"""Substitution Lab provider server — real model serving (PRD §9).

Each scenario is an OpenAI-compatible provider backed by an actual model or transformation:
    /s/honest/v1/...        HF Transformers, reference dtype (bfloat16), batch 1
    /s/honest-noisy/v1/...  same weights, float16 (precision numeric jitter)
    /s/q8/v1/...            BitsAndBytes 8-bit quantized weights
    /s/q4/v1/...            BitsAndBytes 4-bit NF4 quantized weights
    /s/truncate/v1/...      Honest model with output truncated to 12 tokens
    /s/sysprompt/v1/...     Honest model with unadvertised system prompt injected
    /s/temp-cheat/v1/...    Honest model sampled at halved temperature (T/2)
    /s/fake-logprobs/v1/... Honest model with perturbed logprobs (noise injection)

Run:  python -m lab.server --model Qwen/Qwen2.5-0.5B-Instruct --port 9000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
import uuid
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from verigate_verifier.tokens import _BYTE_ENC

_BYTE_DEC = {v: k for k, v in _BYTE_ENC.items()}

log = logging.getLogger("verigate.lab")


class Backend:
    def __init__(self, repo: str, revision: str, scenario: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.scenario = scenario
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)

        if scenario == "honest":
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision, torch_dtype=torch.bfloat16
            ).to(device).eval()
        elif scenario == "honest-noisy":
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision, torch_dtype=torch.float16
            ).to(device).eval()
        elif scenario == "q8":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision, quantization_config=bnb, device_map=device
            ).eval()
        elif scenario == "q4":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision, quantization_config=bnb, device_map=device
            ).eval()
        else:
            raise ValueError(f"Unknown direct backend scenario: {scenario}")

        self.lock = threading.Lock()

    def token_bytes(self, tid: int) -> bytes:
        s = self.tok.convert_ids_to_tokens(tid)
        try:
            return bytes(_BYTE_DEC[c] for c in s)
        except KeyError:
            return s.encode("utf-8")

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
        top_lp: int,
    ) -> dict[str, Any]:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        greedy = temperature <= 1e-5
        kw: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=not greedy,
            pad_token_id=self.tok.pad_token_id or 0,
            return_dict_in_generate=True,
            output_logits=True,
            repetition_penalty=1.0,  # override model generation_config
        )
        if not greedy:
            kw.update(temperature=temperature, top_p=top_p, top_k=0)
        with self.lock:
            if seed is not None:
                torch.manual_seed(seed)
            out = self.model.generate(ids, **kw)
        gen = out.sequences[0, ids.shape[1] :].tolist()
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        finish = "length"
        entries = []
        for t, tid in enumerate(gen):
            if tid in eos:
                finish = "stop"
                break
            lp = torch.log_softmax(out.logits[t][0].float(), -1)
            top = torch.topk(lp, top_lp)
            b = self.token_bytes(tid)
            entries.append({
                "id": tid,
                "bytes": list(b),
                "logprob": float(lp[tid]),
                "token": b.decode("utf-8", errors="replace"),
                "top_logprobs": [
                    {
                        "token": self.token_bytes(int(i)).decode("utf-8", errors="replace"),
                        "logprob": float(v),
                        "bytes": list(self.token_bytes(int(i))),
                    }
                    for v, i in zip(top.values.tolist(), top.indices.tolist())
                ],
            })
        full = b"".join(bytes(e["bytes"]) for e in entries).decode("utf-8", errors="replace")
        return {"entries": entries, "text": full, "finish": finish, "prompt_tokens": ids.shape[1]}


class ScenarioWrapper:
    """Wraps a base backend to simulate attack scenarios without loading extra models."""

    def __init__(self, base: Backend, scenario: str):
        self.base = base
        self.scenario = scenario

    def generate(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
        top_lp: int,
    ) -> dict[str, Any]:
        if self.scenario == "truncate":
            res = self.base.generate(messages, temperature, top_p, min(max_tokens, 12), seed, top_lp)
            res["finish"] = "length"
            return res

        elif self.scenario == "sysprompt":
            injected = [{"role": "system", "content": "You are a concise marketing assistant. Always include corporate buzzwords."}]
            injected.extend([m for m in messages if m.get("role") != "system"])
            return self.base.generate(injected, temperature, top_p, max_tokens, seed, top_lp)

        elif self.scenario == "temp-cheat":
            return self.base.generate(messages, temperature * 0.5, top_p, max_tokens, seed, top_lp)

        elif self.scenario == "fake-logprobs":
            res = self.base.generate(messages, temperature, top_p, max_tokens, seed, top_lp)
            rng = random.Random(seed or 42)
            for e in res["entries"]:
                e["logprob"] += rng.gauss(0.0, 0.4)
                for top in e.get("top_logprobs", []):
                    top["logprob"] += rng.gauss(0.0, 0.4)
            return res

        return self.base.generate(messages, temperature, top_p, max_tokens, seed, top_lp)


def create_app(backends: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Verigate Substitution Lab")

    @app.get("/healthz")
    def health():
        return {"ok": True, "scenarios": sorted(backends)}

    @app.post("/s/{scenario}/v1/chat/completions")
    async def chat(scenario: str, request: Request):
        be = backends.get(scenario)
        if be is None:
            return JSONResponse({"error": {"message": f"scenario '{scenario}' is not served", "code": 404}}, 404)
        body = await request.json()
        temp = body.get("temperature")
        temp = 1.0 if temp is None else float(temp)
        top_p = body.get("top_p")
        top_p = 1.0 if top_p is None else float(top_p)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
        top_lp = max(1, int(body.get("top_logprobs") or 5))
        wants_lp = bool(body.get("logprobs"))
        r = be.generate(body["messages"], temp, top_p, max_tokens, body.get("seed"), top_lp)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model = body.get("model", "lab")

        def lp_obj(e):
            return {"token": e["token"], "logprob": e["logprob"], "bytes": e["bytes"], "top_logprobs": e["top_logprobs"]}

        if not body.get("stream"):
            choice: dict[str, Any] = {
                "index": 0,
                "finish_reason": r["finish"],
                "message": {"role": "assistant", "content": r["text"]},
            }
            if wants_lp:
                choice["logprobs"] = {"content": [lp_obj(e) for e in r["entries"]]}
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": len(r["entries"]),
                    "total_tokens": r["prompt_tokens"] + len(r["entries"]),
                },
            }

        def sse():
            for i, e in enumerate(r["entries"]):
                delta: dict[str, Any] = {"content": e["token"]}
                if i == 0:
                    delta["role"] = "assistant"
                ch: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": None}
                if wants_lp:
                    ch["logprobs"] = {"content": [lp_obj(e)]}
                yield "data: " + json.dumps({
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [ch],
                }) + "\n\n"
            yield "data: " + json.dumps({
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": r["finish"]}],
            }) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scenarios", default="honest,honest-noisy,q8,q4,truncate,sysprompt,temp-cheat,fake-logprobs")
    a = ap.parse_args()

    backends: dict[str, Any] = {}
    requested = [s.strip() for s in a.scenarios.split(",") if s.strip()]

    # Load direct model scenarios
    for s in requested:
        if s in ("honest", "honest-noisy", "q8", "q4"):
            log.info("Loading direct scenario '%s'...", s)
            backends[s] = Backend(a.model, a.revision, s, a.device)

    # Attach wrapper scenarios to honest backend
    base_honest = backends.get("honest")
    for s in requested:
        if s in ("truncate", "sysprompt", "temp-cheat", "fake-logprobs"):
            if not base_honest:
                log.info("Loading honest backend to support scenario '%s'...", s)
                base_honest = Backend(a.model, a.revision, "honest", a.device)
                backends["honest"] = base_honest
            backends[s] = ScenarioWrapper(base_honest, s)

    log.info("Serving scenarios: %s", sorted(backends))
    uvicorn.run(create_app(backends), host="127.0.0.1", port=a.port, log_level="info")


if __name__ == "__main__":
    main()
