"""Substitution Lab scenarios server (PRD §9).

Exposes OpenAI-compatible endpoints `/s/{scenario}/v1/chat/completions`
simulating:
- honest: pure honest reference completions
- honest-noisy: honest completion with slight kernel/temperature jitter
- q4 / q2: quantized degradation simulation
- swap-small / swap-family: model substitutions
- truncate: drops middle context (A5)
- sysprompt: prompt tampering (A6)
- fake-logprobs: returns fake honest logprobs for substitute tokens (A8)
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Verigate Substitution Lab")

SCENARIOS = [
    "honest", "honest-noisy", "q4", "q2",
    "swap-small", "swap-family", "truncate", "sysprompt", "fake-logprobs"
]


def generate_response_text(scenario: str, prompt: str, temp: float) -> tuple[str, list[float]]:
    """Synthesizes text and realistic logprobs based on the scenario."""
    honest_words = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]
    
    if scenario in ("honest", "honest-noisy"):
        text = f"As an AI assistant, I can explain that directly: {prompt[:30]}."
        lps = [-0.15, -0.22, -0.08, -0.45, -0.12, -0.30]
    elif scenario in ("q4", "q2"):
        # Quantized model: chooses lower-probability, slightly off-distribution tokens
        text = f"Notice: response generated with altered quantization weights: {prompt[:25]}."
        lps = [-1.8, -2.4, -3.1, -1.9, -2.8]
    elif scenario == "truncate":
        # Context was truncated
        text = "I do not see the full context of your message, but here is a brief answer."
        lps = [-0.9, -1.2, -1.5, -0.8]
    elif scenario == "sysprompt":
        text = "SYSTEM OVERRIDE DETECTED: Following modified instructions."
        lps = [-2.1, -1.9, -2.5]
    elif scenario == "fake-logprobs":
        # Substitute text with fabricated honest-looking logprobs (Attack A8)
        text = "Serving substitute model tokens while falsifying returned logprobs."
        lps = [-0.05, -0.04, -0.02, -0.03, -0.01]  # overly confident fake logprobs
    else:
        text = f"Alternative model output for: {prompt[:30]}."
        lps = [-1.5, -1.7, -2.0]
    return text, lps


@app.post("/s/{scenario}/v1/chat/completions")
async def chat_completions(scenario: str, request: Request):
    if scenario not in SCENARIOS:
        return JSONResponse({"error": f"Unknown lab scenario '{scenario}'"}, status_code=404)

    body = await request.json()
    messages = body.get("messages", [])
    prompt = messages[-1].get("content", "") if messages else ""
    temp = float(body.get("temperature", 1.0))
    stream = bool(body.get("stream", False))
    wants_lp = bool(body.get("logprobs", False))

    text, lps = generate_response_text(scenario, prompt, temp)
    words = text.split(" ")
    tokens = [w if i == 0 else " " + w for i, w in enumerate(words)]
    token_lps = (lps * ((len(tokens) // len(lps)) + 1))[: len(tokens)]

    req_id = f"lab_{scenario}_{int(time.time()*1000)}"

    if not stream:
        lp_content = [
            {"token": tok, "logprob": lp, "bytes": [ord(c) for c in tok], "top_logprobs": []}
            for tok, lp in zip(tokens, token_lps)
        ]
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }
        if wants_lp:
            choice["logprobs"] = {"content": lp_content}

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "qwen2.5-0.5b-instruct"),
            "choices": [choice],
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(tokens)}
        }

    # Streaming mode (SSE)
    async def sse_gen():
        for i, (tok, lp) in enumerate(zip(tokens, token_lps)):
            delta = {"content": tok}
            if i == 0:
                delta["role"] = "assistant"
            choice = {
                "index": 0,
                "delta": delta,
                "finish_reason": None
            }
            if wants_lp:
                choice["logprobs"] = {
                    "content": [{"token": tok, "logprob": lp, "bytes": [ord(c) for c in tok], "top_logprobs": []}]
                }
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "choices": [choice]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)

        # Final chunk
        yield f"data: {json.dumps({'id': req_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")
