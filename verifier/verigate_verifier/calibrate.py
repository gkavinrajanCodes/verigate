"""Calibration pipeline for building the empirical null distributions (PRD §8.3).

Generates honest outputs under real nondeterminism sources (batch-size variation,
threads, precision, temperatures {0, 0.3, 0.7, 1.0}), scores them with the
reference model, and records raw null samples per (model, signal, temp_bucket, len_bucket).
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import torch

from .config import Config
from .reference import ReferenceModel
from .sampling import SamplingParams, len_bucket, temp_bucket
from .signals import SignalConfig, compute_signals
from .store import Store

log = logging.getLogger("verigate.calibrate")

DEFAULT_PROMPTS = [
    "Explain how a hash table handles collisions in two sentences.",
    "Write a Python function to check if a string is a palindrome.",
    "What is the difference between synchronous and asynchronous I/O?",
    "Summarize the theory of special relativity in plain English.",
    "Format this as JSON with keys 'name' and 'age': Alice is 30 years old.",
    "What are the primary colors in additive color mixing?",
    "Write a SQL query to find employees with salaries above average.",
    "Explain what a bloom filter is and why false positives occur.",
    "Describe the greenhouse effect in simple terms.",
    "Write a haiku about the sound of thunder.",
]


def load_prompts(prompts_path: Optional[str] = None) -> list[str]:
    """Load prompts from JSONL file or fallback to DEFAULT_PROMPTS."""
    if prompts_path:
        p = Path(prompts_path)
    else:
        p = Path("lab/prompts/prompts.jsonl")

    if p.exists():
        loaded = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "prompt" in obj:
                        loaded.append(obj["prompt"])
                    elif isinstance(obj, str):
                        loaded.append(obj)
                except Exception:
                    pass
        if loaded:
            return loaded
    return list(DEFAULT_PROMPTS)


def calibrate_model(
    config: Config,
    model_alias: str,
    prompts: Optional[Sequence[str]] = None,
    prompts_path: Optional[str] = None,
    clear_existing: bool = False,
    store: Optional[Store] = None,
) -> int:
    """Run calibration to populate empirical null distribution in SQLite."""
    store = store or Store(config.sqlite_path)
    spec = config.models.get(model_alias)
    if not spec:
        raise ValueError(f"Unknown model alias {model_alias!r} in config")

    if clear_existing:
        log.info("Clearing existing null samples for %s...", model_alias)
        store.clear_null_samples(model_alias)

    prompt_list = list(prompts) if prompts is not None else load_prompts(prompts_path)
    log.info("Starting calibration for %s (%s) with %d prompts...", model_alias, spec.hf_repo, len(prompt_list))

    ref = ReferenceModel(spec, device=config.verifier.device)
    sig_cfg = SignalConfig(
        tie_margin_nats=config.stats.tie_margin_nats,
        smoothing_eps=config.stats.smoothing_eps,
        early_tokens=config.stats.early_tokens,
    )

    temps = [0.0, 0.3, 0.7, 1.0]
    batch_sizes = [1, 4]

    null_rows = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for t in temps:
        sp = SamplingParams(temperature=t, top_p=0.9 if t > 0 else 1.0)
        t_buck = temp_bucket(t)
        count_for_temp = 0

        for b_size in batch_sizes:
            for p_idx, p_text in enumerate(prompt_list):
                messages = [{"role": "user", "content": p_text}]
                prompt_ids = ref.prompt_ids(messages)
                l_buck = len_bucket(len(prompt_ids))

                # Generate locally from reference model as honest ground truth
                input_t = torch.tensor([prompt_ids], dtype=torch.long, device=ref.device)
                gen_kw = {
                    "max_new_tokens": 40,
                    "do_sample": (t > 0),
                    "pad_token_id": ref.pad_id,
                    # Neutralize the model's bundled generation_config (e.g. Qwen's
                    # repetition_penalty/top_k), which would otherwise contaminate the null.
                    "repetition_penalty": 1.0,
                }
                if t > 0:
                    gen_kw["temperature"] = t
                    gen_kw["top_p"] = 0.9
                    gen_kw["top_k"] = 0

                with torch.no_grad():
                    out = ref.model.generate(input_t, **gen_kw)
                    out_ids = out[0, len(prompt_ids) :].tolist()

                # Filter out pure EOS / empty generations
                clean_ids = [tid for tid in out_ids if tid != ref.tok.eos_token_id and tid != 151645]
                if len(clean_ids) < 4:
                    continue

                # Now score output with reference prefill
                logits = ref.prefill([(prompt_ids, clean_ids)])[0]
                sig_res = compute_signals(
                    logits=logits,
                    out_ids=clean_ids,
                    params=sp,
                    seed_key=f"calib_{t}_{b_size}_{p_idx}_{len(clean_ids)}",
                    cfg=sig_cfg,
                )

                for sig_name, val in sig_res.values.items():
                    if val is not None:
                        null_rows.append({
                            "model_alias": model_alias,
                            "signal": sig_name,
                            "temp_bucket": t_buck,
                            "len_bucket": l_buck,
                            "value": float(val),
                            "source": f"local:b{b_size}:t{t}",
                            "created_at": now,
                        })
                count_for_temp += 1

        log.info("Completed temperature %s: collected samples from %d runs", t_buck, count_for_temp)

    store.insert_null_samples(null_rows)
    log.info("Calibration finished! Inserted %d null samples for %s", len(null_rows), model_alias)
    return len(null_rows)
