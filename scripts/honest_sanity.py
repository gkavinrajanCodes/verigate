"""Honest-provider sanity check: do S2/S3 behave like the null for the model's own samples?

Generates N completions from the reference model with *neutral* sampling
(temperature/top-p only; no repetition penalty, top_k disabled) and scores
them. Under H0, per-request S2 ~ N(0,1) and S3 ~ N(0,1) (roughly), and the
pooled per-token PIT values are Uniform(0,1).

    python scripts/honest_sanity.py --n 40
    python scripts/honest_sanity.py --n 40 --hf-defaults   # keep model's generation_config (shows the pitfall)
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.stats as st
import torch

from verigate_verifier.config import ReferenceSpec
from verigate_verifier.reference import ReferenceModel
from verigate_verifier.sampling import SamplingParams
from verigate_verifier.signals import compute_signals

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about the ocean.",
    "What is a hash table? Answer briefly.",
    "Give three tips for learning a new language.",
    "Describe how a bicycle works in a short paragraph.",
    "Why do we have seasons? Keep it short.",
    "Write a Python function that reverses a string.",
    "What are the benefits of regular exercise?",
    "Explain recursion to a ten year old.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What is the difference between TCP and UDP?",
    "Suggest a name for a coffee shop and explain why.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--hf-defaults", action="store_true")
    a = ap.parse_args()

    ref = ReferenceModel(ReferenceSpec("s", a.model, dtype=a.dtype))
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p)
    print("model generation_config:", ref.model.generation_config.to_dict().get("repetition_penalty"),
          "rep_penalty; top_k", ref.model.generation_config.to_dict().get("top_k"))
    s2s, s3s, us = [], [], []
    for i in range(a.n):
        msgs = [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
        p_ids = ref.prompt_ids(msgs)
        kw = dict(max_new_tokens=a.max_new, pad_token_id=ref.pad_id, do_sample=True,
                  temperature=a.temperature, top_p=a.top_p, top_k=0)
        if not a.hf_defaults:
            kw["repetition_penalty"] = 1.0
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            gen = ref.model.generate(torch.tensor([p_ids], device=ref.device), **kw)[0, len(p_ids):].tolist()
        o = [t for t in gen if t != ref.tok.eos_token_id and t != 151645]
        if len(o) < 4:
            continue
        r = compute_signals(ref.prefill([(p_ids, o)])[0], o, sp, seed_key=f"s{i}", keep_per_token=True)
        s2s.append(r.values["s2"]); s3s.append(r.values["s3"]); us.append(r.per_token.u)

    s2s, s3s, u = np.array(s2s), np.array(s3s), np.concatenate(us)
    print(f"\nrequests={len(s2s)} tokens={len(u)} (mode: {'HF defaults' if a.hf_defaults else 'neutral'})")
    print(f"S2: mean={s2s.mean():+.3f} std={s2s.std(ddof=1):.3f}   (ideal 0, 1)")
    print(f"S3: mean={s3s.mean():+.3f} std={s3s.std(ddof=1):.3f}   (ideal 0, ~1)")
    ks = st.kstest(u, "uniform")
    print(f"pooled PIT KS vs Uniform: stat={ks.statistic:.4f} p={ks.pvalue:.3g}")
    print(f"PIT mean={u.mean():.3f} (ideal 0.5); frac u>0.99={np.mean(u > 0.99):.3f} (ideal 0.01)")


if __name__ == "__main__":
    main()
