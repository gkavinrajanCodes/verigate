"""M0 exit criterion: print per-token reference probabilities for a provider output.

Usage:
    python scripts/m0_score.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--prompt "..."] \
        [--temperature 0.7] [--output "text from a provider"]

If --output is omitted, the reference model itself generates the output
(an "honest" provider), which should give signals consistent with the null.
"""
from __future__ import annotations

import argparse
import sys

import torch

from verigate_verifier.config import ReferenceSpec
from verigate_verifier.reference import ReferenceModel
from verigate_verifier.sampling import SamplingParams
from verigate_verifier.signals import compute_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    ref = ReferenceModel(ReferenceSpec(alias="m0", hf_repo=a.model, revision=a.revision, dtype=a.dtype), device=a.device)
    msgs = [{"role": "user", "content": a.prompt}]
    p_ids = ref.prompt_ids(msgs)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p)

    if a.output is None:
        inp = torch.tensor([p_ids], device=ref.device)
        kw = dict(max_new_tokens=a.max_new, pad_token_id=ref.pad_id, do_sample=not sp.greedy)
        if not sp.greedy:
            kw.update(temperature=a.temperature, top_p=a.top_p, top_k=0)
        with torch.no_grad():
            gen = ref.model.generate(inp, **kw)[0, len(p_ids):].tolist()
        o_ids = [t for t in gen if t != ref.tok.eos_token_id][: a.max_new]
    else:
        o_ids = list(ref.tok.encode(a.output, add_special_tokens=False))

    logits = ref.prefill([(p_ids, o_ids)])[0]
    res = compute_signals(logits, o_ids, sp, seed_key="m0", keep_per_token=True)
    pt = res.per_token

    print(f"\ndevice={ref.device} dtype={a.dtype} prompt_tokens={len(p_ids)} output_tokens={len(o_ids)}")
    print(f"{'pos':>3} {'token':<14} {'p_ref(y)':>9} {'top1':<12} {'p_top1':>7} {'u(PIT)':>7}")
    probs = torch.softmax(logits, -1)
    for t, y in enumerate(o_ids):
        top = int(probs[t].argmax())
        print(f"{t:>3} {ref.tok.decode([y])!r:<14} {float(probs[t, y]):>9.4f} "
              f"{ref.tok.decode([top])!r:<12} {float(probs[t, top]):>7.4f} {pt.u[t]:>7.3f}")
    print("\ntext:", ref.tok.decode(o_ids))
    print("signals:", {k: (None if v is None else round(v, 4)) for k, v in res.values.items()})


if __name__ == "__main__":
    sys.exit(main())
