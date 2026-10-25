"""Send varied chat traffic through the gateway to a provider (drives audits end to end).

    python scripts/send_traffic.py --provider lab-honest --n 60
"""
import argparse
import random

import httpx

PROMPTS = [
    "Explain what a hash table is in two sentences.", "Write a haiku about autumn.",
    "Give three tips for better sleep.", "What causes the tides?",
    "Describe how a binary search works.", "Why is the ocean salty?",
    "Write a short product description for a reusable water bottle.",
    "What is the difference between a list and a tuple in Python?",
    "Suggest a healthy breakfast and explain why it is healthy.",
    "How do vaccines work? Keep it brief.", "Explain recursion with a simple example.",
    "What is inflation, in plain words?", "Write a two-line poem about the moon.",
    "List four uses for a paperclip.", "What is machine learning?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ok = 0
    with httpx.Client(timeout=120) as c:
        for i in range(a.n):
            p = rng.choice(PROMPTS) + f" (variant {rng.randint(0, 10**6)})"
            r = c.post(f"{a.url}/v1/chat/completions", headers={"X-Verigate-Provider": a.provider},
                       json={"model": a.model, "messages": [{"role": "user", "content": p}],
                             "temperature": a.temperature, "max_tokens": a.max_tokens})
            ok += r.status_code == 200
    print(f"{a.provider}: {ok}/{a.n} ok")


if __name__ == "__main__":
    main()
