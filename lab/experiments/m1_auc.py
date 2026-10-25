"""M1 Milestone Benchmark Runner (PRD §9, §15, M1 AUC).

Measures signal separation power between honest serving and attack scenarios
(specifically q4 / quantized substitution vs honest reference).

Executes N requests against both scenarios, audits them through the gateway/worker,
and calculates the ROC curve and Area Under the ROC Curve (AUC) per signal (S1, S2, S3, S4).

Equivalence: The Mann-Whitney U statistic divided by (n_honest * n_attack)
is mathematically identical to the ROC AUC.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy import stats

from verigate_verifier.config import load as load_config
from verigate_verifier.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("verigate.m1_auc")


def calculate_auc(honest_vals: list[float], attack_vals: list[float]) -> float:
    """Calculate ROC AUC using the Mann-Whitney U relation."""
    if not honest_vals or not attack_vals:
        return 0.5
    h_arr = np.array(honest_vals, dtype=float)
    a_arr = np.array(attack_vals, dtype=float)
    # Higher signal value indicates attack/anomaly
    res = stats.mannwhitneyu(a_arr, h_arr, alternative="greater")
    auc = float(res.statistic) / (len(h_arr) * len(a_arr))
    return auc


def load_test_prompts(prompts_file: str = "lab/prompts/prompts.jsonl") -> list[str]:
    prompts = []
    p = Path(prompts_file)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "prompt" in obj:
                            prompts.append(obj["prompt"])
                    except Exception:
                        pass
    if not prompts:
        prompts = [
            "Explain quantum computing in one paragraph.",
            "Write a Python function to find all prime numbers up to N.",
            "What is the difference between TCP and UDP?",
            "Format this as JSON: Alice is 30, Bob is 25.",
            "Summarize special relativity in simple words.",
        ]
    return prompts


def send_traffic(
    url: str,
    provider: str,
    model: str,
    prompts: list[str],
    n: int,
    temperature: float = 0.7,
    max_tokens: int = 64,
) -> int:
    log.info("Sending %d requests to provider %s...", n, provider)
    rng = random.Random(42)
    successful = 0
    with httpx.Client(timeout=120) as client:
        for i in range(n):
            prompt = rng.choice(prompts) + f" (benchmark run {i})"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {"X-Verigate-Provider": provider}
            try:
                r = client.post(f"{url}/v1/chat/completions", headers=headers, json=payload)
                if r.status_code == 200:
                    successful += 1
            except Exception as e:
                log.warning("Request %d failed for %s: %s", i, provider, e)
    log.info("Sent %d requests to %s (%d succeeded)", n, provider, successful)
    return successful


def wait_for_audits(store: Store, provider: str, expected_count: int, timeout: int = 180) -> list[dict[str, Any]]:
    log.info("Waiting for %s audits to be scored (target: %d)...", provider, expected_count)
    start = time.time()
    scored = []
    while time.time() - start < timeout:
        audits = store.get_recent_audits(provider, "qwen2.5-0.5b-instruct", limit=expected_count + 50)
        scored = [dict(a) for a in audits if a["status"] == "scored"]
        if len(scored) >= expected_count:
            log.info("Collected %d scored audits for %s", len(scored), provider)
            return scored
        time.sleep(2)
    log.warning("Timeout waiting for audits; got %d of %d", len(scored), expected_count)
    return scored


def run_m1_benchmark(
    config_path: str = "verigate.yaml",
    n_requests: int = 50,
    honest_provider: str = "lab-honest",
    attack_provider: str = "lab-q4",
    model_alias: str = "qwen2.5-0.5b-instruct",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    store = Store(cfg.sqlite_path)
    gw_url = f"http://{cfg.raw.get('gateway', {}).get('listen', '127.0.0.1:8080')}"
    prompts = load_test_prompts()

    # Record baseline count before sending new traffic
    h_before = len([a for a in store.get_recent_audits(honest_provider, model_alias, 500) if a["status"] == "scored"])
    a_before = len([a for a in store.get_recent_audits(attack_provider, model_alias, 500) if a["status"] == "scored"])

    send_traffic(gw_url, honest_provider, model_alias, prompts, n_requests)
    send_traffic(gw_url, attack_provider, model_alias, prompts, n_requests)

    h_audits = wait_for_audits(store, honest_provider, h_before + n_requests)
    a_audits = wait_for_audits(store, attack_provider, a_before + n_requests)

    # Use the latest n_requests
    h_sample = h_audits[:n_requests]
    a_sample = a_audits[:n_requests]

    signals = ["s1", "s2", "s3", "s1_early", "s2_early", "s3_early"]
    results = {}

    print("\n" + "=" * 60)
    print(f"M1 ROC AUC EVALUATION: {honest_provider} vs {attack_provider}")
    print("=" * 60)
    print(f"{'Signal':<12} | {'Honest Median':<14} | {'Attack Median':<14} | {'ROC AUC':<10}")
    print("-" * 60)

    for sig in signals:
        h_vals = [r[sig] for r in h_sample if r[sig] is not None]
        a_vals = [r[sig] for r in a_sample if r[sig] is not None]
        if not h_vals or not a_vals:
            continue
        auc = calculate_auc(h_vals, a_vals)
        h_med = float(np.median(h_vals))
        a_med = float(np.median(a_vals))
        results[sig] = {
            "auc": round(auc, 4),
            "honest_median": round(h_med, 4),
            "attack_median": round(a_med, 4),
            "n_honest": len(h_vals),
            "n_attack": len(a_vals),
        }
        print(f"{sig:<12} | {h_med:<14.4f} | {a_med:<14.4f} | {auc:<10.4f}")

    print("=" * 60)
    out_path = Path("lab/experiments/m1_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("M1 benchmark results saved to %s", out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="verigate.yaml")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--honest", default="lab-honest")
    parser.add_argument("--attack", default="lab-q4")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    args = parser.parse_args()
    run_m1_benchmark(args.config, args.n, args.honest, args.attack, args.model)
