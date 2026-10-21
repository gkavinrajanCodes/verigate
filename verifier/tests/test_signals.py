import math
import numpy as np
import pytest
import scipy.stats as stats
import torch

from verigate_verifier.sampling import SamplingParams
from verigate_verifier.signals import SignalConfig, compute_signals, per_token_values


def test_pit_uniformity_ks_test():
    """Property test from PRD §17:

    PIT values u_t must be Uniform(0, 1) when tokens are sampled from the
    reference distribution q_t. Kolmogorov-Smirnov test p-value > 0.01 over
    10,000 tokens.
    """
    n_tokens = 10_000
    vocab_size = 50
    torch.manual_seed(42)
    np.random.seed(42)

    # Random logits for 10,000 positions
    logits = torch.randn(n_tokens, vocab_size)
    probs = torch.softmax(logits, dim=-1)

    # Sample actual tokens according to the reference distribution
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    sp = SamplingParams(temperature=1.0)
    cfg = SignalConfig(smoothing_eps=0.0)  # test pure PIT without smoothing distortion

    v_uniform = np.random.uniform(0.0, 1.0, size=n_tokens)
    pt = per_token_values(logits, sampled_tokens, sp, v_uniform, cfg)

    # Run one-sample Kolmogorov-Smirnov test against standard Uniform(0, 1)
    ks_res = stats.kstest(pt.u, "uniform")
    assert ks_res.pvalue > 0.01, f"PIT failed uniformity test: p-value={ks_res.pvalue:.4f} <= 0.01"


def test_s1_tie_handling():
    """PRD §8.2: positions with margin < delta are near-ties and must be skipped."""
    cfg = SignalConfig(tie_margin_nats=0.5)
    sp = SamplingParams(temperature=0.0)  # greedy

    # Case 1: Clear winner (margin = 2.0 - 0.5 = 1.5 >= 0.5)
    logits_clear = torch.tensor([[2.0, 0.5, 0.0]])
    res_clear = compute_signals(logits_clear, out_ids=[0], params=sp, cfg=cfg)
    assert res_clear.values["s1"] == 0.0  # matched, zero mismatch

    # Case 2: Near-tie (margin = 2.0 - 1.9 = 0.1 < 0.5) -> skipped
    logits_tie = torch.tensor([[2.0, 1.9, 0.0]])
    # Provider picked token 1 (mismatch on argmax 0, but within margin)
    res_tie = compute_signals(logits_tie, out_ids=[1], params=sp, cfg=cfg)
    # Since only 1 position and it is ineligible, s1 should be 0.0
    assert res_tie.values["s1"] == 0.0


def test_s3_surprisal_excess():
    """S3 should be near 0 for reference samples and positive for improbable tokens."""
    n_tokens = 500
    vocab_size = 20
    logits = torch.randn(n_tokens, vocab_size)
    probs = torch.softmax(logits, dim=-1)

    # Honest sample
    honest_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    sp = SamplingParams(temperature=1.0)
    res_honest = compute_signals(logits, honest_ids, sp)

    # Dishonest sample: always pick the least probable token
    least_probable = torch.argmin(probs, dim=-1)
    res_dishonest = compute_signals(logits, least_probable, sp)

    assert abs(res_honest.values["s3"]) < 3.0  # roughly within standard deviations
    assert res_dishonest.values["s3"] > 10.0  # extreme positive surprisal


def test_s4_logprob_consistency():
    """S4 measures discrepancy between provider logprob and reference logprob."""
    logits = torch.tensor([[10.0, 0.0]])  # log_softmax is approx [0.0, -10.0]
    out_ids = [0]
    sp = SamplingParams(temperature=1.0)

    # Honest provider returns -0.0001
    res_honest = compute_signals(logits, out_ids, sp, provider_logprobs=[-0.0001])
    assert res_honest.values["s4"] < 0.05

    # Dishonest provider returns fake logprob -5.0
    res_dishonest = compute_signals(logits, out_ids, sp, provider_logprobs=[-5.0])
    assert res_dishonest.values["s4"] > 4.5
