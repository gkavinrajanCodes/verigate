import numpy as np
import pytest
import torch

from verigate_verifier.sampling import SamplingParams, apply_sampling, smoothed_probs


def test_sampling_params_greedy():
    sp0 = SamplingParams(temperature=0.0)
    assert sp0.greedy
    sp1 = SamplingParams(temperature=0.7)
    assert not sp1.greedy


def test_sampling_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    sp = SamplingParams(temperature=0.5)
    scaled = apply_sampling(logits, sp)
    assert torch.allclose(scaled, torch.tensor([[2.0, 4.0, 6.0]]))


def test_sampling_top_k():
    logits = torch.tensor([[1.0, 5.0, 2.0, 10.0, 3.0]])
    sp = SamplingParams(temperature=1.0, top_k=2)
    processed = apply_sampling(logits, sp)
    # top 2 are 10.0 and 5.0; other elements must be -inf
    assert processed[0, 1] == 5.0
    assert processed[0, 3] == 10.0
    assert processed[0, 0] == float("-inf")
    assert processed[0, 2] == float("-inf")
    assert processed[0, 4] == float("-inf")


def test_sampling_top_p():
    # Logits where softmax gives approx [0.05, 0.15, 0.80]
    logits = torch.tensor([[0.0, 1.0986, 2.7725]])  # exp: [1, 3, 16], sum=20 -> [0.05, 0.15, 0.8]
    sp = SamplingParams(temperature=1.0, top_p=0.85)
    processed = apply_sampling(logits, sp)
    # Top token is index 2 (prob 0.8). Index 1 (prob 0.15) brings cumulative to 0.95 >= 0.85.
    # Index 0 (prob 0.05) should be masked out (-inf).
    assert processed[0, 0] == float("-inf")
    assert processed[0, 1] != float("-inf")
    assert processed[0, 2] != float("-inf")


def test_smoothed_probs():
    logits = torch.tensor([[0.0, 0.0]])
    q = smoothed_probs(logits, eps=0.1)
    # Uniform base is 0.5. With eps=0.1, (1-0.1)*0.5 + 0.1/2 = 0.45 + 0.05 = 0.5
    assert torch.allclose(q, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(q.sum(dim=-1), torch.tensor([1.0]))
