"""Sampling-parameter transforms (PRD §8.1).

Order of operations matches Hugging Face `generate` and vLLM:
temperature -> top-k -> top-p, then renormalize. Top-p keeps the smallest
set of most-likely tokens whose cumulative probability reaches `top_p`
(always at least one token).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

GREEDY_EPS = 1e-5


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= GREEDY_EPS

    @classmethod
    def from_job(cls, params: dict | None) -> "SamplingParams":
        params = params or {}
        t = params.get("temperature")
        p = params.get("top_p")
        k = params.get("top_k")
        # OpenAI-compatible defaults when the client did not specify them.
        t = 1.0 if t is None else float(t)
        p = 1.0 if p is None else float(p)
        k = None if (k is None or int(k) <= 0) else int(k)
        return cls(temperature=t, top_p=p, top_k=k)


def apply_sampling(logits: torch.Tensor, sp: SamplingParams) -> torch.Tensor:
    """Return processed logits (masked entries = -inf) for rows of `logits`.

    `logits` has shape [n, V]. For greedy requests the input is returned
    unchanged (callers handle greedy separately).
    """
    x = logits.float()
    if sp.greedy:
        return x
    if sp.temperature != 1.0:
        x = x / sp.temperature
    V = x.shape[-1]
    if sp.top_k is not None and sp.top_k < V:
        kth = torch.topk(x, sp.top_k, dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if sp.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, dim=-1, descending=False)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cum <= (1.0 - sp.top_p)
        remove_sorted[..., -1] = False  # keep at least the most likely token
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        x = x.masked_fill(remove, float("-inf"))
    return x


def smoothed_probs(processed_logits: torch.Tensor, eps: float) -> torch.Tensor:
    """q = (1 - eps) * softmax(processed) + eps / V  (PRD §8.1 ε-smoothing)."""
    q = processed_logits.softmax(dim=-1)
    if eps > 0:
        V = q.shape[-1]
        q = q * (1.0 - eps) + (eps / V)
    return q


def temp_bucket(temperature: Optional[float]) -> str:
    t = 1.0 if temperature is None else float(temperature)
    if t <= GREEDY_EPS:
        return "t0"
    if t <= 0.5:
        return "t0.3"
    if t <= 0.85:
        return "t0.7"
    return "t1"


def len_bucket(prompt_tokens: Optional[int]) -> str:
    n = prompt_tokens or 0
    if n < 512:
        return "short"
    if n < 2048:
        return "medium"
    return "long"


def finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None
