"""Per-token verification signals S1–S4 (PRD §8.2).

All signals are computed from the reference model's next-token distribution
at each output position, conditioned on the full prompt and the provider's
own preceding output tokens. S1–S3 use only the provider's *output tokens*;
S4 additionally uses provider-reported logprobs and is never used alone.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .sampling import SamplingParams, apply_sampling, smoothed_probs

SIGNAL_NAMES = ("s1", "s2", "s3", "s4", "s4_p90", "s1_early", "s2_early", "s3_early")


@dataclass
class SignalConfig:
    tie_margin_nats: float = 0.5   # δ for S1
    smoothing_eps: float = 1e-6    # ε for q_t
    early_tokens: int = 32         # context check window
    chunk_rows: int = 128          # positions processed at once (memory bound)


@dataclass
class PerToken:
    """Per-position intermediate values (useful for diagnostics and tests)."""
    s1_eligible: np.ndarray
    s1_mismatch: np.ndarray
    u: np.ndarray            # randomized PIT values
    z: np.ndarray            # Φ⁻¹(u)
    excess: np.ndarray       # surprisal − entropy
    var: np.ndarray          # variance of surprisal under q
    ref_logprob: np.ndarray  # log p_t(y_t) under the raw reference distribution


@dataclass
class SignalResult:
    values: dict = field(default_factory=dict)
    per_token: Optional[PerToken] = None
    n_scored: int = 0


def pit_seed(key: str) -> int:
    """Deterministic RNG seed for the PIT randomization of one audit."""
    return int.from_bytes(hashlib.sha256(("pit:" + key).encode()).digest()[:8], "big")


def _nan() -> float:
    return float("nan")


@torch.no_grad()
def per_token_values(
    logits: torch.Tensor,
    out_ids: torch.Tensor,
    sp: SamplingParams,
    v_uniform: np.ndarray,
    cfg: SignalConfig,
) -> PerToken:
    """Compute per-position quantities from reference logits [n, V]."""
    n = logits.shape[0]
    dev = logits.device
    out_ids = out_ids.to(dev).long()
    vt = torch.as_tensor(v_uniform, dtype=torch.float64, device=dev)

    elig, mism, us, exc, var, ref_lp = [], [], [], [], [], []
    for a in range(0, n, cfg.chunk_rows):
        b = min(n, a + cfg.chunk_rows)
        lg = logits[a:b].float()
        y = out_ids[a:b]
        logp = torch.log_softmax(lg, dim=-1)
        ref_lp.append(logp.gather(-1, y[:, None]).squeeze(-1).double())

        # ---- S1: tie-aware argmax mismatch (greedy only) ----
        if sp.greedy:
            top2 = torch.topk(logp, 2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            e = margin >= cfg.tie_margin_nats
            m = (y != logp.argmax(dim=-1)) & e
            elig.append(e)
            mism.append(m)

        # ---- distribution for S2/S3 ----
        # Greedy requests have a degenerate q; we score them under the raw
        # (T=1) reference distribution instead. Nulls are stratified by
        # temperature bucket, so this is calibrated like everything else.
        processed = lg if sp.greedy else apply_sampling(lg, sp)
        q = smoothed_probs(processed, cfg.smoothing_eps)
        qy = q.gather(-1, y[:, None])  # [r,1]

        # ---- S2: randomized PIT. Vocabulary ordered by descending q; tokens
        # with probability exactly equal to q(y) form one block, which keeps
        # the randomized PIT exactly uniform under H0.
        greater = (q > qy).to(q.dtype)
        equal = (q == qy).to(q.dtype)
        F_minus = (q * greater).sum(-1).double()
        tie_mass = (q * equal).sum(-1).double()
        us.append(F_minus + vt[a:b] * tie_mass)

        # ---- S3: surprisal excess ----
        logq = torch.log(q)
        surpr = -logq.gather(-1, y[:, None]).squeeze(-1).double()
        H = -(q * logq).sum(-1).double()
        E2 = (q * logq * logq).sum(-1).double()
        exc.append(surpr - H)
        var.append((E2 - H * H).clamp_min(0.0))

    u = torch.cat(us).clamp(1e-12, 1 - 1e-12)
    z = torch.special.ndtri(u)
    if sp.greedy:
        e_all = torch.cat(elig).cpu().numpy()
        m_all = torch.cat(mism).cpu().numpy()
    else:
        e_all = np.zeros(n, dtype=bool)
        m_all = np.zeros(n, dtype=bool)
    return PerToken(
        s1_eligible=e_all,
        s1_mismatch=m_all,
        u=u.cpu().numpy(),
        z=z.cpu().numpy(),
        excess=torch.cat(exc).cpu().numpy(),
        var=torch.cat(var).cpu().numpy(),
        ref_logprob=torch.cat(ref_lp).cpu().numpy(),
    )


def aggregate(pt: PerToken, sp: SamplingParams, sl: slice) -> tuple[float, float, float]:
    """Aggregate per-token values over a slice of positions into (S1, S2, S3)."""
    z = pt.z[sl]
    n = len(z)
    if n == 0:
        return _nan(), _nan(), _nan()
    s1 = _nan()
    if sp.greedy:
        e = pt.s1_eligible[sl]
        if e.sum() > 0:
            s1 = float(pt.s1_mismatch[sl].sum() / e.sum())
        else:
            s1 = 0.0
    s2 = float(z.sum() / math.sqrt(n))
    v = float(pt.var[sl].sum())
    s3 = float(pt.excess[sl].sum() / math.sqrt(v)) if v > 1e-12 else _nan()
    return s1, s2, s3


def compute_signals(
    logits: torch.Tensor,
    out_ids: Sequence[int] | torch.Tensor,
    params: SamplingParams,
    *,
    provider_logprobs: Optional[Sequence[Optional[float]]] = None,
    seed_key: str = "",
    cfg: SignalConfig | None = None,
    keep_per_token: bool = False,
) -> SignalResult:
    """Compute S1–S4 (+ early-window variants) for one audited request.

    logits: reference logits at each output position, shape [n, V]
            (row t predicts output token t given prompt + y_<t).
    """
    cfg = cfg or SignalConfig()
    ids = torch.as_tensor(list(out_ids) if not torch.is_tensor(out_ids) else out_ids)
    n = int(ids.shape[0])
    res = SignalResult(n_scored=n)
    if n == 0:
        res.values = {k: None for k in SIGNAL_NAMES}
        return res

    rng = np.random.default_rng(pit_seed(seed_key))
    v = rng.random(n)
    pt = per_token_values(logits, ids, params, v, cfg)

    s1, s2, s3 = aggregate(pt, params, slice(0, n))
    e1, e2, e3 = aggregate(pt, params, slice(0, min(n, cfg.early_tokens)))

    s4 = s4p90 = _nan()
    if provider_logprobs is not None and len(provider_logprobs) >= n:
        plp = np.array([np.nan if x is None else float(x) for x in provider_logprobs[:n]])
        ok = np.isfinite(plp)
        if ok.any():
            d = np.abs(plp[ok] - pt.ref_logprob[ok])
            s4 = float(np.median(d))
            s4p90 = float(np.percentile(d, 90))

    def clean(x: float) -> Optional[float]:
        return x if (x is not None and math.isfinite(x)) else None

    res.values = {
        "s1": clean(s1), "s2": clean(s2), "s3": clean(s3),
        "s4": clean(s4), "s4_p90": clean(s4p90),
        "s1_early": clean(e1), "s2_early": clean(e2), "s3_early": clean(e3),
    }
    if keep_per_token:
        res.per_token = pt
    return res
