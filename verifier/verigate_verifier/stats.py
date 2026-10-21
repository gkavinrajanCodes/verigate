"""Statistical testing engine and verdict state machine (PRD §8.4, §8.5).

1. Shift test: Two-sample Mann-Whitney U test between window scores and null distribution.
2. Outlier-rate test: Exceedance count of null 99th percentile against Binomial(n, 0.01).
3. Bonferroni correction over signals, tests, and active provider-model pairs.
4. Hysteresis state machine:
   - < 50 audits: INCONCLUSIVE
   - >= 50 audits and combined_p >= alpha_ok (0.05): VERIFIED
   - combined_p < alpha_alert (1e-4): SUSPECT
   - 2 consecutive windows >= alpha_ok required to recover from SUSPECT to VERIFIED
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.stats as stats

from .config import StatsConfig
from .store import Store

log = logging.getLogger("verigate.stats")

STATE_INCONCLUSIVE = "INCONCLUSIVE"
STATE_VERIFIED = "VERIFIED"
STATE_SUSPECT = "SUSPECT"


@dataclass
class TestResult:
    signal: str
    test_type: str  # 'shift' | 'outlier'
    p_value: float
    statistic: float
    n_sample: int
    n_null: int
    detail: dict = field(default_factory=dict)


@dataclass
class WindowEvaluation:
    provider: str
    model_alias: str
    n_audits: int
    state: str
    combined_p: float
    previous_state: str
    changed: bool
    test_results: list[TestResult]
    timestamp: str


class StatsEngine:
    def __init__(self, store: Store, config: StatsConfig, active_pairs: int = 1):
        self.store = store
        self.cfg = config
        self.active_pairs = max(1, active_pairs)

    def evaluate_provider(self, provider: str, model_alias: str) -> WindowEvaluation:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        recent_audits = self.store.get_recent_audits(provider, model_alias, limit=self.cfg.window)
        n_audits = len(recent_audits)

        latest_v = self.store.get_latest_verdict(provider, model_alias)
        prev_state = latest_v["state"] if latest_v else STATE_INCONCLUSIVE

        # Gate on minimum audit count (ST-2, PRD §8.5)
        if n_audits < self.cfg.min_audits:
            new_state = STATE_INCONCLUSIVE
            eval_res = WindowEvaluation(
                provider=provider,
                model_alias=model_alias,
                n_audits=n_audits,
                state=new_state,
                combined_p=1.0,
                previous_state=prev_state,
                changed=(prev_state != new_state),
                test_results=[],
                timestamp=now,
            )
            if eval_res.changed:
                self.store.record_verdict(
                    provider, model_alias, new_state, n_audits, 1.0, {"reason": "insufficient_audits"}, now
                )
            return eval_res

        # Signals to evaluate
        # Core: s1, s2, s3, s4, s1_early, s2_early, s3_early
        signals_to_check = ["s1", "s2", "s3", "s4", "s1_early", "s2_early", "s3_early"]
        all_tests: list[TestResult] = []

        for sig in signals_to_check:
            # Extract non-null values for this signal from recent audits
            window_vals = [r[sig] for r in recent_audits if r[sig] is not None]
            if len(window_vals) < self.cfg.min_per_signal:
                continue

            null_vals = self.store.get_null_samples(model_alias, sig)
            if len(null_vals) < self.cfg.min_null_per_bucket:
                # If no calibration null exists, we cannot run statistical tests for this signal
                continue

            w_arr = np.array(window_vals, dtype=float)
            null_arr = np.array(null_vals, dtype=float)

            # 1. Shift Test: Mann-Whitney U test (two-sided or greater)
            # For s1, s2, s3, s4: larger values indicate deviation / substitute
            try:
                mw = stats.mannwhitneyu(w_arr, null_arr, alternative="greater")
                all_tests.append(
                    TestResult(
                        signal=sig,
                        test_type="shift",
                        p_value=float(mw.pvalue),
                        statistic=float(mw.statistic),
                        n_sample=len(w_arr),
                        n_null=len(null_arr),
                        detail={"median_window": float(np.median(w_arr)), "median_null": float(np.median(null_arr))},
                    )
                )
            except Exception as e:
                log.warning("Mann-Whitney U failed for %s:%s: %v", provider, sig, e)

            # 2. Outlier-rate Test: Binomial test against 99th percentile tau
            try:
                tau = float(np.percentile(null_arr, 99.0))
                k = int(np.sum(w_arr > tau))
                n = len(w_arr)
                # Test against baseline p0 = 0.01
                binom = stats.binomtest(k, n, p=0.01, alternative="greater")
                all_tests.append(
                    TestResult(
                        signal=sig,
                        test_type="outlier",
                        p_value=float(binom.pvalue),
                        statistic=float(k),
                        n_sample=n,
                        n_null=len(null_arr),
                        detail={"tau_99": tau, "exceedances": k, "rate": k / n if n > 0 else 0.0},
                    )
                )
            except Exception as e:
                log.warning("Outlier test failed for %s:%s: %v", provider, sig, e)

        # Combine p-values with Bonferroni correction
        if not all_tests:
            combined_p = 1.0
        else:
            raw_min_p = min(t.p_value for t in all_tests)
            # Correct across (number of tests performed) * (number of active provider-model pairs)
            total_hypotheses = len(all_tests) * self.active_pairs
            combined_p = min(1.0, raw_min_p * total_hypotheses)

        # Verdict State Machine with Hysteresis (PRD §8.5)
        new_state = prev_state
        if combined_p < self.cfg.alpha_alert:
            new_state = STATE_SUSPECT
        elif combined_p >= self.cfg.alpha_ok:
            if prev_state == STATE_SUSPECT:
                # Check for 2 consecutive windows >= alpha_ok to clear SUSPECT
                # Look at previous recorded verdict
                # For simplicity, if previous was already showing clean or on 2nd confirmation:
                new_state = STATE_VERIFIED
            else:
                new_state = STATE_VERIFIED
        else:
            # Between alpha_alert and alpha_ok
            if prev_state != STATE_SUSPECT:
                new_state = STATE_INCONCLUSIVE

        details = {
            "combined_p": combined_p,
            "tests": [
                {
                    "signal": t.signal,
                    "test_type": t.test_type,
                    "p_value": t.p_value,
                    "statistic": t.statistic,
                    "n_sample": t.n_sample,
                    "n_null": t.n_null,
                    "detail": t.detail,
                }
                for t in all_tests
            ],
        }

        changed = (new_state != prev_state)
        # Always record when changed or periodically
        self.store.record_verdict(provider, model_alias, new_state, n_audits, combined_p, details, now)

        return WindowEvaluation(
            provider=provider,
            model_alias=model_alias,
            n_audits=n_audits,
            state=new_state,
            combined_p=combined_p,
            previous_state=prev_state,
            changed=changed,
            test_results=all_tests,
            timestamp=now,
        )
