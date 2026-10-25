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
import json
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

    def _dispatch_webhook(self, payload: dict):
        url = self.cfg.webhook_url
        if not url:
            return
        import threading
        def _send():
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    client.post(url, json=payload)
            except Exception as e:
                log.warning("Webhook dispatch failed to %s: %s", url, e)
        threading.Thread(target=_send, daemon=True).start()

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

        # Exclude mismatched tokenizations from statistical testing (PRD §18)
        valid_audits = [
            r for r in recent_audits
            if r["tokenization"] != "mismatch" and r["status"] == "scored"
        ]
        n_valid = len(valid_audits)

        if n_valid < self.cfg.min_audits:
            new_state = STATE_INCONCLUSIVE
            eval_res = WindowEvaluation(
                provider=provider,
                model_alias=model_alias,
                n_audits=n_valid,
                state=new_state,
                combined_p=1.0,
                previous_state=prev_state,
                changed=(prev_state != new_state),
                test_results=[],
                timestamp=now,
            )
            if eval_res.changed:
                self.store.record_verdict(
                    provider, model_alias, new_state, n_valid, 1.0, {"reason": "insufficient_audits"}, now
                )
            return eval_res

        # Stratified evaluation by temperature bucket (PRD §8.3, §8.4, ST-6)
        # Group audits by temp_bucket
        from .sampling import temp_bucket

        audits_by_temp: dict[str, list] = {}
        for r in valid_audits:
            tb = temp_bucket(r["temperature"])
            audits_by_temp.setdefault(tb, []).append(r)

        signals_to_check = ["s1", "s2", "s3", "s4", "s1_early", "s2_early", "s3_early"]
        all_tests: list[TestResult] = []

        for tb, tb_audits in audits_by_temp.items():
            for sig in signals_to_check:
                # S1 is greedy-only (T=0)
                if tb != "t0" and sig in ("s1", "s1_early"):
                    continue
                # S2 is sampled-only (T>0)
                if tb == "t0" and sig in ("s2", "s2_early"):
                    continue

                window_vals = [r[sig] for r in tb_audits if r[sig] is not None]
                if len(window_vals) < self.cfg.min_per_signal:
                    continue

                # Try stratified null first; fallback to general null if insufficient
                null_vals = self.store.get_null_samples(model_alias, sig, temp_bucket=tb)
                if len(null_vals) < self.cfg.min_null_per_bucket:
                    null_vals = self.store.get_null_samples(model_alias, sig)
                if len(null_vals) < self.cfg.min_null_per_bucket:
                    continue

                w_arr = np.array(window_vals, dtype=float)
                null_arr = np.array(null_vals, dtype=float)

                # 1. Shift Test: Mann-Whitney U test (alternative='greater')
                try:
                    mw = stats.mannwhitneyu(w_arr, null_arr, alternative="greater")
                    all_tests.append(
                        TestResult(
                            signal=f"{sig}:{tb}",
                            test_type="shift",
                            p_value=float(mw.pvalue),
                            statistic=float(mw.statistic),
                            n_sample=len(w_arr),
                            n_null=len(null_arr),
                            detail={
                                "temp_bucket": tb,
                                "median_window": float(np.median(w_arr)),
                                "median_null": float(np.median(null_arr)),
                            },
                        )
                    )
                except Exception as e:
                    log.warning("Mann-Whitney U failed for %s:%s:%s: %s", provider, sig, tb, e)

                # 2. Outlier-rate Test: Binomial test against 99th percentile tau
                # Mathematically requires at least 100 null samples for tau_99 to be well-defined.
                if len(null_arr) >= 100:
                    try:
                        tau = float(np.percentile(null_arr, 99.0))
                        k = int(np.sum(w_arr > tau))
                        n = len(w_arr)
                        binom = stats.binomtest(k, n, p=0.01, alternative="greater")
                        all_tests.append(
                            TestResult(
                                signal=f"{sig}:{tb}",
                                test_type="outlier",
                                p_value=float(binom.pvalue),
                                statistic=float(k),
                                n_sample=n,
                                n_null=len(null_arr),
                                detail={"temp_bucket": tb, "tau_99": tau, "exceedances": k, "rate": k / n if n > 0 else 0.0},
                            )
                        )
                    except Exception as e:
                        log.warning("Outlier test failed for %s:%s:%s: %s", provider, sig, tb, e)

        # Multi-hypothesis Bonferroni correction
        if not all_tests:
            combined_p = 1.0
        else:
            raw_min_p = min(t.p_value for t in all_tests)
            total_hypotheses = len(all_tests) * self.active_pairs
            combined_p = min(1.0, raw_min_p * total_hypotheses)

        # Hysteresis state machine (PRD §8.5)
        prev_details = json.loads(latest_v["details_json"]) if latest_v and latest_v["details_json"] else {}
        prev_streak = prev_details.get("clean_streak", 0)

        new_state = prev_state
        streak = prev_streak

        if combined_p < self.cfg.alpha_alert:
            new_state = STATE_SUSPECT
            streak = 0
        elif combined_p >= self.cfg.alpha_ok:
            if prev_state == STATE_SUSPECT:
                # Require 2 consecutive windows with p >= alpha_ok to clear SUSPECT
                if prev_streak >= 1:
                    new_state = STATE_VERIFIED
                    streak = 2
                else:
                    streak = 1
                    new_state = STATE_SUSPECT
            else:
                new_state = STATE_VERIFIED
                streak = prev_streak + 1
        else:
            # Between alpha_alert and alpha_ok
            streak = 0
            if prev_state != STATE_SUSPECT:
                new_state = STATE_INCONCLUSIVE

        details = {
            "combined_p": combined_p,
            "clean_streak": streak,
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
        # Record verdict whenever state changes, initial record, or periodically every 10 audits
        should_record = changed or (latest_v is None) or (n_valid % 10 == 0)
        if should_record:
            self.store.record_verdict(provider, model_alias, new_state, n_valid, combined_p, details, now)

        if changed and self.cfg.webhook_url:
            self._dispatch_webhook({
                "event": "verdict.changed",
                "provider": provider,
                "model_alias": model_alias,
                "previous_state": prev_state,
                "new_state": new_state,
                "combined_p": combined_p,
                "n_audits": n_valid,
                "timestamp": now,
                "details": details,
            })

        return WindowEvaluation(
            provider=provider,
            model_alias=model_alias,
            n_audits=n_valid,
            state=new_state,
            combined_p=combined_p,
            previous_state=prev_state,
            changed=changed,
            test_results=all_tests,
            timestamp=now,
        )
