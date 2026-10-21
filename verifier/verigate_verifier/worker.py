"""Verifier worker daemon (VF-6, VF-7, VF-8).

Consumes audit jobs from the queue (Redis/PulseDB), batches them by length,
computes reference prefill logits, calculates signals S1–S4, stores the
per-request scores, and deletes the prompt/output text (privacy guarantee).
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from typing import Optional

import redis

from .config import Config
from .reference import ReferenceModel
from .sampling import SamplingParams, len_bucket, temp_bucket
from .signals import SignalConfig, compute_signals
from .stats import StatsEngine
from .store import Store
from .tokens import TokenRecovery

log = logging.getLogger("verigate.worker")


class VerifierWorker:
    def __init__(self, config: Config, store: Optional[Store] = None):
        self.cfg = config
        self.store = store or Store(config.sqlite_path)
        self.r = redis.from_url(config.queue_url, decode_responses=True)
        self.models: dict[str, ReferenceModel] = {}
        self.tokenizers: dict[str, TokenRecovery] = {}
        self.sig_cfg = SignalConfig(
            tie_margin_nats=config.stats.tie_margin_nats,
            smoothing_eps=config.stats.smoothing_eps,
            early_tokens=config.stats.early_tokens,
        )
        self.stats_engine = StatsEngine(
            self.store, config.stats, active_pairs=max(1, len(config.providers))
        )
        self.running = False
        self._load_reference_models()

    def _load_reference_models(self):
        for alias, spec in self.cfg.models.items():
            log.info("Loading reference model for %s: %s", alias, spec.hf_repo)
            ref = ReferenceModel(spec, device=self.cfg.verifier.device)
            self.models[alias] = ref
            self.tokenizers[alias] = TokenRecovery(ref.tok)

    def process_job(self, job_dict: dict) -> bool:
        """Process a single audit job."""
        audit_id = job_dict["audit_id"]
        model_alias = job_dict.get("model_alias")
        provider = job_dict.get("provider")

        ref = self.models.get(model_alias)
        tok_rec = self.tokenizers.get(model_alias)
        if not ref or not tok_rec:
            log.error("No reference model loaded for alias: %s", model_alias)
            return False

        messages = job_dict.get("messages", [])
        output = job_dict.get("output", {})
        params_dict = job_dict.get("params", {})
        sp = SamplingParams.from_job(params_dict)

        # 1. Prompt and Token recovery
        try:
            prompt_ids = ref.prompt_ids(messages)
        except Exception as e:
            log.error("Failed to render/tokenize prompt for audit %s: %v", audit_id, e)
            return False

        rec = tok_rec.recover(output)
        out_ids = rec.ids[: ref.spec.max_scored_tokens]
        if not out_ids:
            log.warning("Audit %s has empty output tokens; skipping", audit_id)
            return False

        # 2. Prefill single-pass over prompt + output
        try:
            logits_list = ref.prefill([(prompt_ids, out_ids)])
            logits = logits_list[0]
        except Exception as e:
            log.error("Prefill failed for audit %s: %v", audit_id, e)
            return False

        # 3. Compute Signals S1–S4
        sig_res = compute_signals(
            logits=logits,
            out_ids=out_ids,
            params=sp,
            provider_logprobs=rec.provider_logprobs,
            seed_key=audit_id,
            cfg=self.sig_cfg,
        )

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t_bucket = temp_bucket(sp.temperature)
        l_bucket = len_bucket(len(prompt_ids))

        # 4. Save to Store (PRD §12, prompt/output text dropped for privacy VF-8)
        audit_row = {
            "audit_id": audit_id,
            "request_id": job_dict.get("request_id"),
            "provider": provider,
            "model_alias": model_alias,
            "received_at": job_dict.get("received_at", now),
            "scored_at": now,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "prompt_tokens": len(prompt_ids),
            "output_tokens": len(rec.ids),
            "scored_tokens": len(out_ids),
            "tokenization": rec.method,
            "prompt_hash": job_dict.get("prompt_hash"),
            "output_hash": job_dict.get("output_hash"),
            "s1": sig_res.values.get("s1"),
            "s2": sig_res.values.get("s2"),
            "s3": sig_res.values.get("s3"),
            "s4": sig_res.values.get("s4"),
            "s4_p90": sig_res.values.get("s4_p90"),
            "s1_early": sig_res.values.get("s1_early"),
            "s2_early": sig_res.values.get("s2_early"),
            "s3_early": sig_res.values.get("s3_early"),
            "status": "scored",
        }
        self.store.record_audit(audit_row)

        # If provider is configured as trusted, add to null samples as well!
        if job_dict.get("trusted", False):
            null_rows = []
            for sname, sval in sig_res.values.items():
                if sval is not None:
                    null_rows.append({
                        "model_alias": model_alias,
                        "signal": sname,
                        "temp_bucket": t_bucket,
                        "len_bucket": l_bucket,
                        "value": sval,
                        "source": f"trusted:{provider}",
                        "created_at": now,
                    })
            self.store.insert_null_samples(null_rows)

        # 5. Evaluate rolling stats and verdicts
        self.stats_engine.evaluate_provider(provider, model_alias)
        return True

    def run(self):
        """Main worker loop consuming jobs from queue."""
        self.running = True
        log.info("Verifier worker started. Listening on %s (%s)", self.cfg.queue_key, self.cfg.queue_url)
        while self.running:
            try:
                # Pop audit ID from list
                pop_res = self.r.brpop(self.cfg.queue_key, timeout=1)
                if not pop_res:
                    continue
                _, audit_id = pop_res
                job_key = self.cfg.job_key(audit_id)
                raw_job = self.r.get(job_key)
                if not raw_job:
                    log.warning("Audit ID %s found in queue but payload %s missing/expired", audit_id, job_key)
                    continue

                job_dict = json.loads(raw_job)
                ok = self.process_job(job_dict)
                # Cleanup job text from queue (VF-8)
                self.r.delete(job_key)
                if ok:
                    log.info("Successfully scored audit %s for %s (%s)", audit_id, job_dict.get("provider"), job_dict.get("model_alias"))
            except redis.ConnectionError:
                time.sleep(2)
            except Exception as e:
                log.exception("Error in verifier worker loop: %v", e)
                time.sleep(0.5)

    def stop(self):
        self.running = False
