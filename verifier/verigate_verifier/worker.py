"""Verifier worker daemon (VF-6, VF-7, VF-8).

Consumes audit jobs from the queue (Redis/PulseDB) using a reliable processing queue pattern,
batches them up to batch_size and max_batch_tokens, computes reference prefill logits in a single
GPU forward pass, calculates signals S1–S4, stores the per-request scores in SQLite,
and deletes the prompt/output text (privacy guarantee).
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from collections import defaultdict
from typing import Any, Optional

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
        # protocol=2 for broad compatibility with Redis 5+ and PulseDB
        self.r = redis.from_url(config.queue_url, decode_responses=True, protocol=2)
        self.main_queue = config.queue_key
        self.processing_queue = f"{config.queue_key}:processing"
        self.dead_queue = f"{config.queue_key}:dead"

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

        # Metrics counters
        self.total_scored_audits = 0
        self.total_scored_tokens = 0

        self._load_reference_models()
        self._recover_stale_processing_jobs()

    def _load_reference_models(self):
        for alias, spec in self.cfg.models.items():
            log.info("Loading reference model for %s: %s", alias, spec.hf_repo)
            ref = ReferenceModel(spec, device=self.cfg.verifier.device)
            self.models[alias] = ref
            self.tokenizers[alias] = TokenRecovery(ref.tok)

    def _recover_stale_processing_jobs(self):
        """Crash recovery: re-enqueue jobs that were in-flight when a prior worker process terminated."""
        try:
            stale_ids = self.r.lrange(self.processing_queue, 0, -1)
            if not stale_ids:
                return
            log.info("Found %d lingering jobs in %s; recovering...", len(stale_ids), self.processing_queue)
            for aid in stale_ids:
                job_key = self.cfg.job_key(aid)
                if self.r.exists(job_key):
                    # Check retry count
                    retries = int(self.r.get(f"verigate:retries:{aid}") or 0)
                    if retries < 3:
                        log.info("Re-enqueuing recovered audit %s (retry %d)", aid, retries)
                        self.r.rpush(self.main_queue, aid)
                    else:
                        log.warning("Audit %s exceeded max retries; moving to DLQ", aid)
                        self.r.rpush(self.dead_queue, aid)
                        self.r.delete(job_key)
                else:
                    log.warning("Audit payload for %s expired; dropping from processing", aid)
                self.r.lrem(self.processing_queue, 1, aid)
        except Exception as e:
            log.warning("Crash recovery check failed: %s", e)

    def _process_item_preparation(self, aid: str) -> Optional[dict[str, Any]]:
        """Fetch payload and prepare token IDs for an audit job."""
        job_key = self.cfg.job_key(aid)
        raw_job = self.r.get(job_key)
        if not raw_job:
            log.warning("Audit ID %s missing payload %s", aid, job_key)
            self.r.lrem(self.processing_queue, 1, aid)
            return None

        try:
            job_dict = json.loads(raw_job)
        except Exception:
            log.error("Corrupted JSON in audit %s; discarding", aid)
            self.r.lrem(self.processing_queue, 1, aid)
            self.r.delete(job_key)
            return None

        model_alias = job_dict.get("model_alias")
        provider = job_dict.get("provider")
        ref = self.models.get(model_alias)
        tok_rec = self.tokenizers.get(model_alias)
        if not ref or not tok_rec:
            log.error("No reference model for alias: %s; dropping audit %s", model_alias, aid)
            self.r.lrem(self.processing_queue, 1, aid)
            self.r.delete(job_key)
            return None

        messages = job_dict.get("messages", [])
        output = job_dict.get("output", {})
        params_dict = job_dict.get("params", {})
        sp = SamplingParams.from_job(params_dict)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            prompt_ids = ref.prompt_ids(messages)
        except Exception as e:
            log.error("Failed to tokenize prompt for audit %s: %s", aid, e)
            self._record_skipped_audit(aid, job_dict, "tokenization_error", now)
            self.r.lrem(self.processing_queue, 1, aid)
            self.r.delete(job_key)
            return None

        rec = tok_rec.recover(output)
        out_ids = rec.ids[: ref.spec.max_scored_tokens]
        if not out_ids:
            log.info("Audit %s has empty output tokens; recording as skipped", aid)
            self._record_skipped_audit(aid, job_dict, "empty_output", now, len(prompt_ids))
            self.r.lrem(self.processing_queue, 1, aid)
            self.r.delete(job_key)
            return None

        return {
            "audit_id": aid,
            "job_dict": job_dict,
            "job_key": job_key,
            "model_alias": model_alias,
            "provider": provider,
            "ref": ref,
            "sp": sp,
            "rec": rec,
            "prompt_ids": prompt_ids,
            "out_ids": out_ids,
            "tokens_count": len(prompt_ids) + len(out_ids),
        }

    def _record_skipped_audit(
        self,
        aid: str,
        job_dict: dict,
        reason: str,
        now: str,
        prompt_tokens: int = 0,
    ):
        self.store.record_audit({
            "audit_id": aid,
            "request_id": job_dict.get("request_id"),
            "provider": job_dict.get("provider"),
            "model_alias": job_dict.get("model_alias"),
            "received_at": job_dict.get("received_at", now),
            "scored_at": now,
            "temperature": job_dict.get("params", {}).get("temperature"),
            "top_p": job_dict.get("params", {}).get("top_p"),
            "prompt_tokens": prompt_tokens,
            "output_tokens": 0,
            "scored_tokens": 0,
            "tokenization": reason,
            "prompt_hash": job_dict.get("prompt_hash"),
            "output_hash": job_dict.get("output_hash"),
            "s1": None, "s2": None, "s3": None, "s4": None, "s4_p90": None,
            "s1_early": None, "s2_early": None, "s3_early": None,
            "status": "skipped",
        })

    def process_batch(self, batch_items: list[dict[str, Any]]):
        """Batched forward prefill over multiple audits in a single GPU pass (VF-6)."""
        if not batch_items:
            return

        # Group items by model_alias
        by_model = defaultdict(list)
        for item in batch_items:
            by_model[item["model_alias"]].append(item)

        providers_to_evaluate = set()
        t_start = time.time()
        tokens_batch = sum(it["tokens_count"] for it in batch_items)

        for model_alias, items in by_model.items():
            ref = items[0]["ref"]
            prefill_pairs = [(it["prompt_ids"], it["out_ids"]) for it in items]

            try:
                # Batched single-pass GPU prefill
                logits_list = ref.prefill(prefill_pairs)
            except Exception as e:
                log.exception("Batched prefill failed for model %s: %s", model_alias, e)
                for it in items:
                    aid = it["audit_id"]
                    self.r.lrem(self.processing_queue, 1, aid)
                continue

            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            for i, it in enumerate(items):
                aid = it["audit_id"]
                job_dict = it["job_dict"]
                job_key = it["job_key"]
                sp = it["sp"]
                rec = it["rec"]
                prompt_ids = it["prompt_ids"]
                out_ids = it["out_ids"]
                logits = logits_list[i]
                provider = it["provider"]

                sig_res = compute_signals(
                    logits=logits,
                    out_ids=out_ids,
                    params=sp,
                    provider_logprobs=rec.provider_logprobs,
                    seed_key=aid,
                    cfg=self.sig_cfg,
                )

                t_bucket = temp_bucket(sp.temperature)
                l_bucket = len_bucket(len(prompt_ids))

                audit_row = {
                    "audit_id": aid,
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

                # Ack on completion: remove from processing list and delete raw text payload
                self.r.lrem(self.processing_queue, 1, aid)
                self.r.delete(job_key)
                self.r.delete(f"verigate:retries:{aid}")
                providers_to_evaluate.add((provider, model_alias))

                self.total_scored_audits += 1
                self.total_scored_tokens += len(out_ids)

        dur = time.time() - t_start
        throughput = (tokens_batch / dur) if dur > 0 else 0
        log.info(
            "Batch completed: %d audits (%d tokens) scored in %.3fs (%.1f tokens/sec)",
            len(batch_items),
            tokens_batch,
            dur,
            throughput,
        )

        # Trigger rolling evaluation periodically
        for p, m in providers_to_evaluate:
            self.stats_engine.evaluate_provider(p, m)

    def run(self):
        """Main worker loop with reliable pop and token-budget batching."""
        self.running = True
        log.info("Verifier worker started. Listening on %s (processing: %s)", self.main_queue, self.processing_queue)

        max_batch_size = max(1, self.cfg.verifier.batch_size)
        max_batch_tokens = max(512, self.cfg.verifier.max_batch_tokens)
        wait_seconds = max(0.01, float(self.cfg.verifier.batch_wait_ms) / 1000.0)

        while self.running:
            try:
                # 1. Reliable Pop: move one job from main_queue to processing_queue
                first_aid = self.r.brpoplpush(self.main_queue, self.processing_queue, timeout=1)
                if not first_aid:
                    continue

                batch_aids = [first_aid]
                batch_tokens = 0

                # 2. Accumulate up to max_batch_size / max_batch_tokens
                deadline = time.time() + wait_seconds
                while len(batch_aids) < max_batch_size and time.time() < deadline:
                    next_aid = self.r.rpoplpush(self.main_queue, self.processing_queue)
                    if not next_aid:
                        time.sleep(0.02)
                        continue
                    batch_aids.append(next_aid)

                # 3. Prepare items
                prepared_items = []
                for aid in batch_aids:
                    item = self._process_item_preparation(aid)
                    if item:
                        if batch_tokens + item["tokens_count"] > max_batch_tokens and prepared_items:
                            # Re-enqueue item that exceeds token budget
                            self.r.lrem(self.processing_queue, 1, aid)
                            self.r.lpush(self.main_queue, aid)
                            break
                        prepared_items.append(item)
                        batch_tokens += item["tokens_count"]

                # 4. Process the batched forward passes
                if prepared_items:
                    self.process_batch(prepared_items)

            except redis.ConnectionError:
                log.warning("Redis connection error; retrying in 2s...")
                time.sleep(2)
            except Exception:
                log.exception("Error in verifier worker loop")
                time.sleep(2)

    def stop(self):
        self.running = False
