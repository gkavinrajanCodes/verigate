"""SQLite storage matching PRD §12.

Stores:
1. `audits`: individual scored requests (hashes only; prompts/outputs deleted after scoring).
2. `null_samples`: empirical calibration distributions per (model, signal, temp_bucket, len_bucket).
3. `verdicts`: historical provider verdicts with combined p-values and signal details.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, Sequence


SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
  audit_id        TEXT PRIMARY KEY,
  request_id      TEXT NOT NULL,
  provider        TEXT NOT NULL,
  model_alias     TEXT NOT NULL,
  received_at     TEXT NOT NULL,
  scored_at       TEXT,
  temperature     REAL,
  top_p           REAL,
  prompt_tokens   INTEGER,
  output_tokens   INTEGER,
  scored_tokens   INTEGER,
  tokenization    TEXT,           -- 'provider_tokens' | 'retokenized' | 'mismatch'
  prompt_hash     TEXT,
  output_hash     TEXT,
  s1 REAL, s2 REAL, s3 REAL, s4 REAL, s4_p90 REAL,
  s1_early REAL, s2_early REAL, s3_early REAL,
  status          TEXT NOT NULL   -- 'scored' | 'skipped' | 'error'
);
CREATE INDEX IF NOT EXISTS idx_audits_pm ON audits(provider, model_alias, received_at);

CREATE TABLE IF NOT EXISTS null_samples (
  model_alias  TEXT,
  signal       TEXT,
  temp_bucket  TEXT,
  len_bucket   TEXT,
  value        REAL,
  source       TEXT,
  created_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_null ON null_samples(model_alias, signal, temp_bucket, len_bucket);

CREATE TABLE IF NOT EXISTS verdicts (
  provider      TEXT,
  model_alias   TEXT,
  state         TEXT,
  n_audits      INTEGER,
  combined_p    REAL,
  details_json  TEXT,
  changed_at    TEXT,
  PRIMARY KEY (provider, model_alias, changed_at)
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # Audits
    # ------------------------------------------------------------------
    def record_audit(self, audit_row: dict[str, Any]):
        cols = [
            "audit_id", "request_id", "provider", "model_alias", "received_at", "scored_at",
            "temperature", "top_p", "prompt_tokens", "output_tokens", "scored_tokens",
            "tokenization", "prompt_hash", "output_hash",
            "s1", "s2", "s3", "s4", "s4_p90",
            "s1_early", "s2_early", "s3_early", "status"
        ]
        placeholders = ", ".join(["?"] * len(cols))
        vals = [audit_row.get(c) for c in cols]
        sql = f"INSERT OR REPLACE INTO audits ({', '.join(cols)}) VALUES ({placeholders})"
        with self._conn() as conn:
            conn.execute(sql, vals)

    def get_recent_audits(self, provider: str, model_alias: str, limit: int = 200) -> list[sqlite3.Row]:
        sql = """
        SELECT * FROM audits
        WHERE provider = ? AND model_alias = ? AND status = 'scored'
        ORDER BY received_at DESC LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(sql, (provider, model_alias, limit)).fetchall()
            return list(rows)

    def count_audits(self, provider: str, model_alias: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM audits WHERE provider = ? AND model_alias = ? AND status = 'scored'",
                (provider, model_alias)
            ).fetchone()
            return row["c"] if row else 0

    # ------------------------------------------------------------------
    # Null Samples (Empirical Calibration)
    # ------------------------------------------------------------------
    def insert_null_samples(self, rows: Sequence[dict[str, Any]]):
        if not rows:
            return
        sql = """
        INSERT INTO null_samples (model_alias, signal, temp_bucket, len_bucket, value, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (r["model_alias"], r["signal"], r["temp_bucket"], r["len_bucket"], r["value"], r["source"], r["created_at"])
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(sql, params)

    def get_null_samples(
        self, model_alias: str, signal: str, temp_bucket: str | None = None, len_bucket: str | None = None
    ) -> list[float]:
        q = "SELECT value FROM null_samples WHERE model_alias = ? AND signal = ?"
        params: list[Any] = [model_alias, signal]
        if temp_bucket:
            q += " AND temp_bucket = ?"
            params.append(temp_bucket)
        if len_bucket:
            q += " AND len_bucket = ?"
            params.append(len_bucket)
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
            return [r["value"] for r in rows if r["value"] is not None]

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------
    def get_latest_verdict(self, provider: str, model_alias: str) -> Optional[sqlite3.Row]:
        sql = """
        SELECT * FROM verdicts
        WHERE provider = ? AND model_alias = ?
        ORDER BY changed_at DESC LIMIT 1
        """
        with self._conn() as conn:
            return conn.execute(sql, (provider, model_alias)).fetchone()

    def get_all_latest_verdicts(self) -> list[dict[str, Any]]:
        sql = """
        SELECT v.* FROM verdicts v
        INNER JOIN (
            SELECT provider, model_alias, MAX(changed_at) as max_changed
            FROM verdicts
            GROUP BY provider, model_alias
        ) latest ON v.provider = latest.provider AND v.model_alias = latest.model_alias AND v.changed_at = latest.max_changed
        """
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]

    def record_verdict(
        self,
        provider: str,
        model_alias: str,
        state: str,
        n_audits: int,
        combined_p: float,
        details: dict,
        changed_at: str,
    ):
        sql = """
        INSERT INTO verdicts (provider, model_alias, state, n_audits, combined_p, details_json, changed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (provider, model_alias, state, n_audits, combined_p, json.dumps(details), changed_at))
