"""Loads verigate.yaml for the Python side (verifier, stats, dashboard, CLI)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ReferenceSpec:
    alias: str
    hf_repo: str
    revision: str = "main"
    dtype: str = "bfloat16"
    chat_template: str = "official"
    max_scored_tokens: int = 512


@dataclass
class ProviderSpec:
    name: str
    base_url: str
    models: dict[str, str]
    api_key_env: str = ""
    request_logprobs: bool = True
    audit_rate: float = 0.05
    trusted: bool = False
    pin_upstream_provider: str = ""

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


@dataclass
class StatsConfig:
    window: int = 200
    min_audits: int = 50
    alpha_ok: float = 0.05
    alpha_alert: float = 1e-4
    tie_margin_nats: float = 0.5
    smoothing_eps: float = 1e-6
    early_tokens: int = 32
    min_null_per_bucket: int = 50
    min_per_signal: int = 20
    webhook_url: str = ""


@dataclass
class VerifierConfig:
    device: str = "auto"
    batch_size: int = 8
    batch_wait_ms: int = 200
    max_batch_tokens: int = 16384


@dataclass
class Config:
    queue_url: str = "redis://localhost:6379/0"
    queue_key: str = "verigate:audits"
    models: dict[str, ReferenceSpec] = field(default_factory=dict)
    providers: dict[str, ProviderSpec] = field(default_factory=dict)
    stats: StatsConfig = field(default_factory=StatsConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    sqlite_path: str = "./data/verigate.db"
    dashboard_listen: str = "127.0.0.1:8090"
    report_key_env: str = "VERIGATE_REPORT_KEY"
    raw: dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    def job_key(self, audit_id: str) -> str:
        return f"{self.queue_key}:job:{audit_id}"


def _pick(cls, d: dict | None):
    d = d or {}
    names = cls.__dataclass_fields__.keys()
    return cls(**{k: v for k, v in d.items() if k in names})


def load(path: str | os.PathLike | None) -> Config:
    if path is None:
        for cand in ("verigate.yaml", "verigate.example.yaml"):
            if Path(cand).exists():
                path = cand
                break
    if path is None:
        raise FileNotFoundError("no verigate.yaml found (pass --config)")
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    gw = raw.get("gateway") or {}
    cfg = Config(raw=raw, path=p)
    cfg.queue_url = gw.get("queue_url", cfg.queue_url)
    cfg.queue_key = gw.get("queue_key", cfg.queue_key)

    for alias, m in (raw.get("models") or {}).items():
        ref = (m or {}).get("reference") or {}
        if "hf_repo" not in ref:
            raise ValueError(f"model {alias!r}: reference.hf_repo is required")
        cfg.models[alias] = ReferenceSpec(
            alias=alias,
            hf_repo=ref["hf_repo"],
            revision=str(ref.get("revision", "main")),
            dtype=ref.get("dtype", "bfloat16"),
            chat_template=ref.get("chat_template", "official"),
            max_scored_tokens=int(m.get("max_scored_tokens", 512)),
        )

    for name, pv in (raw.get("providers") or {}).items():
        pv = pv or {}
        cfg.providers[name] = ProviderSpec(
            name=name,
            base_url=str(pv.get("base_url", "")).rstrip("/"),
            models=dict(pv.get("models") or {}),
            api_key_env=pv.get("api_key_env") or "",
            request_logprobs=bool(pv.get("request_logprobs", True)),
            audit_rate=float(pv.get("audit_rate", 0.05)),
            trusted=bool(pv.get("trusted", False)),
            pin_upstream_provider=pv.get("pin_upstream_provider") or "",
        )

    cfg.stats = _pick(StatsConfig, raw.get("stats"))
    cfg.verifier = _pick(VerifierConfig, raw.get("verifier"))
    st = raw.get("storage") or {}
    cfg.sqlite_path = st.get("sqlite_path", cfg.sqlite_path)
    db = raw.get("dashboard") or {}
    cfg.dashboard_listen = db.get("listen", cfg.dashboard_listen)
    cfg.report_key_env = db.get("report_key_env", cfg.report_key_env)
    return cfg
