"""FastAPI dashboard and JSON API (PRD §7.5, UI-1 through UI-5).

Exposes:
- `GET /api/status`: Provider verdicts, audit counts, last changed time.
- `GET /api/providers/{provider}/{model}`: Window test details and audit score history.
- `GET /api/report/{provider}/{model}`: Signed audit report export.
- `GET /`: Modern, responsive dashboard.
"""
from __future__ import annotations

import hmac
import hashlib
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Config, load as load_config
from .store import Store

app = FastAPI(title="Verigate Dashboard API", version="0.1.0")
_cfg: Optional[Config] = None
_store: Optional[Store] = None


def get_store() -> Store:
    global _cfg, _store
    if _store is None:
        _cfg = load_config(os.environ.get("VERIGATE_CONFIG", "verigate.yaml"))
        _store = Store(_cfg.sqlite_path)
    return _store


def get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config(os.environ.get("VERIGATE_CONFIG", "verigate.yaml"))
    return _cfg


@app.get("/api/status")
def get_status():
    store = get_store()
    verdicts = store.get_all_latest_verdicts()
    return {"status": "ok", "verdicts": verdicts}


@app.get("/api/providers/{provider}/{model}")
def get_provider_details(provider: str, model: str):
    store = get_store()
    latest = store.get_latest_verdict(provider, model)
    recent = store.get_recent_audits(provider, model, limit=100)
    total_audits = store.count_audits(provider, model)

    audits_data = []
    for r in recent:
        audits_data.append({
            "audit_id": r["audit_id"],
            "request_id": r["request_id"],
            "received_at": r["received_at"],
            "temperature": r["temperature"],
            "scored_tokens": r["scored_tokens"],
            "tokenization": r["tokenization"],
            "s1": r["s1"],
            "s2": r["s2"],
            "s3": r["s3"],
            "s4": r["s4"],
        })

    details = json.loads(latest["details_json"]) if latest and latest["details_json"] else {}
    return {
        "provider": provider,
        "model_alias": model,
        "total_audits": total_audits,
        "verdict": dict(latest) if latest else None,
        "details": details,
        "recent_audits": audits_data,
    }


@app.get("/api/report/{provider}/{model}")
def get_signed_report(provider: str, model: str):
    cfg = get_cfg()
    store = get_store()
    latest = store.get_latest_verdict(provider, model)
    if not latest:
        raise HTTPException(status_code=404, detail="No audit data for provider/model")

    total_audits = store.count_audits(provider, model)
    report_data = {
        "provider": provider,
        "model_alias": model,
        "verdict": latest["state"],
        "n_audits": total_audits,
        "combined_p": latest["combined_p"],
        "changed_at": latest["changed_at"],
        "details": json.loads(latest["details_json"]) if latest["details_json"] else {},
    }
    raw_bytes = json.dumps(report_data, sort_keys=True).encode()
    report_key = os.environ.get(cfg.report_key_env, "verigate-default-report-key").encode()
    sig = hmac.new(report_key, raw_bytes, hashlib.sha256).hexdigest()
    return {"report": report_data, "signature": f"sha256:{sig}"}


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Verigate — Model Faithfulness Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 26, 44, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --verified: #10b981;
      --verified-bg: rgba(16, 185, 129, 0.12);
      --suspect: #f43f5e;
      --suspect-bg: rgba(244, 63, 94, 0.12);
      --inconclusive: #f59e0b;
      --inconclusive-bg: rgba(245, 158, 11, 0.12);
      --accent: #6366f1;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at top right, #131b31 0%, var(--bg) 60%);
      color: var(--text);
      font-family: 'Outfit', -apple-system, sans-serif;
      min-height: 100vh;
      padding: 2.5rem 3rem;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }
    .brand { display: flex; align-items: center; gap: 0.75rem; }
    .logo-badge {
      background: linear-gradient(135deg, #6366f1, #a855f7);
      width: 36px; height: 36px;
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; color: white;
    }
    h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
    .subtitle { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.2rem; }
    .nav-stats { display: flex; gap: 2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    .stat-pill { background: var(--card-bg); padding: 0.5rem 1rem; border-radius: 9999px; border: 1px solid var(--border); }
    .stat-val { color: var(--accent); font-weight: 600; margin-left: 0.35rem; }

    .grid { display: grid; grid-template-columns: 1fr; gap: 2rem; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      backdrop-filter: blur(16px);
      padding: 1.5rem;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; }

    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
    th {
      color: var(--text-muted);
      font-weight: 500;
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }
    td { padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.03); vertical-align: middle; }
    tr:hover td { background: rgba(255,255,255,0.02); }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.3rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.78rem;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .badge-VERIFIED { background: var(--verified-bg); color: var(--verified); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-SUSPECT { background: var(--suspect-bg); color: var(--suspect); border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-INCONCLUSIVE { background: var(--inconclusive-bg); color: var(--inconclusive); border: 1px solid rgba(245, 158, 11, 0.3); }

    .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
    .btn {
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn:hover { background: rgba(255,255,255,0.12); }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">V</div>
      <div>
        <h1>Verigate</h1>
        <div class="subtitle">Real-traffic continuous LLM provider verification gateway</div>
      </div>
    </div>
    <div class="nav-stats">
      <div class="stat-pill">MODE: <span class="stat-val">Active Auditing</span></div>
      <div class="stat-pill">SAMPLING: <span class="stat-val">HMAC Secret</span></div>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="card-title">
        <span>Provider Status Matrix</span>
        <button class="btn" onclick="loadData()">↻ Refresh</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Provider</th>
            <th>Model Alias</th>
            <th>Verdict</th>
            <th>Audits Audited</th>
            <th>Combined p-value</th>
            <th>Last Evaluated</th>
          </tr>
        </thead>
        <tbody id="verdicts-table">
          <tr><td colspan="6" style="text-align:center; color: var(--text-muted);">Loading live provider verdicts...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    async function loadData() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const tbody = document.getElementById('verdicts-table');
        if (!data.verdicts || data.verdicts.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted); padding: 2rem;">No provider verdicts recorded yet. Start gateway and verifier to audit live traffic.</td></tr>';
          return;
        }
        tbody.innerHTML = data.verdicts.map(v => {
          const pVal = v.combined_p !== null ? v.combined_p.toExponential(2) : '—';
          return `
            <tr>
              <td style="font-weight: 600;">${v.provider}</td>
              <td class="mono">${v.model_alias}</td>
              <td><span class="badge badge-${v.state}">● ${v.state}</span></td>
              <td class="mono">${v.n_audits}</td>
              <td class="mono">${pVal}</td>
              <td style="color: var(--text-muted); font-size: 0.8rem;">${v.changed_at ? new Date(v.changed_at).toLocaleString() : '—'}</td>
            </tr>
          `;
        }).join('');
      } catch (e) {
        console.error(e);
      }
    }
    loadData();
    setInterval(loadData, 5000);
  </script>
</body>
</html>
"""
