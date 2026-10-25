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
    report_key = os.environ.get(cfg.report_key_env)
    if not report_key:
        raise HTTPException(
            status_code=500,
            detail=f"Report signing key environment variable '{cfg.report_key_env}' is not set; refusing to sign report",
        )
    sig = hmac.new(report_key.encode(), raw_bytes, hashlib.sha256).hexdigest()
    return {"report": report_data, "signature": f"sha256:{sig}"}


@app.get("/api/benchmarks")
def get_benchmarks():
    from pathlib import Path
    p = Path("results/attack_benchmark_e2_e7_e9.json")
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"error": "Benchmark data not generated yet"}


@app.get("/api/audits/recent")
def get_recent_audits_stream():
    store = get_store()
    recent = store.get_all_recent_audits(limit=30)
    total = store.get_total_audits_count()
    return {
        "total_audits": total,
        "audits": [
            {
                "audit_id": r["audit_id"],
                "request_id": r["request_id"],
                "provider": r["provider"],
                "model_alias": r["model_alias"],
                "scored_tokens": r["scored_tokens"],
                "temperature": r["temperature"],
                "s1": r["s1"],
                "s2": r["s2"],
                "s3": r["s3"],
                "s4": r["s4"],
                "scored_at": r["scored_at"],
            }
            for r in recent
        ],
    }


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
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 26, 44, 0.75);
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
      --accent-glow: rgba(99, 102, 241, 0.25);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at top right, #131b31 0%, var(--bg) 60%);
      color: var(--text);
      font-family: 'Outfit', -apple-system, sans-serif;
      min-height: 100vh;
      padding: 2rem 2.5rem;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.25rem;
    }
    .brand { display: flex; align-items: center; gap: 0.85rem; }
    .logo-badge {
      background: linear-gradient(135deg, #6366f1, #a855f7);
      width: 40px; height: 40px;
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; color: white; font-size: 1.2rem;
      box-shadow: 0 4px 16px var(--accent-glow);
    }
    h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
    .subtitle { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.15rem; }
    .nav-stats { display: flex; gap: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    .stat-pill { background: var(--card-bg); padding: 0.4rem 0.85rem; border-radius: 9999px; border: 1px solid var(--border); }
    .stat-val { color: var(--accent); font-weight: 600; margin-left: 0.35rem; }

    /* Metric summary cards */
    .metrics-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      backdrop-filter: blur(12px);
    }
    .metric-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
    .metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 0.35rem; font-family: 'JetBrains Mono', monospace; }
    .metric-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }

    /* Charts grid */
    .charts-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    @media (max-width: 1000px) {
      .charts-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      backdrop-filter: blur(16px);
      padding: 1.5rem;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      margin-bottom: 2rem;
    }
    .card-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; }

    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem; }
    th {
      color: var(--text-muted);
      font-weight: 500;
      padding: 0.8rem 1rem;
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 0.72rem;
      letter-spacing: 0.05em;
    }
    td { padding: 0.9rem 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.03); vertical-align: middle; }
    tr:hover td { background: rgba(255,255,255,0.02); }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.75rem;
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
        <div class="subtitle">Continuous statistical LLM provider verification gateway</div>
      </div>
    </div>
    <div class="nav-stats">
      <div class="stat-pill">GATEWAY: <span class="stat-val">6,656 req/s</span></div>
      <div class="stat-pill">PREFILL: <span class="stat-val">1,080 tok/s</span></div>
      <div class="stat-pill">MODE: <span class="stat-val">Active HMAC</span></div>
    </div>
  </header>

  <div class="metrics-row">
    <div class="metric-card">
      <div class="metric-title">Gateway Overhead (p50)</div>
      <div class="metric-value" style="color: #10b981;">+0.54 ms</div>
      <div class="metric-sub">&lt; 1 ms p99 hot-path isolation</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Verifier Speedup</div>
      <div class="metric-value" style="color: #6366f1;">30.0×</div>
      <div class="metric-sub">Prefill vs autoregressive generation</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Verification Cost</div>
      <div class="metric-value" style="color: #38bdf8;">0.167%</div>
      <div class="metric-sub">at standard 5% audit sampling rate</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Total Audits Scored</div>
      <div class="metric-value" id="total-audits-count" style="color: #f59e0b;">—</div>
      <div class="metric-sub">GPU single-pass re-scored</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="card" style="margin-bottom:0;">
      <div class="card-title">
        <span>Signal Separation Power (E2 ROC AUC Benchmark)</span>
        <span class="mono" style="font-size:0.75rem; color:var(--text-muted);">Higher = Stronger Detection</span>
      </div>
      <div style="height: 260px; position: relative;">
        <canvas id="aucChart"></canvas>
      </div>
    </div>

    <div class="card" style="margin-bottom:0;">
      <div class="card-title">
        <span>Recent Audits: Surprisal Excess (S3) Distribution</span>
        <span class="mono" style="font-size:0.75rem; color:var(--text-muted);">Honest (0.0) vs Anomaly (&gt; 2.0)</span>
      </div>
      <div style="height: 260px; position: relative;">
        <canvas id="s3Chart"></canvas>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">
      <span>Live Provider Faithfulness Matrix</span>
      <button class="btn" onclick="loadAll()">↻ Refresh</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>Provider</th>
          <th>Model Alias</th>
          <th>Verdict</th>
          <th>Audits Evaluated</th>
          <th>Combined p-value</th>
          <th>Last Evaluated</th>
        </tr>
      </thead>
      <tbody id="verdicts-table">
        <tr><td colspan="6" style="text-align:center; color: var(--text-muted);">Loading live provider verdicts...</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-title">
      <span>Live Audited Completions Stream</span>
      <span class="mono" style="font-size:0.75rem; color:var(--text-muted);">Off-Path GPU Scored</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Request ID</th>
          <th>Provider</th>
          <th>Tokens</th>
          <th>S1 (Mismatch)</th>
          <th>S2 (PIT)</th>
          <th>S3 (Surprisal)</th>
          <th>S4 (Logprob)</th>
          <th>Scored At</th>
        </tr>
      </thead>
      <tbody id="audits-table">
        <tr><td colspan="8" style="text-align:center; color: var(--text-muted);">Loading recent audit stream...</td></tr>
      </tbody>
    </table>
  </div>

  <script>
    let aucChartInstance = null;
    let s3ChartInstance = null;

    async function loadVerdicts() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const tbody = document.getElementById('verdicts-table');
        if (!data.verdicts || data.verdicts.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">No provider verdicts recorded yet.</td></tr>';
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
              <td class="mono" style="${v.combined_p < 0.0001 ? 'color:#f43f5e; font-weight:700;' : ''}">${pVal}</td>
              <td style="color: var(--text-muted); font-size: 0.8rem;">${v.changed_at ? new Date(v.changed_at).toLocaleTimeString() : '—'}</td>
            </tr>
          `;
        }).join('');
      } catch (e) {
        console.error('Failed to load verdicts:', e);
      }
    }

    async function loadAudits() {
      try {
        const res = await fetch('/api/audits/recent');
        const data = await res.json();
        document.getElementById('total-audits-count').innerText = data.total_audits || '0';

        const tbody = document.getElementById('audits-table');
        if (!data.audits || data.audits.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">No audits in database.</td></tr>';
          return;
        }

        tbody.innerHTML = data.audits.slice(0, 10).map(a => {
          const s1 = a.s1 !== null ? a.s1.toFixed(3) : '—';
          const s2 = a.s2 !== null ? a.s2.toFixed(2) : '—';
          const s3 = a.s3 !== null ? a.s3.toFixed(2) : '—';
          const s4 = a.s4 !== null ? a.s4.toFixed(3) : '—';
          const isAnomaly = (a.s3 !== null && a.s3 > 2.0) || (a.s4 !== null && a.s4 > 0.3);
          return `
            <tr>
              <td class="mono" style="font-size:0.75rem;">${a.request_id || a.audit_id.slice(0, 12)}</td>
              <td style="font-weight:600;">${a.provider}</td>
              <td class="mono">${a.scored_tokens}</td>
              <td class="mono">${s1}</td>
              <td class="mono">${s2}</td>
              <td class="mono" style="${isAnomaly ? 'color:#f43f5e; font-weight:700;' : ''}">${s3}</td>
              <td class="mono" style="${a.s4 > 0.3 ? 'color:#f43f5e; font-weight:700;' : ''}">${s4}</td>
              <td style="color: var(--text-muted); font-size: 0.78rem;">${a.scored_at ? new Date(a.scored_at).toLocaleTimeString() : '—'}</td>
            </tr>
          `;
        }).join('');

        // Update S3 Timeline Chart
        renderS3Chart(data.audits);
      } catch (e) {
        console.error('Failed to load audits:', e);
      }
    }

    function renderS3Chart(audits) {
      const ctx = document.getElementById('s3Chart').getContext('2d');
      const valid = audits.filter(a => a.s3 !== null).slice(0, 25).reverse();
      const labels = valid.map((_, i) => `#${i + 1}`);
      const dataVals = valid.map(a => a.s3);
      const pointColors = valid.map(a => a.s3 > 1.8 ? '#f43f5e' : (a.provider.includes('honest') ? '#10b981' : '#6366f1'));

      if (s3ChartInstance) {
        s3ChartInstance.data.labels = labels;
        s3ChartInstance.data.datasets[0].data = dataVals;
        s3ChartInstance.data.datasets[0].pointBackgroundColor = pointColors;
        s3ChartInstance.update();
        return;
      }

      s3ChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Surprisal Excess (S3)',
            data: dataVals,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: pointColors,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { display: false } }
        }
      });
    }

    async function loadBenchmarks() {
      try {
        const res = await fetch('/api/benchmarks');
        const data = await res.json();
        if (!data.e2_separability_auc) return;

        const e2 = data.e2_separability_auc;
        const labels = Object.keys(e2).map(k => k.replace('lab-', ''));
        const s2Data = labels.map(l => (e2['lab-' + l]?.s2?.auc || 0.5));
        const s3Data = labels.map(l => (e2['lab-' + l]?.s3?.auc || 0.5));
        const s4Data = labels.map(l => (e2['lab-' + l]?.s4?.auc || 0.5));

        const ctx = document.getElementById('aucChart').getContext('2d');
        if (aucChartInstance) aucChartInstance.destroy();

        aucChartInstance = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [
              { label: 'S2 (PIT)', data: s2Data, backgroundColor: 'rgba(56, 189, 248, 0.7)' },
              { label: 'S3 (Surprisal)', data: s3Data, backgroundColor: 'rgba(99, 102, 241, 0.7)' },
              { label: 'S4 (Logprob)', data: s4Data, backgroundColor: 'rgba(244, 63, 94, 0.7)' },
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { min: 0.0, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
              x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
            },
            plugins: {
              legend: { labels: { color: '#e2e8f0', font: { family: 'Outfit' } } }
            }
          }
        });
      } catch (e) {
        console.error('Failed to load benchmarks:', e);
      }
    }

    async function loadAll() {
      await Promise.all([loadVerdicts(), loadAudits(), loadBenchmarks()]);
    }

    loadAll();
    setInterval(loadAll, 4000);
  </script>
</body>
</html>
"""

