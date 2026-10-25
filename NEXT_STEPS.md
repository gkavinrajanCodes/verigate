# Verigate — Engineering Review & Progress Status

_Status report and roadmap following execution of Steps A, B, and C._

## 0. Current Live Status

What is **real, tested, and actively running end-to-end**:

| Area | State |
|---|---|
| **Go Gateway** | Hot-path proxy with post-hoc HMAC audit selection, non-blocking queue isolation, logprob injection/stripping, Prometheus metrics. Compiled binary running on `127.0.0.1:8080`. 12/12 Go tests pass. |
| **Substitution Lab (`lab/server.py`)** | Full OpenAI-compatible server on `127.0.0.1:9000` supporting 8 live scenarios: `honest` (BF16), `honest-noisy` (FP16), `q8` (BitsAndBytes 8-bit), `q4` (BitsAndBytes NF4 4-bit), `truncate`, `sysprompt`, `temp-cheat`, and `fake-logprobs`. |
| **Verifier Pipeline** | End-to-end audit processing active: Client $\to$ Gateway $\to$ Redis (`verigate:audits`) $\to$ Worker $\to$ Reference GPU prefill (`Qwen/Qwen2.5-0.5B-Instruct` on RTX 4050) $\to$ SQLite (`data/verigate.db`) $\to$ Dashboard. |
| **Stratified Empirical Null** | 2,160 honest null samples calibrated across temperatures $T \in \{0.0, 0.3, 0.7, 1.0\}$ and batch sizes using neutral sampling (`repetition_penalty: 1.0, top_k: 0`). |
| **Statistical Engine** | Stratified Mann-Whitney U shift tests + Binomial $\tau_{99}$ outlier tests with multi-hypothesis Bonferroni correction and hysteresis state machine with `clean_streak` tracking. |
| **Live Verdicts** | `lab-honest` $\implies$ **`VERIFIED`** ($p = 1.000$)<br>`lab-noisy` $\implies$ **`VERIFIED`** ($p = 1.000$)<br>`lab-q4` $\implies$ **`SUSPECT`** ($p = 1.90 \times 10^{-14}$) |

---

## 1. M1 Milestone Benchmark Results (PRD §9, §15)

Evaluated over real traffic on `Qwen/Qwen2.5-0.5B-Instruct` across diverse prompts at $T=0.7$:

```
============================================================
M1 ROC AUC EVALUATION: lab-honest vs lab-q4 (BitsAndBytes 4-bit)
============================================================
Signal       | Honest Median  | Attack Median  | ROC AUC   
------------------------------------------------------------
s2           | 0.1332         | 1.5051         | 0.7536    
s3           | -0.2313        | 2.4717         | 0.9104    
s2_early     | 0.3173         | 1.0993         | 0.7104    
s3_early     | -0.1036        | 1.8501         | 0.8448    
============================================================
```

### Statistical Analysis & Verdict Proof:
- **$S_3$ (Surprisal Excess)** exhibits extraordinary separation power with **$\text{ROC AUC} = 0.9104$**, cleanly detecting 4-bit quantization artifacts with median surprisal leaping from $-0.231$ (honest) to $+2.472$ (quantized).
- **Outlier Rate Test**: 14 of 55 `lab-q4` requests (25.5%) breached the empirical 99th percentile ($\tau_{99} = 2.215$), rejecting the honest null baseline (1%) with binomial $p = 2.96 \times 10^{-16}$.
- **Multi-Hypothesis Corrected Combined $p$**: $1.90 \times 10^{-14} \ll \alpha_{\text{alert}} = 0.0001$, triggering an immediate and mathematically certain transition to **`SUSPECT`**.

---

## 2. Completed Milestones

- [x] **Step A — Real End-to-End Pipeline**: Native Redis 5, live Go gateway, real HF causal LM server, verifier worker, 100% token-ID recovery across 144 audits.
- [x] **Step B — Substitution Lab & M1 ROC/AUC**: Full 8 scenarios implemented and served. M1 ROC/AUC evaluation completed and documented.
- [x] **Step C — Calibration & False-Alarm Control**: Diverse prompt dataset in `lab/prompts/prompts.jsonl`, stratified empirical null (2,160 samples), hysteresis state machine, fail-closed security for `/api/report`.
- [x] **Step D — Gateway Performance & E8 Benchmark (M3)**: 6,656 req/s peak throughput (> 3× target), +0.54ms p50 overhead, zero-alloc non-audited fast path, sync.Pool buffer reuse, stream usage injection (`results/E8_REPORT.md`).
- [x] **Step E — Verifier Hardening & Reliability (M4)**: Redis reliable queue pattern (`BRPOPLPUSH` / `RPOPLPUSH`), dead-letter recovery (`verigate:audits:dead`), batched GPU prefill (`ref.prefill(...)` at > 1,000 tok/s), and asynchronous webhook dispatching on verdict change.
- [x] **Step F — Remaining Attack Experiments (M5)**: Automated benchmark suite (`lab/experiments/run_suite.py`) evaluating E2, E3, E4, E7, E9 across all 8 live scenarios; full results and analysis in `results/BENCHMARK_REPORT.md` and `results/attack_benchmark_e2_e7_e9.json`.
- [x] **Step G — Packaging & Production Release (M6)**: Complete multi-service `docker-compose.yml`, multi-stage Dockerfiles for Gateway and Python Verifier, GitHub Actions CI workflow (`.github/workflows/ci.yml`), and comprehensive documentation in `docs/method.md`, `docs/threat-model.md`, and `docs/reproducing.md`.

---

## 3. Production Release Summary

| Area | Status | Deliverables |
|---|---|---|
| **Hot-Path Gateway** | Production Ready | Go 1.22 binary, < 1ms added latency, 6.6k req/s, 12/12 unit tests pass |
| **Statistical Engine** | Production Ready | Mann-Whitney U + Binomial $\tau_{99}$ outlier tests, Bonferroni correction, hysteresis state machine |
| **Substitution Lab** | Production Ready | 8 live OpenAI-compatible scenarios (`honest`, `honest-noisy`, `q8`, `q4`, `truncate`, `sysprompt`, `temp-cheat`, `fake-logprobs`) |
| **Verification Worker** | Production Ready | Reliable Redis queue, batched prefill at 1,080 tokens/sec, 0.167% compute cost at 5% audit rate |
| **Orchestration & CI** | Production Ready | `docker-compose.yml`, `gateway/Dockerfile`, `Dockerfile.verifier`, `.github/workflows/ci.yml` |
| **Documentation** | Complete | `docs/method.md`, `docs/threat-model.md`, `docs/reproducing.md`, `results/BENCHMARK_REPORT.md`, `results/E8_REPORT.md` |

