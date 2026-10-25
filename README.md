# Verigate

> **A verification gateway that checks whether LLM providers actually serve the model they claim.**

Verigate sits between your client applications and open-weight LLM providers (e.g. Qwen, Llama, DeepSeek). It forwards user traffic with **< 1 ms added latency**, secretly samples a configurable fraction $r$ of completed requests, and audits them off the hot path by re-scoring outputs with a pinned reference copy of the model in a single batched prefill pass.

Statistical hypothesis tests against measured GPU/kernel nondeterminism produce a per-provider verdict: **`VERIFIED`**, **`INCONCLUSIVE`**, or **`SUSPECT`**.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  Client app ──────►│  Gateway (Go)                                │
  (OpenAI SDK)      │  • OpenAI-compatible /v1/chat/completions     │──────► Provider A
                    │  • Non-blocking tee for tokens & logprobs    │──────► Provider B
                    │  • Post-hoc HMAC selection (HMAC < r)        │──────► Substitution Lab
                    └───────────────┬──────────────────────────────┘
                                    │ audit job (prompt, output, params)
                                    ▼
                    ┌──────────────────────────────┐
                    │  Queue: Redis or PulseDB     │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  Verifier (PyTorch + Hugging Face)           │
                    │  • Single-pass prefill over prompt + output  │
                    │  • Signals S1–S4 (PIT, argmax, surprisal)    │
                    │  • Prompt & text deleted after scoring       │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  Stats Engine & SQLite Store                 │
                    │  • Mann-Whitney U shift test                 │
                    │  • Binomial 99th-percentile outlier test     │
                    │  • Bonferroni corrected multi-testing        │
                    │  • Hysteresis verdict state machine          │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  Dashboard & API (FastAPI)   │
                    │  http://localhost:8090       │
                    └──────────────────────────────┘
```

---

## Quickstart

### 1. Requirements

- **Go**: 1.22+
- **Python**: 3.10+ (PyTorch with CUDA recommended; CPU supported)
- **Queue**: Redis or PulseDB (e.g. `docker run -d -p 6379:6379 redis:7-alpine`)

### 2. Install & Configure

```bash
# Clone and enter workspace
git clone https://github.com/your-org/verigate.git
cd verigate

# Copy sample configuration
cp verigate.example.yaml verigate.yaml

# Set an audit secret (defense against selective honesty)
export VERIGATE_AUDIT_SECRET="random-secure-secret-hex"

# Setup python virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .\.venv\Scripts\Activate.ps1
pip install -e ./verifier
```

### 3. Run Components

In separate terminals:

```bash
# 1. Start the Substitution Lab (simulating honest & dishonest endpoints)
python -m verigate_verifier.cli lab up --port 9000

# 2. Build & run the Go Gateway (port 8080)
python -m verigate_verifier.cli gateway --config verigate.yaml

# 3. Start the Verifier Worker (consumes queue, prefill scores on GPU/CPU)
python -m verigate_verifier.cli verifier --config verigate.yaml

# 4. Start the Dashboard (port 8090)
python -m verigate_verifier.cli dashboard --config verigate.yaml
```

Open `http://localhost:8090` in your browser to view the real-time faithfulness matrix and Chart.js graphs.

### Or Run with Docker Compose

```bash
docker-compose up -d
```

---

## Benchmark Highlights

Evaluated on `Qwen/Qwen2.5-0.5B-Instruct` on an NVIDIA RTX 4050 GPU:

| Metric | Target | Measured Result | Status |
|---|---|---|---|
| **Gateway Throughput** | $\ge 2,000\,\text{req/s}$ | **6,656 req/s** (Non-streaming), **5,112 req/s** (Streaming) | 332% of target |
| **Gateway Added Latency** | $< 1.0\,\text{ms}$ | **$+0.54\,\text{ms}$ (p50)**, **$+0.83\,\text{ms}$ (p99)** | Passing |
| **4-bit Quantization Separation** | Clear separation | **ROC AUC = 0.9104** ($S_3$ Surprisal Excess) | Passing |
| **Tampered Logprob Detection** | Clear separation | **ROC AUC = 1.0000** ($S_4$ Logprob Divergence) | Passing |
| **Verifier Prefill Speedup** | $\ge 10\times$ | **1,080 tok/s** vs 36 tok/s generation (**30× speedup**) | Passing |
| **Verification Compute Cost** | Tunable | **0.167% of serving cost** at 5% audit rate | Passing |

See [`results/BENCHMARK_REPORT.md`](results/BENCHMARK_REPORT.md) and [`results/E8_REPORT.md`](results/E8_REPORT.md) for complete empirical tables and methodology.

---

## Running Tests

### Go Gateway Tests
Tests verify byte-identical SSE streaming passthrough, logprob stripping, audit-independent upstream bodies, and graceful queue outage handling:

```bash
cd gateway
go test -v ./...
```

### Python Verifier Tests
Tests verify sampling transforms, S1–S4 signal calculation, Kolmogorov-Smirnov test on randomized PIT uniformity (PRD §17), and verdict state machine transitions:

```bash
pytest -v verifier/tests
```

---

## Documentation

- [**Statistical Verification Method**](docs/method.md): Mathematical formulations of $S_1$–$S_4$, empirical null calibration, stratified Mann-Whitney U tests, and binomial outlier rates.
- [**Threat Model & Attack Taxonomy**](docs/threat-model.md): Detailed adversarial model covering attacks A1 through A9.
- [**Reproduction Guide**](docs/reproducing.md): Step-by-step instructions to replicate every benchmark and table from scratch.

---

## License

Apache-2.0

