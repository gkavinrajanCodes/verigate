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

Open `http://localhost:8090` in your browser to view the real-time faithfulness matrix.

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

## Verification Signals

- **S1 (Tie-aware argmax mismatch)**: Evaluated at $T=0$. Ignores positions where top-1 and top-2 margin is $< 0.5$ nats to tolerate honest kernel nondeterminism.
- **S2 (Randomized PIT score)**: For $T>0$. Maps vocabulary probabilities to uniform cumulative integral transforms. Shifts toward 1 when substitute models favor tokens the reference considers improbable.
- **S3 (Surprisal excess)**: Standardized difference between empirical negative log-probability and sequence entropy under the reference distribution.
- **S4 (Logprob consistency)**: Measures deviation between provider-reported logprobs and reference logits (catches fabricated logprobs, attack A8).
- **Early-token context check**: Evaluates signals over the first 32 tokens to detect prompt tampering (A6) or context truncation (A5) before self-conditioning.

---

## License

Apache-2.0
