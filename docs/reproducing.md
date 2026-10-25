# Reproducing Verigate Benchmarks and Results

This guide provides end-to-end instructions for reproducing every result, figure, and table from the Verigate project on standard developer hardware.

---

## 1. Prerequisites & Environment Setup

### System Requirements
- OS: Linux, macOS, or Windows 10/11
- Go 1.22+ (for Gateway)
- Python 3.10+ (for Verifier, Substitution Lab, and Dashboard)
- GPU: NVIDIA GPU with $\ge 6\,\text{GB}$ VRAM (CUDA), or CPU (slower prefill)
- Redis 5.0+ (or PulseDB)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/gkavinrajanCodes/verigate.git
cd verigate

# Set up Python virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e verifier/
pip install -r requirements.txt  # or install httpx, scipy, numpy, transformers, torch, accelerate

# Build Go gateway
cd gateway
go build -o verigate-gateway ./cmd/verigate-gateway
cd ..
```

---

## 2. Launching Services Locally

Run the following background processes (or use `docker-compose up`):

### Terminal 1: Redis
```bash
redis-server --port 6379
```

### Terminal 2: Substitution Lab (Serving 8 Live Scenarios)
```bash
python -m lab.server --port 9000 --scenarios "honest,honest-noisy,q8,q4,truncate,sysprompt,temp-cheat,fake-logprobs"
```

### Terminal 3: Verigate Gateway
```bash
export VERIGATE_AUDIT_SECRET="dev-secret-change-me"
./gateway/verigate-gateway --config verigate.yaml
```

### Terminal 4: Verifier Worker (GPU Batched Prefill)
```bash
python -m verigate_verifier.cli --config verigate.yaml verifier
```

### Terminal 5: Dashboard API & UI
```bash
python -m verigate_verifier.cli --config verigate.yaml dashboard --port 8090
```

---

## 3. Running Benchmark Experiments

### Milestone M3 (E8: Gateway Latency & Throughput)
```bash
# Start mock upstream
./gateway/mock-upstream --port 9099 &

# Run non-streaming and streaming benchmarks
./gateway/e8-bench --upstream http://127.0.0.1:9099 --gateway http://127.0.0.1:8080 --out results/e8_results.json
```

### Milestones M1 & M5 (E2–E7, E9: Full Attack Suite Benchmark)
```bash
python lab/experiments/run_suite.py --config verigate.yaml --n 25
```

The script automatically executes:
- **E2**: Per-request separability (ROC curve and AUC) for all 7 attacks
- **E3**: Detection sample size ($N_{\text{detect}}$ to reach `SUSPECT`)
- **E4**: Mixing sensitivity (outlier-rate vs shift tests across $p \in \{0.5, 0.7, 0.9\}$)
- **E7**: Verifier prefill throughput and compute cost analysis
- **E9**: 8-bit quantization ($Q_8$) separability against honest FP16 jitter

Results will be generated in:
- `results/attack_benchmark_e2_e7_e9.json`
- `results/BENCHMARK_REPORT.md`
- Dashboard UI on `http://127.0.0.1:8090`
