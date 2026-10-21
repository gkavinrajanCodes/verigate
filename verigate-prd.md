# Verigate — Product Requirements Document

**A verification gateway that checks whether LLM providers actually serve the model they claim.**

| Field | Value |
|---|---|
| Codename | Verigate |
| Owner | Kavin Rajan |
| Status | Draft v1.0 |
| Last updated | October 2026 |
| Type | Open-source infrastructure + research project |

---

## 1. Summary

When a developer pays a third-party provider to run an open-weight model (Llama, Qwen, DeepSeek, Kimi, GLM, gpt-oss), they have no practical way to confirm the provider is running the exact model, at the advertised precision, on the full prompt. Providers have a financial incentive to cut corners: serve a quantized version, a smaller sibling model, a truncated context, or mix a cheaper model into a fraction of requests.

Verigate is an OpenAI-compatible gateway that sits between an application and its providers. It forwards traffic normally, then **audits a secret random sample of real requests off the hot path** by re-scoring the provider's output with a trusted reference copy of the model in a single forward pass (prefill). Statistical tests over many audited requests produce a per-provider verdict: **VERIFIED**, **INCONCLUSIVE**, or **SUSPECT**.

The project has two equal goals:

1. **Product:** a working gateway with near-zero added latency, a verifier service, and a dashboard.
2. **Research:** a rigorous, reproducible measurement of how well software-only verification works against real GPU nondeterminism, including where it fails.

---

## 2. Background and prior work

This is not an empty field. The project must position itself precisely against what exists.

**Benchmark-based vendor checks.** Moonshot AI open-sourced the Kimi Vendor Verifier (KVV), which replays official evaluation sets against third-party deployments and compares tool-calling behavior and benchmark scores to the official API, with a public leaderboard. OpenRouter launched "Exacto" endpoints that route to providers with measurably better tool-calling accuracy. Both prove the problem is real and widespread. Their limitation: they run **known benchmark prompts offline**, which a dishonest provider can detect and serve honestly ("benchmark evasion"), and they measure quality differences rather than model identity.

**Black-box auditing research.** "Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs" (Sunblaze/UC Berkeley, NeurIPS 2025) formalizes substitution as a hypothesis test and evaluates attacks including quantization, randomized substitution, and benchmark evasion. Its central finding is a direct challenge to this project: text-only statistical tests are query-hungry and miss subtle substitutions, and log-probability methods are defeated by inference nondeterminism. The authors argue hardware attestation (TEEs) is the only currently viable strong guarantee.

**Cooperative proofs.** TOPLOC (Prime Intellect, ICML 2025) has the provider commit to a compact locality-sensitive hash of intermediate activations during inference; a verifier recomputes with prefill, much faster than generation, and detects changes to model, prompt, or precision with tolerance for hardware nondeterminism. It requires the provider to cooperate by emitting commits.

**Nondeterminism.** Work such as Thinking Machines Lab's "Defeating Nondeterminism in LLM Inference" (2025) shows that much inference nondeterminism comes from batch-size-dependent kernels, not random noise, which suggests it can be characterized and calibrated.

### The gap Verigate targets

| Approach | Works without provider cooperation | Audits real traffic | Hard to evade | Continuous | Cheap |
|---|---|---|---|---|---|
| Benchmark replay (KVV, Exacto-style) | Yes | No | No | Periodic | Yes |
| Logprob comparison (as evaluated in literature) | Partly | Possible | Partly | Possible | Yes |
| TOPLOC commits | **No** | Yes | Yes | Yes | Yes |
| TEE attestation | **No** (needs special hardware) | Yes | Yes | Yes | Moderate |
| **Verigate** | **Yes** | **Yes** | **Better** (secret sampling of real traffic) | **Yes** | **Yes** (prefill, sampled) |

**Research hypothesis:** published negative results evaluated logprob methods mostly per-query or on small sample sets. Verigate tests whether combining (a) multiple complementary signals, (b) explicit calibration against measured nondeterminism, and (c) sequential testing across hundreds of real requests recovers practical detection power for coarse and medium substitutions, and precisely characterizes where it still fails (expected: FP8/INT8 vs BF16). A careful negative result is an acceptable and publishable outcome.

---

## 3. Goals and non-goals

### Goals

- G1. Detect model substitution, aggressive quantization (≤ 4-bit), context truncation, and randomized mixing by third-party providers of **open-weight** models.
- G2. Add **< 1 ms p99 latency** on the request hot path. All verification runs asynchronously.
- G3. Keep false alarms rare: **≤ 1% false SUSPECT verdicts** per provider per week on honest providers.
- G4. Make verification cost tunable: audit rate `r` controls verifier compute.
- G5. Be indistinguishable to providers: audited and non-audited requests must look identical upstream.
- G6. Ship a reproducible "Substitution Lab" that anyone can run on a laptop to reproduce every result.

### Non-goals (v1)

- Verifying closed models (GPT, Claude, Gemini). There is no reference copy of the weights to compare against.
- Cryptographic or hardware-backed guarantees (TEEs, zero-knowledge proofs). Verigate gives statistical evidence, not proof.
- Being a full-featured commercial gateway (billing, multi-tenant auth, caching). Routing is minimal.
- Detecting harmless serving differences that do not change the output distribution (e.g., a different GPU type).

---

## 4. Users and use cases

| User | Need | Example |
|---|---|---|
| App developer using open-weight models via aggregators | Know which providers are faithful; exclude bad ones | "Provider X's Qwen endpoint has been SUSPECT since Tuesday; route around it." |
| Model lab (like Moonshot with KVV) | Continuous monitoring of how its model is served across vendors on real traffic | Public faithfulness dashboard per vendor |
| Inference provider | Prove it serves honestly; catch its own deployment bugs (wrong chat template, broken quantization) | Run Verigate in CI before releasing a new serving stack |
| Researcher | Reproducible benchmark of verification methods | Use the Substitution Lab to evaluate a new test statistic |

**Primary persona for v1:** a solo developer or small team calling 2–5 providers for the same open-weight model.

---

## 5. Threat model

### Adversary

A provider that wants to reduce its serving cost while charging for the advertised model. It controls the model, weights, kernels, precision, prompt handling, and response contents, including any logprobs it returns. It sees every request Verigate sends but does **not** know Verigate's audit secret or which requests are audited.

### Attacks in scope

| ID | Attack | Description |
|---|---|---|
| A1 | Model swap | Serve a different, cheaper model (e.g., 8B instead of 70B; a different family) |
| A2 | Aggressive quantization | INT4 / Q4 weights instead of BF16 |
| A3 | Mild quantization | FP8 / INT8 instead of BF16 (hardest case; may be partially out of reach) |
| A4 | Randomized mixing | Serve the honest model with probability `p`, a substitute otherwise |
| A5 | Context truncation | Silently drop part of long prompts (e.g., the middle) |
| A6 | Prompt tampering | Inject or modify a system prompt; wrong chat template |
| A7 | Selective honesty | Detect likely audit or benchmark traffic and serve it honestly |
| A8 | Fake logprobs | Return logprobs computed by the honest model or fabricated, while sampling from a substitute |
| A9 | Response caching/replay | Return cached outputs for similar prompts |

### Defenses by design

- **Against A7:** audit selection happens **after** the response completes, using `HMAC(secret, request_id) < r`. The upstream request is identical whether or not it gets audited. If logprobs are requested for any request, they are requested for **all** requests to that provider.
- **Against A8:** the core signals (S1–S3 below) use only the **output tokens**, never the provider's reported logprobs. Provider logprobs (S4) are a secondary signal, and a mismatch between provider-reported logprobs and reference logprobs is itself evidence.
- **Against A5/A6:** the reference model scores the output conditioned on the **full original prompt** with the officially pinned chat template, so outputs generated from a different context score as unlikely.
- **Against A4:** a per-request outlier-rate test (Section 8.4) is sensitive to mixtures, not just mean shifts.

### Out of scope

A provider that serves the honest model on exactly the audited requests is impossible by construction (it cannot know which ones are audited), so the residual risk is statistical: low mixing rates `p` take many requests to detect. The PRD reports detection time as a function of `p`.

---

## 6. System overview

```
                    ┌──────────────────────────────────────────────┐
  Client app ──────►│  Gateway (Go)                                │
  (OpenAI SDK)      │  • OpenAI-compatible /v1/chat/completions     │──────► Provider A
                    │  • Streams response back untouched            │──────► Provider B
                    │  • Tees output tokens + logprobs              │──────► Local lab providers
                    │  • Post-hoc audit selection (HMAC < r)        │
                    └───────────────┬──────────────────────────────┘
                                    │ audit job (prompt, output, params)
                                    ▼
                    ┌──────────────────────────────┐
                    │  Queue: PulseDB (RESP)       │   (Redis-compatible; Redis works too)
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  Verifier (Python, PyTorch + Transformers)   │
                    │  • Loads pinned reference model(s)           │
                    │  • Batched prefill over prompt + output      │
                    │  • Computes per-token signals S1–S4          │
                    │  • Writes per-request scores                 │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │  Stats engine + store (SQLite)                │
                    │  • Null calibration per (model, signal)      │
                    │  • Rolling-window tests per (provider,model) │
                    │  • Verdict state machine                     │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  Dashboard + API (FastAPI)   │
                    └──────────────────────────────┘

  Substitution Lab (separate): local "providers" exposing OpenAI-compatible
  endpoints that are honest, noisy-honest, or dishonest in controlled ways.
```

### Why this split

- **Gateway in Go:** the hot path must add < 1 ms; Go gives easy concurrency, good streaming support, and low overhead. (Rust is an acceptable alternative.)
- **Verifier in Python:** reference scoring needs PyTorch and Hugging Face Transformers.
- **Queue in PulseDB:** dogfoods your own database through its Redis-compatible protocol. Any Redis-compatible server must also work, so a PulseDB bug never blocks the research.
- **SQLite for results:** durable, zero-ops, plenty fast for this volume.

---

## 7. Functional requirements

### 7.1 Gateway

| ID | Requirement | Priority |
|---|---|---|
| GW-1 | Expose `POST /v1/chat/completions`, compatible with the OpenAI Chat Completions format, streaming and non-streaming | P0 |
| GW-2 | Route by `model` field to a configured provider; support explicit provider selection via header `X-Verigate-Provider` | P0 |
| GW-3 | Stream upstream chunks to the client without buffering the full response | P0 |
| GW-4 | Tee the stream: accumulate output text, token strings, token logprobs (if present), `finish_reason`, usage | P0 |
| GW-5 | Per-provider config flag `request_logprobs: true` adds `logprobs: true, top_logprobs: 5` to **every** upstream request to that provider; strip these fields from the client response unless the client requested them | P0 |
| GW-6 | After the response completes, compute `HMAC-SHA256(audit_secret, request_id)`; if its value mapped to [0,1) is `< r`, enqueue an audit job | P0 |
| GW-7 | Audit rate `r` configurable per provider (default 0.05) | P0 |
| GW-8 | Never block or slow the client response on enqueue; if the queue is unavailable, drop the audit and increment a counter | P0 |
| GW-9 | Return header `X-Verigate-Request-Id` on every response | P1 |
| GW-10 | Prometheus metrics: request count, upstream latency, gateway overhead, audits enqueued/dropped | P1 |
| GW-11 | Upstream retries and failover to another provider for the same model | P2 |
| GW-12 | Optional shadow mode: duplicate a sampled request to a second provider for cross-provider comparison | P2 |

### 7.2 Audit job format

Stored in the queue as JSON, deleted after scoring (TTL 24 h as a safety net).

```json
{
  "audit_id": "aud_01J...",
  "request_id": "req_01J...",
  "provider": "together",
  "model_alias": "qwen2.5-7b-instruct",
  "received_at": "2026-10-21T10:15:02Z",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "params": {"temperature": 0.7, "top_p": 0.9, "top_k": null, "max_tokens": 512, "seed": null},
  "output": {
    "text": "...",
    "tokens": ["Hello", ",", " world"],
    "token_logprobs": [-0.01, -0.3, -1.2],
    "top_logprobs": [[{"token": "Hello", "logprob": -0.01}, ...], ...],
    "finish_reason": "stop"
  },
  "prompt_hash": "sha256:...",
  "output_hash": "sha256:..."
}
```

### 7.3 Verifier

| ID | Requirement | Priority |
|---|---|---|
| VF-1 | Load one or more pinned reference models from a config mapping `model_alias → HF repo + revision (commit hash) + dtype + chat template` | P0 |
| VF-2 | Render the prompt with the pinned chat template exactly as the official serving stack does; fail loudly if the template is missing | P0 |
| VF-3 | Recover output token IDs: prefer provider token strings mapped to IDs; fall back to re-tokenizing the output text; record which path was used and flag mismatches | P0 |
| VF-4 | Run a single prefill pass over `prompt + output` and extract, for each output position, the full next-token logits | P0 |
| VF-5 | Compute signals S1–S4 (Section 8) and write a per-request score record | P0 |
| VF-6 | Batch jobs by length to maximize throughput; support CPU and CUDA | P0 |
| VF-7 | Expose throughput metrics: audits/s, tokens/s, queue depth, scoring latency | P1 |
| VF-8 | Delete prompt and output text after scoring; persist only hashes, lengths, and scores | P0 |
| VF-9 | Long outputs: score the first N tokens (default 512) to bound cost | P1 |

### 7.4 Stats engine and verdicts

| ID | Requirement | Priority |
|---|---|---|
| ST-1 | Store a null (honest) distribution per `(model_alias, signal)` built from calibration runs | P0 |
| ST-2 | Maintain a rolling window of the last W audited requests per `(provider, model_alias)`, default W = 200 | P0 |
| ST-3 | Run the window tests in Section 8.4 after every new score, or every 10 scores | P0 |
| ST-4 | Verdict state machine with hysteresis (Section 8.5) | P0 |
| ST-5 | Emit an event (log line + optional webhook) on any verdict change | P1 |
| ST-6 | Stratify results by prompt length bucket and temperature bucket to catch length-dependent cheating | P1 |

### 7.5 Dashboard and API

| ID | Requirement | Priority |
|---|---|---|
| UI-1 | Table of providers × models with current verdict, number of audits, last change time | P0 |
| UI-2 | Per provider: score distributions vs the null distribution, rolling statistic over time | P0 |
| UI-3 | `GET /api/status` returns verdicts as JSON | P0 |
| UI-4 | `GET /api/providers/{provider}/{model}` returns window statistics and history | P1 |
| UI-5 | Export a signed JSON report of a provider's audit history (HMAC-signed with a report key) | P2 |

### 7.6 CLI

```
verigate gateway --config verigate.yaml              # run the gateway
verigate verifier --config verigate.yaml             # run verifier workers
verigate calibrate --model qwen2.5-0.5b-instruct     # build null distributions
verigate audit --provider X --model Y --n 300 \
               --prompts prompts/mixed.jsonl          # active audit using a prompt set
verigate lab up                                      # start Substitution Lab providers
verigate lab eval --scenario all --out results/      # run detection experiments
```

---

## 8. Verification method

This is the core of the project. All signals are computed from the **reference model's distribution at each output position**, conditioned on the full prompt and the provider's preceding output tokens.

### 8.1 Notation

- `x`: the rendered prompt (pinned chat template applied).
- `y = (y_1 … y_n)`: provider output tokens.
- `p_t`: reference next-token distribution at position t, given `x, y_<t`, from raw logits.
- `q_t`: `p_t` transformed by the request's declared sampling parameters (temperature T, then top-k, then top-p, renormalized). Use the same order of operations as vLLM/HF; document it.
- `ε = 1e-6`: smoothing mass added to `q_t` before renormalizing so near-boundary tokens are never assigned zero probability.

### 8.2 Per-token signals

**S1 — Tie-aware argmax mismatch (greedy requests, T = 0).**
For each position, let `g_t = argmax p_t` and `margin_t = log p_t(top1) − log p_t(top2)`. Positions with `margin_t < δ` (default δ = 0.5 nats) are near-ties where honest nondeterminism can flip the choice; skip them.

```
S1 = (# eligible positions where y_t ≠ g_t) / (# eligible positions)
```

Honest providers should have S1 close to 0. After the first mismatch, the output diverges from the reference's greedy path but S1 remains valid because each position is conditioned on the provider's own prefix.

**S2 — Randomized PIT score (sampled requests, T > 0).**
Order the vocabulary by descending `q_t`. Let `F_t(y_t−)` be the total probability of tokens strictly before `y_t` in that order. Draw `V_t ~ Uniform(0,1)` from a seeded RNG and compute:

```
u_t = F_t(y_t−) + V_t · q_t(y_t)
```

If the provider truly sampled from `q_t`, the `u_t` values are i.i.d. Uniform(0,1) (the randomized probability integral transform applied to an autoregressive sequence). A substitute model that favors tokens the reference finds less likely pushes `u_t` toward 1. Aggregate per request:

```
z_t = Φ⁻¹(u_t)                 # standard normal under H0
S2  = (Σ z_t) / sqrt(n)        # ~ N(0,1) under ideal H0
```

Nondeterminism and the ε smoothing make the real null slightly different from N(0,1); that is why calibration (8.3) is required.

**S3 — Surprisal excess (any temperature).**

```
surprisal_t = −log q_t(y_t)
H_t         = entropy(q_t)
V_t         = variance of −log q_t(Y) for Y ~ q_t
S3 = Σ (surprisal_t − H_t) / sqrt(Σ V_t)
```

Mean zero under honest sampling; positive when the output is more surprising to the reference than it should be. Complements S2 and is easier to explain.

**S4 — Logprob consistency (only if the provider returns logprobs).**

```
d_t = provider_logprob(y_t) − log p_t(y_t)
S4  = median(|d_t|),  plus the 90th percentile of |d_t| as S4_p90
```

Large deviations indicate either a different model or fabricated logprobs. S4 is never used alone (attack A8).

**Context check (A5/A6).** Report S1–S3 separately for the first 32 output tokens. Truncation and prompt tampering mostly show up early, before the output becomes self-conditioning.

### 8.3 Calibration (building the null)

For each reference model, generate honest outputs under every nondeterminism source available, then score them with the reference exactly like real audits:

| Noise source | How to produce it |
|---|---|
| Batch-size variation | Generate the same prompts at batch sizes 1, 8, 32 |
| Hardware/kernel variation | CPU vs GPU; different thread counts; different serving engines (HF, vLLM, llama.cpp at full precision) |
| Precision of honest serving | BF16 and FP16 (both considered honest unless the provider advertises otherwise) |
| Sampling settings | T ∈ {0, 0.3, 0.7, 1.0}, top-p ∈ {1.0, 0.9} |
| Prompt mix | Short chat, long context (2k–8k tokens), code, JSON/tool-call, multilingual |

Store, per `(model, signal, temperature bucket, length bucket)`, the empirical null distribution (raw samples, at least 1,000 requests per bucket). Thresholds are null quantiles, never theoretical values.

For remote providers, the operator can additionally mark a "trusted" provider (e.g., the model creator's official API) and use its audited traffic as an extra null sample.

### 8.4 Window tests (per provider × model)

Run on the rolling window of the last W audits:

1. **Shift test.** Two-sample Mann–Whitney U test between window scores and the null sample, per signal. Catches A1, A2, A5, A6 (whole-distribution shifts).
2. **Outlier-rate test.** Let `τ` be the null's 99th percentile per signal. Under H0, about 1% of requests exceed `τ`. Count exceedances `k` in the window; binomial test against rate 0.01. Catches A4 (mixing), where only some requests are bad.
3. **Logprob test.** If S4 is available, same two tests on S4.

Combine p-values across signals and tests with a Bonferroni correction; also correct across the number of monitored `(provider, model)` pairs. Report every individual p-value in the dashboard.

**v2 upgrade:** replace fixed windows with an anytime-valid sequential test (e-values / test martingales or SPRT) so verdicts can be checked after every audit without inflating false alarms.

### 8.5 Verdict state machine

```
               < 50 audits                     combined p < α_alert (1e-4)
  INCONCLUSIVE ──────────────► (stay)   VERIFIED ───────────────────────► SUSPECT
       │  ≥ 50 audits and p ≥ α_ok (0.05)           ▲
       └───────────────────────────────────────────►│
                                                    │ 2 consecutive windows with p ≥ α_ok
  SUSPECT ───────────────────────────────────────────┘
```

- **INCONCLUSIVE:** too few audits, or results between thresholds.
- **VERIFIED:** consistent with the null at current sample size. Always shown with "at N audits, minimum detectable effect is X" so it never overclaims.
- **SUSPECT:** strong evidence of deviation. Shows which signals and tests fired.

The very strict `α_alert` is deliberate: falsely accusing a provider is the worst failure mode for this product.

---

## 9. Substitution Lab

The lab is how the project produces credible results on modest hardware. Every scenario is a local server exposing an OpenAI-compatible API, so the gateway treats it like any real provider.

### 9.1 Reference model

Default: **Qwen2.5-0.5B-Instruct** (fast on CPU), with **Qwen2.5-1.5B-Instruct** as the second tier. Both have official Hugging Face weights and widely available GGUF quantizations for llama.cpp. Pin exact revisions in config.

### 9.2 Scenarios

| Scenario | Implementation | Attack |
|---|---|---|
| `honest` | HF Transformers, BF16/FP32, batch 1 | — |
| `honest-noisy` | Same model, varied batch sizes / threads / device; or llama.cpp at F16 | — (must NOT be flagged) |
| `q8` | llama.cpp GGUF Q8_0 | A3 |
| `q6` | llama.cpp GGUF Q6_K | A3 |
| `q4` | llama.cpp GGUF Q4_K_M | A2 |
| `q2` | llama.cpp GGUF Q2_K | A2 (sanity check, should be easy) |
| `swap-small` | Serve a smaller model of the same family when the reference is 1.5B | A1 |
| `swap-family` | Serve a different family of similar size | A1 |
| `mix-p` | Proxy that routes to `honest` with probability p and `q4` otherwise, p ∈ {0.9, 0.7, 0.5} | A4 |
| `truncate` | Proxy that drops the middle 30% of long prompts before forwarding to `honest` | A5 |
| `sysprompt` | Proxy that prepends a hidden system instruction | A6 |
| `template` | Serve with a wrong or default chat template | A6 (deployment bug) |
| `fake-logprobs` | Sample from `q4`, but report logprobs from `honest` | A8 |
| `temp-cheat` | Ignore requested temperature; always use T = 0.3 | Parameter tampering |

### 9.3 Workloads

Prompt sets stored as JSONL in `lab/prompts/`:

- `chat.jsonl`: everyday instructions (use a permissively licensed public instruction dataset; record the source and license).
- `long.jsonl`: 2k–8k token prompts built from public-domain books or permissively licensed documents.
- `code.jsonl`: short coding tasks.
- `json.jsonl`: structured output / tool-call style prompts.
- Mix ratio for headline experiments: 40% chat, 20% long, 20% code, 20% JSON.

### 9.4 Experiments to run

| ID | Question | Output |
|---|---|---|
| E1 | Do honest-noisy providers stay below thresholds? | False-alarm rate per signal, per bucket |
| E2 | Per-request separability for each attack | ROC curve and AUC per signal per scenario |
| E3 | How many audited requests until SUSPECT? | Detection time (median, p90) per scenario |
| E4 | Mixing sensitivity | Detection time vs mixing fraction p |
| E5 | Effect of temperature | Detection power at T = 0, 0.3, 0.7, 1.0 |
| E6 | Which signals matter | Ablation: S1 / S2 / S3 / S4 alone vs combined |
| E7 | Cost | Verifier tokens/s on CPU and GPU; compute cost as % of serving cost at r = 1%, 5%, 10% |
| E8 | Hot-path overhead | Gateway added latency p50/p99 at 1–200 concurrent clients, audit on vs off |
| E9 | Hardest case | Can Q8 be distinguished from honest-noisy at all? At what N? |

Every experiment writes CSV to `results/` and generates plots with a single command. Each run records git commit, model revisions, hardware, and seeds.

---

## 10. Non-functional requirements

| Category | Requirement |
|---|---|
| Latency | Gateway overhead p99 < 1 ms (excluding upstream time) at 100 concurrent streams |
| Throughput | Gateway sustains ≥ 2,000 req/s non-streaming against a mock upstream on a 4-core machine |
| Verifier cost | Prefill scoring ≥ 10× faster per token than generation on the same hardware |
| Reliability | Gateway serves traffic normally when the queue, verifier, or dashboard is down |
| Privacy | Prompt/output text exists only in the queue until scored, then is deleted; persisted data contains hashes and scores only; audit sampling can be disabled per API key |
| Security | Audit secret loaded from environment variable, never logged; dashboard bound to localhost by default; provider API keys never logged |
| Reproducibility | Every result reproducible from a tagged commit, pinned model revisions, and a seed |
| Portability | Runs on Linux and macOS; CPU-only mode supported end to end |

---

## 11. Configuration

`verigate.yaml`:

```yaml
gateway:
  listen: "0.0.0.0:8080"
  audit_secret_env: VERIGATE_AUDIT_SECRET
  queue_url: "redis://localhost:6379/0"        # PulseDB or Redis
  queue_key: "verigate:audits"

models:
  qwen2.5-0.5b-instruct:
    reference:
      hf_repo: "Qwen/Qwen2.5-0.5B-Instruct"
      revision: "<pin-commit-hash>"
      dtype: "bfloat16"
      chat_template: "official"                # from tokenizer_config.json at that revision
    max_scored_tokens: 512

providers:
  lab-honest:
    base_url: "http://localhost:9001/v1"
    api_key_env: ""
    models: { qwen2.5-0.5b-instruct: "qwen2.5-0.5b-instruct" }   # alias -> provider's model id
    request_logprobs: true
    audit_rate: 0.10
  lab-q4:
    base_url: "http://localhost:9004/v1"
    models: { qwen2.5-0.5b-instruct: "qwen2.5-0.5b-instruct" }
    request_logprobs: true
    audit_rate: 0.10
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: OPENROUTER_API_KEY
    models: { qwen2.5-7b-instruct: "qwen/qwen-2.5-7b-instruct" }
    pin_upstream_provider: "Together"          # aggregator-specific routing option, if supported
    request_logprobs: true
    audit_rate: 0.05

stats:
  window: 200
  min_audits: 50
  alpha_ok: 0.05
  alpha_alert: 0.0001
  tie_margin_nats: 0.5
  smoothing_eps: 1.0e-6

storage:
  sqlite_path: "./data/verigate.db"
```

When auditing through an aggregator, results are only meaningful if the request is pinned to a single upstream provider; otherwise the verdict mixes several providers. Check the aggregator's current documentation for provider-pinning options.

---

## 12. Data model (SQLite)

```sql
CREATE TABLE audits (
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
CREATE INDEX idx_audits_pm ON audits(provider, model_alias, received_at);

CREATE TABLE null_samples (
  model_alias  TEXT, signal TEXT, temp_bucket TEXT, len_bucket TEXT,
  value REAL, source TEXT, created_at TEXT
);
CREATE INDEX idx_null ON null_samples(model_alias, signal, temp_bucket, len_bucket);

CREATE TABLE verdicts (
  provider TEXT, model_alias TEXT, state TEXT,
  n_audits INTEGER, combined_p REAL, details_json TEXT,
  changed_at TEXT,
  PRIMARY KEY (provider, model_alias, changed_at)
);
```

---

## 13. Repository structure

```
verigate/
├── README.md
├── LICENSE                      # Apache-2.0
├── verigate.example.yaml
├── docker-compose.yml           # gateway + PulseDB + verifier + dashboard + lab
├── gateway/                     # Go
│   ├── cmd/verigate-gateway/main.go
│   ├── internal/proxy/          # routing, streaming, tee
│   ├── internal/audit/          # HMAC selection, job building, enqueue
│   ├── internal/config/
│   └── internal/metrics/
├── verifier/                    # Python package
│   ├── verigate_verifier/
│   │   ├── worker.py            # queue consumer, batching
│   │   ├── reference.py         # model loading, chat templates, prefill
│   │   ├── tokens.py            # token ID recovery
│   │   ├── signals.py           # S1–S4
│   │   ├── stats.py             # window tests, combination, verdicts
│   │   ├── calibrate.py
│   │   └── api.py               # FastAPI dashboard + JSON API
│   └── tests/
├── lab/
│   ├── providers/               # scenario servers and proxies
│   ├── prompts/                 # JSONL workloads + SOURCES.md with licenses
│   ├── experiments/             # E1–E9 runners
│   └── plots/
├── results/                     # git-ignored, except published summaries
└── docs/
    ├── method.md                # math in full
    ├── threat-model.md
    └── reproducing.md
```

---

## 14. Technology stack

| Component | Choice | Notes |
|---|---|---|
| Gateway | Go (current stable release) | `net/http`, standard library streaming; no heavy frameworks |
| Verifier | Python 3.11+, PyTorch, Hugging Face Transformers | CUDA optional |
| Stats | NumPy, SciPy | Mann–Whitney, binomial test, normal quantiles |
| Lab: honest serving | Transformers; optionally vLLM on GPU | |
| Lab: quantized serving | llama.cpp `llama-server` (OpenAI-compatible) with GGUF quants | Confirm logprob support in the build used |
| Queue | PulseDB via RESP; Redis-compatible fallback | |
| Storage | SQLite | |
| Dashboard | FastAPI + server-rendered HTML + a small charting library | |
| Packaging | Docker Compose | One command brings up the full stack |
| CI | GitHub Actions | Go tests, Python tests, a 5-minute lab smoke test on CPU |

---

## 15. Milestones

Each milestone has an exit criterion. Do not move on until it is met.

| # | Weeks | Deliverable | Exit criterion |
|---|---|---|---|
| M0 | 1 | Reading + setup: TOPLOC paper, the model-substitution auditing paper and its code, KVV repo; reference model running locally; scoring script for one prompt | Can print per-token reference probabilities for a provider output |
| M1 | 1–2 | Signals S1–S3 as a standalone library + unit tests; lab `honest`, `honest-noisy`, `q4` running | AUC for `q4` vs `honest` reported per signal on 300 requests |
| M2 | 2 | Calibration pipeline and null store; E1 false-alarm experiment | Honest-noisy false-alarm rate ≤ 1% at chosen thresholds |
| M3 | 2 | Go gateway: proxy, streaming tee, HMAC audit selection, enqueue; E8 overhead benchmark | Overhead p99 < 1 ms at 100 concurrent streams vs mock upstream |
| M4 | 1–2 | Verifier worker + stats engine + verdicts, end-to-end through the gateway | `q4` provider turns SUSPECT; honest stays VERIFIED in a live run |
| M5 | 2 | Full lab scenarios + experiments E2–E7, E9 | Results tables and plots generated by one command |
| M6 | 1 | Dashboard, Docker Compose, docs | Fresh clone to running demo in < 15 minutes |
| M7 | 1 | Real-world pilot: 2–3 real providers for one open-weight model, low audit rate | Published report with honest caveats |

Total: roughly 12–14 weeks part-time.

---

## 16. Success metrics

| Metric | Target |
|---|---|
| Gateway overhead p99 | < 1 ms |
| False SUSPECT rate on honest-noisy scenarios | ≤ 1% per simulated week |
| Detection of `q2`, `q4`, `swap-*`, `truncate`, `sysprompt` | SUSPECT within ≤ 200 audited requests |
| Detection of `mix-p` at p = 0.5 | SUSPECT within ≤ 500 audited requests |
| `q8` vs honest | Characterized: either a detection N or a clear demonstration it is not separable from noise |
| Verifier cost at r = 5% | Reported as % of serving compute; target < 5% |
| Reproducibility | Third party reproduces headline numbers from the README |

---

## 17. Testing strategy

- **Unit tests:** sampling transform (temperature/top-k/top-p) matches reference implementations on fixed logits; PIT values are uniform on synthetic sequences sampled from a known distribution (KS test p > 0.01 over 10k tokens); S1 tie handling; HMAC selection rate within ±0.5% of `r` over 100k IDs.
- **Golden tests:** fixed prompt + output pairs with stored expected signal values per model revision (detect accidental changes in scoring).
- **Gateway tests:** streaming passthrough is byte-identical to upstream; client-requested vs injected logprobs handling; queue outage does not affect responses.
- **Property test:** audited and non-audited requests produce byte-identical upstream request bodies.
- **Lab smoke test in CI:** honest vs q2 on 0.5B model, 100 requests, CPU, must separate.

---

## 18. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Nondeterminism makes FP8/INT8 indistinguishable (as prior work suggests) | Weaker headline result | Treat as a finding; focus claims on coarse attacks and mixing; quantify the boundary precisely |
| Chat template mismatch causes false alarms | False SUSPECT verdicts | Pin templates by revision; `template` scenario in lab; early-token signals help diagnose |
| Tokenization mismatch (provider token strings vs reference tokenizer) | Corrupted scores | Prefer provider tokens; flag `mismatch` audits and exclude from tests; report rate |
| Providers don't return logprobs | S4 unavailable | S1–S3 don't need them |
| Aggregators route to multiple upstreams | Mixed verdicts | Require provider pinning when auditing through aggregators |
| Large models too big to run as reference locally | Can't verify 70B-class models on a laptop | Lab uses small models; real pilots use models that fit available hardware, or rented GPU time for verification only |
| Legal/terms-of-service | Accusations could be contested | Report statistics and confidence, not accusations; share findings privately with providers first |
| Privacy of audited prompts | User data exposure | Self-hosted verifier only; text deleted after scoring; per-key opt-out |

---

## 19. Future work (post-v1)

- Anytime-valid sequential testing (e-values) to replace fixed windows.
- TOPLOC-style cooperative mode: a vLLM plugin that emits activation commits, with Verigate verifying them, so cooperative providers can earn a stronger "proven" badge.
- Routing integration: automatically deprioritize SUSPECT providers.
- Public faithfulness leaderboard for popular open-weight models, built from consenting users' audit statistics.
- Tool-call specific signals (argument schema validity, trigger rate), building on the KVV methodology.
- Multi-modal models.

---

## 20. References

- Ong et al., "TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference," ICML 2025. arXiv:2501.16007.
- "Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs," NeurIPS 2025. arXiv:2504.04715. Code: https://github.com/sunblaze-ucb/llm-api-audit
- Moonshot AI, Kimi Vendor Verifier. Blog: https://www.kimi.com/blog/kimi-vendor-verifier · Code: https://github.com/MoonshotAI/Kimi-Vendor-Verifier
- OpenRouter, "Provider Variance: Introducing Exacto." https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/
- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (blog, 2025).
- Gao et al., "Model Equality Testing: Which Model Is This API Serving?" (2024).

---

## 21. Glossary

- **Prefill:** running the model over a full known sequence in one forward pass to get the probability of every next token, without generating. Much cheaper per token than generation.
- **Reference model:** a trusted copy of the claimed model, at pinned weights and precision, run by the verifier.
- **Null distribution:** the distribution of a score when the provider is honest, measured empirically including real nondeterminism.
- **PIT (probability integral transform):** maps a sampled value to its cumulative probability; uniform when the sample really comes from the assumed distribution.
- **Audit rate (r):** fraction of real requests that get re-scored.
- **Mixing attack:** serving a substitute model on a fraction of requests.
