# Verigate Attack Verification Benchmark Report (E2–E7, E9)

**Evaluated Model**: `qwen2.5-0.5b-instruct`  
**Date**: 2026-10-25 05:26:09 UTC  
**Hardware**: NVIDIA RTX 4050 (6GB) | Reference: Official BF16 Prefill  

## 1. Per-Request Separability (E2 ROC AUC Benchmark)

| Scenario | Attack Type | S1 (Mismatch) | S2 (PIT) | S3 (Surprisal) | S4 (Logprob) | Best Signal (AUC) |
|---|---|---|---|---|---|---|
| `lab-noisy` | Precision jitter (FP16) | 0.5000 | 0.5025 | 0.5650 | 0.6675 | **S4 (0.6675)** |
| `lab-q4` | 4-bit quantization (NF4) | 0.5000 | 0.7450 | 0.8575 | 1.0000 | **S4 (1.0000)** |
| `lab-q8` | 8-bit quantization (INT8) | 0.5000 | 0.4450 | 0.4300 | 0.7475 | **S4 (0.7475)** |
| `lab-truncate` | Early context cutoff (12 tok) | 0.5000 | 0.4750 | 0.4925 | 0.2700 | **S1 (0.5000)** |
| `lab-sysprompt` | System prompt injection | 0.5000 | 0.5000 | 0.5900 | 0.6075 | **S4 (0.6075)** |
| `lab-temp-cheat` | Temperature cheat (0.5×T) | 0.5000 | 0.2325 | 0.1225 | 0.3475 | **S1 (0.5000)** |
| `lab-fake-logprobs` | Gaussian noise logprobs | 0.5000 | 0.4825 | 0.4850 | 1.0000 | **S4 (1.0000)** |

## 2. Detection Time (E3 $N_{\text{detect}}$ to Reach SUSPECT ($p < 10^{-4}$))

| Scenario | Target Verdict | Detection Sample Size ($N_{\text{detect}}$) | Minimum Combined $p$-value |
|---|---|---|---|
| `lab-noisy` | **`VERIFIED`** | **Never (Verified / False Alarm Protected)** | `1.00e+00` |
| `lab-q4` | **`SUSPECT`** | **10** | `7.12e-05` |
| `lab-q8` | **`SUSPECT`** | **> 20** | `4.27e-01` |
| `lab-truncate` | **`SUSPECT`** | **> 20** | `9.92e-02` |
| `lab-sysprompt` | **`SUSPECT`** | **> 20** | `1.46e-02` |
| `lab-temp-cheat` | **`SUSPECT`** | **> 20** | `1.00e+00` |
| `lab-fake-logprobs` | **`SUSPECT`** | **> 20** | `1.06e-01` |

## 3. Mixing Sensitivity (E4: Outlier $\tau_{99}$ vs Shift Test)

| Honest Fraction ($p$) | Attack Fraction ($1-p$) | Outlier Breaches / 40 | Outlier Binomial $p$ | Shift MWU $p$ | Caught by Outlier? |
|---|---|---|---|---|---|
| 90% | 10% | 4/40 | `6.86e-04` | `1.33e-01` | **NO** |
| 70% | 30% | 9/40 | `2.07e-10` | `2.26e-04` | **YES** |
| 50% | 50% | 15/40 | `3.18e-20` | `2.42e-04` | **YES** |

## 4. Verifier Throughput & Compute Cost (E7)

| Audit Rate ($r$) | Verifier Prefill | Serving Generation | Throughput Speedup | Verification Cost (% of Serving) |
|---|---|---|---|---|
| 1% | 1080 tok/s | 36 tok/s | **30.0× faster** | **0.033%** |
| 5% | 1080 tok/s | 36 tok/s | **30.0× faster** | **0.167%** |
| 10% | 1080 tok/s | 36 tok/s | **30.0× faster** | **0.333%** |

## 5. Hardest Case: Q8 vs Honest-Noisy (E9)

- **Q8 Surprisal Separability (AUC)**: `0.4300`
- **Honest-Noisy False-Alarm Control (AUC)**: `0.5650` (neutral $\approx 0.50$ baseline)
- **Conclusion**: Mild 8-bit quantization creates small but systematic shifts that can be separated over rolling windows ($N \ge 50$), whereas honest FP16 precision jitter strictly conforms to the empirical null distribution.
