# Verigate Threat Model & Attack Taxonomy

## 1. Adversary Definition and Capabilities

The adversary is an economically rational inference provider offering an API for an open-weight model (e.g. `Qwen2.5-72B`, `Llama-3.1-70B`). The provider seeks to maximize profit by reducing operational compute costs (VRAM footprint, memory bandwidth, latency, tensor parallelism requirements) while charging the full rate for the advertised model.

### Adversary Controls:
- Model architecture, checkpoint, and loaded weights.
- Weight and activation numerical precision (BF16, FP16, FP8, INT8, INT4/NF4, GPTQ, AWQ).
- System prompts, chat templates, and tokenization pipeline.
- Generation runtime hyperparameters and kernels (e.g., FlashAttention, custom cutoffs).
- Upstream HTTP response headers, tokens, and returned logprobabilities.

### Adversary Limitations:
- Does **not** know the gateway's audit secret key (`VERIGATE_AUDIT_SECRET`).
- Does **not** know which incoming requests are selected for post-hoc auditing.
- Cannot distinguish audited from non-audited requests upstream because audit selection occurs **after** the full response is completed and buffered via HMAC hashing:
  $$\text{HMAC}(\text{secret}, \text{request\_id}) < r$$

---

## 2. Attack Taxonomy & Defense Mechanisms

| Attack ID | Attack Name | Adversary Technique | Verigate Detection Mechanism |
|---|---|---|---|
| **A1** | Model Swap | Serves a cheaper sibling model (e.g., 7B instead of 70B, or different family). | $S_3$ Surprisal Excess and $S_2$ PIT immediately spike into extreme outlier regions ($p \ll 10^{-20}$). |
| **A2** | Aggressive Quantization | Loads model in INT4 / NF4 quantization to fit into smaller VRAM (e.g. 1 GPU instead of 4). | $S_3$ Surprisal Excess achieves ROC AUC $\ge 0.91$, catching severe distribution tail distortions. |
| **A3** | Mild Quantization | Loads model in FP8 or INT8. | Stratified Mann-Whitney U tests over rolling windows ($N \ge 50$) detect subtle mean shifts. |
| **A4** | Randomized Mixing | Serves honest model with probability $p$, substitute with probability $1-p$. | **Binomial Outlier Test** on $\tau_{99}$ breaches catches low-frequency substitutions where mean shift is diluted. |
| **A5** | Context Truncation | Truncates long prompts or middle tokens to save prefill compute. | $S_{3,\text{early}}$ and $S_{2,\text{early}}$ evaluate the first 32 tokens conditioned on the full prompt. |
| **A6** | Prompt Tampering | Injects unauthorized system prompts or uses incorrect chat templates. | Evaluated with the official pinned chat template, leading to immediate early token surprisal mismatch. |
| **A7** | Benchmark Evasion | Detects known benchmark queries and serves them honestly; cuts corners on real user traffic. | **Neutralized by design**: Verigate audits live, production user traffic, not synthetic public benchmarks. |
| **A8** | Fake Logprobs | Samples from quantized model, but returns fake logprobs to mask cheating. | Core signals $S_1, S_2, S_3$ use output tokens only; $S_4$ logprob divergence detects synthetic noise. |
| **A9** | Response Caching | Replays cached completions for similar prompts. | High entropy mismatch and context misalignment trigger $S_3$ and $S_2$ alerts. |

---

## 3. Residual Risks and Practical Limits

1. **Ultra-Low Mixing Rates ($p_{\text{attack}} < 0.05$)**:
   - When a provider substitutes only 1 in 50 requests, detection requires larger sample windows ($W \approx 200 - 500$ audits) to achieve statistical significance at $\alpha_{\text{alert}} = 10^{-4}$.
2. **Identical Distillations**:
   - If a provider fine-tunes a substitute model to minimize KL divergence with the reference model on the target domain, separability depends on model capacity disparity.
