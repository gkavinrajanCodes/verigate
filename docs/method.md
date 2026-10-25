# Verigate Verification Method

## 1. Overview and Core Hypothesis

Verigate statistically verifies whether a third-party inference provider is faithfully serving a claimed open-weight language model without modification, aggressive quantization, truncation, or prompt tampering.

Verification is framed as an online hypothesis testing problem:
$$\begin{aligned}
H_0 &: \text{Provider generates outputs from the advertised model } M \text{ under nominal parameters } \theta \\
H_1 &: \text{Provider substitutes a cheaper, smaller, or quantized model, alters prompts, or cheats sampling.}
\end{aligned}$$

Because inference nondeterminism (floating-point associativity across thread scheduling, kernel batching, and GPU architectures) prevents bitwise exact output matching, Verigate relies on **batched reference GPU prefill** to compute the exact conditional probability distribution over output tokens, combined with **stratified empirical null calibration**.

---

## 2. Mathematical Formalization of Signals

For a request with prompt $x$ and generated token sequence $y = (y_1, y_2, \dots, y_n)$:

Let $p_t$ denote the raw logits distribution over the vocabulary emitted by reference model $M$ conditioned on the true context:
$$p_t = P_M(\cdot \mid x, y_{<t})$$

The operational distribution $q_t$ applies temperature scaling $T$, top-$k$, top-$p$ (nucleus) truncation, and smoothing mass $\varepsilon = 10^{-6}$:
$$q_t = \text{Normalize}(\text{Truncate}(\text{Softmax}(p_t / T)) + \varepsilon)$$

### Signal 1: Tie-Aware Argmax Mismatch ($S_1$)
For greedy requests ($T \le 10^{-5}$), let $g_t = \arg\max_v p_t(v)$ and margin $\delta_t = \log p_t(\text{top}_1) - \log p_t(\text{top}_2)$. Positions with $\delta_t < \delta_{\text{tie}} = 0.5\,\text{nats}$ are near-ties susceptible to honest hardware jitter and are excluded:
$$S_1 = \frac{\sum_{t: \delta_t \ge 0.5} \mathbf{1}(y_t \ne g_t)}{\sum_{t} \mathbf{1}(\delta_t \ge 0.5)}$$

### Signal 2: Randomized Probability Integral Transform ($S_2$)
For stochastic sampling ($T > 0$), let $F_t(y_t^-) = \sum_{v \prec y_t} q_t(v)$ represent the cumulative mass of tokens ranked higher than $y_t$. Using an independent uniform draw $V_t \sim \text{Uniform}(0, 1)$:
$$u_t = F_t(y_t^-) + V_t \cdot q_t(y_t)$$
Under $H_0$, $u_t$ is strictly i.i.d. $\text{Uniform}(0, 1)$. Transforming through the standard normal inverse CDF $\Phi^{-1}$:
$$z_t = \Phi^{-1}(u_t) \sim \mathcal{N}(0, 1)$$
$$S_2 = \frac{1}{\sqrt{n}} \sum_{t=1}^n z_t$$
Under an unfaithful model that samples unlikely tokens under $M$, $u_t \to 1$ and $S_2 \gg 0$.

### Signal 3: Surprisal Excess ($S_3$)
Measures the standardized deviation of empirical surprisal from its expected entropy:
$$\text{surprisal}_t = -\log q_t(y_t), \quad H_t = \mathbb{E}_{Y \sim q_t}[-\log q_t(Y)], \quad V_t = \text{Var}_{Y \sim q_t}[-\log q_t(Y)]$$
$$S_3 = \frac{\sum_{t=1}^n (\text{surprisal}_t - H_t)}{\sqrt{\sum_{t=1}^n V_t}}$$
$S_3$ has expected value 0 under honest serving and strongly escalates when quantization shifts generation probabilities into low-density regions.

### Signal 4: Logprob Consistency ($S_4$)
When a provider returns logprobabilities $\ell_t$, the divergence from reference logprobs is computed:
$$d_t = \ell_t - \log p_t(y_t)$$
$$S_4 = \text{median}(|d_t|)$$
Fabricated logprobs or surrogate evaluations create massive spikes in $S_4$.

---

## 3. Stratified Empirical Null Calibration

Rather than relying on theoretical asymptotic distributions, Verigate generates an empirical null distribution $\mathcal{D}_0$ by scoring thousands of honest completions generated across:
- Temperatures: $T \in \{0.0, 0.3, 0.7, 1.0\}$
- Batch sizes: $B \in \{1, 8, 32\}$
- Honest precisions: BF16 and FP16

Quantiles and 99th percentiles $\tau_{99}$ are calculated per stratum $(M, T, \text{length})$.

---

## 4. Multi-Hypothesis Testing & Hysteresis State Machine

For a rolling window of $W$ audits:
1. **Distribution Shift Test**: Two-sample Mann-Whitney U test comparing window scores against the empirical null baseline.
2. **Outlier Rate Test**: Binomial test on the count of requests exceeding $\tau_{99}$ with null parameter $p_0 = 0.01$.
3. **Multi-Hypothesis Correction**: Bonferroni adjustment across tested signals and providers:
   $$p_{\text{combined}} = \min\left(1.0, m \cdot \min_{j} p_j\right)$$

### Verdict States
- **`INCONCLUSIVE`**: $N < 50$ audits or $p_{\text{combined}} \in [\alpha_{\text{alert}}, \alpha_{\text{ok}})$.
- **`VERIFIED`**: $N \ge 50$ and $p_{\text{combined}} \ge \alpha_{\text{ok}} = 0.05$.
- **`SUSPECT`**: $p_{\text{combined}} < \alpha_{\text{alert}} = 10^{-4}$.
- **Recovery Hysteresis**: Returning from `SUSPECT` requires 2 consecutive windows with $p \ge \alpha_{\text{ok}}$ (`clean_streak >= 2`).
