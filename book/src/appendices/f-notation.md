# Appendix F: Notation reference

The symbols the book's derivations use, grouped by domain and pinned to one meaning
each. Where a symbol is genuinely overloaded across domains (the classic offender is
$B$, which the book uses for the bootstrap-resample count, the LoRA
up-projection matrix, and an NF4 quantization block), the collision is flagged
so a reader jumping between chapters is never guessing. Each
table gives the symbol, its meaning as the book uses it, and the chapter where it is
first introduced.

The book leans on a few standing conventions: vectors and matrices are as written in
each chapter (no global bold/roman rule is imposed, because the source chapters set
their own); $\log$ is natural log throughout; expectations are over whatever the
subscript names; and "byte" quantities use binary prefixes for capacity (GiB) and
decimal for rate (GB/s), as in *The hardware baseline* and *Appendix A*.

## Reinforcement learning

| Symbol | Meaning | First introduced |
|---|---|---|
| $s$, $s_t$ | state (here, the prompt / partial generation at step $t$) | The RL problem |
| $a$, $a_t$ | action (here, the next token or a full completion) | The RL problem |
| $\pi_\theta(a \mid s)$ | policy: the model's probability of action $a$ in state $s$, parameters $\theta$ | The RL problem |
| $\theta$ | policy (model) parameters | The policy gradient theorem |
| $\pi_{\text{ref}}$ | frozen reference policy for the KL penalty | GRPO |
| $r$, $r(s,a)$ | reward; in RLVR, the verifiable checker's score | The RL problem / RLVR |
| $R$, $G_t$ | return: (discounted) sum of future rewards from step $t$ | Value functions and Bellman equations |
| $\gamma$ | discount factor, $\gamma \in [0,1]$ | Value functions and Bellman equations |
| $V(s)$, $V^\pi(s)$ | state-value: expected return from state $s$ under $\pi$ | Value functions and Bellman equations |
| $Q(s,a)$, $Q^\pi(s,a)$ | action-value: expected return taking $a$ in $s$, then $\pi$ | Value functions and Bellman equations |
| $A(s,a)$ | advantage, $A = Q - V$ | Actor-critic and GAE |
| $\hat{A}$ | estimated advantage (GAE, or group-relative in GRPO) | Actor-critic and GAE |
| $\lambda$ | GAE bias-variance interpolation parameter | Actor-critic and GAE |
| $\delta_t$ | temporal-difference residual, $r_t + \gamma V(s_{t+1}) - V(s_t)$ | Actor-critic and GAE |
| $\rho_t(\theta)$ | probability ratio $\pi_\theta / \pi_{\theta_{\text{old}}}$ in PPO-clip | Trust regions: from TRPO to PPO |
| $\epsilon$ | PPO clip half-width, ratio clipped to $[1-\epsilon, 1+\epsilon]$ | Trust regions: from TRPO to PPO |
| $\beta$ | KL-penalty coefficient in the RL objective | GRPO |
| $D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})$ | KL divergence from policy to reference | GRPO |
| $G$ | GRPO group size: completions sampled per prompt | GRPO |
| $i$ | GRPO group index, $i = 1 \ldots G$ | GRPO |
| $o_i$ | the $i$-th completion (output) in a group | GRPO |
| $J(\theta)$ | the RL objective (expected return) being maximized | The policy gradient theorem |

```admonish gotcha
$\epsilon$ is the PPO clip width here, but $\epsilon$ also names machine epsilon in
*Tensors, autograd, and number formats* and appears as a small constant in
*RMSNorm*. Context disambiguates: an RL objective versus a floating-point rounding
bound. Likewise $\beta$ is the KL coefficient in RL but bytes-per-parameter in the
quantization/VRAM tables (*Appendix A*); the RL chapters never mix the two.
```

## Probability and statistics

| Symbol | Meaning | First introduced |
|---|---|---|
| $\mathbb{E}[\cdot]$, $\mathbb{E}_{x\sim p}[\cdot]$ | expectation, over the distribution named in the subscript | The policy gradient theorem |
| $\mathrm{Var}[\cdot]$ | variance | REINFORCE and variance reduction |
| $\mathrm{Cov}[\cdot,\cdot]$ | covariance | REINFORCE and variance reduction |
| $p_\theta(x)$ | model's probability of $x$ | The language-modeling objective |
| $\hat{\mu}$, $\hat{\theta}$ | an estimator of a mean / parameter (hat = estimated from data) | The statistics of evals |
| $n$ | eval sample size (number of problems / items) | The statistics of evals |
| $k$ | in pass@k, the number of sampled completions per problem | Metrics and their math |
| $B$ | number of bootstrap resamples | The statistics of evals |
| $\mathrm{CI}_{95}$ | 95% confidence interval (percentile bootstrap by default) | The statistics of evals |
| $\alpha$ | significance level (e.g. 0.05) | The statistics of evals |
| $d$ | effect size (standardized mean difference, Cohen's $d$) | The statistics of evals |
| $\sigma$, $\hat{\sigma}$ | standard deviation and its estimate | The statistics of evals |
| $H(p)$ | entropy of distribution $p$ | The language-modeling objective |
| $H(p,q)$ | cross-entropy of $q$ relative to $p$ | The language-modeling objective |

```admonish gotcha
$d$ is doubly booked and worth watching: it is the statistical effect size here, the
head dimension $d_h$ / model dimension $d_{\text{model}}$ in the transformer tables
below, and appears as a plain dimension $d$ in the roofline matmul argument. The
subscript or the sentence resolves it; when a chapter needs both at once it writes
$d_h$ for the head dimension explicitly.
```

## Transformer / LLM dimensions

| Symbol | Meaning | First introduced |
|---|---|---|
| $d_{\text{model}}$ | model (hidden) width | The transformer block |
| $L$ | number of transformer layers (`num_hidden_layers`) | Where memory goes / KV cache arithmetic |
| $n_{\text{heads}}$, $H$ | number of attention (query) heads | Attention from first principles |
| $H_{kv}$ | number of key/value heads (GQA; `num_key_value_heads`) | KV cache arithmetic |
| $d_h$ | head dimension (`head_dim`, or $d_{\text{model}}/n_{\text{heads}}$) | Attention from first principles |
| $V$ | vocabulary size (context: transformer, not RL value) | Tokenization and embeddings |
| $S$, $S_{\text{ctx}}$ | sequence / context length in tokens | Prefill, decode, and the roofline / KV cache arithmetic |
| $t$ | token position index | The language-modeling objective |
| $x_t$ | the token (or activation) at position $t$ | Tensors, autograd, and number formats |
| $\ell$, $z$ | logits (pre-softmax scores over the vocabulary) | The language-modeling objective |
| $Q, K, V$ | query, key, value matrices in attention | Attention from first principles |
| $J_k$ | Jacobian of layer $k$ (autograd) | Tensors, autograd, and number formats |
| $\bar{x}_k$ | adjoint of activation $x_k$, $\partial L / \partial x_k$ | Tensors, autograd, and number formats |
| $N_{\text{params}}$ | total parameter count of a model | Appendix A |

```admonish gotcha
$V$ is overloaded three ways across the book: the RL state-value function $V(s)$, the
attention value matrix $V$, and the vocabulary size $V$. These live in different
parts (RL, attention, tokenization) and never appear in the same equation, but it is
the collision most likely to trip a reader skimming across parts. $L$ is layers in
the transformer tables and the scalar loss in the autograd derivation; again,
disjoint contexts.
```

## Inference, bandwidth, and the roofline

| Symbol | Meaning | First introduced |
|---|---|---|
| $B_{\text{tok}}$ | KV-cache bytes per token, $2 L H_{kv} d_h\, b$ | KV cache arithmetic |
| $B_{\text{read}}$ | bytes read from memory per generated token (decode) | Prefill, decode, and the roofline |
| $b$ | bytes per cached element (2 BF16, 1 FP8) | KV cache arithmetic |
| $M_{kv}$ | VRAM pool available for the KV cache | KV cache arithmetic |
| $N_{\text{seq}}$ | number of concurrent sequences | KV cache arithmetic |
| $W$, $W_{\text{weights}}$ | weight footprint in bytes | Appendix A |
| $I$ | arithmetic intensity, FLOPs per byte moved | Prefill, decode, and the roofline |
| $I_{\text{ridge}}$ | ridge-point intensity, $P_{\text{peak}}/BW$ | Prefill, decode, and the roofline |
| $BW$ | memory bandwidth (~960 GB/s on the baseline card) | Prefill, decode, and the roofline |
| $P_{\text{peak}}$ | peak compute throughput (FLOP/s) | Prefill, decode, and the roofline |
| $P_{\text{achievable}}$ | $\min(P_{\text{peak}}, I \times BW)$ | Prefill, decode, and the roofline |
| $\text{tok/s}_{\max}$ | decode throughput ceiling, $BW / B_{\text{read}}$ | Prefill, decode, and the roofline |

```admonish gotcha
$b$ (lowercase) is bytes-per-cached-element in the KV formula; the quantization
table below deliberately writes bit-width as $b_{\text{bits}}$, not $b$, to keep
the two apart. $B$ (uppercase) is triply booked, the bootstrap-resample count in
the statistics table, the LoRA up-projection matrix, and an NF4 quantization
block (*Parameter-efficient fine-tuning*). The KV chapter writes $b$ for element
bytes and always states "2 for BF16, 1 for FP8" inline so the meaning travels
with the symbol.
```

## Number formats and quantization

| Symbol | Meaning | First introduced |
|---|---|---|
| $s$ | sign bit (floating point); also the quantization *scale* (context) | Tensors, autograd, and number formats |
| $E$ | stored (biased) exponent field | Tensors, autograd, and number formats |
| $B_{\text{bias}}$ | exponent bias (127 FP32/BF16, 15 FP16, ...) | Tensors, autograd, and number formats |
| $p$ | mantissa bit-count | Tensors, autograd, and number formats |
| $f$ | fractional part of the significand, $f \in [0,1)$ | Tensors, autograd, and number formats |
| $u$ | unit roundoff, $2^{-(p+1)}$ | Tensors, autograd, and number formats |
| $\epsilon_{\text{mach}}$ | machine epsilon, $2^{-p}$ | Tensors, autograd, and number formats |
| $\Delta$ | spacing between representable values (1 ulp), $2^{e-p}$ | Tensors, autograd, and number formats |
| $q$ | stored integer code in integer quantization | Tensors, autograd, and number formats |
| $s_{\text{q}}$ | quantization scale, real $\approx s_{\text{q}}(q - z)$ | Tensors, autograd, and number formats |
| $z$ | zero-point (integer offset in asymmetric quantization) | Tensors, autograd, and number formats |
| $g$ | group/block size for group-wise scales (e.g. 128 AWQ, 32 MXFP4, 64 NF4); chapter 6.3 writes the NF4 block size as $B$ | Quantization: theory and formats |
| $b_{\text{bits}}$ | bit-width of the quantized element (4, 8, 16) | Quantization: theory and formats |
| $\beta$ | effective bytes per parameter for a dtype (VRAM tables) | Appendix A |

```admonish read-along
This table is the reference the *Quantization: theory and formats* derivations and
the *Appendix A* budgets both point back to. When those chapters write $s$ for a
quantization scale, it is the same $s$ shown here, not the floating-point sign bit;
the surrounding equation always makes clear which. The effective-bytes symbol
$\beta$ is what turns a bit-width into the byte-per-parameter figures the VRAM tables
multiply by $N_{\text{params}}$.
```

## Parameter-efficient fine-tuning (LoRA / QLoRA)

The LoRA/QLoRA symbols from *LoRA and QLoRA, mathematically* and *GRPO on 16GB*.
This group collides hard with the RL and statistics tables above, by inheritance
from the source chapters rather than by accident; every clash is flagged below.

| Symbol | Meaning | First introduced |
|---|---|---|
| $r$ | LoRA rank: inner dimension of the low-rank update (collides with reward $r$) | LoRA and QLoRA, mathematically |
| $\alpha$ | LoRA scaling in the $\alpha/r$ update strength (collides with significance level $\alpha$) | LoRA and QLoRA, mathematically |
| $A$ | LoRA down-projection, $A \in \mathbb{R}^{r \times k}$ (Gaussian-initialized) | LoRA and QLoRA, mathematically |
| $B$ | LoRA up-projection, $B \in \mathbb{R}^{d \times r}$ (zero-initialized) (collides with bootstrap count $B$) | LoRA and QLoRA, mathematically |
| $\Delta W$ | the low-rank weight update, $\Delta W = \frac{\alpha}{r} B A$ | LoRA and QLoRA, mathematically |
| $B$ (block) | NF4 quantization block size in elements (QLoRA uses 64); the quantization table above calls this $g$ | LoRA and QLoRA, mathematically |

```admonish gotcha
This group is a collision minefield with the RL and statistics tables, and every
clash is deliberate so the symbols match the source chapters. $r$ is the LoRA rank
here but the reward $r(s,a)$ in the RL table; $\alpha$ is the LoRA scaling factor
here but the significance level in the statistics table; and $B$ is triply booked,
the LoRA up-projection matrix, the bootstrap-resample count (statistics table), and
the NF4 block size the quantization table writes as $g$. Chapter 6.3 sets these
symbols, so the appendix follows it rather than renaming; the surrounding equation
is always what disambiguates.
```
