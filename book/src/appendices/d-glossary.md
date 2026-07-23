# Appendix D: Glossary

Precise, one-paragraph definitions of the book's load-bearing terms, alphabetized
and cross-linked. Each entry defines the term as the book uses it, not in full
generality: where a term has a whole chapter, the definition is the portable
summary and the chapter is the real treatment. Cross-references to other glossary
entries are *italicized*; cross-references to chapters use the chapter's name.

### Advantage

How much better an action (here, a generated token or completion) was than a
baseline expectation, $A = Q - V$ in the classical form: the action-value minus the
state-value. Subtracting the baseline leaves the *policy gradient* unbiased but cuts
its variance, because you reinforce what beat expectation rather than everything
with positive return. In *GRPO* the advantage is computed *group-relative*: the
reward of each completion minus the mean reward of its group, with no learned
*value* function at all. See *Actor-critic and GAE* and *The policy gradient
theorem*.

### AWQ (Activation-aware Weight Quantization)

A weight-only 4-bit quantization method that protects the small fraction of weight
channels which multiply against large-magnitude activations, scaling them before
quantizing so their precision survives. It stores 4-bit integers plus a per-group
(typically group-128) FP16 *scale*, costing ~4.5 effective bits per parameter
(~0.56 byte). It is the book's serving workhorse because quantizing Qwen3-14B to
~7.7 GiB frees several GiB of *KV cache*, buying concurrency and context. Served in
vLLM with `--quantization awq_marlin`. See *Quantization: theory and formats*.

### Backdoor criterion

A graphical rule for choosing what to condition on so a causal effect is
identifiable: a set of variables $Z$ satisfies it (relative to a treatment-outcome
pair) if $Z$ blocks every path from treatment to outcome that starts with an arrow
*into* the treatment, and $Z$ contains no descendant of the treatment. Blocking
those backdoor paths removes *confounding* without opening a *collider*. In this
book it is the tool for asking whether a measured "reasoning delta" between two
models is a clean effect or a spurious association through a shared cause. See
*Identification: backdoor and front-door* and *d-separation*.

### BF16 (bfloat16)

A 16-bit floating-point format laid out (1 sign, 8 exponent, 7 mantissa), keeping
FP32's full exponent range while spending only 7 bits on precision ($2^{-7} \approx
0.008$). The full exponent means nothing overflows or underflows the way it does in
FP16, so BF16 training needs no loss scaling. It is 2 bytes per element and the
default working dtype for both inference and training on the Blackwell baseline
machine. See *Tensors, autograd, and number formats*.

### Bootstrap CI (confidence interval)

An interval estimate for a metric (accuracy, *pass@k*, a mean reward) computed by
resampling the eval set with replacement many times, recomputing the metric on each
resample, and reading percentiles off the resulting distribution. It requires no
parametric assumption about the metric's sampling distribution, which is why it is
the book's default for putting error bars on eval scores and deciding whether a
delta between two models is real or run-to-run jitter. See *The statistics of
evals*.

### Chunked prefill

A vLLM scheduling technique that breaks a long prompt's *prefill* into fixed-size
chunks and interleaves them with ongoing *decode* steps, instead of running the
whole prefill in one blocking pass. It smooths inter-token latency for other
requests when a large prompt arrives, at a small throughput cost. Enabled with
`--enable-chunked-prefill`. See *vLLM internals* and *prefill/decode*.

### Continuous batching

The serving strategy that lets sequences enter and leave a running batch at every
decode step, rather than waiting for a fixed batch to finish together (static
batching). It keeps the GPU busy by backfilling finished slots immediately, and it
raises *decode* throughput by amortizing each weight read across more concurrent
tokens. It is why aggregate throughput and per-sequence latency are different axes.
See *vLLM internals* and *PagedAttention*.

### Cross-entropy

The training loss of a language model: the negative log-probability the model
assigns to the actual next token, averaged over the sequence, $-\sum_t \log
p_\theta(x_t \mid x_{<t})$. Minimizing it makes the model's predicted distribution
match the data distribution; its exponential is *perplexity*. It is the objective of
pretraining and *SFT*, and the reference point RL post-training departs from. See
*The language-modeling objective*.

### d-separation

The graphical criterion that reads conditional independence off a *DAG*: two
variables are d-separated given a conditioning set $Z$ if every path between them is
blocked by $Z$ (blocked at a chain or fork whose middle node is in $Z$, or at a
*collider* whose middle node and all its descendants are outside $Z$). d-separation
is the bridge between the graph and the statistics: separated in the graph implies
independent in any distribution the graph describes. It is the machinery behind the
*backdoor criterion*. See *DAGs and d-separation*.

### DAG (directed acyclic graph)

A graph of variables with directed edges and no cycles, used here to encode causal
assumptions: an arrow $X \to Y$ asserts $X$ is a direct cause of $Y$. The DAG is the
object you reason over with *d-separation* and the *backdoor criterion* to decide
whether an eval comparison is confounded. See *DAGs and d-separation* and *The
ladder of causation*.

### DPO (Direct Preference Optimization)

A post-training method that optimizes a policy directly on preference pairs
(chosen vs rejected responses) without training a separate *reward model* or running
an RL loop. It rewrites the RLHF objective so the optimal policy's log-ratio against
a frozen reference *is* an implicit reward, turning alignment into a single
classification-style loss. It trades the flexibility of on-policy RL for stability
and simplicity. See *Direct alignment: DPO and family*.

### GAE (Generalized Advantage Estimation)

A method for estimating the *advantage* that interpolates between low-variance,
high-bias (one-step) and high-variance, low-bias (full Monte-Carlo) estimates via a
parameter $\lambda \in [0,1]$, applied to an exponentially-weighted sum of temporal-
difference residuals. It is the advantage estimator inside *PPO*. GRPO drops it in
favor of a group-relative baseline. See *Actor-critic and GAE*.

### GQA (Grouped-Query Attention)

An attention variant where multiple query heads share a single key/value head, so
the number of KV heads $H_{kv}$ is smaller than the number of attention heads. It is
the single most important lever on *KV cache* size: Qwen3-8B's 32 query heads share
just 8 KV heads, a 4x cache reduction that is what makes long-context serving fit on
16 GiB at all. See *KV cache arithmetic* and *Attention from first principles*.

### GRPO (Group Relative Policy Optimization)

The RL algorithm at the center of the book: for each prompt it samples a *group* of
$G$ completions, scores them, and sets each completion's *advantage* to its reward
minus the group's mean reward (optionally divided by the group's standard
deviation), then applies a *PPO-clip* update with a *KL divergence* penalty to a
reference policy. It drops PPO's learned *value* critic entirely, which is what
makes it fit on one 16 GiB GPU. Its pathologies (length bias, std collapse) and
fixes are catalogued in *GRPO*; its 16 GiB budget is in *GRPO on 16GB*.

### KV cache

The stored key and value vectors for every past position, kept so that *decode* does
not recompute them each step. Its per-token size is $2 \times L \times H_{kv} \times
d_h \times b$ bytes (keys and values, over all layers, KV-head width, bytes per
element), and it is the single most important number in single-GPU serving because
after the weights it is the entire budget for context and concurrency. See *KV cache
arithmetic*.

### LoRA (Low-Rank Adaptation)

A parameter-efficient fine-tuning method that freezes the base weights and learns a
low-rank update $\Delta W = BA$ (with $B \in \mathbb{R}^{d\times r}$, $A \in
\mathbb{R}^{r\times d}$, rank $r \ll d$) added to selected linear layers. Only the
small adapters carry gradients and optimizer state, which is why training memory
collapses to a fraction of full fine-tuning. Combined with a 4-bit frozen base it
becomes *QLoRA*. See *LoRA and QLoRA, mathematically*.

### MXFP4

A 4-bit *microscaling* floating-point format: each element is an E2M1 float (1 sign,
2 exponent, 1 mantissa, representing only eight magnitudes) and every block of 32
elements shares one 8-bit power-of-two (E8M0) *scale*. The shared scale slides each
block's coarse grid to where its numbers live, giving usable dynamic range at ~4.25
bits per parameter (~0.53 byte). gpt-oss-20b ships in it. See *Quantization: theory
and formats* and *Tensors, autograd, and number formats*.

### NF4 (4-bit NormalFloat)

The 4-bit quantization used by *QLoRA* for the frozen base weights: a non-uniform
grid whose 16 levels are the quantiles of a standard normal, matching the roughly
Gaussian distribution of neural-network weights so each level carries equal
probability mass. With double quantization (quantizing the per-block *scale*s too) it
costs ~4.13 bits per parameter. See *LoRA and QLoRA, mathematically*.

### PagedAttention

vLLM's memory manager for the *KV cache*: instead of one contiguous reservation per
sequence, it allocates the cache in fixed-size blocks (16 tokens by default) from a
shared pool, like virtual-memory paging. This eliminates the fragmentation and
over-reservation of contiguous caches, letting the card hold far more concurrent
sequences, and it is what makes *continuous batching* memory-efficient. See *vLLM
internals*.

### pass@k

An eval metric for generative tasks: the probability that at least one of $k$
sampled completions is correct, usually estimated unbiasedly from a larger sample of
$n$ generations per problem. It rewards a model that can find a solution given a few
tries, which is the natural metric for reasoning tasks with a *verifiable reward*.
pass@1 at temperature 0 is the greedy special case. See *Metrics and their math*.

### PPO-clip

The core update of Proximal Policy Optimization: maximize the *advantage* weighted by
the probability ratio between the new and old policy, but clip that ratio to
$[1-\epsilon, 1+\epsilon]$ so a single step cannot move the policy too far. The clip
is a cheap surrogate for a trust region, keeping updates stable without TRPO's
second-order machinery. *GRPO* uses the same clipped objective with a group-relative
advantage. See *Trust regions: from TRPO to PPO*.

### Perplexity

The exponential of the average *cross-entropy* per token, $\exp(-\frac{1}{N}\sum_t
\log p_\theta(x_t \mid x_{<t}))$, interpretable as the effective number of equally-
likely choices the model is deciding among at each step. Lower is better; it is the
book's intrinsic quality yardstick for measuring the cost of *quantization*. See
*The language-modeling objective* and *Quantization: theory and formats*.

### Policy gradient

The gradient of expected return with respect to policy parameters, which the policy
gradient theorem shows equals $\mathbb{E}[\nabla_\theta \log \pi_\theta(a\mid s)\,
\Phi]$ for a suitable weighting $\Phi$ (return, *advantage*, ...). It lets you
improve a stochastic policy by gradient ascent using only sampled trajectories and
their rewards, no model of the environment required. It is the foundation under
REINFORCE, PPO, and *GRPO*. See *The policy gradient theorem*.

### Prefill / decode

The two phases of autoregressive generation. **Prefill** runs the whole prompt
through the network in one parallel forward pass, filling the *KV cache* and
producing the first token; it is compute-bound (arithmetic intensity scales with
prompt length). **Decode** then generates one token per forward pass, reading the
cache and every weight to emit a single token; it is memory-bandwidth-bound
(intensity ~1). Almost every serving decision is really a choice about which of
these two accounts you are drawing from. See *Prefill, decode, and the roofline*.

### QLoRA (Quantized LoRA)

*LoRA* on top of a base model whose frozen weights are stored in 4-bit *NF4*: the
base is quantized to a fraction of its BF16 size and never updated, while small BF16
adapters carry all the training. It is what lets a 4B policy fine-tune within the 16
GiB budget (base ~1.9 GiB, adapters + optimizer ~0.5 GiB, the rest for activations).
See *LoRA and QLoRA, mathematically* and *Appendix A*.

### RLVR (Reinforcement Learning with Verifiable Rewards)

The training paradigm of the book's loop: instead of a learned *reward model*, the
reward is a deterministic program that checks whether an answer is correct (a math
solution that evaluates, code that passes tests, a format that parses). Because the
reward is a *verifiable reward*, it cannot be gamed the way a learned model can, and
it is exactly what an eval *scorer* already computes, which is the thesis's central
observation: evals are rewards. See *RLVR: reinforcement with verifiable rewards* and
*Scorers as rewards*.

### RoPE (Rotary Position Embedding)

The positional scheme in the model zoo: instead of adding a position vector, it
rotates each query and key by an angle proportional to its absolute position, so
that their dot product depends only on relative position. It is applied inside
attention, extends to longer contexts by scaling the rotation frequencies, and is a
fixed function with no learned parameters. See *Attention from first principles* and
*The transformer block*.

### Roofline

The performance model that plots achievable throughput against arithmetic intensity
(FLOPs per byte moved), capping it at $\min(P_{\text{peak}}, I \times BW)$: below the
ridge point $I_{\text{ridge}} = P_{\text{peak}}/BW$ a kernel is memory-bound, above
it compute-bound. On the baseline machine it yields the hard *decode* throughput
ceiling $\text{tok/s}_{\max} = BW / \text{bytes-per-token}$. See *Prefill, decode,
and the roofline*.

### RMSNorm (Root-Mean-Square Normalization)

The normalization layer in the modern transformer block: it rescales each activation
vector by its root-mean-square (no mean-subtraction, no bias), then applies a learned
per-dimension gain. It is cheaper than LayerNorm and is what Qwen3 and most current
open models use. See *The transformer block*.

### SDA (Space Domain Awareness)

The Thesis Thread's task domain: reasoning about objects in orbit and the space
environment (tracking, conjunction or close-approach screening, maneuver detection,
catalog correlation), where an answer can be checked against ground truth or orbital
mechanics rather than merely judged. That checkability is exactly what makes it a
verifiable-reward domain and the reason the thread uses it end to end. See *Building
the thesis task suite* (chapter 3.11).

### Verifiable reward

A reward signal produced by a deterministic checker rather than a learned model: it
returns a score by actually verifying the output (running the code, checking the
math, matching the required format). Its virtue is that it is hard to hack and
identical to what an eval *scorer* computes, which is what makes *RLVR* the bridge
between evaluation and training in this book. See *RLVR: reinforcement with
verifiable rewards* and *Reward hacking and Goodhart*.
