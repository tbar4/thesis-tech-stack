# Appendix A: VRAM arithmetic tables

Every serving and training decision in this book is an accounting problem against
one number: the 16 GiB of GDDR7 on the RTX 5080 (the baseline machine). This
appendix precomputes that accounting for every model and config the book uses, so
that when a chapter says "the weights are $W$ GiB" or "this leaves $M_{kv}$ for
cache" the number is already sitting here with its formula above it. Everything is
regenerable: each table is preceded by the equation that produced it, so when a
model revision or a dtype changes you re-run the arithmetic rather than trusting a
stale figure.

Two conventions hold throughout, both inherited from *Tensors, autograd, and
number formats* and *KV cache arithmetic*:

- **GiB vs GB.** Memory capacity is binary: $1\ \text{GiB} = 2^{30}\ \text{byte}$.
  Vendor throughput and bandwidth are decimal: $1\ \text{GB} = 10^{9}\ \text{byte}$.
  The card is 16 GiB of capacity; its bandwidth is ~960 GB/s. I keep the units
  explicit because mixing them is a ~7% error that hides exactly at the margin
  where things stop fitting.
- **Measured vs derived.** Every number here is *derived* arithmetic unless it
  carries the tag `(measured on the baseline machine — record value, date,
  driver)`. Derived numbers are exact given their inputs; the inputs that must be
  confirmed on-device (activation overhead, allocator slack, real weight-file
  sizes) are flagged at each table.

## Bytes per parameter, by dtype

The atom of every weight budget is `numel × bytes_per_element`, from *Tensors,
autograd, and number formats*. For the quantized formats the effective figure
includes the per-group scales, so it is not a clean power of two.

$$W_{\text{weights}} = N_{\text{params}} \times \beta_{\text{dtype}}\ \text{byte}$$

```admonish info title="Reading the dtypes"
A one-line gloss on the names in the column below, so the table reads without a
detour:

- **FP32 / FP16 / BF16 / FP8** are floating-point: a sign bit, an exponent (which
  buys range), and a mantissa (which buys precision); fewer total bits buy memory.
  BF16 keeps FP32's full exponent, its whole range, at half the bytes, which is why
  mixed-precision training lives in it.
- **INT8 / INT4** are integer quantization: map weights onto a small grid of evenly
  spaced integers with a stored scale factor.
- **NF4** ("NormalFloat4") is the 4-bit format QLoRA freezes weights in; its 16
  levels sit at the quantiles of a normal distribution (weights are roughly
  Gaussian), so it spends precision where the mass is. ~4.13 effective bits once the
  block scale is counted.
- **MXFP4** ("microscaling FP4") is the 4-bit format gpt-oss ships in: 4-bit floats
  sharing one 8-bit (E8M0) scale per small block.
- **AWQ / GPTQ** are not storage dtypes but post-training quantization *methods*:
  they decide *how* to round a trained model's weights down to ~4 bits (AWQ protects
  activation-salient channels; GPTQ uses second-order / Hessian information), and the
  result is stored as INT4-with-group-scales.

Chapter 1.1 (*Tensors, autograd, and number formats*) walks the float formats bit by
bit; chapter 2.3 (*Quantization: theory and formats*) derives the methods. Appendix D
glosses the acronyms.
```

| dtype | layout (s, e, m) | bits/param | $\beta$ (byte/param) | notes |
|---|---|---|---|---|
| FP32 | 1, 8, 23 | 32 | 4.0 | master weights, optimizer state |
| BF16 / FP16 | 1, 8, 7 / 1, 5, 10 | 16 | 2.0 | default working dtype on Blackwell |
| FP8 (e4m3 / e5m2) | 1, 4, 3 / 1, 5, 2 | 8 | 1.0 | inference weights / KV; native on Blackwell |
| INT4 / AWQ | integer + group scale | ~4.5 | ~0.56 | 4 bits + FP16 scale per group-128 |
| NF4 (+ double-quant) | 4-bit quantile + scale | ~4.13 | ~0.516 | QLoRA base weights; block 64, DQ block 256 |
| MXFP4 (e2m1 + E8M0) | 4-bit float + block scale | ~4.25 | ~0.53 | gpt-oss; block 32, shared power-of-two scale |

```admonish read-along title="Where the fractional bits come from"
INT4/AWQ store one FP16 scale (and sometimes a zero-point) per group of 128
weights: $4 + 16/128 = 4.125$ bits, and real checkpoints round up to ~4.5 with
zero-points and packing overhead. NF4 with double quantization is $4 + 32/64$
(one FP32 absmax per block of 64) compressed by a second quantization of those
absmaxes to $\approx 4 + 0.127 = 4.127$ bits. MXFP4 is $4 + 8/32 = 4.25$ bits.
These are derived from the *Quantization: theory and formats* chapter; confirm the
exact bit-width against the specific checkpoint, since packing conventions differ.
```

## Weight budgets for the serving repertoire

$$W = N_{\text{params}} \times \beta_{\text{dtype}}, \qquad W_{\text{GiB}} = \frac{W}{2^{30}}$$

Parameter counts are the published totals; verify against each model's
`config.json` and safetensors index, since revisions move them.

| Model | $N_{\text{params}}$ | dtype | $\beta$ | $W$ (byte) | $W$ (GiB) |
|---|---|---|---|---|---|
| Qwen3-8B | $8.2\times10^{9}$ | BF16 | 2.0 | $1.64\times10^{10}$ | 15.3 |
| Qwen3-8B | $8.2\times10^{9}$ | FP8 | 1.0 | $8.2\times10^{9}$ | 7.6 |
| Qwen3-14B | $1.48\times10^{10}$ | BF16 | 2.0 | $2.96\times10^{10}$ | 27.6 (does not fit) |
| Qwen3-14B | $1.48\times10^{10}$ | AWQ 4-bit | 0.56 | $8.3\times10^{9}$ | 7.7 |
| gpt-oss-20b | $2.1\times10^{10}$ total | MXFP4 | 0.53 | $\approx1.1\times10^{10}$ | ~10.4* |

*The ~10.4 GiB gpt-oss figure is the naive MXFP4-only estimate (all params at
0.53 byte). The shipped checkpoint is ~12-13 GiB because attention, embeddings,
and the router are kept at higher precision; only the expert MLPs are MXFP4.

```admonish vram-budget title="What actually fits, and why the repertoire is what it is"
The card is 16 GiB. vLLM claims a fraction via `--gpu-memory-utilization`; the
rest is weights + activation/CUDA-graph overhead (~1 GiB, measure it) + KV pool.

- **Qwen3-8B BF16** at 15.3 GiB is right at the ceiling: KV pool is near zero, so
  you run short `--max-model-len` and batch ~1, or drop to FP8 KV. This tension is
  the honest reason the repertoire also carries a 4-bit 14B.
- **Qwen3-14B AWQ** at 7.7 GiB leaves ~5.9 GiB of KV pool at `util=0.92`. This is
  the workhorse: quantizing the weights buys concurrency and context.
- **gpt-oss-20b MXFP4** at ~10.4 GiB is the *naive* figure: it MXFP4-counts all
  $2.1\times10^{10}$ params. The real checkpoint is larger, ~12-13 GiB, because
  attention, embeddings, and the router stay in higher precision and only the
  expert MLPs are MXFP4. So the KV pool it leaves is tighter than 10.4 suggests;
  confirm the on-disk safetensors size before promising context. Being an MoE, its
  *decode* cost is still set by active experts, not total params (see below).

The ~1 GiB overhead and the real weight-file sizes (gpt-oss especially) are the
placeholders to pin on the machine: record value, date, driver.
```

## Per-token KV-cache bytes

From *KV cache arithmetic*, one token caches a K and a V vector per layer, each of
width $H_{kv} \times d_h$:

$$\boxed{\ B_{\text{tok}} = 2 \times L \times H_{kv} \times d_h \times b\ }$$

with $L$ layers, $H_{kv}$ KV heads (GQA makes this smaller than the attention-head
count), $d_h$ head dim, $b$ bytes per cached element (2 for BF16, 1 for FP8), and
the leading 2 for keys and values. Pull $L$, $H_{kv}$, $d_h$ from `config.json`
(`num_hidden_layers`, `num_key_value_heads`, `head_dim`).

| Model | $L$ | $H_{kv}$ | $d_h$ | $B_{\text{tok}}$ BF16 | $B_{\text{tok}}$ FP8 |
|---|---|---|---|---|---|
| Qwen3-4B | 36 | 8 | 128 | 144 KiB | 72 KiB |
| Qwen3-8B | 36 | 8 | 128 | 144 KiB | 72 KiB |
| Qwen3-14B | 40 | 8 | 128 | 160 KiB | 80 KiB |
| gpt-oss-20b | 24 | 8 | 64 | 48 KiB (ceiling) | 24 KiB |

Worked, BF16: Qwen3-8B $= 2\times36\times8\times128\times2 = 147{,}456\ \text{byte}
= 144\ \text{KiB}$; Qwen3-14B $= 2\times40\times8\times128\times2 = 163{,}840 =
160\ \text{KiB}$; gpt-oss $= 2\times24\times8\times64\times2 = 49{,}152 = 48\
\text{KiB}$.

```admonish gotcha
gpt-oss-20b alternates sliding-window and full-attention layers, so its 48 KiB/tok
is an upper bound that overcounts at long context (the sliding layers saturate at
their window). Qwen3-4B and Qwen3-8B share $L=36, H_{kv}=8, d_h=128$, so they have
*identical* per-token KV cost, which matters when the 4B is the training policy and
the 8B is a serving target. Always confirm against the KV-block count vLLM prints
at boot.
```

## Total KV at representative context and concurrency

$$M_{\text{KV}} = N_{\text{seq}} \times S_{\text{ctx}} \times B_{\text{tok}},
\qquad
N_{\text{seq}}^{\max} = \left\lfloor \frac{M_{kv}}{S_{\text{ctx}} \times B_{\text{tok}}} \right\rfloor$$

Context and concurrency trade off on a hyperbola: double the promised context and
you halve the seats. Per-sequence KV (one sequence at the given context):

| $S_{\text{ctx}}$ | Qwen3-8B BF16 | Qwen3-8B FP8 | Qwen3-14B BF16 | gpt-oss BF16 (ceiling) |
|---|---|---|---|---|
| 4,096 | 0.56 GiB | 0.28 GiB | 0.625 GiB | 0.19 GiB |
| 8,192 | 1.12 GiB | 0.56 GiB | 1.25 GiB | 0.38 GiB |
| 16,384 | 2.25 GiB | 1.12 GiB | 2.50 GiB | 0.75 GiB |
| 32,768 | 4.50 GiB | 2.25 GiB | 5.00 GiB | 1.50 GiB |

```admonish vram-budget title="Concurrency for Qwen3-14B AWQ (KV pool ~5.9 GiB)"
Pool: $M_{kv} \approx 16 \times 0.92 - 7.7 - 1 \approx 5.9\ \text{GiB}$ (BF16 KV).

| $S_{\text{ctx}}$ | KV/seq | max concurrent seqs $N_{\text{seq}}^{\max}$ |
|---|---|---|
| 4,096 | 0.625 GiB | $\lfloor 5.9/0.625 \rfloor = 9$ |
| 8,192 | 1.25 GiB | $\lfloor 5.9/1.25 \rfloor = 4$ |
| 16,384 | 2.50 GiB | $\lfloor 5.9/2.50 \rfloor = 2$ |
| 32,768 | 5.00 GiB | $\lfloor 5.9/5.00 \rfloor = 1$ |

Switching KV to FP8 (`--kv-cache-dtype fp8`) halves $B_{\text{tok}}$ and doubles
every seat count for a fraction of a point of eval metric (measure it with the
*Quantization* harness). The 5.9 GiB pool depends on the ~1 GiB overhead estimate;
pin it on the machine and re-derive.
```

## Decode-throughput ceilings (bandwidth budget)

Not VRAM but the other side of the same coin. From *Prefill, decode, and the
roofline*, decode at batch 1 is memory-bound, so tokens per second cannot exceed
bandwidth divided by bytes read per token:

$$\text{tok/s}_{\max} = \frac{BW}{B_{\text{read}}}, \qquad BW = 9.6\times10^{11}\ \text{byte/s}, \qquad B_{\text{read}} \approx W_{\text{weights}}\ (\text{short ctx})$$

| Model | $B_{\text{read}}$ (byte/tok) | $\text{tok/s}_{\max}$ |
|---|---|---|
| Qwen3-8B BF16 | $1.64\times10^{10}$ | ~59 |
| Qwen3-14B AWQ | $8.3\times10^{9}$ | ~116 |
| gpt-oss-20b MXFP4 (active $3.6\times10^{9}$) | $\approx2.5\times10^{9}$ | ~384 |

These are ceilings, not predictions; the real measured tok/s sits under them by the
kernel-overhead + sampling + KV-read gap the roofline lab plots. Record measured
values with date and driver.

## QLoRA training budget for a 4B policy

The training budgets are where the memory chapter's "4-16x inference" claim gets
paid out. From *LoRA and QLoRA, mathematically*: the base weights are frozen in NF4
(no grads, no optimizer state); only the low-rank adapters are trainable, so
optimizer state is tiny. The four accounts are base + adapters + optimizer +
activations.

**Base (frozen, NF4 + double-quant).** $N_{\text{base}} \times \beta_{\text{NF4}}$
with $\beta_{\text{NF4}} \approx 0.516$:
$$4.0\times10^{9} \times 0.516 = 2.06\times10^{9}\ \text{byte} = 1.92\ \text{GiB}$$

**Adapter trainable params.** For LoRA rank $r$ on a linear of shape
$d_{\text{out}} \times d_{\text{in}}$, the update $BA$ adds $r(d_{\text{in}} +
d_{\text{out}})$ params. Summed over target modules $\{$q,k,v,o,gate,up,down$\}$ and
$L=36$ layers of Qwen3-4B ($d_{\text{model}}=2560$, q width $4096$, kv width
$1024$, MLP width $9728$) at $r=16$:

$$N_{\text{LoRA}} \approx 36 \times 16 \times \big[\underbrace{(2560{+}4096)}_{q} + \underbrace{2(2560{+}1024)}_{k,v} + \underbrace{(4096{+}2560)}_{o} + \underbrace{3(2560{+}9728)}_{\text{MLP}}\big] \approx 33\times10^{6}$$

**Adapter, grad, optimizer.** Adapters in BF16, grads in BF16, AdamW keeps two FP32
moments $m,v$ plus an FP32 master copy (from *Where memory goes: training vs
inference*, AdamW = 2 extra copies):

| Account | formula | bytes | GiB |
|---|---|---|---|
| base NF4 (frozen) | $4.0\text{e}9 \times 0.516$ | $2.06\times10^{9}$ | 1.92 |
| adapter params (BF16) | $33\text{e}6 \times 2$ | $6.6\times10^{7}$ | 0.06 |
| adapter grads (BF16) | $33\text{e}6 \times 2$ | $6.6\times10^{7}$ | 0.06 |
| AdamW $m,v$ (FP32×2) | $33\text{e}6 \times 8$ | $2.6\times10^{8}$ | 0.25 |
| FP32 master (optional) | $33\text{e}6 \times 4$ | $1.3\times10^{8}$ | 0.12 |
| **static subtotal** | | | **~2.4** |
| activations (ckpt) | measured | — | 2-5 |

```admonish vram-budget title="QLoRA on Qwen3-4B, r=16"
The static footprint is ~2.4 GiB; the swing variable is activations, which scale
with batch size, sequence length, and whether gradient checkpointing is on. With
checkpointing (trade compute to recompute activations in backward), a
seq-len-2048, small-batch QLoRA on the 4B fits with wide headroom on 16 GiB. The
adapter+optimizer accounts are trivially small precisely because QLoRA freezes the
base: that is the whole point. The activation figure is the one to measure on the
machine (record value, date, driver); everything else is exact.
```

## GRPO training budget on 16 GiB

GRPO (from *GRPO* and *GRPO on 16GB*) adds a generation phase to the QLoRA budget:
for each prompt it samples a group of $G$ completions, scores them, and takes a
group-relative advantage. So on top of the QLoRA static footprint you pay for (a)
the KV cache of the generation rollouts and (b) a reference-policy forward for the
KL term. With Unsloth the generation runs on the *same* 4-bit base, so there is no
second copy of weights; the extra cost is KV and activations.

**Generation KV.** $G$ completions per prompt, each up to $S_{\text{gen}}$ tokens
plus a prompt of $S_{\text{prompt}}$, at the 4B's $B_{\text{tok}} = 144\
\text{KiB}$. I use the actual *GRPO on 16GB* config ($S_{\text{prompt}}=256$,
$S_{\text{gen}}=768$, so $T=1024$) so this line matches that chapter's budget:

$$M_{\text{KV,gen}} = G \times (S_{\text{prompt}} + S_{\text{gen}}) \times B_{\text{tok}}$$

For $G=8$, $S_{\text{prompt}}=256$, $S_{\text{gen}}=768$ ($T=1024$):
$$8 \times 1024 \times 147{,}456 = 1.21\times10^{9}\ \text{byte} = 1.13\ \text{GiB}$$

This is the *live* KV floor; vLLM claims a larger slab (~2.5 GiB) sized by
`gpu_memory_utilization` and pages completions into it. The table below counts
the slab, and the 8-bit AdamW optimizer (Unsloth's default), so its total
reconciles with *GRPO on 16GB*'s ~6.9 GiB rather than the FP32-AdamW QLoRA table
above.

| Account | source | GiB |
|---|---|---|
| CUDA context + allocator + Triton cache | measure (per 7.2) | ~0.8 |
| base NF4 (frozen, shared serve+train) | QLoRA table | 1.93 |
| adapter + grad + 8-bit AdamW | QLoRA table (8-bit opt) | ~0.18 |
| generation KV, vLLM slab ($G{=}8$, 1024 tok; live floor 1.13) | formula above | ~2.5 |
| reference-policy KL (recompute, no weight copy) | activations | measured |
| training activations (ckpt) | measured | ~1.5 |
| **total** | | **~6.9** |

```admonish vram-budget title="GRPO levers on 16 GiB, in order of bluntness"
When GRPO OOMs, pull these in order (each is a term in the budget above):

1. **Group size $G$** — linear in generation KV. $G{:}8\to4$ halves the 1.13 GiB
   live KV (and the slab that pages it).
2. **Generation length $S_{\text{gen}}$** — linear in generation KV *and* in
   rollout time. Cap it to what the reward actually needs.
3. **KV dtype FP8** — halves $B_{\text{tok}}$, halving generation KV.
4. **Batch / gradient accumulation** — trade wall-clock for activation memory;
   accumulate over micro-batches instead of one big batch.
5. **Gradient checkpointing** — already assumed on; it is the difference between
   "fits" and "OOM" for activations.
6. **Policy size** — the repertoire caps training policies at $\le 4$B for exactly
   this reason; a 1.7B policy roughly halves base + activations.

The full budget lands at ~6.9 GiB (mirroring *GRPO on 16GB* line for line),
leaving ~9 GiB of headroom at `util=0.9`. Activations and the reference-KL
forward are the measured quantities; record them with date and driver, and
reconcile against `nvidia-smi` peak.
```

## Regenerating these tables

Every table above is `parameters × formula`. The one-file calculator that produces
them (weights, KV, decode ceiling, QLoRA/GRPO budgets) lives with the KV lab
(`predict.py` in *KV cache arithmetic*) and the roofline lab (`ceiling.py` in
*Prefill, decode, and the roofline*). When a driver update, a vLLM release, or a
model revision changes an input, re-run those scripts rather than editing a number
here by hand, and re-stamp any measured overhead with its new date and driver.
