# Quantization: theory and formats

The last two chapters kept arriving at the same verdict: on 16GB, weight bytes are the tyrant. They set the decode ceiling (bandwidth over bytes-per-token) and they crowd out the KV cache. Quantization is the one lever that attacks both at once. Store weights in 4 bits instead of 16 and you read a quarter of the bytes per token (faster decode) and you free three quarters of the weight footprint (more KV, more concurrency). The catch is that you are throwing away information, and the entire craft is throwing it away where the model will not miss it. This chapter builds the error theory from scratch, then explains AWQ, GPTQ, FP8, and MXFP4 both conceptually and mathematically, and finishes by measuring the perplexity cost of two schemes on a real model.

## Theory

### Uniform quantization and its error

The simplest quantizer maps a real value $x$ in some range onto one of $2^n$ evenly spaced levels. Pick a scale $s$ (the spacing between levels) and, optionally, a zero-point offset $z$. Symmetric uniform quantization (no offset) is:

$$q = \text{round}\!\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot q$$

with $q$ clamped to the representable integer range $[-2^{n-1}, 2^{n-1}-1]$. The reconstructed $\hat{x}$ differs from $x$ by the rounding error $e = \hat{x} - x$. Because rounding sends $x$ to the nearest multiple of $s$, the error lies in $[-s/2, +s/2]$.

```admonish derivation title="The variance of uniform quantization error"
Model the rounding error $e$ as uniform on $[-s/2, s/2]$, a good approximation when $s$ is small relative to the signal's variation. A uniform distribution on an interval of width $s$ has variance:

$$\sigma_e^2 = \frac{1}{s}\int_{-s/2}^{s/2} e^2\, de = \frac{1}{s}\left[\frac{e^3}{3}\right]_{-s/2}^{s/2} = \frac{s^2}{12}$$

So the mean-squared quantization error is $s^2/12$: it grows with the square of the step size. Now tie $s$ to the bit budget. If the values span a range $R = x_{\max} - x_{\min}$ and we have $2^n$ levels, then $s = R / 2^n$ and:

$$\sigma_e^2 = \frac{R^2}{12 \cdot 2^{2n}}$$

Each bit added halves $s$, which quarters the error variance, i.e. **+6 dB of signal-to-quantization-noise per bit**. This is the fundamental exchange rate of quantization, and it tells you two things at once: dropping from 16-bit to 4-bit is a 12-bit reduction and a huge nominal error increase, so the only way it works is if $R$ is kept tiny, per small group of weights, so that the absolute step $s$ stays small even at 4 bits.
```

That last clause is the whole game. The error is $R^2 / (12 \cdot 2^{2n})$. You cannot afford more bits $n$, so you must shrink the range $R$ over which any single scale applies. That is what group-wise scaling does.

It is worth pausing on the numbers to feel how brutal the 16-bit to 4-bit drop is before group-wise scaling rescues it. Going from $n=16$ to $n=4$ is a factor of $2^{2 \times 12} = 2^{24} \approx 1.7\times10^{7}$ increase in error variance for a fixed range $R$. If you applied one scale across a whole weight matrix whose values span, say, a range of 2.0, the 4-bit step would be $2.0 / 16 = 0.125$, coarse enough to visibly wreck the model. Now shrink the range to a 128-weight group where the local span might be 0.2, and the step drops to $0.2 / 16 = 0.0125$, a tenfold improvement from range reduction alone, at the same 4 bits. That is the entire reason 4-bit inference is viable: not because 4 bits is inherently enough, but because a well-chosen local range makes 4 bits behave like many more. The refinements below (AWQ, GPTQ, MXFP4) are all further attacks on the same quantity, either shrinking the effective range for the weights that matter or repairing the error the fixed grid leaves behind.

### Group-wise scales

A per-tensor scale forces one $s$ across millions of weights whose magnitudes vary wildly; a single outlier inflates $R$ and blows up the step for everyone. **Group-wise quantization** partitions each weight matrix into small contiguous groups (typically 64 or 128 weights) and gives each group its own scale $s_g$ (and optionally zero-point). A weight $w_i$ in group $g$ becomes:

$$\hat{w}_i = s_g \cdot \text{round}\!\left(\frac{w_i}{s_g}\right), \qquad s_g = \frac{\max_{j \in g}|w_j|}{2^{n-1}-1}$$

Now $R$ is the dynamic range within a 128-weight neighborhood, not the whole tensor, so the effective step is far smaller and the error much lower for the same bit width. The cost is storage: each group carries a scale (usually FP16). For a group size of 128 that adds $16/128 = 0.125$ bits per weight, and an asymmetric zero-point adds a similar increment, so "4-bit" AWQ or GPTQ actually costs about 4.13 to 4.25 effective bits per parameter (the symmetric, scale-only floor is $4 + 0.125 = 4.125$). Footprint budgeting in the KV chapter rounds this up to a conservative ~4.25 to 4.5 bits (0.56 byte/param) to leave margin for packing and alignment overhead.

```admonish gotcha
Group size is a real knob. Smaller groups (64) mean lower error but more scale overhead and slower kernels; larger groups (128, 256) are leaner and faster but let outliers back in. 128 is the near-universal default because it sits at the knee of that tradeoff. If a quantized model is unexpectedly degraded, checking the group size is a cheap first move.
```

### Where the error actually hurts: outliers and activations

Here is the fact that separates naive quantization from the good schemes. Transformer weight matrices are not the problem; their activations are. A handful of hidden dimensions carry systematically large activation magnitudes (the "outlier features"), and the *product* $Wx$ is what feeds the next layer. Quantizing $W$ uniformly treats every weight column as equally important, but a weight column that multiplies a large-magnitude activation channel contributes far more to the output error than one that multiplies a near-zero channel. Uniform quantization is blind to this; the smart schemes are not.

### AWQ: activation-aware weight quantization

AWQ's insight is that not all weights matter equally, and the ones that matter are identifiable from *activation* statistics. If a weight column is multiplied by a large-magnitude input channel, protecting that column's precision protects the output. AWQ protects it not by giving it more bits (the format is fixed at 4) but by **scaling it up before quantizing and scaling the activation down to compensate**, which pushes the important weights into a range where the fixed step size represents them more faithfully.

```admonish derivation title="The AWQ scaling objective"
Consider one output: $y = Wx = \sum_i w_i x_i$. Insert a per-channel scale $s_i > 0$: rewrite $w_i x_i = (w_i s_i)(x_i / s_i)$. Quantize the scaled weight $w_i' = w_i s_i$ and fold $1/s_i$ into the (already small) activation quantization or the preceding layernorm, which is cheap. The reconstructed output is:

$$\hat{y} = \sum_i Q(w_i s_i)\,\frac{x_i}{s_i}$$

The quantization error on channel $i$ is $\big(Q(w_i s_i) - w_i s_i\big) \cdot x_i / s_i$. Scaling up a channel by $s_i > 1$ shrinks its *relative* rounding error (the step is fixed, the value is larger), and the $1/s_i$ on the activation side is applied in high precision, so the important channels come out cleaner. AWQ chooses the scales to minimize the output reconstruction error, searching over a scale parametrized by the channel's activation magnitude:

$$s_i = \big(\overline{|x_i|}\big)^{\alpha}, \qquad \alpha^\star = \arg\min_{\alpha}\ \big\| WX - Q_\alpha(W)X \big\|$$

where $\overline{|x_i|}$ is the mean absolute activation on channel $i$ (measured on a small calibration set) and $\alpha \in [0,1]$ is a single scalar swept per layer. So AWQ needs calibration data only to estimate $\overline{|x_i|}$; it never touches the labels and never backpropagates. It is a cheap, data-light, forward-only method, which is exactly why it is my default for the 14B.
```

### GPTQ: second-order weight quantization

GPTQ comes at the same goal from a different mathematical direction. Instead of rescaling channels, it quantizes weights **one column at a time and compensates the remaining, not-yet-quantized weights** for the error just introduced, using second-order (curvature) information about the loss.

```admonish derivation title="The GPTQ update, from the OBS view"
GPTQ descends from Optimal Brain Surgeon. The layer's objective is to keep the output close: minimize $\|WX - \hat{W}X\|_2^2$ over quantized $\hat{W}$. The Hessian of this quadratic in the weights of one row is $H = 2XX^\top$, the (scaled) input covariance. When you quantize weight $w_q$ to $\hat{w}_q$, incurring error $\delta_q = \hat{w}_q - w_q$, OBS says the optimal way to absorb that error into the remaining weights is:

$$\Delta w_{-q} = -\frac{\delta_q}{[H^{-1}]_{qq}}\, H^{-1}_{:,q}$$

i.e. push a correction into the surviving weights proportional to the inverse-Hessian column, so the output stays as close as possible. GPTQ applies this greedily column by column, using a Cholesky factorization of $H^{-1}$ to make the sweep efficient. The calibration data enters through $H = XX^\top$: GPTQ needs it to estimate the input covariance, more heavily than AWQ uses its activation means.
```

Conceptually: **AWQ reshapes the weights before rounding so the fixed grid fits the important ones; GPTQ rounds greedily and repairs the damage using curvature.** AWQ is cheaper, more robust to calibration-set choice, and tends to hold up better at low bit widths for the models in this repertoire; GPTQ can be marginally more accurate when its Hessian estimate is good but is fussier. Both land near 4.13 to 4.25 effective bits and both are served by vLLM through Marlin kernels.

### FP8: floating point, not integer

Integer quantization spends its bits on a uniform grid. FP8 spends them on a floating-point grid: a sign, a few exponent bits, a few mantissa bits. The `e4m3` format (1 sign, 4 exponent, 3 mantissa) gives non-uniform spacing, fine near zero and coarse for large magnitudes, which matches how neural network values are distributed (many small, few large). FP8 does not need group scales to handle dynamic range because the exponent handles range for free; it is the natural format for **activations and the KV cache**, and on Blackwell it has native tensor-core support. Weights at FP8 (8 bits) are less aggressive than 4-bit integer schemes, so on this card FP8 is where I reach for KV cache and activation precision, while INT4-group (AWQ) is where I reach for weight compression.

### MXFP4: microscaling 4-bit, and gpt-oss

MXFP4 is the format gpt-oss-20b ships in, and it is a clever hybrid of the two ideas above. It is a 4-bit floating-point element format (`e2m1`: 1 sign, 2 exponent, 1 mantissa) combined with a shared **micro-exponent scale per block of 32 elements**. So each element is a tiny 4-bit float, and every group of 32 shares an 8-bit exponent scale. This is group-wise scaling (small block, shared scale) married to a floating-point element (non-uniform grid), and the block of 32 is small enough that the shared scale keeps effective range tight. The upshot is roughly 4.25 bits per parameter with better outlier tolerance than INT4 at the same budget, which is why a 21B-parameter MoE fits and serves comfortably on 16GB. The relevant point for operations is that MXFP4 is a *native checkpoint format* for gpt-oss (I do not quantize it myself; it arrives that way) and vLLM serves it directly on Blackwell.

```admonish under-the-hood
The reason all these 4-bit schemes are fast and not just small is the dequantize-in-the-kernel trick. Weights sit in VRAM as 4-bit packed integers plus scales; the matmul kernel (Marlin for INT4, native FP4 paths for MXFP4 on Blackwell) loads the packed weights, dequantizes them to BF16/FP8 inside the streaming multiprocessor registers, and multiplies, all without ever writing dequantized weights back to memory. That is what makes quantization move the decode ceiling: the *bytes read from VRAM* are the 4-bit ones, so bandwidth-bound decode genuinely reads a quarter of the data. If the kernel dequantized to VRAM first, you would pay full 16-bit bandwidth and gain nothing on decode speed.
```

### What degrades, and where

Quantization error is not spread evenly across capabilities. Broad, redundant knowledge (common facts, fluent syntax) is robust; the model has many pathways to it. Narrow, precise operations degrade first: multi-step arithmetic, long-chain reasoning where a single wrong token derails the trace, exact format-following, and rare-token recall. For reasoning models this is exactly the sensitive spot, so a quantization that looks fine on perplexity can still lose points on a math eval. The rule I hold to for the thesis: **perplexity is a smoke test, task accuracy is the verdict.** Perplexity catches gross breakage cheaply; the eval suite in Part III catches the reasoning-specific damage that perplexity misses.

### The four formats, side by side

It helps to hold the repertoire's formats in one frame. Each is a different answer to "spend the bits on what?"

| Format | Element | Bits/param (eff.) | Scale granularity | Best for | In the repertoire |
|---|---|---|---|---|---|
| BF16 | float | 16 | none | reference fidelity | Qwen3-8B weights |
| INT4 group (AWQ/GPTQ) | integer | ~4.13-4.25 | per 128-weight group | weight compression | Qwen3-14B weights |
| FP8 e4m3 | float | 8 | per-tensor/channel | activations, KV cache | KV cache option |
| MXFP4 | float e2m1 | ~4.25 | per 32-element block | outlier-tolerant 4-bit | gpt-oss-20b weights |

The pattern reading down the table: integer formats spend a uniform grid and lean on group scales to survive; floating formats spend an exponent to handle range for free, which is why FP8 needs no group scales and why MXFP4 tolerates outliers better than INT4 at the same bit count. There is no universally best format; there is a best format for a given tensor's role and a given hardware's native kernels. On Blackwell all four have fast paths, which is what makes the whole repertoire practical on one card.

## Tooling

Two toolchains cover the repertoire. For AWQ and GPTQ, `llm-compressor` (the maintained successor to AutoAWQ/AutoGPTQ, from the vLLM/Neural Magic side) produces vLLM-loadable checkpoints and runs the calibration forward passes. For measuring the damage, I compute perplexity on a held-out text with a short `transformers` loop, and (in Part III) run the real task suite. Calibration data matters: use a few hundred sequences of text that resembles the serving distribution (for a reasoning model, include chain-of-thought-style text), because AWQ's activation means and GPTQ's Hessian are only as representative as the calibration set. A tiny or off-distribution calibration set is a common cause of avoidable degradation.

## Lab: quantize a small model two ways, measure the perplexity delta

The card cannot quantize a 14B comfortably for a quick lab, so I use a small model to make the mechanism visible and fast; the same commands scale to the repertoire. The artifact is `ppl_report.json` comparing baseline, AWQ, and GPTQ perplexity.

### Set up

```bash title="shell"
uv init quant-lab && cd quant-lab
uv add "llmcompressor>=0.3" "vllm>=0.6" "transformers>=4.45" datasets torch
```

### Quantize both ways

```python title="quant-lab/quantize.py"
"""Quantize a small model with AWQ and with GPTQ using the same calibration set."""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier

def run(model_id, scheme, out_dir):
    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda r: len(r["text"]) > 200).select(range(256))
    # NOTE: modifier kwargs drift across llmcompressor releases; scheme="W4A16"
    # already implies 4-bit weights at group size 128, so bits/group_size are not
    # passed separately. Confirm the signature against your installed version, e.g.
    #   uv run python -c "from llmcompressor.modifiers.awq import AWQModifier; help(AWQModifier)"
    if scheme == "awq":
        recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    else:
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(model=model_id, dataset=ds, recipe=recipe,
            output_dir=out_dir, max_seq_length=512, num_calibration_samples=256)
    print(f"wrote {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    run(args.model, "awq",  "out-awq")
    run(args.model, "gptq", "out-gptq")
```

```bash title="shell"
uv run python quantize.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Measure perplexity for each

```python title="quant-lab/perplexity.py"
"""Sliding-window perplexity on wikitext-2 test, identical for every model."""
import argparse, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

@torch.no_grad()
def perplexity(model_path, stride=512, max_len=2048):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda").eval()
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1",
                                    split="test")["text"])
    enc = tok(text, return_tensors="pt")
    ids = enc.input_ids.to("cuda")
    nll, count = 0.0, 0
    for begin in range(0, ids.size(1) - 1, stride):
        end = min(begin + max_len, ids.size(1))
        window = ids[:, begin:end]
        target = window.clone()
        target[:, :-stride] = -100  # only score the fresh stride
        out = model(window, labels=target)
        n_tok = (target != -100).sum().item()
        nll += out.loss.item() * n_tok
        count += n_tok
        if end == ids.size(1):
            break
    return math.exp(nll / count)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    args = ap.parse_args()
    report = {p: round(perplexity(p), 4) for p in args.paths}
    base = report[args.paths[0]]
    for p, v in report.items():
        report[p] = {"ppl": v, "delta_vs_base": round(v - base, 4)}
    with open("ppl_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
```

```bash title="shell"
uv run python perplexity.py --paths Qwen/Qwen2.5-0.5B-Instruct out-awq out-gptq
```

### What you should see

`ppl_report.json` lists three perplexities: the BF16 baseline first, then AWQ and GPTQ with their deltas. Both quantized deltas should be **small and positive** (perplexity rises a little when you throw bits away) and, for a well-behaved model with a decent calibration set, on the order of a few hundredths to a few tenths of a perplexity point. AWQ and GPTQ should land close to each other; which one wins by a hair depends on the calibration set and the model, and it is not worth over-reading a single run. What you must not see is a large jump (perplexity doubling, or NaN): that signals a broken calibration set, a wrong group size, or a layer that should have been left in higher precision. If the deltas look reassuringly tiny, resist the conclusion that quantization is free. Perplexity is the smoke test; the reasoning-specific damage shows up only when the Part III eval suite runs the quantized model on math and multi-step tasks, and that is where I make the real accept/reject call. Record the three perplexities, the calibration set size, the model revision, date, and driver next to the file.

```admonish substack-seed
"Where a 4-bit model actually breaks." Quantization gets sold as a free lunch: same model, quarter the memory, barely any accuracy loss. The perplexity numbers even back that up. But perplexity averages over everything, and the damage from dropping to 4 bits is not average, it is targeted. It lands on exactly the operations reasoning models are made of: multi-step arithmetic, long chains where one wrong token dooms the answer, precise format-following. This post walks the error theory (six decibels of noise per bit, and why group-wise scaling is the trick that makes 4 bits survivable), then shows the two dominant schemes as two philosophies: AWQ reshapes the weights so the important ones land cleanly on the grid, GPTQ rounds greedily and repairs the wreckage with curvature. The takeaway a reader can act on: quote task accuracy, never perplexity, when you claim a quantized model is "just as good."
```
