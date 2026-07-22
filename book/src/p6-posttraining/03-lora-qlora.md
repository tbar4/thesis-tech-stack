# LoRA and QLoRA, mathematically

Full fine-tuning of a 4B model on a 16GB card is a non-starter, and the arithmetic says so before I even try: the weights alone in BF16 are 8 GiB, and the Adam optimizer state (two FP32 moments plus an FP32 master copy) is another four to six times the weight footprint, so I am tens of GiB over budget before a single activation is stored. LoRA and its quantized cousin QLoRA are the two ideas that make training on this hardware possible at all, and they are not hacks, they rest on a specific and testable hypothesis about the geometry of fine-tuning updates and a specific and clever data format for 4-bit weights. This chapter derives both: the low-rank reparameterization that turns a billion-parameter update into a few-million-parameter one, and the NF4 quantization that squeezes the frozen base into 4 bits without the usual catastrophe. Then it does the thing the whole book is about, a rank sweep measured with real eval deltas through the chapter 3.7 machinery, so the choice of rank stops being folklore and becomes a number with a confidence interval.

## Theory

### The low-rank update hypothesis

Fine-tuning changes a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ into $W_0 + \Delta W$. The hypothesis behind LoRA is that for adapting a pretrained model to a downstream task, the *update* $\Delta W$ has low intrinsic rank, even though $W_0$ itself is full-rank. Intuitively, pretraining already learned the hard, high-dimensional structure of language; adapting it to "answer instructions" or "reason in this format" is a comparatively low-dimensional nudge, a rotation into a few task-relevant directions rather than a wholesale rewrite. If $\Delta W$ really lives near a rank-$r$ subspace with $r \ll \min(d, k)$, then I never need to represent the full $d \times k$ update, I can factor it.

```admonish derivation title="The LoRA reparameterization"
Freeze the pretrained weight $W_0 \in \mathbb{R}^{d \times k}$ and constrain the update to be a product of two thin matrices,

$$\Delta W = \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d,k). \tag{6.3.1}$$

The adapted layer computes, for input $x$,

$$h = (W_0 + \Delta W)\,x = W_0 x + \frac{\alpha}{r}\,B(Ax). \tag{6.3.2}$$

Only $A$ and $B$ are trainable; $W_0$ never receives a gradient. Count the parameters: the full update has $d k$ entries, the factored update has $r(d + k)$. For an illustrative $2560 \times 2560$ attention projection ($d = k = 2560$) with $r = 16$, that is $2560^2 = 6.55\text{M}$ versus $16 \cdot (2560 + 2560) = 81.9\text{k}$, an 80-times reduction *per matrix*, and it compounds over every targeted layer. Across the whole model, LoRA typically makes well under 1% of parameters trainable.

Two design choices make equation (6.3.2) behave. **Initialization:** $A$ is initialized from a small random Gaussian and $B$ is initialized to zero, so at step 0, $\Delta W = B A = 0$ and the adapted model is *exactly* the pretrained model. Training starts from the base behavior and moves away, rather than starting from a random perturbation, which is why LoRA fine-tunes are stable from the first step. **The $\alpha/r$ scaling:** the factor $\alpha/r$ decouples the learning-rate-like magnitude of the update from the rank. Without it, doubling $r$ would roughly double the norm of $\Delta W$ for the same $A, B$ scale, so you would have to re-tune the learning rate every time you changed rank. With it, $\alpha$ sets the effective update strength and $r$ sets the capacity, and the two knobs are (approximately) independent. The gradient only flows into $A$ and $B$:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top \frac{\partial \mathcal{L}}{\partial (\Delta W x)}\, x^\top, \qquad \frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial (\Delta W x)}\, (Ax)^\top, \tag{6.3.3}$$

and crucially the optimizer state (the memory-hungry part) is now sized to $A$ and $B$, not to $W_0$. That is the whole memory win: I still hold $W_0$ in VRAM to compute the forward pass, but I do not hold gradients or Adam moments for it.
```

### Rank, alpha, and target modules

Three knobs follow from equation (6.3.1). **Rank $r$** is the capacity of the update: how many independent directions $\Delta W$ can move in. Too small and the adapter underfits (it cannot express the task nudge); too large and you spend memory and risk overfitting a small dataset, with diminishing returns once $r$ exceeds the true intrinsic rank of the update. Common values are 8 to 64, and the rank sweep in the lab is exactly the experiment that tells you where the returns flatten *for your task*. **Alpha $\alpha$** is the update strength via the $\alpha/r$ scaling; the common convention $\alpha = 2r$ keeps the effective scale roughly constant as you vary rank, which is why the lab holds that ratio fixed so the sweep isolates capacity. **Target modules** is the set of weight matrices you attach adapters to. The attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are the classic minimal set; adding the MLP projections (`gate_proj`, `up_proj`, `down_proj`) covers where most of a transformer's parameters and most of its task-specific capacity actually live, at more memory. For reasoning fine-tunes I target all seven, because the MLP is where a lot of the procedural knowledge sits.

### NF4: quantizing the frozen base

LoRA shrinks the *trainable* footprint but I still hold the full $W_0$ in VRAM. QLoRA's second idea is to hold it in 4 bits instead of 16, using a format designed for the specific distribution neural-network weights actually follow. Pretrained weights are approximately zero-mean Gaussian, and a naive linear 4-bit grid (16 evenly spaced levels) wastes most of its levels in the tails where almost no weights live while under-resolving the dense center. NF4 (4-bit NormalFloat) fixes this by placing its 16 levels at the *quantiles* of a normal distribution, so each level is equally likely to be used, which is information-theoretically optimal for a Gaussian source.

```admonish derivation title="The NF4 construction"
NF4 is a per-block, quantile-based, 4-bit code. Build it in three moves.

**1. Quantile levels.** For a source $\mathcal{N}(0,1)$ with CDF $\Phi$, an information-optimal $k$-level code puts its levels at evenly spaced quantiles. NF4 uses $2^4 = 16$ levels. To guarantee an exact zero (so that a true zero weight quantizes to zero, which preserves sparsity and symmetry), the levels are built from two one-sided quantile grids, $2^3$ levels for the negative side and $2^3 + 1$ for the non-negative side, merged and de-duplicated at zero:

$$q_i = \frac{1}{2}\!\left(\Phi^{-1}\!\Big(\frac{i + 1}{2^{a} + 1}\Big) + \Phi^{-1}\!\Big(\frac{i + 2}{2^{a} + 1}\Big)\right), \tag{6.3.4}$$

evaluated on each side and then normalized so the levels span $[-1, 1]$. The result is a fixed 16-entry lookup table, denser near zero and sparser in the tails, exactly matching where Gaussian weights concentrate.

**2. Per-block absmax scaling.** Real weights are not unit-variance, and their scale varies across the tensor, so NF4 normalizes in small blocks. Partition the weight tensor into contiguous blocks of $B$ elements (QLoRA uses $B = 64$). For each block, compute the absolute maximum $s = \max_j |w_j|$ and store it as the block's scale. Each weight is normalized to $[-1, 1]$ and mapped to the nearest quantile level,

$$\hat{w}_j = s \cdot q_{\,\text{idx}(w_j / s)}, \qquad \text{idx}(u) = \arg\min_i |u - q_i|. \tag{6.3.5}$$

Storage per weight is 4 bits for the index, and the block shares one scale.

**3. Double quantization.** The block scales are themselves numbers, one FP32 per 64 weights, which is $32/64 = 0.5$ bits per weight of overhead, not negligible. Double quantization quantizes the scales too: it takes the FP32 block scales, groups 256 of them, and quantizes that group to 8-bit with its own single FP32 scale. The scale overhead drops from 32 bits per 64 weights to $8/64 + 32/(64 \cdot 256) \approx 0.127$ bits per weight, saving roughly $0.37$ bits per weight, about $0.4$ GiB on a 8B-parameter model, which on a 16GB card is real.

The net cost is $4 + 0.127 \approx 4.13$ bits per weight, and the reconstruction $\hat{w}$ is used *only in the forward and backward matmuls*; the stored master remains the 4-bit code. Gradients flow through the frozen $\hat W_0$ untouched (it is frozen) and into the BF16 LoRA adapters via equation (6.3.3), which is why QLoRA keeps 16-bit-quality adapters on top of a 4-bit base.
```

```admonish vram-budget title="4B QLoRA on the RTX 5080 16GB, line by line"
Take Qwen3-4B: call it $N = 4.0 \times 10^9$ parameters. All figures in GiB ($1\,\text{GiB} = 2^{30}$ bytes). "Measured on the baseline machine, record value, date, driver" applies to every peak-VRAM claim; the numbers below are the *arithmetic floor* you reconcile against.

- **Frozen base in NF4:** $4.13$ bits/param $= 0.516$ bytes/param. $4.0\times10^9 \times 0.516 = 2.06\times10^9$ B $= 1.92$ GiB.
- **LoRA adapters (BF16), $r=16$, all 7 target modules:** roughly $1.5\%$ of $N$ trainable is a loose upper bound; concretely, summing $r(d+k)$ over the targeted matrices lands near $30\text{–}40$M params. Take $35\text{M} \times 2$ B $= 70$ MB $= 0.065$ GiB.
- **Adapter gradients (BF16):** same shape as adapters, $0.065$ GiB.
- **Adam state on adapters (FP32 $m$, $v$, + FP32 master):** $3 \times 4$ B $\times 35\text{M} = 0.42$ GB $= 0.39$ GiB.
- **Activations for the backward pass:** the variable term, set by batch size $\times$ sequence length $\times$ hidden $\times$ layers, and slashed by gradient checkpointing. For batch 2, seq 2048, checkpointed, budget on the order of $2\text{–}4$ GiB.
- **CUDA context + kernels + fragmentation:** $1\text{–}2$ GiB of overhead you do not control.

Floor sum: $1.92 + 0.065 + 0.065 + 0.39 + 3 + 1.5 \approx 6.9$ GiB, comfortably inside 16 GiB, with the activation term as the knob you turn (batch size, sequence length, checkpointing) if you approach the ceiling. Contrast full BF16 fine-tuning: weights $8$ GiB + Adam state $\sim 32$ GiB + activations, which is roughly $40$+ GiB and simply does not fit. That gap, $\sim 7$ GiB versus $\sim 40$ GiB, is why every training lab in this book is QLoRA.
```

### The merge math, and when merging changes behavior

At inference I often want to fold the adapter back into the weights so there is no runtime overhead. From equation (6.3.2), the merge is exact in full precision:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A. \tag{6.3.6}$$

If $W_0$ is stored in BF16, equation (6.3.6) is a lossless (to BF16 rounding) fold and the merged model is behaviorally identical to the adapter-on-base model. The subtlety is QLoRA. There, $W_0$ is *stored* in NF4, and the honest merge is $\text{quant}(W_0) + \frac{\alpha}{r}BA$: you must dequantize the base to BF16, add the BF16 update, and keep the result in BF16, because $\frac{\alpha}{r}BA$ has structure far finer than the NF4 grid and re-quantizing the sum to NF4 would round the delta away. So merging a QLoRA adapter *changes the base dtype* (NF4 to BF16), which quadruples the base footprint and is exactly what you want for a clean deployable checkpoint, but it means "merge then re-quantize to 4-bit" is not the same model as "keep the adapter separate on the 4-bit base," and I have watched people lose their whole fine-tune to that round-trip. Keep the adapter separate for iteration; merge to BF16 only when you freeze a release.

## Tooling

PEFT implements equations (6.3.1)–(6.3.3) as a `LoraConfig` (rank `r`, `lora_alpha`, `target_modules`, dropout) wrapped around a base model; `bitsandbytes` implements the NF4 code of equations (6.3.4)–(6.3.5) with double quantization behind `load_in_4bit=True` and the `bnb_4bit_*` flags; and Unsloth fuses the two so the NF4 dequant and the LoRA matmul happen in one custom kernel rather than two eager ops, which is where its speed and its extra memory headroom come from. `merge_and_unload()` on a PEFT model performs equation (6.3.6). The reused evaluation piece is chapter 3.7's `evalstats`: after each rank's fine-tune I run the frozen thesis task suite and pass paired per-item scores to `evalstats.bootstrap_paired_diff` (and `evalstats.mcnemar` for the paired p-value) to get a delta-versus-base with a 95% bootstrap CI, so the sweep produces intervals, not point estimates.

## Lab

The lab sweeps LoRA rank over $\{4, 8, 16, 32, 64\}$, fine-tunes the 4B QLoRA model at each rank on the same small reasoning-flavored set, evaluates each on the frozen thesis suite, and reports the eval delta versus the untuned base with bootstrap confidence intervals. The artifact is a CSV and a plot showing where accuracy stops improving with rank, which is the empirical version of "find the intrinsic rank of your task's update."

This is a `uv` project in the training environment. From the repo root:

```bash
uv init labs/lora-rank-sweep
cd labs/lora-rank-sweep
uv add "unsloth[cu124]" trl peft datasets matplotlib
# evalstats is the chapter-3.7 module, installed editable from the repo:
uv add --editable ../../packages/evalstats
```

```python title="labs/lora-rank-sweep/sweep.py"
"""LoRA rank sweep with eval deltas via the chapter-3.7 evalstats module.

For each rank in RANKS: QLoRA-SFT a 4B model, score it on the frozen thesis
suite, and compute the paired bootstrap delta vs the untuned base. Writes
artifacts/rank_sweep.csv and artifacts/rank_sweep.png.
"""
from pathlib import Path
import csv

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# chapter-3.7 machinery: paired bootstrap CI + McNemar over per-item scores.
import evalstats as es
# chapter-3.9 frozen suite: returns per-item 0/1 scores for a model.
from thesis_suite import score_model

MODEL = "unsloth/Qwen3-4B-Base"
RANKS = [4, 8, 16, 32, 64]
MAX_SEQ = 2048
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def train_one(rank: int):
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=2 * rank,   # hold alpha = 2r fixed
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    ds = load_dataset("yahma/alpaca-cleaned", split="train[:3000]")
    # Prompt/completion message lists: SFTTrainer renders + masks the prompt.
    ds = ds.map(lambda ex: {
        "prompt": [{"role": "user", "content": ex["instruction"]}],
        "completion": [{"role": "assistant", "content": ex["output"]}]},
        remove_columns=ds.column_names)
    cfg = SFTConfig(
        output_dir=str(OUT / f"run_r{rank}"), max_seq_length=MAX_SEQ,
        completion_only_loss=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=1, learning_rate=2e-4, warmup_ratio=0.03,
        lr_scheduler_type="cosine", bf16=True, logging_steps=25,
        report_to="none", seed=3407,
    )
    SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
    FastLanguageModel.for_inference(model)
    return model, tok


def main() -> None:
    # Baseline scores once, from the untuned base model.
    base, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(base)
    base_scores = score_model(base, tok)          # list[int], per item
    del base
    torch.cuda.empty_cache()

    rows = []
    for rank in RANKS:
        model, tok = train_one(rank)
        tuned_scores = score_model(model, tok)     # paired, same item order
        # Estimate(point, ci_low, ci_high): paired bootstrap of the delta.
        est = es.bootstrap_paired_diff(tuned_scores, base_scores, level=0.95)
        # McNemar for the paired p-value on binary correctness.
        mc = es.mcnemar(tuned_scores, base_scores, exact=True)
        rows.append({
            "rank": rank,
            "base_acc": sum(base_scores) / len(base_scores),
            "tuned_acc": sum(tuned_scores) / len(tuned_scores),
            "delta": est.point, "ci_lo": est.ci_low, "ci_hi": est.ci_high,
            "p_value": mc.pvalue,
        })
        print(f"r={rank:3d}  delta={est.point:+.3f}  "
              f"95% CI [{est.ci_low:+.3f}, {est.ci_high:+.3f}]  p={mc.pvalue:.3f}")
        del model
        torch.cuda.empty_cache()

    csv_path = OUT / "rank_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot delta vs rank with CI whiskers.
    import matplotlib.pyplot as plt
    xs = [r["rank"] for r in rows]
    ys = [r["delta"] for r in rows]
    lo = [r["delta"] - r["ci_lo"] for r in rows]
    hi = [r["ci_hi"] - r["delta"] for r in rows]
    plt.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4)
    plt.axhline(0, color="grey", lw=0.8)
    plt.xscale("log", base=2)
    plt.xlabel("LoRA rank r"); plt.ylabel("accuracy delta vs base")
    plt.title("Rank sweep: eval delta with 95% bootstrap CI")
    plt.tight_layout(); plt.savefig(OUT / "rank_sweep.png", dpi=150)
    print(f"Artifacts: {csv_path.resolve()} and {(OUT/'rank_sweep.png').resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python sweep.py
```

```admonish gotcha
`score_model` must return per-item scores in the *same item order* for the base and every tuned model, because `bootstrap_paired_diff` and `mcnemar` compare matched pairs, that pairing is the entire reason the CI is tight enough to see a small delta (chapter 3.7 derives why paired resampling beats comparing two independent accuracy numbers). If you shuffle the eval set between runs, or the suite's sampling is nondeterministic, the pairing breaks and the CIs blow up. Freeze the suite (chapter 3.9 froze v1.0 for exactly this) and fix the eval seed so the only thing changing between runs is the rank.
```

**What you should see.** The CSV and plot show the eval delta versus the untuned base rising with rank and then flattening, the empirical signature of the low-rank hypothesis: once $r$ exceeds the task's intrinsic update rank, more capacity buys nothing and the confidence intervals of adjacent ranks overlap. Typically the delta from $r=4$ to $r=16$ is real (CI excludes zero) and the delta from $r=16$ to $r=64$ is within noise (CIs overlap), which tells you $r=16$ is the honest choice for this task, more is just memory. All five fine-tunes fit in 16GB by the `vram-budget` above, with peak VRAM creeping up only slightly with rank because the adapter and its optimizer state grow linearly in $r$ but are tiny next to the frozen base. Record peak VRAM per rank and total sweep wall-clock (measured on the baseline machine, record value, date, driver). The headline artifact is a plot that turns "what rank should I use" from a Reddit argument into a measured curve with error bars.

```admonish read-along
This chapter is math-forward and mostly self-contained, but read it against **[RLHF]** ch. 4's treatment of parameter-efficient fine-tuning for the framing of why PEFT methods are the default in the post-training pipeline, and cross-reference the quantization theory in Part II ch. 3 of this book, where the general quantization-error math behind equation (6.3.5) is derived; NF4 is the special case of that theory tuned to a Gaussian weight prior.
```

```admonish substack-seed
"You are not fine-tuning four billion parameters. You are fine-tuning about thirty million, and the other 99 percent are frozen and squeezed into four bits each." A post that derives the two tricks that make single-GPU fine-tuning real: the low-rank hypothesis (equation 6.3.1, fine-tuning is a low-dimensional nudge, so factor it) and NF4 (equation 6.3.4, put your 16 quantization levels at the quantiles of a bell curve because that is where the weights actually are). End on the rank sweep with error bars, the measurement that shows the update really is low-rank, and the punchline that the whole 40-GiB-to-7-GiB gap between what fits and what does not is just these two ideas stacked.
```
