# Where memory goes: training vs inference

This is the chapter the whole book's constraint hangs on. Sixteen gigabytes is a hard wall, and the difference between a lab that runs and a lab that dies with `CUDA out of memory` is almost always a memory-accounting mistake, not a compute one. So I am going to account for every byte, in both regimes. Inference memory is short: weights plus the KV cache, and that is essentially it. Training memory is the interesting one, because reverse-mode autograd (from chapter 1) forces you to store activations, and the AdamW optimizer keeps two extra full-size copies of every parameter, so training a model can cost eight to sixteen times what running it costs. I will derive the backprop-through-a-linear-layer rule that makes the activation-storage requirement concrete, write down the memory-accounting formulas, and then do the byte arithmetic for a 4B model in both modes on the RTX 5080, which is exactly the calculation that tells you what you can and cannot fine-tune on this card.

## Theory

### Inference memory: weights plus KV cache

At inference, with `torch.no_grad()` on, autograd records nothing, so there are no stored activations to speak of beyond the single layer being computed. The memory is two things. First, the **weights**, which is just parameter count times bytes-per-element: a model with $P$ parameters in a dtype of $b$ bytes needs $P \cdot b$ bytes resident. Second, the **KV cache**, the stored keys and values for every past token so the model does not recompute them each step, whose size I derived in the transformer-block chapter (equation 4.9):

$$M_{\text{infer}} = \underbrace{P \cdot b}_{\text{weights}} \;+\; \underbrace{2 \cdot B \cdot L \cdot N \cdot h_{kv} \cdot d_{\text{head}} \cdot b}_{\text{KV cache}} \;+\; (\text{small: current activations, CUDA context}). \tag{6.1}$$

The activations at inference are negligible because you only ever hold one layer's worth at a time (the residual stream is a single $B \times L \times d$ tensor passing through), and the CUDA context and kernels are a fixed few hundred megabytes. So inference is dominated by weights, with the KV cache growing linearly in sequence length and becoming the swing factor at long context. This is why the whole of Part II is about serving: on a fixed card, weights are fixed but the KV cache is where you win or lose, and GQA (equation 4.10) was the first lever for it.

### Training memory: why it explodes

Training adds three new consumers, and together they are why training costs multiples of inference. The first is **gradients**: every trainable parameter needs a same-shaped gradient tensor, another $P$ elements. The second is **optimizer state**, and for AdamW this is the killer. The third is **stored activations**, forced by reverse-mode autograd. Take them in order, starting with the one that requires a derivation.

```admonish derivation title="Backprop through a linear layer, and why it stores the input"
Consider the workhorse operation, a linear layer $Y = XW$, with input $X \in \mathbb{R}^{L \times d_{\text{in}}}$, weight $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, output $Y \in \mathbb{R}^{L \times d_{\text{out}}}$. Suppose the backward pass has handed us the adjoint of the output, $\bar{Y} = \partial \mathcal{L} / \partial Y \in \mathbb{R}^{L \times d_{\text{out}}}$ (the VJP flowing in, from chapter 1's equation 1.5). We need the gradients with respect to both $W$ (to update it) and $X$ (to keep propagating backward).

For the weight gradient, differentiate the scalar loss with respect to a single entry $W_{jk}$. Since $Y_{ik} = \sum_j X_{ij} W_{jk}$, the entry $W_{jk}$ influences the loss through every output $Y_{ik}$ across positions $i$:

$$\frac{\partial \mathcal{L}}{\partial W_{jk}} = \sum_{i} \frac{\partial \mathcal{L}}{\partial Y_{ik}} \frac{\partial Y_{ik}}{\partial W_{jk}} = \sum_i \bar{Y}_{ik}\, X_{ij}. \tag{6.2}$$

That sum over $i$ is exactly a matrix product of $X^\top$ and $\bar{Y}$, so in matrix form:

$$\boxed{\;\frac{\partial \mathcal{L}}{\partial W} = X^\top \bar{Y}\;} \qquad \text{and by the same argument} \qquad \boxed{\;\frac{\partial \mathcal{L}}{\partial X} = \bar{Y}\, W^\top\;}. \tag{6.3}$$

Read equation (6.3) carefully, because the whole training-memory story is hiding in it. To compute the weight gradient $X^\top \bar{Y}$, the backward pass **needs the input $X$** that was fed in during the forward pass. That input is a stored activation. Every linear layer in the network must keep its forward input alive from the forward pass until its backward pass, because equation (6.3) cannot be evaluated without it. This is the concrete, per-layer version of chapter 1's abstract claim that reverse mode trades memory for cheap gradients: the "memory" is literally the $X$ in $X^\top \bar{Y}$, one stored activation tensor per layer, and there are dozens of layers. Inference never pays this because `no_grad` means no backward pass will ever ask for $X$, so $X$ is freed the instant the next layer consumes it.
```

Now the accounting. Consider a full fine-tune with mixed-precision AdamW, the standard recipe. Per parameter, the framework holds: an FP32 **master copy** of the weight (4 bytes, kept in full precision so tiny updates are not lost to BF16's coarse mantissa, per chapter 1's equation 1.8), a BF16 **working copy** used in the forward and backward passes (2 bytes), a **gradient** (2 bytes in BF16, sometimes 4), and AdamW's two optimizer moments, the first moment $m$ and the second moment $v$, each an FP32 same-shaped tensor (4 bytes each). That is the famous accounting:

$$\underbrace{4}_{\text{fp32 master}} + \underbrace{2}_{\text{bf16 weight}} + \underbrace{2}_{\text{grad}} + \underbrace{4}_{\text{Adam } m} + \underbrace{4}_{\text{Adam } v} = 16 \text{ bytes per parameter}. \tag{6.4}$$

AdamW is "2 extra copies" in the sense that $m$ and $v$ are each a full parameter-sized FP32 tensor on top of the weights and gradient; those two moments alone are 8 of the 16 bytes. So the fixed, activation-independent cost of full-fine-tuning a $P$-parameter model is $16P$ bytes, versus $2P$ bytes to merely run it in BF16. That factor of eight, before activations, is most of the "4-16x more" in the chapter title. On top of it sit the stored activations from equation (6.3), whose size scales with batch size, sequence length, and depth:

$$M_{\text{train}} = \underbrace{16 P}_{\text{weights, grads, Adam } m,v} + \underbrace{A \cdot B \cdot L \cdot N}_{\text{stored activations}} + (\text{CUDA context}), \tag{6.5}$$

where $A$ is the per-token, per-layer activation footprint (a handful of $d$-sized and $d_{\text{ff}}$-sized tensors that must survive to backward). The activation term is the one you can actually control at run time, which is where the next idea comes in.

### Gradient checkpointing

Equation (6.5)'s activation term grows with depth $N$ and can dominate for long sequences or large batches, and it is the one term you can trade against compute. **Gradient checkpointing** (activation checkpointing) is the deal: during the forward pass, do *not* store the intermediate activations for every layer; store only a sparse set of "checkpoints" (say, one per layer boundary or every $\sqrt{N}$ layers). Then during the backward pass, when equation (6.3) needs the input $X$ to a layer that was not stored, *recompute* it by re-running the forward pass from the nearest checkpoint. You pay one extra forward pass worth of compute (roughly a 30 percent wall-clock hit) in exchange for cutting the stored-activation memory from $O(N)$ down to $O(\sqrt{N})$ or less. It is the purest expression of the memory-versus-compute tradeoff that reverse-mode autograd sets up: autograd wants to store everything (chapter 1), checkpointing declines to, and recomputes on demand. On a 16GB card it is frequently the difference between a batch size of 1 and a usable batch size, and every training lab in this book turns it on.

The bigger structural escape, which I only flag here and derive in the post-training part, is to *not update most of the parameters at all*. LoRA and its quantized cousin QLoRA freeze the base weights (so they need no gradient, no master copy, and no Adam moments, collapsing the $16P$ term to $2P$ for the frozen base plus $16 \times (\text{tiny})$ for the small adapter) and keep the frozen base in 4-bit (halving even that $2P$). That is how a model whose full fine-tune needs many times the card's capacity still trains on it, and equation (6.4) is exactly the cost structure LoRA is dodging.

```admonish vram-budget title="A 4B model on the RTX 5080 16GB, both modes, byte by byte"
Take a concrete 4-billion-parameter model ($P = 4 \times 10^9$) with a Qwen3-4B-like shape: $N = 36$ layers, $h_{kv} = 8$ key/value heads, $d_{\text{head}} = 128$, working dtype BF16 ($b = 2$ bytes). I use decimal GB ($10^9$ bytes) for the arithmetic and note GiB where the 16GB card's actual $16 \times 2^{30} \approx 17.18 \times 10^9$ byte capacity matters.

**Inference (equation 6.1).**
- Weights: $P \cdot b = 4\times 10^9 \times 2 = 8.0$ GB.
- KV cache at batch $B=1$, context $L = 8192$: $2 \cdot 1 \cdot 8192 \cdot 36 \cdot 8 \cdot 128 \cdot 2 = 1.21 \times 10^9 \approx 1.2$ GB.
- CUDA context + kernels: $\approx 0.6$ GB (measured on the baseline machine, record value, date, driver).
- **Total: $8.0 + 1.2 + 0.6 \approx 9.8$ GB.** Fits in 16 GB with headroom; you could push context to $\sim 32$k tokens (KV grows to $\sim 4.8$ GB) before it tightens. This is why 4B-class inference is comfortable on this card.

**Full fine-tune with mixed-precision AdamW (equations 6.4–6.5).**
- Weights + grads + Adam $m,v$: $16 P = 16 \times 4\times 10^9 = 64$ GB.
- **That single line is already 64 GB, roughly four times the card's entire 16 GB capacity**, before a single activation. Full fine-tuning a 4B model on this GPU is simply impossible; the optimizer state alone (Adam $m$ and $v$, $8P = 32$ GB) is twice the card.
- Ratio to inference weights: $16P / 2P = 8\times$ on the parameter side, and once activations enter, real full-training footprints run $8$–$16\times$ inference, the chapter's headline number, now derived.

**LoRA / QLoRA fine-tune (the version that actually fits).**
- Frozen base in 4-bit (NF4): $P \cdot 0.5 = 4\times 10^9 \times 0.5 = 2.0$ GB.
- LoRA adapters (say rank 16 on the attention and MLP projections, well under 1% of params) with their own Adam state at $16$ bytes each: a few hundred MB, call it $\approx 0.4$ GB.
- Activations with gradient checkpointing at $B=1$, $L=2048$: $\approx 1.5$ GB (measured on the baseline machine, record value, date, driver).
- CUDA context: $\approx 0.6$ GB.
- **Total: $2.0 + 0.4 + 1.5 + 0.6 \approx 4.5$ GB.** Fits comfortably, with room for a larger batch or longer sequences.

The three totals, $\approx 9.8$ GB to serve, $64$ GB to full-fine-tune, $\approx 4.5$ GB to QLoRA, are the entire strategic map of this book on one card: you can serve and QLoRA a 4B model here, you cannot full-fine-tune it, and every technique in the training chapters exists to move a workload from the middle column into the third.
```

## Tooling

The tool is PyTorch's CUDA caching allocator and its introspection API, plus the `torch.utils.checkpoint` module for gradient checkpointing. The allocator does not hand each tensor its own `cudaMalloc`; it grabs large blocks from the driver and sub-allocates, which is why freed tensors do not immediately return memory to the OS (they return to PyTorch's pool) and why `nvidia-smi` shows a higher, stickier number than your tensors' live bytes. The functions that tell the truth about your program's own usage are `torch.cuda.memory_allocated()` (bytes currently held by live tensors), `torch.cuda.max_memory_allocated()` (the peak, which is what actually has to fit), and `torch.cuda.memory_summary()` (a full human-readable table broken down by allocation category). The distinction between *allocated* and *reserved* is the one to internalize: reserved is what PyTorch has taken from the driver, allocated is what your tensors are using, and the gap is pool headroom, not a leak. For checkpointing, `torch.utils.checkpoint.checkpoint` wraps a submodule so its activations are dropped and recomputed, and most model libraries expose a `gradient_checkpointing_enable()` one-liner that flips it on for every block. The whole discipline of training on 16GB is: watch `max_memory_allocated`, keep it under the card's capacity, and reach for checkpointing, quantization, and LoRA (equations 6.4–6.5) when it climbs too high.

```admonish under-the-hood
`max_memory_allocated()` is the number that matters, not the instantaneous `memory_allocated()`, because OOM is triggered by the *peak*, and the peak often occurs mid-backward when a layer's stored activation, its freshly computed gradient, and the optimizer moment about to be updated are all live at once. A program can sit at a comfortable steady-state allocation and still OOM at a transient spike you never see in a snapshot. This is why the lab resets and reads the peak counter rather than sampling the current one, and why the vram-budget above sizes to peak, not average.
```

## Lab

The goal is to *measure* the memory accounting of equations (6.1)–(6.5) on a real small model: load it for inference and read the footprint, then run a training step (forward, loss, backward, optimizer step) and watch the peak climb through the predicted multiple as gradients, optimizer state, and stored activations come online. The artifact is a JSON report of measured peaks in each phase, checked against the byte arithmetic.

```bash
uv init labs/memory
cd labs/memory
uv add torch transformers
```

```python title="labs/memory/measure_memory.py"
"""Measure inference vs training memory and reconcile with the formulas.

Artifact: artifacts/memory_report.json with per-phase peak allocations.
Run on a CUDA machine; on CPU it reports parameter math only.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def gb(nbytes: int) -> float:
    return round(nbytes / 1e9, 3)

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    P = None
    report = {"model": MODEL, "device": device}

    # --- Inference footprint (eq 6.1). ---
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    P = sum(p.numel() for p in model.parameters())
    report["params"] = P
    report["predicted_weights_bf16_GB"] = gb(P * 2)
    report["predicted_full_train_16B_GB"] = gb(P * 16)

    if device == "cuda":
        model = model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        ids = tok("Explain entropy in one sentence.", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model(ids)
        report["measured_inference_peak_GB"] = gb(torch.cuda.max_memory_allocated())

        # --- One training step (eq 6.4/6.5). ---
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)  # allocates m, v
        torch.cuda.reset_peak_memory_stats()
        out = model(ids, labels=ids)
        out.loss.backward()          # allocates grads + stored activations (eq 6.3)
        opt.step()                   # materializes Adam m, v (eq 6.4)
        report["measured_train_step_peak_GB"] = gb(torch.cuda.max_memory_allocated())
        report["train_to_infer_ratio"] = round(
            report["measured_train_step_peak_GB"]
            / report["measured_inference_peak_GB"], 2)
        print(torch.cuda.memory_summary(abbreviated=True))
    else:
        report["note"] = "No CUDA; reporting parameter arithmetic only."

    (OUT / "memory_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Artifact: {(OUT / 'memory_report.json').resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python measure_memory.py
```

```admonish gotcha
The measured training-step peak on a 0.6B model will read a bit *below* the naive $16P$ prediction, for two honest reasons: AdamW's $m$ and $v$ are not allocated until the *first* `opt.step()`, so if you read the peak across exactly one step you catch them, but the FP32 master-copy is only maintained when you use an explicit mixed-precision wrapper (plain `AdamW` on a BF16 model here keeps BF16 moments unless you configure otherwise), so this minimal script measures closer to the "grads + Adam" subset than the full 16-byte recipe. That is a feature: the gap between what this script measures and the $16P$ line in the budget is exactly the FP32-master-copy and activation overhead that a real mixed-precision trainer adds, and naming that gap is the point. For the full $16P$ picture, wrap the step in `torch.autocast` and a `GradScaler` and re-measure.
```

**What you should see.** The script writes `artifacts/memory_report.json` with the parameter count, the predicted BF16 weight footprint ($2P$) and full-train footprint ($16P$), and the measured peaks. On the GPU, the inference peak lands near the predicted weight bytes plus a little context and KV, while the training-step peak is a clear multiple of it (the `train_to_infer_ratio` field), because backward brought gradients and stored activations (equation 6.3) online and `opt.step()` materialized Adam's moments (equation 6.4). Watching that ratio jump from roughly 1 to several in a single script is the entire chapter made empirical: nothing about the model changed, only whether you asked autograd to remember the forward pass and whether an optimizer allocated its state. Record the exact peaks with date and driver (measured on the baseline machine, record value, date, driver); the `memory_summary()` table printed alongside shows the allocated-versus-reserved split so you can see the caching allocator's headroom directly.

```admonish read-along
Read **[MADL]** ch. 8–9 for the backpropagation and optimization backing: ch. 8 derives backprop through the layers (equation 6.3's linear-layer rule is their canonical example, extended to whole networks), and ch. 9 covers the optimizers whose state equation (6.4) is accounting for. Their treatment of why the backward pass needs the forward activations is the long-form version of the boxed result here, and it is the cleanest bridge from chapter 1's abstract reverse-mode claim to this chapter's byte-level consequences.
```

```admonish substack-seed
"Why does training an LLM need eight times the memory of running it? Two matrix transposes and a greedy optimizer." A post built on one boxed equation, the weight gradient of a linear layer is X-transpose times the upstream gradient, which means the backward pass literally cannot run without the input X that the forward pass saw, so every layer has to hoard its activations. Add AdamW keeping two full-size extra copies of every weight (its m and v moments), and you've derived the 16-bytes-per-parameter rule that says a 4B model costs 64 GB to fine-tune and 8 GB to serve. The reader walks away able to look at any model size and any GPU and know, in their head, whether it fits, which is the most practically useful party trick in applied ML.
```
