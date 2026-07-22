# Unsloth internals

The loop this whole book has been circling finally closes in Part VII, and it closes on top of one library. Unsloth is the thing that makes GRPO training of a reasoning model fit on the 16GB card at all, and it does that by being far more invasive than a normal dependency. Most libraries you import sit politely in their own namespace and wait to be called. Unsloth reaches into TRL, transformers, and PEFT at import time and rewrites their hot paths out from under them. That is a wonderful trick when it works and a baffling one when it breaks, so before I run a single training step I want the hood open: what exactly gets patched, why the patched version is smaller and faster, how the in-process vLLM engine shares weights with the trainer instead of duplicating them, and what happens when I fight the versions Unsloth has pinned. This chapter is deliberately heavy on the under-the-hood boxes, because the whole rest of Part VII is spent standing on machinery I want to have already looked inside.

## Theory

### What "patching" actually means here

Unsloth is not a training framework in the way TRL is. TRL owns the training loop: it has a `GRPOTrainer` with a `train()` method, a config object, a data collator, the optimizer step. Unsloth owns almost none of that. What Unsloth owns is a set of *replacements* for the most expensive functions inside the stack TRL sits on, and a loader (`FastLanguageModel`) that installs those replacements before the model is built. When you write `from unsloth import FastLanguageModel` at the very top of your script, the import has side effects: it monkey-patches functions in `transformers` (the attention forward, the RMSNorm, the cross-entropy, the rotary embedding), it patches PEFT's LoRA layers to use fused kernels, and it patches TRL's trainers so that generation during GRPO routes through an in-process vLLM engine instead of `model.generate`. None of this changes the API you call. You still call `GRPOTrainer`. You are just calling a `GRPOTrainer` whose insides have been swapped for faster, leaner versions.

That is why the import-order rule exists and is not superstition. `import unsloth` (or `from unsloth import ...`) has to run *before* `transformers`, `trl`, and `peft` are meaningfully used, because the patches are applied at Unsloth import time and they wrap the target functions as they exist then. If you import and instantiate a transformers model first and Unsloth second, some of the objects you already built are holding references to the unpatched functions, and you get the slow, fat path for those while the patched path applies only to things built later. The failure is silent: nothing errors, your run is just mysteriously using more memory and running slower than the tutorials promised.

```admonish under-the-hood title="What a monkey-patch looks like from the inside"
Concretely, Unsloth does the equivalent of reaching into a module and reassigning attributes:

```python
# Illustrative, not the real source: the shape of what happens at import.
import transformers.models.qwen3.modeling_qwen3 as qwen3
qwen3.Qwen3Attention.forward = _unsloth_fast_attention_forward
qwen3.Qwen3RMSNorm.forward   = _unsloth_fast_rmsnorm_forward
# ... and cross-entropy, rotary, MLP, plus PEFT LoRA linear layers.
```

Because Python looks methods up by name on the class at call time, replacing `Qwen3Attention.forward` means *every* attention module of that class, including ones already constructed, now dispatches to Unsloth's kernel. That is the leverage: a handful of attribute assignments re-routes the entire forward and backward pass of an architecture. It is also the fragility. The patch targets `transformers.models.qwen3.modeling_qwen3` by its exact internal structure. When a transformers release renames that module, changes the attention signature, or splits the RMSNorm out, the assignment either fails loudly (`AttributeError`) or, worse, silently stops matching and you fall back to the slow path. This is the mechanism behind every "works on transformers 4.x.y, breaks on 4.x.z" report, and it is why Unsloth pins the versions it patches against.
```

### The kernels: where the speed and the memory savings come from

The patched functions are almost all Triton kernels, and they win in two distinct ways that are worth separating because they trade against different resources. The first win is **fusion**: doing several elementwise or reduction operations in a single kernel launch so intermediate results never get written to and re-read from GPU memory. The second win is **not materializing activations**: computing a quantity, and its gradient, without ever storing the large intermediate tensor that a naive autograd graph would keep alive for the backward pass. On a 16GB card the second win is the one that decides whether a run fits.

Take the four kernels that matter most for a reasoning-model training step.

**Fused RMSNorm.** RMSNorm computes $y = \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}} \cdot g$ for a gain vector $g$. The naive version is several PyTorch ops (square, mean, rsqrt, multiply, scale), each a separate kernel launch reading and writing the full activation tensor. Unsloth's Triton kernel does the whole thing in one pass over the data, and its custom backward recomputes what it needs rather than storing every intermediate. Fewer launches, less memory traffic, and, critically, fewer saved tensors on the autograd tape.

**Fused rotary embeddings (RoPE).** Applying rotary position encoding to Q and K is a structured multiply-add that the naive path expresses as a pile of slices, concatenations, and elementwise multiplies, each allocating. The fused kernel applies the rotation in place with no intermediate allocations and a matching analytic backward.

**Fused SwiGLU MLP.** The gated MLP computes $\text{down}(\,\text{silu}(\text{gate}(x)) \odot \text{up}(x)\,)$. Naively that is two projections, a SiLU, an elementwise product, and a projection, with the two large hidden-width intermediates (`gate(x)` and `up(x)`, each of size batch by sequence by intermediate-dim) both kept alive for backward. Unsloth fuses the activation and gating and arranges the backward to recompute the SiLU rather than store it, which removes one of the fattest activation tensors in the whole block.

**Fused, chunked cross-entropy.** This is the single most important memory trick and it deserves its own box, below. The short version: computing the loss over a vocabulary of 150,000-plus tokens naively materializes a logits tensor of shape (batch $\times$ sequence $\times$ vocab) in fp32, which for a reasoning trace is enormous, and then a second same-shape tensor for its gradient. Unsloth never lets that full tensor exist.

```admonish under-the-hood title="Why the logits tensor is the villain, and how chunking kills it"
Consider a single sequence of $T = 4096$ tokens (a modest reasoning trace) with a vocabulary of $V \approx 151{,}000$ (Qwen3). The output logits are a $T \times V$ tensor. In fp32 that is

$$
4096 \times 151{,}000 \times 4\ \text{bytes} \approx 2.47\times10^{9}\ \text{bytes} \approx 2.3\ \text{GiB}
$$

for one sequence, and the naive cross-entropy backward wants a gradient tensor of the same shape, so roughly **4.6 GiB of transient memory for the loss of a single sequence**. On a 16GB card, with a group of 8 generations, that alone would blow the budget before any weights or KV cache are counted.

The chunked kernel refuses to materialize $T \times V$. It walks the sequence in blocks, computing the softmax-cross-entropy and its gradient with respect to the hidden states one block at a time, accumulating the scalar loss and writing the gradient straight back into the (small) hidden-state gradient rather than through a full logits gradient. Peak transient memory drops from "the whole $T \times V$ tensor" to "one block's worth", turning a multi-GiB spike into a few hundred MiB. This is the arithmetic reason Unsloth can train long-context reasoning traces where vanilla TRL OOMs, and it is entirely a memory win rather than a speed win: you are not doing less math, you are refusing to store the math's fattest intermediate.
```

The through-line across all four kernels is the same idea from the autograd chapter in Part I: reverse-mode differentiation is cheap in compute but expensive in memory because it must keep forward activations alive for the backward pass. Every Unsloth kernel is a deal to *not* keep a specific activation, recomputing it in the backward instead. That is the same trade as gradient checkpointing, pushed down to the kernel level and hand-tuned per operation.

### The relationship to TRL

It helps to be precise about the division of labor, because "Unsloth trains the model" is wrong in a way that will confuse you when something breaks. TRL provides the algorithm. `GRPOTrainer` implements group-relative policy optimization: it samples a group of completions per prompt, calls your reward functions, computes the group-relative advantages, forms the clipped surrogate objective with the KL-to-reference term, and steps the optimizer. That logic is TRL's, derived in the GRPO chapter of Part V, and Unsloth does not reimplement it.

What Unsloth does is make each of the expensive pieces *inside* that loop cheaper. The forward and backward passes TRL triggers run through Unsloth's fused kernels. The generation step TRL would normally do with `model.generate` is redirected to an in-process vLLM engine (the next section). The model TRL optimizes is a 4-bit base with LoRA adapters, loaded by `FastLanguageModel` rather than `AutoModelForCausalLM`. So the mental model is: TRL is the choreographer, Unsloth is the stunt double it does not know it hired. When TRL calls what it thinks is a normal PyTorch forward, a Triton kernel answers.

```admonish under-the-hood title="How the reference policy avoids a second copy of the model"
GRPO's objective includes a KL penalty against a frozen reference policy $\pi_{\text{ref}}$. Naively that means holding two models in memory: the policy being trained and a frozen reference. On a 16GB card, a second copy of even a 4B model is unaffordable. LoRA makes the second copy unnecessary. Because training only updates the low-rank adapter and the base weights are frozen, the reference policy *is* the current policy with its adapters switched off. To get $\log \pi_{\text{ref}}(y \mid x)$, Unsloth (via PEFT's adapter toggling) disables the LoRA adapters, runs the same base forward, and re-enables them, so a single set of base weights serves as both policy and reference. This is not an Unsloth invention, it is a property of LoRA that TRL exploits, but Unsloth's memory model depends on it: the `vram-budget` in the next chapter has exactly one copy of the base weights, and this is why.
```

### fast_inference: an in-process vLLM engine sharing the weights

The costliest single thing in a GRPO step is not the gradient update, it is the generation. Every step, for every prompt, GRPO samples a whole group of completions (say 8), each potentially hundreds or thousands of tokens of chain-of-thought. Doing that with `model.generate` is painfully slow, because that is naive autoregressive decoding with none of the batching and paging tricks from Part II. So Unsloth's `fast_inference=True` stands up a real vLLM engine *inside the same Python process* as the trainer and routes GRPO's sampling through it.

The obvious worry is memory. vLLM normally loads its own copy of the model weights. If the trainer holds the 4-bit base and vLLM holds a second copy for serving, you have paid for the weights twice, and on 16GB that is fatal. The thing that makes `fast_inference` viable on this card is **weight sharing**: Unsloth arranges for the vLLM engine and the training path to reference the *same* underlying weight tensors rather than each holding an independent copy. There is one set of base weights in VRAM. The trainer reads them for the forward and backward pass; the vLLM engine reads them (plus the current LoRA adapter, applied on top) to generate. When the optimizer updates the adapter, the updated adapter is what the next round of generation uses, so the policy vLLM samples from stays in lockstep with the policy being trained, without ever serializing a checkpoint out and reloading it into a separate server.

```admonish under-the-hood title="Two consumers, one set of weights, and the memory split"
Picture VRAM as holding a single 4-bit weight buffer $W$ and a single LoRA adapter $A$. Two subsystems read them:

- The **trainer** reads $W$ and $A$ for the policy forward/backward, holds gradients and optimizer state for $A$ only, and holds training-time activations.
- The **vLLM engine** reads $W$ and $A$ for generation, and holds its own **KV cache** plus a little scheduler bookkeeping.

The two do not share the KV cache or the optimizer state, but they do share $W$ and $A$. So the memory equation is not `2 x weights + everything else`, it is `1 x weights + optimizer/grad on the adapter + training activations + vLLM KV cache`. The single tunable that splits the card between these two consumers is `gpu_memory_utilization`, which you pass to `FastLanguageModel.from_pretrained`. It is the fraction of the card vLLM is allowed to claim for its KV cache and working set. Set it too high and the trainer's activations OOM; set it too low and vLLM can hold too few concurrent sequences and generation crawls. Tuning that one number is most of what the next chapter's budget is about.
```

There is a subtlety worth stating because it bites people. vLLM likes to own a big contiguous slab of memory up front for its paged KV cache, allocated when the engine initializes, based on `gpu_memory_utilization`. The trainer's peak, by contrast, arrives later, during the backward pass of the longest sequence in a group. Because vLLM grabs its slab first and holds it, the memory the trainer sees available is already net of vLLM's claim. That ordering is why an under-provisioned run often survives generation and then OOMs several steps in, on a backward pass, when a long completion makes training activations spike into space vLLM already fenced off. The fix is always the same: lower `gpu_memory_utilization` to give the trainer more headroom, or shorten the max completion length so the activation spike is smaller.

## Tooling

The tool is `unsloth`, and the surface you actually touch is small: one loader, one adapter-attach call, and a set of environment and version constraints you violate at your peril. Everything else is the patched machinery running underneath TRL.

The loader is `FastLanguageModel.from_pretrained`. It does four things at once: downloads or reads the base model, quantizes it to 4-bit (bitsandbytes NF4) if you asked, installs the Triton patches for that architecture, and, when `fast_inference=True`, initializes the in-process vLLM engine against the shared weights. The important arguments for this book are `model_name`, `max_seq_length` (which bounds both training context and vLLM's max model length), `load_in_4bit`, `fast_inference`, `max_lora_rank` (vLLM needs to know the adapter rank up front to size its LoRA machinery), and `gpu_memory_utilization` (the split from the last box). Then `FastLanguageModel.get_peft_model` attaches the LoRA adapters to the target modules, and from that point the returned `model` is what you hand to TRL's `GRPOTrainer`.

The version discipline is not optional and is the thing this chapter most wants you to internalize. Because Unsloth patches transformers, TRL, and PEFT by their internal structure, it ships against a tested matrix of versions and expects you to stay on it. `uv` is exactly the right tool for this, because the whole point of `uv` in this book is a pinned, lockfile-backed environment that resolves identically tomorrow. The rule I follow: let Unsloth pull in the versions of transformers, trl, peft, vllm, and bitsandbytes that it declares compatible, pin the resulting lock, and treat that lock as part of the experiment's identity (it gets logged to MLflow alongside the git SHA, per the Part 0 tracking spine). When I want to upgrade, I upgrade the whole set deliberately and re-run the acceptance lab, rather than letting a stray `uv add` bump transformers underneath a patched function.

```admonish gotcha title="The three ways people break their Unsloth install"
1. **Import order.** `from unsloth import FastLanguageModel` must come first, before `transformers`, `trl`, or `peft` are imported. Put it at the very top of the file. If a linter reorders your imports alphabetically, exclude that line or you will silently lose the patches.
2. **Bumping a patched dependency.** Running `uv add "transformers>=<newer>"` to get some unrelated feature can move transformers off the version Unsloth patched, and now the attention or RMSNorm patch does not match. Symptoms range from an `AttributeError` at load to a quiet fallback that just uses more memory. Upgrade the whole Unsloth-blessed set together, never one member of it.
3. **A vLLM the fast_inference path does not expect.** `fast_inference` is tied to a vLLM version range because it drives vLLM's engine internals for weight sharing. A mismatched vLLM often surfaces as an error during engine init or a crash the first time GRPO tries to generate. Pin vLLM to what Unsloth wants.
```

```admonish under-the-hood title="Blackwell, the driver, and why the CUDA stack matters here"
The baseline machine is a Blackwell RTX 5080 on the NVIDIA 570-open driver (see the hardware baseline). Unsloth's kernels are Triton, which JIT-compiles to PTX for the target architecture, and the in-process vLLM engine ships CUDA kernels that must have Blackwell (`sm_120`) support. This is the one place where "just pip install it" genuinely fails on new hardware: an older Triton or an older vLLM/PyTorch built before Blackwell support will either refuse to compile a kernel or fall back in a way that erases the speedup. So on this card the version matrix has a hard floor set by the *hardware*, not just by Unsloth's patch targets: PyTorch, Triton, and vLLM all have to be recent enough to emit `sm_120` code against the CUDA toolkit the 570 driver supports. When a lab in this Part reports a clean run, the driver and CUDA versions are part of what made it clean, which is exactly why every measured number in the book is stamped with the driver.
```

## Lab: an Unsloth patch manifest

The point of a "hello world" for a library that works by side effects is to *see the side effects*. So the artifact here is not a trained model, it is a **patch manifest**: a JSON file that records what Unsloth changed about the runtime when I imported it, what the loaded model looks like, and whether the in-process vLLM engine actually came up. It is the diagnostic I reach for first when a later training lab misbehaves, because it answers "is my Unsloth install even doing what it should?" before I go hunting in the training loop.

Set up the project with `uv`, letting Unsloth drag in its blessed versions of the rest of the stack.

```bash title="shell"
uv init labs/unsloth-manifest
cd labs/unsloth-manifest
# Unsloth pins compatible transformers/trl/peft/vllm/bitsandbytes itself.
uv add "unsloth" "unsloth_zoo" "vllm" "trl" "peft" "bitsandbytes"
uv add mlflow
uv lock   # freeze the resolved set; this lockfile is part of the experiment
```

The manifest script imports Unsloth first, snapshots the versions of everything in the patched stack, records which target functions now point at Unsloth code, loads a small 4-bit model with `fast_inference=True`, and confirms the vLLM engine initialized. I use a small policy here (Qwen3-1.7B) so the lab runs quickly; the real training uses a larger one in the next chapter.

```python title="labs/unsloth-manifest/manifest.py"
"""Record what Unsloth patched, then load a 4-bit model with in-process vLLM.

Writes artifacts/patch_manifest.json: the version matrix, which hot-path
functions now resolve to unsloth code, and whether fast_inference came up.
This is the first thing to check when a later GRPO lab misbehaves.
"""
# Unsloth MUST be imported before transformers/trl/peft so its patches apply.
from unsloth import FastLanguageModel  # noqa: E402  (import order is load-bearing)

import json
import importlib.metadata as md
from pathlib import Path

import torch

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

MODEL = "unsloth/Qwen3-1.7B"     # small; swap to a 4B policy in ch. 7.2
MAX_SEQ = 2048
MAX_LORA_RANK = 32


def versions() -> dict:
    pkgs = ["unsloth", "unsloth_zoo", "transformers", "trl", "peft",
            "vllm", "bitsandbytes", "torch", "triton"]
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["cuda"] = torch.version.cuda
    out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    out["capability"] = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    return out


def patched_functions() -> dict:
    """Check whether known hot-path functions now live in an unsloth module.

    We do not assert a specific target (that changes across transformers
    versions); we record where each resolves so a bad install is visible.
    """
    import transformers
    report = {}
    # Look at a couple of representative modules if present.
    candidates = [
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3RMSNorm", "forward"),
        ("transformers.models.qwen2.modeling_qwen2", "Qwen2RMSNorm", "forward"),
    ]
    import importlib
    for mod_name, cls_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            fn = getattr(cls, attr)
            where = getattr(fn, "__module__", "?")
            report[f"{cls_name}.{attr}"] = {
                "resolves_to": where,
                "looks_patched": "unsloth" in (where or "").lower(),
            }
        except (ImportError, AttributeError):
            report[f"{cls_name}.{attr}"] = {"resolves_to": None, "looks_patched": False}
    return report


def main() -> None:
    manifest = {"versions": versions()}

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        fast_inference=True,        # stand up the in-process vLLM engine
        max_lora_rank=MAX_LORA_RANK,
        gpu_memory_utilization=0.5, # give vLLM half the card for KV cache
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=MAX_LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=MAX_LORA_RANK,
        use_gradient_checkpointing="unsloth",  # offloaded checkpointing
    )

    manifest["patched_functions"] = patched_functions()
    manifest["fast_inference_up"] = hasattr(model, "vllm_engine") or hasattr(model, "fast_generate")
    manifest["weights_dtype_4bit"] = any(
        "4bit" in str(getattr(p, "quant_state", "")).lower()
        or p.dtype == torch.uint8
        for p in model.parameters()
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    manifest["params"] = {"trainable": trainable, "total_reported": total,
                          "trainable_fraction": trainable / max(total, 1)}
    manifest["vram_after_load_bytes"] = torch.cuda.memory_allocated()
    manifest["vram_peak_bytes"] = torch.cuda.max_memory_allocated()

    path = OUT / "patch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nArtifact: {path.resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash title="shell"
uv run python manifest.py
```

```admonish gotcha title="`trainable_fraction` should be tiny, and `total_reported` will look wrong"
Two things in the output confuse people. First, `trainable` is the LoRA parameter count and should be well under 1% of the model, because only the adapters train. If it is 100%, `get_peft_model` did not attach or you loaded a full-precision model by accident. Second, `total_reported` from `numel()` under-counts 4-bit weights, because bitsandbytes packs two 4-bit values into one `uint8` element, so `numel()` reports the packed count, not the logical parameter count. Do not read that number as the model's parameter count; read it as a sanity check that the trainable slice is small. The real byte budget comes from the arithmetic in the next chapter, not from `numel()`.
```

### What you should see

The script prints and writes `artifacts/patch_manifest.json`. The `versions` block should show a mutually consistent set (the transformers, trl, peft, vllm, and bitsandbytes that Unsloth resolved), a `cuda` string, and a `device` of "NVIDIA GeForce RTX 5080" with capability `[12, 0]` on the baseline machine, confirming Blackwell `sm_120`. The `patched_functions` block should show at least one entry whose `resolves_to` module contains "unsloth" and `looks_patched: true`, which is the direct evidence that the import did rewrite the hot path; if every entry is `looks_patched: false`, your import order is wrong or your transformers version drifted off Unsloth's patch target, and that is the bug to fix before anything else. `fast_inference_up` should be `true`, meaning the in-process vLLM engine initialized against the shared weights. `trainable_fraction` should be a fraction of a percent. And `vram_peak_bytes` records what the load actually cost on the card: on the baseline machine, note this value with the date and driver (measured on the baseline machine, record value, date, driver), because it is the empirical anchor for the byte-level budget the next chapter builds symbolically. When the arithmetic there says "the base weights plus vLLM's claim should be around N GiB", this manifest's peak is the number you check it against.

```admonish read-along
This chapter has no clean textbook analogue, because Unsloth's internals are a moving library rather than a settled result, but read it against the autograd derivation in Part I ([MADL] ch. 3 to 4). Every kernel trick here (fused RMSNorm, recomputed SwiGLU, chunked cross-entropy) is the same memory-for-compute deal that reverse-mode differentiation forces, applied one operation at a time. If you understood why the backward pass must store activations, you already understand why Unsloth's whole design is about refusing to store them.
```

```admonish substack-seed
"The library that rewrites your other libraries at import time." Most dependencies are polite: they wait in their namespace to be called. Unsloth is not. The moment you import it, it reaches into transformers, TRL, and PEFT and hot-swaps their most expensive functions for hand-tuned Triton kernels, which is why it can train a reasoning model on a 16GB gaming card and also why it breaks the instant you bump transformers by a patch version. The post writes itself around one vivid number: the naive cross-entropy over a 150k-token vocabulary on a 4k-token reasoning trace transiently allocates ~4.6 GiB of memory for a single sequence's loss, and Unsloth's chunked kernel never lets that tensor exist. The deeper hook is that this is the same trade reverse-mode autodiff has always forced (cheap gradients paid for in stored activations), pushed down to the kernel level: every speedup is really a refusal to remember something, recomputed on the way back.
```
