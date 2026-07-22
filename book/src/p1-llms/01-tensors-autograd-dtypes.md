# Tensors, autograd, and number formats

Everything downstream in this book is a program that moves numbers around a GPU and occasionally differentiates them. Before I can talk honestly about attention, memory budgets, or reward gradients, I need the three primitives those programs are built from: the tensor (how numbers are laid out in memory), autograd (how the machine computes derivatives without me deriving them by hand), and the number formats (how a real number gets squeezed into 32, 16, 8, or even 4 bits, and what precision I lose when it does). None of these is a black box once you look at it, and all three come back to bite you at 2am if you treat them as one. So this chapter opens Part I by making them concrete, with the two derivations that the rest of the book leans on: reverse-mode differentiation, and the representation error of floating point.

## Theory

### A tensor is typed memory plus a shape

The word "tensor" scares people off with its physics connotations, but in this stack a tensor is a boringly concrete thing: a flat, contiguous block of bytes in memory, plus a small amount of metadata that tells the library how to read those bytes as a multidimensional array. The metadata is what makes an array a tensor. It is a shape (how many elements along each axis), a stride (how many elements to skip in the flat buffer to move one step along each axis), a dtype (how to interpret each element's bytes as a number), and a device (which piece of hardware's memory the buffer lives in). That is essentially all of it.

Take a $2 \times 3$ matrix of 32-bit floats. The library allocates $2 \times 3 = 6$ elements times 4 bytes each, so 24 contiguous bytes. The shape is $(2, 3)$. The stride, in row-major (C) order, is $(3, 1)$: to move down a row I jump 3 elements in the buffer, to move across a column I jump 1. Indexing element $(i, j)$ is then pure arithmetic on the flat buffer,

$$\text{offset}(i, j) = i \cdot s_0 + j \cdot s_1 = 3i + j \tag{1.1}$$

and reading that element means grabbing 4 bytes at `base + 4 * offset` and interpreting them under the dtype. This is worth internalizing because it explains a pile of behavior that otherwise looks like magic. A transpose does not move any bytes; it just swaps the stride to $(1, 3)$ and the shape to $(3, 2)$, which is why `x.T` is instant and free but leaves you with a *non-contiguous* tensor whose elements are no longer in reading order. A `reshape` that can be expressed as a stride change is free; one that cannot forces a copy. A slice is a new shape, a new stride, and an offset into the *same* buffer, which is why slices alias their parent and writing through one mutates the other. When a kernel later demands a contiguous input and you have handed it a transposed view, the `.contiguous()` call that fixes it is a full memory copy, and on a 16GB card those copies are exactly the kind of thing that quietly doubles your footprint. The stride model is not trivia; it is the difference between a view and a copy, and copies are where VRAM goes to die.

The dtype is the other half of the metadata, and it is the whole back third of this chapter, so hold that thought. The device matters because the RTX 5080's 16GB of VRAM is a separate address space from the 32GB of system DDR5, and moving a tensor across the PCIe bus is neither free nor implicit. `x.to("cuda")` is a real transfer. I will come back to this constantly.

### Autograd is reverse-mode differentiation, and here is why it is cheap

Training is optimization, optimization needs gradients, and the gradients of a modern network with billions of parameters are not something anyone derives by hand. Autograd derives them for you, and the specific algorithm it uses (reverse-mode automatic differentiation, also called backpropagation) is not an approximation and not finite differences. It is the exact chain rule, evaluated in a specific order that makes it astonishingly cheap for the shape of function we care about. That shape is: many inputs (the parameters), one scalar output (the loss). Reverse mode is the right tool for exactly that shape, and it is worth deriving why.

```admonish derivation title="Reverse-mode differentiation from the chain rule"
Write a neural network as a composition of $n$ layers. Each layer is a function, and the whole network maps a parameter/input vector through intermediate activations to a final scalar loss:

$$x_0 \xrightarrow{f_1} x_1 \xrightarrow{f_2} x_2 \;\cdots\; \xrightarrow{f_n} L, \qquad x_k = f_k(x_{k-1}) \in \mathbb{R}^{d_k}, \quad L \in \mathbb{R}. \tag{1.2}$$

The multivariate chain rule says the Jacobian of the composition is the product of the per-layer Jacobians. Let $J_k = \dfrac{\partial x_k}{\partial x_{k-1}} \in \mathbb{R}^{d_k \times d_{k-1}}$ be the Jacobian of layer $k$. Then the gradient of the scalar loss with respect to the input is a chain of matrix products,

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_n} \, J_n \, J_{n-1} \cdots J_1. \tag{1.3}$$

The whole game is the *order* in which you multiply that product. Matrix multiplication is associative, so the answer is the same either way, but the cost is not. Because $L$ is a scalar, $\dfrac{\partial L}{\partial x_n}$ is a row vector (a $1 \times d_n$ object). If I multiply left-to-right,

$$\underbrace{\Big(\big((\tfrac{\partial L}{\partial x_n} J_n) J_{n-1}\big) \cdots\Big)}_{\text{row vector} \times \text{matrix, every step}} \tag{1.4}$$

then at every step I am multiplying a row vector by a matrix, which produces another row vector. I never form a full $d_k \times d_j$ Jacobian; I only ever propagate a vector backward through each layer. Each such step is called a **vector-Jacobian product** (VJP), $v^\top J_k$, and its cost is roughly the same as evaluating the layer forward. So the entire backward pass costs a small constant multiple of one forward pass, independent of how many parameters feed in. That is reverse mode, and equation (1.4) is the reason training a billion-parameter model is even possible.

Contrast forward mode, which multiplies right-to-left and propagates a *column* per input direction: it would cost one forward pass per parameter, which for a billion parameters is a billion forward passes. Reverse mode pays the opposite price: it must *store every intermediate* $x_k$ from the forward pass, because each VJP $v^\top J_k$ needs the activation $x_{k-1}$ to evaluate the local Jacobian at the right point. **That stored-activation requirement is the entire reason training needs far more memory than inference,** and I cash out the byte arithmetic of it in the memory chapter. For now, the one-line summary: reverse mode trades memory for the ability to get all gradients in one backward sweep.

Concretely, define the **adjoint** of each activation as $\bar{x}_k \equiv \dfrac{\partial L}{\partial x_k}$, a vector the same shape as $x_k$. Seed the recursion with $\bar{L} = 1$ (the derivative of the loss with respect to itself), and step backward:

$$\bar{x}_{k-1} = J_k^\top \, \bar{x}_k, \qquad k = n, n-1, \dots, 1. \tag{1.5}$$

Every framework's backward pass is equation (1.5) run once per node, with each layer supplying its own $J_k^\top \bar{x}_k$ rule (its "backward function") rather than a literal Jacobian matrix.
```

The practical upshot for how PyTorch works: as your forward code runs, the library records each operation as a node in a directed acyclic graph, storing the inputs it will need to compute that node's VJP. Calling `.backward()` on the scalar loss walks that graph in reverse topological order, applying equation (1.5) at each node and accumulating adjoints into `.grad` on the leaf tensors (your parameters). The graph is built fresh each forward pass ("define-by-run"), which is why ordinary Python control flow just works. A tensor participates in this only if it has `requires_grad=True`; the activations recorded for the backward pass are held alive until you call `.backward()`, which is why `torch.no_grad()` at inference time is not a nicety but a large memory saving. It tells the engine to skip recording entirely.

### Floating point, bit by bit

Now the dtypes, which is where the 16GB constraint really starts to shape decisions. A real number like $1/3$ has infinitely many digits; a fixed-width binary format has to round it to one of finitely many representable values. Every format in this stack is a specific answer to "how do I spend my bits between range and precision," and the answer determines both how much memory a model eats and how much numerical error it carries. The common scaffold is the IEEE-754-style layout: one sign bit, some exponent bits, and some mantissa (fraction) bits.

```admonish derivation title="Floating-point value and its representation error"
A normalized floating-point number with a $p$-bit mantissa and an exponent bias $B$ encodes the value

$$x = (-1)^{s} \times \Big(1 + \underbrace{\textstyle\sum_{i=1}^{p} b_i 2^{-i}}_{\text{fraction } f \,\in\, [0,\,1)}\Big) \times 2^{\,E - B}, \tag{1.6}$$

where $s$ is the sign bit, the $b_i$ are the mantissa bits, and $E$ is the unsigned integer stored in the exponent field. The leading $1$ is implicit (the "hidden bit"), which buys one extra bit of precision for free. The exponent field selects a **binade**, the interval $[2^{e}, 2^{e+1})$ with $e = E - B$; within a binade the representable numbers are the $2^p$ evenly spaced values you get by walking the mantissa from $0$ to $1 - 2^{-p}$.

The spacing between consecutive representable numbers in the binade $[2^e, 2^{e+1})$ is therefore

$$\Delta = 2^{e} \cdot 2^{-p} = 2^{\,e-p}, \tag{1.7}$$

one unit in the last place (1 ulp). Round-to-nearest maps any real $x$ in that binade to the closest grid point, so the absolute error is at most half a step, $|fl(x) - x| \le \tfrac{1}{2}\,2^{e-p}$. Since $|x| \ge 2^{e}$ everywhere in the binade, the **relative** error is bounded independent of the binade:

$$\frac{|fl(x) - x|}{|x|} \;\le\; \frac{\tfrac{1}{2} 2^{\,e-p}}{2^{\,e}} \;=\; 2^{-(p+1)} \;\equiv\; u. \tag{1.8}$$

The quantity $u = 2^{-(p+1)}$ is the **unit roundoff**, and $\epsilon = 2^{-p}$ (the gap between $1.0$ and the next representable number above it) is **machine epsilon**. Equation (1.8) is the single most useful fact about floating point: relative error is roughly constant across the whole range, set entirely by the mantissa width $p$. More mantissa bits, more precision, full stop. The exponent bits buy you *range* (how large and small $|x|$ can get before you overflow to infinity or underflow toward zero), not precision. This clean split (exponent = range, mantissa = precision) is the lens for reading every format below.

Below the smallest normal exponent, formats fall back to **subnormals**: the implicit leading bit becomes $0$ and the exponent sticks at its minimum, so the numbers fill in the gap between the smallest normal and zero, at the cost of losing relative precision as they shrink toward $0$. Some low-bit formats drop subnormals to save encodings.
```

With equation (1.8) in hand, the formats read off cleanly. I will give each as (sign, exponent, mantissa), its machine epsilon $2^{-p}$, its dynamic range, and where it shows up.

**FP32** (1, 8, 23), bias 127. Machine epsilon $2^{-23} \approx 1.2 \times 10^{-7}$ (about 7 decimal digits), max finite $\approx 3.4 \times 10^{38}$, smallest normal $\approx 1.2 \times 10^{-38}$. This is the "full precision" reference. Master weights, optimizer state, and loss accumulation live here because you want the small numbers to survive being added to big ones. Four bytes per element.

**TF32** (1, 8, 10), NVIDIA's tensor-core format. It keeps FP32's 8 exponent bits (so full FP32 range) but truncates the mantissa to 10 bits, giving $2^{-10} \approx 10^{-3}$ precision. It is stored in a 32-bit slot but multiplied at reduced precision inside the tensor cores; it is a compute mode, not a storage format. Matmuls silently run in TF32 by default on recent hardware unless you ask for true FP32.

**FP16** (1, 5, 10), bias 15. Machine epsilon $2^{-10} \approx 9.8 \times 10^{-4}$, max finite $65504$, smallest normal $\approx 6.1 \times 10^{-5}$. Ten mantissa bits is decent precision, but only five exponent bits means the range is *narrow*, and that narrowness is the classic FP16 training failure: gradients smaller than $\sim 6 \times 10^{-5}$ flush to zero (underflow) and activations above $65504$ blow up to infinity (overflow). This is exactly why FP16 training needs **loss scaling**, multiplying the loss by a big constant to shove the gradients back up into representable range before they underflow, then dividing it back out.

**BF16** (1, 8, 7), bias 127. This is the format that made mixed precision pleasant. It keeps FP32's *full* 8-bit exponent (identical range, $\approx 3.4 \times 10^{38}$, no overflow or underflow drama) and spends only 7 bits on the mantissa, giving coarse precision $2^{-7} \approx 7.8 \times 10^{-3}$. The trade is deliberate: for training, range matters more than precision, because you can tolerate a noisy gradient but you cannot tolerate a gradient that became infinity. BF16 needs no loss scaling because nothing underflows the way it does in FP16. On the RTX 5080 (Blackwell), BF16 is the default working dtype for both my inference and my training labs.

**FP8** comes in two flavors, both 8 bits, differing in how they split the remaining 7 bits after the sign. **E4M3** (1, 4, 3), bias 7, favors precision: max value $448$ (it reclaims the would-be infinity encoding to extend range slightly, so it has no infinities, only NaN). **E5M2** (1, 5, 2), bias 15, favors range: max $57344$, and it keeps IEEE-style inf/NaN. The rule of thumb the hardware vendors use: E4M3 for the forward pass (weights and activations, where you want precision) and E5M2 for gradients (where you want range). FP8 shows up in the fastest inference paths and in FP8 training on Hopper/Blackwell tensor cores; on 16GB it is mostly an inference-time trick.

**INT8** and **INT4** are not floating point at all. They are plain signed integers (range $[-128, 127]$ and $[-8, 7]$) paired with a floating-point **scale** $s$ and sometimes a **zero-point** $z$, so the real value is reconstructed as $x \approx s \cdot (q - z)$ where $q$ is the stored integer. Because the grid is *uniform* (evenly spaced by $s$, unlike floating point's per-binade spacing), integer quantization is a great fit for weight distributions that are roughly bounded and bell-shaped, and it is the workhorse of weight-only quantization. The scale is chosen per-tensor or, better, per-channel/per-group so a few outlier weights do not stretch the grid and wreck everyone else's precision. INT4 weight quantization (with a group size like 128) is how a model that would need many gigabytes in BF16 fits in a couple; I spend a whole chapter on the theory in Part II.

**MXFP4** is the microscaling format that makes 4-bit *floating* point actually usable, and it is worth understanding because gpt-oss ships in it. The element type is E2M1 (1 sign, 2 exponent, 1 mantissa), which by equation (1.6) can represent only the eight magnitudes $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ and their negatives, a laughably coarse grid on its own. The trick is the "MX" part: elements are grouped into blocks of 32, and each block shares one 8-bit power-of-two **scale** (an E8M0 exponent, no mantissa). So the stored cost per element is 4 bits plus $8/32 = 0.25$ bits of shared scale, i.e. 4.25 bits, and the shared scale slides each block's tiny 3-bit grid up or down to wherever that block's numbers actually live. It is INT-style block scaling married to a float element type, and it is the reason a 4-bit model can hold a usable dynamic range. I will read a real MXFP4 checkpoint off disk in the model-zoo chapter.

The through-line: picking a dtype is picking a point on the range-precision-memory triangle that equation (1.8) governs, and on a 16GB card the memory axis is never negotiable, so the craft is in giving up precision (and sometimes range) exactly where the math can absorb it and nowhere else.

## Tooling

The tool that embodies all of this is PyTorch, and specifically two of its subsystems: the tensor/storage model and the `autograd` engine. They are the reason I can write the labs in this book in a few dozen lines instead of a few thousand.

On the tensor side, PyTorch exposes exactly the metadata model from the theory. Every `torch.Tensor` is a view onto a `Storage` (the flat byte buffer) with its own `.shape`, `.stride()`, `.dtype`, `.device`, and a `storage_offset()`. You can see the view-versus-copy distinction directly: `x.t()` and `x[::2]` and `x.view(...)` return tensors sharing `x`'s storage (check with `x.data_ptr()`), while `x.contiguous()` or `x.clone()` or an incompatible `.reshape()` allocate new storage. The dtypes are first-class objects (`torch.float32`, `torch.bfloat16`, `torch.float16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch.int8`, and so on), each carrying an `.itemsize` in bytes, which is all you need to compute a tensor's footprint: `numel() * element_size()`. That formula is the atom of every VRAM budget in this book.

On the autograd side, PyTorch is a define-by-run reverse-mode engine, exactly the algorithm of equations (1.2)–(1.5). Operations on tensors with `requires_grad=True` are recorded into a graph of `Function` nodes; each node stashes whatever forward values its VJP needs (visible as `.saved_tensors`) and exposes a `.grad_fn`. Calling `.backward()` on a scalar seeds $\bar{L} = 1$ and runs the reverse sweep, accumulating into leaf `.grad`. `torch.no_grad()` disables recording (inference, and any bookkeeping you do not want differentiated); `tensor.detach()` cuts a subgraph loose; and mixed precision is handled by `torch.autocast`, which runs matmul-heavy ops in BF16/FP16 while keeping a master copy and the loss accumulation in FP32, i.e. it automates precisely the range-precision split I described. `torch.cuda.amp.GradScaler` is the loss-scaling machinery for the FP16 path.

```admonish under-the-hood
The autograd graph holds your forward activations alive from the moment you run the forward pass until `.backward()` frees them. That is not incidental; it is the memory cost of reverse mode made physical (equation 1.5 needs those activations). Two consequences I rely on all the time: first, forgetting to wrap eval in `torch.no_grad()` can silently inflate inference memory by the entire activation graph; second, gradient checkpointing (Part I, memory chapter) is nothing more than a deal to *not* store some activations and recompute them during backward instead, trading compute for the exact bytes autograd would otherwise pin.
```

## Lab

The goal of this lab is to make the byte-level footprint of a real model layer visible, so that when later chapters claim "the weights are $N$ bytes" you have already watched the arithmetic hold on your own machine. I will load a small open model, walk it parameter by parameter, and print each tensor's shape, dtype, element size, and total bytes, then reconcile the sum against what CUDA reports as allocated. I will also demonstrate the same weights re-cast to BF16 and FP16 so the memory halving is not a claim but a measurement.

This is a `uv` project. From the repo root:

```bash
uv init labs/tensors-dtypes
cd labs/tensors-dtypes
uv add torch transformers safetensors
```

```python title="labs/tensors-dtypes/inspect_dtypes.py"
"""Layer-by-layer dtype and memory inspector for a small model.

Writes a CSV artifact of every parameter's footprint and prints a summary
reconciled against torch.cuda.memory_allocated().
"""
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B"  # small enough to load anywhere; swap freely
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def human(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:8.2f} {unit}"
        nbytes /= 1024

def inspect(model, dtype_label: str, writer) -> int:
    total = 0
    for name, p in model.named_parameters():
        nbytes = p.numel() * p.element_size()
        total += nbytes
        writer.writerow({
            "dtype_label": dtype_label,
            "param": name,
            "shape": "x".join(map(str, p.shape)),
            "dtype": str(p.dtype),
            "elem_bytes": p.element_size(),
            "numel": p.numel(),
            "bytes": nbytes,
        })
    return total

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = OUT / "param_footprint.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dtype_label", "param", "shape", "dtype",
            "elem_bytes", "numel", "bytes",
        ])
        writer.writeheader()

        # Load once in fp32 on CPU so element_size() is unambiguous.
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        n_params = sum(p.numel() for p in model.parameters())

        results = {}
        for label, dtype in [("fp32", torch.float32),
                             ("bf16", torch.bfloat16),
                             ("fp16", torch.float16)]:
            model_cast = model.to(dtype)
            results[label] = inspect(model_cast, label, writer)

        # Move the bf16 copy to the GPU and reconcile with CUDA's accounting.
        allocated = None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            model_gpu = model.to(torch.bfloat16).to("cuda")
            allocated = torch.cuda.memory_allocated()
            print(torch.cuda.memory_summary(abbreviated=True))

    print(f"\nModel: {MODEL}")
    print(f"Total parameters: {n_params:,}")
    for label, total in results.items():
        implied = n_params * {"fp32": 4, "bf16": 2, "fp16": 2}[label]
        print(f"  {label}: summed {human(total)}   "
              f"(n_params x itemsize = {human(implied)})")
    if allocated is not None:
        print(f"CUDA allocated for bf16 weights on GPU: {human(allocated)}")
    print(f"\nArtifact: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python inspect_dtypes.py
```

```admonish gotcha
`model.to(dtype)` casts in place and returns the same object, so the three passes above share one model that I keep re-casting; that is fine for measuring footprint but means you should not keep all three copies live at once expecting three independent models. Also, `torch.cuda.memory_allocated()` will read slightly *higher* than the summed parameter bytes because CUDA's caching allocator rounds each allocation up to a block boundary and there is a fixed context overhead; the point of the reconciliation is that the numbers agree to within that allocator slack, not to the byte.
```

**What you should see.** The script writes `artifacts/param_footprint.csv`, one row per parameter per dtype, which you can open and sort to see exactly which tensors dominate (the embedding/`lm_head` and the MLP `up`/`down`/`gate` projections will be the fattest rows, foreshadowing the memory chapter). In the terminal you get a per-dtype total that matches `n_params × itemsize` to the byte: for a roughly 0.6B-parameter model the FP32 total lands near 2.4 GiB and the BF16 and FP16 totals near 1.2 GiB each, cleanly halved, because equation (1.8)'s bookkeeping is just `numel × element_size`. The CUDA-allocated figure for the BF16 copy on the GPU reads a little above the 1.2 GiB summed figure (allocator rounding plus context), and `memory_summary()` shows that headroom. Record the exact allocated value with date and driver on your run (measured on the baseline machine — record value, date, driver), because that gap between "the arithmetic says 1.2 GiB" and "CUDA says a bit more" is the first hint of the accounting the whole memory chapter is about.

```admonish read-along
This chapter's math backbone is **[MADL]** ch. 2–4: ch. 2 for tensors and linear-algebra-as-memory, ch. 3–4 for the calculus of differentiation that autograd mechanizes. Read equation (1.5) here alongside their treatment of the chain rule, and treat their gradient derivations as the by-hand version of what PyTorch's engine does for you.
```

```admonish substack-seed
"Your GPU doesn't run out of memory because the model is big. It runs out because reverse-mode autodiff has to remember everything it saw on the way forward." A post that starts from the associativity of matrix multiplication (equation 1.4), shows why the scalar-loss shape makes right-to-left the wrong order and left-to-right the cheap one, and lands on the punchline that the price of cheap gradients is stored activations — which is the same sentence, told backward, as "why training needs 4-16x the memory of inference." One derivation, two chapters' worth of intuition, and a title that makes an ML engineer feel seen.
```
