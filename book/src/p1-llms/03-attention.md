# Attention from first principles

Attention is the operation that lets a token look at other tokens and decide, per query, which ones matter. It is the reason transformers work, and it is the single most important thing to be able to derive rather than invoke, because every later chapter (the KV cache and its memory, GQA, the roofline of inference) is an argument about the cost of *this specific computation*. So I am going to build scaled dot-product attention from the ground up: what queries, keys, and values are, why the dot product is the right similarity, why we divide by $\sqrt{d}$ and not by $d$ or nothing, how the softmax turns scores into weights (and what its Jacobian is, because that is what backprop needs), how causal masking makes it autoregressive, and how multi-head attention buys representational capacity for almost free. Then I will implement it in raw PyTorch and check it, number for number, against a real Hugging Face layer.

## Theory

### The problem attention solves

A token embedding on its own is context-free: the vector for "bank" is the same whether the sentence is about a river or a loan. To build a contextual representation, each token needs to pull in information from the other tokens that are relevant *to it*, and "relevant" has to be learned and content-dependent, not fixed by position. Attention is a differentiable, content-addressed lookup that does exactly this. Every token emits a **query** (what am I looking for?), every token exposes a **key** (what do I offer as a match?) and a **value** (what information do I carry?), and each token's output is a weighted average of everyone's values, where the weights come from how well its query matches each key. It is a soft, learnable dictionary lookup: instead of retrieving the one value whose key matches exactly, you retrieve a blend, weighted by match quality.

### Queries, keys, values as learned projections

Start from the input to an attention layer: a sequence of $L$ token vectors stacked into a matrix $X \in \mathbb{R}^{L \times d_{\text{model}}}$, one row per position. Attention does not use $X$ directly; it first projects it into three separate spaces with three learned weight matrices:

$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V, \tag{3.1}$$

with $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, giving $Q, K \in \mathbb{R}^{L \times d_k}$ and $V \in \mathbb{R}^{L \times d_v}$. The point of three separate projections is that the *same* token plays three different roles: the query space is tuned for asking, the key space for being found, the value space for what actually gets copied. These are the only learned parameters in attention (plus an output projection I will add shortly); everything else is fixed arithmetic.

### Scaled dot-product attention, derived

The similarity between query $i$ and key $j$ is their dot product $q_i \cdot k_j = \sum_{m=1}^{d_k} q_{im} k_{jm}$, which is large and positive when the two vectors point the same way and near zero when they are orthogonal. Stacking all pairs gives the score matrix $QK^\top \in \mathbb{R}^{L \times L}$, entry $(i, j)$ being how much token $i$'s query matches token $j$'s key. We convert each query's row of scores into a probability distribution over the $L$ keys with a softmax, then use those probabilities to average the values. The whole operation is one line:

$$\mathrm{Attn}(Q, K, V) = \underbrace{\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{A \,\in\, \mathbb{R}^{L \times L}} V. \tag{3.2}$$

Equation (3.2) is the whole mechanism, and the row-stochastic matrix $A$ (each row sums to 1, all entries nonnegative) is the **attention matrix**: $A_{ij}$ is the fraction of its output that token $i$ pulls from token $j$'s value. The output is $A V \in \mathbb{R}^{L \times d_v}$, each row a convex combination of value vectors. The only piece not yet justified is that $\sqrt{d_k}$ in the denominator, and it is not cosmetic.

```admonish derivation title="Why divide by exactly the square root of d_k"
The scaling exists to keep the dot products from growing with dimension and shoving the softmax into a saturated corner. Model the query and key components, at initialization, as independent random variables with mean $0$ and variance $1$ (this is what standard initialization arranges, and it is the regime that matters because a saturated softmax at step 0 kills the gradient before training can start). Consider one score, the dot product of a query row and a key row:

$$s = q \cdot k = \sum_{m=1}^{d_k} q_m k_m. \tag{3.3}$$

Each term $q_m k_m$ is a product of two independent mean-zero, unit-variance variables, so it has mean $\mathbb{E}[q_m k_m] = \mathbb{E}[q_m]\mathbb{E}[k_m] = 0$ and variance $\mathrm{Var}(q_m k_m) = \mathbb{E}[q_m^2]\mathbb{E}[k_m^2] - 0 = 1 \cdot 1 = 1$. Because the $d_k$ terms are independent, their variances add:

$$\mathbb{E}[s] = 0, \qquad \mathrm{Var}(s) = \sum_{m=1}^{d_k} \mathrm{Var}(q_m k_m) = d_k. \tag{3.4}$$

So the raw scores have standard deviation $\sqrt{d_k}$: they grow with the square root of the head dimension. For a typical head dimension of $128$, that is a spread of about $\pm 11$ before any training, and a softmax fed inputs that large is essentially a hard $\arg\max$ — one weight near $1$, the rest near $0$. That saturation is fatal at initialization, because (as the Jacobian below shows) a saturated softmax has vanishing gradient, so the layer cannot learn.

Dividing the scores by $\sqrt{d_k}$ *before* the softmax rescales the variance back to $1$:

$$\mathrm{Var}\!\left(\frac{s}{\sqrt{d_k}}\right) = \frac{\mathrm{Var}(s)}{d_k} = \frac{d_k}{d_k} = 1. \tag{3.5}$$

That is the whole argument, and it pins the constant exactly. Dividing by $d_k$ (not its root) would over-shrink the scores to variance $1/d_k$, flattening the softmax toward uniform and destroying the model's ability to be selective; dividing by nothing leaves the variance at $d_k$ and saturates. Only $\sqrt{d_k}$ makes the score variance dimension-independent, which is precisely the condition under which the softmax starts life in its responsive, high-gradient regime regardless of how wide you make the head. The constant is a variance-normalization, and equation (3.4) is where the "square root" comes from.
```

### The softmax and its Jacobian

The softmax turns a row of scores $z \in \mathbb{R}^{L}$ into a probability vector $a \in \mathbb{R}^{L}$ via

$$a_i = \frac{e^{z_i}}{\sum_{j=1}^{L} e^{z_j}}, \tag{3.6}$$

which is nonnegative and sums to $1$ by construction. To train through attention, backprop needs the derivative of these probabilities with respect to the scores, and the softmax Jacobian is clean and worth deriving once, because it explains both why saturated softmaxes kill gradients and how the attention weights get their gradient.

```admonish derivation title="The softmax Jacobian"
Differentiate $a_i$ in equation (3.6) with respect to an arbitrary input $z_j$. Write $S = \sum_k e^{z_k}$, so $a_i = e^{z_i}/S$. There are two cases.

**Diagonal ($i = j$).** Using the quotient rule with $\partial S / \partial z_i = e^{z_i}$:

$$\frac{\partial a_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = \frac{e^{z_i}}{S}\left(1 - \frac{e^{z_i}}{S}\right) = a_i (1 - a_i). \tag{3.7}$$

**Off-diagonal ($i \neq j$).** The numerator $e^{z_i}$ does not depend on $z_j$, only $S$ does, and $\partial S / \partial z_j = e^{z_j}$:

$$\frac{\partial a_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} e^{z_j}}{S^2} = -\frac{e^{z_i}}{S}\frac{e^{z_j}}{S} = -a_i a_j. \tag{3.8}$$

Combining both cases with the Kronecker delta $\delta_{ij}$ gives the full Jacobian in one expression:

$$\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j), \qquad \text{i.e.} \qquad J = \mathrm{diag}(a) - a a^\top. \tag{3.9}$$

Two things fall out of equation (3.9) immediately. First, the vector-Jacobian product that backprop actually computes has a tidy form: for an upstream gradient $g$, $J^\top g = a \odot (g - (a^\top g)\mathbf{1})$, an elementwise product with a mean-subtracted upstream — no $L \times L$ matrix ever materialized, which is the VJP principle from the tensors chapter in action. Second, and this is the punchline connecting back to the $\sqrt{d_k}$ argument: when the softmax is saturated (some $a_i \approx 1$, the rest $\approx 0$), every entry of $J$ is near zero, because both $a_i(1-a_i)$ and $a_i a_j$ vanish when the $a$'s are near $0$ or $1$. A saturated softmax has no gradient. That is exactly the failure mode equation (3.4) warned about, and exactly why we scale by $\sqrt{d_k}$ to keep the softmax off its saturated shoulders at initialization.
```

### Causal masking makes it autoregressive

For a language model, token $i$ must predict token $i+1$ using only tokens $1$ through $i$; letting it attend to future tokens would leak the answer. Attention enforces this with a **causal mask** applied to the scores before the softmax: set $S_{ij} = -\infty$ for all $j > i$, so that after the softmax $A_{ij} = 0$ there. Since $e^{-\infty} = 0$, the future tokens get exactly zero weight and drop out of every convex combination, while the surviving weights still renormalize to sum to $1$ over the allowed positions:

$$S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + M_{ij}, \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases} \tag{3.10}$$

The mask is a fixed lower-triangular pattern, not learned. In practice frameworks add a large negative number (like the dtype's most-negative finite value) rather than literal $-\infty$ to avoid NaNs, but the effect is the same: future is unreachable. This lower-triangular structure is why the KV cache works at inference (each new token only ever attends backward, so past keys and values can be stored and reused), which is the entire subject of the KV-cache chapter in Part II.

### Multi-head attention

A single attention head produces one set of weights, one "way of looking." But a token often needs to attend to several things at once for different reasons: the subject it agrees with, the clause it modifies, the earlier mention it refers to. **Multi-head attention** runs $h$ independent attention operations in parallel, each with its own small projections, then concatenates and mixes the results. Split the model dimension into $h$ heads of size $d_k = d_{\text{model}} / h$ each, and for head $t$:

$$\mathrm{head}_t = \mathrm{Attn}(X W_Q^{(t)}, X W_K^{(t)}, X W_V^{(t)}), \qquad t = 1, \dots, h, \tag{3.11}$$

then concatenate the $h$ outputs along the feature axis and apply a final output projection $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:

$$\mathrm{MHA}(X) = \big[\,\mathrm{head}_1 \,\|\, \mathrm{head}_2 \,\|\, \cdots \,\|\, \mathrm{head}_h\,\big]\, W_O. \tag{3.12}$$

The elegance is in the bookkeeping: because each head is size $d_{\text{model}}/h$, the total parameter count of the $h$ heads equals that of a single full-width attention, so multi-head attention buys $h$ different views at essentially the *same* parameter and compute cost as one wide head. It is capacity for almost free, which is why every model uses it. In implementation the per-head projections of equation (3.11) are fused into single $d_{\text{model}} \times d_{\text{model}}$ matrices $W_Q, W_K, W_V$ and then reshaped into $h$ heads, so on disk you see one big projection per role, not $h$ small ones. When I get to GQA in the next chapter, the whole trick will be to keep $h$ separate query heads but *share* keys and values across groups of them, purely to shrink the KV cache; everything about why that is even coherent comes from equations (3.11)–(3.12).

## Tooling

The tool that embodies attention is, at the bottom, `torch.nn.functional.scaled_dot_product_attention` (SDPA), the fused primitive that PyTorch exposes and that Hugging Face models call under the hood. It implements exactly equation (3.2) with the causal mask of (3.10), but it does not build the $L \times L$ matrix $A$ in memory when it can avoid it: on supported hardware it dispatches to a **FlashAttention**-style kernel that computes the softmax-weighted sum in tiles, streaming over the sequence and never materializing the full attention matrix. That matters enormously for the 16GB budget, because the naive $A \in \mathbb{R}^{L \times L}$ scales quadratically with sequence length and is the first thing to blow up at long context; the fused kernel keeps attention's *memory* linear in $L$ even though its *compute* stays quadratic. I will unpack that roofline in Part II, but the tooling point for now is that SDPA is the single call that a hand-written attention and a production model both ultimately reduce to, which is why I can verify one against the other.

On the Hugging Face side, a model's attention module (for example `Qwen3Attention`) is a thin wrapper: it holds the fused `q_proj`, `k_proj`, `v_proj`, and `o_proj` linear layers of equations (3.1) and (3.12), reshapes into heads, applies rotary position embeddings (next chapter) to $Q$ and $K$, and then calls SDPA. Because the wrapper is thin, I can pull the exact projection weights out of a real layer, run the arithmetic myself with the same weights, and expect a numerical match to within floating-point tolerance. That is the whole design of the lab.

```admonish under-the-hood
`scaled_dot_product_attention` picks a backend at runtime (a FlashAttention kernel, a memory-efficient kernel, or a plain math fallback) based on dtype, head dimension, mask type, and hardware. The fused backends fold the $1/\sqrt{d_k}$ scaling, the causal mask, and the softmax into one pass, which means intermediate results (the pre-softmax scores) never exist as a tensor you could inspect — a reason that when I verify by hand I use the explicit math path, where every intermediate is visible, and only then trust the fused path to be the same function computed faster.
```

## Lab

The goal is to implement scaled dot-product multi-head attention in raw PyTorch, straight from equations (3.1)–(3.12), and verify it numerically against a real Qwen3 attention layer by feeding both the same input and the same weights and checking the outputs agree to floating-point tolerance. The artifact is a saved report of the max absolute difference, which is the evidence that "attention" is exactly the arithmetic derived above and nothing hidden.

```bash
uv init labs/attention
cd labs/attention
uv add torch transformers
```

```python title="labs/attention/verify_attention.py"
"""Raw-PyTorch scaled dot-product attention, verified against a HF layer.

Artifact: artifacts/attention_check.json with the max abs difference.
"""
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def manual_attention(x, layer, cfg):
    """Reproduce one attention layer's output from raw ops (eqs 3.1-3.12).

    x: (batch, seq, d_model). Uses the layer's own projection weights.
    Note: we skip RoPE here and disable it on the reference too, so this
    isolates the scaled-dot-product-attention math of this chapter.
    """
    b, s, _ = x.shape
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d_head = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // n_q

    q = layer.q_proj(x).view(b, s, n_q, d_head).transpose(1, 2)   # (b, n_q, s, d)
    k = layer.k_proj(x).view(b, s, n_kv, d_head).transpose(1, 2)  # (b, n_kv, s, d)
    v = layer.v_proj(x).view(b, s, n_kv, d_head).transpose(1, 2)

    # Expand KV heads to match query heads (GQA; trivial when n_kv == n_q).
    rep = n_q // n_kv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)         # eq 3.2/3.5
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    scores = scores + causal                                       # eq 3.10
    attn = torch.softmax(scores, dim=-1)                           # eq 3.6
    out = attn @ v                                                 # eq 3.2
    out = out.transpose(1, 2).contiguous().view(b, s, -1)
    return layer.o_proj(out)                                       # eq 3.12

def main() -> None:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    cfg = model.config
    layer = model.model.layers[0].self_attn

    b, s = 1, 12
    x = torch.randn(b, s, cfg.hidden_size)

    with torch.no_grad():
        mine = manual_attention(x, layer, cfg)

        # Reference: call the same layer but neutralize RoPE by passing
        # position_embeddings of (ones cos, zeros sin) so no rotation happens.
        d_head = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // cfg.num_attention_heads
        cos = torch.ones(b, s, d_head)
        sin = torch.zeros(b, s, d_head)
        ref = layer(x, position_embeddings=(cos, sin), attention_mask=None)[0]

    max_abs = (mine - ref).abs().max().item()
    mean_abs = (mine - ref).abs().mean().item()
    report = {
        "model": MODEL,
        "seq_len": s,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "match": max_abs < 1e-4,
    }
    (OUT / "attention_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Artifact: {(OUT / 'attention_check.json').resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python verify_attention.py
```

```admonish gotcha
The fiddly part of verifying against a real model is neutralizing everything the chapter has *not* covered yet so the comparison isolates scaled-dot-product attention. Two traps: rotary position embeddings (RoPE, next chapter) are applied to Q and K inside the reference layer, so I feed a no-op cos/sin (`cos=1, sin=0`) to disable the rotation and match my RoPE-free manual pass; and I force `attn_implementation="eager"` so the reference runs the explicit math path rather than a fused SDPA kernel whose internal ordering can add a few extra ULPs. If the HF layer's forward signature differs by version (position-embedding handling has changed across `transformers` releases), read the layer's `forward` source and adapt how cos/sin get passed — the arithmetic you are checking does not change, only the plumbing.
```

**What you should see.** The script writes `artifacts/attention_check.json` reporting a max absolute difference between my hand-rolled attention and the real Qwen3 layer that is down at the floating-point-noise floor, on the order of $10^{-5}$ to $10^{-6}$ in FP32, with `"match": true`. That tiny residual is not error in the derivation; it is the accumulated rounding of doing the same real-valued arithmetic in a slightly different operation order (equation 1.8 from the tensors chapter, made visible). The takeaway is concrete and load-bearing for the rest of the book: a production attention layer is *exactly* equations (3.1) through (3.12), same weights, same softmax, same $\sqrt{d_k}$, and once you have watched your own code reproduce it to five decimal places, attention stops being a black box and becomes a thing you can reason about the cost of. Record your exact max-diff (measured on the baseline machine — record value, date, driver), since it depends on dtype and the `transformers` version you pinned.

```admonish read-along
Read **[BLLM]** ch. 3 alongside this: it builds causal multi-head attention step by step in PyTorch and its from-scratch code is the long form of this lab's `manual_attention`. For the linear-algebra and probability underneath (the dot product as similarity, the softmax as a Gibbs distribution, the variance argument for $\sqrt{d_k}$), **[MADL]** ch. 7–8 is the backing, especially their treatment of the softmax and its derivative, which is equation (3.9) written out at length.
```

```admonish substack-seed
"Why does attention divide by the square root of the head dimension? Not for vibes — for variance." A tight post that derives it: dot products of unit-variance vectors have variance equal to the dimension (equation 3.4), a softmax fed inputs of standard deviation eleven is a hard argmax, and a hard argmax has zero gradient (the softmax Jacobian, equation 3.9). Dividing by exactly the square root — not the dimension, not nothing — is the unique choice that makes the score variance one regardless of head width, keeping the softmax in the regime where it can still learn. It's a two-line variance calculation that explains a constant everyone copies without justifying, and it doubles as an intuition pump for why saturated softmaxes are where gradients go to die.
```
