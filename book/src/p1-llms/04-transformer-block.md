# The transformer block

Attention lets a token gather information; it does not, on its own, make a network. The transformer block is the repeating unit that stacks around attention to make a working model: a normalization, the attention sublayer, a residual connection, another normalization, a feed-forward sublayer, another residual. A modern block (Qwen3, Llama, gpt-oss) differs from the 2017 original in four specific, deliberate ways, and this chapter derives all four because each is load-bearing for the memory and speed arguments later. The feed-forward became a *gated* MLP (SwiGLU). LayerNorm became RMSNorm. Absolute position embeddings became rotary (RoPE), which I derive as an actual rotation. And multi-head attention became grouped-query attention (GQA), whose entire justification is KV-cache pressure on cards exactly like the 16GB one. By the end I will assemble a block from these parts and match it, dimension for dimension, against a real Qwen3 config.

## Theory

### The residual stream is the backbone

Before the parts, the skeleton. A transformer block does not *replace* its input with the sublayer output; it *adds* the sublayer output to its input:

$$x' = x + \mathrm{Attn}(\mathrm{Norm}(x)), \qquad x'' = x' + \mathrm{MLP}(\mathrm{Norm}(x')). \tag{4.1}$$

That "$x +$" is the residual connection, and reading equation (4.1) as an addition rather than a transformation is the single most useful frame for the whole architecture. The input $x$ flows down the stack essentially untouched (the "residual stream"), and each sublayer *reads* from it, computes something, and *writes an increment back* by addition. The stream is a running sum; a 36-layer model is 72 sublayers each adding their two cents. This matters for two reasons that recur all book. First, gradients: because the forward pass adds, the backward pass through $x + f(x)$ has Jacobian $I + f'$, so the identity term $I$ guarantees a clean gradient path straight down the stream even when $f'$ is small, this is why deep transformers train at all, and it is the residual restatement of the "identity term keeps gradients alive" idea from the autograd chapter. Second, the norm placement in equation (4.1) puts `Norm` *inside* the sublayer, before attention and before the MLP, never on the residual stream itself. This is **pre-normalization**, and it is what modern models use because it keeps the residual stream un-normalized and the training stable; the original transformer put the norm *after* the addition (post-norm) and was famously twitchy to train without warmup.

### RMSNorm

Every sublayer's input gets normalized first, to keep activations at a controlled scale so the next matmul does not see wildly varying magnitudes. LayerNorm did this by subtracting the mean and dividing by the standard deviation across the feature dimension, then applying a learned scale and shift. RMSNorm drops the mean-centering and the shift entirely, keeping only the scaling by root-mean-square:

$$\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot g_i, \tag{4.2}$$

where $g \in \mathbb{R}^d$ is a learned per-feature gain and $\epsilon$ (around $10^{-6}$) guards the square root. The denominator is the root-mean-square of the feature vector, hence the name. The justification for dropping the mean subtraction is partly empirical (it works as well) and partly efficiency: RMSNorm computes one reduction (a sum of squares) instead of two (mean, then variance), and skips the subtraction, which at the scale of every-token-every-layer is a real saving. There are no bias terms and no re-centering, so on disk an RMSNorm layer is a single $d$-vector of gains, which you will see when I read a config. The important property for later: like all normalizations it is applied per-token across features, so it does not mix information between positions (that is attention's job) and it does not change the sequence length or the memory shape.

### The gated MLP: SwiGLU

The feed-forward sublayer is where most of a transformer's parameters live, and its job is a per-token nonlinear transformation: take each token's $d_{\text{model}}$-vector, blow it up to a wide intermediate dimension $d_{\text{ff}}$, apply a nonlinearity, and project back down. The classic version was two matrices with a ReLU or GELU between them, $\mathrm{MLP}(x) = W_{\text{down}}\, \sigma(W_{\text{up}} x)$. Modern models use a **gated** variant, SwiGLU, which splits the up-projection into two parallel paths and uses one to *gate* the other.

```admonish derivation title="SwiGLU and its parameter-matched width"
SwiGLU uses three weight matrices instead of two. Two of them project up in parallel, a "gate" path and an "up" path, and their outputs are combined by an elementwise product, with the SiLU (a.k.a. swish) nonlinearity $\mathrm{SiLU}(z) = z \cdot \sigma(z)$ on the gate path only:

$$\mathrm{SwiGLU}(x) = W_{\text{down}}\Big(\underbrace{\mathrm{SiLU}(W_{\text{gate}}\, x)}_{\text{gate}} \;\odot\; \underbrace{W_{\text{up}}\, x}_{\text{value}}\Big). \tag{4.3}$$

The intuition is the gate: $\mathrm{SiLU}(W_{\text{gate}} x)$ produces, per intermediate unit, a soft multiplicative mask in roughly $[0, \infty)$ that scales the corresponding unit of $W_{\text{up}} x$. The network learns, per token, which intermediate features to let through and which to suppress, a data-dependent gating that a single static nonlinearity cannot express. This multiplicative interaction is why gated MLPs consistently beat plain ones at equal parameter count.

The parameter accounting has a wrinkle worth deriving, because it is the reason Qwen3's intermediate size looks like an odd number. A plain MLP has two matrices of size $d_{\text{model}} \times d_{\text{ff}}$, so $2\, d_{\text{model}}\, d_{\text{ff}}$ parameters. SwiGLU has *three* matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), so $3\, d_{\text{model}}\, d_{\text{ff}}$ parameters at the same $d_{\text{ff}}$. To keep the parameter budget of the block roughly matched to a plain MLP, implementations shrink the intermediate width by a factor of $2/3$:

$$d_{\text{ff}}^{\text{SwiGLU}} \approx \frac{2}{3}\, d_{\text{ff}}^{\text{plain}}, \qquad \text{so that} \quad 3\, d_{\text{model}} \cdot \tfrac{2}{3} d_{\text{ff}}^{\text{plain}} = 2\, d_{\text{model}}\, d_{\text{ff}}^{\text{plain}}. \tag{4.4}$$

That is why you see intermediate sizes like $3072$ paired with a hidden size of $1024$ (a ratio near $3$, i.e. the $2/3$-adjusted version of a naive $\times 4$ = $4096$) rather than a clean power of two: it is equation (4.4) balancing the three-matrix cost back down to two-matrix territory. When I match the Qwen3 config in the lab, this ratio is one of the numbers I check.
```

### RoPE: position as rotation

Attention as I derived it in the last chapter is *permutation-equivariant*: shuffle the input tokens and the outputs shuffle the same way, because nothing in $QK^\top$ knows position. A language model obviously needs to know order, so position information must be injected. Rotary Position Embedding (RoPE) does this in the most elegant way anyone has found: instead of *adding* a position vector to the embedding, it *rotates* the query and key vectors by an angle proportional to their position, so that the dot product between a query at position $m$ and a key at position $n$ depends only on their *relative* offset $m - n$.

```admonish derivation title="RoPE, from 2D rotation to relative position"
Work in a single 2D subspace first (RoPE splits the head dimension into $d_k/2$ such planes). Take a query 2-vector $q = (q_1, q_2)$ at position $m$ and rotate it by angle $m\theta$ using the standard 2D rotation matrix:

$$R(m\theta) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}, \qquad \tilde{q}_m = R(m\theta)\, q. \tag{4.5}$$

Do the same to the key at position $n$: $\tilde{k}_n = R(n\theta)\, k$. Now compute the attention score, which is the dot product of the rotated query and rotated key:

$$\tilde{q}_m \cdot \tilde{k}_n = (R(m\theta) q)^\top (R(n\theta) k) = q^\top R(m\theta)^\top R(n\theta)\, k. \tag{4.6}$$

Rotation matrices satisfy $R(\alpha)^\top = R(-\alpha)$ and $R(\alpha)R(\beta) = R(\alpha + \beta)$, so $R(m\theta)^\top R(n\theta) = R(-m\theta)R(n\theta) = R\big((n - m)\theta\big)$. Substituting:

$$\tilde{q}_m \cdot \tilde{k}_n = q^\top R\big((n - m)\theta\big)\, k. \tag{4.7}$$

Equation (4.7) is the whole magic: the score depends on position **only through the difference $n - m$**, never through the absolute positions $m$ and $n$ separately. RoPE gives relative-position awareness while implementing it as a per-position rotation you can apply independently to each query and key, with no learned parameters and no extra terms added to the residual stream.

To cover the full head dimension $d_k$, RoPE uses $d_k / 2$ planes, each with its own frequency $\theta_i$ chosen as a geometric progression (the same base-$10000$ schedule as sinusoidal embeddings):

$$\theta_i = \text{base}^{-2(i-1)/d_k}, \qquad i = 1, \dots, d_k/2, \qquad \text{base} \approx 10^4. \tag{4.8}$$

Low-index planes rotate fast (short wavelength, sensitive to nearby tokens), high-index planes rotate slowly (long wavelength, sensitive to far-apart tokens), so the model gets a spectrum of positional resolutions at once. In implementation you never build the $2\times 2$ blocks; you precompute $\cos(m\theta_i)$ and $\sin(m\theta_i)$ for every position $m$ and frequency $i$, and apply equation (4.5) as an elementwise combination of $q$ with a rotated-by-90-degrees copy of itself: $\tilde{q} = q \odot \cos + \mathrm{rotate\_half}(q) \odot \sin$. Those `cos` and `sin` tables are exactly the `position_embeddings` I passed into the reference layer in the attention lab, feeding `cos=1, sin=0` there was setting $m\theta = 0$, i.e. no rotation, which is how I isolated the un-RoPE'd attention math. The **base** in equation (4.8) is the knob behind long-context extension: raising it stretches the wavelengths so the same rotations cover more positions before repeating, which is how models extend context length after pretraining.
```

### GQA: sharing keys and values to shrink the KV cache

The last modern change is the one most tied to my 16GB constraint. At inference, an autoregressive model caches the keys and values of all past tokens so it does not recompute them for every new token (the KV cache, derived fully in Part II). The size of that cache is proportional to the *number of key/value heads*, and for long contexts it becomes the dominant memory cost, often larger than the weights. Grouped-query attention attacks this directly: keep all $h$ query heads (you want the representational capacity of many query views, from the multi-head argument in the last chapter), but let several query heads *share* one key head and one value head.

```admonish derivation title="GQA head-sharing and the KV-cache reduction factor"
Let there be $h$ query heads and $h_{kv}$ key/value heads, with $h_{kv}$ dividing $h$, so each key/value head is shared by a group of $g = h / h_{kv}$ query heads. The three regimes are:

- **Multi-head attention (MHA):** $h_{kv} = h$, every query head has its own K and V ($g = 1$).
- **Grouped-query attention (GQA):** $1 < h_{kv} < h$, groups of $g$ query heads share a K/V.
- **Multi-query attention (MQA):** $h_{kv} = 1$, all query heads share a single K/V ($g = h$).

The KV cache stores, per layer and per token, the keys and values for all $h_{kv}$ key/value heads. Its size in bytes for a sequence of length $L$, batch $B$, head dimension $d_{\text{head}}$, $N$ layers, and $b$ bytes per element is

$$\text{KV bytes} = 2 \cdot B \cdot L \cdot N \cdot h_{kv} \cdot d_{\text{head}} \cdot b, \tag{4.9}$$

where the leading $2$ is for storing both K and V. The only lever that involves the attention head structure is $h_{kv}$, so switching from MHA to GQA cuts the KV cache by exactly the factor

$$\frac{\text{KV}_{\text{MHA}}}{\text{KV}_{\text{GQA}}} = \frac{h}{h_{kv}} = g. \tag{4.10}$$

Qwen3-0.6B, for instance, uses $16$ query heads and $8$ key/value heads, so $g = 2$ and its KV cache is exactly half what full multi-head attention would need, and larger Qwen3 variants push $g$ to $4$ or higher, quartering the cache. Equation (4.10) is why GQA exists: query capacity is cheap (queries are not cached), but key/value capacity is expensive (they *are* cached, per equation 4.9), so GQA spends where it is cheap and saves where it is dear. In the implementation, the shared K/V heads are simply repeated $g$ times to line up with the query heads before the attention dot product, which is exactly the `repeat_interleave(rep, dim=1)` I already wrote in the attention lab, where `rep` is $g$. That line was GQA the whole time; this derivation is why it was there.
```

The through-line for these four changes: RMSNorm and SwiGLU are efficiency and capacity refinements to the sublayers, RoPE is a cleaner way to inject position that gives relative-position awareness for free, and GQA is a direct memory optimization aimed squarely at the KV cache. Three of the four are about doing more with less memory, which is the whole ethos of running this on one card.

## Tooling

The tool is again Hugging Face `transformers`, this time the model *config* and the block module. A model's `config.json` (loaded as a `PretrainedConfig`) is the exact parameterization of everything above: `hidden_size` is $d_{\text{model}}$, `intermediate_size` is $d_{\text{ff}}$ (the $2/3$-adjusted SwiGLU width from equation 4.4), `num_attention_heads` is $h$, `num_key_value_heads` is $h_{kv}$ (equation 4.10's GQA lever), `head_dim` is $d_{\text{head}}$, `num_hidden_layers` is $N$, `rms_norm_eps` is the $\epsilon$ of equation (4.2), `rope_theta` is the base of equation (4.8), and `hidden_act` names the SwiGLU nonlinearity (`silu`). The block module (`Qwen3DecoderLayer`) is equation (4.1) in code: an input RMSNorm, the self-attention with RoPE and GQA, a residual add, a post-attention RMSNorm, the SwiGLU MLP, and another residual add. Because the config *is* the parameterization, I can instantiate my own block from the same config and check that its parameter shapes and count match the real layer exactly, if my SwiGLU width, my head split, and my norm shapes all agree, I have almost certainly assembled the same block.

```admonish under-the-hood
The config carries a subtlety that trips people up: `head_dim` is not always `hidden_size / num_attention_heads`. Qwen3 sets `head_dim` explicitly (for example $128$) even when $h \times 128 \neq$ `hidden_size`, decoupling the attention head geometry from the model width. That is why the q/k/v projections can be non-square (mapping `hidden_size` to `num_heads * head_dim`, a different number) and why my manual attention in the last chapter read `head_dim` from the config rather than dividing. Always trust the explicit `head_dim` field over the division; the division is only a default.
```

## Lab

The goal is to assemble a transformer block from the four derived parts (RMSNorm, SwiGLU, RoPE, GQA attention) in raw PyTorch, parameterized entirely by a real Qwen3 config, and verify that its parameter shapes and total count match the reference `Qwen3DecoderLayer` dimension for dimension. The artifact is a JSON report of every submodule's shape alongside the reference, which proves the block I built is the block Qwen3 ships.

```bash
uv init labs/transformer-block
cd labs/transformer-block
uv add torch transformers
```

```python title="labs/transformer-block/assemble_block.py"
"""Assemble a transformer block from parts and match a Qwen3 config.

Artifact: artifacts/block_shapes.json comparing my block to the real layer.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

class RMSNorm(nn.Module):                                    # eq 4.2
    def __init__(self, d, eps):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.g

class SwiGLU(nn.Module):                                     # eq 4.3
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class GQAAttention(nn.Module):                               # eqs 3.x + 4.9/4.10
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.num_attention_heads
        self.h_kv = cfg.num_key_value_heads
        self.d_head = cfg.head_dim
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.h * self.d_head, bias=False)
        self.k_proj = nn.Linear(d, self.h_kv * self.d_head, bias=False)
        self.v_proj = nn.Linear(d, self.h_kv * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.h * self.d_head, d, bias=False)
        # (RoPE tables omitted for the shape-matching lab; see eq 4.5-4.8)

class Block(nn.Module):                                      # eq 4.1
    def __init__(self, cfg):
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.post_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

def shape_map(module) -> dict:
    return {n: list(p.shape) for n, p in module.named_parameters()}

def main() -> None:
    cfg = AutoConfig.from_pretrained(MODEL)
    mine = Block(cfg)

    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    ref_layer = ref_model.model.layers[0]

    mine_params = sum(p.numel() for p in mine.parameters())
    ref_params = sum(p.numel() for p in ref_layer.parameters())

    report = {
        "model": MODEL,
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "gqa_group_size": cfg.num_attention_heads // cfg.num_key_value_heads,
            "ffn_ratio": round(cfg.intermediate_size / cfg.hidden_size, 3),
            "rope_theta": cfg.rope_theta,
        },
        "my_block_params": mine_params,
        "ref_layer_params": ref_params,
        "my_shapes": shape_map(mine),
        "ref_shapes": shape_map(ref_layer),
    }
    (OUT / "block_shapes.json").write_text(json.dumps(report, indent=2))

    print(f"GQA group size g = {report['config']['gqa_group_size']} "
          f"(KV cache is 1/{report['config']['gqa_group_size']} of MHA)")
    print(f"FFN ratio d_ff/d_model = {report['config']['ffn_ratio']} "
          f"(SwiGLU 2/3-adjusted, eq 4.4)")
    print(f"My block params:  {mine_params:,}")
    print(f"Ref layer params: {ref_params:,}")
    print(f"Artifact: {(OUT / 'block_shapes.json').resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python assemble_block.py
```

```admonish gotcha
The parameter *count* of my block will land very close to the reference but need not match to the exact integer, because I omitted the RoPE inverse-frequency buffer and any tiny extras (some blocks carry per-head q/k norms, Qwen3 in fact adds an RMSNorm on the query and key, `q_norm`/`k_norm`, which the reference has and my minimal block does not). Those are small, and the report prints both counts so you can see the gap and attribute it. The lesson is in the *shapes* matching (the q/k/v/o projections, the three SwiGLU matrices, the two RMSNorm gains), not in a to-the-parameter total; if a shape is wrong, the block is wrong, and that is what the JSON is for.
```

**What you should see.** The script writes `artifacts/block_shapes.json` listing every parameter tensor of my hand-built block next to the real Qwen3 layer's. The projection shapes line up: `q_proj` maps `hidden_size` to `num_attention_heads * head_dim` while `k_proj` and `v_proj` map to the smaller `num_key_value_heads * head_dim`, and you can read the GQA asymmetry straight off those two shapes, the K/V projections are literally $g$ times narrower than Q, which is equation (4.10) made visible in the weights. The printed summary reports the GQA group size (2 for the 0.6B, meaning half-size KV cache), the FFN ratio near 3 (the $2/3$-adjusted SwiGLU width of equation 4.4, not a clean power of two), and the two parameter totals within a fraction of a percent of each other, the small gap explained by the q/k norms and RoPE buffers I left out. Keep the JSON: it is the receipt that "a Qwen3 layer" is exactly RMSNorm + GQA-attention-with-RoPE + RMSNorm + SwiGLU wired residually, four derivations you can now point at.

```admonish read-along
Read **[BLLM]** ch. 4 for the from-scratch transformer block: Raschka builds the residual-plus-norm structure of equation (4.1) and the feed-forward in PyTorch, which is the long form of this lab. For the modern-model specifics, RoPE, RMSNorm, SwiGLU, GQA as Qwen3 actually implements them, **[BRM]** App. C walks the Qwen3 source line by line, and reading equations (4.5)–(4.10) here next to that appendix is the fastest way to connect the rotation math and the head-sharing arithmetic to the code that ships.
```

```admonish substack-seed
"Rotary position embeddings look like a hack until you write down the dot product." A post that derives RoPE in three lines: rotate the query at position m and the key at position n, take their dot product, and watch the absolute positions cancel because a rotation transposed times a rotation is a rotation by the difference (equation 4.7). The payoff line is that the model gets *relative* position awareness with zero learned parameters and nothing added to the residual stream, position becomes geometry. Pair it with the GQA punchline (query heads are free because you don't cache them; key/value heads are expensive because you do) and you've explained two of the four things that separate a 2017 transformer from a 2025 one, both as consequences of caring about memory.
```
