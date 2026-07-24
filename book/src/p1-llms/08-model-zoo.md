# Anatomy of the open-model zoo

An open-weight model on disk is not a mystery; it is a directory of files with a rigid, readable structure, and being able to open that directory and know exactly what you are holding (how many parameters, in what dtype, dense or mixture-of-experts, tied embeddings or not, and under what license) is a basic literacy the rest of the book assumes. This chapter is that literacy. I will read `config.json` as the model's blueprint, read `safetensors` as its weight store (and explain why it replaced the pickle-based format), draw the dense-versus-MoE distinction with the active-versus-total-parameter arithmetic that makes gpt-oss's headline numbers make sense, pin down the Qwen3 and gpt-oss specifics that recur across the book, and take the license question seriously because "open" means several different things and a thesis has to say which one it relies on. The lab is a model-card inspector that reads any local checkpoint and prints its true anatomy.

## Theory

### config.json is the blueprint

Every Hugging Face model ships a `config.json`, and it is the single source of truth for the architecture. Everything I derived across Part I appears in it as a named field, which is why I have been able to say "read it off the config" all chapter long. The essential fields and where they came from: `model_type` names the architecture class; `hidden_size` is $d_{\text{model}}$; `intermediate_size` is the SwiGLU width $d_{\text{ff}}$ (the $2/3$-adjusted number from equation 4.4); `num_hidden_layers` is the block count $N$; `num_attention_heads` and `num_key_value_heads` are $h$ and $h_{kv}$, whose ratio is the GQA group size (equation 4.10); `head_dim` is $d_{\text{head}}$ (which, recall, is not always `hidden_size / num_attention_heads`); `vocab_size` is $V$; `tie_word_embeddings` says whether the embedding and `lm_head` are one tensor or two (the weight-tying decision from the embeddings chapter); `rms_norm_eps` is the RMSNorm $\epsilon$; `rope_theta` is the RoPE base (equation 4.8); `torch_dtype` is the storage dtype; and `max_position_embeddings` is the trained context length. For MoE models there are additional fields (`num_experts`, `num_experts_per_tok`, and often a `moe_intermediate_size`) that I will use below. The point is that a config is not documentation *about* the model, it *is* the model's shape; instantiate the architecture class with it and you have the model minus the weights, which is exactly what the transformer-block lab did.

### safetensors is the weight store

The weights themselves live in one or more `.safetensors` files (sharded when large, with a `model.safetensors.index.json` mapping each tensor name to its shard). The format is deliberately boring, and its design is a direct reaction to the format it replaced. The older PyTorch `.bin`/`.pt` format is a Python **pickle**, which can execute arbitrary code on load, so downloading a `.bin` from an untrusted source and loading it was a genuine remote-code-execution risk. `safetensors` fixes this by storing only data, never code: the file is a small JSON header followed by a contiguous block of raw tensor bytes.

```admonish under-the-hood
A safetensors file is dead simple to parse by hand, which is why the lab can read one without loading the model. The first 8 bytes are an unsigned little-endian integer giving the header length. The next N bytes are a UTF-8 JSON object mapping each tensor's name to its `dtype`, its `shape`, and its byte `data_offsets` `[begin, end)` into the data blob that follows. After the header come the raw tensor bytes, back to back, no framing. Because the layout is just offsets into a flat blob, loading is a zero-copy memory-map: the library `mmap`s the file and hands you tensor views straight onto the mapped bytes, so a 15 GB checkpoint does not need 15 GB of read-into-RAM before you can use it, and moving to GPU streams directly. That header-plus-blob structure is why I can compute a checkpoint's exact parameter count and dtype breakdown by reading only the JSON header, a few kilobytes, without touching the weights, which is the trick the inspector lab uses.
```

The practical consequences I lean on: safetensors is safe to load from the Hub without trusting the uploader's code, it is fast because of the zero-copy mmap, and its header is machine-readable metadata that tells you the dtype and shape of every tensor in the model without instantiating anything. When a chapter says "the weights are $P \times b$ bytes," the safetensors header is where you verify $P$ and $b$ directly.

### Dense versus mixture-of-experts

The biggest architectural fork in the current zoo is dense versus mixture-of-experts (MoE), and it changes what "parameter count" even means. In a **dense** model, every parameter is used for every token: a forward pass touches all $P$ weights. In an **MoE** model, the feed-forward sublayer of each block is replaced by a set of $E$ expert MLPs plus a small **router**, and for each token the router selects only the top-$k$ experts to run (typically $k = 2$ of $E = 64$ or $128$). The token's feed-forward is computed by just those $k$ experts; the other $E - k$ sit idle for that token. So an MoE model has two very different parameter counts, and confusing them is a classic error.

```admonish derivation title="Active versus total parameters in a mixture-of-experts model"
Let a model have a shared part (embeddings, attention, norms, router) with $P_{\text{shared}}$ parameters, and MoE feed-forward layers with $E$ experts each of size $P_{\text{expert}}$ parameters, of which the router activates $k$ per token. The **total** parameter count, what sits on disk and consumes VRAM, counts every expert:

$$P_{\text{total}} = P_{\text{shared}} + N \cdot E \cdot P_{\text{expert}}, \tag{8.1}$$

summed over the $N$ layers that carry experts. The **active** parameter count, what actually does arithmetic for a given token, and therefore what governs compute cost and latency, counts only the $k$ selected experts per layer:

$$P_{\text{active}} = P_{\text{shared}} + N \cdot k \cdot P_{\text{expert}}. \tag{8.2}$$

The ratio $P_{\text{active}} / P_{\text{total}} \approx k / E$ (when experts dominate the count) is the fraction of the model that lights up per token. For gpt-oss-20b this is about $3.6$B active out of $21$B total, and for gpt-oss-120b about $5.1$B active out of $117$B total, so the 120B model computes like a ~5B model but must be *stored* like a 117B one. This split is the entire strategic point of MoE and the entire problem it creates on a 16GB card: **compute scales with active parameters, but memory scales with total parameters** (equation 8.1 is what has to fit in VRAM). MoE buys you the quality of a big model at the inference *speed* of a small one, but it does nothing for the *memory* of a small one, you still have to hold every expert. That is why a 20B-total MoE, even at 4-bit, is a tight or impossible fit on this card while its 3.6B *active* count sounds harmless, and why reading the *total* count off the config (equation 8.1) rather than the marketing "active" number is a survival skill for planning what runs here.
```

### Qwen3 and gpt-oss specifics

Two model families recur throughout this book, and their concrete choices are worth stating once. **Qwen3** comes in dense variants (0.6B, 1.7B, 4B, 8B, and up) and MoE variants (like the 30B-total/3B-active). Across them it uses the modern block I derived in chapter 4: GQA (the 0.6B has 16 query heads and 8 KV heads), SwiGLU, RMSNorm, RoPE, and, a Qwen3-specific detail, per-head RMSNorm on the query and key projections (`q_norm`/`k_norm`), the small extra tensors that made my hand-built block's parameter count come up slightly short in that chapter's lab. The smaller Qwen3 models tie their embeddings; the config's `tie_word_embeddings` tells you which. Qwen3 is my default workhorse because the dense 0.6B–4B range spans "trivially fits" to "fits with care" on 16GB, which is exactly the regime this book lives in.

**gpt-oss** (OpenAI's open-weight release) is the MoE case study: 20B-total and 120B-total, both mixture-of-experts, and notably shipped natively in **MXFP4**, the 4.25-bits-per-weight microscaling format I derived in chapter 1, for the expert weights, which is how the 120B version is even distributable. Reading a gpt-oss checkpoint is the payoff for the dtype chapter: you will see the expert weights stored as packed `uint8` blocks (two fp4 values per byte, so the safetensors header reports them as `U8`, not a 4-bit dtype) alongside E8M0 `uint8` block-scale tensors, not BF16, and the config's expert and top-$k$ fields let you compute equations (8.1) and (8.2) yourself. gpt-oss also uses attention-sink and sliding-window details that I will meet in the inference part; here the point is that it is the concrete example of MoE-plus-low-bit-quantization, the two techniques that let a "large" model touch a small card at all.

### What "open" means for a thesis

Finally, the license, which is not a footnote for work that has to survive a committee. "Open" spans a wide range and the differences are legally real. **Open weights** means the trained parameters are downloadable and runnable, but it says nothing about the training data, the training code, or what you are permitted to do with the model. The permissive end is a true open-source license (Apache 2.0 or MIT), under which you can use, modify, fine-tune, redistribute, and deploy commercially with essentially only an attribution obligation, Qwen3 is Apache 2.0, and gpt-oss is Apache 2.0, which is a large part of why I chose them as the book's spine: a thesis can build on them without a permissions asterisk. The restrictive end is a **custom community license** (Llama's, for instance) that grants broad use but attaches conditions, acceptable-use policies, a scale threshold above which you need a separate grant, naming/attribution requirements, and restrictions on using outputs to train competing models, which are fine for research but must be *read and cited*, because "I fine-tuned Llama" carries obligations that "I fine-tuned Qwen3-Apache" does not. Separately, **open weights** is not **open source** and neither is **open data**: almost no frontier-scale model publishes its training corpus, so reproducibility claims in a thesis have to be scoped to "from these released weights," not "from scratch." The rule I hold myself to: for every model the thesis depends on, record the exact license, its commercial and derivative terms, and whether the weights suffice to reproduce my results without data I do not have, and prefer Apache/MIT models for the load-bearing parts precisely so those answers are clean.

## Tooling

The tools are the Hugging Face Hub client and the `safetensors` library, plus `transformers`' `AutoConfig`. The Hub client (`huggingface_hub`) can list and download a repo's files and, importantly, fetch *just* the metadata, you can read a `config.json` and even a safetensors *header* without pulling the multi-gigabyte weights, using `hf_hub_download` for the small files or the `safetensors` `safe_open` context manager to inspect tensor keys, dtypes, and shapes lazily. `AutoConfig.from_pretrained` parses `config.json` into a typed object with all the fields from the theory. The `safetensors.torch.load_file` and `safe_open` functions give you the zero-copy tensor access described above. Together these let the inspector lab answer "what is this model, really" from metadata alone, which is both fast and safe (no code execution, no full download). This metadata-first approach is how you triage a candidate model, total parameters, dtype, dense-or-MoE, license, before committing disk and VRAM to it, which on a bandwidth- and space-constrained single machine (2TB of local SSD plus the 4TB NAS) is a real workflow, not a nicety.

```admonish gotcha
The `torch_dtype` field in `config.json` is the dtype the model was *saved* in, but for MoE models the expert weights may be stored in a *different* dtype than the config's headline value, gpt-oss keeps most tensors in BF16 but the experts in MXFP4, so a single `torch_dtype` cannot describe the whole checkpoint. The only way to know each tensor's true dtype is to read the safetensors header per tensor, which is exactly why the inspector below iterates tensors rather than trusting the config's one-line dtype. Sizing VRAM from `torch_dtype × param_count` alone will mis-estimate any mixed-dtype checkpoint; sum the actual per-tensor bytes instead.
```

## Lab

The goal is a model-card inspector: point it at a local (or Hub) checkpoint and have it print the true anatomy, parameter count, per-dtype byte breakdown, dense-versus-MoE with active/total split, GQA ratio, tying, and license, reading from the config and the safetensors header without instantiating the model. The artifact is a JSON model card on disk, which becomes the standard triage record for every model the thesis touches.

```bash
uv init labs/model-zoo
cd labs/model-zoo
uv add transformers safetensors huggingface_hub
```

```python title="labs/model-zoo/inspect_model.py"
"""Inspect a checkpoint's anatomy from config + safetensors header only.

Artifact: artifacts/model_card_<name>.json
Usage: uv run python inspect_model.py Qwen/Qwen3-0.6B
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from transformers import AutoConfig

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

DTYPE_BYTES = {  # bytes per element for common safetensors dtypes
    "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I8": 1, "U8": 1, "I32": 4, "I64": 8,
    # safetensors has no MXFP4/F4 dtype: gpt-oss stores 4-bit MoE experts as
    # packed U8 *_blocks (two fp4 values per byte) plus U8 *_scales.
}

def inspect(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id)
    files = list_repo_files(model_id)
    shards = [f for f in files if f.endswith(".safetensors")]

    dtype_params = defaultdict(int)   # dtype -> element count
    dtype_bytes = defaultdict(float)  # dtype -> bytes
    total_params = 0
    for shard in shards:
        local = hf_hub_download(model_id, shard)
        with safe_open(local, framework="pt") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                shape = sl.get_shape()
                dtype = sl.get_dtype()
                n = 1
                for d in shape:
                    n *= d
                # gpt-oss packs each MXFP4 expert matrix into a uint8 *_blocks
                # tensor at two fp4 values per byte, so the logical parameter
                # count is twice the stored element count (the *_scales tensors
                # are E8M0 uint8 block scales, counted as-is). Size storage from
                # the real stored bytes, not the doubled parameter count.
                n_params = n * 2 if key.endswith("_blocks") else n
                total_params += n_params
                dtype_params[dtype] += n_params
                dtype_bytes[dtype] += n * DTYPE_BYTES.get(dtype, 2)

    is_moe = hasattr(cfg, "num_experts") and getattr(cfg, "num_experts", 0)
    card = {
        "model": model_id,
        "model_type": cfg.model_type,
        "total_params": total_params,
        "total_params_readable": f"{total_params/1e9:.2f}B",
        "storage_bytes": int(sum(dtype_bytes.values())),
        "storage_GB": round(sum(dtype_bytes.values()) / 1e9, 3),
        "dtype_breakdown": {
            k: {"params": v, "GB": round(dtype_bytes[k]/1e9, 3)}
            for k, v in dtype_params.items()
        },
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "gqa_group_size": (getattr(cfg, "num_attention_heads", 0)
                           // max(getattr(cfg, "num_key_value_heads", 1), 1)),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "context_length": getattr(cfg, "max_position_embeddings", None),
        "is_moe": bool(is_moe),
    }
    if is_moe:
        E = getattr(cfg, "num_experts", 0)
        k = getattr(cfg, "num_experts_per_tok", 0)
        card["moe"] = {
            "num_experts": E,
            "experts_per_token": k,
            "active_fraction": round(k / E, 4) if E else None,
            "note": "active params ~= shared + (k/E) * expert params (eq 8.2)",
        }
    return card

def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    card = inspect(model_id)
    name = model_id.replace("/", "_")
    path = OUT / f"model_card_{name}.json"
    path.write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"\nArtifact: {path.resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python inspect_model.py Qwen/Qwen3-0.6B
# then try an MoE checkpoint to see the active/total split, e.g.:
# uv run python inspect_model.py Qwen/Qwen3-30B-A3B
```

```admonish gotcha
`list_repo_files` plus `hf_hub_download` on every shard will pull the full weights the first time you inspect a large model, which defeats the "metadata only" goal for a 120B checkpoint. For true header-only inspection on huge models, fetch just the `model.safetensors.index.json` (a small file mapping tensor names to shards and, in newer exports, carrying dtype/shape metadata) and parse that, or read only the first shard's header. The script as written is fine for the small Qwen3 models the book uses day to day; the header-only optimization matters exactly when the model is too big to want on disk, which is the case the metadata-first workflow is *for*. Also, gpt-oss does not expose an MXFP4 dtype to safetensors at all: it stores each expert weight matrix as a packed `uint8` `*_blocks` tensor (two fp4 values per byte) next to an E8M0 `uint8` `*_scales` tensor, so the header reports `U8` and a naive element count undercounts the expert parameters by 2x. The inspector detects the `*_blocks` tensors and doubles their logical parameter count (while still sizing storage from the real `uint8` bytes), which is why its total for gpt-oss matches the published figure instead of coming in at half; the tiny `*_scales` bytes are left counted as-is.
```

**What you should see.** The script writes `artifacts/model_card_<name>.json` and prints a complete anatomy of the checkpoint from metadata: the true total parameter count (verified against the safetensors headers, not the model name), a per-dtype breakdown showing exactly how many parameters and gigabytes sit in each format, the GQA group size read straight from the head counts, whether embeddings are tied, the context length, and, for an MoE model, the expert count, top-$k$, and active fraction that feed equations (8.1) and (8.2). Running it on the dense Qwen3-0.6B shows a clean single-dtype BF16 checkpoint around 1.2 GB; running it on a Qwen3 MoE or gpt-oss checkpoint shows the total-versus-active gap that is the whole point of the mixture-of-experts section, and (for gpt-oss) a mixed-dtype breakdown with the experts as packed `uint8` blocks (two fp4 values per byte, doubled back to their true parameter count) plus their `uint8` block scales. This JSON card is the triage artifact I keep for every model the thesis uses: it answers "will this fit, is it dense or sparse, and can I legally build on it" before I spend disk or VRAM. Record the license separately from the card (the inspector reads architecture, not the license file); pull it from the repo's `LICENSE` and note it in your thesis's model table.

```admonish read-along
Read **[BRM]** ch. 1 for the survey of the open reasoning-model landscape and App. C for the Qwen3 architecture in full source, which is the concrete backing for the Qwen3 specifics here (the GQA head counts, the q/k norms, the SwiGLU and RoPE settings you will see this lab's inspector report). Raschka's appendix walks the exact config fields this chapter reads, so running the inspector on Qwen3-0.6B while reading App. C lets you match every printed number to a line of the architecture it describes.
```

```admonish substack-seed
"A 120-billion-parameter model that computes like a 5-billion-parameter one, and still won't fit on your GPU." A post that untangles the two parameter counts every mixture-of-experts model has: total params (every expert, what you must store) versus active params (the two-of-many experts each token actually uses, what governs speed). The clean punchline is that MoE decouples compute from memory, you get big-model quality at small-model latency, but you pay big-model VRAM regardless, which is exactly why gpt-oss ships its experts in a 4-bit microscaling format just to be distributable. Wrap it with the practical literacy of reading a safetensors header (an 8-byte length, a JSON map of offsets, and a zero-copy mmap that replaced the arbitrary-code-execution risk of pickle files) and you've taught a reader to open any model on the Hub and know what they're actually holding.
```
