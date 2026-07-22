# KV cache arithmetic

The last chapter ended on a threat: as context grows, the KV cache read per token grows with it, and per-sequence throughput falls. This chapter cashes that threat out to the byte. The KV cache is the single most important number in single-GPU serving, because on a 16GB card the weights eat most of the memory and whatever is left is the entire budget you have for context and concurrency. Get the arithmetic right and you can predict, before launching a server, exactly how many sequences you can run at once and how long each one's context can be. Get it wrong and vLLM greets you with an out-of-memory error thirty seconds after boot.

## Theory

### What the cache stores, and why

Attention at position $t$ needs the keys and values of every previous position $1 \ldots t$. Recomputing them each step would make decode quadratic in sequence length. Instead we compute each position's K and V once, during prefill or at the step that produced it, and cache them. Decode then reads the stored K and V and appends one new pair per layer. The cache trades memory for compute, and on this machine memory is the scarce resource, so the trade needs auditing. Without the cache, generating token $t$ would recompute the K and V of all $t-1$ prior positions at every step, making the total decode cost scale with the square of the sequence length; with it, each position's K and V are computed exactly once and then only read. The cache is therefore not an optimization you can toggle off on a whim, it is what makes autoregressive decoding linear instead of quadratic, and the price for that linearity is paid entirely in VRAM.

### The exact per-token byte formula

For one token, one layer stores a K vector and a V vector, each of dimension (number of KV heads) times (head dimension). With grouped-query attention (GQA), the number of KV heads $H_{kv}$ is smaller than the number of attention heads, which is the first big lever on cache size. Summing over all layers, and counting both K and V:

$$\boxed{\ B_{\text{tok}} = 2 \times L \times H_{kv} \times d_h \times b\ }$$

where $L$ is the number of transformer layers, $H_{kv}$ the number of KV heads, $d_h$ the head dimension, $b$ the bytes per element (2 for BF16/FP16, 1 for FP8), and the leading 2 counts keys and values. That is the whole formula. Everything else is plugging in a model's `config.json`.

```admonish read-along title="Pull the four numbers from config.json"
The formula needs exactly four fields, all present in any Hugging Face model's `config.json`: `num_hidden_layers` ($L$), `num_key_value_heads` ($H_{kv}$), `head_dim` ($d_h$, or `hidden_size / num_attention_heads` if absent), and the KV cache dtype you intend to serve with. Open the config for each model in the repertoire and confirm the values I use below before trusting any budget I derive; model revisions change these.
```

### Working the repertoire

I will use these config values (verify them with the read-along; they are from each model's published config):

| Model | $L$ | $H_{kv}$ | $d_h$ | KV dtype |
|---|---|---|---|---|
| Qwen3-8B | 36 | 8 | 128 | BF16 (2 B) |
| Qwen3-14B | 40 | 8 | 128 | BF16 (2 B) |
| gpt-oss-20b | 24 | 8 | 64 | BF16 (2 B) |

```admonish derivation title="Per-token KV bytes for each model"
**Qwen3-8B, BF16.**
$$B_{\text{tok}} = 2 \times 36 \times 8 \times 128 \times 2 = 147{,}456\ \text{byte} = 144\ \text{KiB/token}$$

**Qwen3-14B, BF16.**
$$B_{\text{tok}} = 2 \times 40 \times 8 \times 128 \times 2 = 163{,}840\ \text{byte} = 160\ \text{KiB/token}$$

**gpt-oss-20b, BF16** (ignoring its sliding-window layers, see the gotcha).
$$B_{\text{tok}} = 2 \times 24 \times 8 \times 64 \times 2 = 49{,}152\ \text{byte} = 48\ \text{KiB/token}$$

So one 32,768-token context on Qwen3-8B costs $147{,}456 \times 32{,}768 = 4.83\times10^{9}\ \text{byte} = 4.5\ \text{GiB}$ of cache for a single sequence. On a 16GB card that is enormous relative to what is left after the weights, which is the crux of this chapter.
```

Notice what GQA bought. Qwen3-8B has 32 attention heads but only 8 KV heads, a 4x reduction. Without GQA the cache would be $32/8 = 4\times$ larger, 576 KiB/token, and a single full-context sequence would need 18 GiB, more than the whole card. GQA is not a minor optimization; it is what makes long-context serving on 16GB possible at all.

### The context-length vs concurrency tradeoff

The KV budget is a fixed pool of bytes. Call it $M_{kv}$. You spend it on the product of concurrency and context length:

$$N_{\text{seq}} \times S_{\text{ctx}} \times B_{\text{tok}} \le M_{kv}$$

Rearranged, the maximum number of concurrent sequences you can hold, each at context $S_{\text{ctx}}$:

$$N_{\text{seq}}^{\max} = \left\lfloor \frac{M_{kv}}{S_{\text{ctx}} \times B_{\text{tok}}} \right\rfloor$$

This is a hyperbola: double the context you promise each request and you halve how many requests you can serve at once. There is no free parameter here; the only way to move the curve is to shrink $B_{\text{tok}}$ (fewer KV heads, or a smaller KV dtype) or grow $M_{kv}$ (a smaller/quantized model that frees VRAM).

The hyperbola is worth sitting with because it is the single most consequential shape in single-GPU serving, and it is unforgiving. Suppose I have a 6 GiB pool and a model costing 160 KiB/token. A request stream where everyone uses 2K of context lets me hold about nineteen sequences at once; stretch the promised context to 8K and I am down to four; promise the full 32K and I hold exactly one. The advertised context window and the advertised concurrency are not two independent product decisions I get to make separately, they are the same fixed budget read off two axes, and every honest capacity claim has to name both. A server that says "32K context" and "16 concurrent users" in the same breath, on a pool that only supports their product, is writing a check the KV allocator will bounce. Worse, the failure is not graceful by default: admit too many long requests and vLLM starts preempting (the next chapter's subject), trading throughput for the privilege of not crashing. The way out is to decide the operating point deliberately, size `--max-model-len` and `--max-num-seqs` from this inequality, and let the arithmetic, not optimism, set the numbers.

### FP8 KV cache

The cleanest lever available at serve time is the KV dtype. Storing K and V in FP8 (`e4m3` or `e5m2`) instead of BF16 halves $b$ from 2 to 1, which halves $B_{\text{tok}}$ and therefore doubles either concurrency or context for the same pool. Qwen3-8B drops from 144 to 72 KiB/token; a full 32K context falls from 4.5 to 2.25 GiB.

The cost is precision in the cached activations. K and V are activations, not weights, and attention is fairly tolerant of their quantization because the softmax is a smoothing operation and the scores are dominated by a few large logits. FP8 KV typically moves eval metrics by a fraction of a point (measure it; the quantization chapter's harness is the tool). The `e4m3` variant (4 exponent, 3 mantissa bits) is the usual choice for KV because K and V values are well-behaved in range; `e5m2` trades mantissa for range and is rarely needed here. In vLLM the flag is `--kv-cache-dtype fp8`, and on Blackwell it maps to native FP8 tensor support.

```admonish gotcha
gpt-oss-20b alternates sliding-window attention layers (window ~128 tokens) with full-attention layers. The sliding-window layers never cache more than the window, so their per-token contribution saturates and the naive $B_{\text{tok}}$ formula overcounts the cache at long context. My 48 KiB/token figure treats every layer as full-attention, which is an upper bound. The real long-context cache is smaller and grows more slowly. When you budget for gpt-oss, treat 48 KiB/token as a conservative ceiling and let vLLM's actual allocation (it knows the attention pattern) be the truth. Always confirm against the server's reported number of KV blocks at boot.
```

```admonish vram-budget title="KV budget on the RTX 5080 16GB"
The card has 16 GiB. vLLM reserves a fraction via `--gpu-memory-utilization` (default 0.90); the rest holds weights, activations/CUDA-graph overhead, and the KV pool. Approximate activation + overhead as ~1 GiB (measure it; record value, date, driver). The KV pool is what remains:

$$M_{kv} \approx 16\ \text{GiB} \times \texttt{gpu\_mem\_util} - W_{\text{weights}} - 1\ \text{GiB}$$

**Qwen3-8B at BF16** ($W \approx 15.3\ \text{GiB}$, $B_{\text{tok}} = 144\ \text{KiB}$). At `util=0.95` the pool is $16 \times 0.95 - 15.3 - 1 \approx -1.1\ \text{GiB}$: it does not fit with headroom. This model at full BF16 is right at the edge of the card; you must run short `--max-model-len` and accept batch size near 1, or move to FP8 KV to claw back room. This tension is the honest reason the repertoire also carries a 4-bit 14B.

| max ctx $S$ | KV per seq (BF16) | KV per seq (FP8) |
|---|---|---|
| 4,096 | 0.56 GiB | 0.28 GiB |
| 8,192 | 1.12 GiB | 0.56 GiB |
| 16,384 | 2.25 GiB | 1.12 GiB |
| 32,768 | 4.50 GiB | 2.25 GiB |

**Qwen3-14B at AWQ 4-bit** ($W \approx 7.75\ \text{GiB}$, $B_{\text{tok}} = 160\ \text{KiB}$). At `util=0.92` the pool is $16 \times 0.92 - 7.75 - 1 \approx 5.9\ \text{GiB}$, a comfortable budget. This is why AWQ is the workhorse: quantizing the weights frees several GiB that become KV cache, i.e. concurrency and context.

| max ctx $S$ | KV per seq (BF16 KV) | max concurrent seqs at that ctx |
|---|---|---|
| 4,096 | 0.625 GiB | $\lfloor 5.9 / 0.625 \rfloor = 9$ |
| 8,192 | 1.25 GiB | $\lfloor 5.9 / 1.25 \rfloor = 4$ |
| 16,384 | 2.50 GiB | $\lfloor 5.9 / 2.50 \rfloor = 2$ |
| 32,768 | 5.00 GiB | $\lfloor 5.9 / 5.00 \rfloor = 1$ |

Every number here is derived; the ~1 GiB overhead and the exact weight sizes are the placeholders to pin down on the machine. Record measured values with date and driver.
```

The 14B-AWQ table is the whole argument for the format choice in one place. At a 4K context I can hold nine concurrent reasoning traces; stretch to 32K and I am down to one. The concurrency I can advertise and the context I can promise are the same budget viewed from two ends, and the KV formula is the exchange rate between them.

### The cost of reasoning traces specifically

There is a reason this arithmetic bites harder for reasoning models than for chat models, and it is worth naming because the whole thesis is about reasoning. A chat model answers in a few hundred tokens; a reasoning model thinks out loud, and a hard problem can run its chain of thought to thousands of tokens before it emits a final answer. Every one of those thinking tokens is a position that must be cached, so the effective context per request is not the prompt length, it is the prompt plus the entire trace. On Qwen3-8B at 144 KiB/token, a 4,000-token reasoning trace on top of a 1,000-token prompt occupies $5{,}000 \times 144\ \text{KiB} = 703\ \text{MiB}$ of cache for that one sequence, and it grows as the trace unfolds. This is why the concurrency I can sustain for a reasoning workload is lower than a chat benchmark would suggest, and why FP8 KV is so tempting here: the model that most needs long context is exactly the one whose cache I most want to halve. When Part III runs an eval suite that lets models reason to completion, the KV budget, not the compute, is what caps how many problems I can grade in parallel.

### KV cache in the roofline: the read grows with context

One more connection back to chapter 1 closes the loop. The decode ceiling was $BW / B_{\text{read}}$, and I approximated $B_{\text{read}}$ as just the weight bytes for short context. That approximation frays as the trace lengthens, because each decode step must also read the entire KV cache for the current context. At position $t$ the extra read is $t \times B_{\text{tok}}$, so a Qwen3-8B sequence at 8,000 tokens of context adds $8{,}000 \times 144\ \text{KiB} = 1.1\ \text{GiB}$ of KV read *per token*, on top of the ~16.4 GB of weights. At that context the KV read is a small fraction of the weight read, so the per-token ceiling barely moves; but push to very long contexts, or batch many long sequences, and the KV read starts to dominate and decode slows measurably as the conversation grows. This is the mechanism behind the familiar experience of a long chat getting sluggish near the end: you are not imagining it, the bytes-read-per-token is genuinely climbing, and the KV formula is what quantifies the slowdown.

## Tooling

vLLM does this accounting internally and prints the result at startup. The line to watch is the reported number of **GPU KV cache blocks** (vLLM allocates the cache in fixed-size blocks, typically 16 tokens each; the next chapter dissects why). Multiply blocks by block size to get total cached tokens, which is $N_{\text{seq}} \times S_{\text{ctx}}$ from the formula. The relevant flags:

- `--max-model-len`: the per-sequence context ceiling $S_{\text{ctx}}$. Lowering it shrinks the worst-case reservation and lets vLLM admit more sequences.
- `--max-num-seqs`: a hard cap on concurrent sequences $N_{\text{seq}}$, independent of memory. vLLM takes the min of this and what the KV pool allows.
- `--gpu-memory-utilization`: sets the fraction of the 16 GiB vLLM may claim, hence $M_{kv}$.
- `--kv-cache-dtype fp8`: halves $B_{\text{tok}}$.

The verification loop is: compute $N_{\text{seq}}^{\max}$ by hand, launch with generous `--max-num-seqs`, read the block count vLLM prints, convert to sequences, and check it against my prediction.

## Lab: predict max concurrency, then verify in vLLM

The artifact is `kv_report.json`: my hand-derived prediction next to vLLM's reported capacity, so the two can be diffed.

### Set up

```bash title="shell"
uv init kv-lab && cd kv-lab
uv add "vllm>=0.6" requests
```

### Predict from config

```python title="kv-lab/predict.py"
"""Predict max concurrent sequences from config.json + a VRAM budget."""
import json, argparse

def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes

def predict(layers, kv_heads, head_dim, dtype_bytes, kv_pool_gib, ctx):
    btok = kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes)
    pool = kv_pool_gib * (1024**3)
    per_seq = ctx * btok
    return {
        "kv_bytes_per_token": btok,
        "kv_kib_per_token": round(btok / 1024, 1),
        "ctx": ctx,
        "kv_gib_per_seq": round(per_seq / (1024**3), 3),
        "predicted_max_seqs": int(pool // per_seq),
        "predicted_total_tokens": int(pool // btok),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--dtype-bytes", type=float, default=2.0)
    ap.add_argument("--kv-pool-gib", type=float, required=True,
                    help="VRAM left for KV after weights+overhead (derive from vram-budget)")
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()
    print(json.dumps(predict(args.layers, args.kv_heads, args.head_dim,
                             args.dtype_bytes, args.kv_pool_gib, args.ctx), indent=2))
```

For Qwen3-14B AWQ with a ~5.9 GiB pool at 8K context:

```bash title="shell"
uv run python predict.py --layers 40 --kv-heads 8 --head-dim 128 \
    --dtype-bytes 2 --kv-pool-gib 5.9 --ctx 8192
# predicted_max_seqs: 4  (matches the vram-budget table)
```

### Verify against vLLM

Launch the server and let it report its real KV capacity:

```bash title="shell"
uv run vllm serve Qwen/Qwen3-14B-AWQ --quantization awq_marlin \
    --max-model-len 8192 --max-num-seqs 32 \
    --gpu-memory-utilization 0.92 2>&1 | tee serve.log
```

At startup vLLM logs a line like `# GPU blocks: 2960, # CPU blocks: ...`. This script parses it and computes vLLM's implied max sequences, then writes the comparison report:

```python title="kv-lab/verify.py"
"""Parse vLLM's reported KV blocks and compare to the hand prediction."""
import re, json, argparse
from predict import predict

BLOCK_SIZE = 16  # vLLM default tokens/block; confirm in your serve.log

def parse_gpu_blocks(logpath):
    pat = re.compile(r"GPU (?:KV cache size|blocks)[:=]?\s*([\d,]+)", re.I)
    with open(logpath) as f:
        for line in f:
            m = pat.search(line)
            if m:
                return int(m.group(1).replace(",", ""))
    raise SystemExit("Could not find GPU blocks line; check serve.log format.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="serve.log")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype-bytes", type=float, default=2.0)
    ap.add_argument("--kv-pool-gib", type=float, default=5.9)
    args = ap.parse_args()

    blocks = parse_gpu_blocks(args.log)
    total_cached_tokens = blocks * BLOCK_SIZE
    vllm_max_seqs = total_cached_tokens // args.ctx

    pred = predict(args.layers, args.kv_heads, args.head_dim,
                   args.dtype_bytes, args.kv_pool_gib, args.ctx)
    report = {
        "prediction": pred,
        "vllm_gpu_blocks": blocks,
        "vllm_total_cached_tokens": total_cached_tokens,
        "vllm_implied_max_seqs_at_ctx": int(vllm_max_seqs),
        "agreement": f"{pred['predicted_max_seqs']} predicted vs "
                     f"{int(vllm_max_seqs)} observed",
    }
    with open("kv_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
```

```bash title="shell"
uv run python verify.py --log serve.log --ctx 8192
```

### What you should see

`kv_report.json` on disk with two numbers that should land within one or two sequences of each other. My hand calculation for Qwen3-14B AWQ at 8K context predicts 4 concurrent sequences; vLLM's reported GPU-block count, converted through the 16-token block size, should imply roughly the same total cached tokens and therefore about the same sequence count. A small discrepancy is expected and instructive: vLLM's real overhead is not exactly my 1 GiB estimate, and it rounds context up to whole blocks, so its usable pool differs slightly from my back-of-envelope $M_{kv}$. If the numbers are wildly apart (say I predicted 4 and it reports capacity for 20), the likely culprit is a wrong `--gpu-memory-utilization` assumption or an $H_{kv}$ I read from the wrong config revision; go back and re-pull the four fields. When they agree, I have earned the right to size context and concurrency from arithmetic alone, before ever launching a server, which is exactly the muscle the rest of Part II leans on. Record the measured block count, the derived pool size, date, and driver next to the file.

```admonish substack-seed
"The 4 GiB sentence that decides your context length." Everyone quotes context windows like they are free. On a 16GB GPU they are anything but: one 32K-token conversation on an 8B model reserves 4.5 GiB of pure cache, computed from four integers in a config file (layers times KV-heads times head-dim times two, times two for keys and values). This post turns that one formula into a planning tool. It shows why grouped-query attention is the unsung hero that makes long context fit at all, why 4-bit weights buy you concurrency rather than just speed, and how an FP8 KV cache doubles your seats for a fraction of a percentage point of accuracy. The reader leaves able to answer, on a napkin, "how many users can this box hold at once?" before spending a cent on a server.
```
