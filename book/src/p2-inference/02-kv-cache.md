# KV cache arithmetic

```admonish note
Status: stub. Content pending per the authoring workflow (spec section 8).
This file exists so the book builds clean under `create-missing = false`.
```

**Goal.** Compute exact KV bytes and predict max concurrency, then verify in vLLM.

**Covers.** Exact per-token KV bytes as f(layers, kv-heads, head-dim, dtype); context length vs concurrency; FP8 KV.

## Theory

<!-- The concept, with derivations. -->

```admonish derivation
Full inline derivation goes here: intuition first, then the numbered equations, then the code that implements it.
```

```admonish vram-budget
Byte-level memory arithmetic for the RTX 5080 16GB, line by line, to be filled in.
```

## Tooling

<!-- The tool that embodies the concept and how it actually works. -->

## Lab

<!-- Runnable end-to-end, finishing with an artifact on disk and a "what you should see". -->

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
