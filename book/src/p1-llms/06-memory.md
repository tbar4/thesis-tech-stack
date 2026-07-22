# Where memory goes: training vs inference

```admonish note
Status: stub. Content pending per the authoring workflow (spec section 8).
This file exists so the book builds clean under `create-missing = false`.
```

**Goal.** Account for every byte in training vs inference on the 16GB card.

**Covers.** Forward activations, backprop, optimizer states (AdamW = 2 extra copies), gradient checkpointing, why inference is weights + KV and training is 4-16x more.

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
