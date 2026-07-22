# vLLM internals

```admonish note
Status: stub. Content pending per the authoring workflow (spec section 8).
This file exists so the book builds clean under `create-missing = false`.
```

**Goal.** Open the hood on how vLLM actually serves.

**Covers.** PagedAttention (blocks, block tables, fragmentation); continuous batching and scheduling; prefix caching; chunked prefill; the OpenAI-compatible surface.

## Theory

<!-- The concept, with derivations. -->

## Tooling

<!-- The tool that embodies the concept and how it actually works. -->

```admonish under-the-hood
Implementation internals to be drafted (e.g. block tables, kernel fusion, packing layout).
```

## Lab

<!-- Runnable end-to-end, finishing with an artifact on disk and a "what you should see". -->

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
