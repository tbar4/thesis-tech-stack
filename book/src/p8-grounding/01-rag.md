# RAG over space text

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 8.1).
This stub exists so the book builds under `create-missing = false`.
```

**Goal.** Retrieve the pipeline's article corpus (spacenews / NASA) to ground
qualitative / recency SDA answers, and measure whether it actually helps.

**Covers.** Why RAG is for text not numbers; chunking; a local embedding model
served via vLLM (co-resident memory budget on 16GB); a LanceDB index; a thin
hand-rolled retrieval loop; retrieval metrics (recall@k, MRR) vs end-task metrics;
RAG-as-confound (retrieval leakage, ties to 3.8 and Part IV).

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
