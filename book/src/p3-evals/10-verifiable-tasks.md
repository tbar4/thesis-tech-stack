# From orbital data to verifiable tasks

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 3.10).
This stub exists so the book builds under `create-missing = false`.
```

**Goal.** Turn pipeline snapshots into verifiable SDA task instances (prompt +
machine-checkable ground truth), with a Skyfield/SGP4 physics oracle that both
generates the gold answer and grades the model's answer. Conjunction screening is
the flagship task.

**Covers.** Task families (conjunction screening, orbital-element derivation,
decay/reentry ordering, pass prediction, catalog correlation); the oracle as
verifier plugged into the `thesis_suite` API; numeric tolerance policy; difficulty
calibration (feeds 7.5); item counts from 3.7; dedup against 3.8; output freezes in
3.11.

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
