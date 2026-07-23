# The SDA data pipeline

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 3.9).
This stub exists so the book builds under `create-missing = false`.
```

**Goal.** Turn five live space-data sources (celestrak, space-track, api.nasa.gov,
thespacedevs / Launch Library 2, Spaceflight News API) into a normalized,
provenance-stamped, reproducible store, orchestrated with Airflow 3 (thin DAGs
over plain `uv run` task modules, containerized).

**Covers.** The sources by data type; why an orchestrator rather than cron;
Airflow 3 assets / data-aware scheduling; normalization (Pydantic) and quality
gates (pandera); immutable snapshots stored as Parquet and queried with DuckDB;
DVC-pinned versioning and per-record provenance; the open-data / licensing / auth
boundary as a first-class constraint (space-track is fetch-only).

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
