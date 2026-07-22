# Design spec: grounding the SDA model in real space data

**Status:** approved design, pre-implementation-plan
**Date:** 2026-07-22
**Scope:** three additions to *Evals as Rewards* that ground the Space Domain
Awareness (SDA) reasoning model in real space data, plus the causal comparison
that makes them thesis-load-bearing.

---

## 0. Summary

Today the book's SDA tasks (the frozen thesis suite) and training curriculum are
abstract: "verifiable reasoning tasks" with a placeholder toy checker (the
"elements in A not in B" set-difference examples). This spec replaces that with a
real, physics-grounded SDA stack:

- **A. A data-engineering pipeline** that pulls live space data and turns it into
  verifiable SDA task instances with machine-checkable ground truth.
- **B. RAG** over the pipeline's *text* corpus (news, NASA articles) for
  qualitative/recency grounding.
- **C. MCP tools** that fetch *live numerical* data (conjunctions, catalog,
  ephemerides) at inference.
- **D. A causal four-arm comparison** (base / RL-trained / +RAG / +tools) that
  isolates how much of the SDA reasoning gain is parametric vs retrieval vs tool
  access.

**Why it belongs in this book.** (1) Orbital mechanics is a free, exact reward
oracle, which is what makes "verifiable reward" real for SDA. (2) RAG and tools
are rival explanations for the headline claim ("RL improved SDA reasoning"), so
building them turns "I added RAG" into "I isolated the causal contribution of
each" — which plugs straight into Part IV and the 7.6 delta. (3) It upgrades the
Thesis Thread from a toy to genuine Space Domain Awareness, with **conjunction
screening** as the flagship task.

**Deferred / out of scope.** Training the model to call tools (agentic RL) is
flagged as a post-contract extension, not built. Business-specific (Panoptes)
content stays out; every tool named here is open source, so the book stays
publishable. Non-redistributable data (space-track) is fetch-only, never shipped.

---

## 1. Structure and renumber map

Chosen: renumber into the existing structure (not a new interstitial part label).

### Part III — Evaluation Engineering (insert two, renumber two)

| New # | Chapter | File |
|---|---|---|
| 3.9 | **The SDA data pipeline** (new) | `p3-evals/09-sda-data-pipeline.md` |
| 3.10 | **From orbital data to verifiable tasks** (new) | `p3-evals/10-verifiable-tasks.md` |
| 3.11 | Building the thesis task suite (was 3.9) | `p3-evals/09-thesis-suite.md` → `11-thesis-suite.md` |
| 3.12 | Eval ops (was 3.10) | `p3-evals/10-eval-ops.md` → `12-eval-ops.md` |

### New Part VIII — Grounding: Retrieval and Tools (new dir `p8-grounding/`)

| # | Chapter | File |
|---|---|---|
| 8.1 | **RAG over space text** | `p8-grounding/01-rag.md` |
| 8.2 | **MCP tools for live SDA data** | `p8-grounding/02-mcp-tools.md` |
| 8.3 | **The augmentation arms: what caused the gain** | `p8-grounding/03-augmentation-arms.md` |

### Cascade (parts shift by one)

- Part VIII "Burst and Scale" → **Part IX**; dir `p8-burst/` → `p9-burst/`; prose `8.x` → `9.x`.
- Part IX "Assembly" → **Part X**; dir `p9-assembly/` → `p10-assembly/`; prose `9.x` → `10.x`.
- Appendices A–F unchanged.

### Mechanics

- `git mv` the renamed files/dirs; update every path in `SUMMARY.md`.
- Global prose sweep of chapter-number references: `3.9→3.11`, `3.10→3.12`,
  every `Part VIII`/`8.x` → `Part IX`/`9.x`, every `Part IX`/`9.x` →
  `Part X`/`10.x`. Order the sweep high-to-low to avoid double-rewrites (do the
  9→10 and 8→9 shifts before reusing "8"/"9" for new content).
- The `thesis_suite` "from chapter 3.9" note becomes "from chapter 3.11".
- `create-missing = false`, so **every new file must exist as a stub before the
  build passes**; add stubs first.
- The new PR-CI linkcheck plus a `grep -rEn '\b(3\.9|3\.10|8\.[0-9]|9\.[0-9])\b'`
  audit is the safety net for missed references.

---

## 2. Locked technology stack

Philosophy: standard SDK for anything protocol- or physics-shaped (trust and
correctness); thin hand-rolled glue where the book wants transparency;
embedded/in-process tools over anything needing its own server (one 16GB node).

| Concern | Tool | Notes |
|---|---|---|
| Orchestration | **Apache Airflow 3** | assets / data-aware scheduling; containerized via Docker Compose (extends 0.6); **thin DAGs over plain `uv run` task modules** so every task runs standalone |
| HTTP fetch | `httpx` + **`spacetrack`** lib | `spacetrack` handles space-track auth/login/rate-limits |
| Orbital oracle | **`sgp4`** + **Skyfield** | the verifier's ground-truth propagator; never hand-rolled |
| Validation | **Pydantic** (ingest) + **pandera** (dataframe gates) | typed source models + quality checks |
| Store | raw immutable snapshots → **Parquet**, queried with **DuckDB** | single-machine OLAP, zero server |
| Data versioning | **DVC** | pin snapshots by hash, git-adjacent ("revision, not name") |
| Embeddings | local model (`BAAI/bge-*` / `Qwen3-Embedding`) served via **vLLM** | unifies serving; `sentence-transformers` fallback |
| Vector store | **LanceDB** | embedded, on-disk, no server |
| RAG retrieval loop | **thin hand-rolled** (chunk → embed → top-k → assemble; optional cross-encoder rerank) | legible; LlamaIndex named as the batteries-included alternative, not used |
| MCP server | **official `mcp` SDK / FastMCP** | tools wrap the 3.9 clients + Skyfield oracle; stdio local, streamable-HTTP if needed |
| Tool-use loop / agentic eval | vLLM OpenAI tool-calling + MCP client (thin loop); **Inspect** tools for the eval | reuses 3.4's agentic/sandbox eval |

### uv project / environment layout (extends the two-env doctrine, 0.4)

- `data/` — pipeline + Airflow + DuckDB/DVC/Skyfield (CPU/IO, no GPU).
- `serve/` — gains the embedding-model serving + LanceDB client for RAG.
- `mcp/` — the FastMCP server + source clients (light, IO-bound).
- Each keeps its own `pyproject.toml` + committed `uv.lock`.

---

## 3. New chapters (Goal · Covers · Derivations · Lab → Artifact · Read-along)

### 3.9 The SDA data pipeline

**Goal.** Turn five live space-data sources into a normalized, provenance-stamped,
reproducible store, orchestrated for real (scheduled, incremental, backfillable).

**Covers.** The sources by data type — celestrak (TLE/GP sets, SATCAT, SOCRATES
conjunction reports; public), space-track (authoritative catalog, GP history,
**CDMs / conjunction messages**, decay; auth'd, rate-limited, redistribution-
restricted), api.nasa.gov (NeoWs, DONKI space weather; free key),
thespacedevs / Launch Library 2 + Spaceflight News API (launches, events,
articles; public). The "why an orchestrator, not cron" argument (live feeds →
scheduled incremental pulls + backfill + retries + lineage). The thin-DAG
discipline: task logic in plain `uv run` modules, Airflow only schedules. Airflow
3 assets / data-aware scheduling (the `conjunction_tasks` asset rebuilds when the
`tle_snapshot` asset updates). Normalization to a common schema (objects,
element-sets-at-epoch, events, articles) with Pydantic; pandera quality gates.
Storage as immutable raw snapshots → Parquet, queried with DuckDB. Versioning and
provenance: DVC-pinned snapshots + per-record source/fetch-time/epoch stamps
(same discipline as weight revisions); MLflow dataset logging. **The open-data
boundary as a first-class constraint** (see §5).

**Derivations.** None heavy (this is systems). One boxed argument: why an
immutable content-hashed snapshot makes a task built from it reproducible forever.

**Lab → Artifact.** Stand up containerized Airflow 3; run the DAG to fetch +
normalize + snapshot one epoch of each source; back a snapshot with DVC. Artifact:
a versioned, provenance-stamped SDA data snapshot on the tiers + the DAG.

**Read-along.** [AIE] ch. 8 (dataset engineering).

**Admonitions.** `under-the-hood` (Airflow assets / data-aware scheduling),
`gotcha` (single-node Airflow-3 setup realities; space-track rate limits),
`substack-seed`, `read-along`.

### 3.10 From orbital data to verifiable tasks

**Goal.** Turn snapshots into task instances with machine-checkable ground truth
— the thing that makes "verifiable reward" real for SDA.

**Covers.** Task families, each = prompt + gold answer + verifier + difficulty +
provenance:
- **Conjunction screening (flagship):** propagate two TLEs with SGP4 → time of
  closest approach + miss distance; "do these conjunct within X km in the next
  24h?" Ground truth is deterministic.
- Orbital-element derivation (period, apogee/perigee, inclination, revs/day from a
  TLE).
- Decay/reentry ordering.
- Pass prediction / visibility from a ground site (Skyfield observer geometry).
- Catalog correlation / identification.
The verifier as a **physics oracle**: Skyfield/sgp4 both *generates* the gold
answer and *checks* the model's answer through the same code path, plugged into
the existing `thesis_suite` verify/score API (adds real orbital verifiers next to
the toy ones, then removes the toy ones per the decision to replace them).
Difficulty calibration by solve-rate bands (feeds 7.5); item counts from the 3.7
power analysis; dedup against 3.8 contamination; the output feeds 3.11's freeze.
Numeric tolerance and float-comparison policy for the checker.

**Derivations.** The SGP4/TLE mean-elements → position sketch at reading depth
(enough to trust the oracle); the miss-distance / time-of-closest-approach
formulation; tolerance-band definition for a "correct" numeric answer.

**Lab → Artifact.** Generate a versioned SDA task set (conjunction + element
tasks) from a 3.9 snapshot; show the oracle grading three worked responses.
Artifact: the generated task set + the orbital verifier module.

**Read-along.** [BRM] ch. 3; [AIE] ch. 8.

**Admonitions.** `derivation` (oracle math), `thesis-thread` (the SDA thread
becomes real conjunction tasks), `gotcha` (epoch/frame/units pitfalls in SGP4),
`substack-seed`, `read-along`.

### 3.11 Building the thesis task suite (updated, was 3.9)

Light-touch update only: the suite now freezes from the 3.10 generator; the toy
"set difference" examples are replaced by the real orbital task families; the
`thesis_suite` package note updates to "chapter 3.11". Everything else (freeze,
manifest, datasheet, power-analysis sizing) stays. All downstream references to
"3.9 the suite" become "3.11".

### 8.1 RAG over space text

**Goal.** Retrieve the pipeline's article corpus (spacenews / NASA) to ground
*qualitative / recency* SDA answers, and measure whether it actually helps.

**Covers.** Why RAG is for text and not numbers (you retrieve what you cannot
compute). Chunking; a local embedding model served via vLLM (co-resident memory
budget with the generation model on 16GB); LanceDB index build over a 3.9 article
snapshot; retrieval + prompt assembly; optional cross-encoder rerank. Evaluating
RAG: retrieval metrics (recall@k, MRR) vs end-task metrics; the two are not the
same. **RAG-as-confound:** retrieval can leak the answer (ties to 3.8
contamination) and "is the RAG gain reasoning or lookup?" (sets up 8.3 / Part IV).

**Derivations.** Cosine similarity / nearest-neighbor retrieval; recall@k and MRR;
the embedding-model memory line for the 16GB co-residency.

**Lab → Artifact.** Build the index over a snapshot; run a RAG-vs-parametric eval
on qualitative SDA tasks with `evalstats`. Artifact: the RAG index build + service
+ the RAG-vs-parametric report.

**Read-along.** [AIE] ch. 6 (RAG).

**Admonitions.** `vram-budget` (embedding + generation co-residency), `derivation`
(retrieval metrics), `gotcha` (retrieval leakage as contamination), `substack-seed`,
`read-along`.

### 8.2 MCP tools for live SDA data

**Goal.** Give the model live numerical data (conjunction / catalog / orbital
queries) via MCP tools at inference, with verifiable tool-use.

**Covers.** An MCP server (FastMCP) exposing typed tools that wrap the 3.9 source
clients + Skyfield oracle (`get_tle`, `screen_conjunction`, `catalog_lookup`,
`propagate`); transports (stdio local, streamable-HTTP for the serving host). The
model as MCP client: vLLM OpenAI-compatible tool-calling driving a thin tool-use
loop; and, for the eval, wiring the tools into Inspect (which already owns
agentic/sandboxed tool eval, 3.4). **Verifiable tool-use** (the tool returns
ground truth, so you can check the model both called it and used the result
correctly). Caching + rate-limit/auth discipline (the 3.9 open-data boundary —
tools fetch live, never redistribute — which is *why* live numerical data belongs
behind a tool, not a shipped dataset). When tools beat RAG beat parametric.

**Derivations.** None heavy; a boxed tool-use control-flow trace.

**Lab → Artifact.** Build the MCP server; run a tool-augmented conjunction-
screening eval where the model calls the tool for real orbital data then reasons.
Artifact: the MCP server + the tool-augmented eval.

**Read-along.** [AIE] ch. 6 (agents/tools).

**Admonitions.** `under-the-hood` (building the FastMCP server; the tool-call
loop), `gotcha` (live-data auth/rate-limits; non-determinism from live feeds in an
eval — snapshot-pin for reproducibility), `substack-seed`, `read-along`.

### 8.3 The augmentation arms: what caused the gain

**Goal.** The committee centerpiece. Causally separate how much SDA improvement
comes from parametric RL-training vs retrieval vs tools.

**Covers.** The four arms (base / RL-trained / +RAG / +tools, plus the trained+RAG
and trained+tools combinations) at **matched compute/token budgets** (the 6.6
matched-budget discipline, so an arm doesn't win just by spending more tokens).
Each augmentation as an intervention; RAG-access and tool-access drawn as a DAG
with the confound/mediator roles named, identified with the Part IV backdoor/
front-door machinery. Paired statistics + effect sizes on the shared frozen suite
(3.7); what each arm's delta legitimately claims (e.g. "+tools raised accuracy but
that is tool access, not reasoning"; "the trained arm's gain survives at matched
budget without retrieval, which is the parametric-reasoning claim").

**Derivations.** The matched-budget accounting; the adjusted (backdoor) arm
comparison vs the naive one; paired effect sizes reused from 3.7 / 7.6.

**Lab → Artifact.** The full four-arm eval on the frozen suite, analyzed with
`evalstats` + the causal audit (4.6). Artifact: the arm-comparison report.

**Read-along.** [CAI] Part 3; [RLHF] ch. 7.

**Admonitions.** `thesis-thread` (the strongest form of the claim), `derivation`
(matched budget + adjustment), `substack-seed`, `read-along`.

---

## 4. Cross-cutting updates

- **SUMMARY.md:** new entries + renamed paths + new Part VIII header + renumbered
  Part IX/X headers.
- **Stubs first:** create all five new chapter files (+ any renamed) so
  `create-missing = false` still builds.
- **Cross-references into existing chapters:** 3.4 gains a forward-ref to the 8.2
  agentic tool-use eval; 4.3/4.5 pick up RAG-access and tool-access as named
  confound/mediator examples; 6.6 cross-refs RAG/tools as inference-time
  augmentations; 7.5 sources its curriculum from the 3.10 pipeline; 7.6's delta
  feeds 8.3; the repro package (now 10.3) ships the pipeline + RAG index build +
  MCP server (and documents what is fetch-only).
- **Appendices:**
  - A (VRAM): add the embedding + generation co-residency budget (8.1); note the
    pipeline/MCP are CPU/IO (no GPU line).
  - B (CLI): add `airflow`, `dvc`, `duckdb`, `uv run` task-module invocations, the
    FastMCP server run, and `inspect` tool-eval commands.
  - C (troubleshooting): Airflow-3 single-node gotchas; space-track auth/rate-
    limits; SGP4 epoch/frame/units errors; LanceDB/embedding OOM co-residency.
  - D (glossary): TLE, SGP4, conjunction / CDM, ephemeris, RAG, embedding, vector
    store, reranker, MCP, tool-use, DVC, Parquet, DuckDB, Airflow asset / data-
    aware scheduling.
  - E (reading map): rows for 3.9/3.10 ([AIE] ch. 8), 8.1 ([AIE] ch. 6), 8.2
    ([AIE] ch. 6), 8.3 ([CAI]/[RLHF]); update the renumbered part headers.
  - F (notation): recall@k, MRR, cosine sim; conjunction miss-distance / TCA
    symbols; the tolerance band.

---

## 5. The open-data / licensing / auth boundary (first-class)

| Source | Auth | Rate limit | Redistribute? | In repro package? |
|---|---|---|---|---|
| celestrak | none | courtesy | yes (permissive) | snapshot shipped |
| space-track | account login | strict | **no** (terms restrict) | **fetch-only**, never shipped |
| api.nasa.gov | free key | ~1000/hr | check per-endpoint terms | derived facts only |
| thespacedevs / LL2 + Spaceflight News API | none | courtesy | yes (attribution) | snapshot shipped |

Rule: the public repo and reproducibility package ship only redistributable
snapshots (celestrak, spacedevs) plus *derived* task instances whose gold answers
were computed by the oracle. Space-track data is fetched live at run time and
never committed; this is precisely why the live numerical side belongs behind an
MCP tool rather than a shipped dataset. This boundary is a `gotcha` + a
Thesis-Thread beat, and it is enforced by the repro-package embargo check
(now 10.3).

---

## 6. Reproducibility and artifacts

Every snapshot is content-hashed and DVC-pinned; every task instance carries the
snapshot revision it was built from; every run logs the snapshot hash + git SHA +
uv lock hash to MLflow (0.6 schema). An eval that uses live tools (8.2) pins a
captured snapshot for reproducibility and only hits the live feed in "current"
mode. The four-arm report (8.3) is regenerable from pinned snapshots + the frozen
suite.

---

## 7. Implementation waves (dependency order)

1. **Renumber + stubs** — rename files/dirs, update SUMMARY, sweep number
   references, add the five new stub files. Build stays green. (One PR.)
2. **3.9 pipeline** — Airflow-in-Docker, source clients, snapshotting, DVC.
3. **3.10 verifiable tasks** — the oracle + task families; update 3.11 suite to
   consume it and drop the toy examples.
4. **8.1 RAG** — index + RAG-vs-parametric eval.
5. **8.2 MCP tools** — FastMCP server + tool-augmented eval.
6. **8.3 arms** — the four-arm causal comparison.
7. **Cross-cutting** — appendices, cross-references, reading map, notation.

Each wave is its own PR through the PR-CI gate. Waves 2–6 are hardware-touching
(live APIs run; GPU for 8.1/8.3), so measured numbers stay `(record on the
baseline machine …)` until the Friday hardware is up.

---

## 8. Open questions / decisions still to make during drafting

- Exact embedding model choice (bge vs Qwen3-Embedding) — decide empirically in
  8.1 under a small retrieval-quality bake-off, like the 3.6 judge decision.
- Whether 8.3 includes the trained+RAG+tools "kitchen sink" arm or stops at the
  four principal arms (leaning: include it as a ceiling, flagged as
  non-identifiable for individual attribution).
- Conjunction task realism knobs (screening volume, screening horizon) — set in
  3.10 to hit the difficulty bands.
