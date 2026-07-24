# The SDA data pipeline

Everything the eval side of this book does with Space Domain Awareness assumes a supply of real orbital data: the verifiable conjunction and orbital-element tasks in Chapter 3.10 are generated from element sets, the frozen suite in Chapter 3.11 freezes a slice of them, and the RAG corpus in Chapter 8.1 is the text feeds. None of that is trustworthy if the data underneath it is a folder of CSVs I downloaded once and can no longer explain. Space data is live: TLEs are re-issued as objects are re-observed, conjunction messages arrive continuously, launches and reentries happen on their own schedule. A snapshot I took last Tuesday is not the same as the one I take today, and a thesis claim that cannot say *which* snapshot it was built from is not a claim, it is an anecdote.

So this chapter builds the thing that turns five live feeds into a normalized, provenance-stamped, reproducible store: fetched on a schedule, incrementally, with backfill and retries, every record stamped with where it came from and when, and every snapshot content-hashed and pinned so that a task built from it is reproducible forever. The running constraint, and the reason this is a real engineering chapter and not a `wget` script, is that one of the five sources (space-track) is authenticated, rate-limited, and contractually not redistributable, which forces a hard boundary through the whole design: the public repository ships only the feeds I am allowed to ship, plus derived task instances, and the restricted numerical data is fetched live and never committed. That boundary is what later pushes the live numerical side behind an MCP tool in Chapter 8.2 rather than a shipped dataset.

The stack is deliberately boring and standard: Apache Airflow 3 schedules, **dlt** extracts and lands the public feeds (with the `spacetrack` library for the one authenticated source), plain `uv run` modules freeze each load into an immutable snapshot, Pydantic and pandera gate the shapes, Parquet and DuckDB store and query, and DVC plus MLflow pin and log the provenance. Durable bytes live on the archive tier's MinIO object store (Chapter 0.5), and above the frozen snapshots sits a thin serving layer, Delta Lake tables written with `delta-rs` and SQL transforms organized by `dbt-duckdb`, that exists to be queried, not to be cited. It all lives in the CPU/IO `data/` sub-project, which never touches the GPU.

## Theory

### Five feeds, four data types

The five sources do not map one-to-one onto anything; they map onto four *data types* that I normalize toward, which is what keeps the downstream code from caring which feed a record came from.

- **celestrak** is the public, permissively-licensed backbone: GP/TLE element sets (the orbital state of catalogued objects at an epoch), the SATCAT object catalog, and SOCRATES conjunction reports. No auth, courtesy rate limits, redistributable. This is the feed the conjunction and element tasks are built on, and the one I am allowed to ship.
- **space-track** is the authoritative catalog: full GP history, the official CDMs (conjunction data messages), and decay/reentry data. It needs an account login, enforces strict rate limits, and its terms restrict redistribution. I fetch it live and never commit it.
- **api.nasa.gov** gives NeoWs (near-earth object approaches) and DONKI (space-weather events) behind a free key at roughly 1000 requests/hour. I ship derived facts, not raw dumps, per its per-endpoint terms.
- **thespacedevs** (Launch Library 2) plus the **Spaceflight News API** give launches, events, and news articles: public, courtesy-limited, attribution-required, redistributable.

Those collapse into four normalized tables. **Objects** are catalog entities (a NORAD id, a name, an object type). **Element-sets-at-epoch** are orbital states (the GP/TLE fields at a given epoch). **Events** are things that happen at a time (a launch, a decay, a close approach, a space-weather flare). **Articles** are text with a source and a timestamp. Every source populates one or more of these, and every downstream consumer reads the normalized table, not the raw feed.

### Why an orchestrator, not cron

The instinct on a single machine is a crontab: `0 */6 * * * python fetch.py`. That works until the first thing goes wrong, and with live feeds something always goes wrong. celestrak returns a 403 during a courtesy-limit blip; space-track rate-limits me mid-pull; a NeoWs day comes back empty because the key hit its hourly cap. Cron has no memory of any of this. It does not know a run failed, it will not retry with backoff, it cannot backfill the three days the machine was off, and it keeps no lineage tying "this Parquet file" to "that fetch attempt at that time." I would be reconstructing all of that from log scrollback, which is exactly the failure mode Chapter 0.6 built the tracking spine to kill.

An orchestrator gives me five things cron cannot. **Scheduling** with a real notion of a logical date, so a run "is" a point in time I can reason about. **Incremental pulls** keyed to that date, so I fetch the new window rather than re-pulling the world. **Backfill**, so if the machine was down Friday to Sunday I can materialize those three logical dates deterministically. **Retries with backoff**, so a transient 403 or a rate-limit 429 is handled by policy instead of by me at 2am. And **lineage**, so every snapshot knows which run produced it. Airflow 3 is the orchestrator I use, containerized alongside the MLflow spine from Chapter 0.6.

### The Airflow mechanics this rests on

That list of five is really three Airflow mechanisms, and it is worth naming them precisely because the rest of the chapter leans on them rather than re-deriving them. The first is the **producer/consumer asset model** ([DPA] ch. 4). A task that updates an asset *produces* an asset event; a DAG scheduled on that asset *consumes* those events and runs when they arrive, not when a clock ticks. My `tle_snapshot` and `conjunction_tasks` are a concrete instance of exactly this: the fetch-and-normalize task produces `tle_snapshot` whenever it refreshes the element sets, and the conjunction build consumes that event and rebuilds off it. Airflow also lets me *combine* the two kinds of trigger, which is what the pipeline actually wants. The producer runs on a timer because celestrak has no way to tell me it changed, while every derived asset downstream rebuilds on updates instead of on a second guessed timer that would drift out of step with the pull.

The second is **incremental processing keyed to a data interval** ([DPA] ch. 3). Each DAG run is bound to a slice of time, its data interval, and is expected to process only that slice, which is what turns "pull only what is new" from a vague intention into a precise contract: the run for a given interval fetches that interval's window and nothing else. The mirror image is **backfilling**, re-running the historical intervals to populate the store from the past, which is how three logical dates the machine slept through get materialized deterministically instead of approximated. This is the real substance of "why an orchestrator, not cron". Cron gives me a timer and stops there; data intervals plus backfill plus retries plus lineage are what a live feed actually needs, and they are Airflow features, not things I want to hand-roll around a `crontab` line.

The third is the **thin-DAG mechanics** the next subsection depends on ([DPA] ch. 2): the distinction between tasks and operators, the TaskFlow API that defines a DAG straight from ordinary Python functions, running the whole thing in Docker, and DAG versioning so a run remembers which version of the graph produced it. The TaskFlow API and my standalone `uv run` task modules are two views of one idea, which is to keep the real logic in plain Python you can run without the scheduler. [DPA] arrives at it by lifting functions up into the DAG; I arrive at it by pushing the logic out into modules the DAG shells into. Either way the test is identical, and it is the subject of the next subsection: if the fetching, parsing, and validating only run inside Airflow, you have welded your logic to your orchestrator.

### Thin DAGs: Airflow schedules, `uv run` does the work

The single most important discipline in this chapter is that Airflow does not *do* anything. It schedules. Every piece of real logic (the dlt extract, the Pydantic normalization, the pandera gate, the Parquet write) lives in a plain Python module in the `data/` sub-project that I can run from the command line with `uv run python -m sda_data.tasks.celestrak_tle`, with no Airflow imported anywhere in it. The DAG is a thin shell whose tasks shell out to those modules.

This is not stylistic fussiness; it buys three concrete things. First, **every task is testable and debuggable standalone**: when the celestrak pull breaks, I run the module in a terminal and get a normal Python traceback, not an Airflow task-instance log four clicks deep in a web UI. Second, **the pipeline is not hostage to Airflow's environment**: the task module resolves its own dependencies through the `data/` project's `uv.lock`, so the versions that run in the DAG are byte-identical to the versions that run at my prompt. Third, **the orchestrator is swappable**: if Airflow is ever the wrong tool, the DAG is fifty lines and the thousand lines of real logic do not move. A DAG that reaches into a database, parses JSON, and validates a schema inside a Python task is a DAG you cannot run without Airflow, which means you cannot debug it without Airflow, which means you will debug it slowly.

The same discipline settles the language question. Every task an Airflow DAG invokes is bash or Python, full stop, because those are the languages the whole surrounding stack (dlt, pandera, DuckDB, DVC) already speaks and the ones I can debug at 11pm without a compile step. But notice what the thin-DAG contract actually pins down: a task is a CLI that reads a tier and writes a tier. Nothing in that contract says Python. If a leaf job ever earns a rewrite in Rust (the conjunction screen in 3.10 is the one CPU-bound candidate, and the `sgp4` crate is excellent), it slots behind the identical `BashOperator` line and neither the DAG nor the provenance contract notices. The rule is: reach for Rust when a specific job's profile demands it, never as a default, because the cost of a second toolchain is paid on every debugging session, not just at write time.

```admonish under-the-hood title="Airflow 3 assets and data-aware scheduling"
Airflow 3 renamed Datasets to **Assets** and made them the backbone of data-aware scheduling. An asset is a named, addressable thing a task *produces* (declared as an `outlet`), and a DAG can be *scheduled on* one or more assets instead of (or as well as) a clock. When a task with `outlets=[tle_snapshot]` succeeds, Airflow records an update to that asset, and any DAG whose `schedule=[tle_snapshot]` is immediately eligible to run. The scheduler maintains the asset graph and fires downstream runs off upstream completions, so I never write "after the TLE pull finishes, kick the conjunction build" as an explicit dependency; I declare that the conjunction build consumes the `tle_snapshot` asset and the graph does the wiring. Airflow 3 also exposes an `@asset` decorator that defines an asset-producing DAG in one function, which reads beautifully for simple producers. I use the explicit-`outlets` form in the lab because it keeps the thin-DAG discipline honest: the outlet is metadata on a `BashOperator` that runs my standalone module, so the asset wiring is Airflow's concern and the fetching stays plain Python. The payoff is that the `conjunction_tasks` build rebuilds exactly when the `tle_snapshot` it depends on is refreshed, and never on a guessed timer. This is the producer/consumer model of [DPA] ch. 4 in miniature: the fetch task is the producer, the conjunction build is the consumer, and the recorded asset update is the only coupling between them.
```

### One schema for four data types

Raw feeds are a zoo. celestrak GP JSON uses `MEAN_MOTION` and `NORAD_CAT_ID`; a TLE line-pair encodes the same numbers as fixed-width columns; space-track uses its own casing; the NeoWs payload is a nested dict keyed by date. If every consumer parsed raw feeds, every consumer would carry every feed's quirks, and a change in one feed would ripple everywhere. So I normalize at ingest: each source's task module maps its raw payload onto the common schema (objects, element-sets-at-epoch, events, articles), and everything downstream sees only the schema.

Normalization has two layers, and they do different jobs. **Pydantic** models are the per-record contract: they parse and coerce one raw record into one typed object, reject a record whose fields are missing or the wrong type, and are where the provenance stamps get attached to every record (source, fetch time, epoch, snapshot hash). **pandera** is the per-frame contract: once the records are a dataframe, a pandera schema asserts properties that only make sense across the whole table, like "eccentricity is in [0, 1)", "inclination is in [0, 180] degrees", "NORAD id is unique", "mean motion is a plausible revs-per-day". Pydantic catches the malformed record; pandera catches the record that is individually well-formed but collectively wrong (a duplicate object, an eccentricity of 1.4 that parsed fine as a float but is physically impossible). A record has to pass both to reach the Parquet tier.

### Immutable snapshots and the content address

The storage model is: raw bytes are written once and never mutated, then normalized into Parquet. The raw tier is the source of truth; the Parquet tier is a derived view I can always rebuild from the raw bytes. And the identity of a raw snapshot is not its filename or its fetch date, it is a **content address**: the cryptographic hash of its bytes.

$$
\mathrm{addr}(s) = \texttt{sha256}\big(\mathrm{bytes}(s)\big) \tag{9.1}
$$

I name the raw snapshot file by its own hash, so equation (9.1) is literally the filename. This one move is what makes the whole pipeline reproducible, and it is worth stating the argument precisely rather than gesturing at it.

```admonish derivation title="Why a content-hashed snapshot makes a task reproducible forever"
Take a task instance $t$ built from snapshot $s$ (for example, "propagate these two TLEs and report whether they conjunct within 5 km in the next 24 hours", whose gold answer the Chapter 3.10 oracle computes from the element sets in $s$). I want the property that re-running the generator for $t$ at any point in the future yields the identical gold answer.

The generator is a deterministic function $g$ of the snapshot bytes: $\text{gold}(t) = g(\mathrm{bytes}(s))$ (the oracle is pure numerics, no clock, no network). The only way $\text{gold}(t)$ can change is if $\mathrm{bytes}(s)$ changes. So I bind $t$ to the *content address* of $s$, not its name: the task instance carries $\mathrm{addr}(s)$ from equation (9.1). To reproduce $t$ I resolve $\mathrm{addr}(s)$ back to bytes and check that $\texttt{sha256}$ of those bytes equals the recorded address. By the collision resistance of SHA-256, a match means the bytes are the ones the gold answer was computed from, and $g$ being deterministic then guarantees the same gold answer. If someone re-issues the TLE upstream tomorrow, that is a *different* snapshot $s'$ with a *different* address, and it cannot silently substitute for $s$: the task still points at $\mathrm{addr}(s)$, which still resolves to the original bytes. Naming by content, not by "latest", is precisely what removes the ambiguity that "the TLE for object 25544" would otherwise carry. Immutability plus content addressing turns "the data I used" from a hope into a checkable fact, which is the same discipline the model-revision pinning of Chapter 0.5 and the suite freeze of Chapter 3.11 apply to their own artifacts.
```

DVC is how I make that content address portable and versioned. `dvc add` on a raw snapshot records its hash in a small `.dvc` pointer that lives in git, while the bytes live on the storage tier (and optionally a remote). Pinning is then "by revision, not by name": a task references the snapshot through the git commit that contains its `.dvc` pointer, so checking out that commit resolves exactly those bytes. MLflow closes the loop by logging the snapshot as a dataset input on every run that consumes it, so the tracking spine records "this run used snapshot `sha256:...`" next to the git SHA and lock hash from Chapter 0.6.

### The open-data boundary is a design constraint, not a footnote

The redistribution boundary is not a legal caveat I bolt on at the end; it shapes the storage layout. There are three tiers of committability, and each source is assigned to one:

- **Shippable snapshots.** celestrak and thespacedevs are redistributable (permissive / attribution). Their raw snapshots go on the DVC-tracked raw tier and ride along in the reproducibility package.
- **Derived facts only.** api.nasa.gov data is reduced to the specific derived facts a task needs, per its per-endpoint terms, rather than shipped as a raw dump.
- **Fetch-only, never committed.** space-track data (the authoritative catalog, CDMs, decay) is written to an embargoed live tier that is git-ignored and never DVC-tracked. It is fetched at run time and used in memory; it never enters a committed artifact.

The rule the repository enforces (checked by the repro-package embargo test in Chapter 10.3) is: the public repo and repro package ship only redistributable snapshots plus *derived* task instances whose gold answers the oracle computed. This is exactly why the live numerical side belongs behind an MCP tool in Chapter 8.2: a tool can fetch space-track live at inference and return a ground-truth answer without ever shipping the underlying restricted data, which a static shipped dataset could not do without violating the terms.

```admonish gotcha title="space-track auth, strict rate limits, and the redistribution line"
space-track is the source most likely to burn you, in three ways. First, **auth**: it is a session login, not an API key, so a client has to log in and hold a cookie. I use the `spacetrack` library precisely so I do not hand-roll the login and session handling. Second, **rate limits**: they are strict and enforced (documented limits on requests per minute and per hour, with throttling and temporary blocks for abuse). Hammer it and you get locked out, which on a scheduled pipeline means a whole logical date fails. The `spacetrack` library throttles for you if you let it; do not disable that. Batch queries (ask for many objects in one request) instead of looping one object per request. Third, and most important, **the redistribution line is a code invariant, not a promise**: space-track output must land only on the git-ignored, non-DVC live tier. If you ever find a space-track file staged for commit, that is a bug, and the Chapter 10.3 embargo check exists to fail the build when it happens. Never widen a `git add` glob to "just grab the data folder".
```

## Tooling

The stack is locked, and each piece earns its place. **Apache Airflow 3** is the orchestrator, run in Docker Compose as an extension of the Chapter 0.6 tracking spine (same compose project, same network, so tasks can log to MLflow at `http://mlflow:5000`). It contributes only scheduling, assets/data-aware triggering, retries, and backfill; the DAGs are thin. **dlt** (data load tool) is the ingest framework for the three public feeds (celestrak, api.nasa.gov, thespacedevs): it owns the plumbing that hand-rolled HTTP does not (pagination, transport retries, incremental cursors keyed to a date, and JSON-to-relational normalization), and lands each run to a Parquet staging area through its filesystem destination. Crucially, dlt *stops at the landing zone*: the immutable-snapshot, content-hash, and DVC steps below take a dlt load and freeze it, so the provenance contract does not depend on the extractor. **The `spacetrack` library** wraps space-track's login, session, and rate-limit handling so I never touch raw auth; dlt has no verified source for a session-authenticated, embargoed feed, and that data wants special handling anyway. **Pydantic** models are the per-record ingest contract and **pandera** schemas are the per-frame quality gate. These divide cleanly from dlt's own schema layer: dlt's *schema contracts* govern table **shape** (freeze, evolve, or discard on a new column or type), while pandera asserts the **physics** dlt cannot express (eccentricity in [0, 1), inclination in [0, 180], a plausible mean motion), so the two are complementary rather than redundant. **Parquet** is the normalized storage format (columnar, typed, compresses well), written to the storage tiers and queried in place by **DuckDB**, which reads Parquet directly with zero load step. **DVC** versions the raw snapshots by content hash; **MLflow** logs each snapshot as a dataset input so a run's data provenance sits beside its code and dependency provenance. All of it is CPU/IO work that lives in the `data/` uv sub-project, with its own `pyproject.toml` and committed `uv.lock`, and it never asks for the GPU.

```admonish thesis-thread title="Why dlt, and why it stops at the landing zone"
The honest comparison is not "dlt versus an HTTP client"; they sit at different layers. A raw client does the GET and nothing else, so choosing it means hand-rolling pagination, retry policy, incremental state, and JSON-to-relational normalization myself, for every feed. dlt is the framework that owns exactly that undifferentiated plumbing, and for the paginated, parameterized public APIs (NeoWs, the Spaceflight-News feed) it also ships verified/generated sources, so a new feed is a small resource function rather than a bespoke crawler. That is a real reduction in code I would otherwise write and maintain, which is why the public feeds go through dlt.

The one place dlt's grain runs against this thesis is versioning. dlt's native mode is *incremental merge into a mutating destination*; the spine here is the opposite, an *immutable, content-addressed snapshot pinned by DVC* (equation 9.1), because a task instance in Chapter 3.10 must bind to exact bytes to be reproducible forever. I do not resolve that by giving up the snapshot contract. I resolve it by drawing a boundary: dlt owns **extract to land** (fetch, paginate, normalize, write a Parquet partition), and stops. Each run lands a fresh, run-stamped partition rather than merging in place, so dlt's incremental state never competes with DVC as the versioning authority, exactly the discipline the LanceDB index in 8.1 follows to keep Lance's internal versioning from shadowing DVC. Then the freeze layer takes over unchanged: content-hash the landed bytes, register the immutable raw snapshot, gate the frame with pandera, write the normalized Parquet, DVC-pin, MLflow-log. The payoff is that the artifact the rest of the book depends on, a snapshot I can name by hash and reproduce, is identical whether the extractor was dlt, the `spacetrack` library, or a bare client. The extractor is swappable; the provenance contract is not. celestrak is the one feed where the thin path was never the weak link (a single unauthenticated GET of the whole catalog, no pagination, no cursor), so its module below stays deliberately hand-written as the minimal baseline; the dlt path is shown on NeoWs, where the plumbing it removes is real.
```

```admonish under-the-hood title="The serving layer: Delta tables and dbt above the frozen snapshots"
The content-hashed snapshot tier is built to be *cited*; it is deliberately bad at being *queried* over time, because every refresh is a new immutable file and "the history of object 25544's elements" is scattered across hundreds of them. So above the freeze sits a serving layer with exactly one job: make the accumulated data pleasant to query. Each refresh is appended to a **Delta Lake** table on MinIO, written with `delta-rs` (no Spark, a plain Python writer in the same task module), which buys ACID appends, schema evolution, and a compacted table DuckDB reads directly through its `delta` extension. **dbt-duckdb** then organizes the SQL that turns appended bronze into queryable gold (latest-elements-per-object, close-approach summaries, article rollups) as tested, documented models rather than a folder of loose `.sql` files; DuckDB's delta extension is read-only, so dbt owns the SQL transforms and the thin Python tasks own the Delta writes.

Two invariants keep this layer honest. First, **one writer per table**: every Delta table is owned by exactly one DAG task with `max_active_runs=1`, because `delta-rs` on MinIO has no distributed lock, and a backfill racing a scheduled run is the standard way to corrupt a transaction log. Second, **the serving layer is never the provenance authority**: Delta time travel versions the table, but anything a thesis claim rests on binds to the content address of equation (9.1) and its DVC pin, full stop. The Delta tables are rebuildable from the frozen snapshots by replay, which means `VACUUM` retention, compaction, and even dropping a table are operational choices, not reproducibility events. The moment those two authorities blur (a task instance citing a Delta version, a snapshot existing only inside a Delta table), the whole reproducibility argument of this chapter quietly breaks.
```

```admonish gotcha title="Single-node Airflow 3 realities"
Airflow is designed for clusters, and the defaults assume one. On a single research machine you want the un-fancy configuration and should not fight it. Use `LocalExecutor` (not Celery/Kubernetes, which want a broker and workers you do not have) so tasks run as subprocesses on the one box. Airflow 3 splits the deployment into standalone components: an **API server**, a **scheduler**, and, critically, a **DAG processor**, because in Airflow 3 the scheduler no longer parses DAG files itself. Omit the `dag-processor` service and the stack comes up green while the DAGs list stays empty forever, which is a maddening failure precisely because nothing errors. A **triggerer** service completes the set if you use deferrable operators or event-driven scheduling; it is cheap, so run it. In Compose that is four services plus a metadata database, and you run the `airflow-init` step once to migrate that database before anything else will start. User management changed in Airflow 3: `airflow users create` is gone from core, replaced by the auth-manager interface whose default, the **SimpleAuthManager**, has no CLI at all. You declare users in config instead, `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS` as `username:role` pairs (roles: viewer, user, op, admin), and each password is generated on first boot into `$AIRFLOW_HOME/simple_auth_manager_passwords.json.generated`, editable in place if you want a memorable one. Keep the init container honest with `set -euo pipefail`: the old `airflow users create ... || true` trap goes green on a failed bootstrap and only surfaces later when the UI won't log you in, which is the worst possible default for a bootstrap step. Keep Airflow's metadata database (its own Postgres) entirely separate from your data tiers: it stores DAG runs and task states, not your snapshots, and conflating the two is how people accidentally put orbital data somewhere it does not belong. Turn `catchup` on deliberately, per-DAG, only where backfill is meaningful, or a first `unpause` will stampede a run for every logical date since `start_date`. And mount your `data/` project into the scheduler and worker containers so the `uv run` task modules and their `uv.lock` are visible; the DAG shells out to them, so they have to be on the path the tasks execute in.
```

## Lab

I stand up Airflow 3 next to the MLflow spine, write the standalone task module for the celestrak TLE pull (real httpx, Pydantic, pandera), snapshot it immutably and content-address it, query it with DuckDB, back it with DVC, and wire the thin DAG plus the data-aware downstream asset. The artifact is a versioned, provenance-stamped SDA snapshot on the storage tiers, plus the DAG that produces it.

### The `data/` sub-project

```bash title="shell: create the data sub-project"
uv init data && cd data
uv add "dlt[filesystem,parquet]>=1.0" "httpx>=0.27" "pydantic>=2.7" "pandera>=0.20" \
       "pandas>=2.2" "pyarrow>=16" "duckdb>=1.0" "spacetrack>=1.3" "mlflow>=2.14"
uv add --dev "dvc>=3.50"
# dlt extracts+lands the public feeds; httpx stays for celestrak's single-GET path.
# pandera >= 0.20 splits the pandas backend into its own import path.
```

### Config: the three storage tiers

```python title="data/src/sda_data/config.py"
"""Storage tiers. The live tier is the redistribution boundary, in code."""
from __future__ import annotations

import os
from pathlib import Path

DATA_HOME = Path(os.environ.get("SDA_DATA_HOME", Path(__file__).resolve().parents[3]))

RAW_ROOT = DATA_HOME / "data" / "raw"        # immutable raw bytes, DVC-tracked, shippable
SNAP_ROOT = DATA_HOME / "data" / "snapshots" # normalized Parquet, derived from RAW_ROOT
LIVE_ROOT = DATA_HOME / "data" / "live"      # EMBARGOED: space-track etc. git-ignored, never DVC

# Sources allowed onto the shippable raw tier. space-track is deliberately absent.
REDISTRIBUTABLE = {"celestrak", "spacedevs"}
```

### The snapshot helper: content addressing (equation 9.1)

```python title="data/src/sda_data/snapshot.py"
"""Immutable, content-addressed raw snapshots. The filename IS the hash."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from sda_data.config import RAW_ROOT, LIVE_ROOT, REDISTRIBUTABLE


def content_address(raw: bytes) -> str:
    """addr(s) = sha256(bytes(s)); see equation (9.1)."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def write_raw_snapshot(source: str, kind: str, raw: bytes, fetch_time: datetime) -> tuple[Path, str]:
    """Write bytes ONCE, named by their own hash. Restricted sources go to the
    embargoed live tier and never touch the shippable, DVC-tracked raw tier."""
    addr = content_address(raw)
    digest = addr.split(":", 1)[1]
    root = RAW_ROOT if source in REDISTRIBUTABLE else LIVE_ROOT
    out = root / source / kind / fetch_time.strftime("%Y-%m-%d") / f"{digest}.raw"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():           # identical bytes hash identically: write-once is free
        out.write_bytes(raw)
    return out, addr
```

### The common schema: Pydantic ingest models

```python title="data/src/sda_data/models.py"
"""Per-record ingest contract. Provenance is stamped on EVERY record."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str            # "celestrak"
    fetch_time: datetime   # when we pulled the raw snapshot
    snapshot_hash: str     # content address of the raw bytes, equation (9.1)


class ElementSetAtEpoch(BaseModel):
    """One GP/TLE orbital state at its epoch, normalized across feeds."""
    norad_cat_id: int = Field(gt=0)
    object_name: str
    epoch: datetime                       # the element-set epoch (NOT fetch_time)
    mean_motion: float = Field(gt=0)      # revs/day
    eccentricity: float = Field(ge=0, lt=1)
    inclination: float = Field(ge=0, le=180)     # degrees
    ra_of_asc_node: float = Field(ge=0, le=360)
    arg_of_pericenter: float = Field(ge=0, le=360)
    mean_anomaly: float = Field(ge=0, le=360)
    bstar: float | None = None
    prov: Provenance


def from_celestrak_gp(rec: dict, prov: Provenance) -> ElementSetAtEpoch:
    """Map one CelesTrak GP JSON record onto the common schema."""
    return ElementSetAtEpoch(
        norad_cat_id=int(rec["NORAD_CAT_ID"]),
        object_name=rec["OBJECT_NAME"],
        epoch=datetime.fromisoformat(rec["EPOCH"]),
        mean_motion=float(rec["MEAN_MOTION"]),
        eccentricity=float(rec["ECCENTRICITY"]),
        inclination=float(rec["INCLINATION"]),
        ra_of_asc_node=float(rec["RA_OF_ASC_NODE"]),
        arg_of_pericenter=float(rec["ARG_OF_PERICENTER"]),
        mean_anomaly=float(rec["MEAN_ANOMALY"]),
        bstar=float(rec["BSTAR"]) if rec.get("BSTAR") is not None else None,
        prov=prov,
    )
```

### The pandera quality gate

```python title="data/src/sda_data/gates.py"
"""Per-frame quality gate. Catches what individually-valid records hide."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class ElementSetFrame(pa.DataFrameModel):
    norad_cat_id: Series[int] = pa.Field(gt=0, unique=True)   # no dup objects in one snapshot
    object_name: Series[str] = pa.Field(nullable=False)
    mean_motion: Series[float] = pa.Field(gt=0, le=20)        # revs/day; >~16 is LEO, >20 is garbage
    eccentricity: Series[float] = pa.Field(ge=0, lt=1)        # bound orbits only
    inclination: Series[float] = pa.Field(ge=0, le=180)
    source: Series[str] = pa.Field(isin=["celestrak", "spacedevs"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False   # extra provenance columns are allowed
        coerce = True


class NeoWsFrame(pa.DataFrameModel):
    """Per-frame gate for NASA NeoWs close approaches. dlt lands the shape;
    pandera asserts the physics dlt's schema contract cannot."""
    neo_reference_id: Series[str] = pa.Field(nullable=False)
    close_approach_date: Series[str] = pa.Field(nullable=False)
    miss_distance_km: Series[float] = pa.Field(gt=0)              # positive by definition
    rel_velocity_kms: Series[float] = pa.Field(ge=0, le=100)      # sane approach speed
    absolute_magnitude_h: Series[float] = pa.Field(ge=-5, le=40)  # asteroid H range
    source: Series[str] = pa.Field(isin=["nasa"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False
        coerce = True
```

### The standalone task module (the real work)

This module imports no Airflow. It fetches, normalizes, gates, snapshots, and writes Parquet, and it runs from the CLI. The DAG will just call it.

```python title="data/src/sda_data/tasks/celestrak_tle.py"
"""Fetch one epoch of CelesTrak GP element sets, normalize, snapshot, Parquet.

Standalone:
    uv run python -m sda_data.tasks.celestrak_tle --group active
Airflow only schedules this; every line of logic is here, so it debugs from a
plain terminal with a normal traceback.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import httpx
import pandas as pd

from sda_data.config import SNAP_ROOT
from sda_data.gates import ElementSetFrame
from sda_data.models import Provenance, from_celestrak_gp
from sda_data.snapshot import write_raw_snapshot

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"


def fetch_gp(group: str, timeout: float = 30.0) -> bytes:
    """Pull raw GP JSON bytes. Retries/backoff live in the client, not the DAG."""
    transport = httpx.HTTPTransport(retries=3)   # transient 5xx/connection retries
    headers = {"User-Agent": "sda-thesis-pipeline/0.1 (+research)"}
    with httpx.Client(timeout=timeout, transport=transport, headers=headers) as client:
        r = client.get(CELESTRAK_GP, params={"GROUP": group, "FORMAT": "json"})
        r.raise_for_status()
        return r.content


def run(group: str) -> str:
    fetch_time = datetime.now(timezone.utc)

    # 1. Fetch raw bytes and snapshot them IMMUTABLY, content-addressed.
    raw = fetch_gp(group)
    raw_path, snapshot_hash = write_raw_snapshot("celestrak", f"gp-{group}", raw, fetch_time)
    prov = Provenance(source="celestrak", fetch_time=fetch_time, snapshot_hash=snapshot_hash)

    # 2. Normalize each record through Pydantic (per-record contract).
    records = json.loads(raw)
    rows = [from_celestrak_gp(rec, prov).model_dump() for rec in records]

    # 3. Flatten provenance into columns and gate the FRAME with pandera.
    df = pd.json_normalize(rows)
    df = df.rename(columns={"prov.source": "source",
                            "prov.fetch_time": "fetch_time",
                            "prov.snapshot_hash": "snapshot_hash"}).drop(
        columns=[c for c in df.columns if c.startswith("prov.")], errors="ignore")
    df = ElementSetFrame.validate(df)   # raises on any per-frame violation

    # 4. Write the normalized Parquet view, partitioned by fetch date.
    digest = snapshot_hash.split(":", 1)[1][:16]
    out = SNAP_ROOT / "celestrak" / "element_sets" / fetch_time.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / f"{digest}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    print(f"celestrak gp-{group}: {len(df)} element sets  raw={raw_path}  "
          f"snapshot={snapshot_hash[:19]}...  parquet={parquet_path}")
    return snapshot_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="active", help="CelesTrak GROUP, e.g. active, stations")
    args = ap.parse_args()
    run(args.group)


if __name__ == "__main__":
    main()
```

### A dlt source for the paginated feeds (NASA NeoWs)

celestrak was a single GET, which is the one case where hand-rolled transport was never the weak link. NASA's NeoWs is the opposite: paginated, key-gated, rate-limited, and keyed by date, which is exactly the plumbing dlt is built to own. The module below is the *same shape* as the celestrak one, a standalone `uv run` module with no Airflow, but its fetch-and-normalize half is a dlt resource that lands a run-stamped Parquet partition, and its second half is the identical freeze layer: content-address the landed bytes, gate the frame with pandera, write the normalized snapshot. dlt changed how the bytes arrive; it did not touch how they become a pinned snapshot.

```python title="data/src/sda_data/tasks/nasa_neows.py"
"""Fetch NASA NeoWs close approaches with dlt, land to Parquet, then FREEZE.

dlt owns extraction (pagination, retries, incremental-by-date, JSON->tables) and
lands a per-run Parquet partition through its filesystem destination. It stops
there. The freeze half is the SAME content-hash + pandera + snapshot discipline
the celestrak module uses, so the immutable, content-addressed snapshot the tasks
bind to does not depend on the extractor.

Standalone:
    NASA_API_KEY=... uv run python -m sda_data.tasks.nasa_neows --start 2026-07-01 --end 2026-07-02
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import dlt
import pandas as pd
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.paginators import JSONLinkPaginator

from sda_data.config import SNAP_ROOT
from sda_data.gates import NeoWsFrame
from sda_data.snapshot import write_raw_snapshot

NEOWS = "https://api.nasa.gov/neo/rest/v1"


@dlt.resource(name="neo_close_approaches", write_disposition="replace")
def neows_feed(start: str, end: str):
    """One dlt resource -> one normalized table. dlt follows the paginator and
    retries transient failures; I only yield flat records. Swap this hand-written
    resource for dlt's verified NASA source if you prefer -- the freeze half below
    is unchanged either way."""
    client = RESTClient(base_url=NEOWS,
                        paginator=JSONLinkPaginator(next_url_path="links.next"))
    params = {"start_date": start, "end_date": end,
              "api_key": os.environ["NASA_API_KEY"]}
    for page in client.paginate("/feed", params=params):
        for _day, objects in page["near_earth_objects"].items():
            for o in objects:
                for ca in o["close_approach_data"]:
                    yield {
                        "neo_reference_id": o["neo_reference_id"],
                        "name": o["name"],
                        "close_approach_date": ca["close_approach_date"],
                        "miss_distance_km": float(ca["miss_distance"]["kilometers"]),
                        "rel_velocity_kms": float(
                            ca["relative_velocity"]["kilometers_per_second"]),
                        "absolute_magnitude_h": float(o["absolute_magnitude_h"]),
                    }


def extract_to_landing(start: str, end: str) -> Path:
    """dlt owns this half. write_disposition='replace' into a fresh, run-stamped
    dataset = an immutable landing partition, NOT an in-place merge, so dlt's
    incremental state never competes with DVC as the versioning authority."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    landing = SNAP_ROOT / "_landing" / "nasa_neows" / run_id
    pipe = dlt.pipeline(
        pipeline_name="nasa_neows",
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="neows",
    )
    pipe.run(neows_feed(start, end), loader_file_format="parquet")
    return landing


def freeze(landing: Path) -> str:
    """The freeze layer, identical in spirit to celestrak: hash the landed bytes,
    register the immutable raw snapshot, gate the physics, write the silver Parquet.
    Raw NeoWs bytes are NOT redistributable as a dump, so write_raw_snapshot lands
    them on the embargoed tier (nasa is absent from REDISTRIBUTABLE); the derived
    facts here are what ships."""
    fetch_time = datetime.now(timezone.utc)
    df = pd.concat([pd.read_parquet(f) for f in sorted(landing.rglob("*.parquet"))],
                   ignore_index=True)
    raw = df.to_parquet(index=False)                     # canonical bytes to hash
    _raw_path, snapshot_hash = write_raw_snapshot("nasa", "neows", raw, fetch_time)
    df["source"], df["snapshot_hash"] = "nasa", snapshot_hash
    df = NeoWsFrame.validate(df)                          # physics gate (pandera)
    out = SNAP_ROOT / "nasa" / "neows" / fetch_time.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / f"{snapshot_hash.split(':', 1)[1][:16]}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"nasa neows: {len(df)} approaches  snapshot={snapshot_hash[:19]}...  "
          f"parquet={parquet_path}")
    return snapshot_hash


def run(start: str, end: str) -> str:
    return freeze(extract_to_landing(start, end))         # extract (dlt) -> freeze (ours)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC date, e.g. 2026-07-01")
    ap.add_argument("--end", required=True, help="UTC date, inclusive")
    args = ap.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
```

The `--start`/`--end` window is the Airflow data interval made concrete: the consumer DAG passes the run's logical date, so an incremental pull fetches exactly that day and a backfill re-materializes past days deterministically. The Spaceflight-News feed is the same pattern (a dlt resource, incremental by publish date, frozen the same way) and feeds the Chapter 8.1 RAG corpus rather than the numeric tables.

### Querying the snapshot with DuckDB

DuckDB reads the Parquet in place, so there is no load step. This is also how the Chapter 3.10 task generator will pull element sets out of a pinned snapshot.

```python title="data/src/sda_data/query.py"
"""Query the normalized Parquet tier with DuckDB. No load step, no server."""
from __future__ import annotations

import duckdb

from sda_data.config import SNAP_ROOT


def low_earth_orbit_objects(min_mean_motion: float = 11.25) -> "duckdb.DuckDBPyRelation":
    glob = str(SNAP_ROOT / "celestrak" / "element_sets" / "**" / "*.parquet")
    con = duckdb.connect()
    # DuckDB reads the Parquet snapshots directly; provenance columns come along.
    return con.sql(f"""
        SELECT norad_cat_id, object_name, mean_motion, inclination,
               epoch, snapshot_hash
        FROM read_parquet('{glob}', hive_partitioning = false)
        WHERE mean_motion >= {min_mean_motion}   -- ~LEO: period under ~128 min
        ORDER BY mean_motion DESC
    """)


if __name__ == "__main__":
    rel = low_earth_orbit_objects()
    print(rel.limit(10).df())
    print("rows:", rel.aggregate("count(*) AS n").fetchone()[0])
```

### Backing the snapshot with DVC and logging it to MLflow

```bash title="shell: pin the raw snapshot by content, not by name"
cd data
uv run dvc init --subdir            # first time only, inside the data sub-project
# Track the immutable raw snapshot. DVC records its hash in a small .dvc pointer.
uv run dvc add data/raw/celestrak
git add data/raw/celestrak.dvc data/.gitignore
git commit -m "snapshot: celestrak gp-active $(date -u +%FT%TZ)"
# The bytes are now pinned by content: this git commit resolves EXACTLY these
# bytes. A task references the snapshot by this revision, never by 'latest'.
```

```python title="data/src/sda_data/log_dataset.py"
"""Log a snapshot as an MLflow dataset input, so data provenance sits beside
the git SHA and uv lock hash from Chapter 0.6."""
from __future__ import annotations

import mlflow
import pandas as pd

from sda_data.config import SNAP_ROOT


def log_snapshot(run_ctx, source: str, snapshot_hash: str, parquet_path: str) -> None:
    df = pd.read_parquet(parquet_path)
    dataset = mlflow.data.from_pandas(
        df,
        source=parquet_path,
        name=f"{source}-element-sets",
        digest=snapshot_hash.split(":", 1)[1][:16],   # tie the MLflow digest to addr(s)
    )
    mlflow.log_input(dataset, context="sda-snapshot")
    mlflow.set_tag("snapshot_hash", snapshot_hash)
```

### The thin DAG and the data-aware downstream asset

Now the orchestration, and notice how little of it there is. The producer DAG's tasks are `BashOperator`s that run the standalone modules; the `tle_snapshot` asset is declared as an `outlet`. The consumer DAG carries no clock at all: it is scheduled on the asset, so it rebuilds exactly when the TLE snapshot refreshes.

```python title="data/dags/sda_ingest.py"
"""Thin producer DAG: schedule the standalone task modules, declare assets.
No fetching, parsing, or validating happens HERE; it happens in sda_data.tasks."""
from __future__ import annotations

import pendulum
from airflow.sdk import DAG, Asset
from airflow.providers.standard.operators.bash import BashOperator

# Assets are named, addressable things a task produces.
TLE_SNAPSHOT = Asset("sda://celestrak/element_sets")
ARTICLE_SNAPSHOT = Asset("sda://spacedevs/articles")
NEOWS_SNAPSHOT = Asset("sda://nasa/neows")

RUN = "cd $SDA_DATA_HOME/data && uv run python -m sda_data.tasks.{mod}"

with DAG(
    dag_id="sda_ingest",
    schedule="0 */6 * * *",                 # every 6h; a clock drives the PRODUCER
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,                          # opt into backfill deliberately, not by default
    max_active_runs=1,                      # single node: never overlap runs
    default_args={"retries": 3, "retry_delay": pendulum.duration(minutes=5)},
    tags=["sda", "ingest"],
) as dag:
    fetch_tle = BashOperator(
        task_id="celestrak_tle",
        bash_command=RUN.format(mod="celestrak_tle --group active"),
        outlets=[TLE_SNAPSHOT],             # success => TLE_SNAPSHOT is 'updated'
    )
    fetch_articles = BashOperator(
        task_id="spacedevs_articles",
        bash_command=RUN.format(mod="spacedevs_articles"),   # dlt resource, frozen
        outlets=[ARTICLE_SNAPSHOT],
    )
    fetch_neows = BashOperator(
        task_id="nasa_neows",
        # dlt extracts + lands; the module's freeze step content-addresses it.
        # {{ ds }} is the run's logical date: incremental by data interval.
        bash_command=("cd $SDA_DATA_HOME/data && uv run python -m "
                      "sda_data.tasks.nasa_neows --start {{ ds }} --end {{ ds }}"),
        outlets=[NEOWS_SNAPSHOT],
    )
    # space-track runs here too, writing ONLY to the embargoed live tier.
    fetch_cdms = BashOperator(
        task_id="spacetrack_cdms",
        bash_command=RUN.format(mod="spacetrack_cdms"),
        # no outlet asset: its data is fetch-only and never shipped.
    )
```

```python title="data/dags/sda_conjunction_tasks.py"
"""Data-aware consumer DAG: rebuilds when the TLE snapshot updates. No clock."""
from __future__ import annotations

from airflow.sdk import DAG, Asset
from airflow.providers.standard.operators.bash import BashOperator

TLE_SNAPSHOT = Asset("sda://celestrak/element_sets")

with DAG(
    dag_id="sda_conjunction_tasks",
    schedule=[TLE_SNAPSHOT],                # <- fires on the asset, not a timer
    catchup=False,
    max_active_runs=1,
    tags=["sda", "tasks"],
) as dag:
    # The Chapter 3.10 generator: propagate the fresh element sets into
    # conjunction task instances, each stamped with the snapshot hash.
    build = BashOperator(
        task_id="build_conjunction_tasks",
        bash_command="cd $SDA_DATA_HOME/data && "
                     "uv run python -m sda_data.tasks.build_conjunction_tasks",
    )
```

### Airflow beside the MLflow spine

The compose file extends the Chapter 0.6 tracking spine: same project, shared network, `data/` mounted into the scheduler so the `uv run` modules and their lock are on the execution path.

```yaml title="data/docker-compose.airflow.yml (the artifact, extends the 0.6 spine)"
# Single-node Airflow 3, LocalExecutor, next to the MLflow service from 0.6.
x-airflow-common: &airflow-common
  image: apache/airflow:3.0.0
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-db/airflow
    # Airflow 3 manages users through the auth manager, not `airflow users create`.
    # The default SimpleAuthManager reads users from config as username:role pairs
    # (roles: viewer, user, op, admin). No password here: it is generated on first
    # boot and written to simple_auth_manager_passwords.json.generated (see below).
    AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS: "admin:admin"
    MLFLOW_TRACKING_URI: http://mlflow:5000     # the 0.6 spine, same compose network
    SDA_DATA_HOME: /opt/project
  volumes:
    - ./dags:/opt/airflow/dags                  # thin DAGs
    - ..:/opt/project                           # the data/ sub-project + uv.lock
  depends_on: [airflow-db]

services:
  airflow-db:
    image: postgres:16                          # Airflow METADATA only, not our data
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - airflow-db-data:/var/lib/postgresql/data

  airflow-init:                                 # run ONCE: migrate the metadata DB
    <<: *airflow-common
    entrypoint: /bin/bash
    # `set -euo pipefail`, not `|| true`: an init container that goes green on a
    # failed bootstrap hides the breakage until the UI won't authenticate. The
    # admin user is declared in config above, so migrate is all this step does.
    command: -c "set -euo pipefail; airflow db migrate"

  airflow-apiserver:                            # Airflow 3 splits the API server out
    <<: *airflow-common
    command: api-server
    ports: ["127.0.0.1:8080:8080"]              # localhost only
    restart: unless-stopped

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    restart: unless-stopped

  airflow-dag-processor:                        # REQUIRED in Airflow 3: the
    <<: *airflow-common                         # scheduler no longer parses DAGs.
    command: dag-processor                      # Without this service the UI is
    restart: unless-stopped                     # green and the DAG list is empty.

  airflow-triggerer:                            # deferrable operators + event-
    <<: *airflow-common                         # driven scheduling live here
    command: triggerer
    restart: unless-stopped

volumes:
  airflow-db-data:
```

```bash title="shell: bring it up"
cd data
docker compose -f docker-compose.airflow.yml up airflow-init   # once: DB migrate
docker compose -f docker-compose.airflow.yml up -d   # api-server + scheduler + dag-processor + triggerer
# UI at http://127.0.0.1:8080. Log in as `admin`; SimpleAuthManager generated the
# password on first boot. Read it from the passwords file (edit it in place if you
# want one you can remember; generation only fills in missing entries):
docker compose -f docker-compose.airflow.yml exec airflow-apiserver \
  cat /opt/airflow/simple_auth_manager_passwords.json.generated
# Then unpause sda_ingest to schedule it.
```

**What you should see.** Running the task module directly (`uv run python -m sda_data.tasks.celestrak_tle --group active`) prints one line naming the element-set count, the immutable raw snapshot path, the `sha256:` content address, and the Parquet path (row count measured on the baseline machine, record value, date, driver). On disk you now have an immutable raw snapshot on the shippable `data/raw/celestrak/...` tier named by its own hash, and a normalized `data/snapshots/celestrak/element_sets/...` Parquet derived from it. `uv run python -m sda_data.query` returns the LEO objects straight out of Parquet through DuckDB with their `snapshot_hash` provenance column intact. `dvc add` produces a `celestrak.dvc` pointer that pins those bytes by content into a git commit, so the snapshot is reproducible by revision rather than by name. In the Airflow UI, `sda_ingest` runs on its 6-hour clock, and the moment its `celestrak_tle` task succeeds and marks the `sda://celestrak/element_sets` asset updated, `sda_conjunction_tasks` triggers on its own with no timer, because the asset graph, not a guessed schedule, drives it. The `nasa_neows` task shows the dlt path end to end: dlt paginates the feed and lands a run-stamped Parquet partition under `data/snapshots/_landing/`, then the module's freeze step content-addresses those bytes and writes the derived NeoWs snapshot, so the extractor differs from celestrak but the pinned-snapshot contract is byte-for-byte the same. The space-track task writes only to the git-ignored `data/live/` tier and produces no shippable artifact. The artifact of the chapter is that pair: a versioned, provenance-stamped SDA snapshot sitting correctly across the raw and normalized tiers, DVC-pinned and MLflow-logged, plus the two thin DAGs that produce and consume it. From here, Chapter 3.10 turns these element sets into verifiable conjunction tasks whose gold answers are reproducible forever because they are bound to the content address in equation (9.1).

```admonish read-along
**[DPA] Harenslak et al., *Data Pipelines with Apache Airflow* (2e)** is the direct companion for the orchestration half of this chapter: chapters 2 and 3 build DAGs, run them in Docker, and set up incremental loading and backfilling, and chapter 4 on asset-aware scheduling is the exact Airflow-3 pattern I use so the task asset rebuilds when a snapshot updates. Read it for the orchestrator mechanics I lean on but do not re-derive; read this chapter for what those mechanics buy an eval, which is a snapshot you can name by hash and reproduce forever.

**[AIE] Huyen, *AI Engineering*, chapter 8** makes dataset engineering a first-class discipline: data lineage, quality gates, deduplication, and treating your data as a versioned artifact rather than a pile of files. This chapter is the SDA-specific instance of that discipline with the live-feed and licensing complications made concrete: the lineage is the content address plus DVC pin, the quality gate is Pydantic-per-record plus pandera-per-frame, and the "versioned artifact" is the immutable snapshot. Read the two together and the throughline is that data you cannot name exactly is data you cannot reason about, and an orchestrated, content-addressed pipeline is how you earn the right to name it.
```

```admonish substack-seed
There is a post here about the unglamorous truth that reproducible ML starts with reproducible *data*, and that "reproducible data" from a live feed is harder than it sounds. The hook: your model's evaluation is only as trustworthy as its ability to answer "which exact bytes was this built from", and for anything that changes over time (orbital elements, prices, the news) the honest answer is not a filename or a date, it is a hash. The move that makes it work is refusing to name data by "latest": you name a snapshot by the SHA-256 of its bytes, pin that with DVC, and now a task built on it is reproducible forever because re-running resolves the identical bytes and a deterministic generator gives the identical answer. Add the twist that not all data is yours to keep, that one of your five sources legally cannot be redistributed, and you get the design lesson that a licensing boundary is not a footnote you add at the end but a storage tier you build in from the start, which is exactly why some data belongs behind a live tool instead of in your repo.
```
