# SDA Extraction Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining extraction sources to `data/`: NASA NeoWs, Spaceflight News, and thespacedevs launches through dlt landing + the freeze layer, and space-track as a fetch-only embargoed source.

**Architecture:** Same doctrine as the foundation slice (see `2026-07-24-sda-data-platform-foundation.md`, merged into `data/`). Each public paginated feed is a dlt resource whose page loop runs through the shared `RateLimitedClient` (one rate-limit mechanism for everything), lands a run-stamped Parquet partition, then a shared `freeze_landing()` helper does the identical freeze: content-hash the landed bytes, gate the frame, write the derived Parquet snapshot, append the Delta table. space-track never touches dlt or the shippable tiers: raw and normalized output land only under `live_root()`.

**Tech Stack:** adds `dlt[filesystem,parquet]` and `spacetrack` to the existing foundation stack.

**Redistribution decisions encoded here:** `spaceflightnews` joins `REDISTRIBUTABLE` (The Space Devs, attribution-required, redistributable). `nasa` stays restricted (raw dumps embargoed; *derived* Parquet facts ship, per the book's per-endpoint-terms rule). `spacetrack` output may never leave `live_root()`.

---

## File Structure

```
data/src/sda_data/freeze.py                 # shared freeze_landing() helper (DRY across dlt sources)
data/src/sda_data/silver.py                 # MODIFY: generic append(table, df) + table_uri(name)
data/src/sda_data/config.py                 # MODIFY: REDISTRIBUTABLE += spaceflightnews
data/src/sda_data/gates.py                  # MODIFY: +NeoWsFrame, ArticleFrame, LaunchFrame; ElementSetFrame source isin += spacetrack
data/src/sda_data/tasks/nasa_neows.py       # dlt resource -> landing -> freeze -> delta neo_close_approaches
data/src/sda_data/tasks/spaceflightnews.py  # dlt resource -> landing -> freeze -> delta articles
data/src/sda_data/tasks/spacedevs_launches.py  # dlt resource -> landing -> freeze -> delta launches
data/src/sda_data/tasks/spacetrack_gp.py    # fetch-only, embargoed live tier, NO dlt, NO delta
data/dags/sda_ingest.py                     # MODIFY: add the four tasks + assets
data/.env.example                           # MODIFY: NASA_API_KEY, SPACETRACK_USER/PASSWORD
data/tests/test_freeze.py
data/tests/test_nasa_neows.py
data/tests/test_spaceflightnews.py
data/tests/test_spacedevs_launches.py
data/tests/test_spacetrack_gp.py
data/tests/conftest.py                      # MODIFY: disable dlt telemetry, pin dlt dirs to tmp
```

---

### Task 1: Dependencies, generic silver append, dlt-hermetic conftest

**Files:**
- Modify: `data/pyproject.toml` (dependencies list)
- Modify: `data/src/sda_data/silver.py`
- Modify: `data/tests/conftest.py`
- Test: `data/tests/test_silver.py` (add one test)

- [ ] **Step 1: Add deps**

Run: `cd data && uv add "dlt[filesystem,parquet]>=1.0" "spacetrack>=1.3"`
Expected: resolves and updates `uv.lock`

- [ ] **Step 2: Extend `data/tests/conftest.py`** (replace file)

```python
"""Shared fixtures. Every test runs against a throwaway data home."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dlt_hermetic(monkeypatch, tmp_path):
    """dlt must never phone home or write state outside the test sandbox."""
    monkeypatch.setenv("RUNTIME__DLTHUB_TELEMETRY", "false")
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / ".dlt-data"))


@pytest.fixture
def tmp_data_home(tmp_path, monkeypatch):
    """Point SDA_DATA_HOME at a tmp dir so tiers never touch the real disk."""
    monkeypatch.setenv("SDA_DATA_HOME", str(tmp_path))
    return tmp_path
```

- [ ] **Step 3: Add the failing test** — append this to `data/tests/test_silver.py`:

```python
def test_generic_append_routes_by_table_name(tmp_data_home):
    from sda_data.silver import append, table_uri

    df = frame([7])
    version = append("launches", df)
    assert version == 0
    assert table_uri("launches") == str(config.delta_root() / "launches")
    assert len(DeltaTable(table_uri("launches")).to_pandas()) == 1
```

Run: `cd data && uv run pytest tests/test_silver.py -v` — Expected: new test FAILS (`ImportError`)

- [ ] **Step 4: Rewrite `data/src/sda_data/silver.py`**

```python
"""Serving layer: Delta appends via delta-rs. One writer per table (enforced
by DAG design: each table is written by exactly one task, max_active_runs=1).
This layer is rebuildable from the frozen snapshots; it is NEVER the
provenance authority (book ch. 3.9 serving-layer invariants)."""
from __future__ import annotations

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from sda_data import config


def table_uri(name: str) -> str:
    """Local path by default; the delta tier root moves wholesale to s3:// later."""
    return str(config.delta_root() / name)


def append(name: str, df: pd.DataFrame) -> int:
    """Append one gated frame to a named table; returns the new table version."""
    uri = table_uri(name)
    opts = config.s3_storage_options() or None
    write_deltalake(uri, df, mode="append", storage_options=opts)
    return DeltaTable(uri, storage_options=opts).version()


def element_sets_uri() -> str:
    return table_uri("element_sets")


def append_element_sets(df: pd.DataFrame) -> int:
    return append("element_sets", df)
```

- [ ] **Step 5: Run + commit**

Run: `cd data && uv run pytest tests/test_silver.py tests/test_query.py tests/test_celestrak_tle.py -v` — Expected: all PASS (old callers unchanged)

```bash
git add data/pyproject.toml data/uv.lock data/src/sda_data/silver.py data/tests/conftest.py data/tests/test_silver.py
git commit -m "feat(data): generic delta append by table name; add dlt + spacetrack deps"
```

---

### Task 2: New gates and the redistribution update

**Files:**
- Modify: `data/src/sda_data/config.py` (REDISTRIBUTABLE)
- Modify: `data/src/sda_data/gates.py`
- Modify: `data/tests/test_config.py` (extend one test)
- Test: `data/tests/test_gates.py` (extend)

- [ ] **Step 1: Extend tests**

In `data/tests/test_config.py`, replace `test_redistributable_excludes_restricted_sources` with:

```python
def test_redistributable_excludes_restricted_sources():
    from sda_data import config

    assert "celestrak" in config.REDISTRIBUTABLE
    assert "spacedevs" in config.REDISTRIBUTABLE
    assert "spaceflightnews" in config.REDISTRIBUTABLE
    assert "spacetrack" not in config.REDISTRIBUTABLE
    assert "nasa" not in config.REDISTRIBUTABLE
```

Append to `data/tests/test_gates.py`:

```python
def test_spacetrack_is_a_valid_element_set_source():
    df = good_frame()
    df["source"] = "spacetrack"
    assert len(ElementSetFrame.validate(df)) == 2


def test_neows_frame_rejects_nonpositive_miss_distance():
    import pandas as pd

    from sda_data.gates import NeoWsFrame

    df = pd.DataFrame(
        {
            "neo_reference_id": ["3542519"],
            "close_approach_date": ["2026-07-25"],
            "miss_distance_km": [-1.0],
            "rel_velocity_kms": [10.0],
            "absolute_magnitude_h": [22.0],
            "source": ["nasa"],
            "snapshot_hash": ["sha256:" + "0" * 64],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        NeoWsFrame.validate(df)
```

Run: `cd data && uv run pytest tests/test_config.py tests/test_gates.py` — Expected: FAIL

- [ ] **Step 2: Update `config.py`** — change the constant:

```python
# Sources allowed onto the shippable raw tier. spacetrack/nasa deliberately absent.
REDISTRIBUTABLE = {"celestrak", "spacedevs", "spaceflightnews"}
```

- [ ] **Step 3: Update `gates.py`** — change `ElementSetFrame.source` and append the new frames:

```python
    source: Series[str] = pa.Field(isin=["celestrak", "spacedevs", "spacetrack"])
```

```python
class NeoWsFrame(pa.DataFrameModel):
    """NASA NeoWs close approaches. dlt lands the shape; pandera asserts the
    physics dlt's schema contract cannot."""

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


class ArticleFrame(pa.DataFrameModel):
    """Spaceflight News articles: the text branch feeding the 8.1 RAG corpus."""

    id: Series[int] = pa.Field(gt=0, unique=True)
    title: Series[str] = pa.Field(nullable=False)
    url: Series[str] = pa.Field(nullable=False)
    news_site: Series[str] = pa.Field(nullable=False)
    published_at: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(isin=["spaceflightnews"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False
        coerce = True


class LaunchFrame(pa.DataFrameModel):
    """thespacedevs Launch Library 2 launches (events table feed)."""

    id: Series[str] = pa.Field(nullable=False, unique=True)   # LL2 uuid
    name: Series[str] = pa.Field(nullable=False)
    net: Series[str] = pa.Field(nullable=False)               # no-earlier-than timestamp
    status_name: Series[str] = pa.Field(nullable=False)
    provider: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(isin=["spacedevs"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False
        coerce = True
```

- [ ] **Step 4: Run + commit**

Run: `cd data && uv run pytest tests/test_config.py tests/test_gates.py -v` — Expected: PASS

```bash
git add data/src/sda_data/config.py data/src/sda_data/gates.py data/tests/test_config.py data/tests/test_gates.py
git commit -m "feat(data): gates for neows/articles/launches; spaceflightnews redistributable"
```

---

### Task 3: The shared freeze helper

**Files:**
- Create: `data/src/sda_data/freeze.py`
- Test: `data/tests/test_freeze.py`

- [ ] **Step 1: Write the failing test**

`data/tests/test_freeze.py`:

```python
import pandas as pd
from deltalake import DeltaTable

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import LaunchFrame
from sda_data.silver import table_uri


def make_landing(tmp_data_home) -> None:
    landing = config.snap_root() / "_landing" / "x" / "run1"
    landing.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["u-1", "u-2"],
            "name": ["Falcon 9 | Starlink", "Vulcan | USSF-106"],
            "net": ["2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z"],
            "status_name": ["Go", "TBD"],
            "provider": ["SpaceX", "ULA"],
        }
    ).to_parquet(landing / "part.parquet", index=False)
    return landing


def test_freeze_landing_hashes_gates_and_appends(tmp_data_home):
    landing = make_landing(tmp_data_home)
    snapshot_hash = freeze_landing(
        landing, source="spacedevs", kind="launches",
        gate=LaunchFrame, delta_table="launches",
    )

    digest = snapshot_hash.split(":", 1)[1]
    assert list(config.raw_root().rglob(f"{digest}.raw"))          # frozen, shippable
    snaps = list((config.snap_root() / "spacedevs" / "launches").rglob("*.parquet"))
    assert len(snaps) == 1                                          # derived parquet
    table = DeltaTable(table_uri("launches")).to_pandas()
    assert len(table) == 2
    assert (table.snapshot_hash == snapshot_hash).all()
    assert (table.source == "spacedevs").all()
```

(Tier routing for restricted sources is already covered by
`test_snapshot.py::test_restricted_source_lands_on_live_tier`, since
`freeze_landing` delegates to `write_raw_snapshot`.)

Run: `cd data && uv run pytest tests/test_freeze.py -v` — Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 2: Write `data/src/sda_data/freeze.py`**

```python
"""The freeze layer for dlt landings: identical for every public feed.

dlt owns extract-to-land (fetch, paginate, normalize, write a run-stamped
Parquet partition) and STOPS. This helper takes a landing and does the book's
freeze contract: content-hash the landed bytes, register the immutable raw
snapshot (tier routed by source), stamp provenance, gate the physics, write
the derived Parquet snapshot, append the Delta serving table."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sda_data import config, silver
from sda_data.snapshot import write_raw_snapshot


def freeze_landing(
    landing: Path, *, source: str, kind: str, gate, delta_table: str
) -> str:
    fetch_time = datetime.now(timezone.utc)
    files = sorted(landing.rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    raw = df.to_parquet(index=False)                    # canonical bytes to hash
    _, snapshot_hash = write_raw_snapshot(source, kind, raw, fetch_time)

    df["source"] = source
    df["snapshot_hash"] = snapshot_hash
    df["fetch_time"] = fetch_time
    df = gate.validate(df)

    out_dir = config.snap_root() / source / kind / fetch_time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{snapshot_hash.split(':', 1)[1][:16]}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    version = silver.append(delta_table, df)
    print(
        f"{source} {kind}: {len(df)} rows  snapshot={snapshot_hash[:19]}...  "
        f"parquet={parquet_path}  delta=v{version}"
    )
    return snapshot_hash
```

- [ ] **Step 3: Run + commit**

Run: `cd data && uv run pytest tests/test_freeze.py -v` — Expected: PASS

```bash
git add data/src/sda_data/freeze.py data/tests/test_freeze.py
git commit -m "feat(data): shared freeze_landing helper for dlt landings"
```

---

### Task 4: NASA NeoWs source

**Files:**
- Create: `data/src/sda_data/tasks/nasa_neows.py`
- Test: `data/tests/test_nasa_neows.py`

- [ ] **Step 1: Write the failing test**

`data/tests/test_nasa_neows.py`:

```python
import json

import httpx
from deltalake import DeltaTable

from sda_data import config
from sda_data.ratelimit import RateLimitedClient, TokenBucket
from sda_data.silver import table_uri
from sda_data.tasks.nasa_neows import run

PAGE = {
    "links": {},                       # no next page
    "near_earth_objects": {
        "2026-07-25": [
            {
                "neo_reference_id": "3542519",
                "name": "(2010 PK9)",
                "absolute_magnitude_h": 21.8,
                "close_approach_data": [
                    {
                        "close_approach_date": "2026-07-25",
                        "miss_distance": {"kilometers": "1200000.5"},
                        "relative_velocity": {"kilometers_per_second": "14.2"},
                    }
                ],
            }
        ]
    },
}


def fake_client() -> RateLimitedClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "TESTKEY"
        return httpx.Response(200, content=json.dumps(PAGE).encode())

    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )


def test_run_lands_freezes_and_serves(tmp_data_home, monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", "TESTKEY")
    snapshot_hash = run("2026-07-25", "2026-07-25", client=fake_client())

    # Raw NeoWs dump is NOT redistributable: raw goes to the embargoed live tier.
    digest = snapshot_hash.split(":", 1)[1]
    assert list(config.live_root().rglob(f"{digest}.raw"))
    assert not list(config.raw_root().rglob("*.raw"))

    # Derived facts ship: parquet snapshot + delta table.
    assert list((config.snap_root() / "nasa" / "neows").rglob("*.parquet"))
    table = DeltaTable(table_uri("neo_close_approaches")).to_pandas()
    assert len(table) == 1
    assert table.iloc[0].miss_distance_km == 1200000.5
    assert (table.snapshot_hash == snapshot_hash).all()
```

Run: `cd data && uv run pytest tests/test_nasa_neows.py -v` — Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 2: Write `data/src/sda_data/tasks/nasa_neows.py`**

```python
"""NASA NeoWs close approaches: dlt extract-to-land -> freeze -> delta.

The page loop runs through the shared RateLimitedClient so the hard cap is the
same mechanism as every other source (NASA allows ~1000 req/hr per key). dlt
owns landing (JSON->tables, run-stamped partition) and stops; freeze_landing
does the rest. Raw dumps are embargoed (nasa not in REDISTRIBUTABLE); the
derived facts here are what ships.

Standalone:
    NASA_API_KEY=... uv run python -m sda_data.tasks.nasa_neows --start 2026-07-01 --end 2026-07-02
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import dlt

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import NeoWsFrame
from sda_data.ratelimit import RateLimitedClient, TokenBucket

NEOWS = "https://api.nasa.gov/neo/rest/v1"


def default_client() -> RateLimitedClient:
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=0.25, capacity=4),   # ~900/hr, under the 1000 cap
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


@dlt.resource(name="neo_close_approaches", write_disposition="replace")
def neows_feed(client: RateLimitedClient, start: str, end: str, api_key: str):
    url: str | None = f"{NEOWS}/feed"
    params: dict | None = {"start_date": start, "end_date": end, "api_key": api_key}
    while url:
        page = client.get(url, params=params).json()
        params = None                     # NeoWs next links carry their own query string
        for _day, objects in page["near_earth_objects"].items():
            for o in objects:
                for ca in o["close_approach_data"]:
                    yield {
                        "neo_reference_id": o["neo_reference_id"],
                        "name": o["name"],
                        "close_approach_date": ca["close_approach_date"],
                        "miss_distance_km": float(ca["miss_distance"]["kilometers"]),
                        "rel_velocity_kms": float(
                            ca["relative_velocity"]["kilometers_per_second"]
                        ),
                        "absolute_magnitude_h": float(o["absolute_magnitude_h"]),
                    }
        url = page.get("links", {}).get("next")


def extract_to_landing(client: RateLimitedClient, start: str, end: str, api_key: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    landing = config.snap_root() / "_landing" / "nasa_neows" / run_id
    pipe = dlt.pipeline(
        pipeline_name="nasa_neows",
        pipelines_dir=str(config.data_home() / ".dlt"),
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="neows",
    )
    pipe.run(neows_feed(client, start, end, api_key), loader_file_format="parquet")
    return landing


def run(start: str, end: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    api_key = os.environ["NASA_API_KEY"]
    landing = extract_to_landing(client, start, end, api_key)
    return freeze_landing(
        landing, source="nasa", kind="neows",
        gate=NeoWsFrame, delta_table="neo_close_approaches",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC date, e.g. 2026-07-01")
    ap.add_argument("--end", required=True, help="UTC date, inclusive")
    args = ap.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run + commit**

Run: `cd data && uv run pytest tests/test_nasa_neows.py -v` — Expected: PASS

```bash
git add data/src/sda_data/tasks/nasa_neows.py data/tests/test_nasa_neows.py
git commit -m "feat(data): NASA NeoWs dlt source with embargoed raw + shipped derived facts"
```

---

### Task 5: Spaceflight News source

**Files:**
- Create: `data/src/sda_data/tasks/spaceflightnews.py`
- Test: `data/tests/test_spaceflightnews.py`

- [ ] **Step 1: Write the failing test**

`data/tests/test_spaceflightnews.py`:

```python
import json

import httpx
from deltalake import DeltaTable

from sda_data import config
from sda_data.ratelimit import RateLimitedClient, TokenBucket
from sda_data.silver import table_uri
from sda_data.tasks.spaceflightnews import run

PAGE1 = {
    "next": "https://api.spaceflightnewsapi.net/v4/articles/?offset=1",
    "results": [
        {
            "id": 101,
            "title": "ISS debris avoidance maneuver",
            "url": "https://example.com/a/101",
            "news_site": "NASA",
            "summary": "The ISS moved.",
            "published_at": "2026-07-24T10:00:00Z",
        }
    ],
}
PAGE2 = {
    "next": None,
    "results": [
        {
            "id": 102,
            "title": "Starship launch window",
            "url": "https://example.com/a/102",
            "news_site": "SpaceNews",
            "summary": "A window opens.",
            "published_at": "2026-07-24T11:00:00Z",
        }
    ],
}


def fake_client() -> RateLimitedClient:
    def handler(request: httpx.Request) -> httpx.Response:
        page = PAGE2 if request.url.params.get("offset") == "1" else PAGE1
        return httpx.Response(200, content=json.dumps(page).encode())

    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )


def test_run_follows_pagination_and_serves(tmp_data_home):
    snapshot_hash = run("2026-07-24", client=fake_client())

    # Articles are redistributable: raw snapshot on the shippable tier.
    digest = snapshot_hash.split(":", 1)[1]
    assert list(config.raw_root().rglob(f"{digest}.raw"))

    table = DeltaTable(table_uri("articles")).to_pandas()
    assert set(table.id) == {101, 102}              # both pages landed
    assert (table.source == "spaceflightnews").all()
```

Run: `cd data && uv run pytest tests/test_spaceflightnews.py -v` — Expected: FAIL

- [ ] **Step 2: Write `data/src/sda_data/tasks/spaceflightnews.py`**

```python
"""Spaceflight News (SNAPI v4) articles: the text branch feeding the 8.1 RAG
corpus. Same shape as nasa_neows: dlt lands, freeze_landing freezes.

Standalone:
    uv run python -m sda_data.tasks.spaceflightnews --since 2026-07-01
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import dlt

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import ArticleFrame
from sda_data.ratelimit import RateLimitedClient, TokenBucket

SNAPI = "https://api.spaceflightnewsapi.net/v4/articles/"


def default_client() -> RateLimitedClient:
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=1.0, capacity=5),
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


@dlt.resource(name="articles", write_disposition="replace")
def articles_feed(client: RateLimitedClient, since: str):
    url: str | None = SNAPI
    params: dict | None = {"published_at_gte": since, "limit": 100, "ordering": "published_at"}
    while url:
        page = client.get(url, params=params).json()
        params = None                     # SNAPI next links carry their own query string
        for a in page["results"]:
            yield {
                "id": a["id"],
                "title": a["title"],
                "url": a["url"],
                "news_site": a["news_site"],
                "summary": a.get("summary") or "",
                "published_at": a["published_at"],
            }
        url = page.get("next")


def extract_to_landing(client: RateLimitedClient, since: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    landing = config.snap_root() / "_landing" / "spaceflightnews" / run_id
    pipe = dlt.pipeline(
        pipeline_name="spaceflightnews",
        pipelines_dir=str(config.data_home() / ".dlt"),
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="sfn",
    )
    pipe.run(articles_feed(client, since), loader_file_format="parquet")
    return landing


def run(since: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    landing = extract_to_landing(client, since)
    return freeze_landing(
        landing, source="spaceflightnews", kind="articles",
        gate=ArticleFrame, delta_table="articles",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="published_at lower bound, e.g. 2026-07-01")
    args = ap.parse_args()
    run(args.since)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run + commit**

Run: `cd data && uv run pytest tests/test_spaceflightnews.py -v` — Expected: PASS

```bash
git add data/src/sda_data/tasks/spaceflightnews.py data/tests/test_spaceflightnews.py
git commit -m "feat(data): Spaceflight News dlt source feeding the articles table"
```

---

### Task 6: thespacedevs launches source

**Files:**
- Create: `data/src/sda_data/tasks/spacedevs_launches.py`
- Test: `data/tests/test_spacedevs_launches.py`

- [ ] **Step 1: Write the failing test**

`data/tests/test_spacedevs_launches.py`:

```python
import json

import httpx
from deltalake import DeltaTable

from sda_data.ratelimit import RateLimitedClient, TokenBucket
from sda_data.silver import table_uri
from sda_data.tasks.spacedevs_launches import run

PAGE = {
    "next": None,
    "results": [
        {
            "id": "9d1af13d-8d3b-4450-b56f-6e29ea9b1c07",
            "name": "Falcon 9 Block 5 | Starlink Group",
            "net": "2026-08-01T04:30:00Z",
            "status": {"name": "Go for Launch"},
            "launch_service_provider": {"name": "SpaceX"},
        }
    ],
}


def fake_client() -> RateLimitedClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["limit"] == "100"
        return httpx.Response(200, content=json.dumps(PAGE).encode())

    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )


def test_run_flattens_and_serves(tmp_data_home):
    snapshot_hash = run("2026-07-24", client=fake_client())
    table = DeltaTable(table_uri("launches")).to_pandas()
    assert len(table) == 1
    row = table.iloc[0]
    assert row.status_name == "Go for Launch"
    assert row.provider == "SpaceX"
    assert row.snapshot_hash == snapshot_hash
```

Run: `cd data && uv run pytest tests/test_spacedevs_launches.py -v` — Expected: FAIL

- [ ] **Step 2: Write `data/src/sda_data/tasks/spacedevs_launches.py`**

```python
"""thespacedevs Launch Library 2 launches (events feed).

LL2's free tier is HARSHLY throttled (~15 requests/hour): the default bucket
here is the compliance mechanism, and the DAG's 6-hour cadence uses ~1-2
requests per run. dlt lands, freeze_landing freezes.

Standalone:
    uv run python -m sda_data.tasks.spacedevs_launches --since 2026-07-01
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import dlt

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import LaunchFrame
from sda_data.ratelimit import RateLimitedClient, TokenBucket

LL2 = "https://ll.thespacedevs.com/2.2.0/launch/"


def default_client() -> RateLimitedClient:
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=15 / 3600, capacity=2),   # LL2 free tier ~15/hr
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


@dlt.resource(name="launches", write_disposition="replace")
def launches_feed(client: RateLimitedClient, since: str):
    url: str | None = LL2
    params: dict | None = {"net__gte": since, "limit": 100, "ordering": "net"}
    while url:
        page = client.get(url, params=params).json()
        params = None                     # LL2 next links carry their own query string
        for launch in page["results"]:
            yield {
                "id": launch["id"],
                "name": launch["name"],
                "net": launch["net"],
                "status_name": launch["status"]["name"],
                "provider": launch["launch_service_provider"]["name"],
            }
        url = page.get("next")


def extract_to_landing(client: RateLimitedClient, since: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    landing = config.snap_root() / "_landing" / "spacedevs_launches" / run_id
    pipe = dlt.pipeline(
        pipeline_name="spacedevs_launches",
        pipelines_dir=str(config.data_home() / ".dlt"),
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="ll2",
    )
    pipe.run(launches_feed(client, since), loader_file_format="parquet")
    return landing


def run(since: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    landing = extract_to_landing(client, since)
    return freeze_landing(
        landing, source="spacedevs", kind="launches",
        gate=LaunchFrame, delta_table="launches",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="net lower bound, e.g. 2026-07-01")
    args = ap.parse_args()
    run(args.since)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run + commit**

Run: `cd data && uv run pytest tests/test_spacedevs_launches.py -v` — Expected: PASS

```bash
git add data/src/sda_data/tasks/spacedevs_launches.py data/tests/test_spacedevs_launches.py
git commit -m "feat(data): thespacedevs LL2 launches source with 15/hr rate cap"
```

---

### Task 7: space-track fetch-only (embargoed)

**Files:**
- Create: `data/src/sda_data/tasks/spacetrack_gp.py`
- Test: `data/tests/test_spacetrack_gp.py`

- [ ] **Step 1: Write the failing test**

`data/tests/test_spacetrack_gp.py`:

```python
import json

from sda_data import config
from sda_data.tasks.spacetrack_gp import run

GP = [
    {
        "NORAD_CAT_ID": 25544,
        "OBJECT_NAME": "ISS (ZARYA)",
        "EPOCH": "2026-07-24T06:00:00",
        "MEAN_MOTION": 15.50,
        "ECCENTRICITY": 0.0003,
        "INCLINATION": 51.64,
        "RA_OF_ASC_NODE": 120.0,
        "ARG_OF_PERICENTER": 30.0,
        "MEAN_ANOMALY": 300.0,
        "BSTAR": 0.00012,
    }
]


def test_everything_lands_on_the_live_tier_only(tmp_data_home):
    snapshot_hash = run(fetch=lambda: json.dumps(GP).encode())

    digest = snapshot_hash.split(":", 1)[1]
    assert list(config.live_root().rglob(f"{digest}.raw"))

    # The embargo invariant, as assertions: nothing outside live_root.
    assert not list(config.raw_root().rglob("*"))
    assert not list(config.snap_root().rglob("*"))
    assert not list(config.delta_root().rglob("*"))

    normalized = list(config.live_root().rglob("*.parquet"))
    assert len(normalized) == 1


def test_normalized_view_carries_provenance(tmp_data_home):
    import pandas as pd

    snapshot_hash = run(fetch=lambda: json.dumps(GP).encode())
    df = pd.read_parquet(next(config.live_root().rglob("*.parquet")))
    assert (df.snapshot_hash == snapshot_hash).all()
    assert (df.source == "spacetrack").all()
```

Run: `cd data && uv run pytest tests/test_spacetrack_gp.py -v` — Expected: FAIL

- [ ] **Step 2: Write `data/src/sda_data/tasks/spacetrack_gp.py`**

```python
"""space-track GP pull: fetch-only, EMBARGOED. Everything this module writes
lands under live_root() (git-ignored, never DVC-tracked, never shipped). No
dlt, no shippable parquet, no Delta: the redistribution line is a code
invariant, not a promise (book ch. 3.9).

The `spacetrack` library owns login, session, and per-endpoint throttling;
never disable its rate limiting. Standalone:
    SPACETRACK_USER=... SPACETRACK_PASSWORD=... uv run python -m sda_data.tasks.spacetrack_gp
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd

from sda_data import config
from sda_data.gates import ElementSetFrame
from sda_data.models import Provenance, from_celestrak_gp
from sda_data.snapshot import write_raw_snapshot


def default_fetch() -> bytes:
    from spacetrack import SpaceTrackClient  # deferred: only the real path needs it

    st = SpaceTrackClient(
        identity=os.environ["SPACETRACK_USER"],
        password=os.environ["SPACETRACK_PASSWORD"],
    )
    # Latest GP set, one batched request (never loop per object: rate limits).
    return st.gp(epoch=">now-1", format="json").encode()


def run(fetch: Callable[[], bytes] | None = None) -> str:
    raw = (fetch or default_fetch)()
    fetch_time = datetime.now(timezone.utc)

    # Raw bytes: write_raw_snapshot routes spacetrack to the live tier.
    _, snapshot_hash = write_raw_snapshot("spacetrack", "gp", raw, fetch_time)
    prov = Provenance(source="spacetrack", fetch_time=fetch_time, snapshot_hash=snapshot_hash)

    # Normalized view: same schema as celestrak GP, but it stays embargoed too.
    rows = [from_celestrak_gp(rec, prov).model_dump() for rec in json.loads(raw)]
    df = pd.json_normalize(rows).rename(
        columns={
            "prov.source": "source",
            "prov.fetch_time": "fetch_time",
            "prov.snapshot_hash": "snapshot_hash",
        }
    )
    df = ElementSetFrame.validate(df)

    out_dir = config.live_root() / "spacetrack" / "gp_normalized" / fetch_time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{snapshot_hash.split(':', 1)[1][:16]}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    print(f"spacetrack gp: {len(df)} element sets (EMBARGOED tier)  "
          f"snapshot={snapshot_hash[:19]}...")
    return snapshot_hash


def main() -> None:
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run + commit**

Run: `cd data && uv run pytest tests/test_spacetrack_gp.py -v` — Expected: PASS

```bash
git add data/src/sda_data/tasks/spacetrack_gp.py data/tests/test_spacetrack_gp.py
git commit -m "feat(data): space-track GP pull, embargoed live tier only"
```

---

### Task 8: DAG wiring, env template, full suite, push

**Files:**
- Modify: `data/dags/sda_ingest.py`
- Modify: `data/.env.example`

- [ ] **Step 1: Rewrite `data/dags/sda_ingest.py`**

```python
"""Thin producer DAG: schedule the standalone task modules, declare assets.
No fetching, parsing, or validating happens HERE; it happens in sda_data.tasks."""
from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE_SNAPSHOT = Asset("sda://celestrak/element_sets")
NEOWS_SNAPSHOT = Asset("sda://nasa/neows")
ARTICLE_SNAPSHOT = Asset("sda://spaceflightnews/articles")
LAUNCH_SNAPSHOT = Asset("sda://spacedevs/launches")

RUN = "cd /opt/project/data && uv run python -m sda_data.tasks.{mod}"

with DAG(
    dag_id="sda_ingest",
    schedule="0 */6 * * *",            # clock drives the producer
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,                 # single node + one-writer-per-table
    default_args={"retries": 3, "retry_delay": pendulum.duration(minutes=5)},
    tags=["sda", "ingest"],
) as dag:
    fetch_tle = BashOperator(
        task_id="celestrak_tle",
        bash_command=RUN.format(mod="celestrak_tle --group active"),
        outlets=[TLE_SNAPSHOT],
    )
    fetch_neows = BashOperator(
        task_id="nasa_neows",
        # {{ ds }} is the run's logical date: incremental by data interval.
        bash_command=RUN.format(mod="nasa_neows --start {{ ds }} --end {{ ds }}"),
        outlets=[NEOWS_SNAPSHOT],
    )
    fetch_articles = BashOperator(
        task_id="spaceflightnews",
        bash_command=RUN.format(mod="spaceflightnews --since {{ ds }}"),
        outlets=[ARTICLE_SNAPSHOT],
    )
    fetch_launches = BashOperator(
        task_id="spacedevs_launches",
        bash_command=RUN.format(mod="spacedevs_launches --since {{ ds }}"),
        outlets=[LAUNCH_SNAPSHOT],
    )
    fetch_spacetrack = BashOperator(
        task_id="spacetrack_gp",
        bash_command=RUN.format(mod="spacetrack_gp"),
        # no outlet asset: fetch-only, embargoed, never shipped.
    )
```

- [ ] **Step 2: Extend `data/.env.example`** — append:

```bash
# NASA api.nasa.gov key (free): https://api.nasa.gov
NASA_API_KEY=

# space-track.org credentials. Output is EMBARGOED (live tier only).
SPACETRACK_USER=
SPACETRACK_PASSWORD=
```

And pass them through in `docker-compose.airflow.yml`'s `x-airflow-common.environment`:

```yaml
    NASA_API_KEY: ${NASA_API_KEY:-}
    SPACETRACK_USER: ${SPACETRACK_USER:-}
    SPACETRACK_PASSWORD: ${SPACETRACK_PASSWORD:-}
```

- [ ] **Step 3: Full suite + commit + push**

Run: `cd data && uv run pytest -v` — Expected: all PASS

```bash
git add data/dags/sda_ingest.py data/.env.example data/docker-compose.airflow.yml
git commit -m "feat(data): wire all five sources into the ingest DAG"
git push -u origin claude/mdbook-evals-as-rewards-spec-fvhxw7
```

Update PR #23's body to note the second slice (extraction sources) now included.
