# SDA Data Platform Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `data/` sub-project foundation: config tiers, a rate-limited HTTP client, the content-addressed freeze layer, the celestrak TLE ingest end-to-end, a Delta Lake silver appender, a DuckDB query path, and the corrected single-node Airflow 3 compose stack with a thin DAG.

**Architecture:** Everything follows the book's thin-DAG doctrine (chapter 3.9): Airflow only schedules; all logic lives in plain `uv run` modules under `data/src/sda_data/` that run and test standalone. Storage follows the chapter 0.5 tier contract: local paths are the working tier, MinIO on the NAS is the durable tier (reached via env-configured S3 settings; tests never touch the network). Provenance authority is the content-hashed raw snapshot; the Delta table is a rebuildable serving layer with one writer per table.

**Tech Stack:** Python 3.12, uv, httpx, pydantic v2, pandera, pandas, pyarrow, duckdb, deltalake (delta-rs), pytest. Airflow 3 runs only inside Docker Compose, never in the dev venv.

**Deliberate deviations from the book's inline code (book gets synced in a later docs PR):**
1. `config.py` exposes *functions* (`raw_root()`, not module constants) so tests can redirect tiers via `SDA_DATA_HOME` without import-order games.
2. The rate limiter is a hand-rolled token bucket (30 lines, injectable clock) instead of `pyrate-limiter`, so tests are deterministic and we don't track a fast-moving dependency's API.
3. DuckDB reads Delta via `deltalake` → Arrow (no network extension install needed), not the `delta` extension.

**Not in this plan (later plans):** dlt sources (NeoWs, Spaceflight News, thespacedevs), space-track, dbt gold models, conjunction sieve, Lance/RAG index, MCP server, MLflow logging, DVC wiring.

---

## File Structure

```
data/
  pyproject.toml                  # uv project, pinned deps, pytest config
  .env.example                    # every env var the stack reads, documented
  Makefile                        # test / up / down / doctor / backup targets
  docker-compose.airflow.yml      # Airflow 3: api-server, scheduler, dag-processor, triggerer, postgres
  dags/sda_ingest.py              # thin producer DAG (BashOperator + Asset outlet)
  scripts/backup_offsite.sh       # rclone offsite backup of bronze+DVC buckets (dry-run default)
  src/sda_data/__init__.py
  src/sda_data/config.py          # storage tiers + S3 env settings (functions, env-driven)
  src/sda_data/ratelimit.py       # TokenBucket + RateLimitedClient (httpx wrapper)
  src/sda_data/snapshot.py        # content_address + write_raw_snapshot (freeze layer)
  src/sda_data/models.py          # Pydantic per-record contracts + provenance
  src/sda_data/gates.py           # pandera per-frame physics gates
  src/sda_data/silver.py          # Delta append writer (one writer per table)
  src/sda_data/query.py           # DuckDB over Delta via Arrow
  src/sda_data/doctor.py          # environment sanity checks (make doctor)
  src/sda_data/tasks/__init__.py
  src/sda_data/tasks/celestrak_tle.py  # fetch -> freeze -> gate -> parquet -> delta
  tests/conftest.py               # tmp-tier fixture (SDA_DATA_HOME -> tmp_path)
  tests/test_config.py
  tests/test_ratelimit.py
  tests/test_snapshot.py
  tests/test_models.py
  tests/test_gates.py
  tests/test_silver.py
  tests/test_query.py
  tests/test_celestrak_tle.py
  tests/test_doctor.py
```

---

### Task 1: Project scaffold

**Files:**
- Create: `data/pyproject.toml`
- Create: `data/src/sda_data/__init__.py`
- Create: `data/src/sda_data/tasks/__init__.py`
- Create: `data/tests/conftest.py`
- Create: `data/tests/test_scaffold.py`

- [ ] **Step 1: Write `data/pyproject.toml`**

```toml
[project]
name = "sda-data"
version = "0.1.0"
description = "SDA data platform: ingest, freeze, serve (thesis-tech-stack data/ sub-project)"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.7",
    "pandera>=0.20",
    "pandas>=2.2",
    "pyarrow>=16",
    "duckdb>=1.0",
    "deltalake>=0.19",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sda_data"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Create package inits and the tmp-tier fixture**

`data/src/sda_data/__init__.py` and `data/src/sda_data/tasks/__init__.py` are empty files.

`data/tests/conftest.py`:

```python
"""Shared fixtures. Every test runs against a throwaway data home."""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_data_home(tmp_path, monkeypatch):
    """Point SDA_DATA_HOME at a tmp dir so tiers never touch the real disk."""
    monkeypatch.setenv("SDA_DATA_HOME", str(tmp_path))
    return tmp_path
```

- [ ] **Step 3: Write a scaffold smoke test**

`data/tests/test_scaffold.py`:

```python
def test_package_imports():
    import sda_data  # noqa: F401
```

- [ ] **Step 4: Install and run**

Run: `cd data && uv sync && uv run pytest tests/test_scaffold.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add data/pyproject.toml data/uv.lock data/src data/tests
git commit -m "feat(data): scaffold the sda-data uv sub-project"
```

---

### Task 2: Config — storage tiers and S3 settings

**Files:**
- Create: `data/src/sda_data/config.py`
- Test: `data/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_config.py`:

```python
from pathlib import Path


def test_tiers_hang_off_data_home(tmp_data_home):
    from sda_data import config

    assert config.data_home() == Path(tmp_data_home)
    assert config.raw_root() == tmp_data_home / "data" / "raw"
    assert config.snap_root() == tmp_data_home / "data" / "snapshots"
    assert config.live_root() == tmp_data_home / "data" / "live"
    assert config.delta_root() == tmp_data_home / "data" / "delta"


def test_redistributable_excludes_restricted_sources():
    from sda_data import config

    assert "celestrak" in config.REDISTRIBUTABLE
    assert "spacedevs" in config.REDISTRIBUTABLE
    assert "spacetrack" not in config.REDISTRIBUTABLE
    assert "nasa" not in config.REDISTRIBUTABLE


def test_s3_storage_options_from_env(monkeypatch):
    from sda_data import config

    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    opts = config.s3_storage_options()
    assert opts["AWS_ENDPOINT_URL"] == "https://s3.example.com"
    assert opts["AWS_ACCESS_KEY_ID"] == "k"
    assert opts["AWS_SECRET_ACCESS_KEY"] == "s"


def test_s3_storage_options_empty_when_unconfigured(monkeypatch):
    from sda_data import config

    for var in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert config.s3_storage_options() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`

- [ ] **Step 3: Write `data/src/sda_data/config.py`**

```python
"""Storage tiers and S3 settings. Everything is env-driven and read at call
time (functions, not import-time constants) so tests and containers can
redirect tiers without import-order games.

Tier contract (book ch. 0.5): local paths under SDA_DATA_HOME are the working
tier; MinIO on the NAS (S3_ENDPOINT_URL) is the durable tier. The live tier is
the redistribution boundary, in code.
"""
from __future__ import annotations

import os
from pathlib import Path

# Sources allowed onto the shippable raw tier. spacetrack/nasa deliberately absent.
REDISTRIBUTABLE = {"celestrak", "spacedevs"}


def data_home() -> Path:
    default = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("SDA_DATA_HOME", default))


def raw_root() -> Path:
    """Immutable raw bytes, DVC-tracked, shippable."""
    return data_home() / "data" / "raw"


def snap_root() -> Path:
    """Normalized Parquet, derived from raw_root()."""
    return data_home() / "data" / "snapshots"


def live_root() -> Path:
    """EMBARGOED: space-track etc. git-ignored, never DVC-tracked."""
    return data_home() / "data" / "live"


def delta_root() -> Path:
    """Serving layer: Delta tables. Rebuildable, never the provenance authority."""
    return data_home() / "data" / "delta"


def s3_storage_options() -> dict[str, str]:
    """delta-rs style storage options for the MinIO endpoint, from env.
    Empty dict when unconfigured, so local-path workflows need no setup."""
    mapping = {
        "AWS_ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL", ""),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    }
    return {k: v for k, v in mapping.items() if v}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/config.py data/tests/test_config.py
git commit -m "feat(data): env-driven storage tiers and S3 settings"
```

---

### Task 3: Rate-limited HTTP client

**Files:**
- Create: `data/src/sda_data/ratelimit.py`
- Test: `data/tests/test_ratelimit.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_ratelimit.py`:

```python
import httpx

from sda_data.ratelimit import RateLimitedClient, TokenBucket


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_bucket_allows_burst_up_to_capacity():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=1.0, capacity=3, clock=clock)
    for _ in range(3):
        bucket.acquire()
    assert clock.slept == []          # burst within capacity never sleeps


def test_bucket_blocks_when_empty_and_refills():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=2.0, capacity=1, clock=clock)
    bucket.acquire()                  # drains the bucket
    bucket.acquire()                  # must wait for one token at 2/sec
    assert clock.slept == [0.5]


def test_client_applies_bucket_per_request():
    clock = FakeClock()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    client = RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=1.0, capacity=1, clock=clock),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )
    r1 = client.get("https://example.com/a")
    r2 = client.get("https://example.com/b")
    assert r1.status_code == r2.status_code == 200
    assert calls == ["/a", "/b"]
    assert clock.slept == [1.0]       # second request waited for a token


def test_client_sends_user_agent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200)

    client = RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )
    client.get("https://example.com/")
    assert seen["ua"] == "sda-thesis-pipeline/0.1 (+research)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_ratelimit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.ratelimit'`

- [ ] **Step 3: Write `data/src/sda_data/ratelimit.py`**

```python
"""A hard rate cap for every outbound feed request.

Retries handle transient failures; they are NOT a compliance mechanism. This
token bucket guarantees we stay under a source's published request budget
(celestrak asks for infrequent cached group pulls; space-track enforces
~30/min and ~300/hr). The clock is injectable so tests are deterministic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class _WallClock:
    def time(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class TokenBucket:
    rate_per_sec: float
    capacity: int
    clock: Clock = field(default_factory=_WallClock)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last = self.clock.time()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        now = self.clock.time()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_sec)
        self._last = now
        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) / self.rate_per_sec
            self.clock.sleep(wait)
            self._last = self.clock.time()
            self._tokens = 1.0
        self._tokens -= 1.0


class RateLimitedClient:
    """httpx.Client wrapper: every request pays a token first.

    One instance per source, shared across that source's task modules, so the
    cap holds no matter how many call sites fetch.
    """

    def __init__(
        self,
        bucket: TokenBucket,
        user_agent: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._bucket = bucket
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            transport=transport or httpx.HTTPTransport(retries=3),
        )

    def get(self, url: str, **kwargs) -> httpx.Response:
        self._bucket.acquire()
        response = self._client.get(url, **kwargs)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_ratelimit.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/ratelimit.py data/tests/test_ratelimit.py
git commit -m "feat(data): token-bucket rate-limited HTTP client"
```

---

### Task 4: Freeze layer — content-addressed snapshots

**Files:**
- Create: `data/src/sda_data/snapshot.py`
- Test: `data/tests/test_snapshot.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_snapshot.py`:

```python
import hashlib
from datetime import datetime, timezone

from sda_data import config
from sda_data.snapshot import content_address, write_raw_snapshot

FETCH = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def test_content_address_is_sha256_of_bytes():
    raw = b"orbital bytes"
    expected = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert content_address(raw) == expected


def test_redistributable_source_lands_on_raw_tier(tmp_data_home):
    path, addr = write_raw_snapshot("celestrak", "gp-active", b"abc", FETCH)
    assert path.is_file()
    assert config.raw_root() in path.parents
    assert path.name == addr.split(":", 1)[1] + ".raw"
    assert "2026-07-24" in str(path)


def test_restricted_source_lands_on_live_tier(tmp_data_home):
    path, _ = write_raw_snapshot("spacetrack", "cdm", b"secret", FETCH)
    assert config.live_root() in path.parents
    assert config.raw_root() not in path.parents


def test_write_is_idempotent_for_identical_bytes(tmp_data_home):
    p1, a1 = write_raw_snapshot("celestrak", "gp-active", b"same", FETCH)
    p2, a2 = write_raw_snapshot("celestrak", "gp-active", b"same", FETCH)
    assert p1 == p2 and a1 == a2


def test_snapshot_file_is_never_mutated(tmp_data_home):
    p1, _ = write_raw_snapshot("celestrak", "gp-active", b"v1", FETCH)
    before = p1.read_bytes()
    write_raw_snapshot("celestrak", "gp-active", b"v2", FETCH)  # new addr, new file
    assert p1.read_bytes() == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.snapshot'`

- [ ] **Step 3: Write `data/src/sda_data/snapshot.py`**

```python
"""Immutable, content-addressed raw snapshots. The filename IS the hash
(book eq. 9.1). Restricted sources land on the embargoed live tier and never
touch the shippable, DVC-tracked raw tier."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from sda_data import config


def content_address(raw: bytes) -> str:
    """addr(s) = sha256(bytes(s))."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def write_raw_snapshot(
    source: str, kind: str, raw: bytes, fetch_time: datetime
) -> tuple[Path, str]:
    """Write bytes ONCE, named by their own hash."""
    addr = content_address(raw)
    digest = addr.split(":", 1)[1]
    root = config.raw_root() if source in config.REDISTRIBUTABLE else config.live_root()
    out = root / source / kind / fetch_time.strftime("%Y-%m-%d") / f"{digest}.raw"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():  # identical bytes hash identically: write-once is free
        out.write_bytes(raw)
    return out, addr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_snapshot.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/snapshot.py data/tests/test_snapshot.py
git commit -m "feat(data): content-addressed immutable snapshot freeze layer"
```

---

### Task 5: Pydantic per-record contracts

**Files:**
- Create: `data/src/sda_data/models.py`
- Test: `data/tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_models.py`:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sda_data.models import ElementSetAtEpoch, Provenance, from_celestrak_gp

PROV = Provenance(
    source="celestrak",
    fetch_time=datetime(2026, 7, 24, tzinfo=timezone.utc),
    snapshot_hash="sha256:" + "0" * 64,
)

GP_RECORD = {
    "NORAD_CAT_ID": "25544",
    "OBJECT_NAME": "ISS (ZARYA)",
    "EPOCH": "2026-07-24T06:00:00",
    "MEAN_MOTION": "15.50",
    "ECCENTRICITY": "0.0003",
    "INCLINATION": "51.64",
    "RA_OF_ASC_NODE": "120.0",
    "ARG_OF_PERICENTER": "30.0",
    "MEAN_ANOMALY": "300.0",
    "BSTAR": "0.00012",
}


def test_maps_celestrak_gp_record():
    rec = from_celestrak_gp(GP_RECORD, PROV)
    assert rec.norad_cat_id == 25544
    assert rec.object_name == "ISS (ZARYA)"
    assert rec.mean_motion == pytest.approx(15.50)
    assert rec.prov.snapshot_hash.startswith("sha256:")


def test_bstar_is_optional():
    rec = from_celestrak_gp({**GP_RECORD, "BSTAR": None}, PROV)
    assert rec.bstar is None


def test_unbound_eccentricity_rejected():
    with pytest.raises(ValidationError):
        ElementSetAtEpoch(
            norad_cat_id=1,
            object_name="X",
            epoch=datetime(2026, 1, 1),
            mean_motion=15.0,
            eccentricity=1.4,  # not a bound orbit
            inclination=50.0,
            ra_of_asc_node=0.0,
            arg_of_pericenter=0.0,
            mean_anomaly=0.0,
            prov=PROV,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.models'`

- [ ] **Step 3: Write `data/src/sda_data/models.py`**

```python
"""Per-record ingest contract. Provenance is stamped on EVERY record."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str            # "celestrak"
    fetch_time: datetime   # when we pulled the raw snapshot
    snapshot_hash: str     # content address of the raw bytes


class ElementSetAtEpoch(BaseModel):
    """One GP/TLE orbital state at its epoch, normalized across feeds."""

    norad_cat_id: int = Field(gt=0)
    object_name: str
    epoch: datetime                            # the element-set epoch (NOT fetch_time)
    mean_motion: float = Field(gt=0)           # revs/day
    eccentricity: float = Field(ge=0, lt=1)
    inclination: float = Field(ge=0, le=180)   # degrees
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_models.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/models.py data/tests/test_models.py
git commit -m "feat(data): pydantic per-record contracts with provenance stamps"
```

---

### Task 6: pandera per-frame gates

**Files:**
- Create: `data/src/sda_data/gates.py`
- Test: `data/tests/test_gates.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_gates.py`:

```python
import pandas as pd
import pandera.errors
import pytest

from sda_data.gates import ElementSetFrame


def good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "norad_cat_id": [25544, 48274],
            "object_name": ["ISS (ZARYA)", "CSS (TIANHE)"],
            "mean_motion": [15.50, 15.60],
            "eccentricity": [0.0003, 0.0004],
            "inclination": [51.64, 41.47],
            "source": ["celestrak", "celestrak"],
            "snapshot_hash": ["sha256:" + "0" * 64] * 2,
        }
    )


def test_valid_frame_passes():
    out = ElementSetFrame.validate(good_frame())
    assert len(out) == 2


def test_duplicate_norad_id_rejected():
    df = good_frame()
    df.loc[1, "norad_cat_id"] = 25544
    with pytest.raises(pandera.errors.SchemaError):
        ElementSetFrame.validate(df)


def test_impossible_mean_motion_rejected():
    df = good_frame()
    df.loc[0, "mean_motion"] = 25.0  # > 20 revs/day is garbage
    with pytest.raises(pandera.errors.SchemaError):
        ElementSetFrame.validate(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_gates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.gates'`

- [ ] **Step 3: Write `data/src/sda_data/gates.py`**

```python
"""Per-frame quality gate. Catches what individually-valid records hide."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class ElementSetFrame(pa.DataFrameModel):
    norad_cat_id: Series[int] = pa.Field(gt=0, unique=True)   # no dup objects per snapshot
    object_name: Series[str] = pa.Field(nullable=False)
    mean_motion: Series[float] = pa.Field(gt=0, le=20)        # revs/day; >20 is garbage
    eccentricity: Series[float] = pa.Field(ge=0, lt=1)        # bound orbits only
    inclination: Series[float] = pa.Field(ge=0, le=180)
    source: Series[str] = pa.Field(isin=["celestrak", "spacedevs"])
    snapshot_hash: Series[str] = pa.Field(str_startswith="sha256:")

    class Config:
        strict = False   # extra provenance columns are allowed
        coerce = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_gates.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/gates.py data/tests/test_gates.py
git commit -m "feat(data): pandera physics gates for element-set frames"
```

---

### Task 7: Delta silver appender

**Files:**
- Create: `data/src/sda_data/silver.py`
- Test: `data/tests/test_silver.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_silver.py`:

```python
import pandas as pd
from deltalake import DeltaTable

from sda_data import config
from sda_data.silver import append_element_sets, element_sets_uri


def frame(norads):
    return pd.DataFrame(
        {
            "norad_cat_id": norads,
            "object_name": [f"OBJ-{n}" for n in norads],
            "mean_motion": [15.5] * len(norads),
            "eccentricity": [0.001] * len(norads),
            "inclination": [51.6] * len(norads),
            "source": ["celestrak"] * len(norads),
            "snapshot_hash": ["sha256:" + "0" * 64] * len(norads),
        }
    )


def test_uri_defaults_to_local_delta_tier(tmp_data_home):
    assert element_sets_uri() == str(config.delta_root() / "element_sets")


def test_append_creates_then_appends(tmp_data_home):
    append_element_sets(frame([1, 2]))
    append_element_sets(frame([3]))
    table = DeltaTable(element_sets_uri())
    assert table.version() == 1                     # two commits: v0 create, v1 append
    assert len(table.to_pandas()) == 3              # appends never overwrite


def test_append_returns_new_version(tmp_data_home):
    v0 = append_element_sets(frame([1]))
    v1 = append_element_sets(frame([2]))
    assert (v0, v1) == (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_silver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.silver'`

- [ ] **Step 3: Write `data/src/sda_data/silver.py`**

```python
"""Serving layer: Delta appends via delta-rs. One writer per table (enforced
by DAG design: each table is written by exactly one task, max_active_runs=1).
This layer is rebuildable from the frozen snapshots; it is NEVER the
provenance authority (book ch. 3.9 serving-layer invariants)."""
from __future__ import annotations

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from sda_data import config


def element_sets_uri() -> str:
    """Local path by default; set SDA_DELTA_URI-style env later for s3://."""
    return str(config.delta_root() / "element_sets")


def append_element_sets(df: pd.DataFrame) -> int:
    """Append one gated frame; returns the new table version."""
    uri = element_sets_uri()
    write_deltalake(
        uri,
        df,
        mode="append",
        storage_options=config.s3_storage_options() or None,
    )
    return DeltaTable(uri, storage_options=config.s3_storage_options() or None).version()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_silver.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/silver.py data/tests/test_silver.py
git commit -m "feat(data): delta-rs append writer for the element_sets silver table"
```

---

### Task 8: DuckDB query path over Delta

**Files:**
- Create: `data/src/sda_data/query.py`
- Test: `data/tests/test_query.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_query.py`:

```python
import pandas as pd

from sda_data.query import latest_element_sets
from sda_data.silver import append_element_sets


def frame(norads, snapshot_hash):
    return pd.DataFrame(
        {
            "norad_cat_id": norads,
            "object_name": [f"OBJ-{n}" for n in norads],
            "mean_motion": [15.5] * len(norads),
            "eccentricity": [0.001] * len(norads),
            "inclination": [51.6] * len(norads),
            "source": ["celestrak"] * len(norads),
            "snapshot_hash": [snapshot_hash] * len(norads),
            "fetch_time": pd.to_datetime(["2026-07-24"] * len(norads), utc=True),
        }
    )


def test_latest_wins_per_object(tmp_data_home):
    append_element_sets(frame([1, 2], "sha256:" + "a" * 64))
    newer = frame([2], "sha256:" + "b" * 64)
    newer["fetch_time"] = pd.to_datetime(["2026-07-25"], utc=True)
    append_element_sets(newer)

    df = latest_element_sets()
    assert len(df) == 2                                    # one row per object
    row2 = df[df.norad_cat_id == 2].iloc[0]
    assert row2.snapshot_hash.startswith("sha256:b")       # newest fetch wins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd data && uv run pytest tests/test_query.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.query'`

- [ ] **Step 3: Write `data/src/sda_data/query.py`**

```python
"""Query the Delta serving layer with DuckDB. The table arrives as Arrow via
delta-rs (no network extension install), so this works identically against a
local path or MinIO."""
from __future__ import annotations

import duckdb
import pandas as pd
from deltalake import DeltaTable

from sda_data import config
from sda_data.silver import element_sets_uri


def latest_element_sets() -> pd.DataFrame:
    """One row per object: the element set from the newest fetch."""
    arrow = DeltaTable(
        element_sets_uri(), storage_options=config.s3_storage_options() or None
    ).to_pyarrow_table()
    con = duckdb.connect()
    con.register("element_sets", arrow)
    return con.sql(
        """
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY norad_cat_id ORDER BY fetch_time DESC
            ) AS rn
            FROM element_sets
        ) WHERE rn = 1
        """
    ).df()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd data && uv run pytest tests/test_query.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/query.py data/tests/test_query.py
git commit -m "feat(data): duckdb latest-elements query over the delta table"
```

---

### Task 9: celestrak TLE task module (end-to-end)

**Files:**
- Create: `data/src/sda_data/tasks/celestrak_tle.py`
- Test: `data/tests/test_celestrak_tle.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_celestrak_tle.py`:

```python
import json

import httpx
from deltalake import DeltaTable

from sda_data import config
from sda_data.ratelimit import RateLimitedClient, TokenBucket
from sda_data.silver import element_sets_uri
from sda_data.tasks.celestrak_tle import run

GP_PAYLOAD = [
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
    },
    {
        "NORAD_CAT_ID": 48274,
        "OBJECT_NAME": "CSS (TIANHE)",
        "EPOCH": "2026-07-24T05:00:00",
        "MEAN_MOTION": 15.60,
        "ECCENTRICITY": 0.0004,
        "INCLINATION": 41.47,
        "RA_OF_ASC_NODE": 100.0,
        "ARG_OF_PERICENTER": 20.0,
        "MEAN_ANOMALY": 200.0,
        "BSTAR": 0.00010,
    },
]


def fake_client() -> RateLimitedClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["GROUP"] == "active"
        return httpx.Response(200, content=json.dumps(GP_PAYLOAD).encode())

    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )


def test_run_freezes_gates_and_serves(tmp_data_home):
    snapshot_hash = run("active", client=fake_client())

    # 1. Raw snapshot frozen on the shippable tier, named by its own hash.
    digest = snapshot_hash.split(":", 1)[1]
    raws = list(config.raw_root().rglob(f"{digest}.raw"))
    assert len(raws) == 1

    # 2. Normalized Parquet written under the snapshot tier.
    parquets = list(config.snap_root().rglob("*.parquet"))
    assert len(parquets) == 1

    # 3. Delta silver table got the append with provenance intact.
    table = DeltaTable(element_sets_uri()).to_pandas()
    assert set(table.norad_cat_id) == {25544, 48274}
    assert (table.snapshot_hash == snapshot_hash).all()


def test_run_twice_appends_new_snapshot(tmp_data_home):
    run("active", client=fake_client())
    run("active", client=fake_client())
    table = DeltaTable(element_sets_uri())
    assert table.version() == 1          # v0 create + v1 append
    assert len(table.to_pandas()) == 4   # 2 objects x 2 fetches
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_celestrak_tle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.tasks.celestrak_tle'`

- [ ] **Step 3: Write `data/src/sda_data/tasks/celestrak_tle.py`**

```python
"""Fetch one epoch of CelesTrak GP element sets: fetch -> freeze -> gate ->
Parquet -> Delta append.

Standalone (the DAG only schedules this):
    uv run python -m sda_data.tasks.celestrak_tle --group active
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from sda_data import config
from sda_data.gates import ElementSetFrame
from sda_data.models import Provenance, from_celestrak_gp
from sda_data.ratelimit import RateLimitedClient, TokenBucket
from sda_data.silver import append_element_sets
from sda_data.snapshot import write_raw_snapshot

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"


def default_client() -> RateLimitedClient:
    # celestrak asks for infrequent, cached group pulls. The DAG runs 6-hourly;
    # this cap (1 req / 30s, burst 2) is belt-and-suspenders for manual runs.
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=1 / 30, capacity=2),
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


def run(group: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    fetch_time = datetime.now(timezone.utc)

    # 1. Fetch raw bytes and snapshot them IMMUTABLY, content-addressed.
    raw = client.get(CELESTRAK_GP, params={"GROUP": group, "FORMAT": "json"}).content
    raw_path, snapshot_hash = write_raw_snapshot("celestrak", f"gp-{group}", raw, fetch_time)
    prov = Provenance(source="celestrak", fetch_time=fetch_time, snapshot_hash=snapshot_hash)

    # 2. Normalize per record (Pydantic), then gate the frame (pandera).
    rows = [from_celestrak_gp(rec, prov).model_dump() for rec in json.loads(raw)]
    df = pd.json_normalize(rows).rename(
        columns={
            "prov.source": "source",
            "prov.fetch_time": "fetch_time",
            "prov.snapshot_hash": "snapshot_hash",
        }
    )
    df = ElementSetFrame.validate(df)

    # 3. Normalized Parquet snapshot (the citable derived view).
    digest = snapshot_hash.split(":", 1)[1][:16]
    out_dir = config.snap_root() / "celestrak" / "element_sets" / fetch_time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{digest}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    # 4. Delta append (the queryable serving view; rebuildable, not citable).
    version = append_element_sets(df)

    print(
        f"celestrak gp-{group}: {len(df)} element sets  raw={raw_path}  "
        f"snapshot={snapshot_hash[:19]}...  parquet={parquet_path}  delta=v{version}"
    )
    return snapshot_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="active", help="CelesTrak GROUP, e.g. active, stations")
    args = ap.parse_args()
    run(args.group)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_celestrak_tle.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/tasks/celestrak_tle.py data/tests/test_celestrak_tle.py
git commit -m "feat(data): celestrak TLE task, fetch to freeze to delta end-to-end"
```

---

### Task 10: Doctor — environment sanity checks

**Files:**
- Create: `data/src/sda_data/doctor.py`
- Test: `data/tests/test_doctor.py`

- [ ] **Step 1: Write the failing tests**

`data/tests/test_doctor.py`:

```python
from sda_data.doctor import check_environment


def test_reports_missing_s3_config(tmp_data_home, monkeypatch):
    for var in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    issues = check_environment()
    assert any("S3_ENDPOINT_URL" in issue for issue in issues)


def test_clean_when_configured(tmp_data_home, monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    assert check_environment() == []


def test_reports_unwritable_data_home(monkeypatch):
    monkeypatch.setenv("SDA_DATA_HOME", "/proc/definitely-not-writable")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    issues = check_environment()
    assert any("SDA_DATA_HOME" in issue for issue in issues)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd data && uv run pytest tests/test_doctor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sda_data.doctor'`

- [ ] **Step 3: Write `data/src/sda_data/doctor.py`**

```python
"""`make doctor`: is this machine set up to run the pipeline?
Checks are pure and fast; printing happens only in main()."""
from __future__ import annotations

import os

from sda_data import config


def check_environment() -> list[str]:
    """Return human-readable issues; empty list means healthy."""
    issues: list[str] = []

    for var in ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            issues.append(f"{var} is not set (durable tier unreachable; see .env.example)")

    home = config.data_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor-probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        issues.append(f"SDA_DATA_HOME {home} is not writable: {exc}")

    return issues


def main() -> None:
    issues = check_environment()
    if not issues:
        print("doctor: all checks passed")
        raise SystemExit(0)
    for issue in issues:
        print(f"doctor: FAIL  {issue}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd data && uv run pytest tests/test_doctor.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add data/src/sda_data/doctor.py data/tests/test_doctor.py
git commit -m "feat(data): doctor environment checks"
```

---

### Task 11: Airflow compose, thin DAG, env template, Makefile, backup script

No unit tests here: the DAG and compose only run inside Docker (Airflow is
deliberately absent from the dev venv), and the backup script is verified with
`bash -n` + a dry-run default. Manual verification steps are listed.

**Files:**
- Create: `data/docker-compose.airflow.yml`
- Create: `data/dags/sda_ingest.py`
- Create: `data/.env.example`
- Create: `data/Makefile`
- Create: `data/scripts/backup_offsite.sh`
- Create: `data/.gitignore`

- [ ] **Step 1: Write `data/docker-compose.airflow.yml`** (the corrected book stack: four Airflow services)

```yaml
# Single-node Airflow 3, LocalExecutor. In Airflow 3 the scheduler does NOT
# parse DAGs: the dag-processor service is REQUIRED or the DAG list stays empty.
x-airflow-common: &airflow-common
  image: apache/airflow:3.0.0
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-db/airflow
    AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS: "admin:admin"
    SDA_DATA_HOME: /opt/project/data
    S3_ENDPOINT_URL: ${S3_ENDPOINT_URL:-}
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:-}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:-}
  volumes:
    - ./dags:/opt/airflow/dags
    - ..:/opt/project
  depends_on: [airflow-db]

services:
  airflow-db:
    image: postgres:16          # Airflow METADATA only, never our data
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - airflow-db-data:/var/lib/postgresql/data

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command: -c "set -euo pipefail; airflow db migrate"

  airflow-apiserver:
    <<: *airflow-common
    command: api-server
    ports: ["127.0.0.1:8080:8080"]
    restart: unless-stopped

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    restart: unless-stopped

  airflow-dag-processor:        # REQUIRED in Airflow 3
    <<: *airflow-common
    command: dag-processor
    restart: unless-stopped

  airflow-triggerer:            # deferrable operators + event-driven scheduling
    <<: *airflow-common
    command: triggerer
    restart: unless-stopped

volumes:
  airflow-db-data:
```

- [ ] **Step 2: Write `data/dags/sda_ingest.py`** (thin producer DAG)

```python
"""Thin producer DAG: schedule the standalone task modules, declare assets.
No fetching, parsing, or validating happens HERE; it happens in sda_data.tasks."""
from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE_SNAPSHOT = Asset("sda://celestrak/element_sets")

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
```

- [ ] **Step 3: Write `data/.env.example`**

```bash
# Copy to .env (git-ignored) and fill in. compose reads it automatically.

# Working tier root. Inside compose this is /opt/project/data (set there).
# SDA_DATA_HOME=/data/sda

# Durable tier: the single MinIO instance on the NAS (book ch. 0.5).
S3_ENDPOINT_URL=https://s3.datadazed.com
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Offsite backup remote for scripts/backup_offsite.sh (an rclone remote name).
# BACKUP_REMOTE=b2-offsite:sda-backup
```

- [ ] **Step 4: Write `data/Makefile`**

```makefile
.PHONY: test up down doctor backup

test:
	uv run pytest

doctor:
	uv run python -m sda_data.doctor

up:
	docker compose -f docker-compose.airflow.yml up airflow-init
	docker compose -f docker-compose.airflow.yml up -d

down:
	docker compose -f docker-compose.airflow.yml down

backup:
	./scripts/backup_offsite.sh
```

- [ ] **Step 5: Write `data/scripts/backup_offsite.sh`**

```bash
#!/usr/bin/env bash
# Offsite backup of the buckets whose loss would kill thesis reproducibility:
# bronze snapshots and the DVC remote. Delta/Lance are rebuildable and are
# deliberately NOT backed up. Dry-run by default; APPLY=1 to sync for real.
set -euo pipefail

: "${BACKUP_REMOTE:?set BACKUP_REMOTE (an rclone remote:bucket) in .env}"
: "${S3_ENDPOINT_URL:?set S3_ENDPOINT_URL in .env}"

FLAGS=(--s3-endpoint "$S3_ENDPOINT_URL" --checksum)
if [ "${APPLY:-0}" != "1" ]; then
  echo "dry-run (set APPLY=1 to sync for real)"
  FLAGS+=(--dry-run)
fi

for bucket in sda-bronze sda-dvc; do
  echo "== $bucket -> $BACKUP_REMOTE/$bucket"
  rclone sync ":s3:$bucket" "$BACKUP_REMOTE/$bucket" "${FLAGS[@]}"
done
```

- [ ] **Step 6: Write `data/.gitignore`**

```
.env
data/
__pycache__/
.pytest_cache/
```

- [ ] **Step 7: Verify**

Run: `bash -n data/scripts/backup_offsite.sh && chmod +x data/scripts/backup_offsite.sh`
Expected: no output (syntax OK)

Run: `cd data && uv run pytest`
Expected: all tests PASS

Manual verification (on the real machine, not CI): `make up`, log into
`http://127.0.0.1:8080`, confirm `sda_ingest` appears in the DAG list (this is
the dag-processor working) and that a manual trigger runs `celestrak_tle`.

- [ ] **Step 8: Commit**

```bash
git add data/docker-compose.airflow.yml data/dags data/.env.example data/Makefile data/scripts data/.gitignore
git commit -m "feat(data): airflow 3 compose stack, thin ingest DAG, doctor/backup tooling"
```

---

### Task 12: Full suite, push, PR

- [ ] **Step 1: Run the whole suite**

Run: `cd data && uv run pytest -v`
Expected: all tests PASS (~24 tests)

- [ ] **Step 2: Push and open a draft PR**

```bash
git push -u origin claude/mdbook-evals-as-rewards-spec-fvhxw7
```

Open a draft PR titled "SDA data platform foundation: config, freeze layer, celestrak ingest, Delta silver, Airflow 3 stack" describing the slice and noting the follow-up plans (dlt sources, dbt gold, Lance/RAG, MCP).
