# SDA Gold Layer (dbt-duckdb) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dbt-duckdb project that builds tested gold tables (latest elements, upcoming launches, latest articles, NEO approaches) from the frozen Parquet snapshot tier into a DuckDB gold database, plus the Airflow consumer DAG that rebuilds gold when any source asset updates.

**Architecture:** dbt owns the SQL transforms and their tests; it reads the *snapshot Parquet tier* via `read_parquet(..., union_by_name=true)` globs (the citable tier, same rows the Delta appends carry) and materializes tables into `$SDA_DATA_HOME/data/gold/sda.duckdb`. This avoids both the network-installed DuckDB delta extension and plugin dependencies. Embargoed spacetrack data is deliberately absent from every model. The thin task module `tasks/dbt_build.py` wraps the dbt CLI so the DAG stays a BashOperator and the build runs standalone.

**Tech Stack:** adds `dbt-duckdb` (brings dbt-core). Verification: a pytest integration test seeds all four snapshot tiers, runs `dbt build` (which also runs the schema tests), and asserts the gold tables.

---

## File Structure

```
data/dbt/dbt_project.yml
data/dbt/profiles.yml
data/dbt/models/latest_element_sets.sql
data/dbt/models/upcoming_launches.sql
data/dbt/models/latest_articles.sql
data/dbt/models/neo_approaches.sql
data/dbt/models/schema.yml
data/src/sda_data/tasks/dbt_build.py
data/dags/sda_gold.py
data/tests/test_dbt_gold.py
```

### Task 1: dbt project + models

- [ ] Add dep: `cd data && uv add "dbt-duckdb>=1.8"`

- [ ] `data/dbt/dbt_project.yml`:

```yaml
name: sda_gold
version: "0.1.0"
profile: sda
model-paths: ["models"]
models:
  sda_gold:
    +materialized: table
```

- [ ] `data/dbt/profiles.yml` (env-driven; `SDA_DATA_HOME` is set by the task module):

```yaml
sda:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{{ env_var('SDA_DATA_HOME') }}/data/gold/sda.duckdb"
```

- [ ] `data/dbt/models/latest_element_sets.sql`:

```sql
-- Latest element set per object from the shippable celestrak snapshot tier.
-- Embargoed spacetrack data is deliberately absent from gold.
with all_sets as (
    select *
    from read_parquet(
        '{{ env_var("SDA_DATA_HOME") }}/data/snapshots/celestrak/element_sets/**/*.parquet',
        union_by_name = true
    )
)
select * exclude (rn)
from (
    select *,
           row_number() over (partition by norad_cat_id order by fetch_time desc) as rn
    from all_sets
)
where rn = 1
```

- [ ] `data/dbt/models/upcoming_launches.sql`:

```sql
-- Deduped launches still in the future (net = no-earlier-than).
with all_launches as (
    select *
    from read_parquet(
        '{{ env_var("SDA_DATA_HOME") }}/data/snapshots/spacedevs/launches/**/*.parquet',
        union_by_name = true
    )
),
latest as (
    select * exclude (rn)
    from (
        select *,
               row_number() over (partition by id order by fetch_time desc) as rn
        from all_launches
    )
    where rn = 1
)
select *
from latest
where try_cast(net as timestamptz) >= now()
```

- [ ] `data/dbt/models/latest_articles.sql`:

```sql
-- Deduped article metadata: the queryable face of the 8.1 RAG text branch.
select * exclude (rn)
from (
    select *,
           row_number() over (partition by id order by fetch_time desc) as rn
    from read_parquet(
        '{{ env_var("SDA_DATA_HOME") }}/data/snapshots/spaceflightnews/articles/**/*.parquet',
        union_by_name = true
    )
)
where rn = 1
```

- [ ] `data/dbt/models/neo_approaches.sql`:

```sql
-- Deduped NASA NeoWs close approaches (derived facts; raw dumps stay embargoed).
select * exclude (rn)
from (
    select *,
           row_number() over (
               partition by neo_reference_id, close_approach_date
               order by fetch_time desc
           ) as rn
    from read_parquet(
        '{{ env_var("SDA_DATA_HOME") }}/data/snapshots/nasa/neows/**/*.parquet',
        union_by_name = true
    )
)
where rn = 1
```

- [ ] `data/dbt/models/schema.yml`:

```yaml
version: 2
models:
  - name: latest_element_sets
    description: One row per catalogued object, newest fetch wins. celestrak only.
    columns:
      - name: norad_cat_id
        tests: [unique, not_null]
      - name: snapshot_hash
        tests: [not_null]
  - name: upcoming_launches
    columns:
      - name: id
        tests: [unique, not_null]
  - name: latest_articles
    columns:
      - name: id
        tests: [unique, not_null]
  - name: neo_approaches
    columns:
      - name: neo_reference_id
        tests: [not_null]
```

### Task 2: thin build task + integration test

- [ ] `data/src/sda_data/tasks/dbt_build.py`:

```python
"""Run the dbt gold build. Thin: dbt owns the SQL and its tests; this module
only locates the project, guarantees the gold dir exists, and shells out.

Standalone:
    uv run python -m sda_data.tasks.dbt_build
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sda_data import config

PROJECT_DIR = Path(__file__).resolve().parents[3] / "dbt"


def run() -> None:
    gold_dir = config.data_home() / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("SDA_DATA_HOME", str(config.data_home()))
    subprocess.run(
        ["dbt", "build", "--project-dir", str(PROJECT_DIR),
         "--profiles-dir", str(PROJECT_DIR)],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    run()
```

- [ ] `data/tests/test_dbt_gold.py` — seeds all four snapshot tiers with freeze-shaped parquet, runs `dbt build` (schema tests included), asserts gold:

```python
from datetime import datetime, timezone

import duckdb
import pandas as pd

from sda_data import config
from sda_data.tasks.dbt_build import run

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def seed(subpath: str, df: pd.DataFrame) -> None:
    out = config.snap_root() / subpath
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def seed_all(tmp_data_home) -> None:
    t1 = pd.Timestamp("2026-07-24", tz="UTC")
    t2 = pd.Timestamp("2026-07-25", tz="UTC")
    seed("celestrak/element_sets/2026-07-24/a.parquet", pd.DataFrame({
        "norad_cat_id": [25544, 48274], "object_name": ["ISS", "CSS"],
        "mean_motion": [15.5, 15.6], "eccentricity": [0.0003, 0.0004],
        "inclination": [51.6, 41.5], "source": ["celestrak"] * 2,
        "snapshot_hash": [HASH_A] * 2, "fetch_time": [t1] * 2,
    }))
    seed("celestrak/element_sets/2026-07-25/b.parquet", pd.DataFrame({
        "norad_cat_id": [25544], "object_name": ["ISS"],
        "mean_motion": [15.51], "eccentricity": [0.0003],
        "inclination": [51.6], "source": ["celestrak"],
        "snapshot_hash": [HASH_B], "fetch_time": [t2],
    }))
    seed("spacedevs/launches/2026-07-24/a.parquet", pd.DataFrame({
        "id": ["u-1", "u-2"], "name": ["A", "B"],
        "net": ["2126-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],  # one future, one past
        "status_name": ["Go", "Success"], "provider": ["X", "Y"],
        "source": ["spacedevs"] * 2, "snapshot_hash": [HASH_A] * 2,
        "fetch_time": [t1] * 2,
    }))
    seed("spaceflightnews/articles/2026-07-24/a.parquet", pd.DataFrame({
        "id": [101], "title": ["T"], "url": ["u"], "news_site": ["N"],
        "summary": ["s"], "published_at": ["2026-07-24T10:00:00Z"],
        "source": ["spaceflightnews"], "snapshot_hash": [HASH_A],
        "fetch_time": [t1],
    }))
    seed("nasa/neows/2026-07-24/a.parquet", pd.DataFrame({
        "neo_reference_id": ["354"], "name": ["PK9"],
        "close_approach_date": ["2026-07-25"], "miss_distance_km": [1.2e6],
        "rel_velocity_kms": [14.2], "absolute_magnitude_h": [21.8],
        "source": ["nasa"], "snapshot_hash": [HASH_A], "fetch_time": [t1],
    }))


def test_dbt_build_produces_gold(tmp_data_home):
    seed_all(tmp_data_home)
    run()   # dbt build: models + schema tests; raises on any failure

    con = duckdb.connect(str(config.data_home() / "data" / "gold" / "sda.duckdb"))
    latest = con.sql("select * from latest_element_sets order by norad_cat_id").df()
    assert len(latest) == 2                                   # one row per object
    assert latest[latest.norad_cat_id == 25544].iloc[0].snapshot_hash == HASH_B

    launches = con.sql("select * from upcoming_launches").df()
    assert list(launches.id) == ["u-1"]                       # past launch filtered out

    assert con.sql("select count(*) c from latest_articles").fetchone()[0] == 1
    assert con.sql("select count(*) c from neo_approaches").fetchone()[0] == 1
```

### Task 3: consumer DAG + commit

- [ ] `data/dags/sda_gold.py` — rebuilds when ANY source asset updates (OR-composition):

```python
"""Data-aware consumer DAG: rebuild gold when any snapshot asset updates."""
from __future__ import annotations

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE = Asset("sda://celestrak/element_sets")
NEOWS = Asset("sda://nasa/neows")
ARTICLES = Asset("sda://spaceflightnews/articles")
LAUNCHES = Asset("sda://spacedevs/launches")
GOLD = Asset("sda://gold/tables")

with DAG(
    dag_id="sda_gold",
    schedule=(TLE | NEOWS | ARTICLES | LAUNCHES),   # any-of, no clock
    catchup=False,
    max_active_runs=1,
    tags=["sda", "gold"],
) as dag:
    build = BashOperator(
        task_id="dbt_build",
        bash_command="cd /opt/project/data && uv run python -m sda_data.tasks.dbt_build",
        outlets=[GOLD],
    )
```

- [ ] Run `cd data && uv run pytest tests/test_dbt_gold.py -v` → PASS, then full suite, commit:

```bash
git add data/dbt data/src/sda_data/tasks/dbt_build.py data/dags/sda_gold.py data/tests/test_dbt_gold.py data/pyproject.toml data/uv.lock
git commit -m "feat(data): dbt-duckdb gold layer with tested models and asset-triggered DAG"
```
