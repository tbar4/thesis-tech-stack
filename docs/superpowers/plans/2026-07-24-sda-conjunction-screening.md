# SDA Conjunction Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close-approach screening over the latest element sets: an apogee/perigee band sieve prunes the O(n²) pair space, SGP4 propagates the survivors over a screening window, and detected close approaches append to a `close_approaches` Delta table via an asset-triggered DAG.

**Architecture:** Physics lives in `sda_data/conjunctions.py` (pure, testable: elements → Satrec, radii bands, sieve, cached-position min-distance search with a fine refinement pass). The thin task `tasks/build_conjunctions.py` reads `latest_element_sets()` from the Delta serving layer, screens, and appends. The sieve is the load-bearing efficiency move (book 3.9 adversarial review): band-overlap filtering kills ~95% of pairs before any propagation. Positions are propagated once per *object* over the shared time grid and cached, so pair checks are pure numpy distance math.

**Tech Stack:** adds `sgp4` and an explicit `numpy` dependency.

**Key numerics:** semi-major axis from mean motion (a = (μ/n²)^⅓, μ = 398600.4418 km³/s²); perigee/apogee radii rp = a(1−e), ra = a(1+e); a pair survives the sieve iff its radial bands, padded by `sieve_pad_km`, overlap. SGP4 init: `sgp4init(WGS72, 'i', satnum, jd_epoch − 2433281.5, bstar, 0, 0, ecc, argp_rad, incl_rad, M_rad, n_rad_per_min, raan_rad)`.

---

## File Structure

```
data/src/sda_data/conjunctions.py        # physics: satrec_from_elements, radii, sieve, screen
data/src/sda_data/tasks/build_conjunctions.py  # thin: delta -> screen -> delta append
data/dags/sda_conjunctions.py            # consumer DAG on the TLE asset
data/tests/test_conjunctions.py
```

### Task 1: physics module + tests

- [ ] Add deps: `cd data && uv add "sgp4>=2.23" "numpy>=1.26"`

- [ ] `data/tests/test_conjunctions.py`:

```python
from datetime import datetime, timezone

import pandas as pd
import pytest

from sda_data.conjunctions import orbital_radii, screen, sieve_pairs

EPOCH = datetime(2026, 7, 24, 6, 0, tzinfo=timezone.utc)


def element_row(norad, mean_motion=15.5, ecc=0.0003, mean_anomaly=300.0, incl=51.64):
    return {
        "norad_cat_id": norad, "object_name": f"OBJ-{norad}", "epoch": EPOCH,
        "mean_motion": mean_motion, "eccentricity": ecc, "inclination": incl,
        "ra_of_asc_node": 120.0, "arg_of_pericenter": 30.0,
        "mean_anomaly": mean_anomaly, "bstar": 0.0001,
        "snapshot_hash": "sha256:" + "0" * 64,
    }


def test_orbital_radii_iss_like():
    rp, ra = orbital_radii(15.5, 0.0)
    assert rp == ra                                  # circular
    assert 6700 < rp < 6900                          # ~420 km altitude


def test_sieve_prunes_leo_vs_geo():
    df = pd.DataFrame([element_row(1, mean_motion=15.5),
                       element_row(2, mean_motion=1.0027)])   # GEO
    assert sieve_pairs(df, pad_km=25.0) == []


def test_sieve_keeps_coplanar_leo_pair():
    df = pd.DataFrame([element_row(1), element_row(2, mean_anomaly=300.5)])
    assert sieve_pairs(df, pad_km=25.0) == [(0, 1)]


def test_screen_detects_close_phase_shifted_pair():
    # Identical orbit, mean anomaly offset by 0.01 deg: constant along-track
    # separation of roughly (0.01/360) * 2*pi*a ~ 1.2 km. Must be reported.
    df = pd.DataFrame([element_row(1, mean_anomaly=300.00),
                       element_row(2, mean_anomaly=300.01)])
    hits = screen(df, start=EPOCH, window_hours=1.0, miss_threshold_km=25.0)
    assert len(hits) == 1
    hit = hits.iloc[0]
    assert {hit.norad_i, hit.norad_j} == {1, 2}
    assert hit.miss_km < 5.0


def test_screen_ignores_far_pair():
    # Same orbit, opposite phase: separation ~ 2a, never close.
    df = pd.DataFrame([element_row(1, mean_anomaly=0.0),
                       element_row(2, mean_anomaly=180.0)])
    hits = screen(df, start=EPOCH, window_hours=1.0, miss_threshold_km=25.0)
    assert len(hits) == 0
```

- [ ] `data/src/sda_data/conjunctions.py`:

```python
"""Close-approach screening: sieve, then propagate.

All-vs-all conjunction screening is O(n^2) propagations; the standard fix is a
cheap apogee/perigee band sieve first (two objects whose radial bands never
overlap cannot approach), then SGP4 only on survivors. Positions are
propagated once per OBJECT over the shared grid and cached, so each surviving
pair costs numpy distance math, not another propagation."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec, jday

MU_KM3_S2 = 398600.4418          # Earth GM
TWO_PI = 2.0 * math.pi


def orbital_radii(mean_motion_rev_day: float, eccentricity: float) -> tuple[float, float]:
    """(perigee_radius_km, apogee_radius_km) from mean motion + eccentricity."""
    n_rad_s = mean_motion_rev_day * TWO_PI / 86400.0
    a = (MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    return a * (1.0 - eccentricity), a * (1.0 + eccentricity)


def sieve_pairs(df: pd.DataFrame, pad_km: float = 25.0) -> list[tuple[int, int]]:
    """Index pairs whose padded radial bands overlap. Everyone else is pruned
    without a single propagation."""
    bands = [orbital_radii(r.mean_motion, r.eccentricity) for r in df.itertuples()]
    pairs: list[tuple[int, int]] = []
    for i in range(len(bands)):
        rp_i, ra_i = bands[i]
        for j in range(i + 1, len(bands)):
            rp_j, ra_j = bands[j]
            if rp_i - ra_j > pad_km or rp_j - ra_i > pad_km:
                continue                                  # bands can never meet
            pairs.append((i, j))
    return pairs


def satrec_from_elements(row) -> Satrec:
    """Build an SGP4 satrec straight from the normalized element-set schema."""
    epoch: datetime = row.epoch
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                  epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6)
    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", int(row.norad_cat_id), jd + fr - 2433281.5,
        float(getattr(row, "bstar", 0.0) or 0.0), 0.0, 0.0,
        float(row.eccentricity),
        math.radians(row.arg_of_pericenter),
        math.radians(row.inclination),
        math.radians(row.mean_anomaly),
        row.mean_motion * TWO_PI / 1440.0,        # rev/day -> rad/min
        math.radians(row.ra_of_asc_node),
    )
    return sat


def _positions(sat: Satrec, jds: np.ndarray, frs: np.ndarray) -> np.ndarray | None:
    """(T, 3) TEME positions in km, or None if SGP4 errors anywhere."""
    errs, pos, _vel = sat.sgp4_array(jds, frs)
    if np.any(errs != 0):
        return None
    return pos


def screen(
    df: pd.DataFrame,
    start: datetime,
    window_hours: float = 24.0,
    step_s: float = 60.0,
    miss_threshold_km: float = 25.0,
    pad_km: float = 25.0,
) -> pd.DataFrame:
    """Screen all pairs; return close approaches under the threshold.

    Coarse pass on the shared grid finds each pair's minimum-distance sample;
    a fine pass (1 s steps, +-step_s around it) sharpens tca and miss."""
    pairs = sieve_pairs(df, pad_km=pad_km)
    n_steps = int(window_hours * 3600.0 / step_s) + 1
    offsets = np.arange(n_steps) * step_s
    jd0, fr0 = jday(start.year, start.month, start.day,
                    start.hour, start.minute, start.second)
    jds = np.full(n_steps, jd0)
    frs = fr0 + offsets / 86400.0

    sats = [satrec_from_elements(row) for row in df.itertuples()]
    cache: dict[int, np.ndarray | None] = {}

    def positions(idx: int) -> np.ndarray | None:
        if idx not in cache:
            cache[idx] = _positions(sats[idx], jds, frs)
        return cache[idx]

    hits: list[dict] = []
    for i, j in pairs:
        pi, pj = positions(i), positions(j)
        if pi is None or pj is None:
            continue                                  # decayed / bad elements
        dist = np.linalg.norm(pi - pj, axis=1)
        k = int(np.argmin(dist))
        if dist[k] >= miss_threshold_km:
            continue

        # Fine pass: 1 s resolution around the coarse minimum.
        lo = max(0.0, offsets[k] - step_s)
        hi = min(offsets[-1], offsets[k] + step_s)
        fine = np.arange(lo, hi + 1.0)
        fj = np.full(fine.shape, jd0)
        ff = fr0 + fine / 86400.0
        fi_pos = _positions(sats[i], fj, ff)
        fj_pos = _positions(sats[j], fj, ff)
        if fi_pos is None or fj_pos is None:
            continue
        fdist = np.linalg.norm(fi_pos - fj_pos, axis=1)
        m = int(np.argmin(fdist))
        row_i, row_j = df.iloc[i], df.iloc[j]
        hits.append({
            "norad_i": int(row_i.norad_cat_id),
            "norad_j": int(row_j.norad_cat_id),
            "tca": (start + timedelta(seconds=float(fine[m]))).isoformat(),
            "miss_km": float(fdist[m]),
            "snapshot_hash_i": row_i.snapshot_hash,
            "snapshot_hash_j": row_j.snapshot_hash,
        })
    return pd.DataFrame(hits, columns=[
        "norad_i", "norad_j", "tca", "miss_km", "snapshot_hash_i", "snapshot_hash_j",
    ])
```

- [ ] Run: `cd data && uv run pytest tests/test_conjunctions.py -v` → PASS. Commit:

```bash
git add data/src/sda_data/conjunctions.py data/tests/test_conjunctions.py data/pyproject.toml data/uv.lock
git commit -m "feat(data): conjunction screening with apogee/perigee sieve + SGP4"
```

### Task 2: thin task + DAG + test

- [ ] `data/src/sda_data/tasks/build_conjunctions.py`:

```python
"""Screen the latest element sets for close approaches; append to Delta.

Standalone:
    uv run python -m sda_data.tasks.build_conjunctions --window-hours 24
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from sda_data import silver
from sda_data.conjunctions import screen
from sda_data.query import latest_element_sets


def run(window_hours: float = 24.0, miss_threshold_km: float = 25.0) -> int:
    df = latest_element_sets()
    start = datetime.now(timezone.utc)
    hits = screen(df, start=start,
                  window_hours=window_hours, miss_threshold_km=miss_threshold_km)
    hits["screened_at"] = start.isoformat()
    if hits.empty:
        print("conjunctions: no close approaches under threshold")
        return 0
    version = silver.append("close_approaches", hits)
    print(f"conjunctions: {len(hits)} close approaches  delta=v{version}")
    return len(hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--miss-km", type=float, default=25.0)
    args = ap.parse_args()
    run(window_hours=args.window_hours, miss_threshold_km=args.miss_km)


if __name__ == "__main__":
    main()
```

- [ ] Add to `data/tests/test_conjunctions.py`:

```python
def test_build_task_appends_delta(tmp_data_home):
    from deltalake import DeltaTable

    from sda_data.silver import append_element_sets, table_uri
    from sda_data.tasks.build_conjunctions import run as build_run

    rows = [element_row(1, mean_anomaly=300.00), element_row(2, mean_anomaly=300.01)]
    df = pd.DataFrame(rows)
    df["source"] = "celestrak"
    df["fetch_time"] = pd.Timestamp("2026-07-24", tz="UTC")
    append_element_sets(df)

    count = build_run(window_hours=1.0)
    assert count == 1
    table = DeltaTable(table_uri("close_approaches")).to_pandas()
    assert len(table) == 1 and table.iloc[0].miss_km < 5.0
```

Note: `screen()` starts at "now" in `build_run`; the synthetic elements are epoch
2026-07-24 but SGP4 propagates fine past epoch, and an identical-orbit pair keeps
its constant along-track separation at any start time, so the assertion is stable.

- [ ] `data/dags/sda_conjunctions.py`:

```python
"""Data-aware consumer DAG: screen for conjunctions when elements refresh."""
from __future__ import annotations

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

TLE = Asset("sda://celestrak/element_sets")
CLOSE_APPROACHES = Asset("sda://gold/close_approaches")

with DAG(
    dag_id="sda_conjunctions",
    schedule=[TLE],                 # fires on the asset, not a timer
    catchup=False,
    max_active_runs=1,              # one writer for close_approaches
    tags=["sda", "conjunctions"],
) as dag:
    build = BashOperator(
        task_id="build_conjunctions",
        bash_command="cd /opt/project/data && "
                     "uv run python -m sda_data.tasks.build_conjunctions",
        outlets=[CLOSE_APPROACHES],
    )
```

- [ ] Full suite, commit, push:

```bash
cd data && uv run pytest
git add data/src/sda_data/tasks/build_conjunctions.py data/dags/sda_conjunctions.py data/tests/test_conjunctions.py
git commit -m "feat(data): conjunction build task + asset-triggered DAG"
git push -u origin claude/mdbook-evals-as-rewards-spec-fvhxw7
```
