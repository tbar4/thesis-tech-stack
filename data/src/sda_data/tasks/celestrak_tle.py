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
