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
