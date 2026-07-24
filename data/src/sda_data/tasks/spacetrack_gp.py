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
