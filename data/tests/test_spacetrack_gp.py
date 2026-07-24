import json

import pandas as pd

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
    snapshot_hash = run(fetch=lambda: json.dumps(GP).encode())
    df = pd.read_parquet(next(config.live_root().rglob("*.parquet")))
    assert (df.snapshot_hash == snapshot_hash).all()
    assert (df.source == "spacetrack").all()
