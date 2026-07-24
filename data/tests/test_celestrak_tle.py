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
