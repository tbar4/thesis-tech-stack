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
