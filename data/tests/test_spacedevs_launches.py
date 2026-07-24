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
