import json

import httpx
from deltalake import DeltaTable

from sda_data import config
from sda_data.ratelimit import RateLimitedClient, TokenBucket
from sda_data.silver import table_uri
from sda_data.tasks.spaceflightnews import run

PAGE1 = {
    "next": "https://api.spaceflightnewsapi.net/v4/articles/?offset=1",
    "results": [
        {
            "id": 101,
            "title": "ISS debris avoidance maneuver",
            "url": "https://example.com/a/101",
            "news_site": "NASA",
            "summary": "The ISS moved.",
            "published_at": "2026-07-24T10:00:00Z",
        }
    ],
}
PAGE2 = {
    "next": None,
    "results": [
        {
            "id": 102,
            "title": "Starship launch window",
            "url": "https://example.com/a/102",
            "news_site": "SpaceNews",
            "summary": "A window opens.",
            "published_at": "2026-07-24T11:00:00Z",
        }
    ],
}


def fake_client() -> RateLimitedClient:
    def handler(request: httpx.Request) -> httpx.Response:
        page = PAGE2 if request.url.params.get("offset") == "1" else PAGE1
        return httpx.Response(200, content=json.dumps(page).encode())

    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=100.0, capacity=100),
        transport=httpx.MockTransport(handler),
        user_agent="sda-test/0.1",
    )


def test_run_follows_pagination_and_serves(tmp_data_home):
    snapshot_hash = run("2026-07-24", client=fake_client())

    # Articles are redistributable: raw snapshot on the shippable tier.
    digest = snapshot_hash.split(":", 1)[1]
    assert list(config.raw_root().rglob(f"{digest}.raw"))

    table = DeltaTable(table_uri("articles")).to_pandas()
    assert set(table.id) == {101, 102}              # both pages landed
    assert (table.source == "spaceflightnews").all()
