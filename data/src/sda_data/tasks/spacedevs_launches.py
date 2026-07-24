"""thespacedevs Launch Library 2 launches (events feed).

LL2's free tier is HARSHLY throttled (~15 requests/hour): the default bucket
here is the compliance mechanism, and the DAG's 6-hour cadence uses ~1-2
requests per run. dlt lands, freeze_landing freezes.

Standalone:
    uv run python -m sda_data.tasks.spacedevs_launches --since 2026-07-01
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import dlt

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import LaunchFrame
from sda_data.ratelimit import RateLimitedClient, TokenBucket

LL2 = "https://ll.thespacedevs.com/2.2.0/launch/"


def default_client() -> RateLimitedClient:
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=15 / 3600, capacity=2),   # LL2 free tier ~15/hr
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


@dlt.resource(name="launches", write_disposition="replace")
def launches_feed(client: RateLimitedClient, since: str):
    url: str | None = LL2
    params: dict | None = {"net__gte": since, "limit": 100, "ordering": "net"}
    while url:
        page = client.get(url, params=params).json()
        params = None                     # LL2 next links carry their own query string
        for launch in page["results"]:
            yield {
                "id": launch["id"],
                "name": launch["name"],
                "net": launch["net"],
                "status_name": launch["status"]["name"],
                "provider": launch["launch_service_provider"]["name"],
            }
        url = page.get("next")


def extract_to_landing(client: RateLimitedClient, since: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    landing = config.snap_root() / "_landing" / "spacedevs_launches" / run_id
    pipe = dlt.pipeline(
        pipeline_name="spacedevs_launches",
        pipelines_dir=str(config.data_home() / ".dlt"),
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="ll2",
    )
    pipe.run(launches_feed(client, since), loader_file_format="parquet")
    return landing


def run(since: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    landing = extract_to_landing(client, since)
    return freeze_landing(
        landing, source="spacedevs", kind="launches",
        gate=LaunchFrame, delta_table="launches",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="net lower bound, e.g. 2026-07-01")
    args = ap.parse_args()
    run(args.since)


if __name__ == "__main__":
    main()
