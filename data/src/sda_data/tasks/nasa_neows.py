"""NASA NeoWs close approaches: dlt extract-to-land -> freeze -> delta.

The page loop runs through the shared RateLimitedClient so the hard cap is the
same mechanism as every other source (NASA allows ~1000 req/hr per key). dlt
owns landing (JSON->tables, run-stamped partition) and stops; freeze_landing
does the rest. Raw dumps are embargoed (nasa not in REDISTRIBUTABLE); the
derived facts here are what ships.

Standalone:
    NASA_API_KEY=... uv run python -m sda_data.tasks.nasa_neows --start 2026-07-01 --end 2026-07-02
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import dlt

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import NeoWsFrame
from sda_data.ratelimit import RateLimitedClient, TokenBucket

NEOWS = "https://api.nasa.gov/neo/rest/v1"


def default_client() -> RateLimitedClient:
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=0.25, capacity=4),   # ~900/hr, under the 1000 cap
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


@dlt.resource(name="neo_close_approaches", write_disposition="replace")
def neows_feed(client: RateLimitedClient, start: str, end: str, api_key: str):
    url: str | None = f"{NEOWS}/feed"
    params: dict | None = {"start_date": start, "end_date": end, "api_key": api_key}
    while url:
        page = client.get(url, params=params).json()
        params = None                     # NeoWs next links carry their own query string
        for _day, objects in page["near_earth_objects"].items():
            for o in objects:
                for ca in o["close_approach_data"]:
                    yield {
                        "neo_reference_id": o["neo_reference_id"],
                        "name": o["name"],
                        "close_approach_date": ca["close_approach_date"],
                        "miss_distance_km": float(ca["miss_distance"]["kilometers"]),
                        "rel_velocity_kms": float(
                            ca["relative_velocity"]["kilometers_per_second"]
                        ),
                        "absolute_magnitude_h": float(o["absolute_magnitude_h"]),
                    }
        url = page.get("links", {}).get("next")


def extract_to_landing(client: RateLimitedClient, start: str, end: str, api_key: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    landing = config.snap_root() / "_landing" / "nasa_neows" / run_id
    pipe = dlt.pipeline(
        pipeline_name="nasa_neows",
        pipelines_dir=str(config.data_home() / ".dlt"),
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="neows",
    )
    pipe.run(neows_feed(client, start, end, api_key), loader_file_format="parquet")
    return landing


def run(start: str, end: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    api_key = os.environ["NASA_API_KEY"]
    landing = extract_to_landing(client, start, end, api_key)
    return freeze_landing(
        landing, source="nasa", kind="neows",
        gate=NeoWsFrame, delta_table="neo_close_approaches",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC date, e.g. 2026-07-01")
    ap.add_argument("--end", required=True, help="UTC date, inclusive")
    args = ap.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
