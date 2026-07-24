"""Spaceflight News (SNAPI v4) articles: the text branch feeding the 8.1 RAG
corpus. Same shape as nasa_neows: dlt lands, freeze_landing freezes.

Standalone:
    uv run python -m sda_data.tasks.spaceflightnews --since 2026-07-01
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import dlt

from sda_data import config
from sda_data.freeze import freeze_landing
from sda_data.gates import ArticleFrame
from sda_data.ratelimit import RateLimitedClient, TokenBucket

SNAPI = "https://api.spaceflightnewsapi.net/v4/articles/"


def default_client() -> RateLimitedClient:
    return RateLimitedClient(
        bucket=TokenBucket(rate_per_sec=1.0, capacity=5),
        user_agent="sda-thesis-pipeline/0.1 (+research)",
    )


@dlt.resource(name="articles", write_disposition="replace")
def articles_feed(client: RateLimitedClient, since: str):
    url: str | None = SNAPI
    params: dict | None = {"published_at_gte": since, "limit": 100, "ordering": "published_at"}
    while url:
        page = client.get(url, params=params).json()
        params = None                     # SNAPI next links carry their own query string
        for a in page["results"]:
            yield {
                "id": a["id"],
                "title": a["title"],
                "url": a["url"],
                "news_site": a["news_site"],
                "summary": a.get("summary") or "",
                "published_at": a["published_at"],
            }
        url = page.get("next")


def extract_to_landing(client: RateLimitedClient, since: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    landing = config.snap_root() / "_landing" / "spaceflightnews" / run_id
    pipe = dlt.pipeline(
        pipeline_name="spaceflightnews",
        pipelines_dir=str(config.data_home() / ".dlt"),
        destination=dlt.destinations.filesystem(bucket_url=str(landing)),
        dataset_name="sfn",
    )
    pipe.run(articles_feed(client, since), loader_file_format="parquet")
    return landing


def run(since: str, client: RateLimitedClient | None = None) -> str:
    client = client or default_client()
    landing = extract_to_landing(client, since)
    return freeze_landing(
        landing, source="spaceflightnews", kind="articles",
        gate=ArticleFrame, delta_table="articles",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="published_at lower bound, e.g. 2026-07-01")
    args = ap.parse_args()
    run(args.since)


if __name__ == "__main__":
    main()
