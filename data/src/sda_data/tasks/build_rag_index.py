"""Build the RAG index from the articles Delta table.

Requires the `embed` extra (sentence-transformers) at runtime:
    uv sync --extra embed
    uv run python -m sda_data.tasks.build_rag_index --model BAAI/bge-small-en-v1.5
"""
from __future__ import annotations

import argparse

from deltalake import DeltaTable

from sda_data import config, rag
from sda_data.silver import table_uri


def latest_articles():
    df = DeltaTable(
        table_uri("articles"), storage_options=config.s3_storage_options() or None
    ).to_pandas()
    return (df.sort_values("fetch_time").groupby("id", as_index=False).last())


def run(model_id: str, embedder=None) -> str:
    articles = latest_articles()
    embedder = embedder or rag.default_embedder(model_id)
    return rag.build_index(articles, embedder=embedder, model_id=model_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
