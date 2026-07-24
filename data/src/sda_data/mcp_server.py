"""MCP tools over the serving layer (book 8.2): read-only, provenance intact.

Tools read gold DuckDB, the Delta close_approaches table, and the Lance index
named by the manifest. Nothing fetches live; nothing writes; embargoed
space-track data is structurally absent from every surface served here.

Run (stdio):
    uv run python -m sda_data.mcp_server
"""
from __future__ import annotations

import duckdb
from fastmcp import FastMCP

from sda_data import config, rag
from sda_data.silver import table_uri

mcp = FastMCP("sda-data")


def _gold(sql: str, params: list | None = None) -> list[dict]:
    con = duckdb.connect(str(config.gold_db()), read_only=True)
    try:
        return con.sql(sql, params=params).df().to_dict("records")
    finally:
        con.close()


@mcp.tool
def latest_elements(norad_cat_id: int | None = None, limit: int = 10) -> list[dict]:
    """Latest orbital element set per object (celestrak-derived gold table).
    Optionally filter to one NORAD catalog id."""
    if norad_cat_id is not None:
        return _gold(
            "select * from latest_element_sets where norad_cat_id = ? limit ?",
            [norad_cat_id, limit],
        )
    return _gold("select * from latest_element_sets order by norad_cat_id limit ?", [limit])


@mcp.tool
def upcoming_launches(limit: int = 10) -> list[dict]:
    """Upcoming launches (deduped, future net only), soonest first."""
    return _gold(
        "select * from upcoming_launches order by try_cast(net as timestamptz) limit ?",
        [limit],
    )


@mcp.tool
def close_approaches(max_miss_km: float = 25.0, limit: int = 20) -> list[dict]:
    """Screened close approaches from the latest conjunction run, nearest first.
    Every row carries the snapshot hashes its elements came from."""
    from deltalake import DeltaTable

    df = DeltaTable(
        table_uri("close_approaches"),
        storage_options=config.s3_storage_options() or None,
    ).to_pandas()
    df = df[df.miss_km <= max_miss_km].sort_values("miss_km").head(limit)
    return df.to_dict("records")


@mcp.tool
def search_articles(query: str, k: int = 5) -> list[dict]:
    """Semantic search over the space-news article index (Lance manifest table).
    Returns chunks with article provenance (id, title, url, published_at)."""
    hits = rag.search(query, k=k)
    cols = [c for c in hits.columns if c != "vector"]
    return hits[cols].to_dict("records")


if __name__ == "__main__":
    mcp.run()
