# SDA MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastMCP server exposing the serving layer as tools: `latest_elements` and `upcoming_launches` (gold DuckDB), `close_approaches` (Delta), and `search_articles` (Lance index). Tested in-memory with the FastMCP client; run via `make mcp` (stdio).

**Architecture (book 8.2):** tools read the *serving* tier only — gold DuckDB read-only, Delta via Arrow, Lance via the manifest pointer. Nothing here fetches live or writes. Embargoed space-track data is structurally absent because gold and the shippable Delta tables never contain it. The search tool resolves its embedder from the manifest's model id (sentence-transformers behind the `embed` extra); tests monkeypatch `rag.default_embedder` to the `FakeEmbedder`.

**Tech Stack:** adds `fastmcp` (main), `pytest-asyncio` (dev, `asyncio_mode = "auto"`). Adds `config.gold_db()` helper.

---

## File Structure

```
data/src/sda_data/config.py        # MODIFY: + gold_db()
data/src/sda_data/mcp_server.py
data/Makefile                      # MODIFY: + mcp target
data/tests/test_mcp_server.py
```

### Task 1: server + tests

- [ ] Deps: `cd data && uv add "fastmcp>=2.0" && uv add --group dev "pytest-asyncio>=0.24"`, and in `pyproject.toml` pytest options add `asyncio_mode = "auto"`.

- [ ] `config.py` — append:

```python
def gold_db() -> Path:
    """The dbt-built gold DuckDB database."""
    return data_home() / "data" / "gold" / "sda.duckdb"
```

- [ ] `data/src/sda_data/mcp_server.py`:

```python
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
```

- [ ] `data/tests/test_mcp_server.py`:

```python
import duckdb
import pandas as pd
from fastmcp import Client

from sda_data import config, rag
from sda_data.rag import FakeEmbedder, build_index
from sda_data.silver import append


def seed_gold(tmp_data_home) -> None:
    config.gold_db().parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.gold_db()))
    con.execute("""
        create table latest_element_sets as
        select * from (values
            (25544, 'ISS (ZARYA)', 15.5, 0.0003, 51.6, 'sha256:aaaa'),
            (48274, 'CSS (TIANHE)', 15.6, 0.0004, 41.5, 'sha256:aaaa')
        ) t(norad_cat_id, object_name, mean_motion, eccentricity, inclination, snapshot_hash)
    """)
    con.execute("""
        create table upcoming_launches as
        select * from (values
            ('u-1', 'Falcon 9 | Starlink', '2126-01-02T00:00:00Z', 'Go', 'SpaceX'),
            ('u-2', 'Vulcan | USSF-106', '2126-01-01T00:00:00Z', 'Go', 'ULA')
        ) t(id, name, net, status_name, provider)
    """)
    con.close()


async def test_latest_elements_tool(tmp_data_home):
    seed_gold(tmp_data_home)
    from sda_data.mcp_server import mcp

    async with Client(mcp) as client:
        res = await client.call_tool("latest_elements", {"norad_cat_id": 25544})
        assert res.data[0]["object_name"] == "ISS (ZARYA)"


async def test_upcoming_launches_sorted(tmp_data_home):
    seed_gold(tmp_data_home)
    from sda_data.mcp_server import mcp

    async with Client(mcp) as client:
        res = await client.call_tool("upcoming_launches", {"limit": 2})
        assert [r["id"] for r in res.data] == ["u-2", "u-1"]   # soonest first


async def test_close_approaches_tool(tmp_data_home):
    append("close_approaches", pd.DataFrame([
        {"norad_i": 1, "norad_j": 2, "tca": "2026-07-25T00:00:00+00:00",
         "miss_km": 1.2, "snapshot_hash_i": "sha256:a", "snapshot_hash_j": "sha256:a",
         "screened_at": "2026-07-24T00:00:00+00:00"},
        {"norad_i": 3, "norad_j": 4, "tca": "2026-07-25T00:00:00+00:00",
         "miss_km": 90.0, "snapshot_hash_i": "sha256:a", "snapshot_hash_j": "sha256:a",
         "screened_at": "2026-07-24T00:00:00+00:00"},
    ]))
    from sda_data.mcp_server import mcp

    async with Client(mcp) as client:
        res = await client.call_tool("close_approaches", {"max_miss_km": 25.0})
        assert len(res.data) == 1 and res.data[0]["miss_km"] == 1.2


async def test_search_articles_tool(tmp_data_home, monkeypatch):
    articles = pd.DataFrame([{
        "id": 101, "title": "ISS debris avoidance maneuver", "url": "u1",
        "news_site": "NASA", "published_at": "2026-07-24T10:00:00Z",
        "summary": "The station moved to avoid debris.",
        "snapshot_hash": "sha256:" + "a" * 64,
    }])
    build_index(articles, embedder=FakeEmbedder(), model_id="fake-v1")
    monkeypatch.setattr(rag, "default_embedder", lambda model_id: FakeEmbedder())
    from sda_data.mcp_server import mcp

    async with Client(mcp) as client:
        res = await client.call_tool("search_articles", {
            "query": "ISS debris avoidance maneuver\n\nThe station moved to avoid debris.",
            "k": 1,
        })
        assert res.data[0]["article_id"] == 101
        assert "vector" not in res.data[0]
```

- [ ] Makefile — add:

```makefile
mcp:
	uv run python -m sda_data.mcp_server
```

- [ ] Run `cd data && uv run pytest tests/test_mcp_server.py -v` → PASS; full suite; commit:

```bash
git add data/src/sda_data/config.py data/src/sda_data/mcp_server.py data/Makefile data/tests/test_mcp_server.py data/pyproject.toml data/uv.lock
git commit -m "feat(data): fastmcp server over gold, close approaches, and the rag index"
git push -u origin claude/mdbook-evals-as-rewards-spec-fvhxw7
```
