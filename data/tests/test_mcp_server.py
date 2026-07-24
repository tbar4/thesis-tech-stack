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
