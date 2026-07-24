import numpy as np
import pandas as pd

from sda_data import config
from sda_data.rag import FakeEmbedder, build_index, read_manifest, search

ARTICLES = pd.DataFrame([
    {"id": 101, "title": "ISS debris avoidance maneuver", "url": "u1",
     "news_site": "NASA", "published_at": "2026-07-24T10:00:00Z",
     "summary": "The station moved to avoid debris.",
     "snapshot_hash": "sha256:" + "a" * 64},
    {"id": 102, "title": "Starship launch window", "url": "u2",
     "news_site": "SN", "published_at": "2026-07-24T11:00:00Z",
     "summary": "A launch window opens Friday.",
     "snapshot_hash": "sha256:" + "a" * 64},
])


def test_build_creates_keyed_table_and_manifest(tmp_data_home):
    name = build_index(ARTICLES, embedder=FakeEmbedder(), model_id="fake-v1")
    manifest = read_manifest()
    assert manifest["current"] == name
    assert manifest["model_id"] == "fake-v1"
    assert "fake-v1" in name and name.startswith("chunks_")


def test_exact_match_query_ranks_first(tmp_data_home):
    build_index(ARTICLES, embedder=FakeEmbedder(), model_id="fake-v1")
    # FakeEmbedder is deterministic: identical text -> identical vector -> distance 0.
    gold_text = "ISS debris avoidance maneuver\n\nThe station moved to avoid debris."
    hits = search(gold_text, k=2, embedder=FakeEmbedder())
    assert hits.iloc[0].article_id == 101


def test_reembed_is_new_table_plus_manifest_flip(tmp_data_home):
    import lancedb

    n1 = build_index(ARTICLES, embedder=FakeEmbedder(), model_id="fake-v1")
    n2 = build_index(ARTICLES, embedder=FakeEmbedder(seed=9), model_id="fake-v2")
    assert n1 != n2
    assert read_manifest()["current"] == n2
    db = lancedb.connect(str(config.data_home() / "data" / "lance"))
    # Both tables must still open: the old instrument stays frozen.
    assert db.open_table(n1).count_rows() == 2
    assert db.open_table(n2).count_rows() == 2


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder()
    v1, v2 = e.embed(["same text"]), e.embed(["same text"])
    assert np.allclose(v1, v2)
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)


def test_build_task_reads_delta_and_indexes(tmp_data_home):
    from sda_data.silver import append
    from sda_data.tasks.build_rag_index import run as build_run

    df = ARTICLES.copy()
    df["source"] = "spaceflightnews"
    df["fetch_time"] = pd.Timestamp("2026-07-24", tz="UTC")
    append("articles", df)

    name = build_run("fake-v1", embedder=FakeEmbedder())
    assert read_manifest()["current"] == name
