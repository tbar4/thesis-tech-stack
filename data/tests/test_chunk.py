import pandas as pd

from sda_data.chunk import chunk_articles, chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("a b c", size_words=256, overlap_words=32) == ["a b c"]


def test_long_text_windows_with_overlap():
    words = [f"w{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words), size_words=256, overlap_words=32)
    assert len(chunks) == 3
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-32:] == second[:32]          # boundary fact survives in both


def test_chunk_articles_stamps_provenance():
    df = pd.DataFrame([{
        "id": 101, "title": "T", "url": "u", "news_site": "N",
        "published_at": "2026-07-24T10:00:00Z",
        "summary": " ".join(f"w{i}" for i in range(300)),
        "snapshot_hash": "sha256:" + "a" * 64,
    }])
    chunks = chunk_articles(df)
    assert len(chunks) == 2                     # 300 words + title -> 2 windows
    assert set(chunks.columns) >= {
        "article_id", "title", "url", "published_at", "chunk_idx", "text", "snapshot_hash",
    }
    assert list(chunks.chunk_idx) == [0, 1]
