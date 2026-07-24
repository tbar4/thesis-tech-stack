"""Word-window chunking with provenance stamps (book 8.1: ~256-word windows,
small overlap, source stamps non-negotiable)."""
from __future__ import annotations

import pandas as pd


def chunk_text(text: str, size_words: int = 256, overlap_words: int = 32) -> list[str]:
    words = text.split()
    if len(words) <= size_words:
        return [" ".join(words)]
    step = size_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size_words]
        chunks.append(" ".join(window))
        if start + size_words >= len(words):
            break
    return chunks


def chunk_articles(df: pd.DataFrame, size_words: int = 256, overlap_words: int = 32) -> pd.DataFrame:
    """One row per chunk; every chunk keeps its article's provenance."""
    rows = []
    for a in df.itertuples():
        body = f"{a.title}\n\n{a.summary}" if getattr(a, "summary", "") else a.title
        for idx, text in enumerate(chunk_text(body, size_words, overlap_words)):
            rows.append({
                "article_id": int(a.id), "title": a.title, "url": a.url,
                "published_at": a.published_at, "chunk_idx": idx, "text": text,
                "snapshot_hash": a.snapshot_hash,
            })
    return pd.DataFrame(rows)
