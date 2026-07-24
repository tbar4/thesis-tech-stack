"""LanceDB index build + search with the one-versioning-authority discipline.

Every build writes a FRESH table named by (corpus hash, model id, chunk
params); manifest.json is written LAST and names the live table. Re-embedding
with a new model is a new table + a manifest flip; nothing is upserted in
place, so the old instrument stays frozen and citable (book 8.1)."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from sda_data import config
from sda_data.chunk import chunk_articles


def lance_root():
    return config.data_home() / "data" / "lance"


def corpus_hash(articles: pd.DataFrame) -> str:
    key = ",".join(sorted(f"{a.id}:{a.snapshot_hash}" for a in articles.itertuples()))
    return hashlib.sha256(key.encode()).hexdigest()


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()


class FakeEmbedder:
    """Deterministic hash-seeded embedder for tests and offline dev.
    Identical text -> identical unit vector."""

    dim = 64

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(f"{self.seed}:{t}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            out[i] = rng.standard_normal(self.dim)
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out


def default_embedder(model_id: str):
    """Production path: sentence-transformers behind the optional `embed` extra."""
    from sentence_transformers import SentenceTransformer  # lazy: torch stays optional

    class _ST:
        def __init__(self) -> None:
            self.model = SentenceTransformer(model_id)

        def embed(self, texts: list[str]) -> np.ndarray:
            v = np.asarray(self.model.encode(texts), dtype=np.float32)
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    return _ST()


def build_index(
    articles: pd.DataFrame, *, embedder, model_id: str,
    size_words: int = 256, overlap_words: int = 32,
) -> str:
    import lancedb

    chunks = chunk_articles(articles, size_words, overlap_words)
    vecs = embedder.embed(list(chunks.text))
    records = chunks.to_dict("records")
    for r, v in zip(records, vecs):
        r["vector"] = v.tolist()

    name = (f"chunks_{corpus_hash(articles)[:8]}_{_slug(model_id)}"
            f"_w{size_words}o{overlap_words}")
    root = lance_root()
    root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(root))
    db.create_table(name, data=records, mode="overwrite")  # same key -> same content

    manifest = {"current": name, "model_id": model_id,
                "corpus_hash": corpus_hash(articles),
                "size_words": size_words, "overlap_words": overlap_words}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))  # written LAST
    print(f"rag index: {len(records)} chunks -> {name}")
    return name


def read_manifest() -> dict:
    return json.loads((lance_root() / "manifest.json").read_text())


def search(query: str, k: int = 5, embedder=None) -> pd.DataFrame:
    import lancedb

    manifest = read_manifest()
    embedder = embedder or default_embedder(manifest["model_id"])
    q = embedder.embed([query])[0]
    table = lancedb.connect(str(lance_root())).open_table(manifest["current"])
    return table.search(q).limit(k).to_pandas()
