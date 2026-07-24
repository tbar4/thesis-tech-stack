# SDA RAG Index (LanceDB) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chunk the articles corpus, embed it, and build an immutable LanceDB table keyed by (corpus hash, embedding model, chunk params), with a manifest pointer written last, plus search, the build task, and the asset-triggered DAG.

**Architecture (book 8.1 + the merged sync discipline):** every index build is a *fresh* table named by its full recipe key — never an in-place upsert — and a tiny `manifest.json` (written last) names which table is live. Switching embedding models later = new table + manifest flip; the old instrument stays frozen. The `Embedder` is a protocol: production resolves `sentence-transformers` lazily behind an optional `embed` extra (torch never enters the test env); tests use a deterministic hash-seeded `FakeEmbedder`, and the retrieval test uses an exact-match query (identical text → identical vector → distance 0 → top hit). Vectors are L2-normalized at build and query time so L2 ranking equals cosine ranking (book eq. 1.2). The Lance root lives on the working tier (`$SDA_DATA_HOME/data/lance`); MinIO sync arrives with real credentials via the backup/sync job pattern.

**Tech Stack:** adds `lancedb` (main), `sentence-transformers` (optional extra `embed`, not installed in tests/CI).

---

## File Structure

```
data/src/sda_data/chunk.py          # word-window chunking with provenance stamps
data/src/sda_data/rag.py            # Embedder protocol, build_index, manifest, search
data/src/sda_data/tasks/build_rag_index.py
data/dags/sda_rag.py
data/tests/test_chunk.py
data/tests/test_rag.py
```

### Task 1: chunking

- [ ] `data/tests/test_chunk.py`:

```python
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
```

- [ ] `data/src/sda_data/chunk.py`:

```python
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
```

- [ ] Run + commit: `feat(data): word-window chunking with provenance stamps`

### Task 2: index build, manifest, search

- [ ] Add dep: `cd data && uv add "lancedb>=0.13"` and the optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
embed = ["sentence-transformers>=3.0"]
```

- [ ] `data/tests/test_rag.py`:

```python
import json

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
    assert {n1, n2} <= set(db.table_names())      # old instrument stays frozen


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder()
    v1, v2 = e.embed(["same text"]), e.embed(["same text"])
    assert np.allclose(v1, v2)
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)
```

- [ ] `data/src/sda_data/rag.py`:

```python
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
```

- [ ] Run + commit: `feat(data): lancedb index build with recipe-keyed tables and manifest pointer`

### Task 3: build task + DAG

- [ ] `data/src/sda_data/tasks/build_rag_index.py`:

```python
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
```

- [ ] Test appended to `data/tests/test_rag.py`:

```python
def test_build_task_reads_delta_and_indexes(tmp_data_home):
    from sda_data.silver import append
    from sda_data.tasks.build_rag_index import run as build_run

    df = ARTICLES.copy()
    df["source"] = "spaceflightnews"
    df["fetch_time"] = pd.Timestamp("2026-07-24", tz="UTC")
    append("articles", df)

    name = build_run("fake-v1", embedder=FakeEmbedder())
    assert read_manifest()["current"] == name
```

- [ ] `data/dags/sda_rag.py`:

```python
"""Data-aware consumer DAG: rebuild the RAG index when articles refresh."""
from __future__ import annotations

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

ARTICLES = Asset("sda://spaceflightnews/articles")
RAG_INDEX = Asset("sda://rag/index")

with DAG(
    dag_id="sda_rag",
    schedule=[ARTICLES],
    catchup=False,
    max_active_runs=1,
    tags=["sda", "rag"],
) as dag:
    build = BashOperator(
        task_id="build_rag_index",
        bash_command="cd /opt/project/data && "
                     "uv run python -m sda_data.tasks.build_rag_index",
        outlets=[RAG_INDEX],
    )
```

- [ ] Full suite, commit: `feat(data): rag build task + asset-triggered DAG`
