# RAG over space text

There is a clean line running through this whole part, and it is worth drawing before any code. Some SDA questions are *computed* and some are *looked up*. "Do objects 25544 and 48274 pass within 5 km in the next 24 hours?" has an exact answer that no corpus contains, because it is a function of two orbital states and a propagator; you obtain it by running SGP4, not by searching text, and that is what the oracle and the MCP tools in 8.2 are for. "What did NASA say about the ISS debris-avoidance maneuver last week, and why?" has no closed form at all; the answer lives in prose, in an article that may have been published after the model's training cutoff, and the only way to get it is to *retrieve the text and read it*. Retrieval-augmented generation is the tool for the second kind of question and a category error for the first. You retrieve what you cannot compute. The moment a question has a formula, the honest move is to put the formula behind a tool (8.2) and let the model call it, because a retrieved number is a number you did not verify, from a source you did not control, at a freshness you did not check. So this chapter is deliberately narrow: I build RAG over the pipeline's *article* corpus (the Spaceflight-News / NASA text snapshot from 3.9) to ground **qualitative and recency** SDA answers, and then I spend most of the chapter measuring whether it actually helps, because "I added RAG and the score went up" is exactly the kind of claim that turns out to be a confound when you look closely.

## Theory

### What RAG is, mechanically

RAG is three steps stapled to a generation call. First, **index**: chop the corpus into chunks, embed each chunk into a vector, and store the vectors. Second, **retrieve**: embed the query, find the chunks whose vectors are nearest, and pull their text. Third, **assemble**: paste those chunks into the prompt as context and ask the model to answer using them. Nothing about it is deep; the entire subtlety is in "nearest" (which embedding, which metric), in "chunks" (how you cut the text), and in the evaluation (whether the retrieved text changed the answer for the better). I keep the loop thin and hand-rolled for exactly the reason the memory chapter kept the byte arithmetic explicit: a legible pipeline is one whose failures I can locate. LlamaIndex would give me all three steps behind one `VectorStoreIndex.from_documents(...).as_query_engine()` call, and for a product I would use it; for a thesis where the retrieval step is a suspect in a causal argument, I want every hop visible.

### Chunking

Embeddings summarize a span of text into one vector, and a vector has a fixed budget of meaning. Embed a whole 1,200-word article into a single 384-dimensional point and you get a blurry topical centroid that matches everything about that article and pinpoints nothing. Embed every sentence separately and you shred the context a claim needs to be interpreted. The craft is picking a chunk size that holds one coherent idea. My default for these articles is a token-bounded window of roughly 256 tokens with a small overlap (about 32 tokens) so a fact straddling a boundary survives in at least one chunk, and I keep each chunk stamped with its source article id, title, url, and publication date, because those stamps are what let me both cite the source and, crucially, measure retrieval quality later. The overlap is not free (it inflates the index and lets one fact match twice), so I keep it small; the source stamps are non-negotiable.

### Cosine similarity and nearest-neighbor retrieval

"Nearest" needs a metric, and for text embeddings the metric is cosine similarity, which measures the angle between two vectors and ignores their magnitude.

```admonish derivation
**Cosine similarity and top-$k$ retrieval.** For a query embedding $q \in \mathbb{R}^{d}$ and a chunk embedding $c \in \mathbb{R}^{d}$, the cosine similarity is the normalized inner product,

$$ \cos(q, c) = \frac{q \cdot c}{\lVert q \rVert\, \lVert c \rVert} = \frac{\sum_{i=1}^{d} q_i c_i}{\sqrt{\sum_i q_i^2}\,\sqrt{\sum_i c_i^2}} \in [-1, 1]. \tag{1.1}$$

It is 1 when $q$ and $c$ point the same direction, 0 when orthogonal, and it discards length, which is what you want: a chunk should not rank higher just because it is longer or its embedding happens to have larger norm. Modern embedding models are trained so that semantic closeness shows up as angular closeness, so cosine is the metric they were optimized for.

If you **L2-normalize** every vector at index time, setting $\hat q = q / \lVert q \rVert$ and $\hat c = c / \lVert c \rVert$, then

$$ \cos(q, c) = \hat q \cdot \hat c, \tag{1.2}$$

and cosine similarity collapses to a plain dot product, which is one fused multiply-add per dimension and the reason vector stores prefer normalized vectors. Retrieval is then the top-$k$ selection

$$ \mathrm{TopK}_k(q) = \operatorname*{arg\,max}^{(k)}_{c \in \mathcal{C}}\ \hat q \cdot \hat c, \tag{1.3}$$

the $k$ chunks in the corpus $\mathcal{C}$ with the largest dot product against the query. For a few thousand chunks this is an exact brute-force scan (a single matrix-vector product and a partial sort), which is milliseconds and needs no approximate index at all; the ANN structures (IVF, HNSW) that a vector store can build only start earning their keep in the millions of chunks, and I note where LanceDB switches them on but do not need them at this scale.
```

The practical consequence of equation (1.2) is that I normalize once, at index build, and then every query is a dot product. The practical consequence of equation (1.3) at this corpus size is that I get *exact* nearest neighbors for free, so any retrieval miss is a fact about the embedding or the chunking, never an approximation artifact, which keeps the later causal argument clean.

### Evaluating retrieval: recall@k and MRR

Retrieval has its own quality, separate from whether the final answer is right, and you must measure it separately or you will misattribute failures. The two standard metrics assume each query has a known set of *relevant* chunks (its gold set $R_q$); in this book the gold set is "the chunks belonging to the article the question was written from," which the 3.9 provenance stamps make knowable.

```admonish derivation
**Recall@$k$ and Mean Reciprocal Rank.** Let $Q$ be the set of eval queries, and for query $q$ let $R_q$ be its relevant chunks and $\mathrm{TopK}_k(q)$ the $k$ retrieved chunks from equation (1.3). **Recall@$k$** is the fraction of relevant chunks that made it into the top $k$, averaged over queries,

$$ \text{recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{\bigl| \mathrm{TopK}_k(q) \cap R_q \bigr|}{|R_q|}. \tag{1.4}$$

When each query has a single gold chunk, equation (1.4) reduces to the **hit rate**: the fraction of queries for which the gold chunk appears anywhere in the top $k$. Recall@$k$ answers "did the needed evidence get in front of the model at all," which is the question that matters most for RAG, because a chunk not retrieved cannot help no matter how good the generator is.

**Mean Reciprocal Rank** rewards putting the relevant chunk *high*, not just present. Let $\mathrm{rank}_q$ be the position (1-indexed) of the first relevant chunk in the retrieved list, or $\infty$ if none of the top $k$ is relevant. Then

$$ \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\mathrm{rank}_q}, \qquad \frac{1}{\infty} := 0. \tag{1.5}$$

A first-position hit contributes 1, a second-position hit $\tfrac12$, a fifth-position hit $\tfrac15$, a miss 0. MRR and recall@$k$ can disagree in an informative way: a retriever can have high recall@10 (the evidence is usually somewhere in the ten chunks) but poor MRR (it is usually buried at rank 7), and since a generator attends less reliably to the middle of a long context, that gap predicts that raising $k$ will help recall but a reranker will help the answer.
```

The reason to compute these at all is the central methodological point of the chapter: **retrieval metrics and end-task metrics are not the same number**, and they fail apart in both directions. Retrieval can be perfect while the answer stays wrong, because the generator ignored or misread the chunk; retrieval can be lousy while the answer is right, because the model already knew the fact parametrically and the retrieved junk did no harm. So I always report both, and the end-task metric (did the model answer the SDA question correctly, with vs without retrieval) is the one that licenses any claim of "RAG helped." Retrieval metrics diagnose *why*. This split is not idiosyncratic: [RAG] Part 2 treats retrieval metrics and cosine-similarity evaluation as their own concern, and [GADP] gives explicit retrieval-evaluation metrics alongside the end-task ones, which is the same discipline of scoring the finder and the answerer on separate axes.

### The advanced-RAG ladder, and why I stop at the minimal rung

The pipeline I have described so far (embed the chunks, take the top $k$ by cosine, paste them into the prompt) is what the literature calls **naive RAG**, and it is worth naming the ladder it sits at the bottom of, so the narrowing this chapter does reads as a choice rather than an oversight. [RAG] frames the whole space in three rungs. **Naive RAG** is exactly the minimal loop: embed, retrieve top-$k$, stuff the chunks into the prompt. **Advanced RAG** keeps that core but tunes the retrieval around it, adding steps before and after the nearest-neighbor lookup to raise the odds that the right evidence lands in front of the generator. **Modular RAG** goes further and treats retrieve, rerank, and generate as swappable components in a configurable pipeline, the LangChain-shaped or pattern-catalog build that [LC]'s RAG-systems chapter and [GADP]'s Basic-RAG and index-aware patterns (Patterns 6 and 9) lay out. Almost everything the field calls "advanced" is a refinement of the retrieve step, so it is worth being concrete about which refinements I am declining.

On the retrieval side, four techniques recur. **Hybrid retrieval** combines the dense-embedding cosine of equation (1.3) with a lexical or keyword score such as BM25, so a query that hinges on an exact token (a NORAD id, an agency acronym, a spacecraft name) is not lost to the semantic blur that pure embeddings can smear over rare strings. **Cross-encoder reranking** re-scores a deliberately wide top-$k$ with a heavier model that reads the query and each chunk *together* before the context is assembled, trading compute for a sharper ordering (this is the one advanced step I do wire up, optionally, in the Lab, precisely because the recall-vs-MRR gap from equation (1.5) is what tells you when it would pay). **Query transformation** rewrites or expands the query before retrieval, so a terse or ambiguously phrased question is matched against the corpus it is really asking about rather than its literal surface form. And **contextual retrieval** with **metadata filtering** attaches surrounding context or structured fields (publication date, source, article id) to each chunk so retrieval can sharpen on meaning and filter on facts, a richer use of exactly the provenance stamps I already carry for citation and for measuring recall@$k$.

Above those sit the self-correcting and agentic variants, where retrieval stops being a single fixed hop. **Corrective RAG (CRAG)** grades the retrieved chunks and falls back when they are weak, for instance to a web search, rather than answering from thin evidence. **Self-RAG** hands the retrieve decision to the model itself: it chooses whether to retrieve at all and then critiques what it pulled before trusting it. **Agentic RAG** turns retrieval into a loop, a retrieval agent that reasons, retrieves again, and reasons once more until it judges the evidence sufficient. And **GraphRAG** ([GADP]) changes the index rather than the loop, retrieving over a knowledge graph of entities and relations instead of a flat vector store, so multi-hop questions can be walked rather than guessed. Each of these is a genuine capability, and [LC] and [GADP] cover them for good reason: on a hard corpus they move the numbers.

I stop at the bottom rung on purpose, and the reason is the whole point of this weave rather than a shrug at scope. In this book retrieval is not just an engineering nicety, it is a *variable in a causal claim*, the augmentation axis of the four-arm comparison in 8.3. Every technique on the ladder above adds a moving part, and every added part is a place where the question "did retrieval cause the gain, or was it lookup" gets harder to answer cleanly: a reranker can promote the one chunk that states the answer verbatim (the 3.8 leakage trap wearing a fancier hat), a query rewriter can smuggle the gold phrasing into the match, an agentic loop can retrieve until it stumbles onto the answer and then credit "RAG" for what was really persistence. So I build the minimal legible version (one embedding model, flat exact search, top-$k$, a thin hand-rolled loop) because it is the one whose contribution I can isolate, and I treat the advanced ladder ([RAG]'s modular RAG, [LC] and [GADP]'s hybrid, rerank, CRAG, Self-RAG, and GraphRAG) as the documented next step to reach for *if* the minimal version underperforms on the frozen suite, not as default sophistication to switch on because it exists. Sophistication that I cannot attribute is not an asset in a causal argument, it is a confound with better marketing.

### The embedding-model memory line for 16GB co-residency

Everything above assumes I can produce embeddings on the same card that serves the generation model, and on 16GB that is the binding constraint, so I account for it in bytes exactly as the memory chapter (6.1) did for the generation model.

```admonish derivation
**Why the embedding model is nearly free in KV terms.** An embedding model runs one forward pass per input and reads out a pooled vector (mean- or last-token pooling); it never autoregressively decodes, so it grows **no** KV cache across steps. Its resident cost is therefore weights plus a single prefill's transient activation, with none of the linear-in-tokens KV growth that dominates the generation budget of equation (6.1):

$$ M_{\text{embed}} = \underbrace{P_e \cdot b_e}_{\text{weights}} \;+\; \underbrace{a \cdot B_e \cdot L_e}_{\text{one prefill's activations}}, \tag{1.6}$$

where $P_e$ is the embedding model's parameter count, $b_e$ its bytes-per-weight, and the second term is a transient batch-$B_e$, length-$L_e$ activation buffer that is freed the instant the vector is read out. There is no $2 B L N h_{kv} d_{\text{head}} b$ KV term, which is the whole reason a small embedding model is a cheap lodger.

**The co-residency budget.** To serve embeddings and generation on one card at once, both must fit under the capacity $C$ (about 15.5 GiB usable on the 16GB RTX 5080 after driver and CUDA context):

$$ \underbrace{P_g b_g + M_{\text{KV}}^{(g)}}_{\text{generation (eq.\ 6.1)}} \;+\; \underbrace{P_e b_e}_{\text{embedding weights}} \;+\; M_{\text{ctx}} \;\le\; C. \tag{1.7}$$

Put numbers on it. The generation model is Qwen3-8B at FP8 weights, $P_g b_g \approx 7.7$ GiB (the 2.5 serving figure), with a KV pool of roughly 3.5 GiB at `--max-model-len 8192` and CUDA context near 0.6 GiB, so the generation side is about 11.8 GiB. That leaves $\approx 3.7$ GiB for the embedding lodger. A `bge-small-en-v1.5` encoder ($P_e \approx 33$M, FP16) is $\approx 0.07$ GiB of weights plus a fraction of a GiB of transient activation from equation (1.6); a `bge-large` ($\approx 335$M) is $\approx 0.67$ GiB; a decoder-style `Qwen3-Embedding-0.6B` is $\approx 1.2$ GiB in BF16. All three clear the 3.7 GiB headroom, so the choice is quality-per-byte, not fit, and I pick it by a small retrieval bake-off (recall@$k$ / MRR on a held-out labeled set) exactly the way 3.6 picked a judge. What does *not* fit is a second multi-billion-parameter model held resident purely to embed; equation (1.6) is the reason you never need one.
```

The equation (1.7) budget only bites if you insist on holding both models resident *simultaneously*. You usually do not have to, and the cheaper discipline is to **time-share the GPU**: serve the embedding model alone, build the entire index (an offline, one-time batch job), tear it down, then serve only the generation model. The one place time-sharing pinches is that at eval time each query must also be embedded; but for a *fixed* eval suite the queries are known in advance, so I precompute every query embedding in the same offline pass, persist them, and the eval loop never needs the embedding model resident at all. I default to time-sharing for index build and query precompute, and reach for co-residency (equation 1.7, two vLLM servers each told to take a fraction of the card) only when I want interactive, live retrieval, which 8.2 and 8.3 will.

```admonish vram-budget title="Embedding + generation co-residency on 16GB"
Two ways to fit both, both honest, chosen by whether retrieval must be live.

**Time-share (default for this chapter).** One server at a time.
- Phase A: `vllm serve BAAI/bge-small-en-v1.5 --task embed --gpu-memory-utilization 0.9`. Embed the whole corpus and every eval query in one batch pass. Tear down.
- Phase B: `vllm serve Qwen/Qwen3-8B --quantization fp8 --max-model-len 8192`. Run the eval reading the precomputed query vectors from disk.
- Peak VRAM is just whichever single model is up, so nothing new to budget beyond the 2.5/6.1 generation figure. This is what the lab does.

**Co-resident (for live retrieval, 8.2/8.3).** Both servers up, the card split by `--gpu-memory-utilization` so the fractions sum under 1:
- Generation: `Qwen/Qwen3-8B --quantization fp8 --gpu-memory-utilization 0.75` ($\approx$ 7.7 GiB weights + KV, $\approx 11.6$ GiB).
- Embedding: `BAAI/bge-small-en-v1.5 --task embed --port 8001 --gpu-memory-utilization 0.15` ($\approx$ 0.07 GiB weights + activation + its own context).
- Sum of utilization fractions is 0.90, leaving driver/context headroom. This is equation (1.7) turned into two flags: the embedding model is a cheap lodger *because* equation (1.6) gives it no KV term. Get greedy (a 0.6B decoder-embedder at high `--max-model-len`, or a fat KV pool on the generation side) and the two servers race for blocks and one OOMs at boot; the fix is to shrink the generation KV pool or the embedder, never to hope for a fit.
```

## Tooling

The stack is chosen to keep the loop legible and the serving unified. Embeddings come from a local model served by **vLLM**, the same engine already serving generation, exposing the OpenAI-compatible `/v1/embeddings` endpoint (from 2.4), so the client is the same three-line `openai` call I use everywhere and there is no second serving framework to operate; **`sentence-transformers`** is the in-process fallback when I want embeddings without standing up a server (it is what 3.8's contamination scanner already uses on CPU). The vector store is **LanceDB**: embedded, on-disk, no server process, an Arrow-native table that I query in-process, which suits a one-node thesis the way DuckDB suited the pipeline. The retrieval loop itself is **thin and hand-rolled** (chunk, embed, top-$k$, assemble, with an optional cross-encoder rerank), and I name **LlamaIndex** as the batteries-included alternative I am deliberately not using, for legibility. Scoring the eval reuses **`evalstats`** from 3.7 unchanged: the RAG-vs-parametric comparison is paired over shared items, so it gets a `bootstrap_paired_diff` interval and a `mcnemar` test, never two marginal CIs eyeballed for overlap.

## Lab

The lab builds the index over a 3.9 article snapshot, implements the thin retrieval loop, and runs a RAG-vs-parametric eval on qualitative SDA questions, reporting retrieval metrics and end-task metrics *separately*. The artifact is the index build plus the RAG-vs-parametric report.

### Project setup

The `serve/` uv sub-project gains the LanceDB client and the embeddings tooling; vLLM is already there from Part II.

```bash title="serve/setup-rag.sh"
cd serve
uv add "lancedb>=0.13" "openai>=1.40" "numpy>=1.26" "pandas>=2.2" \
       "pyarrow>=17" "transformers>=4.44"
uv add "sentence-transformers>=3.0"   # in-process fallback + optional reranker
```

### Serve the embedding model

```bash title="serve/bge-embed.sh"
# Phase A of the vram-budget: embedding model alone, whole card.
uv run vllm serve BAAI/bge-small-en-v1.5 \
    --task embed --port 8001 \
    --gpu-memory-utilization 0.90 2>&1 | tee embed-serve.log
```

### Chunk the article snapshot

```python title="serve/rag/chunk.py"
"""Token-bounded chunking of the 3.9 article snapshot, with source stamps."""
from dataclasses import dataclass, asdict
import pandas as pd
from transformers import AutoTokenizer

TOKENIZER = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 256
OVERLAP_TOKENS = 32

@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    title: str
    url: str
    published_at: str
    text: str

def chunk_article(tok, article) -> list[Chunk]:
    """Sliding token window over title + body; each chunk keeps provenance."""
    body = f"{article['title']}. {article['summary']}"
    ids = tok(body, add_special_tokens=False)["input_ids"]
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    out = []
    for start in range(0, max(len(ids), 1), step):
        window = ids[start:start + CHUNK_TOKENS]
        if not window:
            break
        text = tok.decode(window)
        out.append(Chunk(
            chunk_id=f"{article['id']}::{start}",
            article_id=str(article["id"]),
            title=article["title"], url=article["url"],
            published_at=str(article["published_at"]), text=text))
        if start + CHUNK_TOKENS >= len(ids):
            break
    return out

def load_chunks(parquet_path: str) -> list[dict]:
    """Read a snapshot Parquet (objects: id,title,summary,url,published_at)."""
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    df = pd.read_parquet(parquet_path)
    chunks: list[Chunk] = []
    for _, row in df.iterrows():
        chunks.extend(chunk_article(tok, row))
    return [asdict(c) for c in chunks]
```

### Embed and build the LanceDB index

```python title="serve/rag/build_index.py"
"""Embed chunks via vLLM's /v1/embeddings and write a LanceDB table.

Artifact half 1: the on-disk LanceDB index. Time-share Phase A: only the
embedding server is up here; the generation model is not resident.
"""
import numpy as np
import lancedb
from openai import OpenAI
from .chunk import load_chunks

EMBED_URL = "http://localhost:8001/v1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DB_PATH = "artifacts/space_rag.lance"
TABLE = "chunks"
BATCH = 64

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Batched call to the OpenAI-compatible embeddings endpoint."""
    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL,
                                        input=texts[i:i + BATCH])
        vecs.extend(d.embedding for d in resp.data)
    v = np.asarray(vecs, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12   # eq. (1.2)
    return v

def build(parquet_path: str) -> None:
    records = load_chunks(parquet_path)
    client = OpenAI(base_url=EMBED_URL, api_key="EMPTY")
    vecs = embed_texts(client, [r["text"] for r in records])
    for r, v in zip(records, vecs):
        r["vector"] = v.tolist()          # LanceDB reserves the "vector" column
    db = lancedb.connect(DB_PATH)
    tbl = db.create_table(TABLE, data=records, mode="overwrite")
    # A few thousand chunks: exact brute-force cosine (eq. 1.3) is fine and is
    # LanceDB's default with no ANN index built. Uncomment to switch on IVF-PQ
    # once the corpus reaches ~1e6 chunks:
    # tbl.create_index(metric="cosine", num_partitions=256, num_sub_vectors=32)
    print(f"indexed {len(records)} chunks -> {DB_PATH}")

if __name__ == "__main__":
    build("../data/artifacts/articles_snapshot.parquet")
```

### The thin retrieval loop and prompt assembly

```python title="serve/rag/retrieve.py"
"""Embed the query, top-k by cosine (eq. 1.3), assemble the prompt."""
import numpy as np
import lancedb
from openai import OpenAI
from .build_index import EMBED_URL, EMBED_MODEL, DB_PATH, TABLE

def embed_query(client: OpenAI, query: str) -> np.ndarray:
    v = np.asarray(client.embeddings.create(
        model=EMBED_MODEL, input=[query]).data[0].embedding, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def retrieve(tbl, qvec: np.ndarray, k: int = 5) -> list[dict]:
    """LanceDB exact cosine search; distance = 1 - cosine, so we invert it."""
    rows = (tbl.search(qvec).metric("cosine").limit(k)
            .select(["chunk_id", "article_id", "title", "url",
                     "published_at", "text", "_distance"]).to_list())
    for r in rows:
        r["cosine"] = 1.0 - float(r["_distance"])
    return rows

def assemble_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] ({c['published_at']}, {c['url']})\n{c['text']}"
        for i, c in enumerate(chunks))
    return (
        "Use the CONTEXT below to answer the QUESTION. If the context does not "
        "contain the answer, say so; do not invent facts.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nAnswer:")

def open_db():
    return lancedb.connect(DB_PATH).open_table(TABLE)
```

### Optional cross-encoder rerank

Retrieval by bi-encoder cosine (equation 1.3) scores the query and each chunk *independently*; a cross-encoder reads the query and a chunk *together* and scores their relevance jointly, which is more accurate and more expensive, so the standard pattern is to retrieve a wide top-$k$ cheaply and rerank it down.

```python title="serve/rag/rerank.py"
"""Optional cross-encoder rerank: retrieve wide, re-score jointly, keep top-n."""
from sentence_transformers import CrossEncoder

_MODEL = None
def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder("BAAI/bge-reranker-base")  # or vLLM /rerank
    return _MODEL

def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    scores = _model().predict([(query, c["text"]) for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = [chunks[i] | {"rerank_score": float(scores[i])} for i in order[:top_n]]
    return out
```

### The RAG-vs-parametric eval

This is the measurement the chapter exists for. Each eval item is a qualitative SDA question with a gold answer and the `article_id` it was written from (the retrieval gold). I run two arms on the *same* items: **parametric** (the generation model answers from its weights alone) and **RAG** (the same model answers from the assembled context). I record retrieval metrics (recall@$k$, MRR from equations 1.4/1.5) and end-task correctness for both arms, then hand the paired end-task scores to `evalstats`.

```python title="serve/rag/eval_rag.py"
"""RAG vs parametric on qualitative SDA questions.

Artifact half 2: reports retrieval metrics AND end-task metrics separately,
with a paired evalstats diff + CI. Time-share Phase B: generation model up,
query vectors read from the precomputed store (embedding server not resident).
"""
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
import evalstats as es
from .retrieve import open_db, retrieve, assemble_prompt

GEN_URL = "http://localhost:8000/v1"
GEN_MODEL = "Qwen/Qwen3-8B"
K = 5

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def correct(answer: str, gold: str) -> int:
    """Placeholder checker: gold key-phrase containment. A real run scores
    qualitative answers with the 3.6 judge; the paired stats are identical."""
    return int(normalize(gold) in normalize(answer))

def ask(client, prompt: str) -> str:
    r = client.chat.completions.create(
        model=GEN_MODEL, temperature=0.0, max_tokens=256,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""

def run(tasks_path: str, out="artifacts/rag_report.json") -> dict:
    tasks = [json.loads(l) for l in open(tasks_path)]
    tbl, gen = open_db(), OpenAI(base_url=GEN_URL, api_key="EMPTY")
    qvecs = np.load("artifacts/query_vectors.npy")   # precomputed in Phase A

    para, rag = [], []          # 0/1 end-task correctness, shared item order
    recall_hits, recip_ranks = [], []
    for t, qv in zip(tasks, qvecs):
        chunks = retrieve(tbl, qv, k=K)
        got = [c["article_id"] for c in chunks]
        gold = str(t["gold_article_id"])
        recall_hits.append(int(gold in got))                       # eq. (1.4)
        rank = next((i + 1 for i, a in enumerate(got) if a == gold), None)
        recip_ranks.append(1.0 / rank if rank else 0.0)            # eq. (1.5)

        para.append(correct(ask(gen, t["question"]), t["gold_answer"]))
        rag.append(correct(ask(gen, assemble_prompt(t["question"], chunks)),
                           t["gold_answer"]))

    acc_para = es.bootstrap_mean(para, name="acc_parametric")
    acc_rag = es.bootstrap_mean(rag, name="acc_rag")
    delta = es.bootstrap_paired_diff(rag, para, name="rag_minus_parametric")
    mcn = es.mcnemar(rag, para)
    report = {
        "n": len(tasks), "k": K,
        "retrieval": {"recall_at_k": round(float(np.mean(recall_hits)), 4),
                      "mrr": round(float(np.mean(recip_ranks)), 4)},
        "end_task": {"parametric": acc_para.to_dict(),
                     "rag": acc_rag.to_dict(),
                     "paired_delta": delta.to_dict(),
                     "mcnemar_p": mcn.pvalue, "mcnemar_effect": mcn.effect},
        "provenance": "measured on the baseline machine -- record value, date, driver",
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run("../data/artifacts/sda_qualitative_tasks.jsonl")
```

```admonish gotcha title="Retrieval leakage is contamination (ties to 3.8)"
A RAG gain is only interesting if it is grounding, not leakage. Here is the trap. The eval questions in this chapter were *written from* the articles in the corpus (that is what makes `gold_article_id` knowable for recall@$k$). So when retrieval works, the top chunk often contains the answer *verbatim*, and the model's job collapses from "reason about space" to "copy span [1]." A jump in end-task accuracy under those conditions is not evidence of reasoning; it is the pipeline-leakage failure mode of 3.8 wearing a RAG costume, and the recall@$k$/end-task correlation being near 1 is the tell. Two defenses, both mandatory. First, **hold the corpus and the eval-authoring apart**: draw the gold answer from a source the retrievable corpus does not contain verbatim, or transform the question so the answer must be *inferred* from the chunk rather than lifted from it, so a win requires reading, not matching. Second, **report retrieval metrics next to end-task metrics** precisely so a reviewer can see whether the answer tracked the retrieval one-for-one (lookup) or improved more modestly and unevenly (grounding plus reasoning). This is exactly the "is the RAG gain reasoning or lookup?" question that 8.3 answers causally with the augmentation arms, and it is why RAG-access is drawn as a candidate confound in Part IV rather than trusted as a clean intervention. Numbers you could compute, you never retrieve (that is 8.2's tools); text you retrieve, you make sure the model still has to *read*.
```

**What you should see.** Two artifacts land in `serve/rag/artifacts/`: the LanceDB index (`space_rag.lance/`) and the report (`rag_report.json`). The report carries the two families of numbers *separately*, which is the whole point. Retrieval quality shows up as a recall@5 and an MRR: a healthy small-corpus retriever puts the gold article in the top 5 for most queries (recall@5 well above the parametric-only floor) and usually near the top (MRR close to recall@5 means the gold chunk is ranking first; a large gap means a reranker would help). End-task quality shows up as two accuracies with bootstrap intervals and a *paired* delta with its own CI and a McNemar p-value, because RAG and parametric ran on the same items and 3.7's gotcha forbids comparing two marginal CIs by eye. The interesting reading is the *relationship* between the two families: if recall@5 is high but the paired end-task delta's CI straddles zero, retrieval is finding the evidence and the generator is failing to use it (a prompt-assembly or attention problem, not a retrieval one); if the end-task delta is large and tracks recall almost exactly, re-read the leakage gotcha before celebrating, because you may have measured lookup. The measured deltas here are placeholders (measured on the baseline machine, record value, date, driver); what is not a placeholder is the discipline of reporting retrieval and end-task metrics as two separate columns and treating the RAG arm as a suspect intervention rather than a free win. That framing is what hands a clean question to 8.3.

```admonish read-along
[AIE] ch. 6 (RAG and agents) is the production-side companion to this chapter: it treats chunking strategy, embedding-model choice, hybrid (dense + lexical) retrieval, and reranking as the engineering decisions they are, and its evaluation section makes the same split this chapter insists on between retrieval quality and answer quality. Read it for the breadth of options I deliberately narrowed (I chose one embedding model, one metric, exact search, and a thin loop); read this chapter for why, in a thesis where retrieval is a variable in a causal claim, the narrow legible version is the one worth building.

For depth on RAG itself, **[RAG] Rothman, *RAG-Driven Generative AI*** is the dedicated text: its naive, advanced, and modular framing, its retrieval metrics, and its cosine-similarity evaluation map almost one to one onto this chapter's pipeline. **[LC] Auffarth & Kuligin**'s RAG-systems chapter covers the LangChain-shaped build with hybrid and agentic retrieval, and **[GADP] Lakshmanan & Hapke**'s Basic-RAG and index-aware patterns (Patterns 6 and 9) are the pattern-catalog view. I cite the three for the breadth of retrieval technique they lay out, and this chapter for the single legible slice of it I chose to make retrieval a clean variable in a causal claim.
```

```admonish substack-seed
"RAG is for what you can't compute, not for what you can." The most common RAG mistake in a technical domain is retrieving a number. If a question has a formula (a closest-approach distance, an orbital period, a decay date), the right answer comes from running the formula behind a tool, not from finding a sentence that happens to state it, because a retrieved number is unverified, unsourced, and possibly stale. Retrieval earns its keep on the questions with no closed form: what an agency said, why a maneuver happened, what is new since the model's training cutoff. And even there, the honest evaluation reports two numbers, not one: did the retriever find the evidence (recall@k, MRR), and did finding it actually change the answer for the better (a paired end-task delta with a confidence interval). Those two numbers come apart constantly, and the gap between them, high retrieval with no answer gain, or an answer gain that is really just verbatim lookup, is where every interesting RAG bug and every RAG-shaped self-deception lives.
```
