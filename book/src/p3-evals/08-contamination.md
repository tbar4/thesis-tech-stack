# Contamination and dataset hygiene

Here is the failure that makes every other number in this book meaningless. You run an eval, the model scores 82%, you write it down as evidence of reasoning ability, and it turns out the test items were sitting in the model's pretraining corpus verbatim. The model did not reason; it remembered. In the open-weights era this is not a rare accident, it is the default condition you have to actively rule out, because the pretraining corpora are enormous, undisclosed, and scraped from exactly the public web where benchmarks live. Contamination inflates scores, and worse, it inflates them *unevenly* across models depending on what each one memorized, which corrupts the very A-versus-B comparisons that Chapter 3.7 taught us to measure carefully. A confidence interval around a contaminated score is a precise measurement of the wrong thing. So before a dataset is allowed anywhere near the thesis suite, it gets scanned, and this chapter builds the scanner.

## Theory

### What leakage is, and the three places it hides

Leakage is any path by which information about the test answers reaches the model before test time. It hides in three places, and you have to defend all three.

The first is **pretraining contamination**: test items are in the corpus the model was trained on. You cannot inspect that corpus (it is closed or effectively unsearchable), so you cannot prove absence directly; you detect it by proxy and by behavior. The second is **your own pipeline leakage**: in a serve-evaluate-score-train loop you *generate* data and train on it, and if a test item (or a near-duplicate of one) sneaks into the training pool, you have contaminated your own eval with your own hands. This one you *can* control completely, and controlling it is the main hygiene job in this book. The third is **benchmark-internal duplication**: the eval set contains near-duplicate items, so a model that memorizes one gets several "independent" items for free, which breaks the independence assumption behind every CI and test in Chapter 3.7.

Contamination also comes in grades of subtlety. **Verbatim** contamination is the exact item text. **Near-duplicate** contamination is a paraphrase, a reformatting, a translated or lightly-edited version, which slips past exact matching but still hands the model the answer. **Answer/solution leakage** is the sneakiest: the question is novel but its worked solution appeared somewhere, so the model has seen the reasoning even if it has not seen the prompt. A good scan catches the first two mechanically; the third needs behavioral probing.

### Detecting verbatim and near-duplicate overlap: n-grams

The workhorse for verbatim and light-paraphrase overlap is n-gram matching, and it is worth deriving the object rather than treating it as a black box.

```admonish derivation
**Shingling and Jaccard overlap.** Represent a document $D$ as its set of contiguous word $n$-grams (its "shingles"). For $n = 13$ (the length GPT-3's contamination analysis used) the document

$$\text{"the north region fell from 812 to 640 in the last fiscal"}$$

becomes the set of overlapping 13-token windows $S_n(D)$. The overlap between a test item $T$ and a training document $R$ is the **Jaccard similarity** of their shingle sets,

$$ J(T, R) = \frac{|S_n(T) \cap S_n(R)|}{|S_n(T) \cup S_n(R)|}. \tag{8.1}$$

$J = 1$ means identical shingle sets (verbatim duplication up to shingle granularity); $J = 0$ means no shared $n$-gram. The choice of $n$ trades false positives against false negatives: small $n$ (say 5) flags common phrases and over-reports; large $n$ (13+) fires only on substantial verbatim overlap and under-reports paraphrase. A common contamination rule is to flag any test item that shares even a *single* long $n$-gram with the corpus, i.e. $|S_n(T)\cap S_n(R)| \ge 1$ for $n = 13$, which is a stricter and cheaper signal than a Jaccard threshold and is the one worth using for a leakage flag.
```

Comparing every test item against every training document by set intersection is $O(|T|\cdot|R|)$ and does not scale. **MinHash with locality-sensitive hashing (LSH)** turns the Jaccard comparison into a hash lookup: MinHash gives each document a short signature whose collision probability equals its Jaccard similarity, and LSH buckets similar signatures together so you only compare documents that land in the same bucket. That is how you scan a small eval set against a large training pool on a CPU in minutes instead of days.

### Detecting paraphrase: embeddings

N-grams are blind to a paraphrase that shares no long span (`"which region had the biggest revenue drop"` versus `"identify the area with the largest decline in earnings"`). Embedding-based near-duplicate detection catches these: encode each item with a sentence embedding model, and flag pairs whose cosine similarity exceeds a threshold. Cosine similarity of embeddings $u, v$ is $\cos(u,v) = \frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}$, and a threshold around 0.85 to 0.9 catches genuine paraphrase while tolerating incidental topical similarity, though the right threshold is dataset-dependent and you calibrate it by eyeballing the borderline pairs. The two methods are complementary: n-grams catch verbatim overlap embeddings can miss (a long shared span in an otherwise different document dilutes cosine), and embeddings catch semantic overlap n-grams miss. A real scan runs both.

```admonish gotcha
The embedding model is itself trained on the public web, so it can be "contaminated" in a way that matters for you: it might map two of your test items close together because it memorized their shared source, not because they are genuinely paraphrases. This is mostly harmless for near-duplicate *detection* (a false positive just gets a human look), but do not turn around and use the same embedding model as a *judge* of semantic novelty and call that independent evidence. Keep the contamination checker and any downstream judge as separate model families, for the same self-preference reason Chapter 3.6 gave.
```

### Behavioral probing for the leakage you cannot search

When you cannot search the training corpus (the usual case), you probe the model's behavior for memorization. The cheap, honest version: take a test item, mask its final answer or a salient span, and ask the model to fill it in with greedy decoding. If the model completes a *novel* item's masked span at chance and completes a *suspected-contaminated* item's masked span exactly and confidently, that gap is evidence of memorization. More formal membership-inference methods (min-$k$% token probability, guided-versus-unguided prompting) exist and are worth citing in a thesis, but the masked-completion probe is enough to raise a flag that then gets a human decision. Behavioral probing does not prove contamination; it produces a suspicion score that moves an item from "assumed clean" to "manually reviewed."

### Dataset versioning and revision pinning

None of the above means anything if the dataset can change under you. The hygiene rule is that an eval dataset is an *immutable, content-addressed artifact*, and you achieve that in two layers. First, **pin the source revision**: Hugging Face datasets are git repositories with commit hashes, so you load `revision="<commit-sha>"`, never a bare name that silently tracks `main`. Second, **content-hash the frozen set yourself**: after loading and any cleaning, compute a hash over the canonicalized items and record it, so that "the dataset I evaluated on" is a specific 64-hex-character string and not a moving target. Anything you cannot pin and hash, you do not evaluate on. This is the same discipline the hardware baseline chapter applied to measured numbers (a number without a date is a rumor); a dataset without a revision and a content hash is a rumor too.

## Tooling

The scan is deliberately CPU-friendly so it runs on the baseline machine without competing with a served model for VRAM. `datasets` handles revision-pinned loading. `datasketch` provides MinHash and MinHashLSH for the scalable n-gram near-duplicate pass. For the embedding pass, a small sentence-transformer (a `bge-small`-class model, on the order of 130M parameters, a few hundred MB) fits trivially and can run on CPU for a few hundred items, or on the GPU in seconds if one is free; either way it is not the memory pressure in this book. The whole scanner is a few hundred lines and produces one JSON report plus a JSONL of flagged pairs for human review. It does not *decide* to drop items; it surfaces evidence and a recommended action, and a human signs off, because dropping eval items is a decision with statistical consequences (it shrinks $n$, which Chapter 3.7 showed costs you power).

## Lab

We scan a thesis-candidate dataset for leakage on three axes: internal near-duplication, overlap against the training pool we intend to use, and a behavioral memorization probe on a sample. The candidate here is a public verifiable-reasoning set (GSM8K-style math word problems stand in as an open, license-clean example); the training pool is whatever corpus the thesis loop will fine-tune on. The scan emits a contamination report.

### Project setup

```bash title="setup.sh"
uv init contamscan && cd contamscan
uv add "datasets>=2.19" "datasketch>=1.6" "numpy>=1.26"
uv add "sentence-transformers>=3.0"   # small embedding model, CPU-ok
```

### Pinned loading and canonicalization

```python title="contamscan/load.py"
"""Load datasets at pinned revisions and content-hash them."""
import hashlib, json, re
from datasets import load_dataset

def canonical(text: str) -> str:
    """Normalize so trivial formatting differences do not hide overlap."""
    return re.sub(r"\s+", " ", text.strip().lower())

def content_hash(items) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(canonical(it).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def load_pinned(name: str, split: str, revision: str, field: str):
    """revision MUST be a commit sha, never a bare branch name."""
    ds = load_dataset(name, split=split, revision=revision)
    items = [canonical(r[field]) for r in ds]
    return items, {"name": name, "split": split, "revision": revision,
                   "n": len(items), "content_sha256": content_hash(items)}
```

### The n-gram / MinHash pass

```python title="contamscan/ngram.py"
"""13-gram MinHash-LSH near-duplicate detection (Eq. 8.1)."""
from datasketch import MinHash, MinHashLSH

N = 13          # shingle length; see derivation
NUM_PERM = 128  # MinHash signature length (accuracy vs speed)

def shingles(text: str, n: int = N):
    toks = text.split()
    if len(toks) < n:
        return {text}                      # short item: whole-string shingle
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}

def minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m

def build_index(pool_items, threshold=0.5):
    """Index the training pool; threshold is Jaccard for candidate pairs."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    sigs = {}
    for i, txt in enumerate(pool_items):
        m = minhash(txt)
        sigs[i] = m
        lsh.insert(f"pool-{i}", m)
    return lsh, sigs

def scan_against_pool(test_items, lsh, sigs, jaccard_flag=0.8):
    """Flag test items whose nearest pool item exceeds jaccard_flag."""
    hits = []
    for j, txt in enumerate(test_items):
        m = minhash(txt)
        for key in lsh.query(m):            # LSH bucket candidates only
            pool_i = int(key.split("-")[1])
            jac = m.jaccard(sigs[pool_i])   # Eq. 8.1 (MinHash estimate)
            if jac >= jaccard_flag:
                hits.append({"test_idx": j, "pool_idx": pool_i,
                             "jaccard": round(jac, 3), "channel": "ngram"})
    return hits

def internal_dups(test_items, jaccard_flag=0.8):
    """Flag near-duplicate pairs *within* the eval set (breaks independence)."""
    lsh, sigs = build_index(test_items, threshold=0.5)
    seen, hits = set(), []
    for j, txt in enumerate(test_items):
        m = minhash(txt)
        for key in lsh.query(m):
            k = int(key.split("-")[1])
            if k == j or (min(j, k), max(j, k)) in seen:
                continue
            jac = m.jaccard(sigs[k])
            if jac >= jaccard_flag:
                seen.add((min(j, k), max(j, k)))
                hits.append({"a_idx": j, "b_idx": k,
                             "jaccard": round(jac, 3), "channel": "internal"})
    return hits
```

### The embedding pass

```python title="contamscan/embed.py"
"""Cosine near-duplicate detection for paraphrase overlap."""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"   # ~130M params, CPU-friendly

def embed(items, batch_size=64):
    model = SentenceTransformer(MODEL)
    v = model.encode(items, batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=False)
    return np.asarray(v)            # already L2-normalized -> dot = cosine

def scan_paraphrase(test_items, pool_items, cos_flag=0.88):
    """Flag test/pool pairs with cosine >= cos_flag (paraphrase leakage)."""
    T = embed(test_items); R = embed(pool_items)
    sims = T @ R.T                  # cosine, since both normalized
    hits = []
    for j in range(sims.shape[0]):
        i = int(np.argmax(sims[j]))
        c = float(sims[j, i])
        if c >= cos_flag:
            hits.append({"test_idx": j, "pool_idx": i,
                         "cosine": round(c, 3), "channel": "embed"})
    return hits
```

### Orchestration and the report

```python title="contamscan/scan.py"
"""Run all passes, write the contamination report + flagged pairs."""
import json
from pathlib import Path
from .load import load_pinned
from .ngram import build_index, scan_against_pool, internal_dups
from .embed import scan_paraphrase

def run(test_spec, pool_spec, out_dir="runs"):
    test_items, test_meta = load_pinned(**test_spec)
    pool_items, pool_meta = load_pinned(**pool_spec)

    lsh, sigs = build_index(pool_items)
    ngram_hits = scan_against_pool(test_items, lsh, sigs)
    internal_hits = internal_dups(test_items)
    embed_hits = scan_paraphrase(test_items, pool_items)

    flagged_test_idx = sorted(
        {h["test_idx"] for h in ngram_hits + embed_hits}
        | {i for h in internal_hits for i in (h["a_idx"], h["b_idx"])})
    n = test_meta["n"]
    report = {
        "test": test_meta, "pool": pool_meta,
        "counts": {
            "ngram_overlap_items": len({h["test_idx"] for h in ngram_hits}),
            "paraphrase_overlap_items": len({h["test_idx"] for h in embed_hits}),
            "internal_duplicate_pairs": len(internal_hits),
            "total_flagged_items": len(flagged_test_idx),
        },
        "leakage_rate": round(len(flagged_test_idx) / n, 4) if n else 0.0,
        "thresholds": {"ngram_n": 13, "jaccard_flag": 0.8, "cosine_flag": 0.88},
        "recommended_action": (
            "manual review of flagged_pairs.jsonl; drop or replace confirmed "
            "leaks; re-run power analysis (Ch 3.7) after any drop since n shrinks"),
        "provenance": "run on the baseline machine -- record date, driver",
    }
    Path(out_dir).mkdir(exist_ok=True)
    Path(f"{out_dir}/contamination_report.json").write_text(
        json.dumps(report, indent=2))
    with open(f"{out_dir}/flagged_pairs.jsonl", "w") as f:
        for h in ngram_hits + embed_hits + internal_hits:
            f.write(json.dumps(h) + "\n")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    # Pin real commit shas from the dataset's HF page before running.
    test_spec = dict(name="openai/gsm8k", split="test",
                     revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
                     field="question")
    pool_spec = dict(name="openai/gsm8k", split="train",
                     revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
                     field="question")
    run(test_spec, pool_spec)
```

### The behavioral memorization probe

The two mechanical passes catch overlap you can see in the text; they cannot catch the item whose prompt is genuinely novel but whose *answer* the model memorized from a solution posted elsewhere. For that you probe behavior. The cheap, honest version masks a salient span of the item (here, the final numeric answer embedded in a worked target) and asks the model, greedily, to complete it. A model that reconstructs the exact masked answer far more often on suspected items than its own accuracy would predict is showing memorization, not reasoning. This does not prove contamination; it raises a flag that a human then adjudicates.

```python title="contamscan/probe.py"
"""Masked-answer completion probe: does the model *reconstruct* the held-out span?"""
import re
from openai import OpenAI

def mask_answer(target: str) -> tuple[str, str]:
    """Replace the last number in the target with a blank; return (masked, gold)."""
    nums = list(re.finditer(r"-?\d+(?:\.\d+)?", target))
    if not nums:
        return target, ""
    last = nums[-1]
    masked = target[:last.start()] + "____" + target[last.end():]
    return masked, last.group(0)

def probe(items, base_url="http://localhost:8000/v1",
          model="Qwen/Qwen3-14B-AWQ", flag_rate=0.5):
    """Fraction of items whose masked answer the model fills in exactly.
    Greedy decoding so the result is a property of memory, not sampling."""
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    hits = []
    for it in items:
        masked, gold = mask_answer(it["target"])
        if not gold:
            continue
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=16,
            messages=[{"role": "user", "content":
                       f"Fill in the blank exactly.\n{it['question']}\n"
                       f"Answer: {masked}"}])
        if gold in (r.choices[0].message.content or ""):
            hits.append({"id": it["id"], "reconstructed": gold,
                         "channel": "behavioral_probe"})
    rate = len(hits) / max(len(items), 1)
    return {"reconstruction_rate": round(rate, 4),
            "suspected": hits, "flagged": rate >= flag_rate,
            "note": "high reconstruction vs task accuracy => review for leakage; "
                    "measured on the baseline machine -- record date, driver"}
```

Wire the probe's `suspected` list into the same human-review queue the n-gram and embedding passes feed. The interpretation is comparative, not absolute: a reconstruction rate meaningfully above the model's own answer accuracy on the same items is the signal, because a model that can *reason out* the answer will also fill the blank, so only the excess over its earned accuracy is evidence of memorized leakage.

```admonish gotcha
The `revision` shas above are placeholders. Before you run this, open the dataset's Hugging Face page, copy the actual commit sha of the revision you intend to freeze, and paste it in. A bare `revision="main"` (or omitting it) means the dataset can change between your scan and your eval, which defeats the entire point: you would have certified a version you did not actually evaluate. Pin the sha, then record it in the report, then never change it without bumping the suite version in Chapter 3.11.
```

### The artifact and what you should see

The artifact is `runs/contamination_report.json` plus `runs/flagged_pairs.jsonl`. The report records both datasets' names, pinned revisions, and content hashes, the counts from each detection channel (verbatim/near-duplicate n-gram overlap against the pool, paraphrase overlap by embedding, and internal duplicate pairs), an overall leakage rate, the thresholds used, and a recommended action. The flagged-pairs file is the human-review queue: each row names a test item and its most-similar pool item with the similarity that flagged it. On a clean, well-constructed candidate you should see a leakage rate at or near zero and only a handful of borderline embedding hits that turn out to be genuinely different problems that happen to share a topic. On a contaminated candidate you will see clusters of high-Jaccard n-gram hits (near-verbatim overlap) and a leakage rate that is alarmingly non-trivial, at which point the disciplined move is not to quietly delete the flagged items but to review them, decide item by item, and if you drop any, re-run the Chapter 3.7 power analysis because a smaller $n$ buys you less power to detect the training delta you care about. A dataset that passes this scan, with its revision pinned and its content hashed, is a dataset you are allowed to freeze into the thesis suite. The actual leakage rate for the thesis candidate is (measured on the baseline machine -- record value, date, driver).

```admonish read-along
[BRM] builds a reasoning model from scratch and is candid about how much benchmark performance in the wild is memorization rather than reasoning; read its discussion of evaluation alongside this chapter. The framing that pays off for the thesis is that a *verifiable* task with freshly constructed or transformed items has a small contamination surface by design, which is one more reason the next chapter builds the suite out of checkable, novel items rather than borrowing a famous benchmark wholesale. **[AIE]** ch. 8 (dataset engineering) is the production-side companion: deduplication, decontamination, and data lineage as engineering disciplines rather than afterthoughts.
```

```admonish substack-seed
When a new open model posts a huge benchmark score, the first question is not "how did it reason so well" but "did it see the test set during pretraining." In the open-weights era contamination is the default hypothesis, not the edge case, because the training corpora are the same scraped web where the benchmarks live. You cannot search a closed corpus, but you can do the next best thing in an afternoon: scan your eval set for verbatim overlap with 13-gram MinHash, catch paraphrases with sentence embeddings, probe the model with masked-answer completion, and pin your dataset to an exact revision hash so "the test set" is a specific 64-character string and not a moving target. A score you cannot certify as uncontaminated is not a measurement of reasoning; it is a measurement of memory.
```
