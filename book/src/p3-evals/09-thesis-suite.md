# Building the thesis task suite

This is where the last four chapters cash out. We have metrics that mean something (3.2), tasks and scorers that run (3.3, 3.4), a harness for comparability (3.5), a calibrated judge for the residue that resists exact checking (3.6), the statistics that tell signal from noise (3.7), and a scan that certifies a dataset is not memorized (3.8). Now I use all of it to build and freeze the one object the rest of the thesis is measured against: the task suite, version 1.0, tagged and immutable. Everything downstream (the quantization comparisons, the training deltas, the "did reasoning actually improve" claim that the whole book is pointed at) is a measurement taken with this instrument, so the instrument has to be built with the same care you would build a scale you plan to weigh evidence on. This is the Thesis Thread's capstone for Part III: the SDA-flavored example stops being an illustration and becomes a frozen, versioned, datasheeted artifact.

## Theory

### What makes a task suitable for a verifiable-reasoning suite

The defining property is that a response can be *checked, not just liked*. A task earns its place in this suite only if there is a programmatic verifier that maps a model response to correct-or-not without a human and without a judge's opinion. That is the whole reason the thread is "SDA-flavored": these are structured-data reasoning tasks where the answer is a number, a set, a boolean, or a normalized string that a small deterministic checker can validate. Judges (Chapter 3.6) are held in reserve for partial-credit analysis of the reasoning trace; they never decide the primary correctness signal, because a suite whose ground truth depends on a model's opinion inherits that model's biases and cannot anchor a training reward.

Beyond verifiability, four properties separate a good item from a bad one. It must have **unambiguous ground truth**: exactly one correct answer under a stated normalization, or a checker that accepts exactly the set of correct forms. It must be **shortcut-resistant**: a model should not be able to guess it from surface cues or answer it without doing the reasoning, which means distractors and structure that punish pattern-matching. It must sit at a **useful difficulty**: not so easy every model aces it (ceiling, zero discrimination) nor so hard every model fails (floor, also zero discrimination). And it must have a **small contamination surface**: freshly constructed or programmatically transformed items, so the Chapter 3.8 scan comes back clean by construction rather than by luck.

### Difficulty calibration

An item's difficulty is not a property you assign, it is a property you *measure*, and the cheapest measurement is the empirical pass rate of a reference model. Borrowing the vocabulary of item response theory without the full machinery: each item has a **difficulty** (what fraction of capable models get it right) and a **discrimination** (how well it separates stronger models from weaker ones). An item every model passes and an item every model fails both have zero discrimination and are dead weight in a suite meant to detect a training delta, because the delta lives entirely in the items that some models get and some do not. So you pilot: run a reference model (the baseline candidate) over a draft pool, record per-item pass rates, and keep a spread. I bin into three strata (easy: reference pass rate above 0.8; medium: 0.4 to 0.8; hard: below 0.4) and stock all three, because a suite that is all-hard has no ceiling to climb toward and a suite that is all-easy has no signal. The stratification also lets you report per-difficulty deltas later, which is far more informative than a single aggregate number.

### Sizing the suite from the power analysis

Here the thread reconnects to Chapter 3.7, and this is the load-bearing quantitative decision of the chapter. The power analysis (Eq. 7.11) told us exactly how many shared items it takes to detect a paired accuracy gain at 80% power and $\alpha = 0.05$, given a discordant rate $\psi \approx 0.25$: about **194 items** for a ten-point gain, **305 items** for an eight-point gain, and **783 items** for a five-point gain. These are not round numbers I picked; they came out of the derivation and are checked by `evalstats.required_n_mcnemar` in the test suite. Now I trade detectable effect against the cost of hand-verifying items.

```admonish derivation
**Choosing the suite size.** The thesis needs to detect training deltas that are scientifically interesting but not enormous; sub-five-point gains are hard to argue matter, and ten-point gains would be a very strong result I should not design *only* for. Targeting an **eight-point** detectable paired gain sets the required test size at $n = 305$ shared items (Eq. 7.11 with $\delta = 0.08$, $\psi = 0.25$, power 0.80, $\alpha = 0.05$; the formula gives 304.23 and `np.ceil` rounds up to 305). I round to **300 test items**, stratified as 100 easy / 100 medium / 100 hard, plus a separate **60-item dev slice** for prompt and pipeline iteration that is never scored in a reported result. Rounding 305 down to 300 shaves the power by a negligible amount: redo Eq. 7.10's logic and the achieved power at $n = 300$ for $\delta = 0.08$ is within a fraction of a percent of 0.80, well inside the slack already present in the $\psi = 0.25$ assumption.

Two escape hatches are worth stating now so v2.0 has somewhere to go. If a five-point gain must be detectable, the suite has to grow to ~780 items, roughly $2.5\times$ the hand-verification cost. Alternatively, drawing multiple completions per item and analyzing with the clustered bootstrap (Chapter 3.7) reduces within-item variance and buys back some power without adding items, at the cost of more inference. v1.0 commits to 300 items and an eight-point target; the escape hatches are documented, not taken.
```

The dev/test split is a hygiene decision as important as the size. The 300 test items are frozen and never used to tune a prompt, pick a checkpoint, or debug the harness, because every look at the test set with an eye to improving a number is a tiny act of overfitting the eval. All iteration happens on the 60-item dev slice. This is the same firewall a machine-learning practitioner puts between a validation and a test set, applied to eval engineering.

### Freezing v1.0

Freezing is what turns a folder of JSONL into a citable artifact. It has four parts. **Content-hash** every item and the suite as a whole, so "suite v1.0" is a specific SHA-256 and not a description. **Pin provenance**: for every item, record where it came from (generated by which procedure, or transformed from which pinned dataset revision) and its license. **Write a datasheet**: following the Datasheets for Datasets discipline, document motivation, composition, collection, preprocessing, recommended uses, and known limitations, so a future reader (including future me) knows what the suite is and is not for. **Tag it in git**: a signed, immutable `suite-v1.0` tag whose commit contains the frozen JSONL, the manifest with hashes, and the datasheet. After the tag, the rule is absolute: v1.0 never changes. A correction is v1.1, a growth is v2.0, and every reported result names the exact version it used.

## Tooling

The suite lives as JSONL in the Inspect-compatible schema from Chapter 3.3, so the same task and scorer machinery runs it. Each item carries the fields the scorer and the statistics need: a stable `id`, the `domain` and `difficulty` stratum, the `input`/`question`, the `target` answer, a `verifier` type naming which deterministic checker validates it, and `provenance`/`license`. Content hashing uses `hashlib`; MLflow logs the suite version, hashes, and per-stratum counts as a run so the freeze event itself is tracked on the spine. The verifiers are small pure functions (exact-match after normalization, set equality, numeric-with-tolerance, boolean) kept in one module so the checker for every item is auditable in one place.

## Lab

We build the suite, pilot it to calibrate difficulty, freeze it to v1.0 with a manifest and datasheet, and tag it. The generation of items is domain-specific; I show the schema, the verifiers, the difficulty-calibration pilot, and the freeze machinery, with a small illustrative item set that a real run scales to 360 items (300 test + 60 dev).

### Project setup

```bash title="setup.sh"
uv init thesis-suite && cd thesis-suite
uv add "numpy>=1.26" "openai>=1.40"        # openai client to hit the served model
uv add "mlflow>=2.14"
uv add --editable ../evalstats             # the module from Chapter 3.7
```

### The item schema and verifiers

```python title="suite/verifiers.py"
"""Deterministic checkers. The primary correctness signal is never a judge."""
import re
from fractions import Fraction

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def exact_match(response: str, target: str) -> bool:
    return _norm(response) == _norm(target)

def numeric(response: str, target: str, tol: float = 1e-6) -> bool:
    """Extract the last number in the response; compare to target within tol."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(Fraction(target))) <= tol
    except (ValueError, ZeroDivisionError):
        return False

def set_equal(response: str, target: str) -> bool:
    """Compare comma-separated sets, order-insensitive, after normalization."""
    r = {_norm(x) for x in response.split(",") if x.strip()}
    t = {_norm(x) for x in target.split(",") if x.strip()}
    return r == t

def boolean(response: str, target: str) -> bool:
    truthy = {"yes", "true", "1"}; falsy = {"no", "false", "0"}
    r = _norm(response)
    tgt = _norm(target) in truthy
    # grab the model's yes/no verdict token
    hit = next((w for w in re.findall(r"[a-z]+", r) if w in truthy | falsy), None)
    return hit is not None and (hit in truthy) == tgt

VERIFIERS = {"exact": exact_match, "numeric": numeric,
             "set": set_equal, "boolean": boolean}

def check(item: dict, response: str) -> bool:
    return VERIFIERS[item["verifier"]](response, item["target"])
```

```jsonl title="suite/draft/items_sample.jsonl"
{"id": "sda-0001", "domain": "table-reasoning", "verifier": "exact", "input": "Region,Q1,Q2\nNorth,812,640\nWest,900,705\nEast,540,560", "question": "Which region had the largest quarter-over-quarter revenue drop?", "target": "West", "provenance": "generated:table-synth-v1", "license": "CC0-1.0"}
{"id": "sda-0002", "domain": "sql-semantics", "verifier": "numeric", "input": "Table t(amount, status). Rows: (120,paid),(90,paid),(200,unpaid),(150,paid),(80,paid)", "question": "How many rows satisfy amount>100 AND status='paid'?", "target": "2", "provenance": "generated:sql-synth-v1", "license": "CC0-1.0"}
{"id": "sda-0003", "domain": "schema-normalization", "verifier": "boolean", "input": "R(id PK, zip, city); zip -> city holds.", "question": "Is R in third normal form? Answer yes or no.", "target": "no", "provenance": "generated:schema-synth-v1", "license": "CC0-1.0"}
{"id": "sda-0004", "domain": "set-reasoning", "verifier": "set", "input": "A={1,2,3,4,6}; B={2,4,5,6,8}", "question": "List the elements in A but not in B, comma-separated.", "target": "1, 3", "provenance": "generated:set-synth-v1", "license": "CC0-1.0"}
```

### The difficulty-calibration pilot

```python title="suite/pilot.py"
"""Run the reference model over the draft pool; record per-item pass rate."""
import json
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from suite.verifiers import check

REF_MODEL = "Qwen/Qwen3-14B-AWQ"   # the baseline reference; NOT the judge here
N_SAMPLES = 4                       # completions per item for a stable pass rate

def pass_rate(client, item):
    correct = 0
    for _ in range(N_SAMPLES):
        r = client.chat.completions.create(
            model=REF_MODEL,
            messages=[{"role": "user",
                       "content": f"{item['input']}\n\n{item['question']}"}],
            temperature=0.7, max_tokens=512)
        if check(item, r.choices[0].message.content):
            correct += 1
    return correct / N_SAMPLES

def stratum(p):
    return "easy" if p > 0.8 else ("hard" if p < 0.4 else "medium")

def run(draft="suite/draft/items_sample.jsonl", base_url="http://localhost:8000/v1"):
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    items = [json.loads(l) for l in Path(draft).read_text().splitlines()]
    counts = defaultdict(int)
    for it in items:
        p = pass_rate(client, it)
        it["ref_pass_rate"] = round(p, 3)
        it["difficulty"] = stratum(p)
        counts[it["difficulty"]] += 1
    Path("suite/calibrated").mkdir(parents=True, exist_ok=True)
    Path("suite/calibrated/items.jsonl").write_text(
        "\n".join(json.dumps(it) for it in items))
    print("stratum counts:", dict(counts),
          "\nref pass rates: measured on the baseline machine -- record date, driver")
    return items

if __name__ == "__main__":
    run()
```

The pilot uses `N_SAMPLES = 4` completions per item so the pass rate is a stable difficulty estimate rather than a single coin flip, and it uses temperature 0.7 to reflect how the model is actually decoded in the eval, not a degenerate greedy pass. A real run curates the draft pool until each stratum has enough items to hit the target counts (100 test items per stratum plus dev-slice headroom), pruning items that land at pass rate exactly 0 or 1 as non-discriminating.

### Freezing and the manifest

```python title="suite/freeze.py"
"""Freeze the calibrated suite: split, content-hash, manifest, MLflow, git tag."""
import json, hashlib, subprocess, random
from pathlib import Path
import mlflow

VERSION = "1.0"
TEST_PER_STRATUM = 100
DEV_TOTAL = 60
SEED = 20260722   # frozen split seed, recorded so the split is reproducible

def sha256_items(items):
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x["id"]):
        h.update(json.dumps(it, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()

def split(items):
    by = {"easy": [], "medium": [], "hard": []}
    for it in items:
        by[it["difficulty"]].append(it)
    rng = random.Random(SEED)
    test, dev = [], []
    dev_per = DEV_TOTAL // 3
    for s in ("easy", "medium", "hard"):
        pool = by[s][:]; rng.shuffle(pool)
        dev += pool[:dev_per]
        test += pool[dev_per:dev_per + TEST_PER_STRATUM]
    return test, dev

def freeze():
    items = [json.loads(l) for l in
             Path("suite/calibrated/items.jsonl").read_text().splitlines()]
    test, dev = split(items)
    out = Path(f"suite/frozen/v{VERSION}"); out.mkdir(parents=True, exist_ok=True)
    (out / "test.jsonl").write_text("\n".join(json.dumps(i) for i in test))
    (out / "dev.jsonl").write_text("\n".join(json.dumps(i) for i in dev))
    manifest = {
        "name": "thesis-task-suite",
        "version": VERSION,
        "split_seed": SEED,
        "counts": {"test": len(test), "dev": len(dev),
                   "test_per_stratum": {
                       s: sum(i["difficulty"] == s for i in test)
                       for s in ("easy", "medium", "hard")}},
        "sizing_rationale": {
            "target_detectable_delta": 0.08, "power": 0.80, "alpha": 0.05,
            "assumed_discordant_rate": 0.25,
            "required_n_from_ch37": 305, "chosen_n": 300,
            "source": "evalstats.required_n_mcnemar (Eq. 7.11)"},
        "content_sha256": {"test": sha256_items(test), "dev": sha256_items(dev)},
        "verifiers": sorted({i["verifier"] for i in items}),
        "created": "record date, driver on the baseline machine",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    mlflow.set_experiment("thesis-suite")
    with mlflow.start_run(run_name=f"freeze-v{VERSION}"):
        mlflow.log_params({"version": VERSION, "split_seed": SEED,
                           "chosen_n": 300, "target_delta": 0.08})
        mlflow.log_metrics({"n_test": len(test), "n_dev": len(dev)})
        mlflow.set_tags({"suite_test_sha256": manifest["content_sha256"]["test"]})
        mlflow.log_artifacts(str(out), artifact_path=f"suite-v{VERSION}")
    print(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    m = freeze()
    # Tag the freeze in git so v1.0 is immutable.
    subprocess.run(["git", "add", f"suite/frozen/v{VERSION}"], check=True)
    subprocess.run(["git", "commit", "-m",
                    f"Freeze thesis task suite v{VERSION} "
                    f"(n_test={m['counts']['test']}, "
                    f"sha={m['content_sha256']['test'][:12]})"], check=True)
    subprocess.run(["git", "tag", "-a", f"suite-v{VERSION}", "-m",
                    f"Thesis task suite v{VERSION}"], check=True)
```

### The datasheet

```markdown title="suite/frozen/v1.0/DATASHEET.md"
# Datasheet: Thesis Task Suite v1.0

## Motivation
Built to measure verifiable-reasoning ability of open-weight models across the
serve-evaluate-score-train loop, and to serve as the fixed instrument against
which every training delta in this thesis is measured.

## Composition
300 test items + 60 dev items, SDA-flavored structured-data reasoning
(table reasoning, SQL semantics, schema normalization, set reasoning).
Each item has exactly one programmatically verifiable answer. Stratified
100/100/100 easy/medium/hard by reference-model pass rate.

## Sizing
n_test = 300 chosen to detect an 8-point paired accuracy gain at 80% power,
alpha = 0.05, assumed discordant rate 0.25 (evalstats, Eq. 7.11 -> 305, rounded).

## Collection & preprocessing
Items generated or programmatically transformed (small contamination surface);
scanned with the Chapter 3.8 contamination pipeline (13-gram MinHash + embedding).
Difficulty calibrated with a 4-sample pilot at temperature 0.7 on the reference
model. Recorded content SHA-256 in manifest.json.

## Recommended uses & limits
USE: paired A/B comparisons on shared items with the Chapter 3.7 statistics.
DO NOT: tune prompts, pick checkpoints, or debug on test.jsonl (use dev.jsonl).
Known limits: single domain family; verifier normalization may reject unusual
but correct answer forms (audit false negatives before reporting).

## Maintenance
v1.0 is immutable (git tag suite-v1.0). Corrections -> v1.1; growth -> v2.0.
Every reported result cites the version and content hash it used.
```

### The `thesis_suite` convenience package

The `suite/` package above is the builder: it calibrates, freezes, and datasheets. But the chapters downstream (the judge calibration in 3.6, the reward core in Part VII) do not want to reach into `suite/frozen/` and re-implement item loading and verifier dispatch every time. So I add one thin package, `thesis_suite`, that wraps the frozen split and the audited verifiers behind a single stable surface. It builds nothing new: `load_suite` reads the frozen JSONL, and every correctness call routes back into `suite.verifiers` (Chapter 3.9) or the extract-and-compare reward core (Chapter 3.4), so the semantics are exactly what was validated here. Downstream code imports `thesis_suite` and never has to know where the JSONL lives or which checker an item uses.

```python title="thesis_suite/__init__.py"
"""Stable import surface over the frozen thesis task suite (v1.0) and its
verifiers. Chapters 6.x (judges) and 7.x (rewards) import from HERE, not from
suite internals. Thin by design: all real logic lives in suite/verifiers.py
(Ch 3.9) and the frozen JSONL under suite/frozen/ (this chapter)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from suite.verifiers import VERIFIERS, exact_match, numeric

_FROZEN = Path(__file__).resolve().parent.parent / "suite" / "frozen"
_DEFAULT_REV = "v1.0"


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str          # input + question, exactly as the model is asked
    answer: str          # the gold target
    difficulty: str      # easy | medium | hard
    verifier: str        # which deterministic checker validates it


@dataclass(frozen=True)
class Suite:
    items: list[Item]
    revision: str


def _to_item(raw: dict) -> Item:
    prompt = (f"{raw['input']}\n\n{raw['question']}"
              if raw.get("input") else raw["question"])
    return Item(id=raw["id"], prompt=prompt, answer=raw["target"],
                difficulty=raw.get("difficulty", "medium"),
                verifier=raw.get("verifier", "exact"))


def load_suite(rev: str | None = None) -> Suite:
    """Load the frozen TEST split at revision `rev` (default v1.0)."""
    rev = rev or _DEFAULT_REV
    path = _FROZEN / rev / "test.jsonl"
    raw = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return Suite(items=[_to_item(r) for r in raw], revision=rev)


def verify(item: Item, response: str) -> bool:
    """Primary correctness signal for an item: dispatch to its own verifier
    (Ch 3.9). This is the number every reported score is built from."""
    return VERIFIERS[item.verifier](response, item.answer)


def verify_sda_answer(response: str, target: str) -> bool:
    """Low-level extract-and-compare reward core (Ch 3.4): pull the answer out
    of a free-form response and compare to `target` -- numeric extraction first
    (the dominant SDA answer form), exact-match after normalization otherwise.
    This is the pure function Part VII turns into a training reward."""
    return numeric(response, target) or exact_match(response, target)


def verify_correct(prompt: str, response: str, answer: str) -> bool:
    """Verify a (prompt, response, answer) triple when the caller has no Item
    in hand (6.x, 7.x). `prompt` is accepted for call-site symmetry; correctness
    routes through the same reward core as `verify_sda_answer`."""
    return verify_sda_answer(response, answer)


def score_model(client, suite: Suite, *, model: str = "Qwen/Qwen3-14B-AWQ",
                temperature: float = 0.0, max_tokens: int = 512,
                seed: int = 0) -> list[dict]:
    """Score `model` (served behind an OpenAI-compatible `client`) over the
    suite; return one row per item with response, correctness, and difficulty.
    Greedy by default so the score is a fixed measurement, not a coin flip."""
    rows = []
    for it in suite.items:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": it.prompt}],
            temperature=temperature, max_tokens=max_tokens, seed=seed)
        resp = r.choices[0].message.content
        rows.append({"id": it.id, "response": resp,
                     "correct": verify(it, resp), "difficulty": it.difficulty})
    return rows


__all__ = ["Item", "Suite", "load_suite", "verify", "verify_sda_answer",
           "verify_correct", "score_model"]
```

The split of responsibility is deliberate: `verify` is the item-aware primary signal (it honors each item's declared checker, so a `set` item is graded set-wise and a `numeric` item numerically), while `verify_sda_answer` is the item-agnostic reward core that Part VII optimizes against. They agree on the SDA-flavored items the reward is trained on; keeping both means a downstream caller can grade against the frozen suite *or* score a free-form generation with the same package.

```admonish thesis-thread
The SDA thread has been a running illustration since the preface; here it becomes a real artifact. We now hold **thesis task suite v1.0**: 300 frozen, content-hashed, difficulty-stratified, contamination-scanned verifiable-reasoning items plus a 60-item dev slice, sized by the power analysis to resolve an eight-point paired gain, tagged `suite-v1.0` in git with a datasheet. From this point on, every claim the thesis makes ("INT4 matches BF16 on reasoning," "GRPO training moved the reasoning delta") is a paired comparison taken on *this exact instrument*, cited by version and hash. Part IV interrogates whether the deltas measured on it are causal; Parts V through VII generate and score against it and try to move its number. The instrument is built; the rest of the book is readings from it.
```

### The artifact and what you should see

The artifacts are `suite/frozen/v1.0/{test.jsonl, dev.jsonl, manifest.json, DATASHEET.md}` and the immutable git tag `suite-v1.0`, with the freeze event logged to MLflow. Running the pilot then the freeze, you should see the calibration pass sort items into easy/medium/hard by measured reference pass rate, the split place exactly 100 items per stratum into `test.jsonl` and 20 per stratum into `dev.jsonl`, and the manifest record two content hashes (test and dev) plus the full sizing rationale that traces the 300-item choice back to Eq. 7.11 and `evalstats.required_n_mcnemar`. The MLflow run stores the suite as a versioned artifact tagged with its test-set hash, and `git tag` makes v1.0 permanent. The one thing you must verify by eye before trusting the freeze is the verifier false-negative rate: spot-check that the deterministic checkers accept genuinely-correct answers in unusual forms, because a verifier that rejects a right answer silently deflates every score taken on the suite. Once that check passes, you have a frozen instrument, and the discipline for the rest of the book is simple: never score on the test set to make a decision, always cite the version and hash, and let the next parts of the loop do their work against a target that does not move. Actual per-stratum pass rates and the final content hashes are (measured on the baseline machine -- record value, date, driver).

```admonish read-along
[BRM] is the natural companion for suite design: it argues that a reasoning model is only as good as the verifiable signal you can construct, which is exactly why this suite is built from checkable items rather than judge-graded ones. Read its treatment of task construction next to this chapter's difficulty calibration; the two together are the argument for why a small, well-calibrated, verifier-backed suite beats a large, noisy, opinion-graded one for measuring a training delta.
```

```admonish substack-seed
Everyone wants to evaluate their model on a famous benchmark. Almost no one stops to size the benchmark to the question they are actually asking. If you want to prove a training run improved reasoning by, say, eight points, the power analysis tells you the answer before you spend a single GPU-hour: roughly 300 shared items, if a quarter of them flip between the two models. Fewer than that and a real improvement will hide inside the noise; many more and you are hand-verifying items you did not need. Building the suite backward from the effect you want to detect (then stratifying by difficulty, scanning for contamination, freezing to an immutable content hash, and writing a datasheet) is the difference between an instrument you can defend to a committee and a folder of JSON you hope is fine.
```
