# Data and curriculum for RLVR

By now the machinery is built. Chapter 7.2 got a GRPO job running inside the 16GB budget on the baseline machine, and chapter 7.3 wired the Part III verifier scorers in as reward functions. What I have not done yet is decide what to train on. That sounds like a footnote and it is not. In supervised fine-tuning the data is the whole ballgame, but the failure mode is obvious: bad labels, bad model. In reinforcement learning from verifiable rewards the data failure modes are quieter and nastier, because a badly chosen prompt does not produce a wrong gradient, it produces no gradient at all, and a run that is silently learning nothing looks almost exactly like a run that is learning slowly. This chapter is about choosing the prompt set so that the gradient signal is actually there, so that it stays there as the policy improves, and so that not one item of it has leaked from the eval suite I am going to measure the delta against.

The output is a concrete artifact on disk: a training prompt set with a provenance manifest that a committee member could audit line by line. But first I need to explain why the obvious approach (grab a big math dataset and point the trainer at it) wastes most of a single GPU's very limited time.

## Theory

### Why difficulty is the load-bearing variable in RLVR

Recall the shape of the GRPO update from chapter 7.2. For a single prompt $x$, the policy generates a group of $K$ completions, each verifier-scored to a reward $r_i$. The group-relative advantage of completion $i$ is the reward centered and scaled by the group's own statistics:

$$
A_i = \frac{r_i - \bar r}{\sigma_r + \varepsilon}, \qquad \bar r = \frac{1}{K}\sum_{i=1}^{K} r_i, \qquad \sigma_r = \sqrt{\frac{1}{K}\sum_{i=1}^{K}(r_i - \bar r)^2}.
\tag{5.1}
$$

Now stare at the numerator. If every completion in the group gets the same reward, then $r_i = \bar r$ for all $i$, every advantage $A_i$ is zero, and the policy-gradient contribution of that prompt for that step is exactly zero. With a binary correctness reward $r_i \in \{0,1\}$ this happens whenever the prompt is either solved by every rollout or by none of them. The prompt cost you $K$ forward generations, which on a 16GB card is the single most expensive thing the loop does, and it bought you nothing.

So the quantity that actually matters is not "is this a good math problem" but "under the current policy, does this prompt produce disagreement among $K$ rollouts". Let $p$ be the model's per-sample solve probability on prompt $x$. The rewards in a group are $K$ Bernoulli draws, and the expected within-group reward variance, which is what feeds the advantage magnitude, is

$$
\mathbb{E}[\sigma_r^2] = \frac{K-1}{K}\, p(1-p).
\tag{5.2}
$$

That $p(1-p)$ is the whole story of difficulty filtering. It is zero at $p=0$ and $p=1$ and maximized at $p=0.5$. A prompt the model already solves every time ($p \approx 1$) is as useless for training as one it never solves ($p \approx 0$), and for the identical reason: no disagreement, no signal. The informative prompts live in a band around the middle.

```admonish derivation
The probability that a group of $K$ binary rewards is degenerate (all same, hence zero advantage) is $P_{\text{deg}}(p) = p^K + (1-p)^K$. The probability a prompt contributes any gradient at all is $1 - P_{\text{deg}}(p)$. For $K=8$ this is above $0.9$ across roughly $0.2 \le p \le 0.8$, and it falls off a cliff outside that: at $p = 0.95$ it is only about $0.34$, and at $p = 0.99$ about $0.077$. In plain terms, with a group size of 8, a prompt the model solves 95% of the time contributes a usable gradient on barely a third of the steps it appears in. That is the arithmetic justification for a solve-rate band of roughly $[0.2, 0.8]$: it is not a taste, it is where $1 - P_{\text{deg}}$ is large. Larger $K$ widens the usable band (more rollouts, more chances for disagreement) at linear cost in generation time, which is exactly the trade chapter 7.7 ablates.
```

### Solve-rate bands are a measurement, and they drift

Here is the subtlety that separates a curriculum from a static filter. The solve probability $p$ is a property of the prompt *and the current policy together*. At the start of a run I estimate $p$ against the baseline policy $M_0$. But the entire point of training is to raise $p$. A prompt that sits at $p = 0.4$ against the baseline, comfortably in band, will drift toward $p = 0.9$ as the model learns it, and at that point it has left the band and gone quiet. This is the RLVR version of a curriculum: not a fixed ordering I impose, but a moving band the policy walks through as it improves.

There are two honest ways to handle the drift, and I will build for both:

The static-band approach estimates $p$ once against $M_0$, keeps everything in $[p_{\text{lo}}, p_{\text{hi}}]$, and accepts that late in training some prompts will have gone quiet. It is simple, fully reproducible, and its provenance is a single dated measurement. For a one-GPU thesis run of a few hundred steps this is usually enough, because the policy does not move far enough to empty the band.

The online-rebanding approach periodically re-estimates $p$ against the current checkpoint (every few hundred steps, using the rollouts the trainer is already generating so it costs nothing extra) and drops prompts that have crossed $p_{\text{hi}}$, optionally promoting harder prompts from a reserve pool. It keeps the signal dense for longer but its provenance is a schedule of measurements, not one, and it introduces a moving part that chapter 7.7 will want to hold fixed when ablating other things. I default to the static band and treat online rebanding as an ablation, not the baseline.

### Curriculum ordering: does presentation order matter

Given a set of in-band prompts, does the order I feed them in matter? In SFT, curriculum ordering (easy-to-hard) has a mixed empirical record. In RLVR the case is a bit cleaner because of equation (5.2): early in training the model is weak, so the mass of prompts that sit near $p = 0.5$ *for the weak model* are the objectively easier ones. Feeding easy-to-hard is really just feeding "whatever is currently in band first," which is the same online-rebanding idea seen from the ordering side rather than the filtering side.

For a fixed static band I mostly shuffle, because with group-relative advantages the per-prompt signal does not depend on what came before it in the way a value-function bootstrap would. The one ordering choice I do make deliberately is to avoid long runs of near-identical prompts in a single gradient-accumulation window, because that correlates the advantages inside a batch and inflates the effective step size on one narrow skill. A shuffle with a fixed seed handles that for free and keeps the run reproducible.

### Prompt-set size versus epochs

RLVR blurs the SFT notion of an epoch. In SFT, a second epoch shows the model the exact same target tokens again, and overfitting is memorization of those targets. In RLVR, a second pass over a prompt generates *fresh* rollouts and scores them anew, so the model never sees a fixed target to memorize. That makes revisiting prompts far safer than an SFT epoch, but it is not free of risk: the risk is reward hacking and narrow specialization (chapter 7.4's whole subject). A tiny prompt set drilled for many passes will find whatever degenerate strategy maximizes the verifier on those specific items, and it will generalize poorly.

The quantity that a step budget actually fixes is total prompt-exposures. With $N$ unique in-band prompts, $E$ passes, group size $K$, and a per-step prompt count $B$ (the number of distinct prompts per optimizer step), the number of optimizer steps is

$$
S = \frac{N \cdot E}{B}, \qquad \text{total generations} = S \cdot B \cdot K = N \cdot E \cdot K.
\tag{5.3}
$$

On the baseline machine the wall-clock cost is dominated by those $N \cdot E \cdot K$ generations, so for a fixed time budget the product $N \cdot E$ is essentially fixed and the only real decision is how to split it. My rule of thumb, which chapter 7.7 turns into an actual ablation rather than a rule of thumb, is: prefer larger $N$ and $E$ close to 1 until diversity stops helping, because breadth is the cheap insurance against reward hacking, and only spend extra passes once the prompt set is large enough that the model is not memorizing verifier quirks. A few hundred to a couple thousand unique in-band prompts with one to three passes is the region a single-GPU run lives in; I will not pretend to a measured optimum I have not run.

### Dedup against the eval suite is not optional

This is the part a committee will actually try to break. In chapter 7.6 I am going to claim a reasoning delta: metric $M$ moved by $\Delta$ from $M_0$ to $M_1$ on the frozen thesis task suite v1.0 (chapter 3.9). That claim is worth exactly nothing if any training prompt overlaps the eval suite, because then I trained on the test and the "delta" is partly just memorized eval items leaking back. The entire causal story of the thesis (chapter 4.5's control runs exist to protect it) collapses at the first contaminated prompt.

So dedup against the eval suite is a hard gate, and I reuse the contamination machinery from chapter 3.8 rather than inventing a weaker one here. The layers, from cheapest to most thorough:

- Exact-match dedup on the raw prompt string, after Unicode normalization and whitespace collapse. Catches copy-paste duplicates.
- Normalized-match dedup after stripping formatting, lowercasing, and canonicalizing numbers and LaTeX, so that "What is 2+2?" and "what is $2 + 2$" collide. Catches trivially reskinned items.
- N-gram Jaccard overlap: flag any training prompt whose set of character or token n-grams has Jaccard similarity above a threshold with any eval prompt. Catches paraphrase and partial reuse.
- Embedding cosine similarity: embed every training and eval prompt, flag any training prompt within a cosine threshold of any eval prompt. Catches semantic near-duplicates that share no n-grams.

Every flagged prompt is *removed*, and the counts removed at each layer go into the provenance manifest, because "we ran four dedup layers and removed this many at each" is a sentence that survives a committee, and "we deduped" is not.

## Tooling

The tool this chapter builds is a prompt-set builder that turns a candidate source dataset into a training prompt set plus a provenance manifest, running the difficulty estimation against the same served substrate the rest of the loop uses (the chapter 2.6 inference baseline) and the same dedup module the eval side uses (chapter 3.8). It is deliberately boring: a filter is only trustworthy if you can read the whole thing.

Set up the project with uv, in the training environment, never bare pip:

```bash title="shell: create the curriculum builder project"
uv init curriculum
cd curriculum
uv add "datasets>=2.19" "openai>=1.30" "numpy>=1.26" "scikit-learn>=1.4" \
       "sentence-transformers>=3.0" "pydantic>=2.7"
```

The `openai` client is there only because the chapter 2.6 vLLM serving substrate speaks the OpenAI API, so difficulty estimation talks to the local server exactly the way an eval does. Nothing here reaches the network except the one pinned dataset download, and that download records its revision hash.

### The provenance manifest as a first-class object

The manifest is not a log I write at the end. It is a typed object I fill in as I go, so that it is impossible to ship a prompt set whose provenance is incomplete. That is the discipline the whole book runs on (the hardware baseline chapter made the same argument for measured numbers): a value comes stamped or it comes marked as not yet measured, and there is no third category.

```python title="curriculum/provenance.py"
"""Typed provenance manifest for a training prompt set.

Every field here is something a committee could ask about. If a field is
empty at write time, the builder refuses to save. Provenance is a gate,
not a garnish.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SourceSpec(BaseModel):
    name: str                      # e.g. "open-r1/OpenMathReasoning"
    revision: str                  # exact dataset revision / commit hash
    split: str
    license: str
    n_raw: int                     # items before any filtering


class DifficultySpec(BaseModel):
    policy_model: str              # the M0 served on the 2.6 substrate
    policy_checkpoint: str         # path or hash of the served weights
    served_endpoint: str           # e.g. "http://localhost:8000/v1"
    k_rollouts: int
    temperature: float
    max_tokens: int
    band_lo: float
    band_hi: float
    # measured on the baseline machine -- record value, date, driver
    driver: str = "record: e.g. 570-open"
    measured_at: str = "record ISO date"


class DedupSpec(BaseModel):
    eval_suite: str                # e.g. "thesis-suite v1.0 (chapter 3.9)"
    eval_suite_revision: str
    removed_exact: int
    removed_normalized: int
    removed_ngram: int
    removed_embedding: int
    ngram_jaccard_threshold: float
    embedding_cosine_threshold: float


class Manifest(BaseModel):
    prompt_set_name: str
    built_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    machine: str = "MSI Aegis R2 (Core Ultra 9 285, RTX 5080 16GB, Ubuntu 24.04)"
    source: SourceSpec
    difficulty: DifficultySpec
    dedup: DedupSpec
    n_final: int
    seed: int

    def gate(self) -> None:
        """Refuse to represent an incomplete provenance record."""
        if self.n_final <= 0:
            raise ValueError("empty prompt set after filtering; refuse to save")
        if "record" in self.difficulty.measured_at:
            raise ValueError("difficulty measurement date not stamped")
```

### Difficulty estimation against the served baseline

Difficulty estimation is just a small eval: for each candidate prompt, ask the local server for $K$ completions, verifier-score each with the *same* scorer the trainer uses (imported from the chapter 7.3 reward module, so the definition of "solved" is identical on both sides), and record the empirical solve rate $\hat p = \frac{1}{K}\sum r_i$.

```python title="curriculum/difficulty.py"
"""Estimate per-prompt solve rate against the served baseline policy.

Talks to the 2.6 vLLM substrate over the OpenAI API. The scorer is
imported from the 7.3 reward module so 'solved' means exactly what it
means to the trainer -- no second definition to drift.
"""
from __future__ import annotations

import numpy as np
from openai import OpenAI

# The verifier from chapter 7.3, reused verbatim as the difficulty judge.
from rewards import verify_correct  # (x_prompt, completion, gold) -> 0.0 | 1.0


def solve_rate(
    client: OpenAI,
    model: str,
    prompt: str,
    gold: str,
    k: int,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> float:
    """Empirical per-sample solve probability p_hat from k rollouts."""
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        n=k,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    rewards = [verify_correct(prompt, c.text, gold) for c in resp.choices]
    return float(np.mean(rewards))


def in_band(p_hat: float, lo: float, hi: float) -> bool:
    return lo <= p_hat <= hi
```

A note on cost, because it is a single GPU: difficulty estimation is $N_{\text{raw}} \cdot K$ generations, which for a large candidate set can rival the training run itself. Two mitigations keep it honest. First, estimate on a shorter `max_tokens` budget than training uses, since you only need a solve/no-solve verdict, not a polished chain; this is a measured trade (record on the baseline machine — record value, date, driver) but it is usually safe. Second, estimate with a modest $K$ (say 4 to 8): $\hat p$ is a noisy estimate of $p$, and a prompt that lands at $\hat p = 0$ from 4 rollouts might really be $p = 0.15$, so I keep the band edges loose rather than surgical and let a few borderline prompts through rather than paying for a precise $\hat p$ I do not need.

### Dedup, reusing the 3.8 module

I do not reimplement contamination checks here. I import them, because the eval side and the training side must use the identical definition of "overlap" or the gate leaks.

```python title="curriculum/dedup.py"
"""Dedup a candidate prompt set against the frozen eval suite.

Reuses the chapter 3.8 contamination module so 'overlap' is defined
once. Returns the surviving prompts and per-layer removal counts for
the provenance manifest.
"""
from __future__ import annotations

from dataclasses import dataclass

# All four checks come from the 3.8 module; do not redefine them here.
from contamination import (
    normalize_text,
    exact_duplicate_ids,
    ngram_jaccard_overlap,   # returns ids over a Jaccard threshold
    embedding_overlap,       # returns ids over a cosine threshold
)


@dataclass
class DedupResult:
    survivors: list[dict]
    removed_exact: int
    removed_normalized: int
    removed_ngram: int
    removed_embedding: int


def dedup_against_eval(
    candidates: list[dict],
    eval_prompts: list[str],
    ngram_threshold: float = 0.6,
    cosine_threshold: float = 0.85,
) -> DedupResult:
    eval_norm = {normalize_text(p) for p in eval_prompts}

    survivors, r_exact, r_norm = [], 0, 0
    for item in candidates:
        norm = normalize_text(item["prompt"])
        if item["prompt"] in eval_prompts:          # raw exact
            r_exact += 1
            continue
        if norm in eval_norm:                        # normalized exact
            r_norm += 1
            continue
        survivors.append(item)

    ngram_hits = ngram_jaccard_overlap(
        [s["prompt"] for s in survivors], eval_prompts, threshold=ngram_threshold
    )
    survivors2 = [s for i, s in enumerate(survivors) if i not in ngram_hits]

    emb_hits = embedding_overlap(
        [s["prompt"] for s in survivors2], eval_prompts, threshold=cosine_threshold
    )
    survivors3 = [s for i, s in enumerate(survivors2) if i not in emb_hits]

    return DedupResult(
        survivors=survivors3,
        removed_exact=r_exact,
        removed_normalized=r_norm,
        removed_ngram=len(ngram_hits),
        removed_embedding=len(emb_hits),
    )
```

## Lab: build the training prompt set with documented provenance

Now the pieces come together into one script that a committee could run and audit. It downloads a pinned candidate dataset, estimates difficulty against the served baseline, bands it, dedups against the frozen eval suite, and writes two files: the prompt set and its manifest. The gate refuses to write if provenance is incomplete.

```python title="curriculum/build_prompt_set.py"
"""Build a provenance-stamped RLVR training prompt set.

Pipeline:
  1. load a pinned candidate source dataset
  2. estimate solve rate vs the served baseline (2.6 substrate)
  3. keep prompts inside the solve-rate band
  4. dedup against the frozen eval suite (3.8 module, 3.9 suite)
  5. write prompt set + provenance manifest (gated)

No measured throughput/time numbers are invented here; where a real
measurement belongs it is stamped:
  (measured on the baseline machine -- record value, date, driver)
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

from difficulty import solve_rate, in_band
from dedup import dedup_against_eval
from provenance import Manifest, SourceSpec, DifficultySpec, DedupSpec

# ---- configuration (design choices, each justified in the Theory section) ----
SOURCE_NAME = "open-r1/OpenMathReasoning"   # example; swap for your source
SOURCE_REVISION = "PIN_ME_to_a_commit_hash" # exact revision, not "main"
SOURCE_SPLIT = "train"
SOURCE_LICENSE = "record from the dataset card"

POLICY_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # the M0 you serve on 2.6
SERVED_ENDPOINT = "http://localhost:8000/v1"
K_ROLLOUTS = 6
TEMPERATURE = 0.8
MAX_TOKENS = 512                             # short: verdict, not polish
BAND_LO, BAND_HI = 0.2, 0.8
SEED = 0

EVAL_SUITE_PATH = "../thesis_suite_v1.jsonl" # frozen 3.9 suite
EVAL_SUITE_REV = "thesis-suite v1.0"
OUT_DIR = Path("out")


def load_eval_prompts(path: str) -> list[str]:
    return [json.loads(l)["prompt"] for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(base_url=SERVED_ENDPOINT, api_key="not-needed-for-local-vllm")

    ds = load_dataset(SOURCE_NAME, revision=SOURCE_REVISION, split=SOURCE_SPLIT)
    n_raw = len(ds)

    # ---- difficulty banding against the served baseline ----
    banded: list[dict] = []
    for row in ds:
        prompt, gold = row["problem"], row["answer"]
        p_hat = solve_rate(
            client, POLICY_MODEL, prompt, gold,
            k=K_ROLLOUTS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED,
        )
        if in_band(p_hat, BAND_LO, BAND_HI):
            banded.append({"prompt": prompt, "gold": gold, "p_hat": p_hat})

    # ---- dedup against the frozen eval suite ----
    eval_prompts = load_eval_prompts(EVAL_SUITE_PATH)
    dd = dedup_against_eval(banded, eval_prompts)
    final = dd.survivors

    # ---- provenance (gated) ----
    manifest = Manifest(
        prompt_set_name="sda_train_prompts.v1",
        source=SourceSpec(
            name=SOURCE_NAME, revision=SOURCE_REVISION, split=SOURCE_SPLIT,
            license=SOURCE_LICENSE, n_raw=n_raw,
        ),
        difficulty=DifficultySpec(
            policy_model=POLICY_MODEL,
            policy_checkpoint="record: served weights hash",
            served_endpoint=SERVED_ENDPOINT,
            k_rollouts=K_ROLLOUTS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            band_lo=BAND_LO, band_hi=BAND_HI,
            driver="record: e.g. 570-open",
            measured_at="record: ISO date of this run",  # gate() will refuse if left
        ),
        dedup=DedupSpec(
            eval_suite="thesis-suite v1.0 (chapter 3.9)",
            eval_suite_revision=EVAL_SUITE_REV,
            removed_exact=dd.removed_exact,
            removed_normalized=dd.removed_normalized,
            removed_ngram=dd.removed_ngram,
            removed_embedding=dd.removed_embedding,
            ngram_jaccard_threshold=0.6,
            embedding_cosine_threshold=0.85,
        ),
        n_final=len(final),
        seed=SEED,
    )
    manifest.gate()  # raises unless provenance is complete and non-empty

    # ---- write artifacts ----
    with (OUT_DIR / "sda_train_prompts.v1.jsonl").open("w") as f:
        for item in final:
            f.write(json.dumps(item) + "\n")
    (OUT_DIR / "provenance.json").write_text(manifest.model_dump_json(indent=2))

    print(f"raw={n_raw}  in_band={len(banded)}  final={len(final)}")
    print(f"removed: exact={dd.removed_exact} norm={dd.removed_normalized} "
          f"ngram={dd.removed_ngram} emb={dd.removed_embedding}")


if __name__ == "__main__":
    main()
```

Run it with the baseline served on the 2.6 substrate:

```bash title="shell: build the prompt set (baseline must be serving)"
# in one terminal: serve M0 exactly as chapter 2.6 specifies
# vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 ...
uv run python build_prompt_set.py
```

### What you should see

Two files under `curriculum/out/`. The first, `sda_train_prompts.v1.jsonl`, is the training prompt set: every line a prompt, its gold answer, and the measured $\hat p$ that put it in band, with nothing outside $[0.2, 0.8]$ and nothing that survived the dedup gate against the eval suite. The second, `provenance.json`, is the manifest, and it is the artifact that actually matters for the thesis. It names the source dataset and its exact revision, records the served policy and endpoint that difficulty was measured against, states the band edges and rollout budget, and reports how many candidate prompts each of the four dedup layers removed. The console line will read something like `raw=N in_band=M final=L` with `M < N` (most raw prompts fall outside the band, which is the point) and `L <= M` (dedup only ever removes). If `final=0`, the gate throws and nothing is written, which is the builder refusing to hand you a prompt set it cannot stand behind.

The honest number to watch is the ratio `in_band / raw`. If it is very small, your candidate source is mostly too easy or too hard for $M_0$ and you are paying a lot of generation time to keep a little data; if it is close to 1, either your source is already difficulty-calibrated or your band is too loose. Either way it is a measurement (record on the baseline machine — record value, date, driver), and it tells you something real about the match between your data and your model before you have spent a single training step.

```admonish gotcha
The most common way this gate silently leaks is estimating difficulty against a *different* checkpoint than the one chapter 7.6 will call $M_0$. If you difficulty-filter against an instruct model but then start GRPO from a slightly different base, your band is measured against the wrong policy and the "in-band" guarantee is fiction. Pin one checkpoint as $M_0$, serve exactly that for difficulty estimation, and start training from exactly that. The `policy_checkpoint` hash in the manifest exists so this mismatch is auditable rather than invisible.
```

```admonish substack-seed
There is a genuinely counterintuitive post here: in reinforcement learning from verifiable rewards, the problems your model already solves are worthless, and so are the problems it can never solve. All the learning happens in the narrow band where it succeeds maybe a third to two-thirds of the time, because that is the only place the rollouts disagree, and disagreement is the whole signal. It is the pedagogy of the zone of proximal development, except you can measure the zone with one number, $p(1-p)$, and watch it move as the student gets better. The hook: a good curriculum is not the hardest problems or the easiest, it is the ones that are currently a coin-flip, and that band walks up the difficulty ladder on its own as the model climbs.
```
