# What an eval is

An eval is a measurement instrument. That is the whole thesis of Part III compressed into one sentence, and if you take nothing else from these chapters, take that. A thermometer does not rank the weather; it reports a number that stands in for a physical quantity, and the number is only useful if you understand what it measures, how precisely, and where it lies. An eval is the same kind of object pointed at a model. When I run GSM8K against my 8B model and read "62.3%", that number is a reading off an instrument, and my job as the person building the instrument is to know what construct it estimates, how much noise sits on the reading, and which failure modes make the reading a lie.

The leaderboard framing gets this exactly backwards. A leaderboard treats the number as a rank, a single scalar you sort descending, and the model at the top wins. That framing is fine for marketing and useless for engineering. I am not trying to win; I am trying to detect a 2-point change in reasoning quality after a training run on one GPU, and to know whether that 2 points is signal or the instrument wobbling. The rest of Part III is about building instruments precise enough that the deltas I care about are visible above the noise floor. This chapter is the conceptual groundwork: what the instrument is supposed to measure, the two things people constantly confuse when they say "measure a model", the shape of the benchmark landscape, and the ways instruments quietly fail.

```admonish read-along
Pair this chapter with **[BRM] Raschka, *Build a Reasoning Model*, chapter 3**, which frames evaluation as the feedback signal a reasoning model is trained against, and **[RLHF] Lambert, *RLHF*, Part 3**, on evaluation and its discontents. Raschka gives you the training-loop-facing view (the eval is the thing you optimize, so its flaws become the model's flaws); Lambert gives you the measurement-facing skepticism (why published numbers are so often not what they claim). Read this chapter as the bridge: the eval is simultaneously an instrument I read and, later in this book, a reward I optimize, and both roles demand the same construct discipline.
```

## Theory

### Construct validity: what is the number actually about

Borrow the vocabulary from psychometrics, because they have been arguing about this for a century longer than we have. A **construct** is the abstract thing you want to measure but cannot observe directly: "mathematical reasoning ability", "propensity to refuse harmful requests", "instruction-following". You cannot weigh mathematical reasoning on a scale. So you build an **operationalization**: a concrete, scoreable task that you claim stands in for the construct. GSM8K, a set of grade-school word problems scored by exact-match on the final number, is an operationalization of "grade-school arithmetic reasoning".

**Construct validity** is the degree to which the operationalization actually tracks the construct. It is the single most important property of an eval and the one most often ignored, because it cannot be read off the score. A model can score 90% on your benchmark for reasons that have nothing to do with the construct you named. Three classic threats:

- **Construct under-representation.** The task covers a narrow slice of the construct and you generalize the score to the whole thing. GSM8K is arithmetic word problems with small integers; a 90% there tells you almost nothing about multi-step proof, geometry, or reasoning over large numbers. Reporting it as "math ability" over-claims.
- **Construct-irrelevant variance.** The score moves for reasons unrelated to the construct. If a model scores higher because it happened to memorize the answer key (contamination, chapter 8), or because the answer-extraction regex is lenient toward its formatting, that variance is irrelevant to reasoning but shows up in the number.
- **Criterion mismatch.** The thing you score is not the thing you care about. If I care about "produces correct final answers" but score "contains the correct answer somewhere in the reasoning trace", a model that rambles the right number then contradicts itself scores as correct.

Construct validity is why "what an eval is" is not a philosophical warm-up. Every downstream decision, every quantization comparison, every training delta, is an inference from the number back to the construct, and that inference is exactly as sound as the validity of the instrument.

```admonish derivation title="The instrument has a noise floor, and it sets your minimum detectable effect"
Treat an accuracy eval as $N$ independent Bernoulli trials (one per item) with true success probability $p$. The measured accuracy $\hat{p}$ is the sample mean, so its sampling variance is

$$
\operatorname{Var}(\hat{p}) = \frac{p(1-p)}{N}. \tag{1}
$$

The standard error is $\operatorname{SE}(\hat{p}) = \sqrt{p(1-p)/N}$. This is the noise floor of the instrument from sampling alone, before adding decoding stochasticity or grader noise. At $p = 0.5$ (worst case) and $N = 500$ items,

$$
\operatorname{SE} = \sqrt{\frac{0.5 \cdot 0.5}{500}} = \sqrt{0.0005} \approx 0.0224, \tag{2}
$$

so about $\pm 2.2$ percentage points at one standard error, or roughly $\pm 4.4$ points at 95% confidence ($1.96\,\operatorname{SE}$). The blunt consequence: on a 500-item eval, a 2-point improvement is inside the noise. If I want to resolve a 2-point delta reliably I need either more items, a paired/variance-reduced comparison (chapter 7), or a lower-variance metric. Chapter 7 does this properly with McNemar's test and clustered bootstraps; the point here is that the number "62.3%" is meaningless without an $N$ and a construct next to it. Every reading gets units and an interval, or it is not a measurement.
```

### Capability versus propensity

The second confusion is between **capability** and **propensity**, and it is worth its own paragraph because conflating them produces some of the worst reasoning about models I see.

**Capability** is what the model *can* do under favorable conditions: best-of-$k$ sampling, a good prompt, tools available, maybe a hint. It answers "is the ability in there at all?" **Propensity** is what the model *tends* to do under realistic conditions: single sample, default prompt, no hand-holding. It answers "what will actually happen when I deploy this?" A model can have the capability to solve a problem (it gets it right 1 time in 20) and a near-zero propensity to solve it (it gets it right on the first try almost never). These are different constructs and they want different metrics.

This distinction is the reason `pass@k` and `pass@1` both exist and are not interchangeable (chapter 2 derives them). `pass@k` with large $k$ is a capability probe: it asks whether any of $k$ samples succeeds, which surfaces latent ability. `pass@1` (or `pass@1` averaged over samples) is a propensity measure: the everyday hit rate. For reasoning-model training this split is central, because RL with verifiable rewards (Part V) works precisely by taking capability that already exists at high $k$ and dragging it down into propensity at $k=1$. If your eval only measures propensity you cannot see the headroom RL has to work with; if it only measures capability you cannot see whether training actually moved everyday behavior. I want both readings, labeled, and I never let one masquerade as the other.

```admonish gotcha title="A single scalar hides the capability/propensity split"
The most common way to lie to yourself is to report one accuracy number and let the reader assume it means whatever they want. "The model gets 71% on math" could be `pass@1` (propensity) or `pass@8` majority-vote (something in between) or `pass@64` (capability), and these can differ by 20+ points on the same model and dataset. When I later compare a base model to its RL-trained descendant, the *interesting* result is often that `pass@64` barely moved while `pass@1` jumped: RL concentrated existing capability into propensity rather than teaching anything new. That story is invisible if I only logged one number. Always record $k$, the sampling temperature, and the aggregation rule alongside the score.
```

### The benchmark landscape and how instruments fail

The public benchmark ecosystem is large, fast-moving, and mostly built for leaderboards, which means most benchmarks are optimized for discrimination between frontier models rather than for stable measurement of a single model over a training run. Knowing the landscape is knowing which instruments to trust for which readings. The failure modes recur across all of them:

- **Saturation.** When the strong models cluster near 100%, the benchmark has stopped discriminating; the remaining variance is mostly label noise and formatting. MMLU and GSM8K are saturated or near it for frontier models in 2025, which is why harder successors (GPQA, MATH's harder splits, AIME-style problems) exist. For my 8B-to-20B repertoire, GSM8K is *not* saturated, which is exactly why it stays a useful instrument for me even as it stops being one for GPT-scale models. Saturation is relative to the model under test.
- **Contamination.** The test items, or close paraphrases, leaked into pretraining or finetuning data, so the model is partly recalling rather than reasoning. This inflates the score and destroys construct validity, and it is the default assumption for any popular benchmark released before your model's data cutoff. Chapter 8 is entirely about detecting and mitigating this.
- **Prompt and format sensitivity.** The same model on the same items can swing several points from a whitespace change, a different few-shot ordering, or a stricter answer-extraction regex. This is construct-irrelevant variance masquerading as capability. It is also why "published numbers differ" (chapter 5): two labs ran the same benchmark with different harness settings.
- **Answer-extraction brittleness.** For generative tasks you have to pull a final answer out of free text. A regex that expects `#### 42` scores a model that wrote `The answer is 42.` as wrong. The instrument is now measuring formatting compliance, not reasoning.
- **Metric-construct mismatch.** Exact-match on the final token ignores whether the reasoning was valid; a model that guesses right scores identically to one that reasoned right. For reasoning research this is often the wrong criterion, and it is why verifiable scorers and process-level checks (chapter 4) matter.
- **Overfitting the public split.** Because the test set is public, the whole field gradient-descends on it socially: architectures and recipes that happen to help that benchmark propagate. The benchmark slowly stops measuring the construct and starts measuring "resemblance to things that do well on this benchmark".

None of these are reasons to abandon benchmarks. They are reasons to treat every benchmark as an instrument with a datasheet: what it measures, on what population, with what known distortions, at what precision. I keep that datasheet next to every number, and building the habit of writing it down is the lab for this chapter.

### A working taxonomy

I organize evals into four families by construct, because the families demand different tooling and have different failure profiles. This taxonomy runs through the whole of Part III.

1. **Knowledge.** Does the model know facts? Closed-book QA, MMLU-style multiple choice, TriviaQA. Construct: retrieval and recall of stored information. Failure profile: heavily contamination-prone (facts are memorizable), saturates fast, and multiple-choice format lets a model score above chance by elimination without knowing the answer. Cheap to score (exact match on a letter).
2. **Reasoning.** Can the model carry out multi-step inference? GSM8K, MATH, AIME problems, logical-deduction tasks, code (which is reasoning you can execute). Construct: valid multi-step derivation toward a checkable answer. Failure profile: answer-extraction brittleness, and the deep mismatch between "right answer" and "right reasoning". This is the family the thesis lives in, because it is the family where a verifiable final answer gives me a clean, cheap, gameable-but-checkable reward signal for RL.
3. **Agentic.** Can the model act in an environment over multiple turns, using tools, to accomplish a task? SWE-bench (fix a real GitHub issue), WebArena, tool-use suites. Construct: goal-directed behavior under partial observability with real side effects. Failure profile: enormously expensive, high variance, sandbox and environment fragility, and scores that depend as much on the scaffold as on the model. Chapters 3 and 4 build toward this with Inspect's solver and sandbox machinery.
4. **Safety.** Does the model refuse what it should and comply with what it should, and is it robust to adversarial pressure? Refusal benchmarks, jailbreak suites, bias probes. Construct: this is the family where **propensity is the whole point**. I do not care whether the model *can* produce harmful content (it can); I care what it *tends* to do under realistic and adversarial prompting. Safety evals that report capability-style best-of-$k$ numbers are usually measuring the wrong thing.

The taxonomy is not clean at the edges (code is reasoning that is also agentic when you let it run tools; a knowledge question can require reasoning) but it is a good enough carving to organize which instrument I reach for and which failure modes I check first.

```admonish under-the-hood title="Why reasoning evals are the load-bearing family for this book"
The through-line of this whole book is the serve to evaluate to score to train loop, and the reason it closes on one GPU is that reasoning evals with verifiable final answers give a reward signal that is (a) cheap to compute, (b) programmatic and reproducible, and (c) hard to game *at the answer level* even though it is gameable at the process level. A math problem has a right number; a unit test passes or it does not; a symbolic identity holds or it does not. That is what lets the same scorer I write for measurement in chapter 4 become the reward in chapter 7.3. Knowledge evals are too contaminated and too saturated to reward against; agentic evals are too expensive to run in a training loop on 16GB; safety evals need human or strong-judge grading that is too slow and too noisy for per-rollout reward. Reasoning is the sweet spot, and the entire eval-engineering effort in Part III is really an effort to build trustworthy reasoning instruments that double as reward functions.
```

## Tooling

Two tools carry Part III, and this chapter is where I name them and say what each is for; chapters 3 through 5 go deep.

**Inspect** (`inspect_ai`, from the UK AI Safety Institute) is my instrument-building framework. Its mental model matches this chapter's ontology almost one-to-one: a `Task` binds a `Dataset` (the items), a `Solver` (how the model is prompted and, for agentic evals, how it acts over turns), and a `Scorer` (how a completed sample is turned into a number). It produces a structured, replayable log for every run, which is the datasheet-and-audit-trail I keep insisting on. Inspect is the right tool when I am building a *custom* instrument for a construct I care about, and when I need the run to be an auditable object I can re-score later without re-running the model. That last property is what makes it the on-ramp to using the same scorer as a reward.

**lm-evaluation-harness** (`lm-eval`, EleutherAI) is my comparability tool. It is a large library of pre-built, community-vetted task definitions (GSM8K, MMLU, HellaSwag, hundreds more) with fixed few-shot regimes and standardized scoring. Its value is precisely that it is *standardized*: when I want a number that is comparable to a published number, or a clean BF16-versus-INT4 delta on a canonical task, I reach for the harness so that my measurement choices match everyone else's. Chapter 5 is where I run it against my local vLLM endpoint via `local-completions` and produce the first committee-grade table.

The division of labor: harness for comparable, off-the-shelf readings against known instruments; Inspect for custom instruments, agentic solvers, verifiable scorers, and anything that will later feed the training loop. Both talk to the same OpenAI-compatible vLLM endpoint I stood up in Part II, so the model under test is served once and measured by either tool.

```admonish gotcha title="The tool does not confer validity"
It is tempting to think that because Inspect or the harness produced the number, the number is trustworthy. The tooling gives you reproducibility and auditability; it does not give you construct validity. A perfectly reproducible measurement of the wrong construct is a perfectly reproducible mistake. The tools make it easy to run the eval; they do not make the eval a good instrument. That remains a design responsibility, which is why this chapter comes before the tools.
```

## Lab

The artifact for a conceptual chapter is a conceptual artifact made concrete: an **eval datasheet**, a small machine-readable registry that forces me to write down, for each instrument, the construct it estimates, the family it belongs to, the metric, the population, and its known failure modes. This is the habit I want burned in before I run a single real eval: no number without a datasheet. The lab writes a validated registry to disk and a tiny script that refuses to let a datasheet be incomplete, so the discipline is enforced by code rather than by my good intentions.

Set up the project with `uv`, consistent with the two-environment doctrine from Part 0.

```bash
uv init evals-datasheet
cd evals-datasheet
uv add pydantic pyyaml
```

The registry is a YAML file. Each entry is one instrument. The point is that filling it in is uncomfortable: you cannot write "measures math" and move on, because the schema demands a construct statement, a population, and at least one named failure mode.

```yaml title="evals/registry.yaml"
# An eval datasheet. Every instrument I read from gets an entry here BEFORE
# I trust a number off it. This file is the audit trail for construct validity.
instruments:
  - id: gsm8k
    name: "GSM8K"
    family: reasoning            # knowledge | reasoning | agentic | safety
    construct: >
      Multi-step grade-school arithmetic reasoning over small integers,
      producing a single checkable final number.
    operationalization: >
      1319-item test split of natural-language word problems; score by
      exact match on the final integer after chain-of-thought.
    population: "Grade-school arithmetic word problems, small integers."
    metric: exact_match           # see chapter 2
    default_k: 1
    measures: propensity          # capability | propensity
    known_failure_modes:
      - "Answer-extraction brittleness: regex must tolerate varied final-answer phrasing."
      - "Contamination: widely reproduced; assume partial pretraining leakage."
      - "Not saturated for 8B-20B models, so still discriminating at this scale."
    notes: >
      Doubles as an RL reward source in chapter 7 because the final answer
      is cheaply and programmatically verifiable.

  - id: mmlu
    name: "MMLU"
    family: knowledge
    construct: "Broad closed-book factual recall across 57 subjects."
    operationalization: "4-way multiple choice; score exact match on chosen letter."
    population: "Academic and professional exam questions."
    metric: exact_match
    default_k: 1
    measures: capability
    known_failure_modes:
      - "Multiple-choice lets elimination beat true knowledge; chance floor is 25%."
      - "Saturated for frontier models; use with care as a discriminator."
      - "Contamination-prone: facts are directly memorizable."
    notes: "Comparability instrument; run via lm-eval for published-number parity."
```

Now the validator. It is a Pydantic model plus a script that loads the registry and fails loudly if any datasheet is incomplete. This is the enforcement: a registry entry that omits the construct or the failure modes is a runtime error, not a warning I ignore.

```python title="evals/datasheet.py"
"""Validate the eval datasheet registry.

Run: uv run python -m evals.datasheet evals/registry.yaml
Exits non-zero if any instrument is missing the fields that make a number
trustworthy: a construct statement, a population, a metric, and at least one
named failure mode. No number without a datasheet.
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class Family(str, Enum):
    knowledge = "knowledge"
    reasoning = "reasoning"
    agentic = "agentic"
    safety = "safety"


class Measures(str, Enum):
    capability = "capability"
    propensity = "propensity"


class Instrument(BaseModel):
    id: str
    name: str
    family: Family
    construct: str = Field(min_length=20)      # force a real sentence
    operationalization: str = Field(min_length=20)
    population: str = Field(min_length=5)
    metric: str
    default_k: int = Field(ge=1)
    measures: Measures
    known_failure_modes: list[str] = Field(min_length=1)  # at least one
    notes: str = ""

    @field_validator("known_failure_modes")
    @classmethod
    def _no_empty_modes(cls, v: list[str]) -> list[str]:
        if any(not m.strip() for m in v):
            raise ValueError("failure modes must be non-empty strings")
        return v


class Registry(BaseModel):
    instruments: list[Instrument]


def load(path: str | Path) -> Registry:
    data = yaml.safe_load(Path(path).read_text())
    return Registry.model_validate(data)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: datasheet.py <registry.yaml>", file=sys.stderr)
        return 2
    reg = load(argv[1])
    print(f"OK: {len(reg.instruments)} instrument(s) validated\n")
    for ins in reg.instruments:
        print(f"  [{ins.family.value:9}] {ins.name:8} "
              f"measures={ins.measures.value:11} "
              f"k={ins.default_k}  ({len(ins.known_failure_modes)} known failure mode(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

Run it.

```bash
uv run python -m evals.datasheet evals/registry.yaml
```

### What you should see

The script exits 0 and prints one line per instrument, something like:

```
OK: 2 instrument(s) validated

  [reasoning ] GSM8K    measures=propensity  k=1  (3 known failure mode(s))
  [knowledge ] MMLU     measures=capability  k=1  (3 known failure mode(s))
```

The artifact on disk is `evals/registry.yaml`, a validated datasheet, plus `evals/datasheet.py`, the validator that enforces it. To feel why this is the right discipline, delete the `construct:` line from the GSM8K entry and rerun: the script now exits non-zero with a Pydantic error naming the missing field. That failure is the whole point. The registry makes "no number without a datasheet" a property the code checks, not a virtue I have to remember. Every real eval I build in the next chapters gets an entry here first, and by chapter 5 the registry is the index of my instruments and the first place a committee member should look to understand what my numbers actually claim.

```admonish substack-seed
"Your eval is a thermometer, not a scoreboard." Most people read model benchmarks the way they read a sports table: sort descending, top wins. But a benchmark score is a reading off a measurement instrument, and like any instrument it has a construct (what it's actually about), a precision (how much noise sits on the reading), and failure modes (the ways it lies). I'd write the piece around one uncomfortable exercise: take any benchmark number you've quoted and try to fill in its datasheet, the construct it estimates, the population, the metric, and its known distortions. Most numbers can't survive it. The kicker is the capability-versus-propensity split: "the model gets 71% on math" is three different claims depending on whether that's a single try or the best of sixty-four, and the whole story of training reasoning models is dragging capability down into propensity, which you can't even see if you only logged one number.
```
