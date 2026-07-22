# Inspect II: scorers, verifiers, sandboxes

The scorer is the most consequential object in the whole eval, because it is the function that decides what "correct" means, and in this book it has a second life: the scorer I write to *measure* a reasoning model in Part III is the same function that *rewards* it in Part V. That dual role is why I treat scorers with more care than any other part of the pipeline. A lenient scorer inflates a measurement; a lenient scorer used as a reward teaches the model to exploit the leniency. This chapter builds verifiable scorers (regex extraction, symbolic equivalence, sandboxed unit-test execution), explains why model-graded scoring is a last resort for verifiable domains, and treats the Inspect log as an audit object I can re-score without touching the GPU. The lab produces the exact scorer that reappears in chapter 7.3 as a reward function.

```admonish read-along
**[RLHF] Lambert Part 3** is the load-bearing reference here: it draws the line between verifiable rewards (a checker with no opinion) and learned or model-graded rewards (a checker with opinions, hence exploitable), and this chapter is the eval-side realization of that distinction. **[BRM] Raschka ch. 3** shows why a reasoning model wants a programmatic verifier: the final answer is checkable, so the reward can be cheap and honest. Read the sandbox section against Lambert's discussion of reward hacking, since executing model-written code is where measurement and safety meet.
```

## Theory

### The scorer API and what a `Score` is

An Inspect scorer is an async function that receives the finished `TaskState` and the sample's `Target`, and returns a `Score`. The decorator `@scorer(metrics=[...])` registers it and declares how its values aggregate across the dataset (typically `accuracy()` and `stderr()`, the reading and its noise floor from chapter 1). A `Score` carries four things worth keeping distinct:

- `value`: the grade, either a categorical `CORRECT`/`INCORRECT` (stored as `"C"`/`"I"`) or a number. This is what the metric aggregates.
- `answer`: the extracted answer string, so the log shows *what* was graded, not just the grade.
- `explanation`: why this grade, the audit breadcrumb.
- `metadata`: anything else worth recording (the regex that matched, the sympy normal form, the failing test).

Keeping `answer` and `explanation` populated is not cosmetic. They are what make the log a re-readable audit object rather than a column of ones and zeros, and they are what let a committee member (or future-me) reconstruct why item 47 scored the way it did without rerunning anything.

```admonish under-the-hood title="Extraction and verification are two jobs; keep them separate"
Every generative scorer does two things: pull a candidate answer out of free text (extraction), then decide whether that candidate is correct (verification). Tangling them produces scorers that are hard to debug and easy to fool. I always factor them: an `extract(completion) -> str | None` that owns the regex and normalization, and a `verify(candidate, target) -> bool` that owns the correctness judgment (numeric equality, symbolic equivalence, test execution). The separation pays off three ways. First, extraction failures and verification failures show up as different `explanation` strings, so I can tell "the model gave no parseable answer" from "the model gave a wrong answer", which are different construct signals. Second, I can unit-test `verify` in isolation against known pairs. Third, when this becomes a reward in chapter 7, the same factoring lets me shape partial credit (extraction succeeded but wrong answer) differently from total failure (no answer at all), which matters for reward density.
```

### Programmatic versus model-graded, and why verifiable wins for reasoning

There are two families of scorer. **Programmatic** scorers compute correctness with code: string match, numeric equality, symbolic equivalence, running unit tests. They are deterministic, free, fast, and have no opinion, which means they cannot be sweet-talked, cannot be sycophantic, and reproduce exactly. Their limit is that they only work when correctness is checkable in code. **Model-graded** scorers ask another model to judge, via a rubric (chapter 6). They handle open-ended constructs that no regex can, at the cost of introducing a second model's noise, cost, latency, and biases into the measurement, every construct-validity threat from chapter 1 relocated into the grader.

For the reasoning tasks the thesis lives on, the final answer is verifiable, so I use programmatic scorers and reserve model grading for genuinely open-ended work. This is not an aesthetic preference. A verifiable scorer is a *proper* signal in the sense of chapter 2: it is minimized (as an error) only by actually producing the correct answer, so optimizing against it optimizes the construct. A model-graded scorer is a *learned* signal, and optimizing hard against a learned signal finds its blind spots (reward hacking). The single most important design rule of this whole book is: **use the most verifiable scorer the construct allows**, because that scorer is the one I can safely turn into a reward.

### Verifiable scorer types

Three programmatic verifiers cover most of the reasoning surface.

1. **Regex / normalized match.** Extract with a pattern (`ANSWER:\s*(-?\d[\d,]*\.?\d*)`), normalize (strip commas, trailing zeros), compare. Cheap and right for numeric final answers. The risk is brittleness: too strict and you score correct-but-differently-formatted answers as wrong (construct-irrelevant variance), too loose and you score wrong answers as right. The regex is a policy that belongs in the datasheet.
2. **Symbolic equivalence.** For math where `1/2`, `0.5`, and `\frac{1}{2}` are the same answer, string match fails but a computer-algebra check succeeds. Parse both sides and test whether their difference simplifies to zero. This removes format sensitivity that regex cannot, at the cost of a parser that can itself misfire on adversarial strings.
3. **Unit-test execution.** For code, correctness *is* "the tests pass". Extract the code, run it against a test suite, score by pass/fail. This is the most honest reasoning verifier there is (the compiler and the tests have no opinion) and the most dangerous to run, because it executes model-written code. That danger is what sandboxes exist for.

### Sandboxes: executing untrusted output safely

The moment a scorer runs model-generated code, or a solver lets the model call tools that touch a shell, filesystem, or network, I am executing untrusted output on my machine. Inspect's answer is the **sandbox**: an isolated execution environment (Docker by default) that the solver and scorer reach through a `sandbox()` handle, with `sandbox().exec(...)` running commands inside the container rather than on the host. The container is disposable, resource-limited, and network-restricted, so a model that writes `rm -rf /` or tries to exfiltrate data hits container walls, not mine. For the thesis this matters twice: agentic evals (chapter 1's fourth family) need tools, and unit-test scorers need to run code, and both must be sandboxed. On the 16GB baseline machine the sandbox runs on CPU alongside the GPU-resident model server, so it costs container overhead, not VRAM.

```admonish gotcha title="An un-sandboxed code scorer is a remote code execution primitive you built on purpose"
It is tempting, for a quick local run, to score code by `exec`-ing it in the same Python process. Do not, even once, even on "trusted" models. The model's output is untrusted by definition (that is what you are evaluating), and a reasoning model exploring solution space will eventually emit something that deletes a file, opens a socket, or hangs. Worse, when this scorer becomes a reward and the model is *optimized* against it, reward hacking actively searches for scorer exploits, and an un-sandboxed executor turns a reward hack into host compromise. The sandbox is not optional hardening; it is the boundary that lets code execution be a scorer at all. Budget the Docker setup cost once and never run model code outside it.
```

### The log as an audit object, and re-scoring

Because chapter 3's eval log stores every sample's full transcript and model output, scoring is *separable in time* from generation. Inspect exposes this directly: `inspect score` takes an existing `.eval` log and a scorer and produces new scores from the saved transcripts, no model calls. This is the property that makes the log an audit object rather than a receipt. I can run a model once (the expensive part, the GPU part), then score it many ways: fix a too-strict regex, add symbolic equivalence, try a stricter rubric, all against the identical transcripts, for free. It is also how I keep measurements honest across a project: if I improve my verifier in month three, I re-score month-one logs with it and the comparison stays apples-to-apples. And it is the mechanism by which a scorer graduates into a reward. A reward function is exactly "a scorer applied to a rollout", and Inspect's re-scoring is the small-scale rehearsal of the reward loop I build in Part V.

## Tooling

The scorer decorator, the categorical grades, and the aggregation metrics come from `inspect_ai.scorer`; the sandbox handle from `inspect_ai.util`. A verifiable scorer skeleton:

```python
from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target, CORRECT, INCORRECT
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy(), stderr()])
def my_verifier():
    async def score(state: TaskState, target: Target) -> Score:
        answer = extract(state.output.completion)          # extraction
        if answer is None:
            return Score(value=INCORRECT, answer=None, explanation="no parseable answer")
        ok = verify(answer, target.text)                   # verification
        return Score(value=CORRECT if ok else INCORRECT, answer=answer,
                     explanation="symbolic-equal" if ok else "mismatch")
    return score
```

For sandboxed execution the task declares a sandbox (`Task(..., sandbox="docker")`) and the scorer runs code through the handle:

```python
from inspect_ai.util import sandbox
result = await sandbox().exec(cmd=["python", "-c", code], timeout=10)
passed = result.success and result.returncode == 0
```

The `uv` project from chapter 3 already has `inspect-ai`; add `sympy` for symbolic checks. Everything else is the same endpoint and the same served model.

```bash
uv add sympy
```

## Lab

The artifact is `evals/verify.py`, a **verifiable math scorer** factored into extraction and verification, wired into an Inspect task, and re-used against chapter 3's saved logs to prove re-scorability. The verification core is written so that the very same function is importable in chapter 7.3 as a reward. I build the numeric/symbolic verifier fully here and include a sandboxed unit-test scorer as the code-domain counterpart.

```python title="evals/verify.py"
"""Verifiable scorers for reasoning tasks.

The verification core `is_correct(answer, target)` is deliberately a pure
function of (candidate, reference). Chapter 7.3 imports it unchanged as the
verifiable reward: reward = 1.0 if is_correct(model_answer, gold) else 0.0.

Extraction and verification are kept separate (see chapter theory):
  extract_answer(text) -> str | None      # regex + normalization
  is_correct(cand, ref) -> bool           # numeric then symbolic equivalence
"""
from __future__ import annotations

import re

from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(text: str) -> str | None:
    """Pull the last 'ANSWER: ...' payload from a completion, or None."""
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def _normalize(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace("\\", "")
    s = re.sub(r"\s+", "", s)
    return s


def is_correct(candidate: str, reference: str) -> bool:
    """True if candidate equals reference numerically or symbolically.

    Pure, deterministic, no model calls. This is the reward core of ch. 7.3.
    """
    if candidate is None:
        return False
    c, r = _normalize(candidate), _normalize(reference)
    if c == r:
        return True
    # numeric equality with tolerance
    try:
        if abs(float(c) - float(r)) < 1e-6:
            return True
    except ValueError:
        pass
    # symbolic equivalence: does (candidate - reference) simplify to 0?
    try:
        if simplify(parse_expr(c) - parse_expr(r)) == 0:
            return True
    except Exception:
        pass
    return False


@scorer(metrics=[accuracy(), stderr()])
def math_verifier():
    """Programmatic, verifiable scorer. No opinion, reproducible, free."""
    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_answer(state.output.completion)
        if answer is None:
            return Score(value=INCORRECT, answer=None,
                         explanation="extraction failed: no 'ANSWER:' line")
        ok = is_correct(answer, target.text)
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=answer,
            explanation=("verified equal" if ok
                         else f"mismatch: got {answer!r}, want {target.text!r}"),
            metadata={"extracted": answer},
        )
    return score
```

The code-domain counterpart runs model-written code against hidden tests inside a Docker sandbox. It is the same extract-then-verify shape, with verification delegated to the compiler and the tests.

```python title="evals/verify_code.py"
"""Sandboxed unit-test scorer: correctness IS 'the tests pass'.

Requires a Docker sandbox on the Task (sandbox="docker"). The model's code is
NEVER run on the host, only inside the disposable container.
"""
from __future__ import annotations

import re

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    blocks = _CODE_RE.findall(text or "")
    return blocks[-1].strip() if blocks else None


@scorer(metrics=[accuracy(), stderr()])
def unit_test_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        code = extract_code(state.output.completion)
        if code is None:
            return Score(value=INCORRECT, answer=None, explanation="no code block")
        # target.text carries the test harness; concatenate and run in sandbox.
        program = code + "\n\n" + target.text
        try:
            result = await sandbox().exec(cmd=["python", "-c", program], timeout=15)
        except TimeoutError:
            return Score(value=INCORRECT, answer=code, explanation="timeout (possible hang)")
        passed = result.success and result.returncode == 0
        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=code,
            explanation="tests passed" if passed else f"tests failed:\n{result.stderr[:500]}",
        )
    return score
```

Wire the numeric verifier into a task and, critically, re-score chapter 3's existing logs with it to prove the log is an audit object.

```python title="evals/math_task.py"
"""GSM8K-style task scored by the verifiable math_verifier (not match())."""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import chain, generate, prompt_template, system_message

from evals.verify import math_verifier

COT_SYSTEM = (
    "Reason step by step. End with exactly 'ANSWER: <value>' on the final line."
)


def record_to_sample(record: dict) -> Sample:
    return Sample(input=record["question"], target=record["answer"])


@task
def gsm8k_verified() -> Task:
    return Task(
        dataset=json_dataset("evals/data/gsm8k_mini.jsonl", sample_fields=record_to_sample),
        solver=chain(system_message(COT_SYSTEM), prompt_template("{prompt}"), generate()),
        scorer=math_verifier(),
    )
```

Run the verified task, then demonstrate re-scoring: take a log produced in chapter 3 with the crude `match()` scorer and re-score it with `math_verifier` from the saved transcripts, no GPU.

```bash
# 1) fresh run with the verifiable scorer
uv run inspect eval evals/math_task.py@gsm8k_verified --model openai/qwen-8b --limit 5

# 2) re-score an EXISTING chapter-3 log with the new scorer, no model calls
uv run inspect score logs/<chapter3-run>.eval --scorer evals/verify.py@math_verifier

# 3) sandboxed code scorer: wire unit_test_scorer() into a Task with
#    sandbox="docker" (same shape as math_task.py) and run it. Needs Docker.
#    uv run inspect eval evals/code_task.py@code_verified --model openai/qwen-8b --limit 5
```

Prove the reward core is a pure function, independent of Inspect, since that is the property chapter 7.3 relies on:

```bash
uv run python -c "from evals.verify import is_correct; \
print(is_correct('0.5', '1/2'), is_correct('16', '16.0'), is_correct('15', '16'))"
```

### What you should see

The verified run prints `accuracy` and `stderr` as before, but now every graded item has a real `explanation` in the log: "verified equal", or "mismatch: got '18', want '16'", or "extraction failed: no 'ANSWER:' line". Those three explanations distinguish right, wrong, and unparseable, which the chapter-3 `match()` scorer could not, so the same transcripts now yield a more informative measurement. The re-score step (step 2) is the demonstration that matters: it completes in seconds with zero model calls and produces a *new* score column over the *old* transcripts, and if the crude `match()` and the `math_verifier` disagree on any item, that disagreement is the format-sensitivity bug from chapter 3 caught and fixed without spending a single GPU-second. The pure-function check prints `True True False`, confirming `is_correct` handles symbolic (`0.5` = `1/2`) and numeric (`16` = `16.0`) equivalence and correctly rejects `15` vs `16`. On the baseline machine, record the verified accuracy and note how many items changed grade between `match()` and `math_verifier` on the re-score, with date and driver.

The artifact is `evals/verify.py`, and its `is_correct` function is the load-bearing object of the whole book. In chapter 7.3 I import it unchanged and write `reward = 1.0 if is_correct(model_answer, gold) else 0.0`, and the reasoning model is trained to satisfy the exact verifier I validated here as a measurement. That continuity, one function serving first as instrument and then as reward, is why I spent this chapter insisting the scorer be verifiable, factored, sandboxed where it executes anything, and logged as an audit object. A reward is only as trustworthy as the scorer it was born from, and this is where I earn that trust.

```admonish substack-seed
"The scorer you measure with is the reward you'll train against, so write it like money's on it." There's a quiet handoff at the heart of training reasoning models: the function that grades your eval becomes the function that pays your model. That reframes everything about how you write a scorer. A lenient grader is a measurement error you can live with, but a lenient reward is an exploit the model will find and hammer, because optimization is an adversary that searches for your scorer's blind spots. The post makes the case for verifiable scorers, ones with no opinion (a symbolic-math check, a unit-test run) over model-graded ones, and for two disciplines that sound fussy until the handoff makes them essential: separate extraction from verification so you can tell "no answer" from "wrong answer", and never, ever run model-written code outside a sandbox, because a reward hack against an un-sandboxed code scorer isn't a bad number, it's your machine. The closing image is one pure function that lives twice, first as the thing that measures the model, then as the thing that teaches it.
```
