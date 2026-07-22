# Inspect I: tasks, datasets, solvers

Inspect (`inspect_ai`) is the framework I build custom instruments in, and its design is a near-literal encoding of chapter 1's ontology: an eval is a `Task`, a `Task` binds a `Dataset` (the items), a `Solver` (how the model is prompted and, later, how it acts), and a `Scorer` (how a finished sample becomes a number). This chapter covers the first three; scorers get chapter 4, because they are where measurement turns into reward and they deserve their own treatment. By the end I have a first custom task running end-to-end against the vLLM endpoint from Part II, producing a structured log on disk that I can replay and re-score.

```admonish read-along
**[BRM] Raschka ch. 3** builds an eval loop by hand to make the moving parts visible; Inspect is the same loop, factored into reusable abstractions so I stop rewriting the plumbing. Read Raschka's from-scratch loop as the "why", then this chapter as the "how, without the boilerplate". **[RLHF] Part 3** on evaluation infrastructure motivates why an auditable log (not just a final score) is the real deliverable.
```

## Theory

### The four abstractions and the log

Inspect factors an eval into four objects, and the factoring is the point: each is swappable without touching the others, so I can hold the dataset and scorer fixed and vary only the solver to ask "does chain-of-thought help?", or hold the solver fixed and swap the model to ask "does INT4 degrade reasoning?".

- **`Sample`** is one item: an `input` (the prompt, a string or a message list), a `target` (the reference answer), and optional `choices` and `metadata`. A **`Dataset`** is a sequence of samples, loaded from JSON, CSV, a HuggingFace dataset, or built in memory.
- **`Solver`** is a function that transforms a `TaskState` (the running conversation plus scratch state) toward a completed answer. Solvers compose: a chain of `system_message`, `prompt_template`, and `generate` is itself a solver, and a custom multi-turn solver is just a function that calls `generate` more than once.
- **`Scorer`** consumes the finished `TaskState` and the `target` and emits a `Score`. Chapter 4.
- **`Task`** binds a dataset, a solver, and a scorer, plus config (epochs, message limits, the metrics to aggregate).

Running a task produces an **eval log**, a structured record (`.eval` files under `logs/`) containing every sample's full message transcript, the model output, the token usage, the score, and the run's configuration and git state. The log, not the printed accuracy, is the deliverable. It is the audit object from chapter 1 made concrete, and because it stores the full transcripts it can be re-scored later without re-running the model, which is the property that makes Inspect the on-ramp to reward design.

```admonish under-the-hood title="TaskState is the mutable spine every solver reads and writes"
A `Solver` has the signature `async def solve(state: TaskState, generate: Generate) -> TaskState`. The `TaskState` carries `state.messages` (the running chat transcript), `state.user_prompt` (a handle to the first user message, convenient for templating), `state.output` (the latest model completion, populated by `generate`), `state.metadata` (the sample's metadata), and `state.store` (a scratch dict for passing values between solvers in a chain). A solver mutates the state and returns it; the next solver in the chain receives the mutated state. `generate` is an injected callable that runs the model on the current messages and writes the result into `state.output`. This is why multi-turn evals are natural in Inspect: a multi-turn solver just appends messages and calls `generate` again, and each call is one model turn recorded in the log. The whole framework is a disciplined loop over an async state transformer, which is exactly the serve to evaluate loop of this book with the serving half pointed at my local vLLM.
```

### Solvers compose, and composition is where the construct lives

The solver is where I encode *how* I am probing the construct, and small solver choices are large construct choices. A bare `generate` measures the model's zero-shot propensity. Prefixing a `system_message` that says "think step by step" measures propensity under a reasoning prompt, a different construct. Wrapping with `prompt_template` to inject few-shot exemplars measures few-shot capability. A multi-turn solver that lets the model critique and revise its own answer measures self-correction, yet another construct. None of these is "the" measurement; each is a deliberate operationalization, and the solver code *is* the operationalization made executable. This is why I keep solvers explicit and version them: the datasheet from chapter 1 says what construct I claim to measure, and the solver is the proof of what I actually did.

For reasoning models specifically, the solver interacts with the model's own chain-of-thought behavior. A reasoning model already emits a long deliberation before its answer; my solver's job is often just to (a) instruct the output format so the scorer can find the final answer, and (b) optionally cap or shape the reasoning. Over-prompting a reasoning model (stacking my own "think step by step" on top of its native reasoning) can hurt, so the solver is a lever I tune, not a fixed recipe.

### Datasets are more than a list, and every loader choice is a sampling decision

The `Dataset` abstraction looks trivial (a sequence of `Sample`s) but it hides three decisions that quietly change what I measure. First, **where the items come from**: Inspect loads from JSON/JSONL (`json_dataset`), CSV (`csv_dataset`), a HuggingFace hub dataset (`hf_dataset`), or an in-memory list (`MemoryDataset`). For the thesis I keep a local copy of every dataset I evaluate against, because a hub dataset can change under me between runs and silently break reproducibility, which is a comparability failure exactly like chapter 5's "why published numbers differ". A pinned local JSONL is a controlled instrument; a live hub pull is not.

Second, **the record-to-sample mapping**, the `record_to_sample` function, is where I decide what the `input` and `target` actually are, and it is construct-defining. If the raw record has a full chain-of-thought answer and I map `target` to the whole thing, my scorer will grade the reasoning text; if I map `target` to only the final number, my scorer grades the answer. Same dataset, different construct, decided in one function. This is also where I attach `metadata` (difficulty, subject, source split) that I later slice the results by, so a slice like "accuracy on the hardest quartile" is available without re-running.

Third, **how many items and in what order**. Inspect's `--limit` samples a prefix (useful for a fast smoke test, dangerous as a reported number because a prefix is not a random sample), and `--epochs` runs each item multiple times, which is how I get the multiple samples per problem that `pass@k` and majority vote (chapter 2) need. Epochs are the bridge from single-sample propensity to multi-sample capability: `--epochs 8` with a sampling temperature gives me eight rollouts per item, and the scorer aggregates them. The order and seed matter too, because a paired comparison across models (chapter 5) is only valid if both models saw the identical items in the identical order, so I fix the seed and never rely on `--limit` for a number that goes in a table.

## Tooling

### Install and point Inspect at the local endpoint

Inspect talks to any OpenAI-compatible endpoint, which is exactly what vLLM serves, so the model under test is the one I stood up in Part II. Set up the eval project with `uv`.

```bash
uv init thesis-evals
cd thesis-evals
uv add inspect-ai openai
```

Inspect reaches an OpenAI-compatible server through the `openai` provider plus two environment variables pointing at vLLM instead of api.openai.com. Assuming vLLM serves my 8B model under the name `qwen-8b` on port 8000 (from Part II), I put the connection in a `.env` file that Inspect loads automatically.

```bash title=".env"
# Point Inspect's OpenAI provider at the local vLLM endpoint.
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
# vLLM ignores the key but the OpenAI client requires a non-empty value.
```

With that, the model string `openai/qwen-8b` routes to vLLM. Nothing else in Inspect changes, which is the whole appeal of the OpenAI-compatible contract: the eval framework does not know or care that the model is local.

```admonish gotcha title="The model name must match vLLM's served name exactly"
The part after `openai/` is passed verbatim as the `model` field in the OpenAI request, and vLLM rejects a request whose model name does not match what it was launched with (`--served-model-name`). If `inspect eval` returns a 404 or "model not found", check the served name with `curl -s http://localhost:8000/v1/models | python -m json.tool` and make the string after `openai/` match it exactly. This is the single most common first-run failure, and it is a naming mismatch, not an Inspect bug.
```

### The abstractions in code

A minimal task is three objects and a decorator. `@task` marks a function that returns a `Task`; Inspect discovers it by name.

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, system_message, chain
from inspect_ai.scorer import match

@task
def tiny() -> Task:
    return Task(
        dataset=MemoryDataset([
            Sample(input="What is 17 + 26?", target="43"),
        ]),
        solver=chain(system_message("Answer with only the number."), generate()),
        scorer=match(numeric=True),
    )
```

`match(numeric=True)` extracts the last number from the output and compares it to the target, a deliberately crude scorer I use here only so the task runs end-to-end; chapter 4 replaces it with a real verifier. The solver is a chain: set a system message, then generate. That is the whole shape, and everything that follows is elaboration of the dataset and the solver.

## Lab

The artifact is a **task package**: a self-contained GSM8K-style reasoning task with its own dataset, a chain-of-thought solver, a multi-turn self-revision variant, and the eval logs it produces. It is fully local, no network needed, so it runs against vLLM alone and the logs become the reusable audit object I carry into chapter 4.

Lay the package out under the `uv` project from the Tooling section:

```
thesis-evals/
  .env
  evals/
    data/gsm8k_mini.jsonl
    gsm8k_task.py
  logs/            # created by the run
```

First the dataset, a small hand-built JSONL so the lab is self-contained and the answer key is visible. In real use I would load the full GSM8K test split via `hf_dataset`, but a mini set makes the mechanics inspectable and keeps the first run to a few seconds on one GPU.

```json title="evals/data/gsm8k_mini.jsonl"
{"question": "A baker has 3 trays with 12 muffins each. He sells 20 muffins. How many are left?", "answer": "16"}
{"question": "Sara reads 15 pages a day for 6 days, then 25 pages on the 7th day. How many pages total?", "answer": "115"}
{"question": "A tank holds 240 liters. It drains at 8 liters per minute. How many minutes to empty?", "answer": "30"}
{"question": "Tom buys 4 packs of 6 pencils and gives away 9. How many pencils remain?", "answer": "15"}
{"question": "A class of 28 students splits into teams of 4. How many teams?", "answer": "7"}
```

Now the task. It shows the three abstractions doing real work: a `record_to_sample` mapping teaches the `Dataset` layer, a templated chain-of-thought solver teaches prompt templating, and a second `@task` with a custom multi-turn solver teaches multi-turn generation. I keep the crude built-in scorer here on purpose and flag it, so chapter 4 has something concrete to improve.

```python title="evals/gsm8k_task.py"
"""A first custom Inspect task: GSM8K-style reasoning against local vLLM.

Two tasks share one dataset:
  gsm8k_cot        - single-turn chain-of-thought
  gsm8k_revise     - multi-turn: answer, self-critique, then revise

Run (single-turn, 5 items):
  uv run inspect eval evals/gsm8k_task.py@gsm8k_cot \
      --model openai/qwen-8b --limit 5

Scoring here uses the crude built-in match(numeric=True); chapter 4 replaces it
with a verifiable scorer whose logs re-score without re-running the model.
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import match
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    generate,
    prompt_template,
    solver,
    system_message,
)

# --- Dataset -------------------------------------------------------------
# Map one JSONL record into a Sample. This is the Dataset abstraction: the
# eval never sees raw records, only normalized (input, target) pairs.
def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer"],
        metadata={"source": "gsm8k_mini"},
    )


def gsm8k_dataset():
    return json_dataset("evals/data/gsm8k_mini.jsonl", sample_fields=record_to_sample)


# --- Prompt templating ---------------------------------------------------
COT_SYSTEM = (
    "You are a careful math tutor. Reason step by step, then on the final "
    "line write exactly 'ANSWER: <number>' with no units and no extra text."
)

# {prompt} is filled with the sample input. Templating is a solver.
COT_TEMPLATE = "Solve this problem.\n\n{prompt}\n\nShow your work, then the final line."


# --- Single-turn chain-of-thought solver ---------------------------------
@task
def gsm8k_cot() -> Task:
    return Task(
        dataset=gsm8k_dataset(),
        solver=chain(
            system_message(COT_SYSTEM),
            prompt_template(COT_TEMPLATE),
            generate(),
        ),
        scorer=match(numeric=True),  # crude on purpose; ch.4 upgrades this
    )


# --- Multi-turn self-revision solver -------------------------------------
# Demonstrates a multi-turn Solver: generate, inject a critique prompt as a new
# user turn, generate again. Each generate() call is one model turn in the log.
@solver
def self_revise() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)                     # first attempt
        state.messages.append(
            ChatMessageUser(
                content=(
                    "Re-check your arithmetic step by step. If you find an "
                    "error, correct it. End with exactly 'ANSWER: <number>'."
                )
            )
        )
        state = await generate(state)                     # revised attempt
        return state

    return solve


@task
def gsm8k_revise() -> Task:
    return Task(
        dataset=gsm8k_dataset(),
        solver=chain(system_message(COT_SYSTEM), self_revise()),
        scorer=match(numeric=True),
    )
```

Run both variants against the local model. The `@taskname` suffix selects one of the two tasks in the file.

```bash
# single-turn chain-of-thought
uv run inspect eval evals/gsm8k_task.py@gsm8k_cot --model openai/qwen-8b --limit 5

# multi-turn self-revision on the same items
uv run inspect eval evals/gsm8k_task.py@gsm8k_revise --model openai/qwen-8b --limit 5

# open the interactive log viewer
uv run inspect view --log-dir logs
```

### What you should see

Each run prints a live progress table and a final summary line with `accuracy` and `stderr` (the reading and its noise floor together, exactly the chapter-1 discipline the tool enforces for free), plus the token usage and wall-clock time. On the five-item mini set expect the accuracy to be a coarse fraction like 4/5; the number is not the point at this size, the *mechanism* is. Under `logs/` you now have one `.eval` file per run: a structured, replayable record of every sample's full transcript, the model's chain-of-thought, the extracted answer, and the score. Open `inspect view` and you can click into any sample and read exactly what the model saw and produced, including, for `gsm8k_revise`, both the first attempt and the revised second turn as separate messages. On the baseline machine, record the accuracy, wall-clock, and tokens/sec for each variant with the date and driver version, since these are the first real readings off my custom instrument.

The artifact on disk is the task package (`evals/gsm8k_task.py` plus `evals/data/gsm8k_mini.jsonl`) and the `logs/` directory of eval logs. Two things to notice, because they define the next chapter. First, `gsm8k_revise` and `gsm8k_cot` differ only in the solver, one line of substance, yet they measure genuinely different constructs (raw reasoning versus reasoning-with-self-correction), which is the composability claim from the theory section paying off. Second, the scoring is the weak link: `match(numeric=True)` grabs the last number and will misfire whenever the model's format wanders. The log preserves every transcript, so I can fix the scorer and re-score these exact runs without paying for generation again. That re-scorability is what chapter 4 exploits to turn a scorer into a verifiable, auditable reward.

```admonish substack-seed
"Your eval framework should let you change the question without re-running the model." The unglamorous superpower of a well-factored eval harness is that it separates four things that people usually tangle together: the items, how you prompt, how you score, and the run itself. Once they're separate, you can ask genuinely different questions of the same model by changing one of them in isolation, does chain-of-thought help (swap the solver), does INT4 hurt (swap the model), was my grader too lenient (swap the scorer and re-score the saved transcripts, no GPU needed). The post would use one concrete before/after: a five-line single-turn solver and a five-line multi-turn self-revision solver that measure different cognitive abilities of the same model on the same problems, and the fact that the second one costs nothing extra to design because the framework already recorded everything. The deeper point is that the log, not the leaderboard number, is the artifact worth keeping.
```
