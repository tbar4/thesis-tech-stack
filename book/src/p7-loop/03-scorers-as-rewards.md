# Scorers as rewards

Here is the hinge the whole book turns on. In Part III I built scorers as measurement instruments: a scorer takes a model's completion and returns a number that says how good it was, and I was careful to make the good ones programmatic, reproducible, and cheap, precisely because I knew this chapter was coming. In Part V I derived GRPO, whose one external input is a reward: a number that says how good a completion was. Those are the same number. A verifier that can *measure* whether a reasoning trace reached the right answer can *reward* a policy for reaching it, and the only difference is which loop the number feeds. This chapter makes that identity real by taking the actual scorers from the thesis suite and wiring them into `GRPOTrainer` as reward functions. It is the load-bearing chapter of Part VII, because if this join is sloppy, every training result downstream is measuring something other than what I think, and if it is clean, the serve-evaluate-score-train-re-evaluate loop is finally a single closed circuit rather than four separate tools.

## Theory

### A scorer and a reward are the same function wearing two hats

Be precise about the claim, because "evals as rewards" is a slogan and I want the mechanism. A Part III scorer, in the Inspect sense, is a function from (a completed sample, its target) to a `Score`: a value, usually in $[0, 1]$ or a `CORRECT`/`INCORRECT` label, plus metadata explaining the judgment and, critically, the log entry that makes the judgment auditable and re-scorable later. A GRPO reward function, in the TRL sense, is a function from (prompts, a batch of completions, and any passthrough columns) to a list of floats, one per completion. Strip away the framework packaging and both are the same core operation: parse a final answer out of a generated string, decide whether it matches the gold answer, return a number. The verifier at the center, the part that actually knows what "correct" means for this task, does not care whether its output is going to be averaged into an accuracy statistic or centered into a group-relative advantage.

That is why the discipline I imposed in Part III pays off exactly now. Because the thesis-suite scorers were written as pure, deterministic, side-effect-free verifiers (regex extraction, symbolic canonicalization, exact comparison) rather than as model-graded judges baked into an eval harness, I can lift the verifier out and call it in a training loop without dragging the whole harness along. The re-scorable log format from chapter 3.4 was not a nicety; it was the design decision that guaranteed the scorer's core was a standalone function I could reuse here. The alternative, a scorer entangled with Inspect's solver and dataset machinery, would force me to either run a full eval per training step (impossibly slow) or reimplement the verifier (two verifiers that drift apart, which is worse than none).

```admonish thesis-thread title="The SDA task, from instrument to reward"
The recurring worked example of this book is the Space Domain Awareness (SDA) task: a family of verifiable multi-step reasoning problems where the model must show its work and then commit to a final answer in a canonical form, and a verifier decides correctness by *canonicalizing and comparing*, not by string-matching. Concretely, an SDA item gives a problem, expects the model to reason inside `<think>...</think>` and then emit its answer inside `<answer>...</answer>`, and the verifier extracts the answer, normalizes it symbolically (so that `1/2`, `0.5`, and `\frac{1}{2}` all compare equal), and checks it against the gold answer. In Part III chapter 3.4 I wrote exactly this verifier and used it to *measure* a model: run the SDA suite, score each sample, report accuracy with a bootstrap interval from the 3.7 `evalstats` module.

This chapter advances the SDA thread by one decisive step: the same verifier now returns a *reward*. Nothing about the verifier changes. What changes is that its output, instead of being tallied into an accuracy number I read, is centered within a group of completions to form the advantage that GRPO optimizes. The model that was being *measured* on the SDA suite is now being *trained* on it, by the identical function. The pre-training SDA accuracy I recorded in Part III becomes the baseline that chapter 7.6 tests the trained model against, and because it is literally the same scorer on both ends, the delta is honest: I am not measuring with one instrument and training against another. This is the join the entire thesis is built to demonstrate, and it lives in this one substitution.
```

### Sparse correctness, dense format, and why you need both

A pure correctness reward (1 if the final answer is right, 0 otherwise) is the honest signal, and on a hard reasoning task it is also a nearly silent one. Early in training the model gets almost everything wrong, so the reward is 0 for nearly every completion in a group, the group mean is near 0, and the group-relative advantage is near 0 for everyone. There is nothing to learn from a group where every member scored identically, because GRPO's signal is the *spread* within the group. This is the sparse-reward problem from Part V made concrete: correctness alone gives the policy no gradient until it stumbles into a correct answer by chance often enough for the group to differentiate.

The standard fix is to add a **format reward**: a small, dense signal that pays the model for structural compliance it can achieve long before it can reason correctly. Did it produce exactly one `<think>` block and one `<answer>` block? Is the answer parseable as the expected type? These are things a model learns in a handful of steps, and rewarding them does two useful things. First, it gives the group something to differentiate on early, so the advantage is non-zero and learning starts. Second, and less obviously, it makes the *correctness* reward computable at all: a verifier can only check an answer it can extract, so teaching the model to emit a well-formed `<answer>` tag is a precondition for the correctness signal ever firing. Format reward is scaffolding, and like scaffolding it is meant to become irrelevant once the building stands.

The danger, which is the entire subject of the next chapter, is that format reward is *gameable* in a way correctness is not. A model can max out a format reward without ever reasoning, and if the format reward is large relative to correctness, the policy will happily learn to produce beautifully-tagged nonsense. So the two rewards are not peers. Correctness is the objective; format is a small dense nudge toward being able to receive the objective. The scaling has to reflect that hierarchy, which is the next thing to get right.

### Shaping and scaling: what the numbers should be

Reward shaping is choosing the *magnitudes* so the policy optimizes what I mean. TRL sums the reward functions, so if I return correctness in $\{0, 1\}$ and format in $\{0, 0.2\}$, a completion's total reward is in $\{0, 0.2, 1.0, 1.2\}$, and the ordering encodes my intent: a correct, well-formatted answer (1.2) beats a correct but malformed one (1.0) beats a wrong but well-formatted one (0.2) beats wrong-and-malformed (0.0). That ordering is the thing to design deliberately, and the ratio matters more than the absolute values.

```admonish derivation title="Why GRPO cares about reward ratios, not reward scale"
GRPO normalizes rewards within each group before using them. With the standard advantage

$$
A_i = \frac{r_i - \bar r}{\operatorname{std}(r) + \epsilon_{\text{std}}}, \qquad \bar r = \frac{1}{G}\sum_{j} r_j,
$$

multiplying every reward by a constant $c$ multiplies both numerator and denominator by $c$ and leaves $A_i$ unchanged. So the *overall* scale of your reward is invisible to vanilla GRPO; doubling all rewards does nothing. What is **not** invisible is the ratio between reward *components*, because that changes the relative spacing of completions within a group. If correctness contributes $\{0, 1\}$ and format contributes $\{0, w\}$, then two completions that differ only in format are separated by $w$ while two that differ in correctness are separated by $1$, and the advantage will pull the policy toward whichever separation is larger where the group happens to vary. Set $w$ too high (say 0.5) and on groups where everyone is wrong but some are well-formatted, the policy learns formatting as if it were the goal. Set $w$ small (say 0.1 to 0.2) and format is a gentle early nudge that correctness overwhelms as soon as the group starts getting answers right. The design rule falls out of the algebra: pick component ratios, not absolute values, and keep the shaping component well below the objective component. Note the `dr_grpo` loss type drops the std-normalization denominator, which changes this analysis slightly (scale is no longer fully invisible), another reason to name the loss type explicitly.
```

Beyond the format-versus-correctness split, shaping has a few more moves worth knowing. **Graded correctness** replaces the binary with partial credit where the task allows it (a proof that gets three of four steps, a numeric answer within tolerance), which densifies the objective itself rather than adding a proxy. **Bounded penalties** (a small negative reward for, say, exceeding a length cap or emitting no answer tag at all) discourage specific pathologies, but every penalty is a new surface to game, so I add them reluctantly and watch them. And **normalization of the verifier's own output** matters: if a graded verifier returns scores on $[0, 100]$ and the format reward is $0.2$, the format term is numerically irrelevant, so I keep every component on a comparable scale before I reason about their ratios. The governing principle is that the reward is a specification of desired behavior written in the language of numbers, and every term I add is a clause in that specification that the policy will interpret adversarially.

### Asynchronous scoring: the verifier is on the critical path

The last theory piece is a performance fact with a correctness consequence. A GRPO step generates $G$ completions per prompt and must score every one before it can compute advantages, so the verifier is called $G \times (\text{batch})$ times per step, synchronously blocking the optimizer. For a cheap verifier (a regex and an integer compare) this is free. But the interesting thesis-suite verifiers are not always cheap: symbolic canonicalization with a computer-algebra system, or a code-execution verifier that runs the model's output in a sandbox, can take tens or hundreds of milliseconds each, and $G \times \text{batch}$ of those in series can rival the generation time. The step is already generation-bound (last chapter); I do not want to make it verifier-bound on top.

The fix is to score the group *concurrently*. Because the verifiers are pure functions of independent completions, scoring them is embarrassingly parallel: a thread pool (for I/O-bound verifiers like a sandbox subprocess or a CAS call that releases the GIL) or an async gather runs all $G \times \text{batch}$ verifications at once and returns when the slowest finishes, turning a sum of latencies into a max. This is safe *only because* the verifiers are side-effect-free and deterministic, which is again the Part III discipline paying a dividend: a scorer that mutated shared state or depended on call order could not be parallelized this way. Two guardrails come with concurrency: a **timeout** per verification, so one pathological completion (an infinite loop in a code-execution task) cannot stall the whole step, with a timeout scored as 0; and, optionally, a **cache** keyed on the completion text, since a group sometimes contains duplicate completions and re-verifying identical strings is waste. The reward module in the lab builds all of this in, because "the verifier is on the critical path" is exactly the kind of thing that is invisible on the toy task and dominant on the real one.

## Tooling

The tool here is not a new library, it is a small module of my own, `rewards.py`, that adapts the thesis-suite verifiers into TRL's reward-function interface. Its job is to be the single, thin, auditable seam between the Part III scoring world and the Part V training world, and everything I learned about the join goes into keeping that seam honest.

The verifier I am adapting is the SDA scorer from chapter 3.4. In Inspect it is wrapped as a `@scorer` returning a `Score`, but its core is a pure function I exposed deliberately: extract the `<answer>` content, canonicalize it, compare to the target, return a float and a reason. The reward module imports *that function*, not the Inspect wrapper, so training and measurement share one verifier with no reimplementation. Around it the module adds the three things GRPO needs that measurement did not: the TRL calling convention (a list-in, list-out batch function reading passthrough columns), the format reward as a separate dense component, and the concurrency-plus-timeout machinery so a slow verifier does not throttle the step.

```admonish under-the-hood title="What TRL does with your returned numbers"
When you pass `reward_funcs=[correctness, format]`, TRL calls each on the full batch of completions, gets a list of floats from each, and *sums them elementwise* into one reward per completion. It then groups those rewards by prompt (groups of `num_generations`), computes the group mean and, for `grpo`, the group std, and forms $A_i$ per the derivation above. Two consequences for how you write rewards. First, since TRL sums the functions, keeping correctness and format as *separate* functions (rather than one function returning their sum) is purely for logging: TRL logs each reward function's mean separately, so splitting them lets you watch format and correctness diverge in MLflow, which is exactly the signal the next chapter needs to catch hacking. Second, TRL passes every non-`prompt`/`completion` dataset column through as a keyword argument, which is how the gold `target` reaches your verifier; name your dataset column and your function parameter identically or the passthrough silently misses.
```

```admonish gotcha title="Chat-formatted completions and the empty-answer case"
Two parsing traps sit right at the seam. First, when the dataset uses conversational prompts, TRL hands each completion as a list of `{role, content}` messages, not a bare string, so the verifier must pull `completion[-1]["content"]` before parsing; a verifier written for plain strings will try to regex a list and score everything 0, which looks exactly like "the model learned nothing". Second, decide explicitly what an *unparseable* completion scores. A completion with no `<answer>` tag cannot be verified for correctness, and scoring it 0 (same as a wrong answer) is usually right, but it means the correctness and format signals are correlated early (no tag implies both wrong and unformatted), which you want to see reflected in the logged curves. Make the empty-answer policy a named decision in the code, not an accident of how the regex fails.
```

## Lab: train against the thesis suite's scorers

The artifact of this chapter is the reward module itself, `rewards.py`, because it is the reusable seam every subsequent training run in the book imports. The lab wires it into a GRPO run on a slice of the SDA suite and confirms, in MLflow, that the correctness and format rewards move the way the theory says they should. This is the first *real* training run of the thesis: same model and budget as the last chapter, but now the reward is the genuine SDA verifier rather than a toy adder.

Set up the project, reusing the Unsloth-pinned stack and adding the thesis suite and a symbolic library for canonicalization.

```bash title="shell"
uv init labs/grpo-sda
cd labs/grpo-sda
uv add "unsloth" "unsloth_zoo" "vllm" "trl" "peft" "bitsandbytes" mlflow
uv add sympy                     # symbolic canonicalization in the verifier
uv add "thesis-suite @ ../../thesis-suite"   # the Part III scorers, as a local pkg
uv lock
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

The reward module. It imports the SDA verifier from the thesis suite (the *same* function chapter 3.4 measures with), wraps it for TRL, adds a dense format reward, and runs the verifiers concurrently with a per-item timeout.

```python title="labs/grpo-sda/rewards.py"
"""Thesis-suite SDA scorers, adapted into GRPO reward functions.

This is the single seam between Part III measurement and Part V training.
Correctness reuses the EXACT verifier chapter 3.4 measures with; format is a
small dense shaping term; both run concurrently with a timeout so a slow
verifier does not throttle the step. Kept deliberately thin and auditable.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
from functools import lru_cache

# The pure verifier from Part III chapter 3.4. Same function used for
# measurement; imported here, not reimplemented, so the two never drift.
from thesis_suite import verify_sda_answer  # (response, target) -> bool

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# --- shaping constants: ratios matter, not absolute scale (see derivation) ---
CORRECT_W = 1.0     # the objective
FORMAT_W = 0.2      # dense scaffold, kept well below the objective
NO_ANSWER = 0.0     # explicit policy: unparseable completion scores 0 correctness
VERIFY_TIMEOUT_S = 2.0


def _text(completion) -> str:
    """TRL hands conversational completions as message lists; pull the content."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


@lru_cache(maxsize=8192)
def _cached_verify(completion_text: str, target: str) -> bool:
    """Cache identical (completion, target) pairs; verifiers are pure."""
    return bool(verify_sda_answer(completion_text, target))


def _verify_one(completion_text: str, target: str) -> float:
    m = ANSWER_RE.search(completion_text)
    if not m:
        return NO_ANSWER                      # cannot verify what has no answer
    try:
        return CORRECT_W if _cached_verify(completion_text, str(target)) else 0.0
    except Exception:
        return 0.0                            # a verifier that raises is a miss


def correctness_reward(prompts, completions, target, **kwargs):
    """Concurrent, timed correctness scoring via the shared SDA verifier."""
    texts = [_text(c) for c in completions]
    out = [0.0] * len(texts)
    with ThreadPoolExecutor(max_workers=min(16, len(texts) or 1)) as pool:
        futures = {pool.submit(_verify_one, t, tgt): i
                   for i, (t, tgt) in enumerate(zip(texts, target))}
        for fut, i in futures.items():
            try:
                out[i] = fut.result(timeout=VERIFY_TIMEOUT_S)
            except FTimeout:
                out[i] = 0.0                  # pathological completion -> miss
    return out


def format_reward(prompts, completions, **kwargs):
    """Dense scaffold: pay for exactly one well-formed think+answer structure."""
    out = []
    for c in completions:
        t = _text(c)
        ok = (len(THINK_RE.findall(t)) == 1
              and len(ANSWER_RE.findall(t)) == 1)
        out.append(FORMAT_W if ok else 0.0)
    return out


REWARD_FUNCS = [correctness_reward, format_reward]
```

A quick standalone check of the rewards before spending GPU time, which is the habit the earlier gotcha argued for.

```python title="labs/grpo-sda/check_rewards.py"
"""Validate reward functions on hand-written strings before launching a run."""
from rewards import correctness_reward, format_reward

good = "<think>2 and 2</think><answer>4</answer>"
bad_fmt = "the answer is 4"
wrong = "<think>...</think><answer>5</answer>"
comps = [good, bad_fmt, wrong]
tgts = ["4", "4", "4"]

print("correctness:", correctness_reward(None, comps, tgts))  # [1.0, 0.0, 0.0]
print("format    :", format_reward(None, comps))             # [0.2, 0.0, 0.2]
```

```bash title="shell"
uv run python check_rewards.py
```

Now the training script. It is the last chapter's script with the reward source swapped for the module above and the dataset drawn from the SDA suite.

```python title="labs/grpo-sda/train.py"
"""GRPO on the SDA suite with the real thesis-suite verifier as reward."""
from unsloth import FastLanguageModel  # noqa: E402

import hashlib
from pathlib import Path

import mlflow
import torch
from trl import GRPOConfig, GRPOTrainer

from rewards import REWARD_FUNCS
from thesis_suite.data import load_sda_split   # Part III frozen v1.0 suite

MODEL = "unsloth/Qwen3-4B"
MAX_SEQ = 2048
LORA_R = 16
GROUP = 8


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True,
        fast_inference=True, max_lora_rank=LORA_R, gpu_memory_utilization=0.16,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_R, use_gradient_checkpointing="unsloth",
    )

    # The TRAIN split only. The frozen v1.0 test split is never trained on;
    # it is what chapter 7.6 measures the delta against, with the same scorer.
    train_ds = load_sda_split("train")

    cfg = GRPOConfig(
        output_dir="outputs", num_generations=GROUP,
        per_device_train_batch_size=GROUP, gradient_accumulation_steps=2,
        max_prompt_length=512, max_completion_length=1024,
        learning_rate=5e-6, beta=0.02, epsilon=0.2, num_iterations=1,
        loss_type="dr_grpo", temperature=1.0, top_p=1.0,
        max_steps=500, logging_steps=1, save_steps=100,
        report_to="mlflow", run_name="grpo-sda-r16", seed=0,
    )

    torch.cuda.reset_peak_memory_stats()
    mlflow.set_experiment("p7-grpo")
    with mlflow.start_run(run_name="grpo-sda-r16"):
        mlflow.log_params({
            "model": MODEL, "lora_r": LORA_R, "group": GROUP,
            "beta": cfg.beta, "loss_type": cfg.loss_type,
            "correct_w": 1.0, "format_w": 0.2,
            "uv_lock": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()[:12],
        })
        trainer = GRPOTrainer(
            model=model, processing_class=tokenizer,
            reward_funcs=REWARD_FUNCS, args=cfg, train_dataset=train_ds,
        )
        trainer.train()
        mlflow.log_metric("vram_peak_gib", torch.cuda.max_memory_allocated() / 2**30)
        model.save_lora("outputs/sda-adapter")
        mlflow.log_artifacts("outputs/sda-adapter", artifact_path="lora")
        print("adapter saved: outputs/sda-adapter")


if __name__ == "__main__":
    main()
```

```bash title="shell"
uv run python train.py
```

```admonish gotcha title="Train split only, and log the two rewards separately"
The dataset here is the SDA *train* split. The frozen v1.0 *test* split from chapter 3.9 is sacred: it never appears in training, because chapter 7.6 uses it to measure the pre-versus-post delta with the identical verifier, and training on it would contaminate exactly the number the thesis rests on. Second, keep `correctness_reward` and `format_reward` as separate entries in `reward_funcs` even though TRL sums them, so MLflow logs `rewards/correctness_reward/mean` and `rewards/format_reward/mean` as distinct curves. Watching those two diverge is the primary diagnostic for the whole next chapter; collapse them into one function and you go blind to hacking.
```

### What you should see

The artifact is `rewards.py`, the reusable seam, plus `outputs/sda-adapter/`, the LoRA adapter trained against the real SDA verifier, and an MLflow run under `p7-grpo`. In the MLflow UI the two reward curves tell the story the theory predicted. `rewards/format_reward/mean` should rise fast and saturate near its 0.2 ceiling within a few dozen steps, because structural compliance is easy and the dense signal drives it quickly. `rewards/correctness_reward/mean` should start near zero (early groups are almost all wrong, so the signal is sparse) and then, once format compliance makes answers reliably extractable, begin a slower climb as the policy actually gets better at the SDA reasoning. The *shape* to look for is format leading and correctness following, which is format-as-scaffolding working as designed. If correctness stays flat while format saturates, either the task is too hard for this model at this budget (the group never differentiates on correctness) or, more likely on a first run, the verifier is silently rejecting well-reasoned answers because of a canonicalization mismatch, which you debug by dumping a few completions the verifier scored 0 and checking them by hand against `verify_sda_answer`. The `vram_peak_gib` should again land near the last chapter's budget, a little higher for the longer `max_completion_length=1024`. Record the peak, the final reward means, and the step time with date and driver (measured on the baseline machine, record value, date, driver). The deep thing this run establishes is not the reward number, it is that the number came from the same function that will measure the result in chapter 7.6, so the loop is now genuinely closed: one verifier, two hats, no drift.

```admonish read-along
Read this against [RLHF] Lambert on reward design and against the Part III scorer chapter (3.4) that this one consumes. Lambert's discussion of why verifiable rewards are more trustworthy than learned reward models is the theoretical backing for the whole "scorer as reward" move; chapter 3.4 is where the specific SDA verifier this lab imports was built and validated as a measurement instrument. The pairing is the point: read the two chapters as a single argument that the instrument and the reward should be, and here literally are, the same function.
```

```admonish substack-seed
"Your eval and your reward are the same function. Most people write them twice." The slogan is 'evals as rewards', but the mechanism is mundane and that is what makes it powerful: a verifier that checks whether a reasoning trace reached the right answer returns a number, and that number is equally happy being averaged into an accuracy statistic or centered into a reinforcement-learning advantage. Write the verifier once, as a pure deterministic function, and you can measure a model with it on Monday and train against it on Tuesday with zero reimplementation and, crucially, zero drift between what you optimized and what you report. The post's payoff is the pair of curves from a real run: a dense 'format' reward that saturates in minutes because getting the tags right is easy, and a sparse 'correctness' reward that only starts climbing once the model is formatted enough to be graded at all. That lag between the two is scaffolding doing its job, and it is also, as the next post will show, exactly where reward hacking hides.
```
