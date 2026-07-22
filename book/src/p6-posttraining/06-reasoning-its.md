# Reasoning models and inference-time scaling

Up to here every knob in this part changed the *weights*. This chapter is about the orthogonal axis: with the weights frozen, I can spend more *compute per answer* at inference time and get better accuracy, sometimes a lot better. Chain-of-thought lets the model think in tokens before answering; self-consistency samples many chains and votes; best-of-n samples many answers and lets a verifier pick; self-refinement lets the model critique and revise; budget forcing lets me dial the thinking length up or down directly. These are not tricks, they are a second scaling law, compute-per-answer instead of parameters, and the reason this chapter sits in the post-training part is that it interacts with everything else in a way that will wreck my evaluations if I am not careful. A trained model that "beats" a base model might just be spending more inference compute, and unless I hold the compute budget fixed the comparison is meaningless. So the theory here is the menu of inference-time methods and the *compute-per-answer frontier* they trace, and the lab is the matched-budget scaling curves on the thesis suite that keep every later claim in the book honest.

## Theory

### Chain-of-thought: computation as tokens

A transformer does a fixed amount of computation per token: one forward pass, constant depth. A hard problem may need more sequential computation than one forward pass provides. Chain-of-thought (CoT) is the observation that you can buy more computation by letting the model emit intermediate reasoning tokens before its final answer, because each generated token is another forward pass conditioned on all the previous ones, so a long reasoning trace is a variable-depth computation the model builds for itself. Formally, instead of modeling $p(\text{answer} \mid x)$ directly, CoT models

$$p(a \mid x) = \sum_{z} p(a \mid z, x)\, p(z \mid x), \tag{6.6.1}$$

where $z$ is a reasoning trace (the "thinking"). The model samples a $z$, then answers conditioned on it. The reason this helps is that for many problems $p(a \mid z, x)$ is sharply peaked and easy once the right $z$ is written down (the arithmetic is mechanical once you have set up the steps), even though $p(a \mid x)$ marginalized over all traces is diffuse and hard. Reasoning-trained models (via SFT on long traces, chapter 6.2, and RLVR, Part V) are models whose $p(z \mid x)$ has been shaped to put mass on *useful* traces. CoT is the substrate; the training makes the traces good.

### Self-consistency: sample many traces, vote

Equation (6.6.1) is a sum over traces, and a single sampled $z$ is a one-sample Monte Carlo estimate of it, high variance. Self-consistency reduces that variance the obvious way: sample $k$ independent traces $z_1, \dots, z_k$, read off each one's answer, and take the majority vote (the mode of the answer marginal). This works because wrong traces tend to be wrong in *different* ways (they scatter across many wrong answers) while correct traces converge on the *same* right answer, so the correct answer accumulates votes faster than any single wrong one. It is a verifier-free method: no external checker, just agreement among samples. Its cost is exactly $k$ times a single CoT generation, and its accuracy-versus-$k$ curve is the canonical inference-time scaling curve, rising and then saturating as the vote stabilizes.

### Best-of-n with a verifier

If I *have* a verifier (and for reasoning tasks I do, that is the thesis), I can do better than voting. Best-of-n (also called rejection sampling at inference) samples $n$ answers and keeps the one the verifier accepts, or, with a scoring verifier or reward model, the highest-scored one. The distinction from self-consistency is the selection rule: self-consistency picks the most *popular* answer, best-of-n picks the *verified-correct* (or highest-scored) one, which is strictly more informative when a real checker exists.

```admonish derivation title="Why best-of-n scaling is exponential in the wrong direction, then saturates"
Suppose the model solves a task with per-sample success probability $p$ (it emits a correct, verifier-passing answer with probability $p$ on any single try). With a *perfect* verifier that always recognizes a correct answer, best-of-n succeeds if *any* of the $n$ samples is correct. The samples are independent, so the failure probability is $(1-p)^n$ and

$$\text{acc}_{\text{bo}n} = 1 - (1 - p)^{n}. \tag{6.6.2}$$

This is the pass@$n$ curve, and it explains the whole shape of inference-time scaling. For small $p$, early samples buy a lot: going from $n=1$ to $n=2$ raises accuracy by $p(1-p)$, which is substantial. But the marginal gain of the $n$-th sample is $p(1-p)^{n-1}$, which decays geometrically, so the curve saturates: once you are very likely to have already hit a correct sample, more samples are wasted. The half-way point (accuracy $= 1 - \tfrac{1}{2}$ of the gap) arrives around $n \approx \ln 2 / (-\ln(1-p)) \approx 0.69/p$ for small $p$, which says the compute you need scales like $1/p$: a model that solves the task 10% of the time per sample needs roughly $7$ samples to get to about $50\%$ and tens of samples to approach its ceiling.

Two caveats make this the *ceiling*, not the reality. First, real verifiers are imperfect: a verifier with false-positive rate $q$ (accepts a wrong answer) caps the achievable accuracy below 1, because as $n$ grows you eventually sample a wrong-but-accepted answer. Second, equation (6.6.2) assumes the samples are independent and identically distributed, which temperature-based sampling only approximates. Both push the real curve below equation (6.6.2), but the shape, fast early gains, geometric saturation, is exactly what you see.
```

Equation (6.6.2) is also the bridge back to training. RLVR *raises $p$*, the per-sample success probability, by training the policy to put more mass on correct traces. Inference-time scaling *spends samples to convert a given $p$ into higher accuracy*. They are complementary: a better $p$ from training shifts the entire pass@$n$ curve up and left (you reach any target accuracy with fewer samples), which is the cleanest way to see why a good cold-start and RLVR make inference-time scaling *cheaper*, not redundant.

### Self-refinement and budget forcing

Two more methods round out the menu. **Self-refinement** is sequential rather than parallel: the model generates an answer, then is prompted to critique and revise it, possibly for several rounds. It can help when the model can recognize its own errors better than it can avoid them on the first pass, but it has a well-known failure mode where the model "corrects" a right answer into a wrong one, so it is not a free lunch and needs a verifier or a stopping rule to be safe. **Budget forcing** is the most direct knob of all: it controls the *length* of the reasoning trace at decode time, either forcing the model to keep thinking (suppress the end-of-thinking token and inject "Wait, let me reconsider") to spend more compute, or truncating the trace and forcing an answer to spend less. It turns the implicit compute of equation (6.6.1) into an explicit dial, and it is how recent reasoning models expose a "thinking effort" setting. All of these are points on one frontier.

### Verifier types: outcome, process, and generative

Best-of-n and rejection sampling are only as good as the verifier that selects, so it is worth distinguishing the three kinds I might reach for, because they sit at very different points on the trust-versus-availability curve. An **outcome verifier** checks only the final answer against ground truth (does the boxed number equal the gold number, does the code pass the unit tests); it is the gold standard for reasoning tasks because it has essentially no exploitable slack, but it needs a known answer, which you have at training and eval time but not at deployment. A **process reward model** (PRM) scores the *intermediate steps* of a trace, trained on step-level correctness labels; it can catch a trace that stumbles into the right answer by luck and can guide search step by step, but it is a learned reward and therefore hackable in exactly the sense of chapter 6.4, so its verdicts are softer. A **generative verifier** is an LLM-as-judge (Part III) prompted to check the reasoning; it is the most available (no labels, no training) and the least trustworthy, with all the judge-bias failure modes the causal-inference part warns about. For the thesis I lean on outcome verifiers wherever a ground-truth answer exists, precisely because they are the same clean, low-slack signal that makes RLVR stable, and I treat PRM and generative verifiers as research extensions rather than load-bearing. The best-of-n curve in the lab uses an outcome verifier, which is why it represents a near-ceiling on what selection can buy.

### The compute-per-answer frontier and matched-budget evals

Every method above trades inference compute for accuracy, and the honest way to describe a model is not a single accuracy number but a *curve*: accuracy as a function of compute-per-answer. Compute-per-answer is best measured in generated tokens (or FLOPs, or wall-clock, but tokens are the portable currency), summed over all the samples and refinement rounds a method uses to produce one final answer. Self-consistency at $k=16$ costs about $16\times$ the tokens of a single CoT; best-of-$8$ costs about $8\times$; budget forcing to a 4000-token trace costs whatever that trace costs. Plotting accuracy against this token budget puts every method on the same axis, and the upper envelope of those curves is the model's compute-per-answer frontier.

```admonish gotcha
This is the single most important methodological point in the part, and it is where most public model comparisons quietly cheat. If model A is evaluated with self-consistency at $k=32$ and model B with a single greedy sample, A's higher accuracy tells you nothing about which model is better, because A spent $32\times$ the inference compute. Any claim that "training improved the model" must hold the inference budget fixed: same method, same $k$, same token budget, before and after. A trained model that only wins at a *larger* budget has not necessarily improved, it may just have been handed more compute. Every reasoning-delta measurement in this book (Part VII especially) is a *matched-budget* comparison for exactly this reason, and the curves this chapter's lab produces are what let me pick the matched budget honestly, at the point where both models' curves have roughly saturated, so neither is starved.
```

### Parallel versus sequential compute, and diminishing returns

The methods above split into two ways of spending compute, and they do not scale the same. Self-consistency and best-of-n are *parallel*: they draw independent samples that share nothing but the prompt, so they parallelize perfectly on the GPU (one batched vLLM call) and their accuracy follows the saturating pass@$n$ shape of equation (6.6.2). Self-refinement and budget forcing are *sequential*: each step conditions on the previous one, so they cannot be batched the same way and they add latency linearly, but they can in principle reach solutions parallel sampling never would, because a refinement can build on a specific earlier attempt rather than starting fresh. In practice, for verifiable tasks, parallel best-of-n with an outcome verifier is the workhorse because it is cheap to batch and its ceiling is high, while sequential methods earn their latency only when the model genuinely self-corrects. The unifying caveat is diminishing returns: every one of these curves flattens, and past the flattening point I am burning tokens for nothing. The compute-per-answer frontier is therefore not just a description, it is a budgeting tool, it tells me the smallest budget that reaches a target accuracy, which is the number I actually want when I am deciding how much inference to spend serving or evaluating a model.

## Tooling

The inference-time methods live in the *inference/eval* environment (vLLM plus Inspect), kept separate from the training environment per the two-environment doctrine. vLLM does the heavy lifting: its `n` sampling parameter draws multiple completions per prompt in one batched call (efficient because they share the prefill and the paged KV cache from Part II), which is exactly what self-consistency and best-of-n need. Inspect (Part III) supplies the task suite, the solvers, and the scorers, and its scorer *is* the verifier for best-of-n, the same object that will be the reward in the training loop, which is the "evals as rewards" identity showing up again. Budget forcing is implemented at the sampling layer by manipulating stop tokens and `max_tokens` and, for the "keep thinking" variant, by suppressing the end-of-thinking token and re-prompting. The measurement glue is chapter 3.7's `evalstats`: accuracy at each budget comes back as a `bootstrap_mean` with a CI, and before/after-training comparisons at a fixed budget go through `bootstrap_paired_diff` and `mcnemar`, so the scaling curves carry error bars and the matched-budget deltas carry p-values.

## Lab

The lab produces the central figure of the whole methodology: inference-time scaling curves on the frozen thesis suite. For a fixed model, it sweeps the number of samples $k$ and plots two curves against token budget, self-consistency (majority vote) and best-of-n (verifier-selected), each with bootstrap CIs. This shows the pass@$n$-shaped saturation of equation (6.6.2) directly, quantifies how much a verifier beats voting, and marks the matched budget I will use for every training comparison.

This is a `uv` project in the inference/eval environment. From the repo root:

```bash
uv init labs/its-scaling
cd labs/its-scaling
uv add vllm datasets numpy matplotlib
uv add --editable ../../packages/evalstats
```

```python title="labs/its-scaling/scaling_curves.py"
"""Inference-time scaling curves on the frozen thesis suite: self-consistency
(vote) vs best-of-n (verifier), accuracy vs token budget, with bootstrap CIs.
Writes artifacts/its_scaling.csv and artifacts/its_scaling.png.
"""
from collections import Counter
from pathlib import Path
import csv
import re

import numpy as np
from vllm import LLM, SamplingParams

import evalstats as es
# chapter-3.9 frozen suite: load_suite() -> Suite with .items (each Item has
# .prompt, .answer, .verifier) and verify(item, response) -> bool.
from thesis_suite import load_suite, verify

MODEL = "Qwen/Qwen3-4B"
KS = [1, 2, 4, 8, 16, 32]
MAX_K = max(KS)
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def extract_answer(text: str) -> str | None:
    """Canonical final answer for the majority vote. thesis_suite exposes a
    verifier (used below for correctness) but not an extractor, so pull the
    last boxed span or trailing number here to tally self-consistency votes."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def main() -> None:
    items = load_suite("v1.0").items
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096)
    sp = SamplingParams(n=MAX_K, temperature=0.8, top_p=0.95, max_tokens=2048)

    # One batched generation of MAX_K samples per item; sub-sample for small k.
    prompts = [it.prompt for it in items]
    outs = llm.generate(prompts, sp)

    # Per item, cache each sample's extracted answer, verifier verdict, tokens.
    per_item = []
    for it, out in zip(items, outs):
        rec = []
        for comp in out.outputs:
            ans = extract_answer(comp.text)
            rec.append((ans, verify(it, comp.text), len(comp.token_ids)))
        per_item.append(rec)

    rows = []
    for k in KS:
        sc_correct, bo_correct, tokens = [], [], []
        for rec in per_item:
            sub = rec[:k]                      # first k samples
            toks = sum(t for _, _, t in sub)
            tokens.append(toks)
            # self-consistency: majority vote over extracted answers
            votes = Counter(a for a, _, _ in sub if a is not None)
            top = votes.most_common(1)[0][0] if votes else None
            # correct if the voted answer matches (re-verify the voted answer)
            sc_correct.append(1 if top is not None and
                              any(ok for a, ok, _ in sub if a == top and ok) else 0)
            # best-of-n: correct if ANY sample passes the verifier
            bo_correct.append(1 if any(ok for _, ok, _ in sub) else 0)

        sc = es.bootstrap_mean(sc_correct, level=0.95)
        bo = es.bootstrap_mean(bo_correct, level=0.95)
        mean_tokens = float(np.mean(tokens))
        rows.append({"k": k, "mean_tokens": mean_tokens,
                     "sc_acc": sc.point, "sc_lo": sc.ci_low, "sc_hi": sc.ci_high,
                     "bo_acc": bo.point, "bo_lo": bo.ci_low, "bo_hi": bo.ci_high})
        print(f"k={k:2d}  tok/ans={mean_tokens:7.0f}  "
              f"vote={sc.point:.3f}  best-of-n={bo.point:.3f}")

    with (OUT / "its_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import matplotlib.pyplot as plt
    x = [r["mean_tokens"] for r in rows]
    for key, lo, hi, label in [("sc_acc", "sc_lo", "sc_hi", "self-consistency (vote)"),
                               ("bo_acc", "bo_lo", "bo_hi", "best-of-n (verifier)")]:
        y = [r[key] for r in rows]
        yl = [r[key] - r[lo] for r in rows]
        yh = [r[hi] - r[key] for r in rows]
        plt.errorbar(x, y, yerr=[yl, yh], marker="o", capsize=4, label=label)
    plt.xscale("log")
    plt.xlabel("compute per answer (mean generated tokens)")
    plt.ylabel("accuracy on thesis suite v1.0")
    plt.title("Inference-time scaling under matched budget")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "its_scaling.png", dpi=150)
    print(f"Artifacts: {(OUT / 'its_scaling.csv').resolve()}, "
          f"{(OUT / 'its_scaling.png').resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python scaling_curves.py
```

```admonish under-the-hood
Generating `n=32` samples per prompt in vLLM is far cheaper than 32 separate requests, because vLLM shares the prompt's prefill computation across all 32 continuations and packs their KV caches into the paged allocator from Part II. On the 16GB card the constraint is KV-cache memory, not compute: 32 concurrent 2048-token continuations of a 4B model is a real KV footprint, so `gpu_memory_utilization` and `max_model_len` are the knobs that keep it from OOM-ing. If you hit an out-of-memory during generation, lower `max_tokens` or generate the largest $k$ in sub-batches; the science is unchanged, only the batching.
```

**What you should see.** Both curves rise with token budget and then flatten, the pass@$n$ saturation of equation (6.6.2) made visible, and the best-of-n curve sits *above* the self-consistency curve at every budget because a verifier that recognizes correctness beats a popularity vote (voting cannot pick a correct answer that only one sample found, best-of-n can). The gap between the two curves is a direct measure of how much your verifier is worth. Both saturate by $k$ in the low tens, and the token budget where they flatten is the matched budget you will freeze for every training comparison in Part VII, chosen where both curves have plateaued so no model is starved. Record throughput (tokens/sec), peak VRAM, and wall-clock for the full sweep (measured on the baseline machine, record value, date, driver); the $k=32$ generation dominates the runtime and the KV cache dominates the memory. The artifact `its_scaling.png` is the figure that lets me say, for the rest of the book, "compared at a fixed budget of $N$ tokens per answer," and mean it.

```admonish read-along
Pair this with **[BRM]** ch. 4–5 for the reasoning-model methods themselves (chain-of-thought, self-consistency, verification, refinement, and the training that shapes good traces) and **[RLHF]** ch. 7 for how inference-time verification connects back to the reward and the RLVR objective. Read equation (6.6.2) here alongside their pass@$k$ discussion; it is the same curve, and it is the quantitative bridge between "train to raise $p$" and "sample to spend $p$."
```

```admonish substack-seed
"There are two ways to make a model smarter: change its weights, or let it think longer. Most benchmark fights are secretly about the second one." A post built on equation (6.6.2), the pass@$n$ curve, showing that a model with a 10-percent per-try success rate can be pushed near its ceiling with a few dozen samples and a verifier, and that this is a *second* scaling law running orthogonal to parameters. Then the methodological gut-punch: almost every "our model beats theirs" claim you have seen compares models at *different* compute-per-answer budgets, which makes the comparison meaningless, and the fix is a single discipline, matched-budget evaluation, that the scaling-curve figure makes easy.
```
