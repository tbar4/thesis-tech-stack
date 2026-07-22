# Metrics and their math

A metric is the function that turns a scored sample into a number, and picking the wrong one quietly changes the construct you are measuring. Exact-match and token-F1 answer different questions about the same output; `pass@1` and `pass@k` measure propensity and capability respectively (chapter 1); a calibration metric ignores accuracy entirely and asks whether the model's confidence is honest. This chapter is the arithmetic. I derive the two metrics that people most often get subtly wrong, the `pass@k` unbiased estimator and the Brier score, and the lab shows the `pass@k` bias as a concrete number so you can watch the naive estimator lie.

```admonish read-along
**[BRM] Raschka ch. 3** motivates why a reasoning model needs a verifiable, low-variance signal, which is exactly what a well-chosen metric provides; **[RLHF] Lambert Part 3** covers the metric-choice pitfalls from the alignment side. Read the `pass@k` derivation here alongside Raschka's discussion of sampling many candidate solutions: the estimator below is what makes "we sampled 64 and one worked" into an honest number instead of an anecdote. **[AIE]** ch. 3 walks the language-modeling metrics themselves (entropy, cross entropy, bits per byte, perplexity) with the practical caveats on when perplexity comparisons across models mislead; it pairs with this chapter's derivations the way a field guide pairs with anatomy.
```

## Theory

### Exact match

The simplest metric. A sample scores 1 if the extracted answer equals the reference exactly (after normalization), else 0. The eval score is the mean over items, which is just accuracy. Its virtue is that it is unambiguous and cheap; its vice is that "exactly" hides a normalization policy that is doing real work. Lowercasing, stripping punctuation, trimming whitespace, collapsing `1,000` to `1000`, mapping `forty-two` to `42`: every one of these is a decision that moves the score, and every one is construct-irrelevant variance if you get it wrong. For math I normalize numeric answers (strip commas, currency, trailing zeros after a decimal point) before comparing; for short-answer QA I lowercase and strip articles and punctuation. The metric is exact-match; the *policy* is where the judgment lives, and it belongs in the datasheet from chapter 1.

Formally, with items $i = 1 \dots N$, extracted answer $a_i$, reference $r_i$, and a normalizer $\mathcal{N}$,

$$
\text{EM} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\mathcal{N}(a_i) = \mathcal{N}(r_i)\right]. \tag{1}
$$

### Token-level F1

Exact-match is brutal when the answer is a phrase: "the Battle of Hastings" versus "Battle of Hastings" scores 0 under EM but is obviously right. Token-F1, the standard for extractive QA like SQuAD, softens this by treating the answer as a bag of tokens and measuring overlap. Let $P$ be the multiset of predicted tokens and $G$ the gold tokens. Precision is the fraction of predicted tokens that are correct, recall is the fraction of gold tokens recovered, and F1 is their harmonic mean:

$$
\text{prec} = \frac{|P \cap G|}{|P|}, \quad \text{rec} = \frac{|P \cap G|}{|G|}, \quad \text{F1} = \frac{2\,\text{prec}\cdot\text{rec}}{\text{prec}+\text{rec}}. \tag{2}
$$

The intersection is a multiset intersection (token counts matter). F1 rewards partial overlap, which is what you want for phrase answers and exactly what you do *not* want for a math final answer, where "42" and "142" share a token but one is wrong. Match the metric to the construct: F1 for phrase-valued answers, EM for atomic answers.

### pass@k and the unbiased estimator

For generative tasks where a verifier can check each sample (code that runs against tests, math with a checkable answer), the natural capability question is: if the model gets $k$ tries, does at least one succeed? That is `pass@k`. Its true value for a single problem, if each sample independently succeeds with probability $p$, is

$$
\text{pass@}k = 1 - (1-p)^k. \tag{3}
$$

The trouble is that I do not know $p$, and the obvious fix is badly biased. Two wrong-ish and one right way to estimate it:

**The naive per-problem experiment.** Draw exactly $k$ samples, score 1 if any passes. This is unbiased for a *single* $k$, but it wastes samples (you cannot reuse them to estimate `pass@k` at other values of $k$) and it is high variance. Worse, in practice people generate $n > k$ samples once and want `pass@k` for several $k$, which pushes them to the plug-in.

**The plug-in (naive) estimator.** Generate $n$ samples, count $c$ correct, estimate $\hat{p} = c/n$, and plug into equation (3): $\widehat{\text{pass@}k}_{\text{naive}} = 1 - (1-c/n)^k$. This is **biased**, and the lab makes the bias visible.

**The unbiased estimator** (from the Codex/HumanEval paper, Chen et al. 2021). Generate $n \ge k$ samples, count $c$ correct. Estimate `pass@k` as one minus the probability that a randomly chosen size-$k$ subset of the $n$ samples contains no correct sample:

$$
\widehat{\text{pass@}k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}. \tag{4}
$$

```admonish derivation title="Why equation (4) is exactly unbiased for 1 - (1-p)^k"
Draw the $n$ samples i.i.d., so the number correct is $c \sim \text{Binomial}(n, p)$. I claim $\mathbb{E}\!\left[\binom{n-c}{k}\big/\binom{n}{k}\right] = (1-p)^k$, which makes equation (4) unbiased for equation (3). Take the expectation over $c$:

$$
\mathbb{E}\!\left[\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
= \sum_{c=0}^{n} \binom{n}{c} p^c (1-p)^{n-c} \, \frac{\binom{n-c}{k}}{\binom{n}{k}}.
\tag{5}
$$

Use the subset-swap identity $\binom{n}{c}\binom{n-c}{k} = \binom{n}{k}\binom{n-k}{c}$ (choosing $c$ items then $k$ from the rest equals choosing $k$ items then $c$ from the rest). The $\binom{n}{k}$ cancels:

$$
= \sum_{c=0}^{n} \binom{n-k}{c} p^c (1-p)^{n-c}.
\tag{6}
$$

Since $\binom{n-k}{c} = 0$ for $c > n-k$, the sum runs to $n-k$. Pull out $(1-p)^k$:

$$
= (1-p)^k \sum_{c=0}^{n-k} \binom{n-k}{c} p^c (1-p)^{(n-k)-c}
= (1-p)^k \cdot 1,
\tag{7}
$$

because the remaining sum is the full binomial expansion of $\big(p + (1-p)\big)^{n-k} = 1$. Therefore $\mathbb{E}[\widehat{\text{pass@}k}] = 1 - (1-p)^k$, exactly. The naive plug-in has no such guarantee: $f(x) = 1-(1-x)^k$ is concave for $k \ge 2$ (since $f''(x) = -k(k-1)(1-x)^{k-2} < 0$), so by Jensen's inequality $\mathbb{E}[f(\hat{p})] \le f(\mathbb{E}[\hat{p}]) = f(p)$, and the plug-in is biased **low**. The lab measures the gap.
```

To report `pass@k` over a dataset, compute equation (4) per problem and average across problems. Numerically, do not form the binomials directly (they overflow); compute the ratio as a product, which the lab does.

### Majority vote (self-consistency)

`pass@k` asks whether *any* sample is right, which needs a verifier. When you have no verifier at inference time but can sample many chains, **majority vote** (self-consistency) asks which answer the samples *agree* on. Sample $k$ chains, extract each final answer, and return the mode. The metric scores 1 if the modal answer matches the reference. This is a propensity-flavored aggregation that trades compute for accuracy: it works when correct reasoning paths converge on the same answer while errors scatter. Formally, with extracted answers $a_1 \dots a_k$,

$$
\hat{a} = \arg\max_{v} \sum_{j=1}^{k} \mathbb{1}[a_j = v], \qquad \text{score} = \mathbb{1}[\hat{a} = r]. \tag{8}
$$

Majority vote and `pass@k` bracket the model: `pass@k` is optimistic (needs one right answer plus an oracle verifier), majority vote is realistic (no oracle, must self-agree). The gap between them measures how often the model *knows* the answer versus how often it can *find* it without help, which is again the capability/propensity split from chapter 1.

### Rubric scoring

Not every construct has a checkable answer. For open-ended generation (explanations, summaries, dialogue) the score comes from a rubric applied by a grader, human or model. A rubric decomposes quality into criteria (correctness, completeness, coherence) each scored on a scale, then aggregates, usually by a weighted mean:

$$
s = \sum_{m=1}^{M} w_m \, s_m, \qquad \sum_m w_m = 1. \tag{9}
$$

The metric here is only as good as the rubric's inter-rater reliability and the grader's calibration; model-graded rubric scoring is chapter 6's entire subject. The thing to internalize now is that rubric scoring reintroduces every construct-validity threat from chapter 1 at the grader level: a lenient or sycophantic judge is construct-irrelevant variance wearing a lab coat. For the thesis I lean on rubric scoring as little as possible and on verifiable scorers (chapter 4) as much as possible, precisely because a verifier has no opinion.

```admonish under-the-hood title="Macro versus micro averaging changes the number the tool reports"
When a dataset has subgroups of unequal size (GSM8K by difficulty, MMLU by subject), there are two ways to average and they disagree. **Micro-averaging** pools every item and takes one mean, so large subgroups dominate: $\text{acc}_{\text{micro}} = \frac{1}{N}\sum_i \mathbb{1}[\text{correct}_i]$. **Macro-averaging** computes each subgroup's accuracy then averages the subgroups equally, so a tiny subgroup counts as much as a huge one: $\text{acc}_{\text{macro}} = \frac{1}{G}\sum_{g} \text{acc}_g$. MMLU's headline number is macro over 57 subjects; a naive pooled mean would be micro and would differ. Neither is wrong, but they answer different questions ("how often is the model right on a random item" versus "how well does it do across topics evenly"), and reporting one while implying the other is a quiet construct mismatch. Inspect's `accuracy()` metric micro-averages by default; grouped reporting requires either a grouped metric or slicing the log by the `metadata` I attached in the dataset mapping (chapter 3). Whichever I use goes in the datasheet, because a committee comparing my number to a published one needs to know which average produced it.
```

### Calibration and the Brier score

Accuracy asks whether the model is *right*. Calibration asks whether the model's stated confidence is *honest*: when it says 80%, is it right 80% of the time? A model can be accurate and miscalibrated (right often but always says 99%) or calibrated and inaccurate (right 40% of the time and says 40%). For a reasoning model whose confidence I might use to route, abstain, or weight votes, calibration is its own construct with its own metric.

The **Brier score** is the mean squared error of probabilistic predictions. For binary outcomes with forecast probabilities $f_i \in [0,1]$ and outcomes $o_i \in \{0,1\}$,

$$
\text{BS} = \frac{1}{N}\sum_{i=1}^{N}(f_i - o_i)^2. \tag{10}
$$

It ranges from 0 (perfect) to 1 (perfectly wrong), and it is a **proper scoring rule**: it is minimized in expectation by reporting your true belief, so a model cannot game it by hedging. Its value is illuminated by the Murphy decomposition.

```admonish derivation title="The Murphy decomposition: reliability, resolution, uncertainty"
Bin the $N$ predictions into $K$ groups by forecast value, so group $k$ has $n_k$ items all forecasting $f_k$, with observed hit rate $o_k = \frac{1}{n_k}\sum_{i \in k} o_i$. Let $\bar{o} = \frac{1}{N}\sum_i o_i$ be the overall base rate. Then the Brier score decomposes exactly as

$$
\text{BS} = \underbrace{\frac{1}{N}\sum_{k} n_k (f_k - o_k)^2}_{\text{reliability (calibration)}}
\;-\; \underbrace{\frac{1}{N}\sum_{k} n_k (o_k - \bar{o})^2}_{\text{resolution}}
\;+\; \underbrace{\bar{o}(1-\bar{o})}_{\text{uncertainty}}. \tag{11}
$$

**Reliability** is the mean squared gap between forecast confidence and realized frequency; it is 0 exactly when the model is calibrated, and it is the term I want small. **Resolution** rewards forecasts that separate outcomes (bins whose hit rates differ from the base rate); it is *subtracted*, so more resolution is better. **Uncertainty** is the irreducible variance of the outcome itself, set by the base rate, independent of the model. The sketch: write $(f_i - o_i)^2 = ((f_i - o_k) + (o_k - o_i))^2$ within each bin, expand, and note the cross term vanishes because $\sum_{i \in k}(o_i - o_k) = 0$ by definition of $o_k$; then add and subtract $\bar{o}$ inside the outcome-variance piece and use $\frac{1}{N}\sum_i (o_i - \bar{o})^2 = \bar{o}(1-\bar{o})$ for binary outcomes. The decomposition is why Brier beats bare accuracy for calibration work: it separates "is the confidence honest" (reliability) from "is the confidence useful" (resolution).
```

A close cousin worth naming is **Expected Calibration Error**, the reliability term without the squaring, using absolute gaps and probability-weighted bins:

$$
\text{ECE} = \sum_{k} \frac{n_k}{N}\,\bigl|\,f_k - o_k\,\bigr|. \tag{12}
$$

ECE is more interpretable ("on average the confidence is off by 7 points") but is not a proper scoring rule and is sensitive to binning, so I report Brier as the headline and ECE as the human-readable gloss.

```admonish gotcha title="A proper scoring rule cannot be gamed by hedging; accuracy can"
If I reward a model purely on accuracy with a confidence threshold, it learns to state maximal confidence always, because unconfident-but-correct and confident-but-correct score the same. Brier punishes that: stating 99% and being wrong costs $(0.99)^2 \approx 0.98$, near the maximum. This is the same reason verifiable rewards in Part V must be designed as proper incentives; a metric that can be satisfied without doing the thing you care about will be, especially once you optimize against it (Goodhart). Choosing a proper metric is choosing an incentive.
```

## Tooling

Both Inspect and lm-evaluation-harness ship these metrics, and knowing which name maps to which equation saves you from silently comparing apples to oranges. In Inspect, metrics are functions attached to a scorer: `accuracy()` is exact-match's mean (equation 1), `mean()` and `stderr()` give you the reading and its noise floor together (the chapter-1 discipline, enforced by the tool), and `pass_at_k` style aggregation is available for multi-sample tasks. In the harness, the task YAML names a metric (`exact_match`, `acc`, `f1`) and a filter pipeline that does the answer extraction and normalization before the metric sees the string, which is where equation (1)'s normalization policy actually lives.

The load-bearing point for the thesis: whatever metric I use for measurement in Part III, I have to be able to reuse as a reward in Part V, and rewards must be proper and verifier-backed. That biases me hard toward exact-match on verifiable answers and `pass@k` for capability probes, and away from rubric scoring in the training loop. When I do want calibration, I compute Brier from logged per-sample confidences, which both tools can emit into their logs. The metric is not an afterthought bolted onto a run; it is the thing the whole instrument exists to compute, so I pick it first and design the extraction and normalization to serve it.

## Lab

The artifact is a script that estimates `pass@k` both the naive way and the unbiased way against a known ground truth, so the bias is not a claim in a paper but a number on my screen. I fix a true per-sample success probability $p$, simulate a model by drawing $n$ Bernoulli samples per trial, estimate `pass@k` both ways over many trials, and compare each estimator's average to the exact truth $1-(1-p)^k$.

```bash
uv init passk-bias
cd passk-bias
uv add numpy
```

```python title="passk/estimators.py"
"""pass@k estimators: naive plug-in vs the unbiased combinatorial estimator.

The unbiased estimator is from Chen et al. 2021 (HumanEval/Codex):
    pass@k = 1 - C(n-c, k) / C(n, k)
computed as a numerically stable product to avoid binomial overflow.
"""
from __future__ import annotations


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k given c correct out of n samples (n >= k)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        # fewer than k incorrect samples => every size-k subset hits a correct one
        return 1.0
    # 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)   == 1 - C(n-c,k)/C(n,k)
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def pass_at_k_naive(n: int, c: int, k: int) -> float:
    """Biased plug-in estimator: 1 - (1 - c/n)^k."""
    p_hat = c / n
    return 1.0 - (1.0 - p_hat) ** k
```

```python title="passk/experiment.py"
"""Show the pass@k bias numerically.

For a known true p, simulate many 'problems', each with n independent samples,
and estimate pass@k both ways. The unbiased estimator's average should match the
analytic truth 1-(1-p)^k; the naive plug-in should sit noticeably below it.

Run: uv run python -m passk.experiment
"""
from __future__ import annotations

import numpy as np

from passk.estimators import pass_at_k_naive, pass_at_k_unbiased

RNG = np.random.default_rng(0)


def truth(p: float, k: int) -> float:
    return 1.0 - (1.0 - p) ** k


def run(p: float, n: int, k: int, trials: int) -> tuple[float, float, float]:
    naive, unbiased = np.empty(trials), np.empty(trials)
    for t in range(trials):
        c = int(RNG.binomial(n, p))          # samples correct this trial
        naive[t] = pass_at_k_naive(n, c, k)
        unbiased[t] = pass_at_k_unbiased(n, c, k)
    return truth(p, k), float(naive.mean()), float(unbiased.mean())


def main() -> None:
    n, trials = 20, 100_000
    print(f"n={n} samples/problem, {trials:,} simulated problems each row\n")
    header = f"{'p':>5} {'k':>3} {'true':>8} {'naive':>9} {'unbiased':>9} " \
             f"{'naive_bias':>11} {'unb_bias':>9}"
    print(header)
    print("-" * len(header))
    for p in (0.05, 0.1, 0.3):
        for k in (1, 5, 10, 20):
            tru, nav, unb = run(p, n, k, trials)
            print(f"{p:>5.2f} {k:>3} {tru:>8.4f} {nav:>9.4f} {unb:>9.4f} "
                  f"{nav - tru:>+11.4f} {unb - tru:>+9.4f}")
    print("\nnaive_bias = naive_avg - true ; unb_bias = unbiased_avg - true")
    print("The unbiased estimator's bias should be ~0 (Monte-Carlo noise only);")
    print("the naive estimator should be biased LOW and worst near k = n.")


if __name__ == "__main__":
    main()
```

Run it.

```bash
uv run python -m passk.experiment
```

### What you should see

A table where the `unbiased` column tracks the `true` column to within Monte-Carlo noise (the `unb_bias` column near $\pm 0.001$), while the `naive` column sits systematically below `true`, with the gap growing as $k$ approaches $n=20$. At $p=0.1, k=20$ the truth is $1-(0.9)^{20} \approx 0.878$; the naive plug-in averages meaningfully lower not because of any single count (at the mean $c=2$ the naive term $1-(1-2/20)^{20}$ is essentially exact) but because the per-problem estimate is *concave* in $c$ and $c$ has a whole distribution. Jensen's inequality over that distribution pushes the average below the value at the mean, and the probability mass piled at the low counts (a problem with $c=0$ contributes exactly $0$, one with $c=1$ contributes only a small term) is what drags the mean down. The sign is always the same: `naive_bias` is negative, matching the Jensen argument in the derivation, that the concave plug-in is biased low. The magnitude is largest exactly where you are most tempted to use `pass@k`, at large $k$ probing capability.

The artifacts on disk are `passk/estimators.py` (the two estimators, the unbiased one written as a stable product so it survives real $n$ like 64 or 256) and `passk/experiment.py` (the bias demonstration). When I report a `pass@64` capability number for my 8B model later in the book, it comes out of `pass_at_k_unbiased`, and this lab is the receipt showing why the other function would have quietly understated the model's headroom, which is the headroom RL is going to convert into propensity.

```admonish substack-seed
"The pass@k you're quoting is probably biased low, and here's the two-line fix." Everyone reporting how often their model solves a problem in $k$ tries reaches for the obvious estimate: measure the success rate, then compute one-minus-(one-minus-rate)-to-the-$k$. It's wrong, and it's wrong in a specific direction: it understates the model's true capability, worst exactly when you sample many times to probe the ceiling. The post walks through the one-paragraph combinatorics proof that the plug-in is biased by Jensen's inequality, gives the unbiased combinatorial estimator from the Codex paper, and then, the fun part, shows the bias as a live number from a 30-line simulation. The moral generalizes past pass@k: a metric is a claim with a sampling distribution, and "obvious" estimators of nonlinear quantities are biased by default. If you're going to put a number in a table a committee reads, you should be able to say which way it's wrong when it's wrong.
```
