# Measuring the reasoning delta

The training run finished. I have a baseline policy $M_0$ and a trained policy $M_1$, and somewhere between them is the thing the whole thesis is named after: the reasoning delta, the amount by which training on verifiable rewards moved a real measure of reasoning. This chapter is where I earn the word "delta." It would be trivial to run both models on the eval suite, subtract two accuracy numbers, and put an arrow on a slide. It would also be indefensible, because two accuracy numbers with no uncertainty, no pairing, and no effect size are not evidence, they are decoration. A thesis committee will ask three questions in this order: is the difference real, how big is it, and what exactly does it claim. This chapter answers all three, and it does so by reusing the `evalstats` module built in chapter 3.7 rather than inventing statistics on the spot.

The output is a delta-report template: an analysis artifact that consumes the per-item, per-sample eval logs from a pre run and a post run and emits a report a committee can read. I cannot fill it with real numbers here because there is no GPU in this authoring environment, so every measured slot is stamped `(measured on the baseline machine — record value, date, driver)` and the chapter delivers the code and templates that would consume the real numbers.

## Theory

### The pre/post design and why matching is everything

The design is a within-subjects pre/post comparison. The "subjects" are the items of the frozen thesis task suite v1.0 (chapter 3.11), and each item is measured twice: once under $M_0$, once under $M_1$. This is the cleanest design available on one GPU, and its power comes entirely from a discipline that is easy to state and easy to violate: everything except the model must be held identical between the pre and post measurements.

Identical means the same item set (the frozen suite, same revision, no items added or dropped), the same sampling budget (same number of samples $n$ per item), the same decoding parameters (temperature, top-p, max tokens), and the same seeds wherever the serving stack makes seeds reproducible. The reason matching matters so much is not fussiness. It is that the metrics I care about are functions of the sampling budget, and if the budget differs between pre and post then a "delta" partly measures the budget change, not the model. Pass@k in particular is monotonic in k, so measuring $M_0$ at $n=4$ and $M_1$ at $n=8$ would manufacture a delta out of thin air. The matched budget is what makes the subtraction meaningful.

There is a causal-inference reading of this that chapter 4.5 develops in full: training is a $do$-operation on the policy, the pre/post pair is an interrupted time series with a single intervention, and the shared item set is what lets me difference out the item-level confounder (some problems are just harder than others). The control runs of chapter 4.5, especially the random-reward placebo, exist to rule out the alternative explanation that any training at all, regardless of the reward signal, would move the metric. This chapter measures the delta; chapter 4.5 hardens the claim that the delta is caused by the verifiable reward. The trained policy $M_1$ and its measured delta then become the RL-trained arm of chapter 8.3, where the same paired machinery separates this parametric gain from what retrieval and tools add.

### Paired statistics on shared items

Because every item is measured under both models, the natural unit of analysis is the per-item difference, not the two group means. Let $s_0(i)$ and $s_1(i)$ be the per-item scores (for a binary metric, the item's mean-of-$n$ accuracy) for models $M_0$ and $M_1$ on item $i$, over $N$ items. The paired difference is $d_i = s_1(i) - s_0(i)$, and the estimand is its mean:

$$
\bar d = \frac{1}{N}\sum_{i=1}^{N} d_i = \bar s_1 - \bar s_0 .
\tag{6.1}
$$

The mean paired difference equals the difference of means, so pairing does not change the point estimate. What it changes, dramatically, is the uncertainty. Here is why, and it is the single most important equation in the chapter.

```admonish derivation
The variance of the mean paired difference is
$$
\operatorname{Var}(\bar d) = \frac{1}{N}\left(\sigma_0^2 + \sigma_1^2 - 2\rho\,\sigma_0\sigma_1\right),
\tag{6.2}
$$
where $\sigma_0^2, \sigma_1^2$ are the item-score variances under each model and $\rho$ is the correlation between $s_0(i)$ and $s_1(i)$ across items. Compare this to the unpaired (two-sample) variance, which drops the $-2\rho\sigma_0\sigma_1$ term entirely:
$$
\operatorname{Var}_{\text{unpaired}}(\bar s_1 - \bar s_0) = \frac{\sigma_1^2}{N} + \frac{\sigma_0^2}{N}.
\tag{6.3}
$$
Item difficulty is strongly shared between the two models: a problem that is hard for $M_0$ is usually hard for $M_1$, so $\rho$ is large and positive, often $0.7$ or higher for models that differ only by a fine-tune. Plug $\rho = 0.8$ and $\sigma_0 \approx \sigma_1 = \sigma$ into (6.2) and the paired variance is $\frac{2\sigma^2(1-\rho)}{N} = \frac{0.4\sigma^2}{N}$, which is one fifth of the unpaired $\frac{2\sigma^2}{N}$. That factor of five in variance is a factor of $\sqrt{5} \approx 2.2$ in the width of the confidence interval, bought for free by the design choice to reuse the same items. On one GPU, where $N$ is bounded by how many eval generations you can afford, that free variance reduction is the difference between a detectable delta and a shrug.
```

Equation (6.2) is also the argument for the whole enterprise: I do not detect small reasoning deltas by running more items than a datacenter can, I detect them by pairing on shared items so the item-difficulty variance cancels. The `evalstats` module from chapter 3.7 already implements the paired estimators, so the analysis code imports them rather than rederiving them.

### Confidence intervals: the paired bootstrap

I do not trust a normal-theory interval for a bounded, discrete, possibly skewed metric like accuracy, so the confidence interval comes from a bootstrap, exactly the one built in chapter 3.7. The subtlety for a pre/post comparison is that the bootstrap must resample the *pairs*, not the two arms independently, or it destroys the correlation that equation (6.2) says is doing all the work.

There is a second layer of variance that a careful committee will ask about: each item's score is itself an average over $n$ noisy rollouts, so the total uncertainty has a between-item component and a within-item (sampling) component. The clean way to capture both is a cluster bootstrap that resamples items with replacement and, within each resampled item, resamples its $n$ rollouts with replacement. That propagates both sources of noise into the interval. For the headline metric I report the paired item-level bootstrap CI, and I note the cluster bootstrap as the more conservative version; chapter 3.7's module exposes both.

### Significance tests that do not assume normality

A confidence interval that excludes zero is already a significance statement, but committees like a $p$-value next to it, and the honest one for paired data is a permutation test, again from chapter 3.7. Under the null hypothesis that training had no effect, the sign of each paired difference $d_i$ is exchangeable: flipping which model is "pre" and which is "post" should not matter. So I build the null distribution of $\bar d$ by randomly flipping the signs of the $d_i$ many times and computing $\bar d$ each time, and the two-sided $p$-value is the fraction of sign-flipped $\bar d$ at least as extreme as the observed one. This assumes nothing about the shape of the score distribution and it respects the pairing.

For a strictly binary framing (did the item go from wrong to right or right to wrong) the discrete analogue is McNemar's test on the discordant pairs, and I report it when the metric is naturally item-level pass/fail rather than a mean-of-$n$ rate. The two agree in spirit; the permutation test is more general because it handles continuous per-item scores.

### Effect sizes a committee accepts

A $p$-value tells you the delta is not zero. It does not tell you the delta is worth caring about, and with enough items even a trivially small delta becomes "significant." So the report leads with effect sizes, not $p$-values.

The first effect size is just the raw delta in the metric's own units, with its CI: "$M_1$ solved $\bar d$ more of the suite per attempt than $M_0$, 95% CI $[\ell, u]$" (measured on the baseline machine — record value, date, driver). Units matter and this one has them: it is in probability-of-success-per-sample, which is the thing anyone actually cares about.

The second is a standardized effect size, for comparison across metrics and against the literature. For paired data the natural one is

$$
d_z = \frac{\bar d}{s_d}, \qquad s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (d_i - \bar d)^2},
\tag{6.4}
$$

Cohen's $d$ for paired differences: the mean difference in units of the standard deviation of the differences. It answers "how large is the shift relative to how much items disagree about the shift." I report $d_z$ with a bootstrap CI, and I resist the urge to launder it into the unpaired Cohen's $d$, because $d_z$ and the between-groups $d$ are different quantities and conflating them is a classic way to overstate an effect.

```admonish derivation
For a binary item-level outcome (each item either flips wrong-to-right, right-to-wrong, or stays), the discordant-pair effect size is transparent. Let $b$ be the count of items that went wrong-to-right and $c$ the count that went right-to-wrong, out of $N$. Then
$$
\bar d = \frac{b - c}{N},
\tag{6.5}
$$
which is the net flip rate, and McNemar's statistic is $\chi^2 = \frac{(b-c)^2}{b+c}$ on one degree of freedom. The quantity $b + c$ is the number of items that changed at all, and it is a sanity number worth reporting on its own: if $b + c$ is tiny, the model barely changed its item-level verdicts and any $\bar d$ is riding on a handful of items, which is exactly the situation where the bootstrap CI will be honest and wide. Reporting $b$ and $c$ separately also exposes a failure the net rate hides: a run that fixes 20 items and breaks 18 has $\bar d = 2/N$ and looks like mild progress, but it is really churning, and $b=20, c=18$ says so out loud.
```

### Pass@k shifts versus mean shifts, and what each one claims

This is the distinction that separates a defensible thesis claim from an inflated one, and it is subtle enough that it gets its own section.

There are two natural metrics on a set of $n$ samples per item. The mean accuracy (equivalently pass@1 in expectation) is the probability that a single sample is correct. Pass@k is the probability that at least one of $k$ samples is correct. The unbiased estimator of pass@k from $n \ge k$ samples with $c$ correct, from the HumanEval line of work, is

$$
\widehat{\text{pass@}k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} .
\tag{6.6}
$$

Now the important part. A training run can move these two metrics in different directions, and each direction makes a different scientific claim:

If mean accuracy (pass@1) rises but pass@k at large k is flat, the model has become *more reliable at problems it could already sometimes solve*. It concentrated probability mass onto correct solutions it was already capable of sampling. This is real and valuable, but the honest claim is "sharpening," not "the model can now solve problems it could not before." The set of solvable problems (the support at large k) did not grow.

If pass@k at large k rises, the model can now solve problems that were outside its reach at any sampling budget before. That is the stronger claim, capability expansion, and it is the one people usually mean by "the model got smarter." It is also the harder one to demonstrate and the one RLVR is most often accused of *not* delivering, since a live debate holds that RL on verifiable rewards mostly sharpens the base model's existing distribution rather than expanding it.

The committee-grade move is to report both, with matched budgets and the unbiased estimator (6.6), and to state which claim the data supports rather than letting a pass@1 gain masquerade as capability expansion. If pass@1 is up and pass@k is flat, I say "sharpening" and I am right. If both are up, I say "expansion" and I have the pass@k curve to back it. Either way the report shows the pass@k-versus-k curves for $M_0$ and $M_1$ on the same axes, because the *shape* of the gap as $k$ grows is the whole argument.

```admonish thesis-thread
For the Space Domain Awareness (SDA) verifiable-reasoning worked example, this chapter is where the thread finally produces a number. The frozen SDA items in thesis-suite v1.0 are evaluated under $M_0$ (the served baseline) and under $M_1$ (the GRPO-trained checkpoint from the 7.3 run), at a matched sampling budget of $n$ samples per item. The primary estimand is the mean paired difference $\bar d$ in per-sample solve rate on the SDA items, with a paired-bootstrap 95% CI and a sign-flip permutation $p$-value, and the secondary estimand is the pass@k curve for both models so the thread can say honestly whether training sharpened the SDA distribution or expanded it. Every one of those numbers is measured on the baseline machine — record value, date, driver; this chapter delivers the analysis that will consume them, and chapter 7.8 runs the loop that produces them for real. The thread's whole point was to carry one verifiable problem from first eval to trained model, and equation (6.1) applied to the SDA items is that carry made quantitative.
```

## Tooling

The tool is a delta-report generator. It reads two eval-log files (the per-item, per-sample scores that the chapter 3.7 eval harness already emits for a pre run and a post run), aligns them on item id, and produces a report object plus a rendered markdown report plus a pass@k figure. It imports every statistic from `evalstats`; the generator itself only aligns, calls, and formats.

Set it up in the training or analysis environment with uv:

```bash title="shell: create the analysis project"
uv init analysis
cd analysis
uv add "numpy>=1.26" "scipy>=1.13" "matplotlib>=3.8" "pydantic>=2.7"
uv add --editable ../evalstats   # the chapter 3.7 module, reused not rebuilt
```

### The input contract

Both eval runs write the same shape: one record per item, carrying the item id and the list of per-sample binary scores. Matching budgets means both files have the same item ids and the same length-$n$ score lists.

```python title="analysis/schema.py"
"""Input contract for a pre/post delta analysis.

An eval log is a list of ItemResult, one per suite item. The pre and
post logs MUST share item ids and MUST have equal samples-per-item n,
or the analysis refuses to run -- matched budgets are not optional.
"""
from __future__ import annotations

from pydantic import BaseModel


class ItemResult(BaseModel):
    item_id: str
    scores: list[float]   # per-sample binary correctness, length n


class EvalLog(BaseModel):
    model_name: str
    suite: str            # e.g. "thesis-suite v1.0"
    suite_revision: str
    n_samples: int        # samples per item; must match across pre/post
    temperature: float
    max_tokens: int
    # measured on the baseline machine -- record value, date, driver
    driver: str
    measured_at: str
    items: list[ItemResult]
```

### The delta report itself

```python title="analysis/delta_report.py"
"""Generate a committee-grade pre/post delta report.

Consumes two EvalLog files (pre = M0, post = M1) and emits:
  - delta_report.json   (machine-readable, every estimate + CI)
  - delta_report.md     (the template, human-readable)
  - passk_curve.png     (pass@k vs k for both models, matched budget)

All statistics come from evalstats (chapter 3.7). This file aligns,
calls, and formats -- it does not define a single estimator itself.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

from schema import EvalLog
# chapter 3.7 machinery, imported verbatim:
from evalstats import (
    paired_bootstrap_ci,     # (d_i, n_boot, alpha) -> (lo, hi)
    sign_flip_permutation,   # (d_i, n_perm) -> two-sided p-value
    cohens_dz,               # (d_i) -> standardized paired effect
    mcnemar,                 # (b, c) -> (chi2, p)
)


def pass_at_k(scores: list[float], k: int) -> float:
    """Unbiased pass@k from n samples with c correct (eq. 6.6)."""
    n = len(scores)
    c = int(round(sum(scores)))
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def align(pre: EvalLog, post: EvalLog) -> list[str]:
    """Return shared item ids; enforce matched budgets."""
    if pre.n_samples != post.n_samples:
        raise ValueError(
            f"budget mismatch: pre n={pre.n_samples}, post n={post.n_samples}; "
            "matched sampling budgets are required (see eq. 6.6 monotonicity)"
        )
    pre_ids = {it.item_id for it in pre.items}
    post_ids = {it.item_id for it in post.items}
    shared = sorted(pre_ids & post_ids)
    if len(shared) != len(pre_ids) or len(shared) != len(post_ids):
        raise ValueError("item sets differ; pre/post must share the frozen suite")
    return shared


def build(pre: EvalLog, post: EvalLog, n_boot: int = 10_000, n_perm: int = 10_000):
    shared = align(pre, post)
    pre_by = {it.item_id: it for it in pre.items}
    post_by = {it.item_id: it for it in post.items}

    # per-item mean-of-n accuracy under each model
    s0 = np.array([float(np.mean(pre_by[i].scores)) for i in shared])
    s1 = np.array([float(np.mean(post_by[i].scores)) for i in shared])
    d = s1 - s0                                            # eq. 6.1

    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot, alpha=0.05)
    p_perm = sign_flip_permutation(d, n_perm=n_perm)
    dz = cohens_dz(d)                                       # eq. 6.4

    # item-level discordant pairs on the majority verdict (eq. 6.5)
    v0 = (s0 >= 0.5).astype(int)
    v1 = (s1 >= 0.5).astype(int)
    b = int(np.sum((v0 == 0) & (v1 == 1)))   # wrong -> right
    c = int(np.sum((v0 == 1) & (v1 == 0)))   # right -> wrong
    chi2, p_mcnemar = mcnemar(b, c)

    # pass@k curves at matched budget
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= pre.n_samples]
    passk0 = {k: float(np.mean([pass_at_k(pre_by[i].scores, k) for i in shared])) for k in ks}
    passk1 = {k: float(np.mean([pass_at_k(post_by[i].scores, k) for i in shared])) for k in ks}

    corr = float(np.corrcoef(s0, s1)[0, 1])   # the rho of eq. 6.2

    return {
        "suite": pre.suite,
        "suite_revision": pre.suite_revision,
        "n_items": len(shared),
        "n_samples": pre.n_samples,
        "pre_model": pre.model_name,
        "post_model": post.model_name,
        "driver": post.driver,
        "measured_at": post.measured_at,
        "mean_pre": float(s0.mean()),
        "mean_post": float(s1.mean()),
        "delta_mean": float(d.mean()),
        "delta_ci95": [lo, hi],
        "perm_p": p_perm,
        "cohens_dz": dz,
        "pair_correlation_rho": corr,
        "flips_wrong_to_right": b,
        "flips_right_to_wrong": c,
        "mcnemar_chi2": chi2,
        "mcnemar_p": p_mcnemar,
        "pass_at_k_pre": passk0,
        "pass_at_k_post": passk1,
    }
```

The rendering half turns that dict into the report template and the figure. I keep it separate so the numbers and their presentation never entangle.

```python title="analysis/render.py"
"""Render a delta report dict into markdown + a pass@k figure."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEMPLATE = """# Reasoning delta report

**Suite.** {suite} (rev {suite_revision}), {n_items} items, {n_samples} samples/item (matched).
**Models.** pre = {pre_model} -> post = {post_model}.
**Provenance.** measured on the baseline machine (MSI Aegis R2), driver {driver}, {measured_at}.

## Headline delta (mean-of-n / pass@1)
- pre mean solve rate:  {mean_pre:.4f}
- post mean solve rate: {mean_post:.4f}
- **delta (paired):     {delta_mean:+.4f}**, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]
- permutation p (two-sided, sign-flip): {perm_p:.4g}
- paired effect size (Cohen's d_z): {cohens_dz:+.3f}
- pair correlation rho (drives paired-CI width, eq. 6.2): {rho:.3f}

## Item-level flips (discordant pairs, eq. 6.5)
- wrong -> right (b): {b}
- right -> wrong (c): {c}
- McNemar chi2 = {mcnemar_chi2:.3f}, p = {mcnemar_p:.4g}
- net flip rate (b - c)/N = {net_flip:+.4f}

## Claim
{claim}

![pass@k](passk_curve.png)
"""


def _claim(r: dict) -> str:
    ks = sorted(r["pass_at_k_pre"], key=int)
    kmax = ks[-1]
    d1 = r["delta_mean"]
    dpk = r["pass_at_k_post"][kmax] - r["pass_at_k_pre"][kmax]
    if d1 > 0 and dpk <= 0.005:
        return (f"pass@1 rose by {d1:+.4f} while pass@{kmax} is flat "
                f"({dpk:+.4f}): SHARPENING, not capability expansion.")
    if dpk > 0.005:
        return (f"pass@{kmax} rose by {dpk:+.4f}: capability EXPANSION "
                f"(model solves items unreachable at any budget before).")
    return "no material shift in either pass@1 or pass@k at this budget."


def render(r: dict, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)

    ks = sorted(r["pass_at_k_pre"], key=int)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ks, [r["pass_at_k_pre"][k] for k in ks], "o-", label=r["pre_model"])
    ax.plot(ks, [r["pass_at_k_post"][k] for k in ks], "s-", label=r["post_model"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (samples)")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k vs k (matched budget)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "passk_curve.png", dpi=150)

    md = TEMPLATE.format(
        ci_lo=r["delta_ci95"][0], ci_hi=r["delta_ci95"][1],
        rho=r["pair_correlation_rho"],
        b=r["flips_wrong_to_right"], c=r["flips_right_to_wrong"],
        net_flip=(r["flips_wrong_to_right"] - r["flips_right_to_wrong"]) / r["n_items"],
        claim=_claim(r), **r,
    )
    (out_dir / "delta_report.md").write_text(md)
    (out_dir / "delta_report.json").write_text(__import__("json").dumps(r, indent=2))
```

## Lab: the full pre/post analysis of the 7.3 run

The lab wires the pieces into one command that consumes the two eval logs from the 7.3 run and writes the delta-report artifact.

```python title="analysis/run_delta.py"
"""Pre/post delta analysis of the 7.3 GRPO run -> delta-report artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

from schema import EvalLog
from delta_report import build
from render import render


def load(path: str) -> EvalLog:
    return EvalLog.model_validate_json(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre", default="logs/eval_M0.json",
                    help="baseline M0 eval log (chapter 3.7 harness format)")
    ap.add_argument("--post", default="logs/eval_M1.json",
                    help="trained-checkpoint M1 eval log")
    ap.add_argument("--out", default="out",
                    help="output dir for delta_report.{md,json} + passk_curve.png")
    args = ap.parse_args()

    pre = load(args.pre)               # baseline eval log
    post = load(args.post)             # trained-checkpoint eval log
    report = build(pre, post)
    render(report, Path(args.out))
    print(f"delta = {report['delta_mean']:+.4f}  "
          f"95% CI {report['delta_ci95']}  "
          f"perm p = {report['perm_p']:.4g}  "
          f"d_z = {report['cohens_dz']:+.3f}")


if __name__ == "__main__":
    main()
```

```bash title="shell: run the analysis"
uv run python run_delta.py
```

### What you should see

Under `analysis/out/` three files: `delta_report.json`, `delta_report.md`, and `passk_curve.png`. The markdown report is the artifact, and reading top to bottom it tells the committee everything in order. It names the frozen suite and revision and states that the sampling budget was matched, so no one can accuse the delta of being a budget artifact. It gives the pre and post mean solve rates and, in bold, the paired delta with its 95% bootstrap CI and a permutation $p$-value, so "is it real" is answered with an interval and a test rather than an assertion. It reports Cohen's $d_z$ and the pair correlation $\rho$, so "how big" is answered in standardized units and the reader can see the very correlation that equation (6.2) says shrank the interval. It breaks the change into wrong-to-right and right-to-wrong flip counts, so a churning run cannot hide behind a modest net rate. And it ends with an explicit claim line that reads either SHARPENING or EXPANSION based on whether pass@k moved at the largest matched $k$, next to a figure showing both pass@k curves on the same log-scaled axis.

Every number in that report is stamped as measured on the baseline machine, and until the 7.3 run has actually been evaluated on the baseline machine those slots read `record value, date, driver`. That is deliberate: an unfilled delta is impossible to mistake for a real one, and the template is the thing this chapter ships. When chapter 7.8 runs the whole loop for real, this exact command consumes the real logs and the report fills itself in.

```admonish gotcha
The most common way to fake a delta without meaning to is to let the sampling budget drift between pre and post because you re-ran the eval with a different `n` after tweaking something. Equation (6.6) makes pass@k monotonic in $n$-derived $k$, and mean-of-$n$ has its own $n$-dependent variance, so a budget change alone can shift both metrics. The `align` function refuses to run on mismatched `n_samples` for exactly this reason. If it throws, do not "fix" it by trimming one log to match; re-run the shorter eval at the correct budget, because trimming changes which samples you kept and quietly biases the item scores.
```

```admonish substack-seed
Here is a post that will annoy people in the right way: most reported "the model got smarter" results are really "the model got more consistent," and the difference is one plot. If you sample a model many times per problem and measure whether at least one attempt succeeds (pass@k at large k), you see the ceiling of what the model can do at all. If you measure whether a single attempt succeeds (pass@1), you see how reliably it does it. Reinforcement learning on checkable rewards tends to raise the second while leaving the first alone: it sharpens the model onto solutions it could already occasionally find, rather than teaching it genuinely new ones. That is a real and useful improvement, but it is a different claim, and the honest version of "our RL made the model better" specifies which curve moved. The hook: reliability and capability look identical on a bar chart and completely different on a pass@k curve, and only one of them is what everyone thinks you mean.
```
