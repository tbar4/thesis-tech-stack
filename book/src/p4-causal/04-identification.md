# Identification: backdoor and front-door

**Goal.** Turn a causal question into a formula I can compute from observational data, or prove that I cannot. This is the step that converts a DAG plus a query into an estimator, and it is where "the model got better" either becomes defensible or gets honestly abandoned.

**Covers.** Adjustment sets; the backdoor criterion; front-door logic; what is and is not identifiable in the model-versus-model comparisons I actually run.

## Theory

### Identification is the hinge of the whole part

Everything so far has been diagnosis. Chapter 4.1 said my thesis claims live on rung two. Chapters 4.2 and 4.3 gave me the graph and the vocabulary of the ways paths open and close. Identification is the payoff: given a causal query $P(Y \mid \operatorname{do}(X))$ and a DAG, decide whether the query can be written purely in terms of observational quantities $P(\cdots)$ that I can estimate from logged data, and if so, write the formula. If I can, the effect is **identifiable** and I compute it. If I cannot, no amount of data of that kind will answer the question, and the honest move is to say so and either run an experiment or fall back to sensitivity analysis (chapter 4.5).

The reason identification is not automatic is the whole content of chapter 4.3: $P(Y \mid X)$ and $P(Y \mid \operatorname{do}(X))$ differ by exactly the non-causal paths between $X$ and $Y$. The $\operatorname{do}$ operator, applied to the graph, means "delete every arrow into $X$," because intervening on $X$ overrides whatever normally causes it. So identification is the art of accounting for the difference between the mutilated graph (arrows into $X$ removed) and the original, using only observable variables.

### The backdoor criterion

A **backdoor path** from $X$ to $Y$ is any path that starts with an arrow **into** $X$ (i.e. $X \leftarrow \cdots Y$). These are the non-causal paths, the ones that carry confounding, because they connect $X$ to $Y$ through $X$'s causes rather than its effects. The backdoor criterion says: a set $Z$ of variables suffices to identify the effect of $X$ on $Y$ if $Z$ blocks every backdoor path and contains no descendant of $X$. The second condition matters, because a descendant of $X$ might be a mediator or a collider, and conditioning on it either throws away part of the effect or opens a new spurious path.

```admonish derivation title="The backdoor adjustment formula"
Let $Z$ satisfy the backdoor criterion relative to $(X, Y)$ in DAG $G$: (i) $Z$ blocks every path from $X$ to $Y$ that has an arrow into $X$, and (ii) no node in $Z$ is a descendant of $X$. Then the interventional distribution is identified by

$$
P(Y \mid \operatorname{do}(X = x)) = \sum_{z} P(Y \mid X = x, Z = z)\, P(Z = z). \tag{4.1}
$$

Why this holds. Intervening with $\operatorname{do}(X=x)$ produces the mutilated graph $G_{\bar X}$ with all arrows into $X$ removed. In $G_{\bar X}$, the parents of $X$ no longer point at it, so $X$ is independent of any prior causes, and the truncated factorization (2.3) drops the $P(X \mid \operatorname{Pa}(X))$ factor:

$$
P_{\operatorname{do}(x)}(v \setminus x) = \prod_{i : X_i \neq X} P(X_i \mid \operatorname{Pa}(X_i)). \tag{4.2}
$$

Marginalize (4.2) over everything except $Y$ and the backdoor-blocking set $Z$. Because $Z$ blocks all backdoor paths and contains no descendant of $X$, conditioning on $Z$ makes $X$ and $Y$ share only the directed (causal) paths, so within a stratum $Z=z$ the interventional and observational conditionals coincide:

$$
P(Y \mid \operatorname{do}(x), Z=z) = P(Y \mid X=x, Z=z). \tag{4.3}
$$

The distribution of $Z$ is unaffected by intervening on a descendant-free set relative to $Z$ (no arrow into $X$ reaches $Z$ causally through $X$), so $P_{\operatorname{do}(x)}(Z=z) = P(Z=z)$. Averaging (4.3) over that unchanged $Z$ distribution gives (4.1). $\blacksquare$
```

Equation (4.1) is the workhorse. It says: to get the causal effect, estimate the outcome within each stratum of the confounders, then average those stratum-specific outcomes using the confounders' **marginal** distribution, not the distribution conditional on $X$. That reweighting is the entire difference between the adjusted and the naive comparison. The naive comparison, $P(Y \mid X=x)$, uses $P(Z \mid X=x)$ as the implicit weights and so lets the confounder distribution differ between the two treatment arms; the adjusted one forces a common weighting and thereby simulates the experiment I could not run.

### When there is no admissible adjustment set: the front door

Sometimes every candidate confounder is unmeasured, so no observable $Z$ blocks the backdoor paths and (4.1) is unavailable. The front-door criterion rescues a subset of these cases by exploiting a fully mediating variable. Suppose $X$ affects $Y$ only through a mediator $M$ ($X \to M \to Y$), the mediator has no unblocked backdoor to $X$, and every backdoor from $M$ to $Y$ is blocked by $X$. Then even with an unmeasured confounder $U$ sitting on $X$ and $Y$, the effect is identified through $M$.

```admonish derivation title="The front-door derivation"
Assume the front-door graph: an unobserved confounder $U \to X$ and $U \to Y$, a mediator with $X \to M \to Y$, and no direct $X \to Y$ edge and no $U \to M$ edge. The three front-door conditions hold: $M$ intercepts every directed path from $X$ to $Y$; there is no unblocked backdoor from $X$ to $M$; and $X$ blocks every backdoor from $M$ to $Y$.

Decompose the effect into two identifiable pieces. The effect of $X$ on $M$ has no backdoor (nothing confounds $X$ and $M$, since $U$ does not touch $M$), so it is identified directly:

$$
P(M=m \mid \operatorname{do}(x)) = P(M=m \mid X=x). \tag{4.4}
$$

The effect of $M$ on $Y$ is confounded by $U$ through $X$, but $X$ blocks that backdoor, so by backdoor adjustment on $X$:

$$
P(Y \mid \operatorname{do}(m)) = \sum_{x'} P(Y \mid M=m, X=x')\, P(X=x'). \tag{4.5}
$$

Chain them. Intervening on $X$ sets $M$'s distribution via (4.4), and each $M$ value then acts on $Y$ via (4.5), giving the front-door formula:

$$
P(Y \mid \operatorname{do}(x)) = \sum_{m} P(M=m \mid X=x) \sum_{x'} P(Y \mid M=m, X=x')\, P(X=x'). \tag{4.6}
$$

The inner sum over $x'$ deconfounds the $M \to Y$ step; the outer sum propagates $X$'s influence through the mediator. The effect is identified even though $U$ was never measured. $\blacksquare$
```

The front door is beautiful and, in eval practice, rare, because it needs a mediator that captures the **entire** effect and is itself unconfounded with the outcome except through the treatment. Most eval mediators (length, verbosity) fail the "no direct effect" condition, since the model plausibly affects the score through channels besides the mediator. I keep the front door in my kit for one reason: it is the formal statement that a clean, fully-mediating mechanism can rescue identification when confounders are hopeless, and occasionally a well-instrumented pipeline gives me exactly that.

### What is and is not identifiable in eval comparisons

The sober lesson is that most naive model-versus-model comparisons are **not** identified as run, because the two models were served, decoded, and judged under conditions that differ alongside the model. The good news is that the fix is usually cheap: I can measure the confounders (judge, decoding config, prompt template, difficulty mix) and adjust with (4.1), or I can design them away by holding them fixed. An ablation is the cleanest case of all, because randomized or held-fixed design makes the backdoor set empty and (4.1) collapses to the naive difference, which is then valid. The whole point of chapter 4.6's audit is to certify, path by path, that the thesis comparison has a real adjustment set and I have used it.

### The three assumptions hiding inside (4.1)

Backdoor adjustment looks like pure arithmetic, but three assumptions ride along underneath it, and every one of them is a place the estimate can quietly fail even when the code runs clean. Naming them is the difference between "I adjusted for difficulty" and "I adjusted for difficulty, and here is why the adjusted number means what I say it means."

The first is **no unmeasured confounding**, sometimes called conditional exchangeability: the set $Z$ I conditioned on really does block every backdoor path. This is not testable from the data; it is an assumption about the graph, and the graph is an assumption about the world. If a confounder exists that I did not measure and did not put in $Z$, (4.1) returns a number, and the number is wrong, and nothing in the output warns me. This is exactly why chapter 4.5's sensitivity analysis exists: since I cannot prove no unmeasured confounder exists, I quantify how strong one would have to be to matter.

The second is **positivity** (also called overlap): within every stratum $z$ that has nonzero weight, both treatments must actually occur, so that $P(X = x \mid Z = z) > 0$ for the $x$ values I compare. If model $B$ was never evaluated on any hard item, then the hard-item stratum contributes a term I cannot estimate, and (4.1) either silently drops it or extrapolates. In eval practice positivity fails whenever the confounder and the treatment are too tightly coupled, for instance if I only ever ran model $B$ on the easy split. The lab below deliberately keeps overlap positive (every difficulty level sees both models) precisely so the adjustment is estimable; a real pipeline has to check this, because a stratum with no support is a stratum where the causal question has no answer from this data.

The third is **consistency**: the potential outcome under the treatment I assigned equals the outcome I observed, which requires that "model $B$" names a single well-defined intervention and not a bundle of things that varied run to run. If "model $B$" secretly means "model $B$ at whatever temperature I happened to use that day," then the treatment is ill-defined and the effect I estimate is a blur over decoding settings. Pinning the intervention (fixed decoding, fixed template) is what makes consistency hold, and it is the same discipline the control-run protocol in chapter 4.5 enforces from the intervention side.

The do-calculus that Ness develops in [CAI] Part 3 generalizes both formulas above: it is a set of three rewrite rules that is provably **complete**, meaning that if an effect is identifiable from the graph at all, the calculus can derive its formula, and if the calculus cannot, the effect is genuinely non-identifiable from observational data. That completeness is why "not identifiable" is a real verdict and not just an admission that I did not try hard enough. When a tool like `dowhy` reports no valid adjustment set, it is telling me to go collect different data (run the experiment, measure the missing confounder) rather than to reshuffle the data I have.

```admonish read-along title="Go deeper: [CAI] Part 3"
Ness develops identification in [CAI] Part 3: the backdoor criterion, adjustment, and the do-calculus of which the backdoor and front-door formulas are special cases. Read Part 3 for the do-calculus rules if you want the general engine; the two derivations above are the two cases you will actually reach for. Ness also demonstrates the `dowhy` workflow (model, identify, estimate, refute), which is the tooling I lean on next and which automates exactly the step from graph to formula (4.1).
```

```admonish gotcha title="More covariates is not more adjustment"
The reflex to throw every logged column into the adjustment set is the fastest way to break (4.1). If $Z$ contains a collider, or a descendant of the treatment, or a variable on the causal path, the formula still runs and still returns a confident number, but that number is now biased in a direction the arithmetic will never reveal. The backdoor criterion is a filter that says which variables belong in $Z$ and, just as importantly, which must be kept out. A valid adjustment set is often small: for the thesis comparison it is the handful of confounders (difficulty mix, judge, decoding config, template) that sit on genuine backdoor paths, not the full width of the eval log. When in doubt, adjust for causes of the treatment and causes of the outcome, never for their common effects, and never for anything downstream of the treatment.
```

## Tooling

Doing identification by hand is fine for a five-node graph and error-prone past that, so the tool is a library that takes a graph and a query and returns the estimand. In the Python ecosystem that is `dowhy`, whose four-step ritual (model the graph, identify the estimand, estimate it, refute it) maps one-to-one onto this part: modeling is chapters 4.2 and 4.3, identification is this chapter, estimation is the arithmetic below, and refutation is chapter 4.5's placebo and sensitivity checks. `dowhy` will hand me back equation (4.1) with the adjustment set filled in, and it will refuse when no set exists, which is exactly the "not identifiable, go run an experiment" verdict I want it to be willing to give.

For the lab I keep dependencies light and implement the backdoor adjustment (4.1) directly in pandas, because seeing the stratify-then-reweight step in eight lines of code is more instructive than calling a solver, and it makes unmistakable that the adjusted estimate is just the naive estimate with the confounder distribution put back to its marginal. Once the arithmetic is in my hands, swapping in `dowhy` for larger graphs is a mechanical upgrade, not a leap.

## Lab

The lab compares two models on a task where a confounder (item difficulty) differs between the arms, because model $B$ happened to be evaluated on an easier difficulty mix. The naive comparison credits $B$ for the difficulty gap. The backdoor adjustment (4.1) strips it out. I plant a known true effect so I can check which estimate recovers it.

```python
# file: labs/p4-04-identification/adjusted_comparison.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Naive vs backdoor-adjusted model-vs-model comparison.

Graph:  difficulty D --> score S ;  D --> model-assignment (confounding) ;
        model X --> score S.  D is the backdoor-blocking set.

We plant a TRUE effect of model B over A and check that the adjusted
estimator (eq 4.1) recovers it while the naive difference does not.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260722)
TRUE_EFFECT = 0.08   # model B adds 8 points of accuracy at fixed difficulty

def simulate(n: int = 8000) -> pd.DataFrame:
    # difficulty in {easy, med, hard}; easy items score higher
    diff = RNG.choice(["easy", "med", "hard"], n, p=[0.34, 0.33, 0.33])
    base = pd.Series(diff).map({"easy": 0.70, "med": 0.50, "hard": 0.30}).to_numpy()
    # CONFOUNDING: model B was preferentially evaluated on EASY items
    p_B = pd.Series(diff).map({"easy": 0.75, "med": 0.50, "hard": 0.25}).to_numpy()
    model = np.where(RNG.random(n) < p_B, "B", "A")
    lift = np.where(model == "B", TRUE_EFFECT, 0.0)
    p_correct = np.clip(base + lift, 0, 1)
    score = (RNG.random(n) < p_correct).astype(float)
    return pd.DataFrame(dict(difficulty=diff, model=model, score=score))

def naive(df: pd.DataFrame) -> float:
    m = df.groupby("model")["score"].mean()
    return float(m["B"] - m["A"])

def backdoor_adjusted(df: pd.DataFrame, Z=("difficulty",)) -> float:
    # eq 4.1: E[Y|do(B)] - E[Y|do(A)]
    #       = sum_z [ E[Y|X=B,z] - E[Y|X=A,z] ] P(z)
    pz = df.groupby(list(Z)).size() / len(df)
    cond = df.groupby(["model", *Z])["score"].mean()
    est = 0.0
    for z, w in pz.items():
        z = z if isinstance(z, tuple) else (z,)
        key_b = ("B", *z); key_a = ("A", *z)
        if key_b in cond.index and key_a in cond.index:
            est += (cond[key_b] - cond[key_a]) * w
    return float(est)

def main() -> None:
    df = simulate()
    n_naive, adj = naive(df), backdoor_adjusted(df)
    result = {
        "true_effect": TRUE_EFFECT,
        "naive_diff_in_means": round(n_naive, 4),
        "backdoor_adjusted": round(adj, 4),
        "adjustment_set": ["difficulty"],
        "naive_bias": round(n_naive - TRUE_EFFECT, 4),
        "adjusted_bias": round(adj - TRUE_EFFECT, 4),
    }
    out = Path("labs/p4-04-identification")
    out.mkdir(parents=True, exist_ok=True)
    art = out / "adjusted_comparison.json"
    art.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    # show the confounding directly
    print("\nP(difficulty | model):")
    print(pd.crosstab(df.model, df.difficulty, normalize="index").round(3))
    print(f"\nartifact written: {art.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run labs/p4-04-identification/adjusted_comparison.py
```

The artifact is `labs/p4-04-identification/adjusted_comparison.json`, holding the planted true effect, the naive difference in means, the backdoor-adjusted effect, and the bias of each against the truth. The crosstab printed underneath is the confounding made visible: model $B$'s items are disproportionately easy.

**What you should see.** The naive difference in means comes out much larger than the planted 0.08, on the order of 0.20 or more, because $B$ was handed easier items and the naive comparison pays $B$ for the difficulty gap on top of its real 8-point edge. The backdoor-adjusted estimate lands close to 0.08, with a bias near zero (up to sampling noise on 8000 items), because (4.1) compares $A$ and $B$ within each difficulty stratum and then reweights by the overall difficulty mix rather than by each model's self-selected mix. The crosstab confirms the mechanism: $B$'s row is skewed toward "easy" and $A$'s toward "hard." The one-sentence takeaway to carry into the audit is that the naive and adjusted numbers disagree by an amount equal to the confounding, and only the adjusted number estimates the thing "$B$ is better" is supposed to mean. When I run the real thesis comparison, difficulty will not be the only backdoor variable; the judge, decoding config, and prompt template all need to be in $Z$ or held fixed, and chapter 4.6 is where I enumerate them and certify the set is complete.

```admonish substack-seed
Here is a number that is technically true and completely misleading: "Model B scores 20 points higher than Model A." It is true because I measured it. It is misleading because B was quietly evaluated on easier problems. The fix has a name (backdoor adjustment) and a formula that is less scary than it looks: instead of comparing the two models' raw averages, compare them within each difficulty level and then average those gaps using the overall difficulty mix. Do that and B's real edge shrinks from 20 points to 8, which is the honest number. This post derives the adjustment from one idea (intervening on a variable means deleting the arrows that normally point into it) and shows the whole correction in eight lines of pandas. The takeaway for anyone who ships eval tables: your naive comparison and your adjusted comparison differ by exactly the confounding you did not control, so compute both and let the gap tell you how much your setup was lying.
```
