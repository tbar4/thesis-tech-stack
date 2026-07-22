# Ablations, seeds, and experimental design

Chapter 7.6 measured a reasoning delta and gave it a confidence interval. That answers "did the metric move." It does not answer "why," and a thesis committee cares about why at least as much as whether. A delta could come from the correctness reward, or from the format reward riding along, or from the mere fact of training on anything at all, or from a lucky seed. To attribute the delta to a cause I have to vary one thing at a time and watch what the delta does, and to trust the attribution I have to run enough seeds that the between-config difference is not just seed noise. On a datacenter you brute-force this with a full grid and dozens of seeds. On one 16GB card you cannot, so experimental design stops being a formality and becomes a budgeting problem: every run costs real wall-clock time on the baseline machine, the budget is finite, and the design has to spend it where it buys the most inferential power.

This chapter turns that constraint into an actual plan. The output is the thesis experiment plan: a pre-registered, machine-readable document that fixes the hypotheses, the ablation axes, the seed count derived from a power analysis, and the run matrix costed against a single-GPU time budget, before a single run happens. Pre-registration is not bureaucracy here; it is the thing that lets me claim the delta is real rather than the survivor of a hundred quiet forks.

## Theory

### Which ablations actually matter

An ablation removes or varies one component of the system and measures the effect on the outcome. The temptation is to ablate everything; the discipline is to ablate the few components whose effect is both plausibly large and scientifically contested. For the RLVR loop built in this part, three axes carry almost all the interpretive weight.

The reward components come first, because chapter 7.3 built the reward as a sum of a correctness term and a format term, and chapter 7.4 spent a whole chapter on how that sum gets hacked. The question a committee will ask is whether the delta is driven by the model actually solving problems (correctness) or by it learning to satisfy the format checker (format), and the only way to answer it is to run the loop with correctness-only, with format-only, and with the full reward, and compare. Format-only is close to a placebo for reasoning: if it produces nearly the same delta as the full reward, the "reasoning" delta was mostly presentation. This axis directly protects the thesis claim.

The group size $K$ comes second, because chapter 7.2 showed it sets both the variance of the group-relative advantage (equation 5.2's $\frac{K-1}{K}p(1-p)$) and the per-step generation cost, which on 16GB is the dominant cost. Larger $K$ gives lower-variance advantage estimates and a wider usable difficulty band, at linear cost in generations. Whether $K=8$ is enough or $K=16$ is worth the doubled generation time is an empirical question with a real budget consequence, so it earns an axis.

The KL coefficient $\beta$ comes third, because it controls how far the policy is allowed to drift from the reference model, and that trade sits right on top of the reward-hacking story. A small $\beta$ lets the policy move freely and chase reward, including hacked reward; a large $\beta$ anchors it near the reference and can suppress genuine learning along with the hacking. The KL term in the GRPO objective is

$$
\mathcal{L}(\theta) = \mathbb{E}\!\left[\frac{\pi_\theta(a\mid x)}{\pi_{\text{old}}(a\mid x)} A - \beta\, D_{\text{KL}}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)\right],
\tag{7.1}
$$

and $\beta$ is the knob that decides whether "stay close to the reference" or "maximize reward" wins when they conflict. Sweeping $\beta$ across, say, zero, a small value, and a large value maps out that trade-off and is exactly the kind of curve a committee reads as evidence of understanding rather than luck.

There is a fourth axis I do not put in the primary matrix but do run as a control: the random-reward placebo from chapter 4.5, where the reward is replaced with noise. It is not an ablation of a component, it is the negative control that says "training on this loop with no real signal produces no delta," and it belongs to the causal argument of 4.5. I reference it here because a good run matrix budgets for its controls, and the placebo is the most important control the whole thesis has.

### Seeds are a sample size, and power analysis sets it

A single training run at a single seed is one draw from a noisy process. GRPO on one GPU is stochastic in its rollouts, its data order, and its optimization, so two seeds of the identical config land at two different deltas. If I run config A once and config B once and A's delta is larger, I have learned almost nothing, because the gap could be pure seed noise. To compare two configs I need several seeds of each, and "several" is not a vibe, it is the output of a power analysis.

Frame the comparison as a two-sample test: config A produces deltas with mean $\mu_A$ and config B with mean $\mu_B$, each with seed-to-seed standard deviation $\sigma$, and I want to detect a true difference $\Delta = \mu_A - \mu_B$ of a size worth caring about. The number of seeds per config to achieve power $1-\beta$ at significance $\alpha$ is the standard two-sample formula

$$
n_{\text{seeds}} \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\,\sigma^2}{\Delta^2}.
\tag{7.2}
$$

```admonish derivation
Equation (7.2) is worth reading for what it says about a one-GPU budget rather than as a formula to plug. The seed count scales as $\sigma^2 / \Delta^2$: it grows with the square of the noise and shrinks with the square of the effect you want to detect. Two consequences follow immediately. First, halving the smallest effect you care about quadruples the seeds you need, so declaring in advance that you only care about deltas above some threshold is not laziness, it is what makes the experiment affordable. Second, anything that reduces $\sigma$ pays off quadratically, which is why the paired design of chapter 7.6 matters here too: measuring every config's delta on the *same* frozen suite with the *same* seeds for the eval sampling removes a chunk of $\sigma$ before the power analysis even starts. With standard choices $z_{1-\alpha/2} = 1.96$ ($\alpha = 0.05$) and $z_{1-\beta} = 0.84$ (power $0.8$), the leading constant $2(1.96+0.84)^2 \approx 15.7$, so $n_{\text{seeds}} \approx 15.7\,(\sigma/\Delta)^2$. If the seed-to-seed SD is about equal to the effect you want to detect, that is roughly 16 seeds per config, which on one GPU is often unaffordable, and the honest response is to widen $\Delta$ (only claim large effects), shrink $\sigma$ (better pairing, more eval items per run), or accept lower power and say so out loud. The $\sigma$ in (7.2) is a measured quantity: run the center config at several seeds and record the SD of its delta (measured on the baseline machine — record value, date, driver), then let the formula tell you the seed count rather than guessing it.
```

The practical loop is: run a small pilot (the center config at a handful of seeds), measure $\sigma$, pick the smallest $\Delta$ worth claiming, compute $n_{\text{seeds}}$, and if the resulting run matrix does not fit the budget, revise $\Delta$ upward and re-derive, all before running the real matrix. That pilot-then-power sequence is itself pre-registered, so the seed count is a documented consequence of a measured $\sigma$, not a number chosen after seeing which configs looked good.

### The run matrix under a single-GPU time budget

Now the budgeting. Let each training run cost $T$ minutes on the baseline machine (a measured quantity dominated, per chapter 7.2, by the $N \cdot E \cdot K$ generations), and let the total compute budget be $B$ hours. A run matrix of $C$ configurations at $n_{\text{seeds}}$ seeds each, plus the controls, costs

$$
T_{\text{total}} = \big(C \cdot n_{\text{seeds}} + n_{\text{controls}}\big)\cdot T \;\le\; 60\,B \ \text{minutes}.
\tag{7.3}
$$

This inequality is the design constraint, and it forces a shape on the experiment. A full factorial over three reward settings, three group sizes, and three KL values is $3^3 = 27$ configs, times a seed count from (7.2), times $T$, and it will not fit. So I do not run a full factorial. I run a center config (my best guess at a good setting) and vary one axis at a time around it, a one-factor-at-a-time (OFAT) design. OFAT cannot see interactions between axes, which is its real limitation, but it costs $1 + \sum_a (\text{levels}_a - 1)$ configs instead of the product, and on one GPU that difference is the difference between a thesis that runs and one that does not. If the budget allows, I add back the one or two specific interaction cells I most suspect (say, small $\beta$ with format-only reward, the most reward-hackable corner), a fractional design rather than a full grid.

A worked skeleton, with the arithmetic explicit and the numbers marked as the design placeholders they are:

| Axis | Levels | Configs added |
|---|---|---|
| Center config | 1 | 1 |
| Reward component | correctness-only, format-only (full is center) | +2 |
| Group size $K$ | one below, one above center | +2 |
| KL coefficient $\beta$ | one below, one above center | +2 |
| Controls | random-reward placebo (chapter 4.5) | +1 (control) |

That is $C = 7$ configs plus 1 control. At $n_{\text{seeds}}$ seeds per config the run count is $7\,n_{\text{seeds}} + n_{\text{controls}}$, and equation (7.3) turns that into hours once $T$ and $n_{\text{seeds}}$ are filled from measurements. The plan generator below expands exactly this table into a costed matrix so the fit against the budget is checked by arithmetic, not optimism.

### Pre-registration habits

Everything above is worthless if I decide what counts as success after seeing the results. The garden of forking paths is real: with enough metrics, enough subgroups, and enough freedom to choose the analysis after the fact, a delta of zero can be dressed as significant. The defense is pre-registration, and it is a habit, not a ceremony. Before running the matrix I write down, and commit, the following: the primary hypothesis and the primary metric (the mean paired delta on the frozen suite, from chapter 7.6, and nothing else gets to be primary after the fact); the exact ablation axes and their levels; the seed count and the $\sigma$-and-$\Delta$ reasoning that produced it; the analysis (paired bootstrap CI and permutation test, multiple-comparison correction across the ablation arms); the stopping rule (all runs in the matrix complete, no peeking-and-stopping); and the exclusion criteria (what makes a run void, such as a diverged loss or an OOM restart, decided before any run is excluded).

Two of those deserve emphasis on a single GPU. Multiple comparisons: the moment I compare seven configs against the center I am running seven tests, and the family-wise error rate inflates, so the plan pre-commits to a correction (Holm or Benjamini-Hochberg over the arms, from the chapter 3.7 module) rather than reading seven raw $p$-values as if each stood alone. And the stopping rule: it is tempting, when runs are slow and expensive, to stop the matrix early because the trend "looks clear," but optional stopping is one of the most reliable ways to manufacture significance, so the rule is that the matrix runs to completion or the result is reported as incomplete. Committing these before the fact is what converts "I found a significant delta" from a hopeful sentence into a credible one.

## Tooling

The tool is a plan compiler. It reads a human-written pre-registration file (the hypotheses and axes and budget), expands the OFAT matrix, applies the power analysis to fill the seed count from a measured $\sigma$, costs the matrix against the time budget via equation (7.3), and refuses to emit a runnable matrix if it does not fit. The pre-registration file is the source of truth and it is committed before any run.

```bash title="shell: create the experiment-plan project"
uv init expplan
cd expplan
uv add "pydantic>=2.7" "pyyaml>=6.0" "scipy>=1.13"
```

### The pre-registration document

This YAML is the artifact a committee reads first. It is filled in with design choices and placeholder measurements, and the placeholders are stamped so an unmeasured $\sigma$ or $T$ cannot masquerade as real.

```yaml title="expplan/experiment_plan.yaml"
# THESIS EXPERIMENT PLAN (pre-registration).
# Commit this file BEFORE running any config. Measured slots are stamped
# and the compiler refuses to expand a matrix with unfilled measurements.

title: "Reasoning delta of RLVR on the frozen thesis suite"
machine: "MSI Aegis R2 (Core Ultra 9 285, RTX 5080 16GB, Ubuntu 24.04)"
suite: "thesis-suite v1.0 (chapter 3.9)"
primary_metric: "mean paired delta in per-sample solve rate (chapter 7.6)"
primary_hypothesis: >
  The full verifiable reward (correctness + format) produces a positive
  mean paired delta on the frozen suite that exceeds the format-only
  ablation and the random-reward placebo.

analysis:
  ci: "paired bootstrap, 95% (evalstats, chapter 3.7)"
  test: "sign-flip permutation, two-sided (evalstats)"
  multiple_comparison_correction: "holm"     # across ablation arms
  stopping_rule: "run full matrix to completion; no optional stopping"
  exclusion_criteria:
    - "loss diverges (NaN/inf) -> void, re-seed once, then report"
    - "OOM restart mid-run -> void the run, do not silently resume"

power:
  alpha: 0.05
  power: 0.80
  # smallest delta-difference worth claiming, a DESIGN choice:
  min_effect_delta: 0.03
  # seed-to-seed SD of the delta, MEASURED from the pilot:
  seed_sigma: "record: measured on the baseline machine, date, driver"

budget:
  total_hours: 24                    # single-GPU compute budget, a design choice
  minutes_per_run: "record: T measured on the baseline machine, date, driver"

center_config:
  reward: "full"                     # correctness + format
  group_size_K: 8
  kl_beta: 0.02

ablations:
  reward: ["correctness_only", "format_only"]   # full is the center
  group_size_K: [4, 16]                          # 8 is the center
  kl_beta: [0.0, 0.1]                            # 0.02 is the center

controls:
  - name: "random_reward_placebo"    # chapter 4.5 negative control
    reward: "random"
```

### The plan compiler

```python title="expplan/compile_plan.py"
"""Compile the pre-registration YAML into a costed OFAT run matrix.

Steps:
  1. load and validate the plan (measured slots must be filled)
  2. power analysis -> seeds per config (eq. 7.2)
  3. expand OFAT matrix around the center config
  4. cost against the time budget (eq. 7.3); refuse if it does not fit
Emits run_matrix.csv (the runnable plan) only if the budget holds.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml
from scipy.stats import norm


def seeds_per_config(sigma: float, delta: float, alpha: float, power: float) -> int:
    """Two-sample seed count, eq. 7.2, rounded up."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n = 2.0 * (z ** 2) * (sigma ** 2) / (delta ** 2)
    return max(2, math.ceil(n))


def _require_measured(value, label: str) -> float:
    if isinstance(value, str) and "record" in value:
        raise ValueError(f"unfilled measurement: {label} is still stamped '{value}'")
    return float(value)


def expand_ofat(plan: dict) -> list[dict]:
    center = plan["center_config"]
    configs = [{"name": "center", **center}]
    for axis, levels in plan["ablations"].items():
        for lvl in levels:
            cfg = dict(center)
            cfg[axis] = lvl
            configs.append({"name": f"{axis}={lvl}", **cfg})
    return configs


def main() -> None:
    plan = yaml.safe_load(Path("experiment_plan.yaml").read_text())

    sigma = _require_measured(plan["power"]["seed_sigma"], "power.seed_sigma")
    T = _require_measured(plan["budget"]["minutes_per_run"], "budget.minutes_per_run")

    n_seeds = seeds_per_config(
        sigma=sigma,
        delta=plan["power"]["min_effect_delta"],
        alpha=plan["power"]["alpha"],
        power=plan["power"]["power"],
    )

    configs = expand_ofat(plan)
    n_controls = len(plan.get("controls", []))
    total_runs = len(configs) * n_seeds + n_controls
    total_minutes = total_runs * T
    budget_minutes = 60 * plan["budget"]["total_hours"]

    print(f"configs={len(configs)}  seeds/config={n_seeds}  "
          f"controls={n_controls}  total_runs={total_runs}")
    print(f"cost={total_minutes:.0f} min  budget={budget_minutes:.0f} min")

    if total_minutes > budget_minutes:
        raise SystemExit(
            f"OVER BUDGET by {total_minutes - budget_minutes:.0f} min. "
            f"Raise min_effect_delta, cut an axis, or raise total_hours -- "
            f"and re-commit the plan before running anything."
        )

    rows = []
    run_id = 0
    for cfg in configs:
        for s in range(n_seeds):
            rows.append({"run_id": run_id, "seed": s, **cfg})
            run_id += 1
    for ctrl in plan.get("controls", []):
        rows.append({"run_id": run_id, "seed": 0, **ctrl})
        run_id += 1

    with Path("run_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote run_matrix.csv with {len(rows)} runs")


if __name__ == "__main__":
    main()
```

## Lab: produce the thesis experiment plan

The lab is short because the thinking is in the plan, not the code. Fill the two measured slots from a pilot run, then compile.

```bash title="shell: pilot, then compile the plan"
# 1. run the center config at a few seeds (chapter 7.2/7.3 trainer),
#    evaluate each with chapter 7.6, and record:
#      - the seed-to-seed SD of the delta  -> power.seed_sigma
#      - the wall-clock minutes per run     -> budget.minutes_per_run
#    (measured on the baseline machine -- record value, date, driver)

# 2. edit experiment_plan.yaml to replace both 'record: ...' slots

# 3. compile the costed matrix (refuses if over budget)
uv run python compile_plan.py
```

### What you should see

Two committed artifacts. The first is `experiment_plan.yaml` itself, the pre-registration, committed before any real run and carrying its measured slots stamped until the pilot fills them. The second, produced by the compiler, is `run_matrix.csv`: one row per run, each with a run id, a seed, and the full config, expanding the seven OFAT configs across the power-derived seed count plus the random-reward control. The console prints the arithmetic of equation (7.3) in the open: how many configs, how many seeds each, the total run count, the total minutes, and the budget in minutes. If the matrix fits, you get the CSV. If it does not, the compiler refuses and tells you the three honest levers, raise the minimum effect you will claim, cut an axis, or buy more hours, and it makes you re-commit the plan before anything runs. That refusal is the point: the design is forced to be affordable and pre-registered before a single GPU-minute is spent, so when chapter 7.6's delta comes back significant, it is significant under a plan that was fixed in advance rather than fitted to the answer.

The seed count is the number to stare at. If the power analysis demands more seeds than the budget can hold, that is not a bug to route around, it is the experiment telling you the effect you are chasing is too small to detect on one GPU at the noise you have, and the correct response is to claim a larger effect or reduce the noise, both of which the plan makes explicit.

```admonish gotcha
Reusing eval seeds across configs is good (it shrinks $\sigma$ via pairing) but reusing *training* seeds across configs is a trap: if config A and config B share a training seed, their deltas are correlated in a way the two-sample power analysis of equation (7.2) assumes away, and the correction across arms gets muddled. Keep training seeds distinct per run (that is what the `seed` column in the matrix is for) and keep the eval sampling seeds shared across configs. Mixing the two up is a subtle way to both understate your uncertainty and confuse the multiple-comparison correction.
```

```admonish substack-seed
The most useful idea from experimental design for anyone with a small budget is that a power analysis run backwards is a bullshit detector. Instead of asking "how many samples do I need," fix the samples you can actually afford, fix the noise you actually measured, and solve for the smallest effect you could possibly detect. If that smallest detectable effect is larger than any effect worth caring about, the experiment is dead before it starts and no amount of running it will help. On one GPU this happens constantly, and it is liberating rather than depressing: it tells you, in advance and in one line of arithmetic, which questions your hardware can answer and which ones you should stop pretending to ask. The hook: you do not need a bigger computer, you need to know before you start whether the one you have can see the thing you are looking for.
```
