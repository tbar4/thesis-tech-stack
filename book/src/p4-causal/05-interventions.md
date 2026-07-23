# Interventions on models: is the reasoning delta causal?

**Goal.** Treat a training run as an intervention and ask, honestly, whether the pre/post score change it produced is the causal effect of training or an artifact of everything that moved alongside it. Then design the control runs that let me answer.

**Covers.** Training as a do-operation; pre/post designs as interrupted time series; placebo controls (random-reward runs) and negative controls (held-out unrelated tasks); sensitivity analysis for unmeasured confounding.

## Theory

### Training is a do, and pre/post is the weakest design that could work

When I run GRPO on the model, I am performing $\operatorname{do}(\text{train})$: I reach in and change the weights on purpose. That is genuinely rung two, and it is the good news, because unlike a leaderboard comparison I actually control the intervention. The bad news is that my design for measuring its effect, run the eval before, run the eval after, subtract, is the weakest design in the causal toolkit. It is a single-unit, pre/post, before-and-after comparison, and its Achilles heel is that anything else that changed between "before" and "after" is perfectly confounded with the training. Training is not the only intervention the thesis makes on the policy: giving the model retrieval (chapter 8.1) or tools (chapter 8.2) at inference is a $\operatorname{do}$ on its access, and chapter 8.3 uses exactly this machinery to ask whether each augmentation's effect is a genuine reasoning gain or something the augmentation supplied directly.

The list of things that plausibly changed between the two evals is long and boring and lethal: the vLLM version, the decoding seed, the KV-cache settings, the prompt template, the judge revision, the GPU thermal state, even the phase of the sampler's RNG. Any of these that touches the score sits on a backdoor path from "train" to "delta," and the pre/post difference sums the training effect with all of them. Chapter 7.6 computes the delta with careful paired statistics; those statistics tighten the confidence interval around the difference, but a tight interval around a confounded quantity is precision about the wrong number. Statistics answers "is the delta distinguishable from noise." Causal design answers "is the delta the effect of training." I need both, and this chapter is the second.

### Interrupted time series: give the before-and-after a shape

The upgrade from a two-point pre/post to something defensible is to treat the run as an **interrupted time series**. Instead of one measurement before and one after, I take the eval at several checkpoints along training and model the trajectory, so the intervention's effect is a change in level or slope of a curve rather than a single jump I have to take on faith. If the score was already drifting upward before the reward signal kicked in (say, from KL-regularized movement toward the reference), a time series shows that drift and lets me attribute only the extra change to the reward. A two-point design cannot tell drift from effect; a trajectory can.

Formally I fit the score $S_t$ at training step $t$ as a level plus a trend plus an intervention term that switches on at the step $t^\*$ where the causal reward begins to bite:

$$
S_t = \beta_0 + \beta_1 t + \beta_2 \mathbb{1}[t \ge t^\*] + \beta_3 (t - t^\*)\mathbb{1}[t \ge t^\*] + \varepsilon_t. \tag{5.1}
$$

Here $\beta_2$ is the immediate level shift at the intervention and $\beta_3$ is the change in slope after it, and $\beta_1$ is the drift a naive pre/post design would misattribute to training. State the honest version upfront, though: a GRPO run has the verifiable reward active from step 0, so there is no reward-free segment within the run, and the within-run slope is *not* a clean drift estimate. The real "interruption" contrast is not pre-onset versus post-onset inside one run; it is the actual run versus the random-reward *placebo* run of the next section, which supplies the counterfactual drift. So (5.1)'s level and slope terms are read against the placebo trajectory, not within-run pre/post, and reading the effect off $\beta_2$ and $\beta_3$ relative to that placebo is how I stop crediting training for motion that was already underway.

One caveat keeps this honest for my setup: a GRPO run has the verifiable reward active from step 0, so there is no reward-free segment within the run from which to read a clean pre-onset $\beta_1$. Here $t^\*$ is not a switch I flip mid-run; it marks the step where I expect the reward's effect to emerge, and $\beta_1$ is estimated from the early trajectory rather than from a genuinely pre-reward stretch. The counterfactual drift I actually compare the effect against, the trajectory of "the training machinery running with no task signal," comes from the placebo run of the next section, which is why the interrupted time series and the placebo are two halves of one argument rather than independent controls.

### Placebo controls: the random-reward run

The single most convincing control I can run on one GPU is a **placebo**: an identical training run in every respect except that the reward carries no task signal. I replace the verifiable reward with a random reward (or a constant, or a shuffled reward that breaks the response-to-score correspondence), keep the KL penalty, the group size, the learning rate, the number of steps, the decoding config, and the eval harness all identical, and I measure the delta. If the placebo run moves the eval as much as the real run, then whatever I am seeing is not the task reward; it is the optimization machinery, the KL drift, or the eval's own noise. The placebo isolates the reward as the active ingredient by holding constant everything that is not the reward.

This is the causal analogue of a sugar pill, and it does something no statistic can: it gives me a null distribution generated by my actual pipeline, with all its real confounders present, under the hypothesis that the reward does nothing. The real run's delta is only credible as a training effect to the extent that it exceeds the placebo run's delta. A random-reward run that moves the needle is not a nuisance; it is the most useful negative result I can get, because it tells me my "improvement" was mechanism, not signal.

### Negative controls: the held-out unrelated task

A placebo controls the intervention; a **negative control** controls the outcome. I pick a task the training should not affect (an unrelated capability the reward never touched: say, a trivia or formatting task orthogonal to the reasoning skill I trained) and I evaluate on it before and after. If the unrelated task moves as much as the target task, then something global shifted (the judge drifted, the decoding changed, the harness updated), and my target-task delta is contaminated by that global shift rather than reflecting a specific reasoning gain. A negative-control outcome that stays flat while the target moves is strong evidence that the effect is specific to what I trained, which is most of what "the model reasons better" is supposed to mean.

Placebo and negative controls attack the two halves of the confounding story from opposite ends. The placebo asks "if the reward were meaningless, would the score still move?" The negative control asks "if the score moved, did unrelated scores move too?" Passing both is the closest I get, on one GPU without a randomized multi-model experiment, to certifying that the delta is the causal effect of the training I intended.

### Why the single-GPU constraint makes controls cheap, not expensive

It would be easy to read this chapter as a counsel of perfection I cannot afford on one 16GB card, but the economics run the other way. The placebo run reuses the exact same code path, the exact same model, and the exact same eval harness as the main run; the only change is the reward function, which is a few lines. So a placebo costs one more training run of identical size, not a new experimental apparatus, and on a single GPU where I am already time-multiplexing serving and training (chapter 0.4), one more run is a scheduling entry, not a hardware ask. The negative-control task is even cheaper: it is one more eval pass over a small held-out suite, seconds of inference, no training at all. What the single-GPU regime actually forces is discipline about *which* controls earn their slot, because I cannot run twenty of them, and that scarcity is healthy: it makes me pre-commit to the two or three controls that would most change my mind, which is exactly the pre-registration habit chapter 7.7 formalizes. Expensive experiments tempt people to skip controls; cheap ones let me afford the controls that matter, and the reward-swap placebo is the cheapest high-value control in the whole book.

### A note on reading many controls at once

Once I run a placebo, a negative control, and an interrupted-time-series with several checkpoints, I am looking at several comparisons, and several comparisons invite the multiplicity trap: if I run enough contrasts, one will look significant by luck. I handle this the way chapter 3.7 handles it for evals, by deciding in advance which contrasts constitute the claim and treating the rest as descriptive, rather than by hunting through all of them for the one that clears a threshold. The controls are not a fishing expedition for a positive result; they are a fixed set of pre-registered attacks, and the claim survives only if it survives all of them, not if it survives the best-looking one. Framing the controls as a conjunction ("the delta beats the placebo AND the negative control stayed flat AND the trajectory showed a real shift") rather than a disjunction is what keeps multiplicity from creeping back in through the side door, and it is why the protocol below states the decision rule as an explicit AND.

```admonish thesis-thread title="Hardening 7.6's reasoning-delta claim"
Chapter 7.6 will report a reasoning delta on the Space Domain Awareness (SDA) verifiable-reasoning suite: post-training pass@1 minus pre-training pass@1, with paired bootstrap intervals over shared items. That is a rung-one difference with a confidence interval. To promote it to the rung-two claim "GRPO training on the verifiable reward improved SDA reasoning," the thread now commits to three controls, all run on the baseline machine with identical config except the named change (record every value, date, and driver):

1. **Placebo (random-reward) run.** Same recipe, reward replaced by a shuffled reward that destroys the response-to-correctness mapping. Claim survives only if the real delta exceeds the placebo delta beyond their overlapping intervals.
2. **Negative-control task.** A held-out formatting/trivia suite the reward never touched, evaluated pre and post. Claim survives only if this stays flat while SDA moves.
3. **Interrupted time series.** Eval at several checkpoints so the SDA gain is read as a level/slope change (eq 5.1), not a two-point jump, separating reward effect from pre-existing KL drift.

The delta from 7.6 is the numerator. These three controls are the denominator that makes it an effect rather than a coincidence. Chapter 4.6 folds all three into the audit, and chapter 7.7 schedules them into the run matrix.
```

### Sensitivity analysis: how strong would a hidden confounder have to be?

Even after placebo and negative controls, some unmeasured confounder might remain. I cannot rule it out, but I can quantify how strong it would have to be to explain away my result, and that quantity is itself a defensible, committee-facing number. The cleanest single summary is the **E-value**: the minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need to have with both the treatment and the outcome to fully account for the observed effect. A large E-value means only an implausibly strong hidden confounder could overturn the finding; a small one means a mild unmeasured cause could.

```admonish gotcha title="A placebo that is not identical is not a placebo"
The whole power of the random-reward run comes from its being byte-for-byte identical to the main run except for the reward. If I accidentally change the number of steps, the learning rate, the group size, or the decoding seed alongside swapping the reward, the placebo no longer isolates the reward, and a difference between the two runs could be any of the things I let vary. The most common self-inflicted version is running the placebo for fewer steps "to save time," which quietly makes it a weaker intervention and flatters the main run by comparison. The discipline is to change exactly one thing, verify config equality in the logs the same way the main pipeline does (chapter 4.6's T3), and treat any drift in the invariants as a failed control that must be rerun, not explained away.
```

```admonish read-along title="Go deeper: [CAI] Part 3 on robustness and refutation"
Ness frames placebo tests, negative controls, and sensitivity analysis in [CAI] Part 3 as the "refute" step of the model-identify-estimate-refute loop: having identified and estimated an effect, you attack it. Read that section for the philosophy (an estimate you have not tried to break is not yet evidence) and for the do-calculus view of why a random-reward placebo is a valid null. The E-value construction below follows VanderWeele and Ding; Ness's treatment of unmeasured confounding gives the graphical intuition for what the E-value is bounding.
```

## Tooling

The tooling is a control-run protocol: a machine-readable specification of the runs I must launch, what each one holds fixed, what it varies, and the decision rule that says whether the main claim survives. Writing it as a file rather than as prose does two things. It forces me to pre-commit to the comparisons before I see the numbers, which is the pre-registration habit chapter 7.7 formalizes, and it becomes an executable checklist the experiment plan can consume directly. The protocol is the artifact of this chapter and the input to 7.7.

Alongside the protocol I need one computation: the E-value, so that the sensitivity claim is a number and not a hand-wave. Given an observed effect expressed as a risk ratio $\mathrm{RR}$ (with $\mathrm{RR} \ge 1$; invert if below), the E-value is

$$
\text{E-value} = \mathrm{RR} + \sqrt{\mathrm{RR}\,(\mathrm{RR} - 1)}. \tag{5.2}
$$

For an effect on a difference scale (like a pass-rate delta) I first convert to an approximate risk ratio, report the E-value for the point estimate and, more importantly, for the confidence-interval limit nearest the null, because that is the one an adversary attacks. Equation (5.2) is small enough to keep in my head and load-bearing enough to end an argument.

## Lab

The lab does two things: it emits the control-run protocol as a versioned YAML the experiment plan will adopt, and it computes E-values for a plausible reasoning delta so the sensitivity claim is concrete. No GPU is required here; the protocol describes the GPU runs, and the E-value is arithmetic. The actual pre/post numbers are placeholders to be filled from real runs on the baseline machine.

```python
# file: labs/p4-05-interventions/build_control_protocol.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pyyaml>=6.0"]
# ///
"""Emit the control-run protocol (adopted by chapter 7.7) and compute
E-values for the reasoning-delta claim.

The pre/post rates below are PLACEHOLDERS. Replace with measured values from
the baseline machine (record value, date, driver) before citing.
"""
from __future__ import annotations
import math
from pathlib import Path
import yaml

def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value for a risk ratio (rr >= 1)."""
    if rr < 1:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))

def rate_to_rr(p_post: float, p_pre: float) -> float:
    return p_post / p_pre

PROTOCOL = {
    "protocol": "thesis control-run protocol",
    "version": "1.0",
    "consumed_by": ["p7-loop/07-ablations (experiment plan)",
                    "p4-causal/06-causal-audit (audit v1.0)"],
    "estimand": "causal effect of GRPO verifiable-reward training on SDA pass@1",
    "invariants_held_fixed_across_all_runs": [
        "vLLM version + flags", "decoding seed + temperature + top_p",
        "prompt template", "judge revision + seed", "eval item set (frozen v1.0)",
        "KL coefficient", "group size", "learning rate", "step count",
    ],
    "runs": [
        {"id": "main", "reward": "verifiable SDA reward",
         "purpose": "estimate the reasoning delta"},
        {"id": "placebo_random_reward", "reward": "shuffled reward (breaks response->score map)",
         "purpose": "null generated by the real pipeline; main must exceed this"},
        {"id": "placebo_constant_reward", "reward": "constant reward",
         "purpose": "isolate KL/optimizer drift from task signal"},
    ],
    "outcomes": [
        {"id": "target", "task": "SDA verifiable-reasoning suite v1.0",
         "expect": "moves under main, flat-ish under placebo"},
        {"id": "negative_control", "task": "held-out formatting/trivia suite",
         "expect": "flat under all runs; if it moves, a global shift is confounding"},
    ],
    "design": {
        "type": "interrupted time series",
        "checkpoints": "eval at >=4 training steps to separate drift from effect (eq 5.1)",
        "statistics": "paired bootstrap over shared items (see 7.6)",
    },
    "decision_rule": (
        "Promote the delta to a causal claim only if: (a) main delta exceeds "
        "both placebo deltas beyond overlapping bootstrap intervals; (b) the "
        "negative-control task stays within noise; (c) the interrupted-time-"
        "series level/slope term is significant beyond pre-intervention drift; "
        "(d) the E-value at the near-null CI limit exceeds a pre-set threshold."
    ),
    "sensitivity": {"method": "E-value (VanderWeele-Ding, eq 5.2)",
                    "threshold_note": "set threshold before seeing results"},
}

def main() -> None:
    out = Path("labs/p4-05-interventions")
    out.mkdir(parents=True, exist_ok=True)
    art = out / "control_run_protocol.yaml"
    art.write_text(yaml.safe_dump(PROTOCOL, sort_keys=False))

    # E-value illustration with PLACEHOLDER rates
    p_pre, p_post = 0.52, 0.61          # placeholder pre/post pass@1
    p_ci_low = 0.55                     # placeholder near-null CI limit for post
    rr_point = rate_to_rr(p_post, p_pre)
    rr_ci = rate_to_rr(p_ci_low, p_pre)
    print("PLACEHOLDER rates (replace with baseline-machine measurements):")
    print(f"  pre={p_pre}  post={p_post}  post_CI_low={p_ci_low}")
    print(f"  RR(point) = {rr_point:.3f}  E-value = {e_value(rr_point):.2f}")
    print(f"  RR(CI)    = {rr_ci:.3f}  E-value = {e_value(rr_ci):.2f}"
          "   <-- the one an adversary attacks")
    print(f"\nprotocol written: {art.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run labs/p4-05-interventions/build_control_protocol.py
```

The artifact is `labs/p4-05-interventions/control_run_protocol.yaml`: the frozen list of runs, invariants, outcomes, the interrupted-time-series design, the decision rule, and the sensitivity method. That file is what chapter 7.7 reads to build its run matrix and what chapter 4.6 cites as the intervention half of the audit.

**What you should see.** The protocol YAML on disk, and two E-values on stdout. With the placeholder rates, the point-estimate risk ratio is about 1.17 (0.61 over 0.52) giving an E-value near 1.6, and the CI-limit risk ratio near 1.06 gives an E-value near 1.3. Read those numbers as: an unmeasured confounder would need associations of roughly 1.6-fold (point) or 1.3-fold (worst case) with both training and score to explain the whole effect away. Whether that is "reassuringly large" or "worryingly small" is a judgment I make against a threshold I set **before** seeing the data, which is why the protocol records the threshold as a pre-commitment rather than a post-hoc rationalization. The real reasoning delta and its interval come from the baseline machine in chapter 7.6; when I have them, I re-run this to get the true E-values and record them with date and driver. The durable lesson is that the delta alone is never the claim: the claim is the delta plus a placebo it beat, a negative control that stayed flat, a trajectory that showed a real level shift, and an E-value large enough that no plausible hidden confounder rescues the null.

```admonish substack-seed
Your model's score went up after training. Congratulations, you have measured a difference. You have not yet measured an effect, and the gap between those two words is where most "our method improves reasoning" claims quietly fail. The cure is three controls you can run on a single GPU. Run the exact same training with the reward shuffled into noise (a placebo): if the score still climbs, your gain was the optimizer, not the reward. Evaluate on a task you never trained on (a negative control): if it climbs too, something global drifted and your "improvement" is contamination. And take the eval at several checkpoints instead of just before-and-after, so you can see whether the score was already rising on its own. This post makes the case that a reasoning delta is a numerator, and these controls are the denominator that turns it into evidence, with a copy-pasteable protocol and a one-line sensitivity number (the E-value) that tells you how strong a hidden confounder would have to be to erase your result.
```
