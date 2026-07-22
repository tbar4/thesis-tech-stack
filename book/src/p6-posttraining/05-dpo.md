# Direct alignment: DPO and family

The reward-model-then-RL pipeline of the last two chapters is correct, powerful, and a genuine pain to run on one GPU: a separate reward model to train and hold in memory, a sampling loop, a value function or group baseline, a KL estimator, and all the PPO or GRPO plumbing from Part V, with the policy and reference and reward all fighting for 16 GiB. Direct Preference Optimization is the observation that for the specific RLHF objective in equation (6.1.2) you do not need any of that. You can solve the constrained optimization in closed form, notice that the reward model is *implicit* in the optimal policy, invert the relationship, and substitute it back into the Bradley-Terry loss of the last chapter. What drops out is a single supervised-looking loss on preference pairs with no reward model, no sampling, and no RL loop at all. This chapter derives that in full, because the derivation is one of the most elegant results in post-training and because the *implicit reward* it exposes is the concept that makes sense of the entire variant zoo (IPO, KTO, ORPO) that followed. Then I am honest about when DPO's offline nature costs you and on-policy RL earns its keep, which for this thesis is most of the time.

## Theory

### The RLHF objective has a closed-form optimum

Start from the KL-regularized reward-maximization objective, equation (6.1.2), the thing PPO spends thousands of GPU-hours approximating. The key fact is that this objective, for a *fixed* reward function $r$, has an exact closed-form solution for the optimal policy. Deriving it is the first half of DPO.

```admonish derivation title="The optimal KL-regularized policy"
Fix a prompt $x$ and a reward $r(x, y)$. The objective for the policy $\pi$ is

$$\max_{\pi}\; \mathbb{E}_{y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] \;-\; \beta\, \mathbb{D}_{\text{KL}}\!\big[\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big]. \tag{6.5.1}$$

Write the KL out as an expectation and pull everything under one $\mathbb{E}_{y \sim \pi}$. Maximizing (6.5.1) is minimizing its negative:

$$\min_{\pi}\; \mathbb{E}_{y \sim \pi}\!\left[\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y)\right]. \tag{6.5.2}$$

Now the trick: fold the reward into the log by defining a normalized target distribution. Introduce the partition function

$$Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big), \tag{6.5.3}$$

and define

$$\pi^{*}(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big). \tag{6.5.4}$$

$\pi^{*}$ is a valid probability distribution by construction ($Z$ normalizes it). Substitute (6.5.4) into (6.5.2) by adding and subtracting $\log Z(x)$; the objective becomes

$$\min_{\pi}\; \mathbb{E}_{y \sim \pi}\!\left[\log \frac{\pi(y \mid x)}{\pi^{*}(y \mid x)}\right] - \log Z(x) \;=\; \min_{\pi}\; \mathbb{D}_{\text{KL}}\!\big[\pi(\cdot\mid x)\,\|\,\pi^{*}(\cdot\mid x)\big] - \log Z(x). \tag{6.5.5}$$

$Z(x)$ does not depend on $\pi$, so the whole problem is: minimize a KL divergence to $\pi^{*}$. A KL divergence is minimized (to zero, its floor) exactly when the two distributions are equal. Therefore the optimal policy is

$$\boxed{\;\pi^{*}(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big).\;} \tag{6.5.6}$$

Equation (6.5.6) is exact and completely general: for *any* reward, the RLHF-optimal policy is the reference distribution reweighted by the exponentiated reward. PPO exists only because $Z(x)$ is an intractable sum over all sequences, so you cannot write $\pi^{*}$ down and sample from it directly. DPO's move is to not sample from it at all.
```

### Inverting the reward: the implicit reward

Equation (6.5.6) says the optimal policy is a function of the reward. DPO reads it the other way: solve for the reward as a function of the optimal policy. This is the pivot of the whole method.

```admonish derivation title="The DPO loss, in full"
Take the log of equation (6.5.6) and solve for $r(x, y)$:

$$\log \pi^{*}(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \tfrac{1}{\beta} r(x, y) - \log Z(x), \tag{6.5.7}$$

$$\Longrightarrow\quad r(x, y) = \beta\, \log \frac{\pi^{*}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x). \tag{6.5.8}$$

Equation (6.5.8) is the **implicit reward**: any reward function is recoverable from its optimal policy as $\beta$ times the log-ratio of that policy to the reference, plus a prompt-dependent constant $\beta \log Z(x)$. Now identify the optimal policy $\pi^{*}$ with the policy we are training, $\pi_\theta$ (this is the modeling assumption: we assume our model *is* the optimum for *some* reward, and we solve for the $\theta$ that makes that reward fit the data). Define the **DPO implicit reward**

$$\hat r_\theta(x, y) = \beta\, \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}. \tag{6.5.9}$$

Now plug this into the Bradley-Terry preference model from chapter 6.4. Recall equation (6.4.2): the preference probability depends only on the *difference* of rewards. Take the difference of the implicit rewards for the winner and loser of a preference pair:

$$\hat r_\theta(x, y_w) - \hat r_\theta(x, y_l) = \beta\, \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\, \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}. \tag{6.5.10}$$

**Here is the miracle.** The intractable constant $\beta \log Z(x)$ from equation (6.5.8) is the *same* for both $y_w$ and $y_l$ (it depends only on the prompt $x$), so it *cancels in the difference*. The partition function that forced PPO to exist has vanished. Recall the Bradley-Terry negative log-likelihood of equation (6.4.3), which trains on a preference pair by maximizing the log-probability that the winner beats the loser: $\mathcal{L} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)$. Substitute equation (6.5.10) into it:

$$\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right].\;} \tag{6.5.11}$$

Equation (6.5.11) is DPO. It is the reward-model loss of chapter 6.4 with the reward replaced by the policy-versus-reference log-ratio. There is no reward model (the reward is implicit in $\pi_\theta$), no sampling loop (you use the fixed preference pairs), and no RL (it is a supervised loss you backprop through directly). You need only the trained policy and a frozen copy of the reference, and you evaluate four log-probabilities per pair: $\pi_\theta$ and $\pi_{\text{ref}}$ on each of $y_w$ and $y_l$.
```

### Reading the DPO gradient

The loss is elegant; its gradient is what tells you what it actually *does*.

```admonish derivation title="What the DPO update pushes on"
Let $u = \beta\big[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\big]$ be the implicit-reward margin. Differentiating equation (6.5.11),

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\,\beta\,\mathbb{E}\Big[\underbrace{\sigma(-u)}_{\text{weight}}\,\big(\nabla_\theta \log \pi_\theta(y_w\mid x) - \nabla_\theta \log \pi_\theta(y_l\mid x)\big)\Big]. \tag{6.5.12}$$

Read it in three parts. The bracket is "increase the log-prob of the winner, decrease the log-prob of the loser," the intuitively obvious thing. The weight $\sigma(-u)$ is the implicit-reward *error*: it is large when the current model has the margin wrong (loser ranked above winner, $u < 0$) and small once the winner is confidently preferred ($u \gg 0$), so DPO, like the RM loss it descends from, spends its gradient on the pairs it currently gets wrong and coasts on the ones it has learned. The $\pi_{\text{ref}}$ terms have no $\theta$ in them so they vanish from the gradient, but they are *not* inert: they set the reference point in $u$, which is why DPO does not simply maximize winner-probability and minimize loser-probability without bound (that would be the unregularized limit). The reference keeps the update anchored, playing exactly the role the KL term played in equation (6.5.1); $\beta$ is the same regularization strength, now a plain loss coefficient.
```

A subtlety worth flagging, because it bites in practice: nothing in equation (6.5.12) forces the winner's *absolute* probability up. The loss only cares about the *margin*, so a common DPO failure is that both $\log \pi_\theta(y_w)$ and $\log \pi_\theta(y_l)$ *fall* over training, the loser just falls faster. The model gets the preference right while making the winning responses themselves less likely, which can degrade generation quality even as the DPO metrics improve. This is one of the motivations for the variant zoo below, and it is why I watch the raw chosen-logprob, not just the loss.

### When DPO, when RL

DPO's price for its elegance is that it is *offline*. It optimizes against a fixed set of preference pairs collected once; it never samples new responses from the current policy and never gets fresh judgment on them. On-policy RL (PPO, GRPO, RLVR) does exactly that: it samples from the current policy and scores those samples, so it can discover and reinforce responses better than anything in a static dataset, and it keeps getting signal in the regions the policy actually drifts into. The tradeoff shakes out cleanly. DPO wins when you have good preference data and want alignment-of-style cheaply, it is dramatically simpler and lighter, one training run, no sampling, fits a 16GB card with room to spare. On-policy RL wins when you can *verify* correctness and want to push capability past the data, because the exploration is the whole point and a verifier gives clean per-sample signal, which is precisely the RLVR regime this thesis targets. So for this book, DPO is the method I understand deeply and reach for on preference-style tasks, and GRPO-on-verifiable-rewards (Part V and Part VII) is the workhorse for the reasoning delta. They are not competitors so much as two points on the offline-to-online axis.

### The variant zoo

The implicit-reward view of equation (6.5.9) generates the whole family; each variant changes one modeling choice.

- **IPO** (Identity Preference Optimization) replaces the logistic Bradley-Terry link with a squared loss on the margin, targeting a fixed margin rather than pushing $\sigma(u)$ toward 1. This addresses DPO's tendency to overfit deterministic preferences (when a pair is always labeled the same way, the logistic loss wants $u \to \infty$, which IPO's bounded target prevents).
- **KTO** (Kahneman-Tversky Optimization) drops the *pairwise* requirement entirely: it works from *unpaired* binary "good response / bad response" labels using a prospect-theory-inspired value function. This matters because paired data is expensive and unpaired thumbs-up/down signal is everywhere; KTO trades the clean Bradley-Terry footing for cheaper supervision.
- **ORPO** (Odds Ratio Preference Optimization) folds the SFT and preference stages into *one* loss with no reference model at all, adding an odds-ratio preference penalty to the ordinary SFT cross-entropy. No reference copy means less memory and one fewer stage, at the cost of the clean KL-anchor interpretation.

The through-line: DPO showed that preference alignment is a supervised loss in disguise, and every variant is a different answer to "which supervised loss, on which data, with which anchor." Understanding equation (6.5.11) is understanding all of them.

### Two knobs that decide whether DPO works: beta and length normalization

Two practical choices dominate DPO outcomes, and both fall out of the derivation rather than being tuning folklore. The first is $\beta$, which equation (6.5.9) reveals is not a learning rate but the *strength of the reference anchor*, the same $\beta$ that scaled the KL term in equation (6.5.1). A small $\beta$ (say 0.01) lets the policy move far from the reference, which chases the preferences hard and risks the degenerate "both logprobs collapse" regime; a large $\beta$ (say 0.5) keeps the policy tightly pinned to the SFT model, which is safe but barely moves. The typical 0.1 is a compromise, and the right value depends on how much you trust your preference data: trustworthy pairs justify a smaller $\beta$, noisy pairs demand a larger one. The second knob is *length normalization*. The implicit reward of equation (6.5.9) is a sum of per-token log-ratios, so it scales with sequence length, which means DPO inherits the exact length bias I warned about for reward models in chapter 6.4: if winners are systematically longer than losers, the margin can be raised by length alone. This is why length-normalized variants (dividing each log-ratio by its token count, as in the "R-DPO" and SimPO-style corrections) exist, and why I watch mean response length over DPO training the same way I watch it over RL. A DPO run whose "improvement" is entirely a length increase is the offline twin of RM length-hacking, and the same defenses apply.

## Tooling

TRL's `DPOTrainer` implements equation (6.5.11) directly, and its in-family variants are one config flag away (`loss_type="ipo"` for IPO); the ones that change the *data* or drop the reference are separate trainers, `KTOTrainer` for KTO on *unpaired* good/bad labels and `ORPOTrainer` for ORPO. It expects a dataset with `prompt`, `chosen`, `rejected` fields, and it manages the reference model for you: with a LoRA policy, the reference is simply the base model with the adapter *disabled*, so you do not pay for a second full model in VRAM, a genuinely important trick on 16GB that Unsloth and PEFT make automatic. The trainer computes the four log-probabilities per pair, forms the margin, and applies the logistic loss. Evaluation is the same as any alignment run: score the DPO'd model on the frozen thesis suite and report the delta versus the SFT starting point through chapter 3.7's `evalstats.bootstrap_paired_diff` and `evalstats.mcnemar`, so "DPO helped" is an interval and a p-value.

## Lab

The lab DPO-tunes the SFT model from chapter 6.2 on a preference set and, crucially, *watches the implicit reward*: it logs the chosen and rejected log-probabilities and the implicit-reward margin over training, so the derivation's abstractions (equations 6.5.9, 6.5.12) become curves you can see. The artifact is the DPO'd adapter plus a training-dynamics plot that exposes the "both logprobs fall" failure mode if it happens.

This is a `uv` project in the training environment. From the repo root:

```bash
uv init labs/dpo-align
cd labs/dpo-align
uv add "unsloth[cu124]" trl peft datasets matplotlib
uv add --editable ../../packages/evalstats
```

```python title="labs/dpo-align/train_dpo.py"
"""DPO-align the chapter-6.2 SFT model, logging the implicit reward margin
(equation 6.5.9) so the derivation becomes a curve. Writes the DPO adapter
and artifacts/dpo_dynamics.png.
"""
from pathlib import Path

import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

SFT_ADAPTER = "../sft-4b/artifacts/adapter"   # cold start from chapter 6.2
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def main() -> None:
    # Policy = base + the chapter-6.2 SFT adapter (the actual cold start).
    # Loading the SAVED adapter continues training from the SFT weights; a
    # fresh get_peft_model would zero-init a NEW adapter and DPO from base,
    # discarding the cold start. Unsloth reads the base off the adapter config.
    # DPOTrainer then uses this same model with the adapter disabled as the
    # frozen reference, so no second full model sits in VRAM.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=SFT_ADAPTER, max_seq_length=2048, load_in_4bit=True, dtype=None)

    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:4000]")

    cfg = DPOConfig(
        output_dir=str(OUT / "run"),
        beta=0.1,                        # the beta of equations (6.5.9)/(6.5.11)
        loss_type="sigmoid",             # DPO link; set "ipo" for IPO. KTO is a
                                         # separate KTOTrainer on unpaired data.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1, learning_rate=5e-6,
        warmup_ratio=0.05, lr_scheduler_type="cosine",
        bf16=True, max_length=1024, max_prompt_length=512,
        logging_steps=10, report_to="none", seed=3407,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()

    model.save_pretrained(str(OUT / "dpo_adapter"))

    # TRL logs rewards/chosen, rewards/rejected, rewards/margins per step.
    hist = trainer.state.log_history
    steps = [h["step"] for h in hist if "rewards/chosen" in h]
    chosen = [h["rewards/chosen"] for h in hist if "rewards/chosen" in h]
    rejected = [h["rewards/rejected"] for h in hist if "rewards/rejected" in h]
    margin = [h["rewards/margins"] for h in hist if "rewards/margins" in h]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(steps, chosen, label="implicit reward: chosen")
    plt.plot(steps, rejected, label="implicit reward: rejected")
    plt.plot(steps, margin, label="margin (chosen - rejected)")
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("step"); plt.ylabel(r"$\beta \log \pi_\theta/\pi_{ref}$")
    plt.title("DPO implicit-reward dynamics"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "dpo_dynamics.png", dpi=150)
    print(f"Artifacts: {(OUT / 'dpo_adapter').resolve()}, "
          f"{(OUT / 'dpo_dynamics.png').resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python train_dpo.py
```

```admonish gotcha
Watch the two implicit-reward curves, not just the margin. The margin (equation 6.5.10) will rise, that is all the loss optimizes, but a *healthy* DPO run keeps the chosen implicit reward roughly flat or rising while the rejected one falls. If *both* fall and the margin rises only because the rejected one falls faster, you are in the degenerate regime I warned about under equation (6.5.12): the model is getting the preference right by making good responses less likely too, which shows up as worse generations despite a "better" DPO loss. Fixes are a larger $\beta$ (stronger reference anchor), a lower learning rate, or switching to IPO (`loss_type="ipo"`), whose bounded target does not chase the margin to infinity. This plot is the whole reason the lab logs the raw rewards.
```

**What you should see.** Over one epoch the implicit-reward margin climbs steadily from around 0 into positive territory (the model learns to prefer the chosen responses), and in a healthy run the chosen curve stays near flat while the rejected curve drifts down, so the boxed loss of equation (6.5.11) is doing what the derivation promised. Because the reference is the adapter-disabled base, peak VRAM is close to the chapter 6.3 SFT budget, well inside 16 GiB, with the caveat that DPO forwards four sequences per pair (policy and reference on chosen and rejected) so the throughput is lower than SFT, plan for it to be a few times slower per example. Record wall-clock, tokens/sec, and peak VRAM (measured on the baseline machine, record value, date, driver). The headline artifact is `dpo_dynamics.png`, which turns equations (6.5.9) and (6.5.12) from symbols into three curves you can point at, and the saved DPO adapter, which you can then score against the SFT baseline through `evalstats` to put a CI on "did direct alignment help."

```admonish read-along
Pair this with **[RLHF]** ch. 8, the direct-alignment chapter: it walks the same derivation from the KL-regularized objective (equation 6.5.1) through the closed-form optimum (equation 6.5.6) to the DPO loss (equation 6.5.11), and it surveys the IPO/KTO/ORPO variants with the tradeoffs I summarized. Read it alongside chapter 6.4 here, because the DPO loss is quite literally the reward-model loss of that chapter with the implicit reward of equation (6.5.9) substituted for the learned $r_\phi$.
```

```admonish substack-seed
"The reward model was hiding inside the language model the whole time." A post that derives DPO the way it deserves: start from the RLHF objective everyone spends fortunes approximating with PPO, show it has an exact closed-form optimum (equation 6.5.6, the policy is just the reference reweighted by exponentiated reward), then invert it to read the reward off the policy (equation 6.5.9) and watch the intractable partition function cancel in the preference *difference* (equation 6.5.10). What is left is a supervised loss you can train on a laptop-class GPU. End on the honest catch: it is offline, so it cannot explore, which is exactly why verifiable-reward RL still earns its cost when you are chasing capability instead of style.
```
