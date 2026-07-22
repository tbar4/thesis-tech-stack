# Reward models and preference data

A reward model is the field's answer to a hard question: how do you turn "this response is better than that one," a judgment humans make easily and specify precisely never, into a differentiable scalar you can optimize against a billion times? You cannot ask annotators to write down a number on an absolute scale, because human absolute ratings are noisy, drift over a session, and are incomparable between people. But you can ask them to *compare* two responses and pick the better one, which is stable and cheap. The reward model is the machine that ingests a pile of those pairwise picks and emits a scalar score function $r_\phi(x, y)$ consistent with them. The bridge from "pick the better one" to "here is a scalar" is a fifty-year-old model of paired choice, Bradley-Terry, and deriving the reward-model loss from it is the whole theoretical content of this chapter. It also matters even though this thesis mostly uses *verifiable* rewards rather than learned ones, because the RM loss is the direct ancestor of the DPO loss in the next chapter, and because understanding how a learned reward gets *hacked* is the thing that keeps me honest when my "verifiable" reward turns out to have a loophole.

## Theory

### Pairwise preferences as data

The raw material is a dataset of comparisons. Each record is a prompt $x$ and two responses, with a human (or a stronger model) label saying which is preferred. Write the preferred response as $y_w$ (winner) and the dispreferred as $y_l$ (loser), so a record is $(x, y_w, y_l)$. That is deliberately minimal: no numeric rating, no absolute quality, just an ordering. The modeling question is how to explain a *whole dataset* of such orderings with a single scalar score function, and the answer requires a probabilistic model of how a latent score produces a noisy binary choice.

### Bradley-Terry and the RM loss

Bradley-Terry is the model that says: each item has a latent positive "strength," and the probability that item $i$ beats item $j$ in a comparison is its share of the two strengths. For reward modeling, the latent strength of response $y$ given prompt $x$ is $\exp(r_\phi(x, y))$, i.e. the reward is the log-strength. That single modeling choice turns the pile of orderings into a likelihood I can maximize.

```admonish derivation title="From Bradley-Terry to the reward-model loss"
**The Bradley-Terry model.** Assign each response a positive strength $s(y) = \exp\big(r_\phi(x, y)\big)$. Bradley-Terry posits that the probability the winner is preferred over the loser is the winner's share of total strength,

$$P(y_w \succ y_l \mid x) = \frac{s(y_w)}{s(y_w) + s(y_l)} = \frac{e^{r_\phi(x, y_w)}}{e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}}. \tag{6.4.1}$$

Divide numerator and denominator by $e^{r_\phi(x, y_w)}$ and the expression collapses into a logistic function of the reward *difference*. Let $\Delta = r_\phi(x, y_w) - r_\phi(x, y_l)$; then

$$P(y_w \succ y_l \mid x) = \frac{1}{1 + e^{-(r_\phi(x, y_w) - r_\phi(x, y_l))}} = \sigma\big(\Delta\big), \tag{6.4.2}$$

where $\sigma$ is the logistic sigmoid. Equation (6.4.2) is the crux and it is worth pausing on: the preference probability depends *only on the difference of rewards*, never on their absolute level. This is the mathematical statement that a reward model is identified only up to an additive constant, because $r_\phi(x, \cdot) + c$ gives the identical preference probabilities. The RM has no natural zero, which is exactly why the RL stage of equation (6.1.2) needs a KL anchor to a reference and why "calibration" of an RM is subtle.

**The loss.** Given a dataset $\mathcal{D} = \{(x, y_w, y_l)\}$ of independent comparisons, the likelihood of the observed preferences is $\prod \sigma(\Delta)$. Maximizing it is minimizing the negative log-likelihood, which is the reward-model loss,

$$\mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\,\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]. \tag{6.4.3}$$

This is just binary logistic regression on the reward difference, with the "label" being "the winner won." Its gradient makes the training dynamics transparent:

$$\nabla_\phi \mathcal{L}_{\text{RM}} = -\,\mathbb{E}\Big[\big(1 - \sigma(\Delta)\big)\,\big(\nabla_\phi r_\phi(x, y_w) - \nabla_\phi r_\phi(x, y_l)\big)\Big]. \tag{6.4.4}$$

The weight $(1 - \sigma(\Delta))$ is the model's *error*: it is near 1 when the model wrongly scores the loser above the winner (large negative $\Delta$) and near 0 once the winner is confidently ranked above the loser (large positive $\Delta$). So training pushes up the winner's reward and pushes down the loser's, hard on the pairs it currently gets wrong and barely at all on the pairs it already ranks correctly. That self-limiting weight is why a converged RM does not blow the reward gap to infinity: correctly ordered pairs stop contributing gradient.
```

### The architecture of a reward model

Concretely, a reward model is the pretrained (usually SFT'd) transformer with its language-modeling head replaced by a scalar head: a single linear layer mapping the final hidden state to one number. You feed the full sequence $(x, y)$ through, take the hidden state at the last token (or a pooled representation), and the scalar head reads off $r_\phi(x, y)$. Training initializes from the SFT model so the RM inherits its understanding of the domain, then optimizes equation (6.4.3) over the preference pairs. The batch structure is nice: each example is a (winner, loser) pair sharing a prompt, both are forwarded, the two scalars are subtracted, and the logistic loss is applied. Everything after the forward pass is a two-element logistic regression.

### Calibration, and why RMs get hacked

Two failure modes matter, and they are two sides of one coin.

The first is **miscalibration in the ordinary statistical sense**: does the RM's confidence match reality? If the RM says $\sigma(\Delta) = 0.9$ for a set of pairs, are 90% of them actually won by the response it favors? A well-calibrated RM's stated preference probabilities match empirical win rates, and you check it the same way you check a classifier, with a reliability diagram. RMs are frequently overconfident, especially out of the distribution of pairs they were trained on, which matters because the RL stage will drive the policy into exactly those out-of-distribution regions.

The second, and the one that actually bites, is **reward hacking**, and equation (6.4.3) explains why it is inevitable rather than a bug. The RM is trained to be correct *on the distribution of pairs it saw*, which are pairs of reasonable SFT-model outputs. The RL policy of equation (6.1.2), by construction, searches for the highest-reward response it can find, which means it actively seeks out the regions where the RM is *wrong in the optimistic direction*, responses the RM scores highly that a human would not actually prefer. Since the RM only ever learned the reward difference on in-distribution pairs (equation 6.4.2 has no absolute anchor), it extrapolates arbitrarily outside them, and the optimizer exploits every one of those extrapolation errors. This is Goodhart's law made mechanical: the RM is a *proxy* for human preference, optimizing the proxy hard turns the proxy into a target, and the correlation between proxy and true preference breaks exactly where the optimizer pushes. The symptoms are familiar: length exploitation (the RM learned "longer is often better" and the policy generates padding), format tics, sycophancy, confident-sounding wrongness. The KL penalty $\beta$ in equation (6.1.2) is the main defense, it keeps the policy near the distribution where the RM is still trustworthy, but a strong enough optimizer with a loose enough $\beta$ will find the holes.

```admonish gotcha
The most common RM hack is length, and it is subtle because it is partly real: in the preference data, longer answers genuinely are better on average (they are more thorough), so the RM correctly learns a positive length-reward correlation. The policy then discovers it can raise reward just by padding, which is *not* what the annotators meant. If you see mean response length climbing over RL steps while human-judged quality is flat, that is length hacking, and it is why serious RM pipelines length-normalize the reward or add explicit length penalties. The deeper lesson for this thesis: this is exactly the argument for *verifiable* rewards. A unit-test-passes reward or a boxed-answer-correct reward has far less exploitable slack than a learned RM, which is the whole reason RLVR is more stable on a small budget than classic RLHF. But "far less" is not "none," and Part VII's reward-hacking chapter shows the verifier loopholes that survive.
```

### The quality of preference data, and ensembling as a hedge

Two practical levers decide whether an RM is worth anything, and both follow from the loss. The first is the *agreement rate* of the preference labels. Equation (6.4.3) treats every pair as a clean vote, but human annotators agree with each other only some fraction of the time (inter-annotator agreement on open-ended preference is often in the 60 to 75 percent range), and that label noise sets a ceiling: an RM cannot be more reliable than the signal it was trained on, and a pair where annotators split 50/50 is teaching the model a coin flip. The mitigation is to filter to high-agreement pairs, or to weight pairs by annotator confidence, which is the RM-stage version of chapter 6.2's quality-over-quantity rule. The second lever is *distribution coverage*: the RM is trustworthy only where it has seen pairs, and the RL optimizer will drive the policy toward the edges of that coverage (that is the whole hacking mechanism), so preference data should span the response styles the policy might explore, not just the ones the SFT model currently produces.

A common hedge against both label noise and coverage gaps is a **reward-model ensemble**: train several RMs on different data subsets or seeds and combine their scores, often taking the mean minus a multiple of the disagreement (a lower-confidence-bound reward). The intuition is directly anti-hacking: where the RMs *agree*, the reward is trustworthy; where they *disagree*, the policy is probing a region none of them learned well, and penalizing disagreement steers the optimizer back toward the trustworthy interior. Ensembling costs more memory, which on 16GB is exactly the constraint that pushes me toward verifiable rewards instead, but it is the standard answer when a learned reward is unavoidable, and it is worth knowing because a single-RM pipeline with no disagreement signal is the one that gets hacked silently.

### Why this chapter matters even when I do not train an RM

For verifiable reasoning tasks I mostly skip the RM entirely and use a programmatic checker, so why derive all this? Three reasons. First, the DPO loss of chapter 6.5 is equation (6.4.3) with a *specific* reward substituted in, and you cannot understand DPO's implicit reward without the Bradley-Terry starting point here. Second, reward hacking is the central risk of the whole "evals as rewards" thesis, and equations (6.4.2)–(6.4.4) are the cleanest place to see *why* optimizing a learned proxy is structurally different from optimizing a verifier. Third, plenty of real reasoning setups use a hybrid, a verifier for correctness plus a learned reward or judge for style or process, and the moment a learned component enters the reward, the hacking analysis applies. So the RM is both a tool I sometimes use and the theoretical object the next two chapters are built out of.

## Tooling

TRL's `RewardTrainer` implements equation (6.4.3) directly. It expects a dataset of `chosen`/`rejected` text pairs, wraps the base model with `AutoModelForSequenceClassification` (num_labels=1, which *is* the scalar head), forwards both members of each pair, and applies the logistic loss on the score difference. PEFT LoRA sits on top exactly as in SFT, so an RM is trainable in the same 16GB budget as chapter 6.3's SFT, the scalar head is a rounding error on top of the frozen base. Evaluation of an RM is preference *accuracy* on a held-out split (what fraction of held-out pairs does it rank correctly), and I run that through chapter 3.7's `evalstats.bootstrap_mean` for a confidence interval and, for RM-A-versus-RM-B comparisons, `evalstats.mcnemar` on the per-pair correctness, because "RM A is 2 points better" is exactly the kind of noisy paired claim that chapter exists to discipline.

## Lab

The lab trains a small reward model on a preference set, then does the thing that makes the theory tangible: it plots a reliability diagram (calibration) and demonstrates a length hack by measuring the correlation between response length and assigned reward. The artifact is the trained RM plus a calibration/length report, which is the evidence base for "this RM is trustworthy in-distribution and here is exactly the axis along which it will be gamed."

This is a `uv` project in the training environment. From the repo root:

```bash
uv init labs/reward-model
cd labs/reward-model
uv add "unsloth[cu124]" trl peft datasets numpy matplotlib
uv add --editable ../../packages/evalstats
```

```python title="labs/reward-model/train_rm.py"
"""Train a small LoRA reward model from Bradley-Terry (equation 6.4.3),
then audit it: preference accuracy with a CI, a calibration curve, and a
length-hack correlation. Artifacts land in artifacts/.
"""
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

import evalstats as es

MODEL = "Qwen/Qwen3-4B"          # SFT'd base; RM head added below
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # num_labels=1 IS the scalar reward head r_phi(x, y).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, torch_dtype=torch.bfloat16,
        load_in_4bit=True, device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:4000]")
    eval_ds = load_dataset("trl-lib/ultrafeedback_binarized", split="test[:800]")

    cfg = RewardConfig(
        output_dir=str(OUT / "run"), per_device_train_batch_size=2,
        gradient_accumulation_steps=8, num_train_epochs=1,
        learning_rate=1e-4, bf16=True, max_length=1024,
        logging_steps=25, report_to="none",
    )
    trainer = RewardTrainer(
        model=model, args=cfg, processing_class=tok,
        train_dataset=ds, peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(OUT / "rm"))

    # ---- Audit the RM on the held-out split ----
    model.eval()

    @torch.no_grad()
    def reward(text: str) -> float:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=1024).to(model.device)
        return float(model(**ids).logits.squeeze())

    correct, deltas, lengths_pref = [], [], []
    for ex in eval_ds:
        rw = reward(ex["chosen"])
        rl = reward(ex["rejected"])
        correct.append(1 if rw > rl else 0)
        deltas.append(rw - rl)
        # length-hack probe: reward vs char length of the chosen response
        lengths_pref.append((len(ex["chosen"]), rw))

    acc = es.bootstrap_mean(correct, level=0.95)
    print(f"pref accuracy = {acc.point:.3f}  "
          f"95% CI [{acc.ci_low:.3f}, {acc.ci_high:.3f}]")

    # Calibration: bin by predicted P(win) = sigmoid(delta), compare to empirical.
    p_win = 1 / (1 + np.exp(-np.array(deltas)))
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p_win, bins) - 1
    rel = []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            rel.append(((bins[b] + bins[b + 1]) / 2,
                        float(np.mean(np.array(correct)[m])), int(m.sum())))

    # Length hack: Pearson correlation between response length and reward.
    L = np.array([a for a, _ in lengths_pref], float)
    R = np.array([b for _, b in lengths_pref], float)
    length_corr = float(np.corrcoef(L, R)[0, 1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    xs = [r[0] for r in rel]; ys = [r[1] for r in rel]
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax[0].plot(xs, ys, "o-", label="RM")
    ax[0].set_xlabel("predicted P(chosen wins)"); ax[0].set_ylabel("empirical win rate")
    ax[0].set_title("Reliability diagram"); ax[0].legend()
    ax[1].scatter(L, R, s=4, alpha=0.3)
    ax[1].set_xlabel("chosen response length (chars)"); ax[1].set_ylabel("reward")
    ax[1].set_title(f"Length vs reward (r = {length_corr:+.2f})")
    fig.tight_layout(); fig.savefig(OUT / "rm_audit.png", dpi=150)

    (OUT / "rm_audit.txt").write_text(
        f"pref_accuracy {acc.point:.4f} CI [{acc.ci_low:.4f}, {acc.ci_high:.4f}]\n"
        f"length_reward_pearson {length_corr:+.4f}\n")
    print(f"Artifacts: {(OUT / 'rm').resolve()}, {(OUT / 'rm_audit.png').resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python train_rm.py
```

```admonish under-the-hood
`RewardTrainer` does not compute per-token anything; it forwards the `chosen` and `rejected` sequences, reads the scalar at the final non-pad position of each, subtracts them to form $\Delta$, and applies `-log_sigmoid(delta)`, which is equation (6.4.3) verbatim. The reason the pad token id must be set on the model config is that "final non-pad position" is where the scalar head is read, and a wrong pad id reads the score off a padding token, which silently trains garbage. If your RM accuracy sits at chance (0.5) with a falling loss, check the pad handling first; it is the single most common RM-training footgun.
```

**What you should see.** After one epoch on 4000 pairs, held-out preference accuracy should land somewhere in the 0.65 to 0.75 range with a bootstrap CI a few points wide (a perfect RM is not the goal; beating chance decisively is). The reliability diagram should track the diagonal loosely in the middle and sag away from it at the confident ends, the visible signature of the overconfidence that RL will later exploit. The length-versus-reward scatter will show a *positive* Pearson correlation, often in the 0.2 to 0.4 range, which is the length hack sitting right there in the trained reward: the model has learned, partly correctly and partly exploitably, that longer chosen responses score higher, and that is the exact axis a policy would game under equation (6.1.2). Record training wall-clock and peak VRAM (measured on the baseline machine, record value, date, driver); the RM trains in the same envelope as the chapter 6.3 SFT because the scalar head is negligible. The artifacts, the saved RM and `rm_audit.png`, are the evidence that this reward is useful in-distribution and precisely where it is not.

```admonish read-along
Pair this with **[RLHF]** ch. 5, the reward-modeling chapter: it develops Bradley-Terry (equation 6.4.1), the logistic RM loss (equation 6.4.3), the practical architecture of the scalar head, and the calibration-and-hacking discussion at length, and it is the reference the next chapter's DPO derivation assumes you have read, because DPO is this loss with the reward re-expressed through the policy.
```

```admonish substack-seed
"You can't teach a model what 'good' means, but you can teach it to break ties, and that turns out to be enough, right up until it isn't." A post that derives the reward model from Bradley-Terry (equation 6.4.2, the preference depends only on the *difference* of scores, so a reward model has no natural zero) and then shows the trap baked into equation (6.4.3): the RM is only correct on the pairs it saw, an RL optimizer's entire job is to find responses the RM scores highly, so it hunts down exactly the RM's blind spots. Close on the length-hack scatter from the lab, a real trained reward that quietly loves long answers, as the concrete face of Goodhart's law and the reason verifiable rewards exist.
```
