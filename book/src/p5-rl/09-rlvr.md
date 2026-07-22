# RLVR: reinforcement with verifiable rewards

This is the chapter the whole thesis is built on, so I want to state its thesis plainly before deriving anything. Everything in Part V so far, the policy gradient, actor-critic, PPO's clip, GRPO's group baseline, is machinery for turning a *reward signal* into a *better policy*. RLVR (Reinforcement Learning with Verifiable Rewards) is a claim about where that reward signal should come from: not a learned reward model that approximates human preference and can be gamed, but a *verifier*, a program that checks whether an answer is actually correct. On tasks where correctness is checkable (math with a known answer, code that must pass tests, a proof a checker accepts, a format a parser validates), the reward is the verifier's verdict, and that changes the character of the whole enterprise. The reward is grounded, cheap, and un-hackable in the usual sense, and it turns the eval harness (Part III) into the training environment. That last sentence is the thesis: the thing that measures the model is the thing that trains it, and this chapter makes the identification formal. I will lay out binary and graded verifiable rewards, work carefully through the sharp and genuinely unsettled question of *when RL adds capability over supervised fine-tuning*, catalogue the failure modes that bite on real runs, and end with the formal statement of the loop that Part VII implements on the baseline machine.

## Theory

### Verifiable rewards, binary and graded

A verifiable reward is a function $v(q, o) \to \mathbb{R}$ that takes a prompt $q$ and a completed response $o$ and returns a score computed by *checking*, not by a learned model's opinion. The cleanest case is binary:

$$r = v(q, o) = \mathbb{1}[\,\text{answer extracted from } o \text{ matches the reference}\,] \in \{0, 1\}. \tag{9.1}$$

For grade-school math this is "parse the final number and compare to the known answer"; for code it is "run the unit tests, reward 1 if all pass." Binary rewards are the sharpest possible signal (correct or not, no ambiguity) and they are what DeepSeek-R1-Zero used to bootstrap reasoning from a base model with nothing but rule-based checkers. But binary is also the *sparsest* signal, which is the source of half the failure modes below.

Graded rewards soften the cliff. Instead of all-or-nothing, return partial credit:

$$r = v(q, o) = \alpha \cdot (\text{fraction of unit tests passed}) + \beta \cdot \mathbb{1}[\text{format valid}] + \cdots \in [0, 1]. \tag{9.2}$$

Code with 7 of 10 tests passing gets 0.7, not 0; a response in the right format but wrong answer might get a small shaping term for at least being parseable. Graded rewards give the policy a gradient to climb even before it produces a fully correct answer, which matters enormously for hard prompts where binary reward is zero for thousands of samples and the policy never gets a foothold. The tradeoff is that every shaping term is a surface the policy can exploit (reward the format and the model learns to emit perfect format around wrong answers), so graded reward design is where "verifiable" starts to leak back toward "hackable" and must be watched. The thesis default is binary where the task allows it and minimal, defensible grading where binary is too sparse to learn from.

The crucial property both share, and the reason RLVR is different from RLHF, is that $v$ is a *fixed function*, not a learned model with parameters. It cannot drift, it costs no VRAM (it is Python, or a subprocess), and it cannot be gamed by finding adversarial inputs to a reward network because there is no network. This is the single fact that makes the whole loop fit on one GPU, and I will cash it out in the vram-budget below.

### When does RL add over SFT? The pass@k reshaping question

Here is the question that actually matters, and it is genuinely unsettled, so I will present it honestly rather than as settled doctrine. Supervised fine-tuning (SFT) on correct solutions already improves a model. What does RLVR add on top? There are two hypotheses, and the truth is some contested mixture.

**Hypothesis A: RL reshapes the sampling distribution without adding new capability.** The base (or SFT) model, sampled enough times, can *already* produce a correct solution to many problems it fails on a single try. Its pass@k (probability that at least one of $k$ samples is correct) is high even where its pass@1 is low. On this view, RLVR does not teach the model anything it could not already do; it *concentrates probability mass* onto the correct solutions the base model could already occasionally sample, converting pass@k performance into pass@1 performance.

```admonish derivation title="Pass@k reshaping, made precise"
Fix a prompt and let $p$ be the probability that a single sample from the policy is correct (pass@1). Assuming independent samples, the probability that at least one of $k$ samples is correct is

$$\text{pass@}k = 1 - (1 - p)^k. \tag{9.3}$$

This is concave and saturating in $k$: even a small $p$ gives high pass@$k$ for large $k$ (with $p = 0.05$, pass@100 $= 1 - 0.95^{100} \approx 0.994$). Hypothesis A says RLVR's effect is to *raise $p$* toward a value the base model's pass@$k$ curve already promised was achievable, without expanding the *set* of problems where $p > 0$. Concretely, if the base model has $p_{\text{base}} > 0$ on a problem, RL can push $p \to$ near 1; but if $p_{\text{base}} = 0$ (the correct solution is never sampled, at any $k$), RL has nothing to reinforce, because the policy gradient can only upweight trajectories it actually samples. This is the sharp, testable prediction: **RL cannot create probability mass where there was exactly zero.** The policy gradient $\nabla_\theta J = \mathbb{E}[\hat{A}\,\nabla_\theta\log\pi_\theta]$ only ever reweights sampled outcomes, so a solution the policy assigns probability zero gets gradient zero, forever. Under Hypothesis A, plot pass@$k$ versus $k$ for the base model and the RL model: the RL model wins decisively at small $k$ (higher pass@1) but the two curves *converge* at large $k$, because RL only moved mass around within the base model's already-reachable set.
```

**Hypothesis B: RL can add genuinely new capability.** The empirical picture is more complicated than Hypothesis A allows. Some studies (Yue et al. 2025, "Does RL Really Incentivize Reasoning Beyond the Base Model?") find exactly the crossover the derivation predicts, base model catching up at large $k$, supporting A. Others find RL-trained models that solve problems the base model does not solve at any sampled $k$ within a large budget, and that improvements persist with continued training and better exploration, suggesting RL sometimes does more than reshape. The reconciliation most people now hold: RL's *primary, reliable* effect is pass@k-to-pass@1 reshaping (Hypothesis A is the safe bet for what you will observe on a modest run), but the *reachable set* is not truly fixed, because exploration (temperature, sampling, longer training, curriculum) can surface solutions that were effectively-but-not-exactly zero probability, and once sampled they can be reinforced into the model's repertoire. The honest thesis-committee statement is: on a single-GPU budget with modest compute, expect RLVR to mostly *sharpen* what SFT and the base model already make reachable, and treat any evidence of genuinely new capability as a claim requiring the large-$k$ pass@k control to substantiate.

This distinction is not academic for the thesis, it dictates the *evaluation*. If I only measure pass@1 I cannot tell reshaping from new capability, so the eval harness must report pass@k across a range of $k$, which is exactly the statistics chapter of Part III feeding back into the RL chapter. The measurement and the training are the same instrument, again.

### Failure modes

Three failure modes recur, and all three are consequences of the reward structure rather than bugs.

**Reward sparsity.** With a binary reward (equation 9.1) on hard prompts, the policy samples $G$ responses and *all* of them are wrong, so every reward is zero, the group mean is zero, the advantage (GRPO's equation 8.1) is zero, and there is no gradient. The model learns nothing from its hardest, most valuable problems. This is the flip side of "un-hackable": a verifier that says only "no" gives no direction to improve. Mitigations are graded rewards (equation 9.2, partial credit gives a foothold), curriculum (train on prompts near the model's current ability so groups have reward variance), and dynamic sampling (GRPO chapter, filter out all-zero groups so compute goes to prompts with signal).

**Exploration collapse (entropy collapse).** RL on a verifiable reward is a positive-feedback loop: the policy upweights what works, which makes it sample those things more, which gives it less diverse data, which narrows it further. Left unchecked the policy's entropy crashes, it commits to a narrow band of solution strategies, and it stops discovering. You see it as the entropy metric falling toward zero and the pass@k curve flattening because every sample is now nearly identical. Mitigations are the KL-to-reference penalty (GRPO's $\beta$ term keeps the policy from sprinting away from the diverse base model), clip-higher (DAPO, chapter 8, lets rare good tokens climb so exploration is not throttled by the upper clip), and entropy bonuses. The whole design tension of RLVR is exploit-versus-explore, and it is the bandit chapter's dilemma returning at the scale of a language model.

**Reward hacking / specification gaming.** Even a verifier can be gamed if it is a proxy for what you want rather than the thing itself. Reward the format and get well-formatted nonsense; reward test-passing and get code that special-cases the test inputs; reward answer-matching and get a model that learns to emit the answer string in a way the parser accepts without doing the reasoning. Verifiable reward is *harder* to hack than a learned reward model (there is no adversarial input to a frozen network), but "verifiable" is not "unhackable," it just moves the vulnerability from the reward model's weights to the verifier's specification. The mitigation is verifier hygiene: check the extraction is robust, hold out the reasoning from the reward where possible, and audit high-reward samples by hand, which is again an eval activity.

### The eval-harness-as-environment framing

Now the framing the thesis lives on, stated carefully. In the RL formalism (Part V chapter 1) an *environment* is defined by a state space, an action space, a transition function, and a reward function. Claim: for RLVR on verifiable reasoning tasks, *the eval harness is the environment*, and every piece lines up.

The state is the prompt plus the tokens generated so far. The action is the next token. The transition is deterministic and trivial (append the token, the language model itself provides the dynamics). And the reward function *is the verifier the eval harness already runs to score the model*. This is the identification: the same harness that, in Part III, generated model outputs on a task set and scored them to produce an accuracy number, is, in Part V, generating rollouts and scoring them to produce a reward. Evaluation generates $(q, o, v(q,o))$ triples to *report* a number; RLVR generates the same triples to *train on* them. The harness does not know or care which you are doing. That is why the thesis title is "Evals as Rewards": the eval is not a separate measurement bolted onto training, it is literally the reward channel of the RL environment, and building one good harness gives you both your metric and your training signal from a single artifact. This is the whole architectural bet of the thesis, and it is what makes a serve-evaluate-score-train loop coherent on one machine.

```admonish thesis-thread title="The thesis loop, formally"
Let $\pi_\theta$ be the policy (a language model), $\mathcal{D}$ a dataset of verifiable prompts $q$ each with reference answer $a^\star_q$, and $v(q, o) \to [0,1]$ the verifier: a fixed program that scores a completed response $o$ against $a^\star_q$ (equation 9.1 binary, or 9.2 graded). The RLVR environment is the tuple

$$\mathcal{E} = \big(\;\mathcal{S},\; \mathcal{A},\; T,\; R\;\big), \qquad \begin{aligned}
\mathcal{S} &= \{\,(q, o_{<t})\,\}\ \text{(prompt + partial generation)},\\
\mathcal{A} &= \text{vocabulary (next token)},\\
T\big((q,o_{<t}), a\big) &= (q,\, o_{<t}\,\Vert\, a)\ \text{(deterministic append)},\\
R\big((q,o)\big) &= v(q, o)\ \text{at the terminal token, } 0\text{ elsewhere},
\end{aligned} \tag{9.4}$$

and the reward function $R$ **is the eval harness's scorer**. The training objective is the expected verifier score,

$$\max_{\theta}\; \mathbb{E}_{q\sim\mathcal{D}}\;\mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}\big[\,v(q, o)\,\big], \tag{9.5}$$

optimized by GRPO (chapter 8) so no value model is needed. The loop that Part VII runs on the baseline machine (MSI Aegis R2, RTX 5080 16GB) is one iteration of stochastic optimization of (9.5):

1. **Serve.** Load $\pi_\theta$ into vLLM. For a batch of prompts $q \sim \mathcal{D}$, sample a group of $G$ completions $\{o_1,\dots,o_G\}$ each (this is generation, the throughput bottleneck).
2. **Evaluate.** Run the *eval harness* over the $(q, o_i)$ pairs exactly as it would to report a benchmark number.
3. **Score.** The harness's verifier returns $r_i = v(q, o_i)$; these rewards *are* the eval scores.
4. **Train.** Compute group-relative advantages $\hat{A}_i = (r_i - \bar{r})/\text{std}(r)$ (equation 8.1), take a GRPO step on the clipped-plus-KL objective (equation 8.2), updating $\theta$.
5. **Re-evaluate.** Periodically run the harness as a held-out *measurement* (pass@1 and pass@k, per Hypothesis A/B) to track whether the policy is improving or merely reshaping, then return to step 1 with the updated $\pi_\theta$.

The identity that makes this one loop rather than two systems is $R = v = $ (the harness scorer). Steps 2-3 during training and step 5 as measurement call the *same code*. Evals are rewards.
```

## Tooling

The tooling for RLVR is the assembly of everything Part III and Part V built: an eval harness that can both score and, structurally, serve as a reward function, plus GRPO from chapter 8. In practice the reward function passed to TRL's `GRPOTrainer` (its `reward_funcs` argument) is *literally* a Python callable with the signature `(prompts, completions, **kwargs) -> list[float]`, and the whole thesis claim is that this callable should be your eval harness's scorer, not a `RewardModel`. For a math task it wraps the same answer-extraction-and-compare the harness uses to compute accuracy; for code it shells out to the same sandboxed test runner. Frameworks purpose-built for this (verl, and the reward-function ecosystems around it) formalize the verifier as a first-class component, but the minimal version is a function, and the minimalism is the point: a reward that is a function has no weights to store, no server to stand up, and no drift to monitor.

```admonish under-the-hood
The reason "the verifier is a function" is load-bearing and not a throwaway: a *learned* reward model in the RLHF pipeline is a full neural network that must be held in VRAM during training, queried on every rollout, and (worse) is itself an approximation that the policy learns to exploit, which is why RLHF needs the KL leash so badly. Swapping it for a verifier removes an entire model from the resident set *and* removes the adversarial dynamic, because a frozen `assert answer == expected` has no gradient for the policy to hack toward. What remains hackable is the *specification* (the failure modes above), but that is a code-review problem, not a moving-target-model problem, and code review is something one person on one GPU can actually do. The whole reason RLVR democratized reasoning-model training is this collapse of "reward model" into "unit test."
```

```admonish vram-budget
RLVR is what makes the thesis fit in 16GB, and here is the accounting stated once, completely. A full RLHF-PPO setup on the baseline machine (RTX 5080 16GB, Blackwell) must hold four model-shaped things: policy (trained), value model (chapter 6), reference policy (KL), and reward model (scoring). RLVR-GRPO holds *two*: the policy (trained, ~12+ bytes/param in BF16) and a frozen reference (forward-only, ~2 bytes/param). The value model is gone because GRPO uses a group baseline (chapter 8); the reward model is gone because the verifier is a Python function with *zero* VRAM cost. Those two deletions, roughly, are what buy a 1.5-3B policy enough headroom to train with a real group size $G$ and sequence length on one consumer card. Record the actual resident peak with `torch.cuda.max_memory_allocated()` for your configuration (measured on the baseline machine — record value, date, driver). The through-line of the entire book: the memory you save by making the reward a verifier instead of a network is exactly the memory that lets the loop close on one GPU.
```

## Lab

The goal of this lab is to build the verifier-as-reward-function that the Part VII training loop plugs into, and to demonstrate the pass@k-versus-pass@1 distinction (equation 9.3) that decides whether RL is reshaping or adding capability, using a mock policy whose per-problem success probabilities I control. This is the measurement instrument the thesis loop's step 5 depends on, built and tested in isolation before any GPU is involved, so that when the real run produces pass@k curves I already know how to read them.

This is a `uv` project. From the repo root:

```bash
uv init labs/rlvr-verifier
cd labs/rlvr-verifier
uv add numpy matplotlib
```

```python title="labs/rlvr-verifier/verifier_and_passk.py"
"""RLVR building blocks: a verifiable reward function with the exact signature
GRPOTrainer expects, and the pass@k estimator that distinguishes RL's
distribution-reshaping (Hypothesis A) from added capability (Hypothesis B).
"""
from pathlib import Path
import csv, re
import numpy as np

OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# ---- The verifier: this IS the reward function passed to GRPOTrainer --------
def extract_answer(text: str) -> str | None:
    """Pull the final integer from a '#### 42'-style or last-number response."""
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return m[-1] if m else None

def reward_func(prompts, completions, references, **kwargs) -> list[float]:
    """Binary verifiable reward (eq 9.1). Same code the eval harness scores with."""
    out = []
    for comp, ref in zip(completions, references):
        pred = extract_answer(comp)
        out.append(1.0 if pred is not None and pred == str(ref) else 0.0)
    return out

# ---- Unbiased pass@k estimator (Chen et al. 2021 combinatorial form) ---------
def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """P(at least one of k drawn (without replacement) from n is correct)."""
    if n_samples - n_correct < k:
        return 1.0
    # 1 - C(n-c, k)/C(n, k), computed stably in log space
    from math import comb
    return 1.0 - comb(n_samples - n_correct, k) / comb(n_samples, k)

def main():
    # --- verifier smoke test ---
    comps = ["The answer is #### 42", "so we get 41", "twenty (=20)"]
    refs = [42, 42, 20]
    print("reward_func:", reward_func(None, comps, refs))  # [1.0, 0.0, 0.0]

    # --- pass@k curves for a 'base' vs an 'RL' policy over 50 problems ---
    # Hypothesis A: RL raises pass@1 by concentrating mass, but only on problems
    # the base model could already sometimes solve (p_base > 0). Where p_base=0,
    # RL cannot help (zero gradient on never-sampled solutions).
    n_problems, N = 50, 128
    p_base = rng.beta(1.2, 4.0, size=n_problems)      # many low, few high
    p_base[rng.random(n_problems) < 0.2] = 0.0        # 20% truly unreachable
    # RL sharpens reachable problems toward 1, leaves unreachable at 0.
    p_rl = np.where(p_base > 0, np.minimum(1.0, p_base * 4.0 + 0.15), 0.0)

    def curve(ps, ks):
        cols = {}
        for k in ks:
            vals = []
            for p in ps:
                n_correct = rng.binomial(N, p)
                vals.append(pass_at_k(N, n_correct, k))
            cols[k] = float(np.mean(vals))
        return cols

    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    base_curve, rl_curve = curve(p_base, ks), curve(p_rl, ks)

    csv_path = OUT / "passk.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "base_pass@k", "rl_pass@k"])
        w.writeheader()
        for k in ks:
            w.writerow({"k": k, "base_pass@k": base_curve[k], "rl_pass@k": rl_curve[k]})

    print("\n  k   base   RL")
    for k in ks:
        print(f"{k:>4}  {base_curve[k]:.3f}  {rl_curve[k]:.3f}")
    print(f"\nRL wins at k=1 by {rl_curve[1]-base_curve[1]:+.3f}; "
          f"gap at k=128 is {rl_curve[128]-base_curve[128]:+.3f} "
          f"(converges: reshaping, not new capability)")
    print(f"Artifact: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python verifier_and_passk.py
```

```admonish gotcha
The `pass_at_k` here uses the unbiased combinatorial estimator (draw $k$ without replacement from $N$ samples), not the naive $1-(1-\hat p)^k$ with an empirical $\hat p$, because plugging a noisy point estimate of $p$ into equation (9.3) is biased for small sample counts. Use the combinatorial form whenever you report pass@k from a finite number of generations, which you always are. And note the deliberate 20% of problems with `p_base = 0`: those are the ones RL provably cannot fix on this budget (zero probability means zero gradient), and they are why the base and RL curves *both* plateau below 1.0 at large $k$. If your real run shows the RL model solving problems the base model never touches at large $k$, that is the signature of Hypothesis B (genuinely new capability) and it is a claim worth double-checking, not assuming.
```

**What you should see.** The verifier smoke test prints `[1.0, 0.0, 0.0]`: the first completion's extracted answer matches, the second is wrong, the third parses `20` correctly (the reward function is doing exactly what the eval harness's scorer does). The pass@k table shows the RL policy winning decisively at $k=1$ (much higher pass@1, the reshaping payoff) while the gap *shrinks* as $k$ grows and the two curves converge at $k=128$, the numerical signature of Hypothesis A: RL concentrated mass onto solutions the base model could already sometimes reach, but did not expand the reachable set (both plateau below 1.0 because of the 20% unreachable problems). This is the exact figure the thesis loop's step 5 produces on real data, and reading it correctly, "did pass@1 rise because we reshaped, or because we added capability," is the difference between a defensible thesis claim and an overclaim. The CSV is the artifact you carry into Part VII to compare against a real base-versus-RL pass@k measurement.

```admonish read-along
Read this against **[RLHF]** ch. 7, which covers reinforcement learning with verifiable rewards, the reward-function-versus-reward-model distinction, and the reasoning-model recipes directly, and **[BRM]** ch. 6 for the reasoning-from-verifiable-reward construction in the DeepSeek-R1 lineage. For the pass@k debate, Yue et al. (2025) "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" is the key skeptical result behind Hypothesis A, and Chen et al. (2021) "Evaluating Large Language Models Trained on Code" is the source of the unbiased pass@k estimator in the lab. The Lambert overview and the Tulu/OLMo RLVR work are the applied references for building verifiers at scale.
```

```admonish substack-seed
"Your benchmark and your reward function are the same program." The one-idea version of the whole thesis: when a task is verifiable, the code that scores a model on the test set and the code that hands out reward during RL are literally identical, so building one good eval harness gives you both your metric and your training signal for free. The post can carry a reader from "reward models are neural networks you have to store and that models learn to hack" to "a unit test has no weights and nothing to hack," land the VRAM punchline (deleting the reward model and the value model is exactly what makes reasoning-model RL fit on a gaming GPU), and close on the genuinely open question that keeps it honest: does RL teach the model anything new, or just sharpen what it could already occasionally do? The hook is the collapse of two things everyone treats as separate, measurement and training, into one artifact, with a title that reframes a benchmark as a reward.
```
