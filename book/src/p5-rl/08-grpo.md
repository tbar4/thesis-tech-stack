# GRPO

PPO works, but the last two chapters left it carrying an expensive passenger: the value model. On a 16GB card that critic is the difference between fitting and overflowing, and it exists for one job, turning a single end-of-sequence reward into a dense per-token advantage via GAE. Group Relative Policy Optimization (GRPO), introduced by the DeepSeek team and the workhorse behind DeepSeek-R1, asks a sharp question: what if I get the advantage a different way, from *comparison within a group of samples*, and delete the critic entirely? This chapter derives GRPO straight from the PPO objective, shows precisely which term the group-relative advantage replaces, works through the KL-to-reference regularizer and why the specific "k3" estimator is the right choice, disentangles the token-versus-sequence aggregation question that causes most of the confusion in real implementations, and catalogues the pathologies (length bias, standard-deviation collapse) along with the DAPO-style fixes that patch them. GRPO is the algorithm Part VII actually runs on the baseline machine, so this is the one to get exactly right.

## Theory

### The idea: replace the critic with a group baseline

Go back to the advantage, $A(s,a) = Q(s,a) - V(s)$. The value $V(s)$ is a baseline, an estimate of "how good is this state on average," and PPO learns it with a whole second network. But there is an older, cheaper way to get a baseline that the REINFORCE chapter already flagged: sample several actions from the same state and use their *average* reward as the baseline. If from prompt $q$ I draw a group of $G$ complete responses $\{o_1, \dots, o_G\}$ and score each with reward $r_i$, then the group mean $\bar{r} = \frac{1}{G}\sum_i r_i$ is an empirical, per-prompt estimate of the state value, and $r_i - \bar{r}$ is an empirical advantage: how much better response $i$ was than its peers on the same prompt. No critic, no bootstrap, no GAE. This is the entire conceptual move of GRPO. The critic $V_\phi(s)$ is replaced by a Monte Carlo baseline computed fresh for each prompt from its own group of samples.

GRPO goes one step further and *normalizes* by the group's spread, giving the group-relative advantage

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}, \tag{8.1}$$

which is assigned identically to every token in response $i$ (every token in a sequence shares that sequence's advantage). The mean-subtraction is the variance-reducing baseline; the standard-deviation division is a whitening step that keeps the advantage scale roughly constant across prompts of wildly different difficulty, so the learning rate does not have to absorb the fact that one prompt's rewards live in $[0,1]$ and another's cluster at $0.9$. I will return to that division, because it is also the source of one of GRPO's pathologies.

### Deriving the GRPO objective from PPO

Now put equation (8.1) into the PPO machinery. GRPO keeps PPO's clipped surrogate exactly (chapter 7's equation 7.7) and adds an explicit KL penalty to a frozen reference policy $\pi_{\text{ref}}$, optimizing

$$\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\,\{o_i\}\sim\pi_{\text{old}}}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(&\min\big(r_{i,t}(\theta)\hat{A}_i,\;\text{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_i\big)\\[-2pt]
&-\;\beta\,D_{\text{KL}}\big[\pi_\theta\,\|\,\pi_{\text{ref}}\big]_{i,t}\Big)\Bigg],
\end{aligned} \tag{8.2}$$

where $r_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}\mid q, o_{i,<t})}$ is the same per-token importance ratio as PPO, and $\beta$ weights the KL regularizer.

```admonish derivation title="GRPO as PPO with two substitutions"
Start from PPO's per-token objective (equation 7.7), $\mathbb{E}_t[\min(r_t\hat{A}_t, \text{clip}(r_t,1\pm\epsilon)\hat{A}_t)]$. GRPO makes exactly two changes.

**Substitution 1: the advantage.** PPO computes $\hat{A}_t$ from GAE, which needs $V_\phi$ and the TD residuals $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. In the LLM setting the per-token reward is zero except at the end, so all the interior $\delta$'s are pure value differences, which is the *only* reason a critic was needed (chapter 6). GRPO throws that out and sets $\hat{A}_{i,t} = \hat{A}_i$, the group-relative scalar of equation (8.1), *constant across all tokens $t$ of response $i$*. The per-token structure GAE provided is gone; every token in a good response gets the same positive advantage, every token in a bad one the same negative advantage. This is a coarser credit assignment, and it is the price of dropping the critic, but with a group of samples per prompt the mean-and-std baseline is a low-variance estimate that works well when the reward is a clean end-of-sequence verdict (which is exactly the RLVR setting the thesis lives in).

**Substitution 2: the KL location.** PPO usually folds a per-token KL-to-reference penalty into the *reward* before computing advantages (so the KL flows through GAE). GRPO instead adds the KL as a separate term in the *loss*, equation (8.2)'s $-\beta D_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$, kept out of the advantage entirely. This is cleaner to reason about (the clip governs the trust region against $\pi_{\text{old}}$; the KL governs drift against the frozen $\pi_{\text{ref}}$, two different jobs) and it is where the specific k3 estimator enters, next.

That is the whole derivation. GRPO is PPO with the advantage sourced from a group mean/std instead of a value network, and the reference-KL moved from the reward into the loss. Everything else, the clip, the ratio, the multiple epochs per rollout, is inherited unchanged.
```

The payoff is immediate and is the reason GRPO exists for people on small hardware: **no value model.** The critic's weights, gradients, and optimizer state, the single most expensive optional component of PPO (chapter 6's vram-budget), are simply gone. What remains is the policy (trained), a frozen reference (forward-only), and, in the RLVR setting, a verifier that is a Python function rather than a reward network. That is the configuration that fits a 1.5-3B policy comfortably on the RTX 5080.

### The KL estimators: k1, k2, k3

The KL term needs care because I estimate it from samples, not in closed form, and the naive estimator is bad. I am estimating $D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \mathbb{E}_{x\sim\pi_\theta}[\log\frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)}]$ using samples $x$ drawn from $\pi_\theta$ (the tokens I actually generated). Define the likelihood ratio $\rho = \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)}$ (reference over current, note the direction). Schulman's three estimators are:

$$k_1 = -\log\rho = \log\frac{\pi_\theta}{\pi_{\text{ref}}}, \qquad k_2 = \tfrac{1}{2}(\log\rho)^2, \qquad k_3 = \rho - 1 - \log\rho. \tag{8.3}$$

```admonish derivation title="Why k3 and not k1 or k2"
All three are estimators of the same KL, but they differ in bias and variance, and k3 is the one GRPO uses. Work through the three.

**k1 is unbiased but high-variance.** By definition $\mathbb{E}_{x\sim\pi_\theta}[-\log\rho] = \mathbb{E}[\log\frac{\pi_\theta}{\pi_{\text{ref}}}] = D_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$ exactly, so k1 is unbiased. But KL is a nonnegative quantity, and $k_1 = -\log\rho$ is negative whenever $\pi_{\text{ref}}(x) > \pi_\theta(x)$, which happens for plenty of individual samples. So a single-sample k1 estimate is frequently negative, which is nonsense for a divergence, and its variance is large. Optimizing against a term that is negative half the time injects noise.

**k2 is low-variance but biased.** $k_2 = \tfrac12(\log\rho)^2$ is always nonnegative (good) and low-variance, and it happens to be a decent approximation because for distributions that are close, $D_{\text{KL}} \approx \tfrac12\mathbb{E}[(\log\rho)^2]$ to second order. But it is *biased*: its expectation is not the KL except in the limit of identical distributions, and the bias grows as $\pi_\theta$ drifts from $\pi_{\text{ref}}$, which is exactly when you care.

**k3 is unbiased AND nonnegative.** Here is the construction. Start from the identity that for the true KL, $\mathbb{E}_{x\sim\pi_\theta}[\rho] = \mathbb{E}[\frac{\pi_{\text{ref}}}{\pi_\theta}] = \sum_x \pi_\theta(x)\frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)} = \sum_x \pi_{\text{ref}}(x) = 1$. So $\mathbb{E}[\rho - 1] = 0$: the quantity $\rho - 1$ has zero mean and I can add any multiple of it to k1 without changing the expectation. Choose the multiple that makes the result nonnegative:

$$k_3 = (\rho - 1) + (-\log\rho) = \rho - 1 - \log\rho.$$

Its expectation is $\mathbb{E}[\rho - 1] + \mathbb{E}[-\log\rho] = 0 + D_{\text{KL}} = D_{\text{KL}}$, so it is **unbiased**. And it is **always nonnegative**, because $f(\rho) = \rho - 1 - \log\rho$ has $f'(\rho) = 1 - 1/\rho$ (zero at $\rho=1$), $f''(\rho) = 1/\rho^2 > 0$ (convex), so its global minimum is $f(1) = 0$; for all $\rho > 0$, $\rho - 1 \ge \log\rho$, hence $k_3 \ge 0$. It is a convex, always-positive, unbiased, low-variance estimator, the best of k1 and k2 at once. That is why equation (8.2)'s KL term is implemented as $k_3 = \rho - 1 - \log\rho$ with $\rho = \pi_{\text{ref}}/\pi_\theta$ evaluated per token.
```

A useful sanity check that the k3 gradient behaves: even though k3 is a nonlinear function of $\rho$, its gradient with respect to $\theta$ points in the KL-reducing direction, and near $\rho = 1$ it reduces to the familiar penalty. The practical note is just that you must get the *direction* of $\rho$ right ($\pi_{\text{ref}}$ over $\pi_\theta$); flip it and the estimator is no longer the KL you want.

### Token versus sequence aggregation

Equation (8.2) as written normalizes each response by its own length $|o_i|$ (the inner $\frac{1}{|o_i|}\sum_t$) and then averages over the $G$ responses. This is a *choice*, and it is one of the most consequential and most confused details in GRPO. There are two axes: how to reduce over tokens within a response, and how to weight responses of different lengths against each other.

```admonish derivation title="Length normalization and where the bias hides"
Write the per-response loss contribution generically as $\frac{1}{Z_i}\sum_{t=1}^{|o_i|} \ell_{i,t}$, where $\ell_{i,t}$ is the per-token clipped objective and $Z_i$ is a normalizer. Two natural choices:

**Per-sequence mean ($Z_i = |o_i|$).** This is the original GRPO of equation (8.2). Each response contributes the *average* per-token loss, so every response counts equally regardless of length. But look at what it does to gradients: a token in a short response is divided by a small $|o_i|$ and a token in a long response by a large one, so **individual tokens in long responses get smaller gradients**. When the advantage $\hat{A}_i$ is negative (a wrong answer), this means long wrong answers are penalized *less per token* than short wrong answers. The optimizer discovers it can dilute penalty by being verbose, and length creeps up. This is the notorious GRPO **length bias**, and it is baked into the per-sequence normalizer.

**Per-token mean over the whole batch ($Z = \sum_i |o_i|$, one shared denominator).** Sum every token's loss across the whole group and divide by the total token count. Now every token carries equal weight regardless of which response it lived in, so a long wrong answer accrues penalty proportional to its length, killing the verbosity incentive. This is the DAPO / Dr. GRPO fix: move the normalization from per-sequence to per-token-over-the-batch. The tradeoff is that long responses now dominate the loss in proportion to their length, which is usually what you want for reasoning (longer correct chains should get more total credit) but changes the effective learning dynamics, so it interacts with the learning rate.
```

The one-line rule I use: per-sequence normalization treats *responses* as the unit and quietly rewards length; per-token (batch-level) normalization treats *tokens* as the unit and removes that particular bias. For reasoning-model training the per-token variant is now the common default, and I will use it in the Part VII lab.

### Pathologies and the DAPO-style fixes

Two failure modes recur, both traceable to specific terms above.

**Length bias**, just derived, from per-sequence normalization. Fix: per-token (batch-level) loss normalization, and DAPO additionally adds an explicit "overlong" soft penalty that shapes the reward down for responses that run past a length budget, so the model is discouraged from rambling to game the normalizer.

**Standard-deviation collapse**, from the division in equation (8.1). When every response in a group gets the *same* reward (all correct, or all wrong, which is extremely common with binary verifiable rewards on easy or impossible prompts), the group std is zero and equation (8.1) is $0/0$. Implementations add an $\epsilon$ to the denominator, so the advantage becomes zero and the prompt contributes *no gradient at all*. That is a silent waste: a large fraction of your batch can be prompts the model already fully solves or cannot touch, contributing nothing, and it also introduces a subtle scale artifact because prompts that happen to have small nonzero std get *amplified* advantages (dividing by a tiny number), overweighting near-degenerate prompts. Two fixes: **Dr. GRPO** argues for dropping the std division entirely (use only $r_i - \bar{r}$, an unnormalized advantage, removing the amplification artifact), and **DAPO's dynamic sampling** filters out prompts whose group is all-correct or all-wrong and resamples until the batch is full of prompts with reward variance, so every gradient step spends its compute on prompts that actually carry signal.

**Clip-higher**, DAPO's third tweak, addresses entropy collapse. Recall from chapter 7 that PPO's clip is symmetric, $[1-\epsilon, 1+\epsilon]$. DAPO decouples the two sides, $[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$ with $\epsilon_{\text{high}} > \epsilon_{\text{low}}$ (for example $0.28$ versus $0.20$). The rationale falls straight out of the chapter-7 piecewise analysis: for a good, currently-low-probability token, the upper clip at $1+\epsilon$ caps how fast its probability can rise, which throttles exploration of promising-but-rare tokens and lets the policy collapse onto a few high-probability moves. Raising the *upper* bound alone lets rare good tokens climb faster (more exploration) without loosening the lower bound that guards against over-suppression. It is a targeted widening of exactly one arm of the clip geometry.

```admonish gotcha
The most expensive silent bug in a from-scratch GRPO run is std collapse eating your batch. With a binary verifiable reward and a policy that already solves, say, 60% of the easy prompts, a large chunk of every group comes back all-1 or all-0, contributes zero advantage, and your effective batch size is a fraction of what you think it is; loss curves look calm while learning crawls. Always log the fraction of prompts with nonzero reward std per step. If it is low, you are paying full generation cost for a handful of useful gradients, and DAPO's dynamic sampling (resample until the group has reward variance) is not a nice-to-have, it is the difference between the run learning and stalling. This is measured behavior on your data (measured on the baseline machine — record the nonzero-std fraction, date, driver).
```

## Tooling

The tool is TRL's `GRPOTrainer` (and its close cousins in verl and OpenRLHF). It embodies equation (8.2) directly: for each prompt it generates $G$ completions (the `num_generations` argument, typically 4 to 16), scores them with a user-supplied `reward_funcs` list (which for RLVR is just Python functions returning floats, no reward model in VRAM), computes the group mean and std to get $\hat{A}_i$ per equation (8.1), and optimizes the clipped-plus-KL loss. The knobs map one-to-one onto the theory: `beta` is the KL coefficient $\beta$, `epsilon` and (in newer versions) `epsilon_high` are the clip bounds including DAPO's clip-higher, `loss_type` selects the aggregation (`"grpo"` for per-sequence, `"dr_grpo"` or `"bnpo"`-style for the per-token batch normalization that removes length bias), and `scale_rewards` toggles the std division from equation (8.1) so you can run the Dr. GRPO unnormalized variant.

```admonish under-the-hood
GRPO's compute profile is *generation-bound*, harder than PPO's, because it needs $G$ samples per prompt instead of one. That is the real cost of dropping the critic: you trade the value network's memory for more rollouts. On the RTX 5080 this is exactly why the serve step of the thesis loop uses vLLM rather than naive `model.generate`, since producing $G=8$ completions per prompt for a batch of prompts is the throughput bottleneck of the entire training step, and vLLM's paged-attention batching is what makes it tractable. The advantage computation itself (equation 8.1) is a trivial mean and std over $G$ scalars per prompt, microseconds, so essentially all the wall-clock time is generation plus the backward pass; the group baseline is free. Weight sync between the training copy and the vLLM inference copy after each update is the other moving part, and on one GPU that means the served policy and the trained policy are the same weights, updated in place.
```

```admonish vram-budget
This is where GRPO earns its place in the thesis. On the baseline machine (RTX 5080 16GB, Blackwell), the resident set for a GRPO run on a 1.5-3B policy is: the policy (weights + grads + optimizer state, ~12+ bytes/param in BF16 mixed precision), a frozen reference policy for the k3 KL (weights only, forward in `no_grad`, ~2 bytes/param), and *no value model and no reward model*, because the advantage is the group baseline and the reward is a verifier function. Compared to PPO's four-model burden (chapter 7), GRPO carries roughly two, and the two it drops (critic and reward model) are the two that most often push a small-hardware run over the 16GB edge. The one remaining tension is the group size $G$: larger $G$ gives a lower-variance baseline but multiplies the activation memory and KV-cache of the generation step, so on 16GB there is a real ceiling on $G \times$ (max sequence length) that you tune empirically. Record the peak with `torch.cuda.max_memory_allocated()` for your $G$ and policy size (measured on the baseline machine — record value, date, driver).
```

## Lab

The goal of this lab is to compute the two things GRPO adds to PPO, the group-relative advantage (equation 8.1) and the three KL estimators (equation 8.3), on synthetic data where I know the answer, and to *watch k3 beat k1 and k2* on the bias-and-variance criterion the derivation claimed. This isolates the two novel pieces so that when the full GRPO run in Part VII uses them, they are already understood in miniature. Pure arithmetic, no GPU, no model.

This is a `uv` project. From the repo root:

```bash
uv init labs/grpo-pieces
cd labs/grpo-pieces
uv add numpy matplotlib
```

```python title="labs/grpo-pieces/grpo_pieces.py"
"""The two novel pieces of GRPO, in isolation:
  (1) group-relative advantage (eq 8.1) over G samples per prompt, and
  (2) the k1/k2/k3 KL estimators (eq 8.3), comparing bias and variance
      against the exact KL between two known categorical distributions.
"""
from pathlib import Path
import csv
import numpy as np

OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# ---- Piece 1: group-relative advantage --------------------------------------
def group_advantage(rewards, scale_by_std=True, eps=1e-6):
    mean = rewards.mean()
    if scale_by_std:
        return (rewards - mean) / (rewards.std() + eps)
    return rewards - mean            # Dr. GRPO variant (no std division)

# ---- Piece 2: KL estimators over a known pair of distributions --------------
def kl_exact(p, q):
    return float(np.sum(p * np.log(p / q)))

def kl_estimators(p, q, n_samples):
    """Sample x ~ p (= pi_theta), form rho = q(x)/p(x) = pi_ref/pi_theta."""
    idx = rng.choice(len(p), size=n_samples, p=p)
    rho = q[idx] / p[idx]
    logrho = np.log(rho)
    k1 = -logrho                      # unbiased, high variance, can be < 0
    k2 = 0.5 * logrho**2             # biased, nonnegative
    k3 = rho - 1.0 - logrho          # unbiased, nonnegative
    return k1, k2, k3

def main():
    # --- advantage demo: a group where 2 of 6 responses are correct ---
    rewards = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    print("rewards:", rewards.tolist())
    print("A (std-normalized):", np.round(group_advantage(rewards), 3).tolist())
    print("A (Dr.GRPO, no std):", np.round(group_advantage(rewards, False), 3).tolist())
    # std-collapse case: every response identical
    flat = np.ones(6)
    print("A when all-correct (std collapse):",
          np.round(group_advantage(flat), 3).tolist(), "-> zero gradient")

    # --- KL estimator comparison ---
    p = np.array([0.4, 0.3, 0.2, 0.1])          # pi_theta
    q = np.array([0.1, 0.2, 0.3, 0.4])          # pi_ref (deliberately far)
    true_kl = kl_exact(p, q)
    print(f"\nExact KL[pi_theta || pi_ref] = {true_kl:.4f}")

    rows = []
    for n in (1, 4, 16, 64, 256):
        trials = 20000 // n
        m1 = np.array([kl_estimators(p, q, n)[0].mean() for _ in range(trials)])
        m2 = np.array([kl_estimators(p, q, n)[1].mean() for _ in range(trials)])
        m3 = np.array([kl_estimators(p, q, n)[2].mean() for _ in range(trials)])
        for name, m in (("k1", m1), ("k2", m2), ("k3", m3)):
            rows.append({"n": n, "estimator": name,
                         "mean": m.mean(), "bias": m.mean() - true_kl,
                         "std": m.std(),
                         "frac_negative": float((m < 0).mean())})

    csv_path = OUT / "kl_estimators.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n","estimator","mean","bias","std","frac_negative"])
        w.writeheader(); w.writerows(rows)

    print("\n n  est   bias      std     frac<0")
    for r in rows:
        print(f"{r['n']:>3} {r['estimator']}  {r['bias']:+.4f}  "
              f"{r['std']:.4f}  {r['frac_negative']:.2f}")
    print(f"\nArtifact: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python grpo_pieces.py
```

```admonish gotcha
Watch the `frac_negative` column for k1: at $n=1$ a large fraction of single-sample k1 estimates are *negative*, which is the concrete face of "k1 is unbiased but noisy and violates the nonnegativity of KL." k3's `frac_negative` is exactly 0 at every $n$ (it is nonnegative by construction, the convexity argument in the derivation), and its `bias` stays near zero at every sample count while k2's `bias` is visibly nonzero and does not vanish as $n$ grows (bias is not a variance problem, so more samples do not fix it). That three-column contrast, k1 noisy and sign-violating, k2 biased, k3 clean on both, is the entire case for equation (8.3)'s choice, made numerically.
```

**What you should see.** For the advantage: the std-normalized advantages for `[1,0,0,1,0,0]` come out symmetric around zero with the two correct responses getting equal positive values and the four wrong ones equal negative values, the Dr. GRPO variant gives the same signs at a different (unnormalized) scale, and the all-correct group prints all-zeros, the std-collapse case that contributes no gradient. For the KL estimators: k3's bias column hovers at essentially zero for every $n$ while its std shrinks as $n$ grows (unbiased, consistent), k1 shares the near-zero bias but with larger std and a nonzero `frac_negative` at small $n$, and k2 shows a persistent nonzero bias that more samples never remove. The CSV lets you plot bias-versus-$n$ for all three and see k2's flat bias floor against k1 and k3 converging to the true KL. This is the miniature of the two pieces the Part VII GRPO run assembles into a full training step.

```admonish read-along
Read this against **[BRM]** ch. 6-7, which build GRPO for reasoning models with the token-level implementation details and walk the DeepSeek-R1 recipe this chapter derives, and **[RLHF]** ch. 6 for how GRPO sits in the broader policy-optimization family alongside PPO. The primary sources are Shao et al. (2024) "DeepSeekMath" (which introduces GRPO, equation 8.2), the DeepSeek-R1 report (2025) for the reasoning application, Yu et al. (2025) "DAPO" for clip-higher and dynamic sampling, and Liu et al. (2025) "Dr. GRPO" for the length-bias and std-division analysis. Schulman's "Approximating KL Divergence" note is the source for the k1/k2/k3 estimators of equation (8.3).
```

```admonish substack-seed
"How to do PPO without the second neural network." GRPO's whole trick, told in one paragraph: PPO learns a value network just to compute a baseline, and that network is the thing most likely to blow your VRAM budget on a single GPU, so GRPO deletes it and gets the baseline for free by sampling a group of answers to the same prompt and comparing them to their own average. The post can carry the reader from "advantage is reward minus a baseline" through "the group mean *is* a baseline" to the two twists that make it work in practice, the k3 KL estimator (why the obvious estimator of KL is negative half the time and how a one-line convexity fix repairs it) and the length bias hiding in a denominator. The hook: the algorithm behind DeepSeek-R1 is, at its core, PPO with the expensive part replaced by "sample eight answers and grade on a curve."
```
