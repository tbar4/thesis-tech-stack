# REINFORCE and variance reduction

The policy gradient theorem of Chapter 5.4 gave me an exact expression for $\nabla_\theta J$, but it is an expectation, and I cannot evaluate it in closed form for anything real. I have to *estimate* it by sampling. The moment you estimate a gradient by sampling, you inherit a new adversary that has nothing to do with bias and everything to do with noise: variance. REINFORCE is the most direct Monte Carlo estimator of the policy gradient, and left raw it is so high-variance as to be nearly useless. This chapter derives REINFORCE, then derives the three standard variance-reduction moves, reward-to-go, baselines, and the advantage, proving that each one leaves the gradient *unbiased* while shrinking its variance. The baseline unbiasedness proof is the theoretical seed of GRPO, so I give it in full. The lab is the payoff of the whole part: a REINFORCE agent on a toy task with a matplotlib figure showing the gradient-variance curve collapse the instant a baseline is switched on.

## Theory

### REINFORCE: Monte Carlo policy gradient

Start from the trajectory form of the theorem, Chapter 5.4 Equation (5). Sample $N$ trajectories $\tau^{(1)},\dots,\tau^{(N)}$ from the current policy and replace the expectation with its sample mean:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\left(\sum_{t=0}^{T_i-1}\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\right)R(\tau^{(i)}). \tag{1}
$$

This is REINFORCE (Williams, 1992). The update rule is gradient ascent, $\theta \leftarrow \theta + \alpha\,\widehat{\nabla_\theta J}$. In words: for each sampled trajectory, push up the log-probability of every action it took, scaled by the trajectory's total return. Good trajectories make all their actions more likely; bad trajectories make all their actions less likely. It is beautifully simple and, as stated, badly behaved, for a reason I want to make precise before fixing it.

The trouble is that the single scalar $R(\tau)$ multiplies the log-prob gradient of *every* action in the trajectory, including actions that had nothing to do with the outcome. If one lucky reward inflates $R(\tau)$, REINFORCE dutifully reinforces every token that happened to be in that sequence, noise and signal alike. The estimator is unbiased, but its variance grows with trajectory length and with the scale of the returns, and for a language model with hundreds of tokens per episode that variance is enormous.

### Reward-to-go: the first free variance cut

The first fix follows from causality: **an action cannot influence rewards that were already collected before it was taken.** So weighting $\nabla\log\pi_\theta(a_t\mid s_t)$ by the *full* return $R(\tau)$, which includes rewards from steps $0,\dots,t-1$, is adding pure noise. Replace the full return with the *reward-to-go*, the return accumulated from step $t$ onward.

```admonish derivation title="Reward-to-go is unbiased"
Let $G_t = \sum_{k=t}^{T-1}\gamma^{k-t} r_{k+1}$ be the reward-to-go from step $t$. I claim the past rewards contribute zero to the gradient, so dropping them changes nothing in expectation. Consider a single past reward $r_{j+1}$ with $j < t$ and the action term at step $t$. Its contribution to the gradient is

$$
\mathbb{E}\big[\, \nabla_\theta\log\pi_\theta(a_t\mid s_t)\; r_{j+1} \,\big].
$$

Condition on everything up to and including step $j+1$ (which fixes $r_{j+1}$ and the state $s_t$). Inside that conditioning, $r_{j+1}$ is a constant, and $a_t$ is still drawn fresh from $\pi_\theta(\cdot\mid s_t)$. The expectation of the score function under its own distribution is zero, a fact I prove in the next box:

$$
\mathbb{E}_{a_t\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big] = 0.
$$

By the tower property, the whole term vanishes. Since this holds for every past reward $r_{j+1}$ with $j<t$, replacing $R(\tau)$ by the reward-to-go $G_t$ at each step leaves the gradient unbiased:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, G_t\right]. \tag{2}
$$

We removed variance (the noisy past rewards) at zero cost in bias. That is the ideal kind of trade, and it is worth internalizing that for a *terminal-reward* LLM task with $\gamma=1$, the reward-to-go $G_t$ equals the single final reward $R_T$ for every $t$, so reward-to-go and full-return coincide and this particular cut buys nothing. It buys a great deal in dense-reward or long-horizon tasks, which is why I keep it in the general form.
```

### The score function integrates to zero

The identity I just used is the linchpin of every variance-reduction result in RL, so I prove it standalone.

```admonish derivation title="The expected score is zero"
For any state $s$ and any $\theta$,

$$
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\big]
= \sum_a \pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s).
$$

Apply the log-derivative trick $\pi_\theta\,\nabla\log\pi_\theta = \nabla\pi_\theta$ (Chapter 5.4, Eq. 4) to each term:

$$
= \sum_a \nabla_\theta\pi_\theta(a\mid s)
= \nabla_\theta \sum_a \pi_\theta(a\mid s)
= \nabla_\theta\, 1 = 0. \tag{3}
$$

The gradient and the sum swap because the action set is finite and fixed. The last step is the punchline: probabilities sum to one, a constant, whose gradient is zero. This single fact, that the policy is *normalized*, is why you can subtract things from the return without biasing the gradient. Hold onto it for the baseline proof.
```

### Baselines: subtracting a number for free

Reward-to-go still leaves the returns on an arbitrary scale. If every completion for a prompt scores between $+8$ and $+10$, REINFORCE reinforces *all* of them strongly (every $G_t$ is large and positive), even though relative to each other some are worse. What matters for learning is not the absolute return but how a return compares to what you *expected*. So subtract a *baseline* $b(s_t)$, a number that can depend on the state but not on the action taken:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\right]. \tag{4}
$$

The remarkable fact is that *any* action-independent baseline leaves the gradient exactly unbiased, so you may choose $b$ purely to minimize variance.

```admonish derivation title="Baselines are unbiased (the GRPO seed)"
I need to show the baseline contributes zero to the gradient in expectation, i.e.

$$
\mathbb{E}\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\, b(s_t)\right] = 0.
$$

Look at a single step $t$ and condition on the state $s_t$ (which fixes $b(s_t)$, since the baseline depends only on the state). Pull the constant $b(s_t)$ out of the inner expectation over the action:

$$
\mathbb{E}_{s_t}\Big[\, b(s_t)\;\mathbb{E}_{a_t\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big] \,\Big].
$$

By Equation (3), the inner expectation is zero. So the whole term is $\mathbb{E}_{s_t}[\,b(s_t)\cdot 0\,] = 0$, for every $t$. Summing over $t$, the baseline's total contribution is zero, hence (4) is unbiased for *any* function $b(s_t)$ that does not depend on the action. $\qquad\blacksquare$

**This is the theorem GRPO stands on.** GRPO samples a group of $K$ completions for one prompt, computes each completion's reward $R_k$, and uses the group mean $\bar{R} = \frac{1}{K}\sum_k R_k$ as the baseline. Because $\bar R$ depends only on the prompt state, not on which specific completion you are updating, the proof above says subtracting it is unbiased. The group-relative advantage $R_k - \bar R$ (usually also divided by the group's standard deviation) is precisely a baseline-subtracted return, and no learned value network is anywhere in sight. Everything special about GRPO is contained in this box plus the choice "let the baseline be the group mean."
```

### Why a baseline reduces variance, and the optimal choice

Unbiasedness says the baseline does not hurt; now I show a good baseline helps. Write the per-step estimator as $g(a) = \nabla_\theta\log\pi_\theta(a\mid s)\,(G - b)$. Its variance, for a scalar parameter to keep the algebra clean, is $\mathrm{Var}[g] = \mathbb{E}[g^2] - (\mathbb{E}[g])^2$, and the second term is fixed (it is the true gradient squared, baseline-independent by the proof above). So minimizing variance means minimizing $\mathbb{E}[g^2]$.

```admonish derivation title="The variance-minimizing baseline"
Let $\ell = \nabla_\theta\log\pi_\theta(a\mid s)$. Minimize over $b$:

$$
\mathbb{E}\big[\ell^2 (G-b)^2\big] = \mathbb{E}[\ell^2 G^2] - 2b\,\mathbb{E}[\ell^2 G] + b^2\,\mathbb{E}[\ell^2].
$$

This is a quadratic in $b$. Differentiate and set to zero:

$$
\frac{d}{db}\Big(-2b\,\mathbb{E}[\ell^2 G] + b^2\,\mathbb{E}[\ell^2]\Big) = -2\,\mathbb{E}[\ell^2 G] + 2b\,\mathbb{E}[\ell^2] = 0
\;\Longrightarrow\;
b^\star = \frac{\mathbb{E}[\ell^2 G]}{\mathbb{E}[\ell^2]}. \tag{5}
$$

The optimal baseline is a $\ell^2$-weighted average of the return, which is *close to* the expected return $\mathbb{E}[G] = v_\pi(s)$ but weighted by the squared score. In practice people use the plainer choice $b(s) = v_\pi(s)$, the value function, because it is intuitive ("compare the return to what you expected from this state") and nearly optimal. That single substitution, $b(s) = v_\pi(s)$, is what turns REINFORCE into an actor-critic method (Chapter 5.6): the "critic" is just a learned estimate of the variance-reducing baseline.
```

### Advantage as the general form

Put reward-to-go and a value baseline together. With $b(s_t) = v_\pi(s_t)$ and recalling that $\mathbb{E}[G_t\mid s_t, a_t] = q_\pi(s_t,a_t)$, the weight on the log-prob gradient becomes, in expectation,

$$
G_t - v_\pi(s_t) \;\longrightarrow\; q_\pi(s_t,a_t) - v_\pi(s_t) \;=\; A_\pi(s_t,a_t), \tag{6}
$$

the *advantage* from Chapter 5.2. The advantage is the general form of the policy gradient weight, and every method in the rest of Part V is a different way of estimating $A_\pi$:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\, A_\pi(s_t,a_t)\right]. \tag{7}
$$

- **REINFORCE with baseline** estimates $A$ by $G_t - b$ with $b$ a running average or the value function.
- **Actor-critic and GAE** (5.6) estimate $A$ with a learned critic and bootstrapping.
- **PPO** (5.7) uses the same $A$ but clips the update to stay in a trust region.
- **GRPO** (5.8) estimates $A$ by $(R_k - \bar R)/\mathrm{std}(R)$ across a group, no critic at all.

Seen this way, the whole progression from REINFORCE to GRPO is one question asked repeatedly: *what is the cheapest low-variance unbiased estimate of the advantage that fits in 16 GB?*

```admonish read-along title="Read-along: Sutton & Barto Chapter 13; Raschka BRM Chapter 6"
[S&B] Section 13.3 is REINFORCE, my (1) and (2); Section 13.4 adds the baseline and states the unbiasedness result, my (4) and the proof box; Section 13.5 is REINFORCE-with-baseline as a stepping stone to actor-critic, my (6). Their Figure 13.2 shows the variance reduction empirically, which is exactly what the lab below reproduces from scratch. For the LLM translation, [BRM] Chapter 6 sidebars connect the advantage form (7) to how reasoning-model training frames the reward-minus-baseline signal per completion; read those sidebars after the lab, when the group-mean-baseline idea from the GRPO seed box is fresh, and the connection to $(R_k-\bar R)$ will click.
```

## Tooling

The tooling is `matplotlib` for the artifact plus NumPy for the agent, both already in the uv project from Chapter 5.1. I deliberately avoid a deep-learning framework here so that every line of the gradient estimator is visible and the variance measurement is unambiguous. The one methodological tool worth naming is *measuring* gradient variance directly: draw many independent gradient estimates at a fixed $\theta$ and take the variance across them, rather than trusting the loss curve to tell you whether your estimator is noisy. That measurement is the figure.

## Lab

The task is a contextual bandit that stands in for one-step LLM generation: a "prompt" (context) is one of a few categories, the "action" is one of several tokens, and a verifier rewards the correct token for each context. I train a softmax policy with REINFORCE, once with no baseline and once with a running-mean baseline, and at each step I estimate the *variance* of the gradient across a fresh minibatch. The artifact is a matplotlib figure with two panels: reward-over-time (both variants learn) and gradient-variance-over-time (the baseline variant is dramatically lower). That variance gap, at equal final reward, is the entire lesson of the chapter made visual.

```python title="labs/reinforce_variance.py"
"""REINFORCE on a contextual bandit (a stand-in for one-step LLM
generation), with and without a baseline. Measure the variance of the
gradient estimator at each step and plot it. Artifact: reinforce_variance.png

Context = "prompt" category. Action = "token". Reward = 1 if the action
is the correct token for that context, else 0 (a toy verifier).

Run:  uv run python labs/reinforce_variance.py
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless: write a file, no display
import matplotlib.pyplot as plt

N_CONTEXTS = 4
N_ACTIONS = 5
CORRECT = np.array([0, 2, 4, 1])          # the "right token" per context
STEPS = 400
BATCH = 32                                 # completions sampled per step
LR = 0.5
SEED = 0


def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reward(context, action):
    return 1.0 if action == CORRECT[context] else 0.0


def grad_log_pi(probs_row, action):
    """d/dlogits log pi(action) for a softmax = onehot(action) - probs."""
    g = -probs_row.copy()
    g[action] += 1.0
    return g


def train(use_baseline: bool, rng):
    # Policy parameters: one logit vector per context.
    theta = np.zeros((N_CONTEXTS, N_ACTIONS))
    baseline = np.zeros(N_CONTEXTS)        # running mean reward per context
    rewards_hist, gradvar_hist = [], []

    for step in range(STEPS):
        probs = softmax_rows(theta)
        # Sample a batch of (context, action, reward).
        contexts = rng.integers(N_CONTEXTS, size=BATCH)
        batch_reward = 0.0
        # Accumulate per-sample gradients so we can measure their spread.
        per_sample_grads = np.zeros((BATCH, N_CONTEXTS * N_ACTIONS))
        grad_sum = np.zeros_like(theta)

        for i, c in enumerate(contexts):
            a = rng.choice(N_ACTIONS, p=probs[c])
            r = reward(c, a)
            batch_reward += r
            advantage = r - (baseline[c] if use_baseline else 0.0)
            g = np.zeros_like(theta)
            g[c] = advantage * grad_log_pi(probs[c], a)
            grad_sum += g
            per_sample_grads[i] = g.ravel()
            if use_baseline:
                # Update running-mean baseline for this context (eq: b ~ E[G]).
                baseline[c] += 0.05 * (r - baseline[c])

        rewards_hist.append(batch_reward / BATCH)
        # Variance of the gradient estimator: trace of the covariance of
        # the per-sample gradients (total variance across all components).
        gradvar_hist.append(float(per_sample_grads.var(axis=0).sum()))
        # Gradient ASCENT on J.
        theta += LR * grad_sum / BATCH

    return np.array(rewards_hist), np.array(gradvar_hist)


def smooth(x, k=15):
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    r_no, v_no = train(use_baseline=False, rng=rng)
    rng = np.random.default_rng(SEED)        # same seed: fair comparison
    r_yes, v_yes = train(use_baseline=True, rng=rng)

    print(f"final reward  (no baseline): {r_no[-50:].mean():.3f}")
    print(f"final reward  (baseline)   : {r_yes[-50:].mean():.3f}")
    print(f"mean grad-var (no baseline): {v_no.mean():.4f}")
    print(f"mean grad-var (baseline)   : {v_yes.mean():.4f}")
    print(f"variance reduction factor  : {v_no.mean() / max(v_yes.mean(), 1e-9):.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(r_no, alpha=0.35, color="tab:red")
    ax1.plot(r_yes, alpha=0.35, color="tab:blue")
    ax1.plot(range(len(smooth(r_no))), smooth(r_no), color="tab:red",
             label="no baseline")
    ax1.plot(range(len(smooth(r_yes))), smooth(r_yes), color="tab:blue",
             label="with baseline")
    ax1.set_title("Reward over training (both learn)")
    ax1.set_xlabel("step"); ax1.set_ylabel("mean batch reward"); ax1.legend()

    ax2.plot(v_no, color="tab:red", alpha=0.8, label="no baseline")
    ax2.plot(v_yes, color="tab:blue", alpha=0.8, label="with baseline")
    ax2.set_title("Gradient-estimator variance (lower is better)")
    ax2.set_xlabel("step"); ax2.set_ylabel("trace of grad covariance")
    ax2.set_yscale("log"); ax2.legend()

    fig.tight_layout()
    fig.savefig("reinforce_variance.png", dpi=130)
    print("\nwrote reinforce_variance.png")
```

**What you should see.** The printed summary reports that both variants reach essentially the same final reward (both solve the toy task, climbing toward a mean batch reward near $1.0$), while the mean gradient variance with a baseline is several times lower than without, a variance-reduction factor typically in the range of roughly 2x to 5x on this problem with these seeds. The saved figure `reinforce_variance.png` makes it unmistakable: the left panel shows two reward curves rising to the same ceiling, so the baseline costs nothing in final performance, and the right panel, on a log scale, shows the blue (baseline) variance curve sitting well below the red (no-baseline) one for the whole run. That separation at equal reward is the empirical face of the unbiasedness proof: the baseline moved variance, not bias. This is the exact mechanism GRPO scales up, swapping my per-context running-mean baseline for a per-prompt group-mean baseline, and it is why a critic-free method can train a reasoning model on a single RTX 5080 16GB (when you run the real GRPO loop in Part VII, log this same gradient-variance and entropy pair: measured on the baseline machine — record value, date, driver).

```admonish gotcha title="A baseline that depends on the action is a bug, not a feature"
The unbiasedness proof hinges on the baseline being independent of the action taken. It is tempting, and wrong, to peek at the sampled action when computing the baseline, for instance by using that completion's own reward as its own baseline, which makes the advantage identically zero and kills learning. GRPO's group mean is safe precisely because it averages over the whole group, so from any single completion's perspective the baseline is an action-independent constant (to leading order). If you ever find your advantages mysteriously collapsing to zero or your gradient going biased, check first whether your baseline has secretly become a function of the action you are updating.
```

```admonish substack-seed
Post angle: "Why grading on a curve is the trick that makes AI training work." REINFORCE, the foundational recipe for learning from trial and error, has a crippling flaw: it is so noisy that it barely learns, because it can't tell a genuinely good attempt from a lucky one. The fix is almost philosophical: don't reward an attempt for its raw score, reward it for beating expectations, for how much better it did than the average attempt at the same problem. That's grading on a curve, and there's a clean proof that it speeds up learning without ever distorting *what* the model learns, only how quickly. The essay lands on the fact that this exact "compare each answer to the group average" move is the beating heart of GRPO, the algorithm behind today's open reasoning models, and that you can watch the noise drop in a chart you can generate on a laptop in under a second.
```
