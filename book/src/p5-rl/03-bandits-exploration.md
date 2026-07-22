# Bandits, exploration, and sampling

Strip an MDP down to a single state and you get a *bandit*: no transitions to reason about, no discounting, just the purest form of the one dilemma that never goes away in reinforcement learning. Do I take the action that looks best right now, or do I try something else to learn whether "best" is really best? That is exploration versus exploitation, and it is not a footnote. It is the reason temperature exists as a decoding knob, it is the reason RL fine-tuning can quietly strangle itself through "entropy collapse," and it is why every serious RLVR recipe carries some term whose only job is to keep the policy from getting too sure of itself too soon. This chapter derives the bandit machinery lightly and spends most of its energy on the LLM translation, because the translation is where the money is.

## Theory

### The $k$-armed bandit

You face $k$ actions (arms). Each time you pull arm $a$ you receive a reward drawn from a fixed but unknown distribution with mean

$$
q_*(a) \;=\; \mathbb{E}[R_t \mid A_t = a]. \tag{1}
$$

This $q_*(a)$ is the action value from Chapter 5.2, collapsed to a single state so I can drop the state argument. If you knew every $q_*(a)$ you would pull $\arg\max_a q_*(a)$ forever and be done. You do not, so you estimate. Let $Q_t(a)$ be your estimate of $q_*(a)$ at time $t$. The natural estimator is the sample average of the rewards seen so far from arm $a$:

$$
Q_t(a) \;=\; \frac{\sum_{i=1}^{t-1}\, R_i \,\mathbb{1}[A_i = a]}{\sum_{i=1}^{t-1}\, \mathbb{1}[A_i = a]}. \tag{2}
$$

By the law of large numbers $Q_t(a) \to q_*(a)$ as the count of pulls of $a$ grows, but only if you keep pulling $a$. That "if" is the whole problem.

```admonish derivation title="Incremental mean: why you never store the history"
Storing every past reward to recompute (2) is wasteful. Let $Q_n$ be the estimate after $n{-}1$ rewards and $R_n$ the $n$-th reward for a given arm. Then

$$
Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} R_i
        = \frac{1}{n}\left( R_n + \sum_{i=1}^{n-1} R_i \right)
        = \frac{1}{n}\left( R_n + (n-1)Q_n \right).
$$

Split the last expression and simplify:

$$
Q_{n+1} = \frac{1}{n}\big( R_n + nQ_n - Q_n \big) = Q_n + \frac{1}{n}\big( R_n - Q_n \big). \tag{3}
$$

This is the archetypal RL update: **new estimate = old estimate + step size × (target − old estimate)**. The term $(R_n - Q_n)$ is a prediction error, and $1/n$ is a step size that shrinks over time. Replace $1/n$ with a constant $\alpha$ and you get an exponentially-weighted recency-biased average, which is what you want in a *nonstationary* problem where $q_*(a)$ drifts. RL fine-tuning of an LLM is emphatically nonstationary, because the policy you are learning changes the distribution of states you visit, so constant step sizes are the norm there.
```

### The exploration-exploitation dilemma and three ways to handle it

If you always pull $\arg\max_a Q_t(a)$ (pure *greedy*), you can lock onto an arm that got lucky early and never discover a better one whose first few pulls were unlucky. You need to *explore*. Three classic mechanisms, in rising order of sophistication:

**$\varepsilon$-greedy.** With probability $1-\varepsilon$ exploit (pull the greedy arm), with probability $\varepsilon$ explore (pull a uniformly random arm). Dead simple, and it works, but it explores indiscriminately: a clearly terrible arm gets pulled as often as a promising-but-uncertain one.

**Upper confidence bound (UCB).** Explore in proportion to how *uncertain* you are, not uniformly. Pick

$$
A_t = \arg\max_a \left[\, Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \,\right], \tag{4}
$$

where $N_t(a)$ is the number of pulls of $a$ so far and $c>0$ tunes the exploration bonus. Arms pulled rarely have a large bonus and get tried; arms pulled often have a small bonus and are trusted. This "optimism in the face of uncertainty" is provably near-optimal for stationary bandits.

**Boltzmann / softmax.** Convert estimated values into a probability distribution and sample. This is the one that matters for LLMs, so it gets its own equation:

$$
\pi(a) \;=\; \frac{\exp\!\big(Q_t(a)/\tau\big)}{\sum_{b}\exp\!\big(Q_t(b)/\tau\big)}. \tag{5}
$$

The parameter $\tau > 0$ is a *temperature*. As $\tau \to 0$ the softmax concentrates all mass on the greedy arm (pure exploitation); as $\tau \to \infty$ it flattens to uniform (pure exploration). Look at (5) and then look at how a language model picks its next token, and you should feel a jolt of recognition, because they are the same equation.

### Regret, lightly

How do you score an exploration strategy? By *regret*: the cumulative gap between the reward you got and the reward the best arm would have given.

$$
L_T \;=\; \sum_{t=1}^{T}\big(\, q_*(a^*) - q_*(A_t) \,\big), \qquad a^* = \arg\max_a q_*(a). \tag{6}
$$

Pure greedy can suffer *linear* regret: if it locks onto a suboptimal arm, the per-step gap never closes and $L_T$ grows like $T$. Good strategies achieve *logarithmic* regret, $L_T = O(\ln T)$, and the Lai-Robbins result says you cannot do better than logarithmic in general. I will not prove that bound (it is a KL-divergence argument on how many pulls it takes to distinguish two reward distributions), but the intuition is worth stating plainly.

```admonish derivation title="Why regret is at least logarithmic (intuition, not a proof)"
To be confident that a suboptimal arm $a$ is really worse than the best arm $a^*$, you have to pull $a$ enough times that its sample-average reward separates from $a^*$'s by more than the noise. If the reward noise has scale $\sigma$ and the value gap is $\Delta_a = q_*(a^*) - q_*(a)$, the standard error of $Q(a)$ after $n$ pulls is $\sigma/\sqrt{n}$, so you need roughly

$$
\frac{\sigma}{\sqrt{n}} \lesssim \Delta_a \quad\Longrightarrow\quad n \gtrsim \frac{\sigma^2}{\Delta_a^2}
$$

pulls to be sure. Each of those pulls costs you $\Delta_a$ in regret, contributing $\sim \sigma^2/\Delta_a$ to $L_T$ for that arm. The $\ln T$ appears once you insist the arm stay separated over the *whole* horizon: to keep the total probability of ever mistaking $a$ for the best arm bounded across $T$ rounds, a union bound over those rounds asks each confidence interval to fail with probability $\sim 1/T$, and a Gaussian tail reaches that level only when its half-width shrinks like $\sigma\sqrt{\ln T / n}$. Setting that width below $\Delta_a$ forces $n \gtrsim (\sigma^2/\Delta_a^2)\,\ln T$, which is the $\ln T$ factor on the pull count. The takeaway for LLM training: **exploration is not free, but the cost of confirming an arm is bad is bounded, so a little exploration buys a lot of insurance.** Refuse to explore and you risk linear regret, which in fine-tuning shows up as a policy that commits early to a mediocre reasoning style and never escapes it.
```

### Temperature as the LLM's exploration knob

A language model produces a vector of logits $z = (z_1, \dots, z_{|\mathcal{V}|})$ over the vocabulary at each step. The sampling distribution is a tempered softmax, exactly (5) with the logits playing the role of the value estimates:

$$
\pi_\tau(a \mid s) \;=\; \frac{\exp(z_a / \tau)}{\sum_{b}\exp(z_b / \tau)}. \tag{7}
$$

So temperature is literally Boltzmann exploration bolted onto next-token prediction. At $\tau = 0$ (greedy decoding, argmax) the model exploits: it always emits its single most likely token, deterministic and safe and often bland. Crank $\tau$ up and the model explores, sampling rarer tokens, which is what you want during RL rollouts because **you cannot learn from an answer you never generate**. If your rollout temperature is too low, every completion for a given prompt is nearly identical, the reward signal has no variation to learn from, and gradient estimates collapse toward zero. If it is too high, completions are gibberish and rewards are uniformly low for the wrong reason. Choosing rollout temperature is choosing an exploration rate, and it is one of the few genuinely load-bearing hyperparameters in RLVR.

There is a clean derivative fact that makes the knob's behavior precise: temperature is a smooth interpolation between the uniform distribution and the point mass on the argmax.

```admonish derivation title="Temperature interpolates uniform and greedy"
Consider two tokens with logit gap $\delta = z_1 - z_2 > 0$. Their probability ratio under (7) is

$$
\frac{\pi_\tau(1)}{\pi_\tau(2)} = \frac{e^{z_1/\tau}}{e^{z_2/\tau}} = e^{\delta/\tau}.
$$

As $\tau \to 0^+$, the exponent $\delta/\tau \to +\infty$, so the ratio blows up and token 1 takes essentially all the mass: the distribution collapses to the argmax. As $\tau \to \infty$, the exponent $\delta/\tau \to 0$, so the ratio $\to 1$ and the two tokens become equally likely: the distribution flattens toward uniform over the whole vocabulary. Because this holds for every pair of tokens, the entire distribution slides monotonically from a point mass ($\tau\to 0$) to uniform ($\tau\to\infty$) as $\tau$ increases. Temperature is a single dial spanning pure exploitation to pure exploration, which is exactly the bandit tradeoff of (5).
```

### Top-k and top-p: truncation is exploration control too

Temperature is the smooth exploration dial, but in practice LLM decoding usually stacks a second, sharper control on top of it: truncated sampling. *Top-k* keeps only the $k$ highest-probability tokens and renormalizes the softmax over just those, zeroing the rest. *Top-p* (nucleus) sampling keeps the smallest set of tokens whose cumulative probability first exceeds $p$, so the number of candidates flexes with how peaked the distribution is. Both are, in the bandit language, hard *restrictions of the action set* before sampling: they forbid the long tail of low-value arms entirely rather than merely making them unlikely.

This matters for RL rollouts in a way that is easy to get wrong. Truncation removes exploration in the tail. If you generate RLVR rollouts with an aggressive top-k of, say, $k=1$, you have made decoding greedy no matter what temperature says, and every completion for a prompt will be identical, giving you zero learnable variation. Conversely, leaving the full tail open at high temperature can let the policy wander into token sequences it will never actually deploy at inference time, creating a train-inference mismatch. The practical rule I follow is to make rollout sampling match, or be slightly more exploratory than, the decoding settings you will evaluate under, so that the reward signal reflects behavior you actually intend to keep. Truncation is not a rounding detail; it is a second exploration knob sitting in series with temperature.

### Contextual bandits: the bridge to full RL

The pure $k$-armed bandit has one state, which is why it is the cleanest place to study exploration. A *contextual bandit* adds a state that you observe before choosing, but still with no transitions: each round you see a context $x$, pick an action, get a reward, and the context for the next round is drawn independently. Action values become context-dependent, $q_*(x,a)$, and the policy becomes a conditional distribution $\pi(a\mid x)$. This is exactly the sequence-level view of LLM generation from Chapter 5.1: the prompt is the context, the whole completion is the single action, the verifier is the reward, and there is no transition to a "next prompt" that your action influenced. GRPO, at its core, treats reasoning-model training as a contextual bandit, which is why the exploration machinery of this chapter, temperature and entropy, is the machinery that actually governs it, and why the value-function and multi-step credit-assignment apparatus of Chapter 5.2 can be sidestepped. The lab in Chapter 5.5 is literally a contextual bandit for that reason.

### Entropy, and how RL fine-tuning collapses it

The right way to *measure* how much a policy explores is its entropy. For the next-token distribution,

$$
H(\pi(\cdot\mid s)) \;=\; -\sum_{a} \pi(a\mid s)\,\log \pi(a\mid s). \tag{8}
$$

Entropy is maximized ($\log|\mathcal{V}|$ nats) by the uniform distribution and is zero for a deterministic policy. Here is the failure mode that bites RL fine-tuning, and it is worth taking seriously because it is not hypothetical, it is the default trajectory of a naive run.

Policy-gradient training pushes probability mass toward actions that earned high reward. That is the whole idea. But nothing in the bare objective pushes *back*, so the policy keeps sharpening: the entropy (8) falls, the rollouts for a given prompt become more and more alike, exploration dies, and the model converges prematurely onto whatever reasoning pattern happened to work early. This is *entropy collapse*. Once entropy is near zero the gradient signal starves, because with no variation among sampled completions there is no advantage to learn from (Chapter 5.5 makes "advantage needs variation" precise). The model gets stuck, sometimes at a genuinely mediocre policy, and no amount of further training rescues it because it can no longer generate the alternatives it would need to discover something better.

```admonish gotcha title="Entropy collapse is a silent killer, so instrument it"
The reason entropy collapse is dangerous is that the reward curve can look *fine* while it happens. Reward climbs, then plateaus, and you shrug and call it converged, when actually the policy has gone deterministic and simply cannot improve. Always log the mean per-token entropy (8) of your rollouts as a first-class metric next to reward. Two standard countermeasures, both of which you will meet again in the GRPO and PPO chapters:

- **Entropy bonus.** Add $+\beta\,H(\pi)$ to the objective so the optimizer is paid to stay uncertain. This is the softmax-policy analogue of $\varepsilon$-greedy: a direct incentive to keep exploring.
- **KL penalty to a reference policy.** Add $-\lambda\,\mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)$ to anchor the trained policy near the original model, which was high-entropy and fluent. This both prevents collapse and stops the policy from drifting into degenerate reward-hacking text.

On the baseline machine, entropy and KL are cheap scalars to compute from logits you already have, so there is no excuse not to log them (record the actual entropy trajectory of your first GRPO run: measured on the baseline machine — record value, date, driver).
```

```admonish read-along title="Read-along: Sutton & Barto Chapter 2; Grokking AI Chapter 10"
[S&B] Chapter 2 is the canonical bandit chapter. Section 2.2 is the $\varepsilon$-greedy setup, Section 2.4 derives the incremental mean, my (3); Section 2.6 covers nonstationarity and constant step sizes; Section 2.7 is UCB, my (4); Section 2.8 is the gradient-bandit / softmax view that anticipates (5) and the whole policy-gradient story of Chapter 5.4. Read Section 2.3 on the ten-armed testbed alongside the lab below, which is a miniature of it. For a gentler on-ramp, [GAIA] Chapter 10 walks through reinforcement learning with a slot-machine framing and Q-learning intuition and almost no calculus, which is a nice way to build the exploration-exploitation feel before returning to Sutton and Barto's formal treatment.
```

## Tooling

The tool is the simulator: a ten-armed testbed you can run in a second to *feel* the difference between exploiting and exploring, and a temperature sweep that connects the bandit softmax directly to LLM decoding. Reuse the uv project from Chapter 5.1 (`uv add numpy matplotlib` if you have not already).

## Lab

Two artifacts. First, an $\varepsilon$-greedy vs greedy comparison on a ten-armed bandit that prints how often pure greedy locks onto the wrong arm. Second, a temperature sweep over a fixed logit vector that prints entropy as a function of $\tau$, making (7) and (8) concrete and previewing entropy collapse.

```python title="labs/bandit_and_temperature.py"
"""Two demos in one file.

(A) 10-armed bandit: greedy vs epsilon-greedy, averaged over runs.
    Shows that refusing to explore can lock onto a suboptimal arm.
(B) Temperature sweep: entropy of a tempered softmax vs tau, the
    exploration knob an LLM exposes as its sampling temperature.

Run:  uv run python labs/bandit_and_temperature.py
"""
from __future__ import annotations
import numpy as np


# ---------- (A) ten-armed testbed ----------
def run_bandit(epsilon: float, steps: int, rng: np.random.Generator):
    k = 10
    q_star = rng.normal(0.0, 1.0, size=k)      # true means
    best = int(np.argmax(q_star))
    Q = np.zeros(k)                            # estimates
    N = np.zeros(k)                            # pull counts
    optimal_pulls = 0
    for t in range(steps):
        if rng.random() < epsilon:
            a = rng.integers(k)                # explore
        else:
            a = int(np.argmax(Q))              # exploit
        reward = rng.normal(q_star[a], 1.0)    # noisy reward
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]         # incremental mean, eq (3)
        optimal_pulls += (a == best)
    return optimal_pulls / steps


def bandit_demo(runs=500, steps=1000):
    rng = np.random.default_rng(1)
    for eps in (0.0, 0.01, 0.1):
        rate = np.mean([run_bandit(eps, steps, rng) for _ in range(runs)])
        label = "greedy" if eps == 0 else f"eps-greedy(e={eps})"
        print(f"{label:<22} optimal-arm rate over {steps} steps: {rate:.3f}")


# ---------- (B) temperature = LLM exploration knob ----------
def softmax(z, tau):
    z = z / tau
    z = z - z.max()                            # numerical stability
    e = np.exp(z)
    return e / e.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())       # nats, eq (8)


def temperature_demo():
    # A plausible next-token logit vector: one clear favorite, a few
    # contenders, a long tail. Same shape an LLM produces per step.
    logits = np.array([6.0, 5.2, 4.8, 3.0, 2.5, 1.0, 0.5, 0.0, -1.0, -2.0])
    h_max = np.log(len(logits))
    print(f"\nmax possible entropy (uniform): {h_max:.3f} nats")
    for tau in (0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0):
        p = softmax(logits, tau)
        top = p.max()
        print(f"tau={tau:<4} entropy={entropy(p):.3f} nats   "
              f"P(argmax)={top:.3f}")


if __name__ == "__main__":
    print("=== (A) exploration on a 10-armed bandit ===")
    bandit_demo()
    print("\n=== (B) temperature sweep (LLM decoding knob) ===")
    temperature_demo()
```

**What you should see.** In part (A), pure greedy ($\varepsilon = 0$) lands on the optimal arm markedly less often than $\varepsilon = 0.1$ over 1000 steps, typically well under it, because greedy sometimes commits to an arm that got a lucky first pull and never re-checks, the linear-regret trap made visible. In part (B), entropy is tiny at $\tau = 0.1$ with $P(\text{argmax})$ near 1 (this is what a collapsed policy looks like) and climbs steadily toward the uniform ceiling of $\ln 10 \approx 2.303$ nats as $\tau$ grows, with $P(\text{argmax})$ falling toward $0.1$. That single monotone column of entropies is exactly the dial you turn when you set rollout temperature for RL fine-tuning: too low and you have preemptively collapsed, too high and you are sampling noise. Keeping that dial in the productive middle, and logging entropy so you notice when training pushes it down on its own, is most of what "exploration" means in practice for a reasoning model.

```admonish substack-seed
Post angle: "The one knob that decides whether your AI keeps learning or gives up." Temperature is usually explained as a creativity slider for chatbots, but the deeper story is that it is a fifty-year-old idea from slot-machine math, the explore-versus-exploit dilemma, wearing a modern coat. When you train a reasoning model, the model is constantly tempted to become a know-it-all that always gives the same answer, and the moment it does, it stops being able to learn, because it can no longer surprise itself into finding a better answer. The essay connects casino odds to why a training run can look healthy on the reward chart while quietly going brain-dead (entropy collapse), and why the fix is to literally pay the model to stay a little unsure of itself.
```
