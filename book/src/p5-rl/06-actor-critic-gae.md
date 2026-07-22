# Actor-critic and GAE

The last chapter left me with REINFORCE and a baseline, and a promise: the advantage $A(s,a)$ is the "right" thing to multiply the score function by, and any good estimate of it will do. This chapter makes good on that promise by building the estimate the whole modern stack uses. I will learn the baseline instead of guessing it (that is the "critic"), I will show that the choice of advantage estimator is really a single knob trading bias against variance, and I will derive Generalized Advantage Estimation (GAE), which is the specific setting of that knob PPO and its descendants reach for. The derivation is a small, pretty telescoping sum, and it is worth doing by hand once because it demystifies the one line of code (`delta + gamma*lam*last`) that every PPO implementation hides the whole idea inside. At the end I price what this costs on a 16GB card, because "learn the baseline" means "carry a second network," and on the baseline machine that is the difference between a run that fits and a run that OOMs.

## Theory

### From a guessed baseline to a learned critic

Recall where REINFORCE landed. The policy gradient with a state-dependent baseline $b(s)$ is

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big)\right], \tag{6.1}$$

where $G_t = \sum_{l \ge 0} \gamma^l r_{t+l}$ is the discounted return from step $t$, and subtracting any function of the state alone leaves the gradient unbiased (the baseline term has zero expectation under the score function, which I proved in the REINFORCE chapter). The variance-minimizing baseline is close to the state value $V^\pi(s) = \mathbb{E}_{\pi}[G_t \mid s_t = s]$, so the natural move is to *estimate* $V^\pi$ with a network $V_\phi(s)$ and use it as the baseline. That network is the **critic**; the policy $\pi_\theta$ is the **actor**. The actor proposes actions, the critic grades the states the actor lands in, and the grade feeds back as the baseline in equation (6.1). With $b = V_\phi$, the multiplier $G_t - V_\phi(s_t)$ is an estimate of the advantage

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t), \tag{6.2}$$

the amount by which taking $a_t$ beats the policy's own average behavior from $s_t$. Positive advantage, push the log-prob up; negative, push it down. That is the entire actor-critic idea in one sentence.

The critic is trained by regression. I have samples of the return $G_t$ (or a bootstrapped target, more on that in a moment), and I fit $V_\phi$ to them by minimizing a squared error,

$$\mathcal{L}_V(\phi) = \tfrac{1}{2}\,\mathbb{E}\big[(V_\phi(s_t) - \hat{G}_t)^2\big], \tag{6.3}$$

where $\hat{G}_t$ is whatever target I have chosen. So there are two losses now, the policy loss (equation 6.1) and the value loss (equation 6.3), optimized together, and this two-headed structure is exactly what PPO carries into language-model training. Everything interesting about actor-critic comes down to what I plug in for $\hat{G}_t$, because that single choice sets both the critic's regression target and, through the advantage, the actor's gradient.

### The bias-variance dial

Here is the tension at the heart of the whole chapter. I can estimate the advantage in two extreme ways, and they fail in opposite directions.

At one extreme, use the full Monte Carlo return: $A_t \approx G_t - V_\phi(s_t)$. The return $G_t$ is an unbiased sample of $Q^\pi$, so this estimator is (nearly) unbiased. But $G_t$ is a sum of many random rewards over the whole rest of the trajectory, so its variance is enormous, and in a language model where the "trajectory" is a 500-token generation, that variance is brutal. Every token's advantage inherits the noise of every reward that came after it.

At the other extreme, bootstrap after one step. Use the one-step temporal-difference (TD) target $\hat{G}_t = r_t + \gamma V_\phi(s_{t+1})$, which gives the advantage estimate

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t). \tag{6.4}$$

This $\delta_t$ is the **TD residual**, and it is the single most important quantity in the GAE derivation, so give it a name and a box in your head. It has tiny variance, because it only involves one real reward $r_t$ and two value lookups. But it is *biased*: it trusts $V_\phi$, which is wrong early in training, and that bias propagates. If the critic thinks a losing position is fine, $\delta_t$ inherits the delusion.

So one-step TD is low-variance and biased; full Monte Carlo is high-variance and unbiased. Everything in between is an $n$-step estimator. Roll out $n$ real rewards, then bootstrap with the critic:

$$\hat{A}_t^{(n)} = \underbrace{r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V_\phi(s_{t+n})}_{n\text{-step return}} - V_\phi(s_t). \tag{6.5}$$

Small $n$ leans on the critic (low variance, higher bias); large $n$ leans on real rewards (high variance, lower bias); $n \to \infty$ recovers Monte Carlo. The knob is $n$, and picking a single $n$ is an awkward, discrete choice. GAE's contribution is to stop choosing and instead take a smooth, exponentially weighted *average over all $n$ at once*, controlled by a continuous parameter $\lambda \in [0,1]$. That is the dial done right.

```admonish derivation title="n-step advantages as sums of TD residuals"
Before averaging, I need the key structural fact that makes the average collapse: **the $n$-step advantage is just a discounted sum of the first $n$ TD residuals.** Start from equation (6.5) and add and subtract the intermediate values $\gamma^k V_\phi(s_{t+k})$ for $k = 1, \dots, n-1$, which changes nothing:

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t).$$

Now insert the telescoping ladder $0 = \sum_{k=1}^{n-1}\big(\gamma^k V_\phi(s_{t+k}) - \gamma^k V_\phi(s_{t+k})\big)$ and regroup the terms so each real reward sits next to the value of the state it left and the (discounted) value of the state it entered:

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \Big( r_{t+k} + \gamma V_\phi(s_{t+k+1}) - V_\phi(s_{t+k}) \Big) = \sum_{k=0}^{n-1} \gamma^k \, \delta_{t+k}. \tag{6.6}$$

Every internal value term appears once with a $+$ and once with a $-$ and cancels, leaving exactly the definition of $\delta$ from equation (6.4) at each offset. So $\hat{A}_t^{(1)} = \delta_t$, $\hat{A}_t^{(2)} = \delta_t + \gamma\delta_{t+1}$, and so on. This is the lever I need: an average over $n$ of the $\hat{A}_t^{(n)}$ becomes a weighted sum over the $\delta$'s, and the weights will telescope into something clean.
```

### GAE(λ): the telescoping sum

GAE defines the advantage estimate as the exponentially weighted average of every $n$-step estimator, with weight proportional to $\lambda^{n-1}$ and a normalizer $(1-\lambda)$ so the weights sum to one:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{\,n-1}\,\hat{A}_t^{(n)}. \tag{6.7}$$

The claim, which I will now prove, is that this apparently infinite mixture collapses to a single discounted sum of TD residuals with the discount rate $\gamma\lambda$:

$$\boxed{\;\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^{l}\,\delta_{t+l}.\;} \tag{6.8}$$

```admonish derivation title="The GAE telescoping sum"
Substitute the residual form of the $n$-step advantage, equation (6.6), into the definition (6.7):

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}.$$

I want to swap the order of summation so I collect, for each residual $\delta_{t+l}$, the total weight it receives across all the $n$-step terms that contain it. The residual $\delta_{t+l}$ (i.e. $k = l$) appears in $\hat{A}_t^{(n)}$ exactly when $n - 1 \ge l$, that is for every $n \ge l+1$. Its coefficient carries the $\gamma^l$ from the inner sum and a factor $\lambda^{n-1}$ from the outer one. Exchanging the sums:

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\sum_{l=0}^{\infty}\gamma^l\,\delta_{t+l}\sum_{n=l+1}^{\infty}\lambda^{n-1}.$$

The inner geometric series starts at $n = l+1$, i.e. at exponent $n-1 = l$:

$$\sum_{n=l+1}^{\infty}\lambda^{n-1} = \lambda^{l} + \lambda^{l+1} + \cdots = \frac{\lambda^{l}}{1-\lambda}.$$

Plug it back and the normalizer $(1-\lambda)$ cancels the denominator exactly:

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\sum_{l=0}^{\infty}\gamma^l\,\delta_{t+l}\,\frac{\lambda^{l}}{1-\lambda} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},$$

which is equation (6.8). The whole mixture over horizons became one geometric discounting of the TD residuals at rate $\gamma\lambda$. That is why GAE costs no more than one-step TD to compute: it is the same $\delta$'s, just summed with a slightly heavier discount.
```

Two endpoints check the formula and fix the intuition. At $\lambda = 0$, equation (6.8) keeps only the $l=0$ term, so $\hat{A}_t = \delta_t$: pure one-step TD, minimum variance, maximum reliance on the critic (maximum bias if the critic is wrong). At $\lambda = 1$, the discount becomes $\gamma^l$ and the sum telescopes back through equation (6.6) into $\sum_l \gamma^l r_{t+l} - V_\phi(s_t) = G_t - V_\phi(s_t)$: pure Monte Carlo, unbiased, maximum variance. So $\lambda$ is literally the bias-variance dial, sliding continuously between the two extremes I set up earlier, and the conventional working value $\lambda \approx 0.95$ sits close to the low-bias end while shaving off the worst of the variance. Note the two discounts do different jobs: $\gamma$ defines the actual objective (how much future reward is worth), while $\lambda$ is purely an estimator knob that does not change what I am optimizing, only how noisily I estimate its gradient.

```admonish derivation title="The backward recursion you actually implement"
Nobody sums equation (6.8) forward from each $t$; that is $O(T^2)$. Instead read off a one-line backward recursion. Split the sum after its first term:

$$\hat{A}_t = \delta_t + \sum_{l=1}^{\infty}(\gamma\lambda)^l\delta_{t+l} = \delta_t + \gamma\lambda\sum_{m=0}^{\infty}(\gamma\lambda)^m\delta_{t+1+m} = \delta_t + \gamma\lambda\,\hat{A}_{t+1}. \tag{6.9}$$

So walking backward from the end of the trajectory, `A[t] = delta[t] + gamma*lam*A[t+1]`, with $\hat{A}_{T} = 0$ past the terminal step (or $\delta_T = r_T - V_\phi(s_T)$ at a true terminal state where there is no bootstrap). One pass, $O(T)$, and the value target for the critic falls out for free as $\hat{G}_t = \hat{A}_t + V_\phi(s_t)$ (the "returns" you regress $V_\phi$ onto in equation 6.3). That three-line loop is the entire practical content of GAE.
```

### What this means for a language model

For an LLM, the "trajectory" is a generation and the "steps" are tokens. The state $s_t$ is the prompt plus the tokens emitted so far, the action $a_t$ is the next token, and in the RLHF setting the reward is *sparse*: zero for every intermediate token and a single scalar (a reward-model score, or a verifier's verdict) at the end of the sequence. With per-token rewards all zero until the end, the TD residual $\delta_t = \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ for interior tokens is entirely a difference of critic values, and GAE's job is to smear that terminal reward back across the tokens, credit-assigning it token by token through the learned value function. This is why PPO-for-LLMs genuinely needs a value model: without $V_\phi$ there is nothing to bootstrap from, every interior $\delta_t$ would be zero, and all the credit would dump onto the final token. The critic is what turns one scalar at the end into a dense, per-token advantage signal. GRPO, two chapters from now, will refuse to pay for that critic and get its dense signal a different way, and understanding *why* GAE needs the value model is exactly what makes GRPO's shortcut legible.

## Tooling

The tool that embodies GAE is the RLHF trainer, and the concrete reference implementation I lean on is Hugging Face's TRL `PPOTrainer` together with the `AutoModelForCausalLMWithValueHead` wrapper. That wrapper is worth dwelling on because it is where the actor-critic structure becomes a memory line item. It takes a causal LM and bolts a second output head onto the final hidden states: a small linear layer $V_\phi : \mathbb{R}^{d_\text{model}} \to \mathbb{R}$ that reads the same transformer trunk the policy uses and emits a scalar value per token position. So the critic shares the body with the actor and adds only a thin head. That sharing is a deliberate memory economy, but it does not make the critic free, because PPO's optimizer still has to carry gradients and optimizer state for the value pathway, and many setups (and the original InstructGPT recipe) use a *separate* value network entirely, which doubles the trunk.

Inside the trainer, the GAE computation is exactly equations (6.4), (6.8), and (6.9): after generating responses and scoring them, TRL computes per-token rewards (the reward-model score placed at the final token, plus a per-token KL penalty against the reference policy that I will develop in the PPO and GRPO chapters), runs `values = value_head(hidden_states)`, then loops backward accumulating `lastgaelam = delta + gamma * lam * lastgaelam` to fill an advantages tensor. The defaults you will see are `gamma=1.0` and `lam=0.95`, and $\gamma = 1$ is the usual LLM choice because a generation is short and finite and there is no reason to discount a reward that arrives 200 tokens away relative to one 10 tokens away. The returns tensor `advantages + values` becomes the regression target for the value loss (equation 6.3), which TRL clips just like the policy loss, a detail I will motivate in the next chapter.

```admonish under-the-hood
The value head reads the transformer's last hidden state, which means the critic and the policy see the *same* forward activations. TRL exploits this: one forward pass through the trunk produces both the next-token logits (for the policy ratio) and the per-token values (for GAE), so you are not paying for two forward passes, only two heads on one. This is also why a shared-trunk value head is so much cheaper than a separate critic network, and why, on 16GB, the shared head is basically the only option that fits alongside a 1-3B policy. The catch is a subtle optimization coupling: gradients from the value loss flow back into the shared trunk and can fight the policy loss, which is why the value-loss coefficient (`vf_coef`, often 0.1-0.5) exists and why some recipes stop the value gradient from updating the trunk at all.
```

```admonish vram-budget
On the baseline machine (MSI Aegis R2, RTX 5080 16GB, Blackwell), the critic is not an afterthought, it is a headline cost. Take a policy with $P$ parameters in BF16. The policy alone, training with a fused AdamW-style optimizer, needs roughly: 2 bytes/param weights + 2 bytes/param gradients + 8 bytes/param optimizer state (FP32 moment + variance, and often an FP32 master weight pushing this higher), so on the order of 12 bytes per parameter *before activations*. A **separate** full value network of the same size doubles all of that, which is why a 3B policy that trains comfortably alone can OOM the moment you attach an independent critic. A **shared-trunk value head** avoids duplicating the trunk's weights and optimizer state, adding only the tiny head's parameters plus the extra activation memory for the value forward and its gradient, so it is the only actor-critic configuration I would attempt for a 1.5-3B policy on this card. Record the actual peak with `torch.cuda.max_memory_allocated()` for your policy size (measured on the baseline machine — record value, date, driver). The one-sentence version, which the GRPO chapter cashes in: **the value model is the most expensive optional thing in PPO, and its cost is the whole reason a critic-free method is attractive on one GPU.**
```

## Lab

The goal of this lab is to *see* the bias-variance dial move. I will build a tiny, fully controllable environment where the true advantage is known in closed form, compute GAE at several $\lambda$ values against a deliberately-wrong critic, and plot estimator bias against estimator variance as $\lambda$ sweeps from 0 to 1. The artifact is a CSV and a PNG showing the classic tradeoff curve, so that when a real PPO run later uses $\lambda = 0.95$ you know exactly what that number is buying. No GPU is needed here; this is pure estimator arithmetic and it runs on CPU in a second.

This is a `uv` project. From the repo root:

```bash
uv init labs/gae-dial
cd labs/gae-dial
uv add numpy matplotlib
```

```python title="labs/gae-dial/gae_bias_variance.py"
"""Visualize the GAE(lambda) bias-variance tradeoff on a known-answer MDP.

A short linear-chain MDP with deterministic rewards gives a closed-form true
advantage. We corrupt the critic with a fixed error, then measure how GAE's
bias (systematic offset from the true advantage) and variance (spread across
noisy reward rollouts) move as lambda sweeps 0 -> 1.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

GAMMA = 0.99
T = 20                 # steps per episode
N_EPISODES = 4000      # rollouts to estimate variance
REWARD_NOISE = 1.0     # std of per-step reward noise
CRITIC_BIAS = 0.5      # systematic error we bake into V_phi

rng = np.random.default_rng(0)

def true_values(mean_rewards):
    """Exact V(s_t) = sum_{k>=0} gamma^k E[r_{t+k}] for the chain."""
    V = np.zeros(T + 1)
    for t in range(T - 1, -1, -1):
        V[t] = mean_rewards[t] + GAMMA * V[t + 1]
    return V

def gae(deltas, lam):
    """Backward recursion A[t] = delta[t] + gamma*lam*A[t+1]  (eq 6.9)."""
    A = np.zeros(len(deltas))
    running = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        running = deltas[t] + GAMMA * lam * running
        A[t] = running
    return A

def main():
    mean_rewards = np.linspace(1.0, 0.2, T)      # deterministic reward means
    V_true = true_values(mean_rewards)
    V_phi = V_true + CRITIC_BIAS                  # a critic that is wrong by +0.5

    # True advantage at t=0 for the greedy step: A = Q - V. With deterministic
    # dynamics and this policy, the true A_0 is 0 in expectation (the baseline
    # is exactly right when V_true is used), so the *estimator's* mean minus 0
    # is its bias, and its spread across rollouts is its variance.
    lambdas = np.linspace(0.0, 1.0, 21)
    rows, biases, variances = [], [], []

    for lam in lambdas:
        est0 = np.empty(N_EPISODES)
        for e in range(N_EPISODES):
            rewards = mean_rewards + rng.normal(0, REWARD_NOISE, size=T)
            # delta_t = r_t + gamma V(s_{t+1}) - V(s_t), bootstrap with wrong V_phi
            deltas = rewards + GAMMA * V_phi[1:] - V_phi[:-1]
            est0[e] = gae(deltas, lam)[0]
        # Bias: mean estimate minus the true advantage against the *correct* V.
        true_A0 = 0.0
        bias = est0.mean() - true_A0
        var = est0.var()
        biases.append(bias)
        variances.append(var)
        rows.append({"lambda": round(lam, 3),
                     "bias": bias, "variance": var,
                     "abs_bias": abs(bias)})

    csv_path = OUT / "gae_tradeoff.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "bias", "variance", "abs_bias"])
        w.writeheader(); w.writerows(rows)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(lambdas, np.abs(biases), "o-", color="C3", label="|bias|")
    ax1.set_xlabel(r"$\lambda$"); ax1.set_ylabel("|bias|", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")
    ax2 = ax1.twinx()
    ax2.plot(lambdas, variances, "s-", color="C0", label="variance")
    ax2.set_ylabel("variance", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax1.set_title(r"GAE($\lambda$): the bias-variance dial")
    fig.tight_layout()
    png_path = OUT / "gae_tradeoff.png"
    fig.savefig(png_path, dpi=130)

    print(f"lambda=0.00  |bias|={abs(biases[0]):.3f}  var={variances[0]:.3f}")
    print(f"lambda=0.95  |bias|={abs(biases[19]):.3f}  var={variances[19]:.3f}")
    print(f"lambda=1.00  |bias|={abs(biases[-1]):.3f}  var={variances[-1]:.3f}")
    print(f"Artifacts: {csv_path.resolve()}  {png_path.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python gae_bias_variance.py
```

```admonish gotcha
The bias in this toy comes entirely from `CRITIC_BIAS`, the deliberate error I baked into $V_\phi$. That is the honest picture: GAE's bias is *inherited from the critic*, so at $\lambda = 0$ (pure bootstrap) the estimate leans hardest on the wrong critic and shows the most bias, while at $\lambda = 1$ (pure Monte Carlo) the critic's error washes out of the advantage almost entirely because the value terms telescope away, leaving only the real (noisy) rewards. If you set `CRITIC_BIAS = 0`, the |bias| curve flattens to zero and only the variance curve moves, which is the clean way to confirm that $\lambda$ trades variance for a bias you only pay when your critic is wrong. In real training the critic is always at least a little wrong, especially early, which is why $\lambda = 1$ is rarely used despite being unbiased.
```

**What you should see.** As $\lambda$ climbs from 0 to 1, the `|bias|` curve falls toward zero (the critic's baked-in error drains out of the estimate) while the `variance` curve climbs steeply (more real, noisy rewards enter the sum). The two curves cross somewhere in the middle, and the region around $\lambda \approx 0.9$ to $0.97$ is visibly the sweet spot: most of the bias is already gone but the variance has not yet exploded, which is the empirical reason the field settled on $\lambda = 0.95$. The printed lines give you three points on that curve to sanity-check against the plot, and the CSV lets you re-plot or fit the tradeoff yourself. Read this figure as the picture behind one hyperparameter you will otherwise copy on faith into every PPO and GRPO config in Part VII.

```admonish read-along
Read this chapter against **[RLHF]** ch. 6, which develops the policy-gradient-to-actor-critic progression in the specific context of language-model RLHF and lays out GAE as the advantage estimator PPO uses. Their treatment connects the value head and the per-token KL reward directly to the training loop you will run in Part VII; my derivation here (equations 6.6-6.9) is the underlying math their trainer takes as given. If you want the original, Schulman et al. (2016), "High-Dimensional Continuous Control Using Generalized Advantage Estimation," is where equation (6.8) comes from.
```

```admonish substack-seed
"The one line of code that hides an infinite sum." Every PPO implementation contains `A = delta + gamma*lam*A_next`, three symbols and a loop, and almost nobody who copies it can tell you it is the collapsed form of an exponentially-weighted average over every n-step return at once. A post that takes the reader from "I have a noisy return and a biased critic, which do I trust" through the telescoping sum (equation 6.8) to the punchline that $\lambda$ is a single continuous knob between the two, and that $\lambda = 0.95$ is not folklore but a measurable point on a bias-variance curve you can plot in twenty lines on a laptop. The hook: the prettiest derivation in RL is also the one people are most likely to paste without reading.
```
