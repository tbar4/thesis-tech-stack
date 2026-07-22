# The policy gradient theorem

This is the load-bearing chapter of Part V. Everything before it was scaffolding; everything after it, REINFORCE, actor-critic, PPO, GRPO, is a variation on the single result I derive here. The policy gradient theorem tells you how to nudge the parameters of a stochastic policy so that its expected reward goes up, and it does so with a formula that, astonishingly, does not require you to differentiate through the environment or the reward. That one property, "environment-gradient-free," is the entire reason you can train a language model against a black-box verifier, a unit-test suite, or a human thumbs-up. If the theorem needed a differentiable reward, RLVR would be impossible, because "did this proof check out" is not a differentiable function of the tokens. It needs only that you can *sample* and *score*. Let me earn that claim carefully, twice, by two different routes.

## Theory

### Parameterized policies

Replace the lookup-table policy of earlier chapters with a differentiable function of parameters $\theta$. For a language model, $\theta$ is the network weights and the policy is the tempered softmax over logits from Chapter 5.3,

$$
\pi_\theta(a \mid s) = \frac{\exp\big(z_\theta(s)_a\big)}{\sum_{b}\exp\big(z_\theta(s)_b\big)}, \tag{1}
$$

where $z_\theta(s)$ is the logit vector the network computes for state (context) $s$. The only thing I will use about $\pi_\theta$ is that it is differentiable in $\theta$ and that it is a proper probability distribution, $\sum_a \pi_\theta(a\mid s) = 1$ for every $s$. That normalization identity is quietly the source of the whole theorem, as you will see.

### The objective

I want to maximize expected return. Write a full trajectory as $\tau = (s_0, a_0, r_1, s_1, a_1, \dots, s_{T})$ and its return as $R(\tau) = \sum_{t=0}^{T-1}\gamma^{t} r_{t+1}$. The probability of a trajectory under the policy factorizes into the initial-state distribution, the policy, and the environment dynamics:

$$
p_\theta(\tau) = \rho_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)\, p(s_{t+1}\mid s_t, a_t). \tag{2}
$$

The objective is the expected return over trajectories the policy induces:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\big[R(\tau)\big] = \sum_\tau p_\theta(\tau)\, R(\tau). \tag{3}
$$

I want $\nabla_\theta J$. The obstacle is that $\theta$ sits inside $p_\theta(\tau)$, which contains the environment term $p(s_{t+1}\mid s_t,a_t)$ that I do not know and cannot differentiate. The log-derivative trick makes the obstacle vanish.

### The log-derivative trick

```admonish derivation title="The log-derivative (score function) identity"
For any distribution $p_\theta(x)$ that is differentiable in $\theta$, the gradient of the distribution relates to the gradient of its log by the chain rule applied to $\log$:

$$
\nabla_\theta \log p_\theta(x) = \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}
\quad\Longrightarrow\quad
\nabla_\theta p_\theta(x) = p_\theta(x)\,\nabla_\theta \log p_\theta(x). \tag{4}
$$

The right-hand identity is the whole trick. It converts a gradient of a probability into that probability times a gradient of a log-probability, and the "that probability times" part is exactly the weight an expectation already carries. So a gradient of an expectation becomes an expectation of a gradient, which is something you can estimate by sampling.
```

Now push (4) through the objective. This is the trajectory route to the theorem, and it is the one that makes the environment-gradient-free property blindingly obvious.

```admonish derivation title="Policy gradient, route 1: the trajectory derivation"
Differentiate (3) and apply the log-derivative trick (4) with $x = \tau$:

$$
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta p_\theta(\tau)\, R(\tau)
= \sum_\tau p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)
= \mathbb{E}_{\tau\sim p_\theta}\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big].
$$

Now expand $\log p_\theta(\tau)$ using the factorization (2). Taking logs turns the product into a sum:

$$
\log p_\theta(\tau) = \log \rho_0(s_0) + \sum_{t=0}^{T-1}\Big[\log \pi_\theta(a_t\mid s_t) + \log p(s_{t+1}\mid s_t, a_t)\Big].
$$

Take the gradient with respect to $\theta$. Here is the pivotal observation: **the initial-state term $\log\rho_0(s_0)$ and every environment term $\log p(s_{t+1}\mid s_t,a_t)$ do not depend on $\theta$.** The environment is what it is; the parameters only touch the policy. So their gradients are exactly zero and they drop out:

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

Substituting back gives the trajectory form of the policy gradient theorem:

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta}\!\left[\, \Big(\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t\mid s_t)\Big)\, R(\tau) \,\right]\;}. \tag{5}
$$

Stare at (5). To estimate the gradient you need only: (i) the ability to *sample* trajectories from the policy, and (ii) a *scalar* reward $R(\tau)$ for each one. You never differentiate the reward. You never differentiate the environment. You never even need to know the environment dynamics. The gradient lives entirely in $\nabla_\theta\log\pi_\theta$, which is the gradient of the model's own log-probabilities, something autograd hands you for free.
```

### Why the environment-gradient-free form is the entire reason RLHF exists

Let me hammer this, because it is the thesis in miniature. Suppose the theorem had come out as $\nabla_\theta J = \mathbb{E}[\dots \nabla_\theta R(\tau)]$, requiring the gradient of the reward with respect to the parameters. Then you could only train against rewards that are differentiable functions of the model output. But the rewards I care about are things like:

- Does this generated Python program pass its unit tests? (A boolean from a subprocess.)
- Does this math answer equal the reference after symbolic simplification? (A SymPy call.)
- Did an Inspect scorer mark this transcript correct? (A grader, possibly another model.)
- Did a human prefer response A over response B? (A click.)

None of these is a differentiable function of the tokens. Every one of them is a *sampling-plus-scoring* black box. Equation (5) says that is exactly, and only, what you need. The policy gradient theorem is the mathematical permission slip that lets an eval score, the object of Parts III and IV, stand in as $R(\tau)$ and drive learning. That is why I can call this book "Evals as Rewards" and mean it literally: the theorem is what makes the substitution legal.

```admonish gotcha title="The reward is a constant with respect to the gradient"
A frequent confusion: since $R(\tau)$ is inside the expectation in (5), doesn't its dependence on $\theta$ matter? No. For a *fixed sampled trajectory* $\tau$, the number $R(\tau)$ is just a scalar coefficient multiplying $\nabla_\theta\log\pi_\theta$. The trajectory's $\theta$-dependence was already fully accounted for by the log-derivative trick when it turned $\nabla p_\theta$ into $p_\theta \nabla\log p_\theta$. You do not, and must not, additionally backprop through $R$. In code this means you compute the reward under `torch.no_grad()` (or in a separate process entirely, like a verifier), detach it, and use it purely as a weight on the log-prob gradient. Backpropagating through a verifier would be both wrong and, for a subprocess unit-test runner, impossible.
```

### The second route: the episodic policy gradient theorem via value functions

Route 1 is the cleanest derivation and the one that matters most for LLMs, but Sutton and Barto reach the same destination from the value-function side, and seeing both routes converge is reassuring and also reveals the $q_\pi$ structure that actor-critic methods exploit. I will derive the episodic form, where the objective is the value of the start state, $J(\theta) = v_{\pi_\theta}(s_0)$.

```admonish derivation title="Policy gradient, route 2: the value-function unrolling"
Recall from Chapter 5.2 that $v_\pi(s) = \sum_a \pi_\theta(a\mid s)\, q_\pi(s,a)$. Differentiate with the product rule:

$$
\nabla_\theta v_\pi(s) = \sum_a \Big[\, \nabla_\theta \pi_\theta(a\mid s)\, q_\pi(s,a) + \pi_\theta(a\mid s)\, \nabla_\theta q_\pi(s,a) \,\Big].
$$

The first term is what we want (it contains $\nabla\pi$); the second term is a nuisance because $q_\pi$ also depends on $\theta$. But $q_\pi(s,a) = \sum_{s',r}p(s',r\mid s,a)[r + \gamma v_\pi(s')]$, and here $p$ and $r$ do **not** depend on $\theta$ (same environment-gradient-free fact as route 1), so

$$
\nabla_\theta q_\pi(s,a) = \gamma \sum_{s'} p(s'\mid s,a)\,\nabla_\theta v_\pi(s').
$$

The nuisance term is itself a discounted copy of $\nabla_\theta v_\pi$ at the next state. So the recursion for $\nabla_\theta v_\pi(s)$ has the same self-similar shape as the Bellman equation. Define the shorthand $\phi(s) = \sum_a \nabla_\theta\pi_\theta(a\mid s)\, q_\pi(s,a)$ for the "local" first term. Then

$$
\nabla_\theta v_\pi(s) = \phi(s) + \gamma \sum_a \pi_\theta(a\mid s)\sum_{s'} p(s'\mid s,a)\,\nabla_\theta v_\pi(s').
$$

The double sum is exactly the one-step state-transition probability under the policy, call it $\Pr(s\to s', 1, \pi)$. Unroll the recursion repeatedly, substituting the formula into itself, and the gradient becomes a sum of $\phi$ evaluated at every state, weighted by the (discounted) probability of reaching that state from $s_0$ in any number of steps:

$$
\nabla_\theta v_\pi(s_0) = \sum_{s}\Big(\sum_{k=0}^{\infty}\gamma^k \Pr(s_0 \to s, k, \pi)\Big)\phi(s) = \sum_s \eta(s)\,\phi(s),
$$

where $\eta(s) = \sum_k \gamma^k \Pr(s_0\to s,k,\pi)$ is the (unnormalized) discounted state-visitation measure. Writing $\mu(s) = \eta(s)/\sum_{s'}\eta(s')$ for the normalized on-policy distribution and folding the constant into a proportionality gives the **policy gradient theorem** in its canonical Sutton and Barto form:

$$
\boxed{\;\nabla_\theta J(\theta) \;\propto\; \sum_s \mu(s)\sum_a q_\pi(s,a)\, \nabla_\theta\pi_\theta(a\mid s)\;}. \tag{6}
$$

Finally, convert (6) to the sampling-friendly expectation form with the log-derivative trick $\nabla\pi = \pi\nabla\log\pi$ once more:

$$
\nabla_\theta J(\theta) \propto \sum_s \mu(s)\sum_a \pi_\theta(a\mid s)\, q_\pi(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)
= \mathbb{E}_{s\sim\mu,\, a\sim\pi_\theta}\big[\, q_\pi(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s) \,\big]. \tag{7}
$$
```

Equations (5) and (7) are the same theorem seen from two sides. Route 1 gives the trajectory-level form with the full return $R(\tau)$ as the weight; route 2 gives the state-action form with the action value $q_\pi(s,a)$ as the weight. They are reconciled by noting that $R(\tau)$ is an unbiased single-sample estimate of $q_\pi$, which is the observation the next chapter builds REINFORCE on. The proportionality constant in (6) and (7) is the average episode length in the episodic case and is $1$ in the continuing case with the right normalization; in practice it is absorbed into the learning rate, so I will stop writing $\propto$ and write $=$ from here on.

### The score function for a softmax policy

The abstract $\nabla_\theta\log\pi_\theta$ becomes concrete and cheap for the softmax policy (1), and since the lab below and every real LLM training step compute exactly this quantity, it is worth deriving once. The result is one of the most-used gradients in all of machine learning, and it is startlingly clean.

```admonish derivation title="Gradient of log-softmax: onehot minus probability"
Let the logits be $z = z_\theta(s)$ and $\pi_a = \pi_\theta(a\mid s) = e^{z_a}/\sum_b e^{z_b}$. I want $\partial \log\pi_a / \partial z_j$, the sensitivity of the chosen action's log-probability to each logit. Write $\log\pi_a = z_a - \log\sum_b e^{z_b}$ and differentiate the two pieces.

The first piece: $\partial z_a/\partial z_j = \mathbb{1}[a=j]$, the indicator that $j$ is the action taken. The second piece, the log-normalizer, differentiates to the softmax itself:

$$
\frac{\partial}{\partial z_j}\log\sum_b e^{z_b} = \frac{e^{z_j}}{\sum_b e^{z_b}} = \pi_j.
$$

Subtracting,

$$
\frac{\partial \log\pi_a}{\partial z_j} = \mathbb{1}[a=j] - \pi_j, \tag{8}
$$

which in vector form is $\nabla_z \log\pi_a = \mathbf{e}_a - \pi$, "onehot of the action minus the probability vector." Chain this through the network with $z = z_\theta(s)$ to get $\nabla_\theta\log\pi_\theta(a\mid s)$; the $\mathbf{e}_a - \pi$ factor is the part unique to the policy-gradient loss, and it is exactly what appears in the lab's `grad_log_pi`. Read (8) intuitively: taking action $a$ pushes its own logit up (the $+1$ from the onehot) and pushes *every* logit down in proportion to its current probability (the $-\pi_j$), which is precisely the "make what I did more likely, make everything else a little less likely" behavior REINFORCE will scale by the reward.
```

Notice that the expected value of (8) under $\pi$ is $\sum_a \pi_a(\mathbf{e}_a - \pi) = \pi - \pi = 0$, the score-function-integrates-to-zero identity I will need constantly in Chapter 5.5. It falls out of the softmax gradient for free.

### On-policy by construction, and the crack that PPO widens

One more property of (5) and (7) deserves flagging now, because it is the seam along which every later algorithm is built. The expectation is taken over trajectories sampled from $\pi_\theta$, the *same* policy whose gradient you are computing. The estimator is *on-policy*: it is only valid for data generated by the current parameters. The instant you take a gradient step, $\theta$ changes, and strictly speaking your previous rollouts are stale, drawn from a policy that no longer exists.

For a single ascent step per batch of rollouts this is fine, and it is exactly what REINFORCE does. But generating rollouts is the expensive part on one GPU, all that vLLM decoding, and you would love to take *several* optimizer steps on one batch of samples to amortize the cost. Doing so violates the on-policy assumption. The repair is importance sampling: reweight each sample by the ratio of the new policy's probability to the old one's,

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}, \tag{9}
$$

which corrects the expectation back to being over $\pi_{\theta_{\text{old}}}$. That ratio (9) is the protagonist of the PPO and GRPO chapters: PPO *clips* it to keep the reused updates from wandering too far from the sampling policy, and GRPO inherits the same ratio. So the humble on-policy caveat here is the reason trust-region methods exist at all. Keep (9) in mind; it is the first thing that changes when you move past pure REINFORCE.

```admonish read-along title="Read-along: Sutton & Barto, Chapter 13"
[S&B] Chapter 13 is the home of this material. Section 13.1 motivates policy parameterization; Section 13.2 states and proves the policy gradient theorem, which is my route 2, Equation (6), and their proof is the unrolling I did in the second derivation box; Section 13.3 turns it into REINFORCE via the log-derivative step, my (7), and is the direct lead-in to Chapter 5.5. The trajectory derivation (route 1, my (5)) is the form more common in the deep-RL and RLHF literature; Schulman's writing and the PPO lineage use it, and it is the one to keep in mind when you read the GRPO chapter. If you read only one section, read 13.2 with my second box open beside it.
```

## Tooling

The tool is autograd plus a sanity check. The single most valuable habit when implementing any policy-gradient method is to verify the estimator (5) against a brute-force finite-difference gradient on a tiny problem, because a sign error or a missing `log` is otherwise invisible: the loss still goes down sometimes, just for the wrong reasons. I reuse the uv project from Chapter 5.1 and add nothing; NumPy is enough to make the point without dragging in a deep-learning framework.

## Lab

The artifact numerically confirms the policy gradient theorem. I build a one-step (bandit) softmax policy over three actions with a fixed reward per action, compute the analytic policy gradient estimator from (5) by sampling, and compare it against a finite-difference estimate of the true $\nabla_\theta J$. Agreement to a couple of decimals is the proof, in code, that "sample the log-prob gradient and weight by reward" really does climb $J$.

```python title="labs/policy_gradient_check.py"
"""Numerically verify the policy gradient theorem on a 3-armed softmax
bandit. Compare the score-function (log-derivative) estimator, eq (5),
against a finite-difference estimate of the true gradient of J(theta).

If they agree, the theorem's sampling estimator is correctly implemented.

Run:  uv run python labs/policy_gradient_check.py
"""
from __future__ import annotations
import numpy as np

REWARDS = np.array([1.0, -0.5, 2.0])     # fixed reward per action (the "verifier")
N_ACTIONS = len(REWARDS)


def softmax(theta):
    z = theta - theta.max()
    e = np.exp(z)
    return e / e.sum()


def J(theta):
    """True objective: expected reward = sum_a pi(a) R(a).
    Here the environment is trivial so we can compute J in closed form,
    which is what lets us finite-difference it as ground truth."""
    return float(softmax(theta) @ REWARDS)


def finite_difference_grad(theta, eps=1e-6):
    """Ground-truth gradient of J via central differences."""
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        d = np.zeros_like(theta)
        d[i] = eps
        g[i] = (J(theta + d) - J(theta - d)) / (2 * eps)
    return g


def grad_log_pi(theta, a):
    """d/dtheta log pi(a): for a softmax, this is (onehot(a) - pi)."""
    pi = softmax(theta)
    onehot = np.zeros_like(theta)
    onehot[a] = 1.0
    return onehot - pi


def score_function_grad(theta, n_samples, rng):
    """Monte Carlo estimate of nabla J via eq (5): average of
    R(a) * grad_log_pi(a) over sampled actions."""
    pi = softmax(theta)
    est = np.zeros_like(theta)
    for _ in range(n_samples):
        a = rng.choice(N_ACTIONS, p=pi)
        reward = REWARDS[a]                       # scalar, detached
        est += reward * grad_log_pi(theta, a)     # weight log-prob grad by reward
    return est / n_samples


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    theta = np.array([0.3, -0.2, 0.1])

    fd = finite_difference_grad(theta)
    print(f"finite-difference grad (ground truth): {fd}")

    for n in (100, 1_000, 100_000):
        sf = score_function_grad(theta, n, rng)
        err = np.linalg.norm(sf - fd)
        print(f"score-function grad, n={n:>7}: {sf}   ||err||={err:.4f}")

    # Take a few gradient-ascent steps and watch J climb.
    print("\ngradient ascent on J using the score-function estimator:")
    lr = 0.2
    for step in range(6):
        g = score_function_grad(theta, 20_000, rng)
        theta = theta + lr * g
        print(f"step {step}: J(theta)={J(theta):.4f}  pi={softmax(theta).round(3)}")
```

**What you should see.** The score-function estimator (5) converges to the finite-difference ground truth as the sample count grows: the error norm shrinks from order $10^{-1}$ at 100 samples to order $10^{-2}$ or better at $10^5$ samples, the classic Monte Carlo $1/\sqrt{n}$ shrinkage. That convergence *is* the policy gradient theorem verified: two completely different computations, one differentiating the reward-weighted log-prob and the other perturbing $\theta$ and measuring $J$, agree. The final block does gradient ascent and you watch $J(\theta)$ climb toward $2.0$ while the policy concentrates on action index 2, the highest-reward arm. Notice what the estimator never touched: the reward vector was used only as a scalar multiplier, never differentiated. Swap `REWARDS` for a call to a unit-test runner or an Inspect scorer and the exact same code trains a language model. That substitutability is the whole game.

```admonish substack-seed
Post angle: "The one equation that lets you train an AI on things you can't do calculus on." Normally, to teach a neural network you need a smooth, differentiable measure of wrongness, which is why so much of AI is quietly limited to tasks you can express as a tidy math function. The policy gradient theorem is the escape hatch: it proves you can improve a model using nothing but the ability to try something and get a yes-or-no score back, even when that score comes from running code, checking a proof, or a person clicking a button. The essay's arc is that a single algebra move (a trick for turning the gradient of a probability into the probability times the gradient of its logarithm) is what separates "AI that can only optimize smooth math" from "AI you can train against a real-world grader," and that this is the hidden hinge under every reasoning model shipped in the last two years.
```
