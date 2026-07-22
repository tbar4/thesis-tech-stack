# Trust regions: from TRPO to PPO

I now have a policy gradient (Part V chapter 4), a learned baseline, and a low-variance advantage estimate from GAE (chapter 6). In principle I could just take gradient steps and be done. In practice, if I do that naively, the policy collapses, and it collapses in a way that is not a bug in my code but a fundamental feature of on-policy RL. This chapter is about why, and about the two fixes that define modern policy optimization: TRPO, which solves the problem correctly but expensively with a hard trust-region constraint, and PPO, which approximates that constraint with a clipped objective so simple you can implement it in five lines and so robust it became the default for essentially all of RLHF. I will derive the surrogate objective from the performance-difference lemma, sketch the TRPO constrained problem, and then do the PPO clip in full, including the piecewise case analysis that is the whole reason the clip works. The geometry of that clip is the payoff: once you see which way each piece pushes, PPO stops being a magic formula and becomes obvious.

## Theory

### Why a naive policy-gradient step destroys the policy

The policy gradient tells me a *direction* in parameter space, not a *distance*. The gradient $\nabla_\theta J$ is a statement about an infinitesimal step; it says nothing about how far I can move before the linear approximation it is built on stops being true. And in policy optimization the approximation breaks down viciously fast, for a reason specific to RL that supervised learning does not share: the data distribution depends on the parameters I am changing. In supervised learning, a too-large step gives a worse model on a *fixed* dataset, and the next batch corrects it. In RL, a too-large step gives a worse *policy*, that worse policy generates the next batch of data, and if the policy has moved somewhere terrible it collects terrible data and the estimate of the next gradient is computed on that garbage. There is no fixed dataset to fall back to. A single overlarge step can push the policy into a region where it emits degenerate outputs, those outputs get near-zero or misleading reward, the gradient signal collapses, and the run never recovers. This is the "falling off a cliff" failure, and it is why you cannot just crank the learning rate.

The deeper issue is that the natural distance in *parameter* space (Euclidean distance on $\theta$) has nothing to do with the distance in *policy* space (how different the resulting action distributions are). A tiny change in $\theta$ can enormously change $\pi_\theta$ if you are in a sensitive region, and a large change in $\theta$ can barely move $\pi_\theta$ elsewhere. So bounding $\|\Delta\theta\|$ is the wrong constraint. What I actually want to bound is how much the *policy distribution* moves, and the natural currency for that is the KL divergence $D_{\text{KL}}(\pi_{\text{old}} \,\|\, \pi_\theta)$. Keep the new policy inside a small KL ball around the old one and the data distribution cannot lurch, the advantage estimates I computed under $\pi_{\text{old}}$ stay approximately valid, and each step is a safe, monotonic-ish improvement. That KL ball is the "trust region," the region where I trust my local model of the objective. Everything below is machinery for optimizing inside it.

### The surrogate objective and the performance-difference lemma

To optimize offline (using a batch of data from $\pi_{\text{old}}$ to take a step that produces $\pi_\theta$), I need to express the *new* policy's performance in terms of *old*-policy samples. The exact bridge is the performance-difference lemma (Kakade and Langford, 2002):

$$J(\pi_\theta) - J(\pi_{\text{old}}) = \mathbb{E}_{s \sim d^{\pi_\theta}}\,\mathbb{E}_{a \sim \pi_\theta(\cdot\mid s)}\big[A^{\pi_{\text{old}}}(s, a)\big], \tag{7.1}$$

where $d^{\pi_\theta}$ is the discounted state-visitation distribution under the *new* policy. The lemma is exact and beautiful, but it has a problem that is the crux of everything: the expectation is over states visited by $\pi_\theta$, the very policy I am trying to find, so I cannot sample from it yet.

```admonish derivation title="From the exact lemma to the surrogate objective"
Proof sketch of (7.1) first, because it is short and it shows where the advantage comes from. Write the discounted return as a telescoping sum of advantages along a trajectory drawn from the new policy $\pi_\theta$, using $A^{\pi_{\text{old}}}(s,a) = r(s,a) + \gamma V^{\pi_{\text{old}}}(s') - V^{\pi_{\text{old}}}(s)$:

$$\mathbb{E}_{\tau \sim \pi_\theta}\!\Big[\sum_{t\ge0}\gamma^t A^{\pi_{\text{old}}}(s_t,a_t)\Big] = \mathbb{E}_{\tau\sim\pi_\theta}\!\Big[\sum_{t\ge0}\gamma^t\big(r_t + \gamma V^{\pi_{\text{old}}}(s_{t+1}) - V^{\pi_{\text{old}}}(s_t)\big)\Big].$$

The $V^{\pi_{\text{old}}}$ terms telescope: $\sum_t \gamma^t(\gamma V(s_{t+1}) - V(s_t)) = -V(s_0)$ in expectation. So the right side is $\mathbb{E}_{\tau\sim\pi_\theta}[\sum_t \gamma^t r_t] - \mathbb{E}[V^{\pi_{\text{old}}}(s_0)] = J(\pi_\theta) - J(\pi_{\text{old}})$, which is (7.1).

Now the move that makes it usable. I cannot sample states from $d^{\pi_\theta}$, so I **approximate** it with the old policy's visitation $d^{\pi_{\text{old}}}$, which is valid as long as $\pi_\theta$ is close to $\pi_{\text{old}}$ (close policies visit similar states). That gives the *surrogate objective*:

$$L_{\pi_{\text{old}}}(\pi_\theta) = \mathbb{E}_{s \sim d^{\pi_{\text{old}}}}\,\mathbb{E}_{a \sim \pi_\theta(\cdot\mid s)}\big[A^{\pi_{\text{old}}}(s,a)\big]. \tag{7.2}$$

The inner expectation is still over $a \sim \pi_\theta$, which I also do not have samples from (my actions came from $\pi_{\text{old}}$). Fix that with **importance sampling**: reweight old-policy actions by the ratio of new to old probability,

$$\mathbb{E}_{a\sim\pi_\theta}[A] = \mathbb{E}_{a\sim\pi_{\text{old}}}\Big[\tfrac{\pi_\theta(a\mid s)}{\pi_{\text{old}}(a\mid s)}A\Big],$$

which is exact (it is just multiplying and dividing by $\pi_{\text{old}}$ inside the sum over actions). Defining the **probability ratio**

$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}, \tag{7.3}$$

the surrogate becomes something I can estimate entirely from old-policy rollouts:

$$L(\theta) = \mathbb{E}_t\big[\,r_t(\theta)\,\hat{A}_t\,\big]. \tag{7.4}$$

This is the objective every trust-region method maximizes. Note $r_t(\theta_{\text{old}}) = 1$ and $\nabla_\theta L\big|_{\theta_{\text{old}}} = \mathbb{E}[\hat{A}_t \nabla_\theta \log\pi_\theta]$, so at the starting point the surrogate's gradient is exactly the policy gradient. The surrogate is a *local model* of the true objective that agrees with it to first order, and is trustworthy only while $\pi_\theta$ stays near $\pi_{\text{old}}$, which is precisely why it must be paired with a trust region.
```

The two approximations I just made (swapping $d^{\pi_\theta}$ for $d^{\pi_{\text{old}}}$, and trusting importance sampling with a ratio that could be anything) are both only valid near $\pi_{\text{old}}$. Push $\pi_\theta$ far and the surrogate $L(\theta)$ stops predicting the true $J(\pi_\theta)$, the ratio $r_t$ can blow up, and maximizing $L$ happily walks you off the cliff. So the surrogate is necessary but not sufficient; it must be constrained.

### TRPO: the hard constraint

TRPO makes the trust region explicit. Maximize the surrogate subject to a hard KL constraint:

$$\max_{\theta}\; \mathbb{E}_t\big[r_t(\theta)\,\hat{A}_t\big] \quad\text{subject to}\quad \mathbb{E}_t\big[D_{\text{KL}}\big(\pi_{\text{old}}(\cdot\mid s_t)\,\|\,\pi_\theta(\cdot\mid s_t)\big)\big] \le \delta. \tag{7.5}$$

```admonish derivation title="Why TRPO reduces to a natural-gradient step"
Approximate the two pieces near $\theta_{\text{old}}$. The surrogate is linear to first order, $L(\theta) \approx g^\top(\theta - \theta_{\text{old}})$ with $g = \nabla_\theta L$ the policy gradient. The KL constraint is quadratic to *second* order, because its first-order term vanishes (KL is minimized at zero when the policies match, so its gradient there is zero):

$$\bar{D}_{\text{KL}}(\theta) \approx \tfrac{1}{2}(\theta - \theta_{\text{old}})^\top F\,(\theta - \theta_{\text{old}}),$$

where $F = \mathbb{E}\big[\nabla_\theta\log\pi_\theta\,\nabla_\theta\log\pi_\theta^\top\big]$ is the Fisher information matrix (the Hessian of the KL at the origin). So (7.5) becomes: maximize $g^\top\Delta\theta$ subject to $\tfrac12\Delta\theta^\top F\Delta\theta \le \delta$. The Lagrangian solution is the **natural gradient**:

$$\Delta\theta \;\propto\; F^{-1} g, \qquad \Delta\theta = \sqrt{\tfrac{2\delta}{g^\top F^{-1} g}}\;F^{-1}g. \tag{7.6}$$

This is the correct, geometry-aware step: it moves along the gradient *preconditioned by the inverse Fisher*, so a fixed KL budget $\delta$ translates into an appropriately-sized parameter step no matter how sensitive the local policy is. TRPO computes $F^{-1}g$ without forming $F$ (which for a large network is astronomically big) using conjugate gradient on Fisher-vector products, then backtracks along the step to enforce the exact constraint and the actual surrogate improvement.
```

TRPO works and comes with a monotonic-improvement guarantee, but equation (7.6) is a lot of machinery: conjugate-gradient inner loops, Fisher-vector products, a line search, all per update. For a transformer with billions of parameters and a training loop I want to run on one GPU, this is both a memory and an engineering burden. PPO's entire pitch is: get 90% of the trust-region benefit with a first-order method that needs none of that.

### PPO: clipping as a soft trust region

PPO throws away the explicit KL constraint and instead *bakes the trust region into the objective itself*, so that plain SGD/Adam cannot want to move too far. The clipped surrogate is

$$\boxed{\;L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\,\hat{A}_t,\;\; \text{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\big)\Big],\;} \tag{7.7}$$

where $\text{clip}(x, a, b) = \max(a, \min(x, b))$ and $\epsilon$ is a small constant, typically $0.1$ to $0.2$. Two things are happening: the ratio is clipped to the interval $[1-\epsilon, 1+\epsilon]$, and then the objective takes the *minimum* of the clipped and unclipped terms. The minimum is what makes it a lower bound (a pessimistic surrogate), and the direction of the clip flips depending on the sign of the advantage. That sign-dependence is the whole design, so I will take it apart case by case.

```admonish derivation title="PPO-clip, the full piecewise analysis"
Fix a single timestep and drop the subscript: ratio $r = r_t(\theta)$ (which is $1$ at $\theta_{\text{old}}$ and moves as $\theta$ changes), advantage $\hat{A}$, clip width $\epsilon$. The per-sample objective is $\ell(r) = \min\big(r\hat{A},\,\text{clip}(r,1-\epsilon,1+\epsilon)\hat{A}\big)$. Split on the sign of $\hat{A}$.

**Case 1: $\hat{A} > 0$ (this action was good, I want to increase its probability, i.e. push $r$ up).**
The unclipped term $r\hat{A}$ rises without bound as $r$ increases. The clipped term is $\min(r, 1+\epsilon)\hat{A}$, capped at $(1+\epsilon)\hat{A}$. Taking the min of the two:
- For $r \le 1+\epsilon$: clip does nothing, $\ell(r) = r\hat{A}$, ordinary gradient pushing $r$ up.
- For $r > 1+\epsilon$: clipped term $(1+\epsilon)\hat{A}$ is the smaller one, so $\ell(r) = (1+\epsilon)\hat{A}$, a **constant**. Its gradient in $\theta$ is zero.

So once the new policy has made this good action more than $(1+\epsilon)$ times as likely as the old policy did, the objective flatlines and stops rewarding further increases. The incentive to keep climbing is switched off exactly at the trust-region boundary.

**Case 2: $\hat{A} < 0$ (this action was bad, I want to decrease its probability, i.e. push $r$ down).**
Now $r\hat{A}$ becomes *more positive* as $r$ decreases (a negative advantage times a shrinking ratio), so the objective wants $r$ small. The clipped term is $\max(r, 1-\epsilon)\hat{A}$ (the clip's lower arm binds now), floored at $(1-\epsilon)\hat{A}$. Because $\hat{A}<0$, the min of the two picks the *more negative*, i.e. the smaller, value:
- For $r \ge 1-\epsilon$: clip inactive, $\ell(r) = r\hat{A}$, gradient pushing $r$ down.
- For $r < 1-\epsilon$: clipped term $(1-\epsilon)\hat{A}$ is now the *smaller (more negative)* term, and since we take the min, $\ell(r) = (1-\epsilon)\hat{A}$, again a **constant** with zero gradient.

So once the new policy has driven this bad action below $(1-\epsilon)$ times its old probability, the objective flatlines and stops rewarding further suppression. Symmetric to case 1.

**The unified statement.** The clip removes the gradient incentive to move the ratio *further past the boundary in the direction the advantage wants*. It does **not** clip when the ratio moves in the "wrong" direction (a good action becoming less likely, or a bad action becoming more likely); there the full unclipped gradient applies, so PPO can always *correct* an overshoot, it just refuses to *chase* one. Taking the min is what guarantees this asymmetry: $L^{\text{CLIP}}$ is a pessimistic lower bound on the unclipped surrogate, tight at $r=1$ and only ever pulling the objective down, never inflating it. That pessimism is the soft trust region.
```

The geometry is worth stating in plain words because it is the mental model I carry into every PPO and GRPO run. Picture $\ell(r)$ as a function of the ratio $r$, with the old policy sitting at $r=1$. For a good action, the objective is a ramp going up and to the right that hits a flat ceiling at $r = 1+\epsilon$. For a bad action, it is a ramp going up and to the left (toward smaller $r$) that hits a flat floor at $r = 1-\epsilon$. In both cases there is a flat plateau *beyond* the boundary in the "desired" direction and a live slope on the near side and in the "undo" direction. Gradient ascent slides up the ramp until it reaches the plateau, then stops, per token, per sample. No Fisher matrix, no line search, no explicit KL, just a clamp and a min. The clip width $\epsilon$ is the radius of the trust region measured in ratio space, and $\epsilon = 0.2$ meaning "don't let any single token's probability move by more than about 20% per update round" is a genuinely useful one-line summary.

```admonish gotcha
The clip alone does **not** bound the KL divergence, a subtlety that trips people who read "PPO replaces the KL constraint." Clipping only kills the gradient once a ratio is already outside $[1-\epsilon,1+\epsilon]$; a single large update, or many small correlated ones, can still carry the ratio far past the boundary before the gradient dies, and the min does not pull it back, it only stops pushing. That is why practical PPO for LLMs keeps a *separate* explicit KL penalty against a reference policy in the reward (the per-token KL I mentioned in the GAE chapter), and why implementations often add early-stopping on measured KL per epoch. The clip is a soft, per-token trust region; it is necessary but, on its own, not a hard guarantee. Hold this thought, because GRPO's KL term is exactly this separate penalty, and DAPO's "clip-higher" is exactly a modification of $\epsilon$.
```

## Tooling

The tool is again TRL's `PPOTrainer`, now read through the lens of equation (7.7). One PPO iteration does four things: generate responses from the current policy (this snapshots $\pi_{\text{old}}$, whose log-probs are cached), score them with a reward model or verifier and compute per-token rewards including the KL-to-reference penalty, compute advantages $\hat{A}_t$ with GAE from the previous chapter, then take several gradient epochs over the batch optimizing $L^{\text{CLIP}}$ plus the clipped value loss plus an entropy bonus. The ratio $r_t(\theta)$ is computed as `exp(new_logprob - old_logprob)` per token, where `old_logprob` is the cached value from generation time and `new_logprob` comes from a fresh forward pass under the current $\theta$; this is why PPO does multiple gradient steps per rollout batch (the "$K$ epochs"), because after the first step $\theta \ne \theta_{\text{old}}$ and the ratio genuinely starts to move away from 1, which is the entire point of the clip.

```admonish under-the-hood
"On-policy" is a spectrum in practice, and the ratio is what buys the slack. Strictly on-policy would mean one gradient step per rollout, wasting the expensive generation. Importance sampling (equation 7.3) lets PPO reuse each batch for several epochs by correcting for the growing mismatch between $\pi_\theta$ and the $\pi_{\text{old}}$ that generated the data, and the clip is what keeps that reuse from going off the rails when the ratio drifts too far. So the clip is not just a safety rail on step size, it is what makes PPO *sample-efficient enough to afford at all*: without it you could not safely take more than one step per generation, and generation is the costliest part of the loop. On the baseline machine (RTX 5080 16GB), where a generation pass with vLLM is the throughput bottleneck, squeezing 2-4 gradient epochs out of each rollout batch is the difference between a training run that finishes overnight and one that does not.
```

```admonish vram-budget
PPO's memory footprint on 16GB is dominated by carrying *four* model roles at once: the policy being trained (weights + grads + optimizer state), the value model (chapter 6, cheap if it is a shared head, expensive if separate), a frozen reference policy for the KL penalty (weights only, no grads, but still a full forward pass), and the reward model or verifier (weights only, or free if the verifier is a Python function, which is exactly the RLVR advantage the thesis leans on). For a 1.5-3B policy in BF16 this is tight but feasible on the RTX 5080 if the value head is shared and the reference/reward passes run in `no_grad`; push to a separate value network or a large reward model and you overflow. Record the peak with `torch.cuda.max_memory_allocated()` per configuration (measured on the baseline machine — record value, date, driver). This four-model burden is precisely what GRPO attacks: drop the value model (chapter 6's expensive optional thing) and, in the RLVR setting, drop the reward model too by using a verifier, and suddenly you are carrying policy plus a frozen reference, which is what fits comfortably on one card.
```

## Lab

The goal of this lab is to make the clip's geometry undeniable by plotting $\ell(r)$ itself, for both advantage signs, and overlaying the unclipped surrogate so you can see exactly where and why the objective flattens. This is the picture from the derivation, rendered from the same arithmetic PPO runs on every token. It is pure math, no GPU, no model, and it runs instantly. Seeing the two ramps-into-plateaus once is worth more than re-reading the min-of-clip formula ten times.

This is a `uv` project. From the repo root:

```bash
uv init labs/ppo-clip-geometry
cd labs/ppo-clip-geometry
uv add numpy matplotlib
```

```python title="labs/ppo-clip-geometry/clip_geometry.py"
"""Plot the PPO-clip per-sample objective as a function of the ratio r.

Reproduces the piecewise picture from the derivation: for A>0 a rising ramp
that ceilings at r=1+eps, for A<0 a rising-to-the-left ramp that floors at
r=1-eps. Overlays the unclipped surrogate r*A to show where the clip bites.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
EPS = 0.2

def clipped_obj(r, A, eps=EPS):
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    return np.minimum(unclipped, clipped)

def main():
    r = np.linspace(0.0, 2.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, A, title in [(axes[0], +1.0, r"$\hat{A} > 0$ (good action)"),
                         (axes[1], -1.0, r"$\hat{A} < 0$ (bad action)")]:
        ax.plot(r, r * A, "--", color="0.6", label=r"unclipped $r\hat{A}$")
        ax.plot(r, clipped_obj(r, A), color="C0", lw=2.4, label=r"$L^{CLIP}$")
        for x in (1 - EPS, 1.0, 1 + EPS):
            ax.axvline(x, color="0.85", lw=1, zorder=0)
        ax.set_title(title); ax.set_xlabel(r"ratio $r_t(\theta)$")
        ax.set_ylabel("per-sample objective"); ax.legend(loc="best")
        ax.annotate(r"$1-\epsilon$", (1 - EPS, ax.get_ylim()[0]),
                    ha="center", va="bottom", fontsize=8, color="0.4")
        ax.annotate(r"$1+\epsilon$", (1 + EPS, ax.get_ylim()[0]),
                    ha="center", va="bottom", fontsize=8, color="0.4")

    fig.suptitle(r"PPO-clip geometry ($\epsilon = 0.2$): the plateau is the trust region")
    fig.tight_layout()
    png = OUT / "clip_geometry.png"
    fig.savefig(png, dpi=130)

    # Print the four regimes as a table so the plateaus are checkable numerically.
    print("A>0:  r=0.9 -> L=%.2f   r=1.3 -> L=%.2f (ceiling at 1+eps)"
          % (clipped_obj(0.9, 1.0), clipped_obj(1.3, 1.0)))
    print("A<0:  r=1.1 -> L=%.2f   r=0.7 -> L=%.2f (floor at 1-eps)"
          % (clipped_obj(1.1, -1.0), clipped_obj(0.7, -1.0)))
    print(f"Artifact: {png.resolve()}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python clip_geometry.py
```

```admonish gotcha
Read the plateaus carefully against the derivation, because the intuition most people carry is half wrong. On the $\hat{A}>0$ panel the objective flatlines for $r > 1+\epsilon$ (stop chasing a good action once it is already 20% more likely), but it keeps a live downward slope for $r < 1-\epsilon$ where a good action has become *less* likely, so PPO will still fight to bring it back. On the $\hat{A}<0$ panel it is mirrored: flat for $r < 1-\epsilon$, but a live slope for $r > 1+\epsilon$ where a bad action got *more* likely. The clip is one-sided per case; it disables the incentive to overshoot in the desired direction while leaving the correction always available. If your mental model was "PPO clips whenever the ratio leaves the band," this plot is the correction: it clips only on the far side of the direction the advantage is pushing.
```

**What you should see.** Two panels. Left ($\hat{A}>0$): a straight line rising with $r$ that abruptly goes flat at $r=1+\epsilon=1.2$, with the dashed unclipped line continuing up past it, so the gap between them is the reward PPO is deliberately declining. Right ($\hat{A}<0$): a line rising as $r$ *decreases*, going flat at $r=1-\epsilon=0.8$, again with the unclipped dashed line diverging below. The four printed numbers pin the plateaus: for $\hat{A}>0$, $r=1.3$ gives exactly $1+\epsilon = 1.2$ (clamped) while $r=0.9$ gives $0.9$ (unclamped); for $\hat{A}<0$, $r=0.7$ gives $-(1-\epsilon) = -0.8$ (clamped) while $r=1.1$ gives $-1.1$ (unclamped, still being corrected). That asymmetry, live on the correction side and flat on the overshoot side, is the entire trust-region behavior of PPO in one figure, and it is the picture I want in your head when GRPO reuses this exact clip in the next chapter.

```admonish read-along
Read this against **[RLHF]** ch. 6 for the RLHF-specific PPO recipe (the per-token KL reward, the value head, the practical loss composition) and **[BRM]** ch. 6, which walks the clipped objective and its implementation in the reasoning-model context this thesis targets. My derivation of the surrogate from the performance-difference lemma (equations 7.1-7.4) is the theoretical backing both treatments assume; the primary sources are Schulman et al. (2015) "Trust Region Policy Optimization" for equations (7.5)-(7.6) and Schulman et al. (2017) "Proximal Policy Optimization Algorithms" for the clip (7.7).
```

```admonish substack-seed
"Why you can't just turn up the learning rate on a policy." The one-paragraph version of the whole cliff problem: in supervised learning a bad step is corrected by the next batch, but in RL the policy *generates* the next batch, so a bad step poisons its own future data and there is no fixed ground truth to fall back to. TRPO's answer is a careful KL-constrained natural-gradient step (correct, expensive); PPO's answer is a clip so simple it fits on one line, and the post's payoff is the geometry: two ramps into two plateaus, live on the side that corrects mistakes and flat on the side that would chase them. The hook is that the most-deployed RL algorithm on earth is, geometrically, just "stop pushing once you've pushed enough," and you can plot the whole thing in twenty lines with no GPU.
```
