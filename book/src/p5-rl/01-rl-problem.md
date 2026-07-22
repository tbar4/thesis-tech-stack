# The RL problem

Everything in Part V is aimed at one payoff: taking an eval score, the same number I spent Parts III and IV learning to trust, and turning it into a training signal. That is what "evals as rewards" means literally. Before I can do that I need the vocabulary of reinforcement learning, and I need it mapped cleanly onto what a language model actually does when it generates text. This chapter builds the frame. It is theory-forward, but I close with a tiny artifact so the abstractions have something to bite on.

The claim I want you to hold through the whole part is this: **generating a sequence of tokens is a sequential decision problem, and a verifier that scores the finished sequence is a reward function.** Once you believe that sentence, GRPO and RLVR stop looking like exotic machinery and start looking like the obvious thing to do.

## Theory

### The agent and environment interface

Reinforcement learning studies an *agent* that interacts with an *environment* over discrete time steps. At each step $t = 0, 1, 2, \dots$ the agent observes a state $S_t \in \mathcal{S}$, chooses an action $A_t \in \mathcal{A}$, and one step later receives a scalar reward $R_{t+1} \in \mathbb{R}$ and lands in a new state $S_{t+1}$. That is the entire contract. Everything the agent controls flows through $A_t$; everything the environment controls flows through the pair $(R_{t+1}, S_{t+1})$.

The choice to index the reward as $R_{t+1}$ rather than $R_t$ is a convention from Sutton and Barto that I am going to keep, because it saves grief later. The reward is a consequence of the action, so it is emitted by the environment at the same instant the next state is, and it shares that state's index. A trajectory (also called a history or, in the episodic case, an episode) is the interleaved sequence

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots \tag{1}
$$

The agent's behavior is summarized by a *policy* $\pi(a \mid s)$, a conditional distribution over actions given the state. That is all a policy is: a rule, possibly stochastic, for turning what you see into what you do.

### From arbitrary histories to Markov decision processes

In full generality the next state and reward could depend on the entire history. That is unworkable, so we assume the state is a sufficient statistic for the future. Formally, the environment satisfies the *Markov property* if

$$
\Pr\{S_{t+1}=s', R_{t+1}=r \mid S_0,A_0,\dots,S_t,A_t\} = \Pr\{S_{t+1}=s', R_{t+1}=r \mid S_t,A_t\}. \tag{2}
$$

The present state carries everything from the past that matters for what comes next. When (2) holds, the whole interaction is described by a single function, the *dynamics*:

$$
p(s', r \mid s, a) \;=\; \Pr\{S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a\}. \tag{3}
$$

A *Markov decision process* (MDP) is the tuple $(\mathcal{S}, \mathcal{A}, p, \gamma)$ where $p$ is the dynamics of (3) and $\gamma \in [0,1]$ is a discount factor I will justify in a moment. From $p$ you can derive every other quantity you might want, for instance the expected reward for a state-action pair,

$$
r(s,a) \;=\; \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a] \;=\; \sum_{r} r \sum_{s'} p(s', r \mid s, a). \tag{4}
$$

The Markov property is not a law of nature. It is a modeling choice about what you put in the state. If your chosen state leaves out something the future depends on, you have a *partially observable* problem wearing an MDP costume, and your algorithms will quietly misbehave. Keep that in your pocket; it comes back when I map LLMs onto MDPs, because there the choice is unusually clean.

### The reward hypothesis, and why a scalar is enough

Before the return, a word on the reward itself, because the design of $R$ is where "evals as rewards" lives. Reinforcement learning rests on what Sutton and Barto call the *reward hypothesis*: that everything we mean by goals and purposes can be encoded as the maximization of the expected value of a single scalar signal. This is a strong claim and it is easy to underrate. It says you do not get a vector of objectives, you do not get gradients from the environment, you get one number per outcome, and the agent's entire job is to make the expectation of that number large.

For my purposes the reward hypothesis is liberating rather than limiting, because an eval already *is* a scalarizer. A verifier that returns pass or fail, an Inspect scorer that returns a graded number in $[0,1]$, an exact-match check against a reference answer: each collapses a rich, messy transcript down to exactly the one scalar the reward hypothesis asks for. Parts III and IV were, in retrospect, an extended argument that these scalars are trustworthy and causally meaningful. Part V takes them at face value and feeds them in as $R$. The catch, which returns with force in Part VII on reward hacking, is that the agent optimizes the number you actually wrote down, not the intention behind it, so a sloppy scalar buys you a policy that games it.

### The return: what the agent actually maximizes

The agent does not maximize the next reward. It maximizes cumulative reward over the future. The *return* from time $t$ is

$$
G_t \;=\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \;=\; \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}. \tag{5}
$$

The single most useful fact about the return is that it obeys a one-step recursion. This recursion is the seed from which every Bellman equation in the next chapter grows, so I will derive it slowly.

```admonish derivation title="The return recursion"
Start from the definition (5) and peel off the first term:

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \\
    &= R_{t+1} + \gamma\left( R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots \right).
\end{aligned}
$$

The parenthesized series is exactly the definition of $G_{t+1}$, just with every index shifted up by one:

$$
G_{t+1} = R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+2}.
$$

Substituting gives the recursion I will lean on constantly:

$$
G_t = R_{t+1} + \gamma\, G_{t+1}. \tag{6}
$$

This is not an approximation and it does not assume the Markov property. It is pure algebra on the definition of the return. It says the value of the future decomposes into the immediate reward plus a discounted copy of the same problem one step later. Every dynamic-programming and temporal-difference method exploits exactly this self-similarity.
```

### Why discount, and what $\gamma$ buys you

The discount factor $\gamma$ does two jobs at once. First, it encodes a preference for sooner rewards over later ones, which is often what we actually want. Second, and more technically, it guarantees the infinite sum in (5) converges when rewards never stop. Suppose rewards are bounded, $|R_t| \le R_{\max}$ for all $t$. Then

$$
|G_t| \;\le\; \sum_{k=0}^{\infty} \gamma^k R_{\max} \;=\; \frac{R_{\max}}{1-\gamma}, \tag{7}
$$

using the geometric-series identity $\sum_{k=0}^\infty \gamma^k = 1/(1-\gamma)$ valid for $0 \le \gamma < 1$. So any $\gamma < 1$ makes the return a finite, well-defined random variable no matter how long the interaction runs. The quantity $1/(1-\gamma)$ has a nice reading as an *effective horizon*: with $\gamma = 0.99$ you are effectively planning about 100 steps ahead, with $\gamma = 0.9$ about 10 steps ahead.

```admonish derivation title="The geometric series bound, spelled out"
Let $S = \sum_{k=0}^{\infty}\gamma^k$. Multiply by $\gamma$: $\gamma S = \sum_{k=1}^{\infty}\gamma^k = S - 1$. Solve $\gamma S = S - 1$ for $S$ to get $S(1-\gamma) = 1$, hence $S = 1/(1-\gamma)$ whenever $|\gamma| < 1$. The bound (7) then follows by the triangle inequality applied term by term to (5), pulling $R_{\max}$ out of every term.
```

### Episodic versus continuing, unified

Two regimes show up. In *continuing* tasks the interaction runs forever and you need $\gamma < 1$ for the return to make sense. In *episodic* tasks the interaction breaks into finite episodes that end in a special *terminal* state, after which everything resets. Games, mazes, and, crucially, a single generated response are episodic.

You might think these need separate math, but there is a standard trick that unifies them. Model episode termination as entering an *absorbing state* that transitions only to itself and emits reward $0$ forever. Then a finite episode

$$
R_1, R_2, \dots, R_T, \underbrace{0, 0, 0, \dots}_{\text{absorbing}}
$$

has a return identical to the finite sum $\sum_{k=1}^{T}\gamma^{k-1}R_k$, because every added term is zero. With this device I can write the return as the infinite sum (5) in both regimes and never worry about which one I am in again. In episodic tasks I am also free to set $\gamma = 1$, since the sum is finite anyway, and RLHF-style setups almost always do exactly that.

### Mapping LLM generation onto an MDP

Now the payoff. I want to see a language model generating a completion as an agent moving through an MDP, because that is the lens that makes reward-based training legible.

Fix a prompt $x$, a token vocabulary $\mathcal{V}$, and an autoregressive model $\pi_\theta$. Define the pieces:

- **State.** The state at step $t$ is the prompt plus everything generated so far, $S_t = (x, a_0, a_1, \dots, a_{t-1})$. At $t=0$ the state is just the prompt, $S_0 = x$.
- **Action.** The action $A_t = a_t \in \mathcal{V}$ is the next token the model emits. The action space is the vocabulary, on the order of $|\mathcal{V}| \approx 1.5 \times 10^5$ tokens for a modern tokenizer (source: typical Llama-3 and Qwen-2 vocabulary sizes, roughly 128k to 152k entries).
- **Policy.** The policy is the model's next-token distribution, $\pi_\theta(a_t \mid S_t) = \pi_\theta(a_t \mid x, a_{<t})$, computed by a softmax over the output logits. The policy *is* the model. This identity is the whole reason RL applies to LLMs.
- **Transition.** The dynamics are *deterministic and known*: appending token $a_t$ to the state yields $S_{t+1} = (x, a_{<t}, a_t)$ with probability one. There is no stochastic environment fighting you. This is a very special MDP.
- **Termination.** The episode ends when the model emits an end-of-sequence token or hits a length cap $L_{\max}$. That terminal step is where the interesting reward lives.
- **Reward.** In the RLVR setting a verifier reads the finished completion and returns a scalar: $1$ if the answer checks out, $0$ otherwise, or some graded score from an Inspect scorer. Intermediate rewards are usually zero, so the reward is *sparse* and *terminal*.

Written out, the return for a generated sequence with a single terminal reward $R_T$ and $\gamma = 1$ collapses to something almost embarrassingly simple:

$$
G_0 = \sum_{t=1}^{T}\gamma^{t-1}R_t = R_T \quad \text{when } R_1 = \cdots = R_{T-1} = 0,\ \gamma = 1. \tag{8}
$$

Every token in the sequence shares the credit or blame for that one terminal number. The entire difficulty of policy-gradient methods, and the reason the next four chapters exist, is figuring out how to distribute that single scalar back across the hundreds of token-level decisions that produced it.

```admonish gotcha title="Token-level versus sequence-level actions"
There are two legitimate ways to slice the same generation, and they lead to genuinely different algorithms, so do not blur them.

**Token-level (fine-grained) MDP.** Each token is a separate action, exactly as above. The episode has $T$ steps, one per generated token. Value-based and actor-critic methods (PPO) live here, because they want a value estimate at every token position.

**Sequence-level (bandit) MDP.** Treat the whole completion as a *single* action drawn from a policy over sequences, $\pi_\theta(\mathbf{a}\mid x) = \prod_{t}\pi_\theta(a_t\mid x,a_{<t})$. Now there is one step, one action, one reward. This is a contextual bandit (Chapter 5.3), and it is the mental model behind GRPO, which scores whole completions and never asks about per-token value.

Both are correct. The sequence-level view is simpler and, on one 16GB GPU, cheaper, because it dodges a per-token critic. The token-level view gives finer credit assignment at the cost of a second network. Part V walks from one view to the other.
```

```admonish read-along title="Read-along: Sutton & Barto, Chapter 3"
This chapter is my compression of [S&B] Chapter 3, "Finite Markov Decision Processes." Read it alongside: their Sections 3.1 through 3.3 set up the agent-environment interface and the dynamics $p(s',r\mid s,a)$; Section 3.3 defines the return and gives the recursion (6); Section 3.4 handles the unified episodic and continuing notation with the absorbing-state trick I used above. When they introduce value functions in Section 3.5, stop and jump to my Chapter 5.2, which picks up exactly there. If you want the return recursion re-derived even more slowly, their Equation 3.9 is my (6).
```

## Tooling

The "tool" for this chapter is not a library, it is a discipline: writing an explicit environment wrapper so that generation and scoring share one interface. Even though a real LLM rollout runs through vLLM (Part II) and an Inspect scorer (Part III), it pays to prototype the MDP structure on something you can print to a terminal. The pattern I use everywhere later is a small class exposing `reset()`, `step(action)`, and a terminal reward, deliberately mirroring the Gymnasium API so that swapping the toy environment for the real generate-then-verify loop is a drop-in change.

Set up a scratch project with uv, no bare `pip` anywhere:

```bash
uv init rl-foundations
cd rl-foundations
uv add numpy matplotlib
```

## Lab

The artifact for 5.1 is a runnable sequence-generation MDP with a terminal reward, plus a function that computes discounted returns and verifies the recursion (6) numerically. It is intentionally not an LLM; it is the LLM's MDP skeleton with the transformer replaced by a stub, so that the *structure* is what you inspect. Later chapters bolt learning onto exactly this skeleton.

```python title="labs/mdp_skeleton.py"
"""A minimal sequence-generation MDP: the skeleton an LLM rollout fills in.

State  = prompt tokens + tokens generated so far (deterministic transition).
Action = next token from a tiny vocabulary.
Reward = terminal only, from a verifier that checks a target property.

Run:  uv run python labs/mdp_skeleton.py
"""
from __future__ import annotations
import numpy as np

VOCAB = ["0", "1", "<eos>"]          # tiny toy vocabulary
EOS = VOCAB.index("<eos>")
L_MAX = 6                            # length cap = forced termination


class SequenceMDP:
    """Deterministic, known-dynamics MDP over token sequences."""

    def __init__(self, prompt: str):
        self.prompt = prompt

    def reset(self):
        self.tokens: list[int] = []
        return self._state()

    def _state(self):
        # The Markov state is the full prefix: prompt + generated tokens.
        return (self.prompt, tuple(self.tokens))

    def step(self, action: int):
        """Append a token. Return (next_state, reward, done)."""
        self.tokens.append(action)
        done = action == EOS or len(self.tokens) >= L_MAX
        reward = self._verify() if done else 0.0   # sparse, terminal
        return self._state(), reward, done

    def _verify(self) -> float:
        """Toy verifier: reward 1.0 if the emitted bits (before <eos>)
        contain strictly more 1s than 0s, else 0.0."""
        bits = [t for t in self.tokens if t != EOS]
        ones = sum(1 for t in bits if VOCAB[t] == "1")
        zeros = sum(1 for t in bits if VOCAB[t] == "0")
        return 1.0 if ones > zeros else 0.0


def rollout(mdp: SequenceMDP, policy, rng: np.random.Generator):
    """Run one episode under a policy that maps state -> action probs."""
    state, done = mdp.reset(), False
    rewards, actions = [], []
    while not done:
        probs = policy(state)
        action = rng.choice(len(VOCAB), p=probs)
        state, reward, done = mdp.step(action)
        actions.append(action)
        rewards.append(reward)
    return actions, rewards


def discounted_return(rewards: list[float], gamma: float) -> float:
    """G_0 = sum_k gamma^k R_{k+1}, computed directly from the definition."""
    return sum((gamma ** k) * r for k, r in enumerate(rewards))


def returns_by_recursion(rewards: list[float], gamma: float) -> list[float]:
    """G_t = R_{t+1} + gamma * G_{t+1}, computed backwards. Returns G_t for all t."""
    g = 0.0
    out = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def uniform_policy(_state):
    # A deliberately dumb policy: uniform over the vocabulary.
    return np.full(len(VOCAB), 1.0 / len(VOCAB))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mdp = SequenceMDP(prompt="emit-mostly-ones:")

    gamma = 1.0
    for ep in range(5):
        actions, rewards = rollout(mdp, uniform_policy, rng)
        seq = "".join(VOCAB[a] for a in actions)
        g_def = discounted_return(rewards, gamma)
        g_rec = returns_by_recursion(rewards, gamma)[0]
        assert abs(g_def - g_rec) < 1e-12, "recursion must match definition"
        print(f"ep {ep}: seq={seq:<8} reward={rewards[-1]:.0f} "
              f"G0(def)={g_def:.2f} G0(rec)={g_rec:.2f}")

    # Empirical success rate of the uniform policy over many episodes.
    wins = sum(rollout(mdp, uniform_policy, rng)[1][-1] for _ in range(2000))
    print(f"\nuniform-policy success rate: {wins / 2000:.3f} "
          f"(this is J(pi) for the dumb policy; RL will push it up)")
```

**What you should see.** Five short episodes print with their generated bit-strings and a terminal reward of 0 or 1, and for every episode the return computed from the definition (5) matches the return computed from the recursion (6) to machine precision, which is the assertion firing silently. The final line reports the uniform policy's success rate, roughly $0.3$ to $0.4$, and that number is $J(\pi)$, the expected return of the current policy. Every later chapter is a machine for making that number go up. When you eventually swap `SequenceMDP` for a real vLLM rollout and `_verify` for an Inspect scorer, none of the surrounding structure changes, and that invariance is the entire point of writing it this way. If you want a timed rollout on the real stack instead of the toy, plug in the vLLM server from Part II and record throughput (measured on the baseline machine — record value, date, driver).

```admonish substack-seed
Post angle: "Your chatbot is playing a video game, and you already know the rules." The hook is that text generation is a Markov decision process where tokens are button presses, the screen never lies (transitions are deterministic and known, unlike Mario), and the score only appears at the very end when a grader looks at the whole run. Walk a general reader from the geometric-series intuition for discounting to the punchline that "one number at the end has to teach a thousand keystrokes," which is precisely the credit-assignment problem that the rest of the series solves. The MSI Aegis R2 with a single RTX 5080 16GB is the whole arcade cabinet, which is a fun visual for "you can do this at your desk."
```
