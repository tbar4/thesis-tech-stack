# Agentic RL: training the model to use its tools

```admonish business title="Business scope: outside the thesis"
This chapter is not part of the thesis contribution. Parts 0 through X are the thesis: the serve-evaluate-score-train loop and the causal reasoning-delta claim, complete and defensible on their own. Part XI is where that research becomes a product. Nothing here is a thesis claim, carries a thesis result, or is required to defend the thesis; it is forward-looking engineering for the business the loop is being commercialized into. It is written to be publishable: it shows the general technique, and deliberately keeps proprietary product specifics (customer workflows, internal system names, go-to-market) out.
```

In the thesis, the tools were always handed to the model. Chapter 8.2 built the MCP tool surface and a thin loop that *drives* the model through a tool call, and chapter 8.3's `+tools` arm measured what happens when a fixed policy is given that surface at inference. In both, the *decision* to call `screen_conjunction`, and the arguments to call it with, came from a harness and a prompt I wrote, not from a policy I trained. The model reasoned over tool outputs; it did not learn a tool-use strategy. That was the right choice for the thesis, because the thesis needed matched tool access across arms so the causal reasoning delta (7.6, 8.3) stayed clean. It is the wrong choice for a product, where I want an autonomous Space Domain Awareness (SDA) agent that decides *for itself* when a question needs the oracle, calls the tool correctly, and stops calling once it has what it needs.

This chapter closes that gap. It extends the RLVR machinery of Part V and the scorers-as-rewards join of 7.3 to *agentic* reinforcement learning: the policy's action space now includes emitting tool calls, the environment (the 8.2 FastMCP server) returns tool results mid-rollout, and the reward credits both the correct final answer and correct, efficient tool use. This is a capability I want in the product, an agent that uses its instruments well, and it is deliberately *not* a thesis claim, for a reason I will make precise in the theory: training the tool-calling policy confounds reasoning with tool strategy, which is exactly the confound 8.3 spent a whole chapter isolating. That confound is a feature here (I want the best agent) and a bug there (the thesis wants the clean parametric contrast), and that difference is why this work lives in Part XI and not in Part VII. Everything measured below is a *product* metric, reported with the book's placeholder "(measured on the baseline machine, record value, date, driver)", never a thesis result.

## Theory

### The action space grows: a rollout is now interleaved

In the thesis loop (5.9, equation 9.4) a rollout was a flat sequence: prompt, then a stream of policy-generated tokens, then a terminal reward. The action at each step was the next token, and the transition was a trivial append. Agentic RL keeps that skeleton but enlarges what a "step" can produce and who produces it. A rollout is now an *interleaved* sequence of four kinds of span:

$$
\tau \;=\; \big(\; x,\; a_1,\; o_1,\; a_2,\; o_2,\; \ldots,\; a_T,\; y \;\big),
\tag{1.1}
$$

where $x$ is the prompt, each $a_t$ is a policy-generated span (reasoning tokens followed, optionally, by a tool call in the model's tool-call syntax), each $o_t$ is a tool-result span *returned by the environment* (the 8.2 server calling `sda_core` and the 3.10 oracle), and $y$ is the final answer span the policy emits once it stops calling tools. The 8.2 tool-call loop is exactly the process that unrolls (1.1); agentic RL runs that same loop *inside* the training rollout instead of at inference only.

The load-bearing detail, and the one that separates a correct agentic-RL implementation from a broken one, is that the tool-result spans $o_t$ are **not policy actions**. The environment wrote them, not $\pi_\theta$, so the policy must not be rewarded or penalized for their tokens. This is precisely analogous to how the prompt tokens $x$ are excluded from the loss: I never ask the policy to explain the prompt it was given, and I must not ask it to explain the oracle output it was handed. Concretely, I attach a token-level mask $m$ to the trajectory,

$$
m_i \;=\;
\begin{cases}
1 & \text{token } i \text{ lies in some policy span } a_t \text{ or in the answer } y,\\[2pt]
0 & \text{token } i \text{ lies in the prompt } x \text{ or any tool-result span } o_t,
\end{cases}
\tag{1.2}
$$

and every per-token term in the GRPO objective is multiplied by $m_i$ before it is summed. TRL already carries a `completion_mask` that zeroes padding tokens after the end-of-sequence marker so they contribute no gradient; agentic RL is, mechanically, that same mask *extended* to also zero the environment-authored tool-result tokens. Get this mask wrong and the policy is trained to predict the oracle's numbers as if it had generated them, which teaches it to hallucinate tool outputs, the exact opposite of what I want.

### The reward: correctness dominant, tool use instrumental

The thesis reward (7.3) summed a dominant correctness term and a small dense format term. The agentic reward keeps correctness dominant and adds two tool-use terms, one that pays for a *valid, necessary* tool call and one that *charges* for waste:

$$
r(\tau) \;=\;
\underbrace{r_{\text{correct}}\big(y, \text{oracle}\big)}_{\text{objective (3.10 verdict)}}
\;+\;
\underbrace{\lambda_{\text{tool}} \sum_{t=1}^{T} \mathbb{1}\big[\text{call } t \text{ valid} \wedge \text{necessary}\big]}_{\text{tool-use shaping}}
\;-\;
\underbrace{\lambda_{\text{cost}}\big(n_{\text{redundant}} + c\,\ell\big)}_{\text{cost: redundancy + latency}},
\tag{1.3}
$$

where $r_{\text{correct}}$ is the same 3.10 oracle verdict the thesis verifier uses (the `conjunction` boolean matched against gold), $\mathbb{1}[\cdot]$ is 1 only when a call parsed to valid arguments *and* changed the answer the model would otherwise have given, $n_{\text{redundant}}$ counts repeat or superfluous calls, $\ell$ is a latency proxy (wall-clock or summed tool time), and $\lambda_{\text{tool}}, \lambda_{\text{cost}}, c$ are small weights kept well below the correctness scale, for the ratio reason derived in 7.3 (GRPO sees component *ratios*, not absolute scale).

```admonish note title="A naive tool-use bonus is a reward-hacking magnet"
The reward-hacking lesson of 7.4 transfers directly, and the tool-use term is a textbook trap. If $\lambda_{\text{tool}}$ simply pays for *any* valid call, the policy discovers the cheapest possible hack: call the tool on every prompt, whether or not the question needs it, and collect the bonus. The training reward climbs while the agent learns to be pointlessly chatty with its instruments, which is a worse product, not a better one. Two disciplines defuse it, and I use both. First, gate the bonus on *necessity*: a call earns $\lambda_{\text{tool}}$ only when it *changes the answer*, that is, when removing the tool result would flip the model's verdict, so a call that made no difference is worth nothing. Second, keep correctness strictly dominant and treat tools as purely instrumental: the only reason to call a tool is that it raises $r_{\text{correct}}$, and the shaping term is a small nudge toward doing so cleanly, never a payout the policy can live on. The held-out check from 7.4 still applies: watch a strict, tools-blind correctness verifier alongside the training reward, and if the tool-call rate climbs while held-out correctness stalls, the bonus is being gamed.
```

### Credit assignment across a multi-step trajectory

GRPO's group-relative advantage (5.9, and the derivation in 7.3) needs no change in form. For a prompt $x$ I sample a group of $G$ full trajectories $\{\tau_1, \ldots, \tau_G\}$, each unrolled through however many tool calls it chose to make, score each with the scalar return $R(\tau_i) = r(\tau_i)$ from equation (1.3), and center within the group:

$$
A_i \;=\; \frac{R(\tau_i) - \bar R}{\operatorname{std}(R) + \epsilon}, \qquad \bar R = \frac{1}{G}\sum_{j=1}^{G} R(\tau_j).
\tag{1.4}
$$

The advantage $A_i$ is a single scalar per trajectory, broadcast to every *unmasked* token of that trajectory (the $m_i = 1$ tokens of (1.2)). This is **trajectory-level** credit assignment: the whole interleaved rollout, all its reasoning and all its tool calls, shares one return, and the group baseline is what tells a good trajectory from a bad one at matched prompt. It is the simplest thing that works and it is what I ship first. **Turn-level** credit, assigning separate advantages to individual tool-calling steps, is the obvious refinement (a trajectory that made one brilliant call and three wasteful ones deserves per-turn discrimination that a single scalar cannot express), but it needs a per-turn value estimate or a process reward, which reintroduces machinery GRPO deleted to fit on 16GB (the value model, per the 5.9 VRAM budget). I keep credit at the trajectory level for the same budget discipline as 7.2, and I let the cost term in (1.3) carry the "punish the wasteful calls" signal that turn-level credit would otherwise provide.

### Why this is a product concern, not a thesis one

Here is the reason the fence around this Part is not bureaucratic. The thesis claim (8.3) is a *causal* statement isolated at **matched tool access**: the `+tools` arm gives every model the same tool surface, so any measured reasoning delta cannot be explained by one arm having better tools than another. The four-arm design spends its whole effort holding tool access fixed precisely so the parametric reasoning channel is the only thing that moves.

Training the tool-calling policy does the opposite on purpose. When I optimize (1.3), the resulting checkpoint is better at SDA questions for two entangled reasons at once: it may reason better *and* it certainly uses tools better, and no single-arm contrast can separate the two, because I changed both in the same gradient. On the 8.3 causal graph, this reopens the tool-access and injected-context-correctness paths that the thesis worked to hold shut, and it does so through the weights, so I could not even attribute it with the matched-budget machinery. That is a fine, in fact desirable, thing for a product: I want the best agent, and I do not care to decompose *why* it is best. It is a disqualifying confound for the thesis's clean parametric claim. So the same operation, "train the model to use its tools," is a product win and a thesis contaminant, and that is exactly why it sits in Part XI, downstream of a defense, and not in Part VII.

## Tooling

The stack is the book's, reused wholesale, because the point of Part XI is a smooth research-to-product transition and not a new toolchain. GRPO comes from TRL exactly as in 7.2 (Unsloth-backed, LoRA, `fast_inference` vLLM), the tools are the 8.2 FastMCP server wired in as the rollout *environment*, the reward is the 3.10 Skyfield oracle verdict from 7.3 plus the tool-use terms of (1.3), vLLM serves the rollouts, and MLflow tracks the run. The 16GB budget discipline of 7.2 is unchanged: LoRA or QLoRA adapters, a small group size $G$, capped completion length, and the same `gpu_memory_utilization` headroom the thesis runs used.

The one genuinely new seam is that the rollout is no longer TRL's single-turn generate. A vanilla `GRPOTrainer` generates one completion per prompt and scores it; an agentic rollout must run the 8.2 tool-call loop (generate a span, dispatch any tool call to the MCP session, inject the result, generate again) and return both the concatenated token ids *and* the extended loss mask of (1.2). I get this by subclassing `GRPOTrainer` and overriding the generation step so it delegates to the 8.2 loop, then hands TRL a `completion_mask` that is already zeroed on the tool-result spans. Everything downstream of the mask (the group-relative advantage, the clipped objective, the KL leash) is untouched TRL. The tool-calling loop is the 8.2 loop reused inside the training rollout, which is the whole reason 8.2 kept that loop small and protocol-clean.

```admonish gotcha title="The tool-result mask is the entire correctness of the method"
Two failure modes both look like "the agent got worse," and both trace to the mask of equation (1.2). First, if the tool-result tokens are left *unmasked* ($m_i = 1$ on $o_t$), GRPO trains the policy to predict the oracle's output, so under sampling the model starts *fabricating* tool results that look like the oracle's JSON instead of waiting for the real call, which is catastrophic for a grounded agent. Second, if the mask is misaligned by even a few tokens (an off-by-one where the tool-call closing tag or a chat-template boundary is counted on the wrong side), the advantage is applied to the wrong tokens and training silently degrades. Build the mask from the *same* tokenization the rollout emitted, span by span, never by re-tokenizing the joined string (chat templates and special tokens do not round-trip token-for-token), and unit-test it by asserting that the masked-in token count equals the sum of the policy spans' lengths before you spend any GPU time. This is the agentic-RL analogue of the 7.3 gotcha about parsing chat-formatted completions: the seam between the rollout and the loss is where the subtle, silent bugs live.
```

## Lab

The lab is a minimal agentic-RL training loop on the conjunction-screening task. Each rollout lets the policy call the 8.2 `screen_conjunction` MCP tool, the environment injects the oracle result, the policy continues reasoning and commits a verdict, and the reward is the 3.10 oracle verdict plus a small validated-tool-call bonus minus a redundant-call cost, per equation (1.3), with tool-result tokens masked from the loss per (1.2). This is a *product* artifact (a tool-use-trained adapter and a behavior report), not a thesis result, and every headline number is a placeholder.

Set up under a new lab project, reusing the 7.2 training stack and the 8.2 MCP sub-project as a dependency.

```bash title="shell"
uv init labs/agentic-rl-sda
cd labs/agentic-rl-sda
uv add "unsloth" "unsloth_zoo" "vllm" "trl" "peft" "bitsandbytes" mlflow
uv add "mcp[cli]" skyfield sgp4            # the 8.2 server + 3.10 oracle deps
uv add --editable ../../mcp                # the 8.2 FastMCP tools + sda_core
uv lock
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export SDA_MODE=pinned                     # reproducible oracle ground truth (8.2)
export SDA_SNAPSHOT=$(cat ../../mcp/.snapshot_hash)
```

**The reward function**, extending 7.3's. Correctness reuses the 3.10 oracle verdict exactly; the tool-use shaping and cost terms of (1.3) read the rollout's recorded tool-call trace, which the environment attaches to each completion.

```python title="labs/agentic-rl-sda/agentic_rewards.py"
"""Agentic reward (eq 1.3): dominant 3.10 oracle correctness + a necessity-gated
tool bonus - a redundancy/latency cost. PRODUCT reward, not a thesis result.

Weights are ratios (7.3): correctness=1.0 dominates; tool terms are small nudges.
The rollout attaches a `trace` per completion: the list of tool calls it made,
each with {name, args, valid, changed_answer, seconds}.
"""
from __future__ import annotations

import re

CORRECT_W = 1.0        # objective: the oracle verdict (dominant)
TOOL_W = 0.15          # pay a VALID + NECESSARY call (necessity-gated, per the note)
COST_W = 0.10          # charge redundant calls
LATENCY_W = 0.02       # charge summed tool seconds (a gentle efficiency pressure)

_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def _final_verdict(text: str) -> str | None:
    m = _VERDICT_RE.search(text or "")
    return m.group(1).upper() if m else None


def correctness_reward(prompts, completions, gold_verdict, traces, **kwargs):
    """3.10 oracle correctness: does the model's final verdict match gold."""
    out = []
    for c, gold in zip(completions, gold_verdict):
        v = _final_verdict(c if isinstance(c, str) else c[-1]["content"])
        out.append(CORRECT_W if (v is not None and v == str(gold).upper()) else 0.0)
    return out


def tool_use_reward(prompts, completions, traces, **kwargs):
    """Necessity-gated bonus minus redundancy/latency cost (eq 1.3 tool terms).

    A call scores the bonus only if it parsed to valid args AND changed the
    answer; repeat/superfluous calls and slow calls are charged. This is the
    anti-hacking shape from 7.4: a pointless call is worth <= 0, never > 0.
    """
    out = []
    for trace in traces:
        bonus = TOOL_W * sum(
            1 for call in trace if call["valid"] and call["changed_answer"]
        )
        n_redundant = sum(
            1 for call in trace if not call["valid"] or not call["changed_answer"]
        )
        latency = sum(call["seconds"] for call in trace)
        out.append(bonus - COST_W * n_redundant - LATENCY_W * latency)
    return out


REWARD_FUNCS = [correctness_reward, tool_use_reward]
```

**The rollout and environment glue.** This runs the 8.2 tool-call loop against the policy, records the tool trace the reward reads, and builds the loss mask of (1.2) span by span from the tokens the rollout actually emitted. It returns, per prompt, the completion token ids and the aligned mask that zeroes tool-result tokens.

```python title="labs/agentic-rl-sda/rollout.py"
"""Agentic rollout: run the 8.2 tool loop with the trained policy as client,
inject oracle results as the environment, and build the (1.2) loss mask.

The mask is the whole ballgame (see the gotcha): tool-result spans are the
ENVIRONMENT's tokens, so m_i = 0 there, exactly like prompt tokens. We build it
from the emitted token ids span by span, never by re-tokenizing the joined text.
"""
from __future__ import annotations

import json
import time

import sda_core          # the 8.2 data+physics core (screen_conjunction -> 3.10 oracle)

TOOL_NAME = "screen_conjunction"


def _oracle_call(args: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    a = sda_core.get_tle(args["norad_id_a"])
    b = sda_core.get_tle(args["norad_id_b"])
    result = sda_core.screen_conjunction(
        a, b, window_h=args.get("window_h", 24),
        threshold_km=args.get("threshold_km", 5.0),
    )
    return result, time.perf_counter() - t0


def agentic_rollout(engine, tokenizer, prompt_ids, sampling, max_turns=4):
    """One trajectory (eq 1.1). Returns (completion_ids, mask, trace, text).

    engine.generate(...) is the in-process vLLM engine (Unsloth fast_inference),
    used the same way 7.2 generates, but here re-invoked each turn.
    """
    ids: list[int] = []           # completion token ids (everything after prompt)
    mask: list[int] = []          # (1.2): 1 = policy span, 0 = tool-result span
    trace: list[dict] = []
    context_ids = list(prompt_ids)
    prev_verdict = None

    for _ in range(max_turns):
        # 1) policy span a_t: the model reasons and maybe emits a tool call.
        out = engine.generate(context_ids, sampling)     # returns token ids
        span = out.token_ids
        ids += span
        mask += [1] * len(span)                           # policy tokens: unmasked
        context_ids += span
        text = tokenizer.decode(span)

        call = _parse_tool_call(text)                     # None if the model answered
        if call is None:
            break

        # 2) environment span o_t: the oracle result, injected as a tool message.
        args = call["args"]
        valid = _args_valid(args)
        result, seconds = ({}, 0.0)
        changed = False
        if valid:
            result, seconds = _oracle_call(args)
            verdict = "YES" if result.get("conjunction") else "NO"
            changed = verdict != prev_verdict              # did the call move the answer?
            prev_verdict = verdict
        tool_msg = _render_tool_message(result if valid else {"error": "invalid args"})
        tool_ids = tokenizer.encode(tool_msg, add_special_tokens=False)
        ids += tool_ids
        mask += [0] * len(tool_ids)                        # ENVIRONMENT tokens: masked
        context_ids += tool_ids
        trace.append({"name": TOOL_NAME, "args": args, "valid": valid,
                      "changed_answer": changed, "seconds": seconds})

    assert len(ids) == len(mask)                           # alignment invariant
    return ids, mask, trace, tokenizer.decode(ids)
```

**The TRL GRPO wiring.** A thin `GRPOTrainer` subclass swaps the single-turn generation for `agentic_rollout` and hands TRL a `completion_mask` already zeroed on tool-result spans. Everything after the mask (advantage, clip, KL) is stock TRL, exactly as 7.2 used it.

```python title="labs/agentic-rl-sda/train_agentic.py"
"""GRPO with an agentic (tool-calling) rollout. PRODUCT run, not a thesis result.
Same 7.2 budget: Unsloth LoRA, small group, capped length, one 16GB card."""
from unsloth import FastLanguageModel  # noqa: E402

import mlflow
import torch
from trl import GRPOConfig, GRPOTrainer

from agentic_rewards import REWARD_FUNCS
from rollout import agentic_rollout
from data import load_conjunction_split          # (norad_a, norad_b, gold_verdict)

MODEL, LORA_R, GROUP = "unsloth/Qwen3-4B", 16, 8


class AgenticGRPOTrainer(GRPOTrainer):
    """Override generation to run the tool loop and extend the loss mask (1.2).

    TRL already multiplies the per-token loss by `completion_mask`; we return a
    mask that is 0 on tool-result tokens, so environment tokens get no gradient,
    exactly as prompt tokens never do."""

    def _rollout_group(self, prompt_ids, sampling):
        comps, masks, traces = [], [], []
        for _ in range(self.args.num_generations):
            ids, mask, trace, _text = agentic_rollout(
                self.llm, self.processing_class, prompt_ids, sampling,
                max_turns=4,
            )
            comps.append(ids); masks.append(mask); traces.append(trace)
        # `traces` rides through to the reward funcs as a passthrough column (7.3).
        return comps, masks, {"traces": traces}


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=4096, load_in_4bit=True,
        fast_inference=True, max_lora_rank=LORA_R, gpu_memory_utilization=0.16,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_R, use_gradient_checkpointing="unsloth",
    )

    train_ds = load_conjunction_split("train")   # product split; NOT the 3.11 frozen suite

    cfg = GRPOConfig(
        output_dir="outputs", num_generations=GROUP,
        per_device_train_batch_size=GROUP, gradient_accumulation_steps=2,
        max_prompt_length=512, max_completion_length=2048,   # room for tool turns
        learning_rate=5e-6, beta=0.02, epsilon=0.2, loss_type="dr_grpo",
        temperature=1.0, max_steps=400, logging_steps=1, save_steps=100,
        report_to="mlflow", run_name="agentic-sda-r16", seed=0,
    )

    torch.cuda.reset_peak_memory_stats()
    mlflow.set_experiment("p11-product")          # product experiment, fenced off
    with mlflow.start_run(run_name="agentic-sda-r16"):
        mlflow.log_params({"model": MODEL, "lora_r": LORA_R, "group": GROUP,
                           "tool_w": 0.15, "cost_w": 0.10, "part": "XI-product"})
        trainer = AgenticGRPOTrainer(
            model=model, processing_class=tokenizer,
            reward_funcs=REWARD_FUNCS, args=cfg, train_dataset=train_ds,
        )
        trainer.train()
        mlflow.log_metric("vram_peak_gib", torch.cuda.max_memory_allocated() / 2**30)
        model.save_lora("outputs/agentic-sda-adapter")
        mlflow.log_artifacts("outputs/agentic-sda-adapter", artifact_path="lora")


if __name__ == "__main__":
    main()
```

```bash title="shell"
uv run python train_agentic.py
```

```admonish gotcha title="Train on a product split, never the 3.11 frozen suite"
The 3.11 frozen suite is the thesis's measurement instrument, and it stays sacred here for the same reason it did in 7.3: 8.3 and 7.6 score the thesis arms on it, and any weight update that saw those items contaminates the thesis numbers. This product run trains on a *separate* conjunction split drawn from the pinned 8.2 snapshot, and it reports its own product metrics on its own product held-out set. The fence between the thesis and the product is a data fence too: the agentic adapter must never touch a frozen-suite item, so that a defense committee reading Part VIII sees numbers this Part could not have moved.
```

**What you should see.** The training run logs the two reward functions separately in MLflow (`rewards/correctness_reward/mean` and `rewards/tool_use_reward/mean`), the same split-and-watch discipline as 7.3 and 7.4, so I can see correctness lead while the tool term settles. The behavior signals are the product story: a **tool-call rate** (fraction of rollouts that call `screen_conjunction` at least once) that should rise from wherever the base policy sits toward "calls when the question is numeric, abstains when it is not," an **unnecessary-call rate** (calls with `changed_answer=False`) that should *fall* as the cost term of (1.3) bites, and a **final accuracy** on the product held-out set. Every one of those is a placeholder in this write-up: record the tool-call rate, the unnecessary-call rate, the final accuracy, the mean tool calls per trajectory, the peak VRAM, and the step time, all with the snapshot hash, date, and driver (measured on the baseline machine, record value, date, driver). The natural product comparison is the trained agentic adapter against the 8.3 `+tools` arm (the *handed*-tools baseline): the thesis arm was given the tool and told to use it, this agent decided to, and the interesting product metric is whether learned tool-use matches or beats handed tool-use at a lower unnecessary-call rate. The artifacts are `outputs/agentic-sda-adapter/` (the tool-use-trained LoRA adapter) and a short agent-behavior report (the reward curves plus the tool-call and unnecessary-call rates over training). Both are product artifacts. Neither is a thesis result, and neither is scored on the frozen suite.

```admonish read-along
**[RLHF] ch. 7** is the backing for this chapter's training side: it covers reinforcement learning for tool use and agents, the reward-function-versus-reward-model distinction that the 5.9 chapter built on, and the reasoning-and-acting recipes that equation (1.1)'s interleaved trajectory formalizes. Read its treatment of multi-step, tool-using rollouts against the loss-mask detail of (1.2): [RLHF] motivates *why* you train the policy to act, and this chapter's mask is the concrete answer to *how* you keep the environment's tokens out of the gradient while doing so. One caution worth stating plainly: agentic RL is a fast-moving area, and the specific mechanics (turn-level credit, tool-call formats, rollout frameworks) are shifting quickly, so treat the trajectory-level, single-tool loop here as a deliberately minimal, correct baseline to build the product on, not as the settled state of the art.
```

```admonish substack-seed title="Product angle, post-contract (not for the pre-defense thesis series)"
This seed is a *business* post, to be written after the commercial contract lands and explicitly outside the thesis Substack series that runs before the defense. The hook: "Handing a model a tool is not the same as teaching it to use one." The thesis proved a reasoning gain at matched tool access, which required *giving* every model the same tools; the product needs the opposite, an agent that decides for itself when to reach for its instruments. The post would walk the reward of equation (1.3), correctness dominant, a necessity-gated tool bonus, a redundancy-and-latency cost, and land the counterintuitive engineering point that the hard part is not rewarding tool use but *penalizing* it correctly, because a naive tool bonus is the easiest reward hack in the book (the model just calls the tool constantly). The payoff figure is the unnecessary-call rate falling over training while accuracy holds: an agent learning restraint, which is the behavior a customer actually pays for. It would keep the customer specifics out and show only the general technique, and it would be clear that this is product engineering downstream of a thesis that already stands on its own.
```

## The thesis stands without this Part

I want to close by restating the fence, because it is the reason this chapter exists in the book at all. Parts 0 through X are the thesis, and they are complete without a single line of Part XI: the serve-evaluate-score-train loop is closed (5.9, 7.8), the causal reasoning delta is measured and defended at matched tool access (7.6, 8.3), and none of those results depends on the agent trained here. This chapter takes the same machinery one step past the thesis, into a product capability (an autonomous SDA agent that learns its own tool-use policy), and that step deliberately confounds the very channels the thesis worked to separate, which is why it is a product win and a thesis contaminant at once, and why it is fenced into this Part. Everything measured here is a product metric, carried with the "(measured on the baseline machine, record value, date, driver)" placeholder and never presented as thesis evidence. The genuinely proprietary parts of the product (the customer workflows this agent runs inside, the internal system it plugs into, the go-to-market) stay out of the book and live in internal documents; what remains here is the general, publishable technique, written down now so the research-to-product transition is smooth if the contract lands before the defense. The thesis does not wait on any of it.
