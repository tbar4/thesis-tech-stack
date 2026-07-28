# The post-training landscape

A pretrained language model is a spectacular autocomplete and almost nothing else. It has read a large slice of the internet and learned to continue text in a way that is statistically faithful to what it saw, which means it will happily continue a math problem with a plausible-looking wrong answer, continue a question with three more questions, or continue a polite request with a refusal it scraped from some forum. Everything I actually want from a model I use, following instructions, staying helpful, showing its reasoning, stopping when it is done, is bolted on after pretraining, in a phase the field calls post-training. This part of the book is about that phase done the way a single-GPU shop can afford it, and this opening chapter draws the map: what pretraining leaves me, what the canonical SFT then reward-model then RL pipeline adds, why the direct-alignment shortcut collapses two of those stages into one, and where reasoning training (the thing this whole thesis is chasing) actually sits on that map.

## Theory

### Pretraining sets the prior; post-training sets the behavior

Pretraining optimizes one objective, next-token prediction, over a corpus so large that the model has to learn general structure to compress it: syntax, facts, code, arithmetic patterns, the shape of an argument. I derived that objective in Part I as the per-token cross-entropy averaged over the corpus,

$$\mathcal{L}_{\text{pre}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\!\left[\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(x_t \mid x_{<t})\right], \tag{6.1.1}$$

and the crucial thing about equation (6.1.1) is that it treats every token in the corpus as equally worth predicting. The model learns $p(\text{text})$, the distribution of internet-shaped strings, and that is a prior over language, not a policy for being useful. The knowledge is in there. A base model can do arithmetic and cite facts. What it lacks is any reason to prefer the helpful continuation over the merely probable one, because during pretraining those were the same thing by construction.

Post-training changes the objective, not the architecture. I keep the same transformer and the same weights as a starting point, and I retrain (usually a small fraction of the weights, via the LoRA machinery of the next two chapters) against a signal that says which continuations are wanted. The three classical ways to supply that signal define the three classical stages, and they differ in exactly one respect: how expensive and how informative the supervision is.

### The adaptation continuum, and when not to climb it

Before I draw the three-stage pipeline, it helps to zoom out one level, because retraining the weights is only one region of a larger space. **[CUST]** ch. 1-2 frames every way of adapting a model to a task as a single continuum, a ladder from the cheapest and least invasive intervention to the most. The bottom rung is prompt engineering: change nothing about the model, just change what I put in the context. The next rung is retrieval, feeding the model relevant documents at inference time (RAG, which I build in chapter 8.1). Above that sits supervised fine-tuning, where I finally touch the weights but only via the parameter-efficient adapters of chapters 6.2 and 6.3. Higher still is preference tuning (DPO, chapter 6.5), and at the top is full reinforcement learning. Each rung buys a deeper, more durable change in behavior, and each charges more for it in data, compute, and engineering surface. The whole of this Part VI, and the RLVR loop the thesis builds in Part VII, live near the top of that ladder. The retrieval and tool-use machinery of Part VIII lives near the bottom, which is exactly why those chapters are cheaper to stand up and why I reach for them first when a task does not obviously demand new weights.

The reason to keep the continuum in view is that it reframes the opening question. A book that is entirely about training makes it tempting to assume the answer is always "train," but **[CUST]** ch. 1.5 is blunt that fine-tuning carries real and recurring costs: curated data I have to build and keep, compute I have to spend every time the base model updates, a maintenance burden that does not exist for a prompt, and the standing risk that pushing the weights toward one behavior quietly degrades the general ability I started with. Often the honest answer is a lower rung. The first question is therefore not "how do I fine-tune" but "should I move the weights at all, or is prompting or retrieval already enough." That is the same discipline this thesis enforces everywhere else through `evalstats`: the intervention has to actually cause a measured gain over the cheaper baseline, or it does not earn its place. Climbing a rung I did not need is not a neutral choice, it is cost and risk I took on for a delta I never verified.

None of this is a ladder I climb once and abandon the rungs below. **[CUST]** ch. 1.7 and 2.3 treat choosing a starting technique as a matter of matching the rung to the use case rather than always reaching for the highest one, and ch. 2.2 makes the point that production systems usually stack rungs rather than pick a single one: a retrieval layer feeding a fine-tuned model, say, each doing the part it is best at. That is precisely the shape of the four-arm comparison I run in chapter 8.3, where trained weights, retrieval, and tools are separate arms I can combine and measure against each other rather than rival religions. A grounding detail makes the mapping concrete: **[CUST]** does its worked examples on the Qwen3 instruct family, the same models this book serves and trains, so its guidance about which rung a given task wants lands on the same hardware reality my labs do, the single 16GB card and everything that fits on it.

### The SFT then RM then RL pipeline

The pipeline that produced the first generation of instruction-following assistants has three stages, run in order, each consuming the output of the last.

**Stage 1, supervised fine-tuning (SFT).** I collect prompt-response pairs where the response is a demonstration of the behavior I want, and I train the model to imitate the response with the same cross-entropy objective as pretraining, except that I only score the response tokens (the masking derivation is the spine of the next chapter). This is cheap to run and its supervision is dense: every token of every demonstration is a gradient signal. Its ceiling is set by the demonstrations. The model learns to sound like the data, so if the data is a few thousand curated instruction-response pairs, I get an instruction-follower whose quality is bounded by those examples. SFT teaches format, tone, and the basic contract of "answer the question," and it is the cold start for everything that follows.

**Stage 2, the reward model (RM).** Demonstrations are expensive and they cap the model at human-writing quality. Preferences are cheaper: it is far easier for an annotator to look at two responses and say which is better than to write the ideal response from scratch. So I collect pairs $(y_w, y_l)$, a "winner" and a "loser" for the same prompt, and I train a separate model, the reward model, to assign a scalar score $r_\phi(x, y)$ that ranks winners above losers. Chapter 6.4 derives that training objective from the Bradley-Terry model of pairwise choice. The RM is not the thing I ship; it is a learned, differentiable stand-in for human judgment that I can query millions of times.

**Stage 3, reinforcement learning (RL).** Now I optimize the SFT model (the policy) to produce responses the reward model scores highly, using the policy-gradient machinery of Part V, with a KL penalty pinning the policy near the SFT model so it does not wander off into gibberish that happens to fool the RM. The canonical objective, the one the whole DPO derivation in chapter 6.5 starts from, is

$$\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[\, r_\phi(x, y) \,\big] \;-\; \beta\, \mathbb{D}_{\text{KL}}\!\big[\pi_\theta(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big]. \tag{6.1.2}$$

The reward signal here is sparse (one scalar per whole response) and the optimization is *on-policy*: the model generates its own attempts and is graded on them, rather than being handed the answer. That is what lets RL push past the demonstration ceiling: the model can discover responses better than anything in the SFT set, as long as the RM can recognize them. It is also what makes RL finicky, because a sparse learned reward is exactly the kind of thing a capable optimizer learns to hack, a theme I return to in 6.4 and again in Part VII.

```admonish derivation title="Why the three stages are ordered the way they are"
The ordering is forced by an information-versus-cost tradeoff. Write the supervision each stage provides as a number of bits per training example. SFT gives roughly $T$ tokens times $\log_2 V$ bits of vocabulary constraint per example, dense and cheap, but every bit is a bit about *imitating the demonstration*, so the target distribution is capped at the demonstrator. A preference label gives exactly one bit per comparison ("$y_w \succ y_l$"), which is far sparser, but it is a bit about *relative quality*, a dimension SFT cannot express because SFT has no notion of "worse." RL then spends those quality bits on-policy, where the model's own samples supply the exploration that neither of the earlier stages had.

So the pipeline is a curriculum in supervision type: dense-imitation first to get a competent, on-format starting policy (you cannot usefully RL a base model that does not yet answer questions, because its samples are all off-distribution and the reward signal is noise); then a cheap quality signal distilled into an RM; then sparse on-policy optimization against that signal, fenced by the KL term of equation (6.1.2) so the dense knowledge from stages 1 and pretraining is not destroyed. Skip stage 1 and stage 3 has nothing coherent to explore from. Skip stage 2 and stage 3 has no reward to optimize. The order is not a convention, it is a dependency chain.
```

### The direct-alignment shortcut

Stages 2 and 3 together are a lot of moving parts: a second model to train and hold in memory, a sampling loop, a KL estimator, a value function or a group-baseline, all of the PPO or GRPO plumbing from Part V. On a 16GB card this is genuinely painful, because the RM and the policy and the reference and the optimizer state are all competing for the same VRAM. The direct-alignment insight, which chapter 6.5 derives in full, is that for the specific objective in equation (6.1.2) you can solve for the optimal policy in closed form, substitute it back, and discover that the reward model is *implicit* in the policy itself. The upshot is a single supervised-style loss on the preference pairs, no separate RM, no sampling loop, no RL. DPO (Direct Preference Optimization) is the canonical member of that family, and it is why a single-GPU shop can do preference alignment at all: it turns stages 2 and 3 into one more SFT-shaped training run.

The tradeoff is real and I will be honest about it throughout: direct alignment optimizes against a *fixed* set of preferences (the ones you collected), so it cannot explore beyond them the way on-policy RL can. It is offline. For alignment-of-style it is usually enough and dramatically cheaper. For pushing capability on tasks where you can *verify* correctness, on-policy RL earns its cost back, which is exactly the regime this thesis lives in.

### Where reasoning training sits

The reason this thesis exists is that a fourth kind of supervision changed the picture: verifiable rewards. If the task is a math problem or a unit-tested coding problem, I do not need a learned reward model at all, because I have a *programmatic* checker that says right or wrong with no ambiguity and no hackable slack (or much less of it). That collapses stage 2 entirely: the reward is a function I wrote, the same scorer my evals use. Stage 3 becomes RL against that verifier, which the field calls RLVR (reinforcement learning with verifiable rewards) and which Part V builds up to. The whole conceit of the book, evals as rewards, is the observation that the scorer in my eval harness and the reward function in my training loop are the same object viewed from two angles.

On the landscape, then, reasoning training sits like this. Pretraining gives the prior. SFT (chapter 6.2) is the cold start, and for reasoning models the SFT set is increasingly *reasoning traces*, long chain-of-thought demonstrations, sometimes distilled from a bigger teacher (chapter 6.7). LoRA and QLoRA (chapter 6.3) are how I afford any of this on 16GB. Reward models and DPO (chapters 6.4, 6.5) are the alignment-of-style branch I mostly do not need for verifiable tasks but must understand, because they are the intellectual parent of the RLVR objective and because reward hacking is the failure mode I have to defend against. Inference-time scaling (chapter 6.6) is the orthogonal axis: even with a fixed model, I can spend more compute per answer (sampling more, verifying, refining) and get better accuracy, and any honest comparison of a trained model against a base model has to hold that inference budget fixed. Distillation (chapter 6.7) is the pragmatic finale: at small scale, copying a good teacher's traces via SFT often beats running RL from scratch, and it is the natural extension once the core contract of the thesis loop is met.

That is the whole part in one paragraph. The rest of it fills in the derivations and the labs, but the map does not change: dense imitation to cold-start, a quality signal (learned or verifiable) to define the objective, on-policy optimization or its offline shortcut to chase that objective, and inference-time compute as a separate knob I must control for whenever I claim a delta.

## Tooling

There is no single tool for "post-training." There is a small, coherent stack that the rest of this part uses, and it is worth naming now so the labs do not feel like they appear from nowhere.

The training side is three libraries that layer cleanly. **PEFT** (Parameter-Efficient Fine-Tuning) supplies LoRA and QLoRA: it wraps a frozen base model and injects the small trainable adapters chapter 6.3 derives. **TRL** (Transformer Reinforcement Learning) supplies the training loops shaped like each stage: `SFTTrainer` for stage 1, `RewardTrainer` for the RM, `DPOTrainer` for direct alignment, and `PPOTrainer`/`GRPOTrainer` for the RL stage. **Unsloth** sits underneath both as an optimization layer: it patches the model's attention and MLP kernels and its LoRA path with hand-written Triton kernels and a memory-frugal backward pass, which is what makes a 4B model trainable in 16GB rather than 24GB. On the baseline machine (MSI Aegis R2, RTX 5080 16GB Blackwell, 32GB DDR5, Ubuntu 24.04, NVIDIA 570-open) Unsloth is not optional; it is the difference between the labs running and OOM-ing. I dig into its internals in Part VII; here I just use it.

The evaluation side is the machinery I already built. The `evalstats` module from chapter 3.7 is the reused core: paired-difference tests and bootstrap confidence intervals over the thesis task suite, so that every "this fine-tune helped" claim in this part comes with an interval and a p-value rather than a vibe. Every lab from 6.3 onward closes the loop by running the frozen task suite before and after training and reporting the delta through `evalstats`, which is the concrete meaning of "evals as rewards" at the level of my own workflow: the eval is how I know the training did anything.

Everything is a `uv` project, one directory per lab, following the two-environment doctrine: the training environment (Unsloth/TRL/PEFT, which pins a specific torch) stays separate from the inference/eval environment (vLLM, Inspect), because their dependency graphs fight. The landscape lab below only needs the light inference side.

## Lab

The point of this lab is to *see* post-training, not to do it yet. A base model and its instruction-tuned sibling share an architecture and most of their weights, but post-training leaves fingerprints, and the two most legible ones are the chat template (the exact string format the model was tuned to expect) and the behavior on a bare instruction. I will load a base model and its instruct counterpart, dump both tokenizers' chat templates, and generate a short completion from each on the same raw prompt, then write the comparison to disk. This makes concrete the claim that the knowledge is already in the base model and post-training supplies the contract for getting at it.

This is a `uv` project. From the repo root:

```bash
uv init labs/landscape-fingerprint
cd labs/landscape-fingerprint
uv add transformers torch accelerate
```

```python title="labs/landscape-fingerprint/fingerprint.py"
"""Show the post-training fingerprint: chat template + raw-prompt behavior
for a base model vs its instruction-tuned sibling.

Writes artifacts/fingerprint.md and prints a short summary. CPU-runnable,
but far faster on the baseline GPU.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same family, same size: only post-training differs.
BASE = "Qwen/Qwen3-4B-Base"
INSTRUCT = "Qwen/Qwen3-4B"
PROMPT = "Give me three tips for sleeping better."

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


@torch.no_grad()
def raw_continue(tok, model, prompt: str, n_new: int = 120) -> str:
    """Feed the prompt with NO chat template: pure autocomplete."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=n_new, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def chat_answer(tok, model, prompt: str, n_new: int = 120) -> str:
    """Feed the prompt through the model's own chat template."""
    if tok.chat_template is None:
        return "(no chat template on this tokenizer)"
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=n_new, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    lines = ["# Post-training fingerprint\n"]
    for label, name in [("BASE", BASE), ("INSTRUCT", INSTRUCT)]:
        tok, model = load(name)
        has_template = tok.chat_template is not None
        lines.append(f"## {label}: `{name}`\n")
        lines.append(f"- has chat template: **{has_template}**")
        lines.append(f"\n**Raw autocomplete on {PROMPT!r}:**\n")
        lines.append("```\n" + raw_continue(tok, model, PROMPT).strip() + "\n```\n")
        lines.append("**Through the chat template:**\n")
        lines.append("```\n" + chat_answer(tok, model, PROMPT).strip() + "\n```\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (OUT / "fingerprint.md").write_text("\n".join(lines))
    print(f"Artifact: {(OUT / 'fingerprint.md').resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python fingerprint.py
```

```admonish gotcha
The base model often has *no* chat template at all (`tok.chat_template is None`), which is itself the point: the template is a post-training artifact, added by whoever did the instruction tuning, not a property of the architecture. If you force a chat template onto a base model by borrowing the instruct model's, you will get a plausible-looking answer, because the knowledge is present, but the model was never trained to honor the special tokens, so quality and stop behavior are unreliable. That mismatch, a template at inference the weights never saw in training, is one of the most common silent-failure modes in this whole part, and chapter 6.2 makes it a first-class citizen.
```

**What you should see.** The artifact `artifacts/fingerprint.md` lays the two models side by side. The base model's raw autocomplete on "Give me three tips for sleeping better." tends to *continue the prompt as text*, often with more list-like questions or a meandering paragraph, because equation (6.1.1) only ever taught it to be plausible. Its chat-template answer is erratic or empty because it has no template. The instruct model's raw autocomplete is already somewhat answer-shaped (post-training leaks into the base behavior), and its chat-template answer is a clean, bounded, three-item list that stops when done. Same architecture, same rough weights, one round of post-training in between: that visible gap is the entire subject of this part. Record wall-clock and peak VRAM for the 4B loads on your run (measured on the baseline machine, record value, date, driver), since two 4B models in BF16 is about 16 GiB of weights alone and you may need to load them one at a time on the 16GB card, which the script already does by deleting each model before the next.

```admonish read-along
This chapter maps to **[RLHF]** ch. 1–3: ch. 1 for the framing of post-training as a distinct phase with its own objectives, ch. 2 for the SFT-then-preference pipeline as a whole, and ch. 3 for the problem setup and notation (policy, reference, reward, KL) that equation (6.1.2) uses and that every later chapter in this part inherits. Read those three alongside this map before diving into the derivations that start in 6.2. **[AIE]** ch. 2's post-training section gives the same map at survey altitude (why SFT then preference tuning, and what each stage buys); it is a useful cross-check that the taxonomy here is the field's consensus rather than this book's invention.

**[CUST] Bahree & Tok, *LLM Customization and Fine-Tuning*** widens the frame: its adaptation continuum (chapters 1 and 2) places this whole part on a ladder that starts below it, at prompting and retrieval, and its "when not to adapt" section is the honest counterweight a training-focused book needs. Read it for the decision that comes before any of the derivations here, which is whether to touch the weights at all, and note that it works on the same Qwen3 instruct family this book serves, so its examples land on the same hardware reality.
```

```admonish substack-seed
"A base model already knows how to help you. It just has no reason to." The post that starts from equation (6.1.1), the claim that pretraining learns $p(\text{text})$ and not a policy, and walks the reader through the fingerprint lab: same 4B model, base versus instruct, same prompt, and the base one answers your question by asking three more. Then the reveal that the entire multi-billion-dollar apparatus of RLHF is three ideas stacked, imitate, rank, optimize, and that the cheap direct-alignment shortcut and the verifiable-reward variant are both just clever ways to skip the expensive middle. A whole field's shape in one screenful, anchored on a demo anyone can rerun.
```
