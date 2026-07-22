# Distillation

There is a slightly deflating result at the heart of small-scale reasoning work, and it is worth stating plainly at the top: for a 4B model on a 16GB card, the fastest way to a good reasoning model is often not to run reinforcement learning at all, but to copy a bigger model's homework. Distillation, in the form this chapter cares about, means taking reasoning *traces* from a strong teacher model and using them as SFT data for a weaker student. It is not the classic logit-matching distillation of small-image-classifier fame (though I will connect the two); it is trace distillation, and when the traces are good it routinely beats from-scratch RL at this scale, for a reason the theory below makes precise. This chapter derives why, connects rejection sampling as the quality filter that makes distillation work, and places distillation where it belongs in the thesis: the *post-contract extension*, the thing I do once the core serve-evaluate-score-train loop is proven, to squeeze out more capability than my own RLVR run can find on its own.

## Theory

### Two kinds of distillation

Classical knowledge distillation trains a student to match a teacher's full output *distribution*: for each input, minimize the KL from the teacher's softmax (softened by a temperature) to the student's, so the student learns not just the teacher's top answer but its whole spread of probabilities, the "dark knowledge" in the runner-up logits. The loss is

$$\mathcal{L}_{\text{KD}} = \mathbb{E}_{x}\Big[\tau^2\, \mathbb{D}_{\text{KL}}\big(p_T^{\tau}(\cdot \mid x)\,\|\,p_S^{\tau}(\cdot \mid x)\big)\Big], \tag{6.7.1}$$

where $p^{\tau}$ is the temperature-$\tau$ softened distribution. This is beautiful and it needs the teacher's *logits*, which for a closed API teacher you do not have and for an open teacher costs you a synchronized forward pass. For LLM reasoning at small scale the dominant, practical variant is coarser and cheaper: **sequence-level distillation**, where the teacher just *generates* full reasoning traces and the student does ordinary SFT (chapter 6.2, masked cross-entropy) on them. You throw away the soft distribution and keep only the sampled sequence, which is a lossy approximation of equation (6.7.1) but requires nothing but the teacher's text output. Every "distilled" open reasoning model you have seen is this: a strong model generated chains of thought, and a smaller model was fine-tuned to imitate them.

```admonish derivation title="Why trace distillation is SFT on a better data distribution"
Sequence-level distillation is exactly the SFT objective of equation (6.2.3), with one change: the demonstrations come from the teacher policy $\pi_T$ instead of from humans. The student minimizes

$$\mathcal{L}_{\text{distill}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D},\, (z, a) \sim \pi_T(\cdot \mid x)}\Big[\log \pi_\theta(z, a \mid x)\Big], \tag{6.7.2}$$

where $(z, a)$ is a teacher-generated trace-plus-answer and the loss is masked to the completion as always. This is maximum likelihood, so its fixed point is the student policy closest (in forward KL) to the teacher's trace distribution that the student's capacity can represent. Compare that to the human-demonstration SFT of chapter 6.2, which targets the *human* trace distribution. The distillation win is entirely about *which distribution you are imitating*: a strong reasoning teacher produces traces that are correct more often, structured more consistently, and formatted the way a verifier wants, so imitating $\pi_T$ moves the student toward a better demonstrator than a pile of human-written or scraped traces would.

Now the comparison to RL, which is the point of the chapter. RLVR (Part V) optimizes the student's own samples against a verifier: it needs the student to *already* produce correct traces with nonzero probability so the reward signal is informative (the cold-start argument of chapter 6.2), and it explores from there. Its per-example signal is one scalar (correct/incorrect), sparse. Distillation's per-example signal is a *full correct trace*, dense: every token of a good teacher trace is a gradient toward the right procedure, equation (6.7.2) versus the one-bit reward of RL. So at small scale, where the student's own solve rate is low and RL's exploration is slow and noisy, distillation's dense supervision from a teacher that has *already solved the problem* is simply more sample-efficient. RL earns its cost when there is no teacher better than you, or when you want to push past the teacher; distillation earns its cost when a better teacher exists and you just want its capability cheaply. At 4B on 16GB, a better teacher almost always exists.
```

### Rejection sampling: the quality filter

Teacher traces are not uniformly good. A strong teacher still gets some problems wrong, and imitating a wrong trace teaches a wrong *procedure*, which is worse than teaching a wrong fact because the student generalizes the bad reasoning. The fix, and the thing that makes distillation actually work, is **rejection sampling**: generate teacher traces *only for problems with known answers*, verify each trace with the same checker my evals use, and keep only the traces that reach the correct answer. This is where the "evals as rewards" identity pays off yet again, the verifier that scores my benchmark is the filter that curates my distillation set. Formally, I am sampling from the teacher conditioned on correctness,

$$\pi_T^{\text{filt}}(z, a \mid x) \propto \pi_T(z, a \mid x)\,\mathbb{1}[\,a = a^{\star}(x)\,], \tag{6.7.3}$$

where $a^{\star}(x)$ is the ground-truth answer and $\mathbb{1}[\cdot]$ is the verifier. Distilling on $\pi_T^{\text{filt}}$ instead of raw $\pi_T$ removes the wrong-procedure traces entirely, and it is strictly better data for the same reason chapter 6.2 preached quality over quantity. Note that this is exactly rejection-sampling fine-tuning (the "RFT" / "STaR"-style loop) when the teacher *is* a stronger checkpoint of the student itself: sample, keep the correct ones, SFT on them, repeat. Self-distillation via rejection sampling is a real and cheap capability lever, and it needs no reward model and no RL machinery, just a verifier and a sampling budget.

### Forward KL, off-policy data, and the exposure-bias caveat

There is a subtlety in equation (6.7.2) worth surfacing, because it explains both why distillation works and where it frays. Maximum-likelihood imitation minimizes the *forward* KL, $\mathbb{D}_{\text{KL}}(\pi_T \,\|\, \pi_\theta)$, which is mode-covering: the student is penalized for putting *low* probability anywhere the teacher puts *high* probability, so it tries to cover all of the teacher's behavior, including modes it cannot represent well at 4B. That is usually benign for reasoning traces (the teacher's mass is concentrated on good traces after rejection sampling) but it is why a heavily distilled small model can feel like it is imitating a register it does not fully command. The deeper issue is *exposure bias*, the same one that afflicts all SFT: the student is trained on the teacher's trajectories but at inference it conditions on *its own* prefix, and once it makes a token the teacher never would, it is off the training distribution with no supervision for how to recover. On-policy distillation variants address this by having the teacher score or correct the *student's own* rollouts rather than only its own, which reintroduces some of RL's on-policy benefit at more cost. For the thesis scale I stick with off-policy trace distillation plus rejection sampling and accept the exposure-bias caveat, because the alternative pulls me back toward the RL machinery distillation was supposed to avoid. Naming the caveat is what keeps the "distillation beats RL here" claim honest: it beats RL on sample-efficiency at low solve rates, not on robustness at the margins.

### The subtle costs

Distillation is not free of failure modes, and honesty requires naming them. First, the student cannot exceed the teacher's *reachable* capability by imitation alone; equation (6.7.2)'s fixed point is bounded by the teacher's trace distribution, so pure distillation asymptotes at "a smaller, faster version of the teacher on this task," not beyond it. Second, distilled traces can induce *style over substance*: the student learns to produce teacher-shaped reasoning that *looks* right, including on problems where it does not actually follow the logic, which is a form of the same proxy-gaming risk from chapter 6.4 wearing a different hat. Third, there are real licensing and terms-of-service constraints on distilling from closed models that a thesis has to respect. The mitigations are rejection sampling (equation 6.7.3, keep only verified-correct traces), mixing distilled data with your own RLVR-discovered traces so the student is not purely imitative, and evaluating on held-out problems the teacher never saw so you measure generalization, not memorized teacher tics.

### Where distillation sits in the thesis

The thesis's core deliverable is the closed loop: serve a model, evaluate it, use the scorer as a reward, train with RLVR on 16GB, re-evaluate, measure the reasoning delta with the chapter 3.7 statistics. Distillation is deliberately *outside* that contract, and it is the natural *extension* once the contract is met, for two reasons. First, it is the strongest baseline the RLVR result must be compared against: if a simple rejection-sampling distillation from a good teacher matches or beats my RLVR run at matched inference budget (chapter 6.6), then my RL contribution is honest work only if I report that. Second, it composes: the best small reasoning models are distillation *then* RLVR, cold-start the student on verified teacher traces to lift its solve rate, then run RLVR to push past the teacher on the tasks where the student's exploration finds something new. So distillation is both the baseline I must beat and the on-ramp that makes my RL cheaper, which is exactly why it closes this part rather than opening it.

### Sampling temperature and the diversity of the distillation set

One knob quietly decides how much a distillation run is worth: the teacher's sampling temperature during trace generation. At temperature zero the teacher emits its single most-likely trace per problem, which is high-quality but low-diversity, so the student sees essentially one way to solve each problem and learns a narrow procedure. At a higher temperature (0.7 to 1.0) the teacher produces varied traces, different solution paths, different phrasings, which teaches the student a *distribution* over ways to reason rather than one canonical path, and that diversity is what makes the distilled student robust to problems that need a slightly different approach. Rejection sampling (equation 6.7.3) makes higher temperature safe, because the wrong traces that temperature also produces get filtered out, leaving a diverse-but-correct set. So the recipe is: sample warm for diversity, verify hard for correctness, and keep multiple correct traces per problem when they take genuinely different paths. This is the distillation-set analogue of the quality-and-coverage argument that ran through the SFT and reward-model chapters, and it is why the lab samples at temperature 0.8 with $k=4$ rather than greedily.

## Tooling

Distillation reuses the entire stack already built, which is the point. Generation of teacher traces is a vLLM job in the inference environment: batched sampling of $k$ traces per problem, the same machinery as chapter 6.6's best-of-n. Verification and filtering (equation 6.7.3) is the Inspect scorer from Part III, the same verifier that is the eval and the reward. Training on the filtered traces is TRL's `SFTTrainer` with a LoRA adapter in the training environment, byte-for-byte the chapter 6.2 lab with a different dataset. Evaluation of the distilled student against the RLVR student and the SFT baseline scores each in-process model on the frozen suite with `thesis_suite.score_model_local` (no served endpoint, so I grade the resident checkpoint directly) and feeds the paired per-item scores to chapter 3.7's `evalstats.bootstrap_paired_diff` and `evalstats.mcnemar` at a matched budget. There is no new library here, distillation is a *composition* of the tools, which is itself the lesson: once the loop's components exist, distillation is a data-curation pattern over them, not a new system.

## Lab

The lab builds a rejection-sampled distillation set from a teacher, SFT-distills the 4B student on it, and compares the distilled student against the plain SFT cold-start on the frozen thesis suite, so the "distillation beats vanilla SFT (and is a real baseline for RL)" claim is measured with a CI. The artifacts are the filtered trace dataset and the eval delta.

This spans both environments. First, generate and filter teacher traces (inference environment):

```bash
uv init labs/distill
cd labs/distill
uv add vllm datasets
uv add --editable ../../packages/evalstats
```

```python title="labs/distill/gen_traces.py"
"""Generate teacher reasoning traces and reject the wrong ones (equation 6.7.3).
Writes artifacts/distill_set.jsonl of verified-correct traces.
"""
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from datasets import load_dataset

# Teacher: a stronger open reasoning model the license permits distilling from.
TEACHER = "Qwen/Qwen3-14B"
K = 4                                  # traces per problem; keep the correct ones
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)

from thesis_suite import verify_sda_answer   # chapter-3.9 verifier


def main() -> None:
    # Problems WITH known answers (rejection sampling needs a^*(x)).
    problems = load_dataset("openai/gsm8k", "main", split="train[:2000]")
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096)
    sp = SamplingParams(n=K, temperature=0.8, top_p=0.95, max_tokens=1024)

    prompts = [p["question"] for p in problems]
    outs = llm.generate(prompts, sp)

    kept = 0
    with (OUT / "distill_set.jsonl").open("w") as f:
        for prob, out in zip(problems, outs):
            gold = prob["answer"].split("####")[-1].strip()
            for comp in out.outputs:
                if verify_sda_answer(comp.text, gold):     # keep correct only
                    f.write(json.dumps({
                        "instruction": prob["question"],
                        "output": comp.text,
                    }) + "\n")
                    kept += 1
                    break     # one verified trace per problem is enough
    total = len(problems)
    print(f"kept {kept}/{total} verified traces "
          f"({100*kept/total:.1f}% solve rate on the teacher)")
    print(f"Artifact: {(OUT / 'distill_set.jsonl').resolve()}")


if __name__ == "__main__":
    main()
```

```bash
uv run python gen_traces.py
```

Then distill and compare (training environment; the `distill_set.jsonl` carries over):

```python title="labs/distill/distill_sft.py"
"""SFT-distill the 4B student on verified teacher traces, then compare against
the plain cold-start SFT on the frozen suite. Writes artifacts/distill_delta.txt.
"""
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

import evalstats as es
# chapter-3.9 frozen suite: load the frozen v1.0 items, and grade an in-process
# model over them (no served endpoint) with the same verifiers 3.9 froze.
from thesis_suite import load_suite, score_model_local

BASE = "unsloth/Qwen3-4B-Base"
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
SUITE = load_suite()                # frozen thesis suite v1.0 (chapter 3.9)


def suite_scores(model, tok) -> list[int]:
    """Per-item 0/1 correctness over the frozen suite, in fixed suite order so
    the distilled and baseline students yield matched pairs for evalstats.
    Reduces the `{id, response, correct, difficulty}` rows of
    `score_model_local` to the binary vector McNemar and the bootstrap consume."""
    return [int(r["correct"]) for r in score_model_local(model, tok, SUITE)]


def student():
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=2048, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    return model, tok


def train_on(ds, tag):
    model, tok = student()
    # Prompt/completion message lists: SFTTrainer renders + masks the prompt
    # (no packing over a pre-rendered `text`, which would score the whole seq).
    ds = ds.map(lambda ex: {
        "prompt": [{"role": "user", "content": ex["instruction"]}],
        "completion": [{"role": "assistant", "content": ex["output"]}]},
        remove_columns=ds.column_names)
    cfg = SFTConfig(output_dir=str(OUT / f"run_{tag}"), max_seq_length=2048,
                    completion_only_loss=True,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    num_train_epochs=1, learning_rate=2e-4, warmup_ratio=0.03,
                    lr_scheduler_type="cosine", bf16=True, logging_steps=25,
                    report_to="none", seed=3407)
    SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
    FastLanguageModel.for_inference(model)
    return model, tok


def main() -> None:
    # Distilled student: trained on verified teacher traces.
    distill_ds = load_dataset(
        "json", data_files=str(OUT / "distill_set.jsonl"), split="train")
    dm, dtok = train_on(distill_ds, "distill")
    distill_scores = suite_scores(dm, dtok)
    del dm; torch.cuda.empty_cache()

    # Baseline student: plain human-demo SFT on the same Alpaca slice as
    # chapter 6.2, loaded straight from HF (6.2 trains on this slice but does
    # not dump a jsonl, so read it here rather than from a nonexistent file).
    base_ds = load_dataset("yahma/alpaca-cleaned", split="train[:3000]")
    bm, btok = train_on(base_ds, "baseline")
    base_scores = suite_scores(bm, btok)
    del bm; torch.cuda.empty_cache()

    est = es.bootstrap_paired_diff(distill_scores, base_scores, level=0.95)
    mc = es.mcnemar(distill_scores, base_scores, exact=True)
    report = (f"distill_acc {sum(distill_scores)/len(distill_scores):.4f}\n"
              f"baseline_acc {sum(base_scores)/len(base_scores):.4f}\n"
              f"delta {est.point:+.4f} 95% CI [{est.ci_low:+.4f}, {est.ci_high:+.4f}]\n"
              f"mcnemar p {mc.pvalue:.4f}\n")
    (OUT / "distill_delta.txt").write_text(report)
    print(report)
    print(f"Artifact: {(OUT / 'distill_delta.txt').resolve()}")


if __name__ == "__main__":
    main()
```

```bash
uv run python distill_sft.py
```

```admonish gotcha
The distillation comparison is only honest if the distilled student and the baseline student are evaluated on problems the *teacher never generated traces for*. If your thesis-suite items overlap with the GSM8K training problems you distilled on, you are measuring memorization, not reasoning transfer, which is the contamination failure mode from Part III wearing a distillation costume. Keep the distillation source problems and the eval suite strictly disjoint, and prefer an eval suite drawn from a different distribution than the distillation set so a real generalization delta is what you measure. This is the single most common way a distillation result is accidentally inflated.
```

**What you should see.** The teacher's solve rate on the source problems (printed by `gen_traces.py`) tells you your yield: a 14B teacher on GSM8K-style problems should verify-correct on the large majority of items, so you keep on the order of 1500 to 1800 traces from 2000 problems. The distilled student, trained on those verified traces, should beat the plain-SFT baseline on the frozen suite by a delta whose bootstrap CI excludes zero and whose McNemar p-value is small, the measured statement that dense correct traces beat generic instruction data for reasoning. Both training runs fit the chapter 6.3 QLoRA budget; the teacher generation is the memory-heavy step and runs in the inference environment where the 14B model in BF16 (about 28 GiB) exceeds 16 GiB, so on the baseline machine you either quantize the teacher to 4-bit to fit or generate the traces during a Lambda burst (Part VIII) and bring the `distill_set.jsonl` home. Record teacher solve rate, kept-trace count, both training wall-clocks, and the eval delta with its CI (measured on the baseline machine, record value, date, driver). The `distill_delta.txt` artifact is the baseline number every RLVR result in Part VII must be reported against.

```admonish read-along
Pair this with **[BRM]** ch. 8, the distillation chapter, which develops trace distillation and rejection-sampling fine-tuning for reasoning models in depth and situates them against RL, exactly the "when does distillation beat RL at small scale" question equation (6.7.2)'s derivation answers here. Read it as the capstone of the reasoning-methods arc that ran through **[BRM]** ch. 4–7, and as the direct setup for this book's Part VII, where distillation is the baseline the RLVR loop is measured against.
```

```admonish substack-seed
"At small scale, the fastest path to a good reasoning model is to copy a better model's homework, and check the answers before you copy." A post on why sequence-level distillation (equation 6.7.2, SFT on a teacher's traces) beats from-scratch RL for a 4B model: RL gives you one bit of signal per attempt and needs you to already be able to solve the problem, while a good teacher hands you a full correct chain of thought, dense supervision toward the right procedure. The catch and the fix in one move: teachers are sometimes wrong, so rejection-sample with a verifier (equation 6.7.3) and keep only the traces that reach the right answer, which is the same verifier that scores your evals and rewards your RL. Distillation, evaluation, and reward turn out to be the same object again.
```
