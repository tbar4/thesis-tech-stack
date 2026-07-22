# SFT and instruction tuning

Supervised fine-tuning is the least glamorous stage in this part and the one everything else depends on. It is just next-token prediction again, the same objective from Part I, pointed at a small pile of curated prompt-response pairs instead of the whole internet. The subtlety is entirely in the details that a from-the-textbook description skips: which tokens actually get a gradient, how the prompt gets wrapped so the model recognizes it, how you pack short examples so you do not waste the GPU, and why a few thousand good demonstrations beat a few million scraped ones. Get those right and SFT is a reliable way to turn a base model into a competent instruction-follower and, more to the point for this thesis, into a cold-start policy that RLVR can actually explore from. Get the masking wrong and you will train the model to parrot your prompts, which is a bug that produces a loss curve that looks perfect and a model that is quietly broken. So this chapter derives the masked cross-entropy loss carefully, then builds a LoRA SFT of a 4B model end to end.

## Theory

### Chat templates are part of the data, not a wrapper around it

A base model saw raw text. An instruction model has to be told, in-band, where the user's turn ends and its own turn begins, and it learns that boundary from special tokens inserted by a *chat template*. A chat template is a deterministic function that takes a list of role-tagged messages and renders them into one flat string with delimiter tokens. Qwen3, for instance, renders a user turn roughly as `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`, and the model is trained so that the tokens after that final `assistant\n` are its job to produce. The template is not cosmetic. It is the exact string distribution the model's post-training weights expect, and if I train with one template and serve with another (or serve a base model through a template it never saw, the gotcha from 6.1) the model is off-distribution and everything degrades. So in SFT the template is applied at *training* time, the same call the serving stack will make at inference, and the two must be byte-identical. This is why the labs always render through `tokenizer.apply_chat_template` rather than hand-building strings.

### Masked cross-entropy: only score the completion

Here is the single most important idea in the chapter. An SFT example is a (prompt, response) pair, rendered by the template into one token sequence. If I trained the plain language-modeling loss of equation (6.1.1) on that whole sequence, I would be teaching the model to predict the *prompt* tokens too, which is worse than useless: it wastes capacity learning to generate the very questions I am going to hand it at inference, and it dilutes the gradient that should be teaching it to answer. The fix is a per-token mask that zeroes out the loss on the prompt and keeps it only on the response. Let me derive exactly what that does to the objective and the gradient, because the mask is not a heuristic, it is a change to the likelihood being maximized.

```admonish derivation title="Masked cross-entropy for SFT"
Take one rendered example as a token sequence $x = (x_1, \dots, x_T)$. Define a binary mask $m_t \in \{0, 1\}$ that is $1$ on completion (assistant) tokens and $0$ on prompt (user/system/template) tokens. The autoregressive factorization of the whole sequence is unchanged,

$$\log \pi_\theta(x) = \sum_{t=1}^{T} \log \pi_\theta(x_t \mid x_{<t}), \tag{6.2.1}$$

but I do not want to maximize the joint likelihood of the sequence. I want to maximize the *conditional* likelihood of the response given the prompt. Split the sequence into a prompt part $x_{\le p}$ and a completion part $x_{>p}$ (with $m_t = 1$ exactly for $t > p$ in the simple single-turn case). The conditional is

$$\log \pi_\theta(x_{>p} \mid x_{\le p}) = \sum_{t=p+1}^{T} \log \pi_\theta(x_t \mid x_{<t}) = \sum_{t=1}^{T} m_t \,\log \pi_\theta(x_t \mid x_{<t}), \tag{6.2.2}$$

where the last equality is the whole trick: multiplying each term by $m_t$ drops the prompt terms exactly, and the surviving terms are unchanged because $\pi_\theta(x_t \mid x_{<t})$ *already conditions on the full prefix* $x_{<t}$, prompt included. Masking the loss does not hide the prompt from the model; the prompt still flows through attention as context. It only removes the prompt tokens as prediction *targets*. That distinction is the entire point.

Averaging over a dataset $\mathcal{D}$ of examples and dividing by the number of scored tokens gives the SFT objective as a masked, length-normalized cross-entropy,

$$\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\!\left[\frac{\sum_{t=1}^{T} m_t \,\log \pi_\theta(x_t \mid x_{<t})}{\sum_{t=1}^{T} m_t}\right]. \tag{6.2.3}$$

Now the gradient, so it is clear what the mask does to learning. With $z_t$ the logits at position $t$ and $\mathrm{softmax}(z_t)$ the predicted distribution, the per-position gradient of cross-entropy with respect to the logits is the familiar "prediction minus one-hot,"

$$\frac{\partial \mathcal{L}_{\text{SFT}}}{\partial z_t} \;\propto\; m_t\,\big(\mathrm{softmax}(z_t) - e_{x_t}\big), \tag{6.2.4}$$

where $e_{x_t}$ is the one-hot vector for the true next token. The mask multiplies the *entire* per-position gradient, so prompt positions contribute exactly zero to the update. Practically this means the loss you watch during training is an average over completion tokens only, and it is directly comparable across examples only if you normalize by $\sum_t m_t$ as in equation (6.2.3), which is why the choice between per-token and per-example averaging (mean over tokens versus mean over sequences) actually changes what you optimize, and why mismatched normalization between two runs makes their loss curves incomparable.
```

The implementation of equation (6.2.4) in every framework is a sentinel label. The tensor of target labels is set to the true token id on completion positions and to a special ignore index (in PyTorch, `-100`) on prompt positions; `torch.nn.functional.cross_entropy` skips any position whose label is `-100`. So "masking the loss" is concretely "set the prompt labels to `-100`," and the single most common SFT bug is forgetting to do it, which trains equation (6.1.1) on the whole sequence and gives you a model that continues prompts instead of answering them.

### Sequence packing

Instruction examples vary wildly in length. If I pad every example in a batch to the longest one, a batch with one 2000-token example and seven 200-token examples wastes roughly 78% of the compute on padding tokens that contribute nothing (their labels are `-100` too). Packing fixes this: concatenate multiple short examples into one fixed-length sequence (say 2048 tokens), back to back, so there is little or no padding. The catch is attention. If example B is packed right after example A in the same 2048-token window, and I use ordinary causal attention, B's tokens can attend back into A, which is cross-contamination: B learns to "predict" its answer partly from an unrelated earlier example. The correct fix is a block-diagonal attention mask (often called an attention mask for packing, or handled by FlashAttention's variable-length `cu_seqlens` path) that forbids attention across the example boundaries, so each packed example attends only to itself.

```admonish under-the-hood
Modern SFT trainers implement packing through FlashAttention's variable-length API rather than a materialized 2D mask. Instead of building a $T \times T$ boolean mask (which would cost $T^2$ memory and defeat the purpose), they pass a `cu_seqlens` vector marking where each example starts and stops inside the packed buffer, and the kernel restricts each query's attention to its own segment. That is why packing with a naive `attention_mask` and packing with proper `cu_seqlens` can give different loss curves for the same data: the naive version leaks context across boundaries. TRL's `SFTTrainer` with `packing=True` on a FlashAttention-2 backend does the right thing; if you hand-roll packing you must supply the boundaries or you are training on contaminated context.
```

Packing buys real throughput (often 2 to 4 times, depending on your length distribution) at zero cost to correctness when done right, and on a 16GB card where every token of compute is scarce, that throughput is the difference between an overnight run and a two-day one.

### Dataset quality over quantity

The uncomfortable empirical fact about SFT is that it is far more sensitive to the *quality* of demonstrations than to their *count*. The intuition follows straight from equation (6.2.3): SFT is imitation, so its fixed point is the demonstrator's distribution. A thousand carefully written, correct, well-formatted, stylistically consistent responses move the model toward a thousand-example-quality demonstrator. A million noisy, contradictory, format-inconsistent responses move it toward a noisy demonstrator, and no amount of them buys past that ceiling, they only teach the average of the mess. This is the "quality over quantity" result that repeated small-set fine-tuning studies keep rediscovering: a few thousand curated examples routinely beat orders of magnitude more scraped ones for instruction-following. For reasoning models the same logic applies with teeth, because a reasoning demonstration encodes not just the answer but a whole chain of thought, and one wrong step in a long trace teaches the model a wrong *procedure*, not just a wrong fact.

### Hyperparameters that matter, and catastrophic forgetting

SFT has few knobs but two of them decide whether it helps or harms. The first is the number of epochs. SFT sets are small, and the temptation is to train for many passes to squeeze them, but a small set trained for too long overfits: the model memorizes the demonstrations, its held-out loss turns up while its train loss keeps falling, and it starts reproducing training answers verbatim regardless of the actual question. One to three epochs is the usual safe range, and I watch a held-out slice of the SFT data to catch the turn. The second is the learning rate, which for LoRA SFT (roughly $1\times10^{-4}$ to $2\times10^{-4}$) is much higher than full fine-tuning would tolerate, because the low-rank adapters of chapter 6.3 start from zero and need a large step to move at all. The deeper risk behind both is *catastrophic forgetting*: aggressive SFT on a narrow set can erode the broad capabilities pretraining installed, so the model gets better at your instruction format and worse at everything else. LoRA is itself a partial defense (the frozen base cannot be overwritten, only added to), which is one more reason every SFT in this book is a LoRA rather than a full fine-tune, and it is why I evaluate on the broad thesis suite after SFT and not only on held-out instruction data: the number I care about is whether general reasoning survived the cold start.

### SFT as the cold start for RLVR

This is why SFT matters to the thesis even though the headline method is RL. RLVR optimizes the policy by having it *sample* attempts and rewarding the correct ones (Part V). That only works if the policy already samples correct attempts with nonzero probability often enough for the reward signal to be more than noise; if the policy solves the task 0% of the time, every sample gets reward zero and the gradient is flat. SFT on a modest set of correct reasoning traces (often distilled from a stronger teacher, chapter 6.7) lifts that base solve rate off the floor and, just as importantly, teaches the *format* the verifier expects (put the final answer in `\boxed{}`, use the `<think>` tags, stop cleanly). A model that reasons in the right format at a 15% solve rate is a workable cold start; a base model that emits the answer in an unparseable form at 2% is not. SFT is the on-ramp that makes the RL loop's reward signal informative, and getting the cold-start SFT right is the highest-leverage thing I do before any RL at all.

## Tooling

The tool is TRL's `SFTTrainer`, wrapping a PEFT LoRA adapter, accelerated by Unsloth. `SFTTrainer` does four things that map directly onto the theory: it applies the chat template to render examples, it builds the completion-only label mask (via a data collator, so equation (6.2.3) is what actually gets optimized), it optionally packs sequences with correct boundary handling, and it runs the training loop with the usual optimizer and scheduler. PEFT supplies the LoRA adapter so I am training a few tens of millions of parameters instead of four billion (the math is chapter 6.3). Unsloth patches the whole stack, its `FastLanguageModel.from_pretrained` returns a model whose forward and backward are replaced with fused Triton kernels and whose LoRA path is memory-optimized, which is what keeps a 4B SFT inside 16GB.

The one piece of configuration that carries the whole masking derivation is completion-only training. TRL's `DataCollatorForCompletionOnlyLM` (or the `assistant_only_loss`/`completion_only` options depending on version) takes the response template string, finds where the assistant turn starts in each rendered example, and sets every label before it to `-100`. That collator is equation (6.2.4)'s mask made real; if you skip it and train on raw rendered text, you are back to the whole-sequence loss and the parroting bug. I verify it is on by inspecting a batch's labels before training, which the lab does.

## Lab

The lab is a complete LoRA SFT of a 4B model on a small, high-quality instruction set, ending in a saved adapter on disk and a before/after generation. I deliberately keep the dataset small to make the quality-over-quantity point tangible: a few thousand examples, one short epoch, and a visibly more instruction-following model at the end. This adapter is also the literal cold-start artifact the later RLVR labs load.

This is a `uv` project in the *training* environment (kept separate from the eval/inference environment per the two-environment doctrine). From the repo root:

```bash
uv init labs/sft-4b
cd labs/sft-4b
uv add "unsloth[cu124]" trl peft datasets
# torch comes pinned by unsloth's extra; do not add it separately.
```

```python title="labs/sft-4b/train_sft.py"
"""LoRA SFT of a 4B model on a small instruct set, on one 16GB GPU.

Produces a saved LoRA adapter in artifacts/adapter/ plus a before/after
generation sample. This adapter is the cold-start for the RLVR labs.
"""
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

MODEL = "unsloth/Qwen3-4B-Base"
MAX_SEQ = 2048
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def build_model():
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,        # QLoRA: base weights in NF4 (chapter 6.3)
        dtype=None,               # let unsloth pick bf16 on Blackwell
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                     # LoRA rank
        lora_alpha=32,            # alpha = 2r, the common default
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tok


def format_and_mask(tok):
    """Split each example into prompt/completion message lists. SFTTrainer
    renders them through the chat template and, on prompt-completion data,
    masks the prompt tokens automatically (equation 6.2.4), so only the
    completion is scored, without needing a pre-rendered `text` column."""
    def fmt(ex):
        user = ex["instruction"] + ("\n\n" + ex["input"] if ex.get("input") else "")
        return {
            "prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": ex["output"]}],
        }
    return fmt


def main() -> None:
    model, tok = build_model()

    # A small, curated instruction set. Cap it to make the point that a few
    # thousand good examples is enough for a usable cold start.
    ds = load_dataset("yahma/alpaca-cleaned", split="train[:3000]")
    ds = ds.map(format_and_mask(tok), remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=str(OUT / "run"),
        max_seq_length=MAX_SEQ,
        # No packing here: completion-only masking works cleanly on
        # prompt/completion data, but packing a pre-rendered sequence would
        # score the whole thing. Keep them separate (chapter 6.2 theory).
        completion_only_loss=True,        # equation (6.2.3): mask the prompt
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,    # effective batch 16
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        report_to="none",
        seed=3407,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)

    # Sanity-check the mask BEFORE training: prompt labels must be -100.
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    scored = int((batch["labels"] != -100).sum())
    total = int(batch["labels"].numel())
    print(f"[mask check] scored tokens {scored} / {total} "
          f"({100*scored/total:.1f}%), the rest are masked prompt tokens.")

    trainer.train()

    model.save_pretrained(str(OUT / "adapter"))
    tok.save_pretrained(str(OUT / "adapter"))

    # Before/after is really "with adapter" here; show one generation.
    FastLanguageModel.for_inference(model)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain what a p-value is in two sentences."}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=120, do_sample=False)
    sample = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    (OUT / "sample.txt").write_text(sample)

    print(f"Adapter saved to {(OUT / 'adapter').resolve()}")
    print(f"Sample generation in {(OUT / 'sample.txt').resolve()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python train_sft.py
```

```admonish gotcha
Watch the `[mask check]` line before you let training run to completion. If it reports something near 100% of tokens scored, your completion-only masking is *not* active (the response-template match failed, common when the chat template's assistant marker does not match what the collator is looking for), and you are about to train the whole-sequence loss of equation (6.1.1) and get a prompt-parroting model. The fix is to make sure the collator's response template matches the exact assistant-turn opener your chat template emits. A healthy 4B SFT on Alpaca-style data usually scores somewhere in the 40 to 70 percent range depending on prompt/response length balance; the exact figure does not matter, but "essentially everything is scored" is the alarm.
```

**What you should see.** Training a LoRA over a 4B QLoRA base for one epoch on 3000 examples runs comfortably in 16GB (the QLoRA budget is chapter 6.3's `vram-budget`), and the masked loss from equation (6.2.3) should fall from somewhere around 1.5 to 2.0 nats/token to under 1.0 over the epoch, roughly monotonically, with the usual noise. The artifacts are a LoRA adapter directory (tens of MB, not GB, because you only saved the low-rank deltas) and a `sample.txt` whose answer to "Explain what a p-value is in two sentences" is now a clean, two-sentence, on-topic answer that stops, rather than the base model's meander. Record wall-clock, tokens/sec, and peak VRAM for the run (measured on the baseline machine, record value, date, driver); on the RTX 5080 a 3000-example epoch of 4B QLoRA is a matter of minutes to low tens of minutes, and peak VRAM should sit a few GiB under the 16 GiB ceiling. This adapter is the cold start every later training lab in the book loads.

```admonish read-along
Pair this with **[RLHF]** ch. 4 for instruction tuning as the first post-training stage and the data-quality argument, and **[BLLM]** ch. 7, which builds instruction fine-tuning from scratch, including the label masking of equation (6.2.4) written out by hand. Raschka's from-scratch masking loop is the concrete, no-framework version of what TRL's completion-only collator does for you; read the two together and the `-100` sentinel stops being magic. **[AIE]** ch. 7 rounds this out with the practical when-to-finetune decision framework and the memory arithmetic from the production side.
```

```admonish substack-seed
"The most expensive bug in fine-tuning costs you nothing to write and everything to debug: you forgot to mask the prompt." A post built entirely around equation (6.2.4), the one-line difference between training a model to *answer* your questions and training it to *ask* them. Show the mask check, show the two loss curves that look identical, show the two models that behave nothing alike, and land on the deeper point that SFT is imitation, so its ceiling is your data, which is why three thousand good examples beat three million scraped ones and why the whole reasoning-model game starts with getting the cold-start traces right.
```
