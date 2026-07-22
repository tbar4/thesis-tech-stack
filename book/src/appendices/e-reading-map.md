# Appendix E: Reading map

**Goal.** Pair the eight reference books to the chapters they serve, and suggest
an order to read them in relative to the book's five authoring waves.

I lean on eight reference books. None is quoted at length anywhere in the text;
instead each chapter carries a `read-along` pointer by short key, and this
appendix is the master index behind those pointers. Local copies live in
`references/` at the repo root (see that directory's `README.md`); they are not
distributed with the book.

## The eight books

| Key | Book | Primary parts served |
|---|---|---|
| **[S&B]** | Sutton & Barto, *Reinforcement Learning: An Introduction* (2e) | Part V |
| **[RLHF]** | Lambert, *RLHF: LLM Alignment and Post-Training* | Parts IV–VI (esp. ch. 5–8) |
| **[BRM]** | Raschka, *Build a Reasoning Model (From Scratch)* | Parts III, V, VI (ch. 3–8; App. C Qwen3 source) |
| **[BLLM]** | Raschka, *Build a Large Language Model (From Scratch)* | Part I |
| **[MADL]** | Chaudhuri, *Math and Architectures of Deep Learning* | Part I math spine (ch. 2–9) |
| **[GAIA]** | Hurbans, *Grokking AI Algorithms* (2e) | Pre-reading only; RL intuition (ch. 10) |
| **[CAI]** | Ness, *Causal AI* | Part IV |
| **[AIE]** | Huyen, *AI Engineering* | Breadth complement: Parts II–III, VI, and 7.5 (ch. 3–4, 7–9) |

Three of these earn special notes. **[GAIA]** is deliberately light-duty: it
overlaps the other texts at lower depth, so it serves as optional pre-reading
for RL intuition, never as a chapter dependency. **[CAI]** earns a full part of
its own (Part IV) because judge-model and benchmark-comparison pitfalls are
confounding problems, and framing evaluation claims causally is a committee-grade
differentiator. **[AIE]** is the production-engineering complement: it covers the
same territory as Parts II, III, and VI at survey depth but from the builder's
seat (evaluation methodology, AI-as-judge, finetuning, dataset engineering,
inference optimization), so I cite it where a chapter benefits from the
systems-level framing rather than for derivations.

## Suggested order

The books are not read cover-to-cover front-to-back; they are read alongside the
parts they serve, in roughly the authoring-wave order (spec §8).

1. **Before anything (optional):** skim **[GAIA]** ch. 10 for gentle RL
   intuition. Nothing downstream depends on it.
2. **With Part I (the theory spine):** read **[MADL]** ch. 2–9 as the math
   backbone and **[BLLM]** ch. 2–5 as the from-scratch build companion. These
   two run in parallel with Part I chapter for chapter.
3. **With Parts II–III (inference and evals):** **[BRM]** ch. 1–3 and its
   App. C (Qwen3 source) back the model-anatomy and eval-framing chapters.
4. **With Part IV (causal):** **[CAI]** Parts 1–3, in order, are the spine.
5. **With Part V (RL foundations):** **[S&B]** is the backbone (ch. 2–4 and
   ch. 13 carry the most weight), with **[BRM]** ch. 6–7 and **[RLHF]** ch. 6
   arriving as the material turns to LLM-specific policy optimization.
6. **With Part VI (post-training):** **[RLHF]** ch. 1–8 is the throughline;
   **[BRM]** ch. 4–5 and ch. 8 cover reasoning-time scaling and distillation.
7. **Anytime, in parallel (optional):** **[AIE]** reads well cover to cover as
   the production-context companion; its evaluation chapters (ch. 3–4) pair
   with Part III, finetuning (ch. 7) with Part VI, dataset engineering (ch. 8)
   with 3.8/7.5, and inference optimization (ch. 9) with Part II.

## Per-chapter pairings

Every entry below mirrors the `read-along` admonition in the named chapter.
Chapters with no reference pairing are hands-on or self-contained (Part 0, most
of Part II, and Part IX lean on tooling and the book's own artifacts rather than
outside reading; 2.6 picks up an [AIE] pairing). Part VII does carry read-along pointers, because the loop
chapters lean on [RLHF], [S&B], and [MADL] for the algorithm and kernel math
even as the labs are hands-on; its per-chapter rows are below.

### Part I — How LLMs Actually Work

| Chapter | Read-along |
|---|---|
| 1.1 Tensors, autograd, and number formats | [MADL] ch. 2–4 |
| 1.2 Tokenization and embeddings | [BLLM] ch. 2 |
| 1.3 Attention from first principles | [BLLM] ch. 3; [MADL] ch. 7–8 |
| 1.4 The transformer block | [BLLM] ch. 4; [BRM] App. C |
| 1.5 The language-modeling objective | [BLLM] ch. 5 |
| 1.6 Where memory goes: training vs inference | [MADL] ch. 8–9 |
| 1.7 Sampling and decoding | [BRM] ch. 2 |
| 1.8 Anatomy of the open-model zoo | [BRM] ch. 1, App. C |

### Part II — Inference Engineering

| Chapter | Read-along |
|---|---|
| 2.6 Benchmarking without lying to yourself | [AIE] ch. 9 |

### Part III — Evaluation Engineering

| Chapter | Read-along |
|---|---|
| 3.1 What an eval is | [BRM] ch. 3; [RLHF] Part 3; [AIE] ch. 3–4 |
| 3.2 Metrics and their math | [BRM] ch. 3; [RLHF] Part 3 |
| 3.6 Judge models | [RLHF]; [AIE] ch. 3 (AI-as-judge) |
| 3.7 The statistics of evals | [CAI] |
| 3.8 Contamination and dataset hygiene | [BRM]; [AIE] ch. 8 |

### Part IV — Causal Inference for Evaluation

Part-wide spine: **[CAI] Parts 1–3.**

| Chapter | Read-along |
|---|---|
| 4.1 The ladder of causation | [CAI] ch. 1–2 |
| 4.2 DAGs and d-separation | [CAI] Part 2 |
| 4.3 Confounding, colliders, and selection | [CAI] Parts 2–3 |
| 4.4 Identification: backdoor and front-door | [CAI] Part 3 |
| 4.5 Interventions on models | [CAI] Part 3 |

### Part V — Reinforcement Learning Foundations

| Chapter | Read-along |
|---|---|
| 5.1 The RL problem | [S&B] ch. 3 |
| 5.2 Value functions and Bellman equations | [S&B] ch. 3–4 |
| 5.3 Bandits, exploration, and sampling | [S&B] ch. 2; [GAIA] ch. 10 |
| 5.4 The policy gradient theorem | [S&B] ch. 13 |
| 5.5 REINFORCE and variance reduction | [S&B] ch. 13; [BRM] ch. 6 sidebars |
| 5.6 Actor-critic and GAE | [RLHF] ch. 6 |
| 5.7 Trust regions: from TRPO to PPO | [RLHF] ch. 6; [BRM] ch. 6 |
| 5.8 GRPO | [BRM] ch. 6–7; [RLHF] ch. 6 |
| 5.9 RLVR: reinforcement with verifiable rewards | [RLHF] ch. 7; [BRM] ch. 6 |

### Part VI — Post-Training, the LLM Way

| Chapter | Read-along |
|---|---|
| 6.1 The post-training landscape | [RLHF] ch. 1–3 |
| 6.2 SFT and instruction tuning | [RLHF] ch. 4; [BLLM] ch. 7; [AIE] ch. 7 |
| 6.3 LoRA and QLoRA, mathematically | [RLHF] ch. 4; Part II ch. 3 |
| 6.4 Reward models and preference data | [RLHF] ch. 5 |
| 6.5 Direct alignment: DPO and family | [RLHF] ch. 8 |
| 6.6 Reasoning models and inference-time scaling | [BRM] ch. 4–5; [RLHF] ch. 7 |
| 6.7 Distillation | [BRM] ch. 8 |

### Part VII — The Loop

| Chapter | Read-along |
|---|---|
| 7.1 Unsloth internals | [MADL] ch. 3–4 |
| 7.2 GRPO on 16GB | [RLHF]; [S&B] ch. 13 |
| 7.3 Scorers as rewards | [RLHF] |
| 7.4 Reward hacking and Goodhart | [RLHF] |
| 7.5 Data and curriculum for RLVR | [AIE] ch. 8 |

### Part VIII — Burst and Scale

| Chapter | Read-along |
|---|---|
| 8.3 When to burst | [RLHF] ch. 6 (infra notes) |

```admonish read-along
When a chapter's `read-along` pointer and this table disagree, the chapter is
the source of truth: update this appendix, not the chapter. This index accretes
as chapters are drafted, so pairings for chapters still in stub form may be
refined when their `read-along` blocks are written.
```
