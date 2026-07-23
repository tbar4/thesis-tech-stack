# Summary

[Preface](frontmatter/preface.md)
[How to read this book](frontmatter/how-to-read.md)
[The hardware baseline](frontmatter/hardware-baseline.md)

# Part 0 — The Machine

- [Why this machine, this stack](p0-machine/01-why-this-machine.md)
- [Firmware first: BIOS and the Windows send-off](p0-machine/02-firmware-first.md)
- [Ubuntu 24.04 on Blackwell](p0-machine/03-ubuntu-on-blackwell.md)
- [uv and the two-environment doctrine](p0-machine/04-uv-two-envs.md)
- [Storage tiers and cache discipline](p0-machine/05-storage-tiers.md)
- [Containers and the tracking spine](p0-machine/06-containers-mlflow.md)
- [Acceptance: proving the machine](p0-machine/07-acceptance.md)

# Part I — How LLMs Actually Work

- [Tensors, autograd, and number formats](p1-llms/01-tensors-autograd-dtypes.md)
- [Tokenization and embeddings](p1-llms/02-tokenization-embeddings.md)
- [Attention from first principles](p1-llms/03-attention.md)
- [The transformer block](p1-llms/04-transformer-block.md)
- [The language-modeling objective](p1-llms/05-lm-objective.md)
- [Where memory goes: training vs inference](p1-llms/06-memory.md)
- [Sampling and decoding](p1-llms/07-sampling.md)
- [Anatomy of the open-model zoo](p1-llms/08-model-zoo.md)

# Part II — Inference Engineering

- [Prefill, decode, and the roofline](p2-inference/01-roofline.md)
- [KV cache arithmetic](p2-inference/02-kv-cache.md)
- [Quantization: theory and formats](p2-inference/03-quantization.md)
- [vLLM internals](p2-inference/04-vllm-internals.md)
- [vLLM operations](p2-inference/05-vllm-ops.md)
- [Benchmarking without lying to yourself](p2-inference/06-benchmarking.md)

# Part III — Evaluation Engineering

- [What an eval is](p3-evals/01-what-an-eval-is.md)
- [Metrics and their math](p3-evals/02-metrics.md)
- [Inspect I: tasks, datasets, solvers](p3-evals/03-inspect-tasks.md)
- [Inspect II: scorers, verifiers, sandboxes](p3-evals/04-inspect-scorers.md)
- [lm-evaluation-harness and comparability](p3-evals/05-lm-eval.md)
- [Judge models](p3-evals/06-judges.md)
- [The statistics of evals](p3-evals/07-eval-statistics.md)
- [Contamination and dataset hygiene](p3-evals/08-contamination.md)
- [The SDA data pipeline](p3-evals/09-sda-data-pipeline.md)
- [From orbital data to verifiable tasks](p3-evals/10-verifiable-tasks.md)
- [Building the thesis task suite](p3-evals/11-thesis-suite.md)
- [Eval ops](p3-evals/12-eval-ops.md)

# Part IV — Causal Inference for Evaluation

- [The ladder of causation](p4-causal/01-ladder.md)
- [DAGs and d-separation](p4-causal/02-dags.md)
- [Confounding, colliders, and selection in eval pipelines](p4-causal/03-confounding-evals.md)
- [Identification: backdoor and front-door](p4-causal/04-identification.md)
- [Interventions on models: is the reasoning delta causal?](p4-causal/05-interventions.md)
- [Causal audit of the thesis eval design](p4-causal/06-causal-audit.md)

# Part V — Reinforcement Learning Foundations

- [The RL problem](p5-rl/01-rl-problem.md)
- [Value functions and Bellman equations](p5-rl/02-values-bellman.md)
- [Bandits, exploration, and sampling](p5-rl/03-bandits-exploration.md)
- [The policy gradient theorem](p5-rl/04-policy-gradient.md)
- [REINFORCE and variance reduction](p5-rl/05-reinforce.md)
- [Actor-critic and GAE](p5-rl/06-actor-critic-gae.md)
- [Trust regions: from TRPO to PPO](p5-rl/07-trpo-ppo.md)
- [GRPO](p5-rl/08-grpo.md)
- [RLVR: reinforcement with verifiable rewards](p5-rl/09-rlvr.md)

# Part VI — Post-Training, the LLM Way

- [The post-training landscape](p6-posttraining/01-landscape.md)
- [SFT and instruction tuning](p6-posttraining/02-sft.md)
- [LoRA and QLoRA, mathematically](p6-posttraining/03-lora-qlora.md)
- [Reward models and preference data](p6-posttraining/04-reward-models.md)
- [Direct alignment: DPO and family](p6-posttraining/05-dpo.md)
- [Reasoning models and inference-time scaling](p6-posttraining/06-reasoning-its.md)
- [Distillation](p6-posttraining/07-distillation.md)

# Part VII — The Loop

- [Unsloth internals](p7-loop/01-unsloth-internals.md)
- [GRPO on 16GB](p7-loop/02-grpo-16gb.md)
- [Scorers as rewards](p7-loop/03-scorers-as-rewards.md)
- [Reward hacking and Goodhart](p7-loop/04-reward-hacking.md)
- [Data and curriculum for RLVR](p7-loop/05-data-curriculum.md)
- [Measuring the reasoning delta](p7-loop/06-reasoning-delta.md)
- [Ablations, seeds, and experimental design](p7-loop/07-ablations.md)
- [Capstone: the loop end-to-end](p7-loop/08-capstone.md)

# Part VIII — Grounding: Retrieval and Tools

- [RAG over space text](p8-grounding/01-rag.md)
- [MCP tools for live SDA data](p8-grounding/02-mcp-tools.md)
- [The augmentation arms: what caused the gain](p8-grounding/03-augmentation-arms.md)

# Part IX — Burst and Scale

- [Containerizing the stack](p9-burst/01-containers.md)
- [The Lambda workflow](p9-burst/02-lambda-workflow.md)
- [When to burst](p9-burst/03-when-to-burst.md)

# Part X — Assembly

- [From logs to figures](p10-assembly/01-logs-to-figures.md)
- [Writing the methodology chapter](p10-assembly/02-methodology.md)
- [The reproducibility package](p10-assembly/03-repro-package.md)
- [The Substack map](p10-assembly/04-substack-map.md)

# Part XI — Beyond the Thesis: Toward a Product

- [Agentic RL: training the model to use its tools](p11-beyond-thesis/01-agentic-rl.md)

---

[Appendix A: VRAM arithmetic tables](appendices/a-vram-tables.md)
[Appendix B: CLI cheatsheets](appendices/b-cli-cheatsheets.md)
[Appendix C: Troubleshooting bestiary](appendices/c-troubleshooting.md)
[Appendix D: Glossary](appendices/d-glossary.md)
[Appendix E: Reading map](appendices/e-reading-map.md)
[Appendix F: Notation reference](appendices/f-notation.md)
