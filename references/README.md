# Reference books

This directory is where local copies of the **fifteen** reference books live.
They are meant to stay **local-only** and out of version control: they are large,
copyrighted binaries, and this repo is public. The `.gitignore` here keeps them
untracked — do not force copyrighted PDFs into the repo (note that GitHub's web
"upload files" bypasses `.gitignore`).

Because the files are untracked, the "Local file" column is **declarative**: it
records the filename each book should have if present, not proof that it is.

## The eight core books — the theory spine

| Key | Book | Local file | Zotero |
|---|---|---|---|
| **[S&B]** | Sutton & Barto, *Reinforcement Learning: An Introduction* (2e) | `BartoSutton-compressed.pdf` | `PQ8GA7DF` |
| **[RLHF]** | Lambert, *RLHF: LLM Alignment and Post-Training* | `Reinforcement_Learning_from_Human_Feedba.pdf` | `8CL9GQKZ` |
| **[BRM]** | Raschka, *Build a Reasoning Model (From Scratch)* | `Build_a_Reasoning_Model_(From_Scratch).pdf` | `QJPWVCT6` |
| **[BLLM]** | Raschka, *Build a Large Language Model (From Scratch)* | `Build_a_Large_Language_Model_(From_Scrat.pdf` | `CS6FKEIS` |
| **[MADL]** | Chaudhuri, *Math and Architectures of Deep Learning* | `Math_and_Architectures_of_Deep_Learning.pdf` | `NXS35G7P` |
| **[GAIA]** | Hurbans, *Grokking AI Algorithms* (2e) | `Grokking_AI_Algorithms_Second_Edition.pdf` | `SSNY43NN` |
| **[CAI]** | Ness, *Causal AI* | `Causal_AI.pdf` | `3N3MEFUH` |
| **[AIE]** | Huyen, *AI Engineering* | `AI Engineering - Chip Huyen.epub` | — |

## Practical and framework references

These back the grounding chapters (Part VIII), the data pipeline (3.9), the
post-training landscape (6.1), and the agentic-RL chapter (11.1). They were
added to the reading map after this file was first written, which is why they
were missing from it until 2026-08-02.

| Key | Book | Local file | Zotero |
|---|---|---|---|
| **[RAG]** | Rothman, *RAG-Driven Generative AI* | — | — |
| **[LC]** | Auffarth & Kuligin, *Generative AI with LangChain* (2e) | — | — |
| **[GADP]** | Lakshmanan & Hapke, *Generative AI Design Patterns* | — | — |
| **[DPA]** | Harenslak et al., *Data Pipelines with Apache Airflow* (2e) | — | `WZ3ZL3XT` |
| **[AGENTS]** | Infante, *AI Agents and Applications* | — | `835NEF62` |
| **[CUST]** | Bahree & Tok, *LLM Customization and Fine-Tuning* | — | — |
| **[PDL]** | Dürr, Sick & Murina, *Probabilistic Deep Learning* | — | `3NH74HUK` |

## Gaps

Five books are cited by the text but have **no Zotero record**: [AIE], [RAG],
[LC], [GADP], [CUST]. [AIE] is the odd one — it has a local copy but no library
entry. Until they are added, nothing tracks whether they have been read.

## See also

- `../book/src/appendices/e-reading-map.md` — per-chapter pairings, suggested
  reading order, and why each book earns its place.
- `../book/reading-map.toml` — the machine-readable twin of that appendix,
  carrying the Zotero item keys above. Consumers read the TOML, not the prose.
