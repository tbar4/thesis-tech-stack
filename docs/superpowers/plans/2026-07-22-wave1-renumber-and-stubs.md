# Wave 1 — Renumber + Stubs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the book to make room for the space-data additions — insert two new Part III chapters and a new Part VIII, renumber the affected chapters/parts, and add empty-but-valid stubs — so the book still builds green under `create-missing = false` before any content lands.

**Architecture:** Pure structural change. `git mv` the renamed files/dirs, add five stub files, rewrite `SUMMARY.md`, then sweep every prose cross-reference to the renumbered chapters/parts using *context-anchored* replacements (never bare numbers, to avoid corrupting values like `28 GiB` or `0.9`). Verification is: `mdbook build` passes, linkcheck is clean, and a grep audit shows zero stale references.

**Tech Stack:** mdbook, git, grep/sed, the design spec at `docs/superpowers/specs/2026-07-22-space-data-grounding-design.md`.

**Reference — the renumber map:**
| Old | New |
|---|---|
| (new) | 3.9 The SDA data pipeline |
| (new) | 3.10 From orbital data to verifiable tasks |
| 3.9 Building the thesis task suite | **3.11** |
| 3.10 Eval ops | **3.12** |
| (new Part VIII) | Grounding: Retrieval and Tools (8.1 RAG, 8.2 MCP tools, 8.3 augmentation arms) |
| Part VIII — Burst and Scale (8.1–8.3) | **Part IX** (9.1–9.3) |
| Part IX — Assembly (9.1–9.4) | **Part X** (10.1–10.4) |

**File structure after this wave:**
- `book/src/p3-evals/09-sda-data-pipeline.md` (new stub)
- `book/src/p3-evals/10-verifiable-tasks.md` (new stub)
- `book/src/p3-evals/11-thesis-suite.md` (moved from `09-thesis-suite.md`)
- `book/src/p3-evals/12-eval-ops.md` (moved from `10-eval-ops.md`)
- `book/src/p8-grounding/01-rag.md`, `02-mcp-tools.md`, `03-augmentation-arms.md` (new stubs)
- `book/src/p9-burst/` (moved from `p8-burst/`)
- `book/src/p10-assembly/` (moved from `p9-assembly/`)
- `book/src/SUMMARY.md` (rewritten headers + entries)
- ~18 chapter/appendix files with cross-reference edits

---

### Task 1: Baseline audit + branch

**Files:** none (recon only)

- [ ] **Step 1: Confirm a clean branch off latest main**

```bash
cd /home/user/thesis-tech-stack
git fetch origin main --quiet
git checkout -B claude/mdbook-evals-as-rewards-spec-fvhxw7 origin/main
git status --short   # expect empty
```

- [ ] **Step 2: Capture the baseline reference counts (for the final audit)**

```bash
cd book/src
grep -rEn 'Part VIII|Part IX' --include=*.md | tee /tmp/audit-parts-before.txt | wc -l
grep -rEn 'chapter 3\.9|chapter 3\.10|Chapter 3\.9|Chapter 3\.10|Ch 3\.10|3\.9 |3\.9\)|3\.10 |3\.10\)' --include=*.md | tee /tmp/audit-ch3-before.txt | wc -l
```

Expected: non-zero counts. These files are the worklist for Tasks 5–6; keep them.

---

### Task 2: Move the two Part III chapters that shift, then add the two new stubs

**Files:**
- Move: `book/src/p3-evals/09-thesis-suite.md` → `11-thesis-suite.md`
- Move: `book/src/p3-evals/10-eval-ops.md` → `12-eval-ops.md`
- Create: `book/src/p3-evals/09-sda-data-pipeline.md`
- Create: `book/src/p3-evals/10-verifiable-tasks.md`

- [ ] **Step 1: Move the shifting files first (frees the 09/10 slots)**

```bash
cd /home/user/thesis-tech-stack/book/src
git mv p3-evals/09-thesis-suite.md p3-evals/11-thesis-suite.md
git mv p3-evals/10-eval-ops.md    p3-evals/12-eval-ops.md
```

- [ ] **Step 2: Create the 3.9 stub**

Write `book/src/p3-evals/09-sda-data-pipeline.md`:

```markdown
# The SDA data pipeline

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 3.9).
This stub exists so the book builds under `create-missing = false`.
```

**Goal.** Turn five live space-data sources (celestrak, space-track, api.nasa.gov,
thespacedevs / Launch Library 2, Spaceflight News API) into a normalized,
provenance-stamped, reproducible store, orchestrated with Airflow 3 (thin DAGs
over plain `uv run` task modules, containerized).

**Covers.** The sources by data type; why an orchestrator rather than cron;
Airflow 3 assets / data-aware scheduling; normalization (Pydantic) and quality
gates (pandera); immutable snapshots stored as Parquet and queried with DuckDB;
DVC-pinned versioning and per-record provenance; the open-data / licensing / auth
boundary as a first-class constraint (space-track is fetch-only).

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
```

Note: the outer fence in the file uses THREE backticks; the nested ```` ```admonish ```` blocks above are shown with the standard three backticks. When you actually create the file, the admonish blocks are ordinary three-backtick fences inside the markdown body (they are content, not nested code fences), so no four-backtick escaping is needed here.

- [ ] **Step 3: Create the 3.10 stub**

Write `book/src/p3-evals/10-verifiable-tasks.md`:

```markdown
# From orbital data to verifiable tasks

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 3.10).
This stub exists so the book builds under `create-missing = false`.
```

**Goal.** Turn pipeline snapshots into verifiable SDA task instances (prompt +
machine-checkable ground truth), with a Skyfield/SGP4 physics oracle that both
generates the gold answer and grades the model's answer. Conjunction screening is
the flagship task.

**Covers.** Task families (conjunction screening, orbital-element derivation,
decay/reentry ordering, pass prediction, catalog correlation); the oracle as
verifier plugged into the `thesis_suite` API; numeric tolerance policy; difficulty
calibration (feeds 7.5); item counts from 3.7; dedup against 3.8; output freezes in
3.11.

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
```

- [ ] **Step 4: Verify the four files exist with the right names**

```bash
ls book/src/p3-evals/ | grep -E '^(09|10|11|12)-'
```
Expected: `09-sda-data-pipeline.md 10-verifiable-tasks.md 11-thesis-suite.md 12-eval-ops.md`

- [ ] **Step 5: Commit**

```bash
git add book/src/p3-evals/
git commit -m "Part III: add SDA pipeline + verifiable-tasks stubs; shift suite/eval-ops to 3.11/3.12"
```

---

### Task 3: Rename the Part VIII/IX directories and add the Part VIII stubs

**Files:**
- Move dir: `book/src/p8-burst/` → `book/src/p9-burst/`
- Move dir: `book/src/p9-assembly/` → `book/src/p10-assembly/`
- Create: `book/src/p8-grounding/01-rag.md`, `02-mcp-tools.md`, `03-augmentation-arms.md`

- [ ] **Step 1: Rename the shifting directories (high number first)**

```bash
cd /home/user/thesis-tech-stack/book/src
git mv p9-assembly p10-assembly
git mv p8-burst p9-burst
mkdir p8-grounding
```

- [ ] **Step 2: Create `book/src/p8-grounding/01-rag.md`**

```markdown
# RAG over space text

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 8.1).
```

**Goal.** Retrieve the pipeline's article corpus (spacenews / NASA) to ground
qualitative / recency SDA answers, and measure whether it actually helps.

**Covers.** Why RAG is for text not numbers; chunking; a local embedding model
served via vLLM (co-resident memory budget on 16GB); a LanceDB index; a thin
hand-rolled retrieval loop; retrieval metrics (recall@k, MRR) vs end-task metrics;
RAG-as-confound (retrieval leakage, ties to 3.8 and Part IV).

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
```

- [ ] **Step 3: Create `book/src/p8-grounding/02-mcp-tools.md`**

```markdown
# MCP tools for live SDA data

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 8.2).
```

**Goal.** Give the model live numerical SDA data (conjunction / catalog / orbital
queries) via MCP tools at inference, with verifiable tool-use.

**Covers.** A FastMCP server wrapping the 3.9 clients + Skyfield oracle; transports
(stdio / streamable-HTTP); the model as MCP client via vLLM tool-calling; wiring
tools into Inspect for the agentic eval (3.4); verifiable tool-use; caching and the
open-data boundary (tools fetch live, never redistribute).

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
```

- [ ] **Step 4: Create `book/src/p8-grounding/03-augmentation-arms.md`**

```markdown
# The augmentation arms: what caused the gain

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 8.3).
```

**Goal.** Causally separate how much SDA improvement comes from parametric
RL-training vs retrieval vs tools.

**Covers.** The four arms (base / RL-trained / +RAG / +tools, and combinations) at
matched compute/token budgets (6.6); each augmentation as an intervention, with
RAG-access and tool-access drawn as a DAG and identified with the Part IV
backdoor/front-door machinery; paired stats + effect sizes (3.7 / 7.6); what each
arm's delta legitimately claims.

## Theory

## Tooling

## Lab

```admonish thesis-thread
The strongest form of the thesis claim advances here. To be drafted.
```

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
```

- [ ] **Step 5: Verify structure**

```bash
ls book/src/ | grep -E 'p8-grounding|p9-burst|p10-assembly'
ls book/src/p8-grounding/   # expect 01-rag.md 02-mcp-tools.md 03-augmentation-arms.md
```

- [ ] **Step 6: Commit**

```bash
git add book/src/
git commit -m "Add Part VIII (Grounding) stubs; rename Burst->p9, Assembly->p10"
```

---

### Task 4: Rewrite SUMMARY.md

**Files:** Modify `book/src/SUMMARY.md`

- [ ] **Step 1: Replace the Part III block** (add two entries, rename the last two paths)

Find the Part III list and make it read exactly:

```markdown
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
```

- [ ] **Step 2: Insert the new Part VIII and renumber the two parts after it**

Replace the current `# Part VIII — Burst and Scale` and `# Part IX — Assembly` blocks with:

```markdown
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
```

(Keep the exact chapter filenames already in `p9-burst/` and `p10-assembly/`; only the directory prefix changed.)

- [ ] **Step 3: Verify every SUMMARY path resolves**

```bash
cd /home/user/thesis-tech-stack/book
python3 - <<'PY'
import re, os
src="src"; s=open(os.path.join(src,"SUMMARY.md")).read()
missing=[l for l in re.findall(r"\]\(([^)]+\.md)\)", s) if not os.path.exists(os.path.join(src,l))]
print("missing:", missing or "none")
PY
```
Expected: `missing: none`

- [ ] **Step 4: Commit**

```bash
git add book/src/SUMMARY.md
git commit -m "SUMMARY: new Part III chapters, new Part VIII, renumber Burst/Assembly"
```

---

### Task 5: Sweep the Part-VIII/IX prose references (do the high number first)

**Files:** every file in `/tmp/audit-parts-before.txt` plus any `\bPart (VIII|IX)\b` / `chapter 9\.[0-9]` / `\(Part VIII\)` hit. Known: `appendices/e-reading-map.md`, `appendices/d-glossary.md`, `p10-assembly/02-methodology.md`, `p10-assembly/03-repro-package.md`, `p10-assembly/04-substack-map.md`, `p6-posttraining/07-distillation.md`, `p9-burst/*`.

**Rule: anchor every replacement so bare numbers are never touched.** Only these forms are chapter/part references: `Part VIII`, `Part IX`, `Part X`, `chapter 9.N`, `Chapter 9.N`, `(9.N)`, `9.N (`, `(Part VIII)`. Do the IX→X shift before the VIII→IX shift so the newly-written "Part IX" is not re-shifted.

- [ ] **Step 1: Assembly part IX → X (part header refs, not the new Part VIII)**

```bash
cd /home/user/thesis-tech-stack/book/src
# Prose "Part IX" currently ALWAYS means Assembly (the only Part IX today). Safe.
grep -rln 'Part IX' --include=*.md | while read f; do
  sed -i 's/Part IX/Part X/g' "$f"
done
```

- [ ] **Step 2: Assembly chapter numbers 9.1–9.4 → 10.1–10.4 (anchored)**

Assembly chapters are referenced as `chapter 9.1/9.2/9.3/9.4`, `(9.3)`, `9.1`, `9.2`. Apply only the anchored forms:

```bash
cd /home/user/thesis-tech-stack/book/src
for n in 1 2 3 4; do
  grep -rln "chapter 9\.$n\|Chapter 9\.$n\|(9\.$n)\|9\.$n (" --include=*.md | while read f; do
    sed -i "s/chapter 9\.$n/chapter 10.$n/g; s/Chapter 9\.$n/Chapter 10.$n/g; s/(9\.$n)/(10.$n)/g; s/9\.$n (/10.$n (/g" "$f"
  done
done
```

- [ ] **Step 3: Manually resolve any remaining bare `9.N` assembly refs**

```bash
grep -rEn '\b9\.[1-4]\b' --include=*.md | grep -viE 'gib|gb|0\.9|1\.9|2\.9|figure|table|v9|9\.[0-9]{2}'
```
Read each hit; if it names an assembly chapter (e.g. "9.1's figure pipeline"), edit it to `10.N` by hand. If it is a numeric value, leave it.

- [ ] **Step 4: Burst part/chapter VIII → IX (anchored)**

```bash
cd /home/user/thesis-tech-stack/book/src
grep -rln 'Part VIII' --include=*.md | while read f; do sed -i 's/Part VIII/Part IX/g' "$f"; done
for n in 1 2 3; do
  grep -rln "chapter 8\.$n\|Chapter 8\.$n\|(8\.$n)\|8\.$n (" --include=*.md | while read f; do
    sed -i "s/chapter 8\.$n/chapter 9.$n/g; s/Chapter 8\.$n/Chapter 9.$n/g; s/(8\.$n)/(9.$n)/g; s/8\.$n (/9.$n (/g" "$f"
  done
done
```

**Caution:** the NEW Part VIII (grounding) SUMMARY block already says "Part VIII" — but SUMMARY.md was committed in Task 4 and these seds run over chapter files, not SUMMARY. Confirm `SUMMARY.md` was NOT rewritten by this step:

```bash
grep -n 'Part VIII — Grounding\|Part IX — Burst\|Part X — Assembly' SUMMARY.md
```
Expected: all three present and correct.

- [ ] **Step 5: Manually resolve remaining bare `8.N` burst refs**

```bash
grep -rEn '\b8\.[1-3]\b' --include=*.md | grep -viE 'gib|gb|0\.8|1\.8|2\.8|figure|table'
```
Hand-edit true burst-chapter references to `9.N`; leave numeric values.

- [ ] **Step 6: Commit**

```bash
git add book/src/
git commit -m "Renumber prose refs: Part IX->X, Part VIII->IX, and their chapter numbers"
```

---

### Task 6: Sweep the Part III references (suite 3.9 → 3.11, eval-ops 3.10 → 3.12)

**Files:** every hit in `/tmp/audit-ch3-before.txt`. Known: `appendices/d-glossary.md`, `appendices/e-reading-map.md`, `p10-assembly/02-methodology.md`, `p10-assembly/03-repro-package.md`, `p3-evals/06-judges.md`, `p3-evals/07-eval-statistics.md`, `p3-evals/08-contamination.md`, plus Part VII chapters that cite the suite.

**Key fact:** every existing reference to `3.9`/`3.10` today means the suite / eval-ops (the new 3.9/3.10 chapters have no inbound references yet). So all existing chapter-anchored `3.9`→`3.11` and `3.10`→`3.12`. Do `3.10`→`3.12` first (so `3.10` is not caught by a `3.1` pattern), then `3.9`→`3.11`.

- [ ] **Step 1: eval-ops 3.10 → 3.12 (anchored)**

```bash
cd /home/user/thesis-tech-stack/book/src
grep -rln '3\.10' --include=*.md | while read f; do
  sed -i 's/chapter 3\.10/chapter 3.12/g; s/Chapter 3\.10/Chapter 3.12/g; s/Ch 3\.10/Ch 3.12/g; s/(3\.10)/(3.12)/g; s/3\.10 (eval ops)/3.12 (eval ops)/g' "$f"
done
```

- [ ] **Step 2: suite 3.9 → 3.11 (anchored)**

```bash
cd /home/user/thesis-tech-stack/book/src
grep -rln '3\.9' --include=*.md | while read f; do
  sed -i 's/chapter 3\.9/chapter 3.11/g; s/Chapter 3\.9/Chapter 3.11/g; s/(3\.9)/(3.11)/g; s/3\.9 (frozen suite/3.11 (frozen suite/g; s/3\.9 thesis/3.11 thesis/g; s/3\.9 (/3.11 (/g; s/3\.9)/3.11)/g' "$f"
done
```

- [ ] **Step 3: Resolve remaining bare `3.9` / `3.10` chapter refs by hand**

```bash
grep -rEn '\b3\.(9|10)\b' --include=*.md | grep -viE 'gib|gb/s|gflop|3\.9\.|v3\.9'
```
For each hit: if it refers to the suite, `→ 3.11`; eval-ops `→ 3.12`; if it is a version/number/measured value, leave it. Note the two NEW chapters may legitimately introduce fresh `3.9`/`3.10` self-references — those are correct and stay.

- [ ] **Step 4: Fix the reading-map (appendix E) rows explicitly**

In `appendices/e-reading-map.md`: the `3.10 Eval ops` row → `3.12 Eval ops`; the "with 3.10" pointer (AIE ch.10 line) → "with 3.12"; the `3.2` etc. rows are unchanged. Do NOT add rows for the new 3.9/3.10/8.x chapters yet (that is content for the drafting waves).

- [ ] **Step 5: Fix the glossary suite pointer**

In `appendices/d-glossary.md`: `the thesis task suite* (chapter 3.9)` → `(chapter 3.11)`.

- [ ] **Step 6: Commit**

```bash
git add book/src/
git commit -m "Renumber prose refs: suite 3.9->3.11, eval-ops 3.10->3.12"
```

---

### Task 7: Verify — build, linkcheck, and stale-reference audit

**Files:** none (verification)

- [ ] **Step 1: Structural build (preprocessor-free, catches SUMMARY/link errors fast)**

```bash
cd /tmp && rm -rf tb && mkdir tb && cat > tb/book.toml <<'TOML'
[book]
title="t"
src="/home/user/thesis-tech-stack/book/src"
[build]
create-missing=false
build-dir="out"
[output.html]
TOML
/root/.cargo/bin/mdbook build tb 2>&1 | tail -5
```
Expected: "Book building has started" / "Running the html backend" with **no error** and no "not found" for any SUMMARY entry.

- [ ] **Step 2: Stale-reference audit (must be zero)**

```bash
cd /home/user/thesis-tech-stack/book/src
echo "stale Part headers:"; grep -rEn 'Part VIII — Burst|Part IX — Assembly' --include=*.md || echo OK
echo "stale suite refs:"; grep -rEn 'chapter 3\.9\b|\(3\.9\)|3\.9 \(frozen' --include=*.md || echo OK
echo "orphan check (files on disk vs SUMMARY):"
python3 - <<'PY'
import re,os
src="."; s=open("SUMMARY.md").read(); linked=set(re.findall(r"\]\(([^)]+\.md)\)", s))|{"SUMMARY.md"}
disk=[os.path.relpath(os.path.join(r,f),src) for r,_,fs in os.walk(src) for f in fs if f.endswith(".md")]
print("orphans:", [f for f in disk if f not in linked] or "none")
print("missing:", [l for l in linked if not os.path.exists(l)] or "none")
PY
```
Expected: `OK`/`OK`, `orphans: none`, `missing: none`.

- [ ] **Step 3: Push and open the PR (the PR-CI runs the authoritative build + linkcheck)**

```bash
cd /home/user/thesis-tech-stack
for i in 1 2 3 4; do git push -u origin claude/mdbook-evals-as-rewards-spec-fvhxw7 2>&1 | tail -2 && break || sleep $((2**i)); done
```
Then open a draft PR titled "Wave 1: renumber Part III + add space-data stubs" against `main`, body summarizing the renumber map. Watch the `build` check; if linkcheck fails on a missed reference, grep for it and fix on the branch (drive-to-green).

---

## Self-review

- **Spec coverage:** this plan implements spec §1 (structure/renumber) and the stub-first requirement of §4. It deliberately does NOT touch content (§3 chapters), the tech stack (§2), cross-reference *additions* for new chapters, or appendix new-chapter rows — those belong to later waves and are called out as such.
- **Placeholder scan:** the chapter bodies are intentional stubs (Goal/Covers from the spec + a status admonish), which is the correct artifact for `create-missing = false`; they are not plan placeholders.
- **Ordering hazards handled:** IX→X before VIII→IX; 3.10→3.12 before 3.9→3.11; move-then-create for the 09/10 slots; SUMMARY committed before the prose sed sweep so the seds do not corrupt the new Part VIII header.
- **Disambiguation:** every sed is anchored (`chapter`, parentheses, trailing space-paren) so `28 GiB`, `0.9`, `~8`, version strings are never rewritten; each sweep ends with a manual bare-number grep to catch anchored-miss references.
