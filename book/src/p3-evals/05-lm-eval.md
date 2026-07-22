# lm-evaluation-harness and comparability

Inspect is for building my own instruments. `lm-evaluation-harness` (EleutherAI's `lm-eval`) is for reading off *standard* instruments in a way that is comparable to how everyone else read them. The two are complementary: when I want a novel construct or a verifiable reward, I build it in Inspect (chapters 3 and 4); when I want a GSM8K number that means the same thing my number means to a committee member who has seen a hundred GSM8K numbers, I run the harness, because the harness fixes the thousand small choices (few-shot count, prompt format, answer extraction, stop sequences) that otherwise make two "GSM8K scores" incomparable. This chapter explains why published numbers differ, how to drive the harness against my local vLLM endpoint, and when a harness *delta* is trustworthy even when the absolute number is not. The lab produces the first committee-grade artifact of the thesis: a GSM8K quantization-degradation table, the same 8B model at BF16 and INT4, measured under one identical harness config.

```admonish read-along
**[RLHF] Lambert Part 3** is blunt about how much published eval numbers move under harness settings and why cross-paper comparisons are mostly noise; this chapter is the operational response, control the harness so your deltas mean something. **[BRM] Raschka ch. 3** motivates why quantization-degradation measurement matters for a one-GPU project: INT4 is what makes the model fit and run fast, so I have to know what it costs in reasoning accuracy, and the harness is how I price it.
```

## Theory

### The harness is a big library of frozen operationalizations

The harness's value is that its tasks are *frozen*: a task like `gsm8k` is a versioned YAML that pins the dataset, the few-shot regime, the prompt template, the answer-extraction filters, and the metric. When I run `gsm8k` I get the community's agreed operationalization of the GSM8K construct, not my personal one, and that agreement is exactly what makes my number comparable to a published one (provided the config also matches). The YAML is worth understanding because every field is a construct decision from chapter 1 made explicit.

A GSM8K-style task YAML has roughly this shape (abbreviated):

```yaml
task: gsm8k
dataset_path: gsm8k
dataset_name: main
output_type: generate_until
doc_to_text: "Question: {{question}}\nAnswer:"     # prompt template
doc_to_target: "{{answer}}"
num_fewshot: 5                                       # the few-shot regime
generation_kwargs:
  until: ["Question:", "\n\n"]                       # stop sequences
  do_sample: false                                   # greedy
filter_list:                                          # answer extraction
  - name: strict-match
    filter:
      - function: regex
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
  - name: flexible-extract
    filter:
      - function: regex
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        group_select: -1
metric_list:
  - metric: exact_match
```

Two fields deserve emphasis. **`num_fewshot`** sets the regime: GSM8K is conventionally 5-shot, and a 0-shot number and an 8-shot number are different measurements of different constructs (recall-with-exemplars versus zero-shot reasoning), not comparable. **`filter_list`** defines answer extraction, and GSM8K ships two filters: `strict-match` demands the model emit the exact `#### N` format from the few-shot exemplars, while `flexible-extract` grabs the last number anywhere in the output. These can differ by several points on the same run, because they are measuring "reasoning plus format compliance" versus "reasoning" (chapter 1's answer-extraction brittleness, now a named knob). A committee-grade table reports both.

### Why published numbers differ

When two papers report different GSM8K scores for ostensibly the same model, the model is rarely the reason. The reasons are almost always harness-level, and knowing the list is how I avoid being fooled and how I make my own numbers defensible:

- **Few-shot count.** 0, 5, and 8 shot are different constructs. A paper reporting 8-shot against another's 5-shot is comparing different measurements.
- **Prompt format and chat template.** Whether the harness wraps the prompt in the model's chat template, and which exemplar formatting it uses, moves the score. A reasoning model prompted without its chat template can badly underperform its true behavior.
- **Answer-extraction filter.** `strict-match` versus `flexible-extract`, or a bespoke regex, changes what counts as a correct answer independent of the reasoning.
- **Greedy versus sampled, and stop sequences.** Greedy decoding is reproducible; sampled decoding adds variance and its own mean. Different `until` stop strings truncate reasoning differently.
- **Dataset version and split.** GSM8K has variants and the harness pins a version; a mismatch silently changes items.
- **Normalization.** Whether `1,000` equals `1000`, trailing-zero handling, currency stripping.
- **CoT or not.** Whether the template elicits chain-of-thought at all. For reasoning models this is enormous.

The practical upshot is a discipline, not a lament: I record the full harness config (task version, `num_fewshot`, filters, `gen_kwargs`, chat-template flag, harness commit) next to every number, and the harness does this for me by writing the config into its results JSON. That config block is part of the committee-grade artifact.

```admonish gotcha title="An absolute harness number is only comparable to a published number if the whole config matches"
It is tempting to run `gsm8k` once, get 61.2%, and compare it to a blog post's 74%. That comparison is almost certainly invalid: different few-shot count, chat template, filter, or decoding. The harness gives you comparability *only if you match the config that produced the number you're comparing to*, and you usually cannot fully reconstruct someone else's config. So I treat cross-source absolute comparisons as weak evidence at best. What the harness gives me reliably is *internal* comparability, two of my own runs under one identical config, which is exactly what a quantization-degradation table needs.
```

### When to trust a harness delta

Here is the load-bearing idea of the chapter. An absolute harness number carries all the config-dependence above, so its trustworthiness as a cross-source claim is low. But a **delta between two of my own runs under a single fixed config** cancels almost all of that dependence. If I hold the task version, few-shot count, filters, decoding, chat template, prompt, item set, and machine fixed, and change *only* the thing under study (BF16 to INT4 weights), then the difference in scores isolates the effect of that one change. The shared config is a control: every construct-irrelevant knob is held constant, so its contribution subtracts out of the delta. This is why a quantization-degradation *table* is scientifically stronger than any single number in it. The absolute 63.1% might not be comparable to anyone else's 63.1%, but "INT4 costs 1.4 points versus BF16 on this machine under this config" is a clean, defensible causal-flavored claim, because the comparison is paired and everything else is nailed down.

The delta is trustworthy when: the two runs share one config, the item set is identical (same seed, same order), decoding is greedy (removing sampling variance), and the difference exceeds the sampling noise floor from chapter 1 ($\text{SE} = \sqrt{p(1-p)/N}$, roughly $\pm 1.3$ points at 95% confidence on the full 1319-item GSM8K test set near $p=0.6$). The delta is *not* trustworthy when the runs differ in any hidden setting, or when the difference is smaller than the noise floor, in which case the honest report is "no detectable degradation at $N=1319$", not "zero degradation".

```admonish derivation title="What noise floor must a quantization delta clear on full GSM8K"
The GSM8K test set has $N = 1319$ items. Modeling accuracy as a binomial mean (chapter 1, equation 1), the standard error at accuracy $\hat{p}$ is $\sqrt{\hat{p}(1-\hat{p})/N}$. At $\hat{p} = 0.60$,

$$
\operatorname{SE} = \sqrt{\frac{0.60 \cdot 0.40}{1319}} = \sqrt{\frac{0.24}{1319}} \approx \sqrt{1.82\times10^{-4}} \approx 0.0135,
$$

so about $\pm 1.35$ percentage points at one SE, or $\pm 2.6$ points at 95% for a *single* run. But the quantization comparison is **paired** (same items, same order), so the relevant quantity is the standard error of the *difference*, which is smaller than the naive sum because the two runs are positively correlated: items are correct or wrong together far more often than chance. Chapter 7 does this properly with McNemar's test on the discordant pairs; the preview is that pairing can shrink the effective noise floor on the delta to well under a point, which is why a 1-to-2 point INT4 degradation is detectable on the full set even though it would be invisible on a 200-item slice. The table must run the full 1319 items, and the delta's significance is a paired test, not two independent confidence intervals eyeballed for overlap.
```

### The memory arithmetic that makes this comparison matter on one GPU

Quantization is not an academic curiosity on the baseline machine; it is what makes the model fit. The weight footprint follows directly from bytes-per-parameter. An 8B model has $N \approx 8\times10^9$ parameters, so at BF16 (2 bytes/param) the weights alone need

$$
8\times10^9 \times 2\ \text{bytes} = 1.6\times10^{10}\ \text{bytes} \approx 16\ \text{GB},
$$

which saturates the RTX 5080's 16 GB before a single byte of KV cache, so BF16 8B runs only with a short context (or a slightly sub-8B checkpoint) on this card. INT4 (roughly 0.5 bytes/param plus small scale and zero-point overhead) drops the weights to

$$
8\times10^9 \times 0.5\ \text{bytes} \approx 4\ \text{GB},
$$

freeing about 12 GB for KV cache and higher throughput. So the table is not just "does INT4 hurt accuracy"; it is "INT4 is what lets this model breathe on 16 GB, and here is the accuracy I pay for that". The whole thesis runs on quantized models, and this table is the price tag.

## Tooling

The harness reaches my vLLM server through the `local-completions` model type, which speaks the OpenAI *completions* API that vLLM serves at `/v1/completions`. (There is also `local-chat-completions` for the chat endpoint; for GSM8K few-shot with raw prompt formatting, `local-completions` against the completions endpoint is the standard choice.) Install with `uv`.

```bash
uv init quant-table
cd quant-table
uv add "lm-eval" openai
```

The invocation names the endpoint, the served model, and the task config explicitly, so the config is captured in the command and in the results JSON:

```bash
lm_eval \
  --model local-completions \
  --model_args base_url=http://localhost:8000/v1/completions,model=qwen-8b,num_concurrent=4,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 1 \
  --gen_kwargs temperature=0 \
  --output_path results/bf16 \
  --log_samples
```

`--log_samples` writes every per-item prompt, completion, and score to disk, the harness's version of chapter 4's audit object, so I can inspect exactly what was graded. `--output_path` gets a results JSON containing both filter scores (`strict-match` and `flexible-extract`), their stderrs, and the full config block.

```admonish under-the-hood title="The harness sends prompts; vLLM must be the same server for both runs"
The comparison's validity depends on *nothing changing but the weights*. That means I serve BF16, run the harness, stop the server, serve the INT4 build of the same base model (same tokenizer, same chat template, same vLLM version and flags except quantization), and run the identical harness command. If I change the vLLM version, the max context, the tokenizer, or the sampling between runs, I have contaminated the delta with a second variable. The clean protocol is: one harness command, saved verbatim, run twice against two server configs that differ only in `--quantization`. Record the vLLM launch flags for both servers alongside the table, because those flags are part of the experiment.
```

## Lab

The artifact is a **committee-grade quantization-degradation table**: GSM8K (5-shot, full 1319 items, greedy) on the 8B model at BF16 and at INT4, under one identical harness config, with both filters, stderrs, the paired delta, and the derived memory footprints. Because there is no GPU in this authoring environment, the measured accuracy cells carry the honest placeholder and get filled on the baseline machine; the arithmetic cells (memory) are computed here, and the script that builds the table from the harness JSON is complete and runnable.

Run the harness twice against the two server configs. First serve BF16 (from Part II), run; then serve the INT4/AWQ build of the same base model, run the identical command with a different `--output_path`.

```bash
# --- run A: BF16 server on :8000, served as qwen-8b ---
lm_eval --model local-completions \
  --model_args base_url=http://localhost:8000/v1/completions,model=qwen-8b,num_concurrent=4,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --gen_kwargs temperature=0 \
  --output_path results/bf16 --log_samples

# --- stop BF16 server, serve INT4/AWQ build of the SAME base model, then: ---
lm_eval --model local-completions \
  --model_args base_url=http://localhost:8000/v1/completions,model=qwen-8b,num_concurrent=4,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --gen_kwargs temperature=0 \
  --output_path results/int4 --log_samples
```

The table builder reads both results JSONs, extracts the two filter scores and stderrs, computes the paired-relevant delta and the derived memory footprints, and writes a markdown table plus a machine-readable JSON. It degrades gracefully: if a results file is missing (no GPU here), it emits the placeholder row so the table structure is reviewable before the run.

```python title="quant/build_table.py"
"""Build the GSM8K BF16-vs-INT4 quantization-degradation table.

Reads lm-eval results JSON from results/bf16 and results/int4, extracts the
strict-match and flexible-extract exact_match scores with stderrs, computes the
delta, derives weight-memory footprints from bytes-per-parameter, and writes a
committee-grade markdown table + a JSON sidecar.

Run: uv run python -m quant.build_table
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

PARAMS = 8.03e9          # 8B model parameter count (record exact for your ckpt)
PLACEHOLDER = "(record on baseline machine: value, date, driver)"


def weight_gb(bytes_per_param: float) -> float:
    return PARAMS * bytes_per_param / 1e9


def load_scores(output_dir: str) -> dict | None:
    """lm-eval writes results_*.json under output_path; grab the newest."""
    files = sorted(glob.glob(f"{output_dir}/**/results_*.json", recursive=True))
    if not files:
        return None
    data = json.loads(Path(files[-1]).read_text())
    g = data["results"]["gsm8k"]
    return {
        "strict": g.get("exact_match,strict-match"),
        "strict_se": g.get("exact_match_stderr,strict-match"),
        "flex": g.get("exact_match,flexible-extract"),
        "flex_se": g.get("exact_match_stderr,flexible-extract"),
    }


def fmt(v: float | None, se: float | None) -> str:
    if v is None:
        return PLACEHOLDER
    pct, pse = 100 * v, 100 * (se or 0)
    return f"{pct:.1f} ± {pse:.1f}"


def delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return PLACEHOLDER
    return f"{100 * (b - a):+.1f} pp"


def main() -> None:
    bf16 = load_scores("results/bf16")
    int4 = load_scores("results/int4")
    mem_bf16, mem_int4 = weight_gb(2.0), weight_gb(0.5)

    rows = [
        ("Precision", "Weights (GB, derived)", "GSM8K strict",
         "GSM8K flexible", "Delta vs BF16 (flexible)"),
        ("BF16", f"{mem_bf16:.1f}",
         fmt(bf16 and bf16["strict"], bf16 and bf16["strict_se"]),
         fmt(bf16 and bf16["flex"], bf16 and bf16["flex_se"]), "n/a"),
        ("INT4 (AWQ)", f"{mem_int4:.1f}",
         fmt(int4 and int4["strict"], int4 and int4["strict_se"]),
         fmt(int4 and int4["flex"], int4 and int4["flex_se"]),
         delta(bf16 and bf16["flex"], int4 and int4["flex"])),
    ]

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for j, r in enumerate(rows):
        lines.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
        if j == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    table_md = "\n".join(lines)

    caption = (
        "\nGSM8K, 5-shot, greedy, full 1319-item test set, one identical "
        "lm-eval config; INT4 is AWQ of the same base checkpoint. "
        "Weights derived at 2.0 / 0.5 bytes-per-param over "
        f"{PARAMS:.2e} params. Delta significance is a paired McNemar test "
        "(chapter 7), not overlapping single-run intervals."
    )

    Path("results").mkdir(exist_ok=True)
    Path("results/quant_table.md").write_text(table_md + "\n" + caption + "\n")
    Path("results/quant_table.json").write_text(json.dumps(
        {"bf16": bf16, "int4": int4,
         "weights_gb": {"bf16": mem_bf16, "int4": mem_int4}}, indent=2))
    print(table_md + "\n" + caption)
    print("\nwrote results/quant_table.md and results/quant_table.json")


if __name__ == "__main__":
    main()
```

Build the table.

```bash
uv run python -m quant.build_table
```

### What you should see

Before the GPU runs (in this authoring environment) the script still succeeds, computing the derived memory column and emitting the placeholder in every measured cell, so the table's structure is reviewable now:

```
| Precision  | Weights (GB, derived) | GSM8K strict                              | GSM8K flexible                            | Delta vs BF16 (flexible)                  |
|------------|-----------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| BF16       | 16.1                  | (record on baseline machine: value, date, driver) | (record on baseline machine: value, date, driver) | n/a                                       |
| INT4 (AWQ) | 4.0                   | (record on baseline machine: value, date, driver) | (record on baseline machine: value, date, driver) | (record on baseline machine: value, date, driver) |
```

The derived weights column is real arithmetic: 16.1 GB at BF16 (which is why the model barely fits on the 5080's 16 GB, leaving almost nothing for KV cache) and 4.0 GB at INT4 (which is what frees the card). After running both `lm_eval` invocations on the baseline machine, rerun the builder and the measured cells fill with `XX.X ± Y.Y` for both filters, plus a signed `Delta vs BF16` in percentage points. The expected shape, to be confirmed by measurement, is a small INT4 degradation on the order of a point or two on flexible-extract, with strict-match sometimes moving more because format compliance is fragile under quantization; whatever the number, it must be reported with its stderr and adjudicated against the paired noise floor from the derivation, not eyeballed. Record the vLLM launch flags for both servers, the harness commit, the date, and the driver alongside the table.

The artifacts are `results/quant_table.md` (the committee-grade table with its caption naming every config choice) and `results/quant_table.json` (the machine-readable scores for later plots), plus the `--log_samples` per-item logs from both runs. This is the first table in the thesis a committee can actually trust, and the reason it is trustworthy is everything this chapter argued: the absolute numbers may not be comparable to anyone else's, but the *delta* is a controlled paired comparison under one frozen config, so "INT4 costs this much reasoning accuracy on this machine" is a defensible measurement rather than a vibe. Chapter 7 tightens the significance claim; chapter 9 turns this procedure into the thesis's standing evaluation suite.

```admonish substack-seed
"Stop comparing benchmark numbers across papers. Compare deltas you control." Almost every argument about model quality cites absolute benchmark scores as if they were comparable, and they mostly aren't: the same model gets wildly different GSM8K numbers depending on few-shot count, prompt format, answer-extraction regex, and whether decoding was greedy. An evaluation harness doesn't fix that by making absolute numbers universal; it fixes it by letting you freeze every one of those knobs and change exactly one thing. The post's worked example is a quantization-degradation table: the same 8-billion-parameter model at full precision and at 4-bit, run through one identical config, so the difference between them isolates the one variable you care about. The punchline is that the absolute 63% might mean nothing to anyone else, but "4-bit costs 1.4 points of reasoning here, and buys back 12 gigabytes of GPU memory" is a real, defensible, decision-grade fact, and the memory half of it is just multiplication you can do before you ever touch the GPU.
```
