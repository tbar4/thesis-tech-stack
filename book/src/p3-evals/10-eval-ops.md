# Eval ops

Every chapter in this part produced an artifact: scored responses, a judge reliability report, the `evalstats` module, a contamination report, the frozen suite. This chapter is about the boring infrastructure that keeps all of those from rotting: where the files live, how they get logged, when they move to the archive tier, and the one operating rule that saves more time and compute than any other in the book. That rule is: **never re-generate what you can re-score.** Generation is the expensive, stochastic, hard-to-reproduce step; scoring is cheap and repeatable. If you keep them separate and persist the generations immutably, then changing a scorer, fixing a verifier, or swapping a judge costs seconds instead of GPU-hours, and every re-analysis is provably taken on the same responses. This chapter turns that principle into an MLflow schema, a directory layout, an archive policy, and a re-scoring workflow, and packages them as a runbook you can follow at 2am without thinking.

## Theory

### The two axes: generation and scoring

An eval result is the composition of two very different operations. **Generation** runs the model: it consumes VRAM, takes real wall-clock time, depends on the serving stack and the seed, and is only approximately reproducible because decoding is stochastic and kernels are nondeterministic across driver versions. **Scoring** runs a function over the saved generations: it is cheap, CPU-bound, deterministic given the scorer version, and perfectly reproducible. The single most consequential design decision in eval ops is to treat these as separate, persisted stages rather than one fused pipeline.

When they are fused (generate-and-score in one script that throws away the raw text) you pay the generation cost every time you touch anything. Fix a bug in the answer-extraction regex? Regenerate. Add a per-stratum breakdown? Regenerate. Try the challenger judge from Chapter 3.6? Regenerate. Each regeneration also *changes the responses* (different seed, different decode), so you can no longer tell whether a metric moved because your scorer changed or because the generations changed. When they are separated (generate once, persist, score many times) all of those become re-scores over a fixed, content-addressed set of responses, and the only thing that varies is the thing you meant to vary.

```admonish gotcha
The subtle trap is re-generating "just to be safe" when you actually only changed the scorer. It feels harmless, but it silently confounds your comparison: you wanted to measure the effect of the new scorer and instead you measured the new scorer *plus* a fresh draw of generations. If you ever find yourself re-running the model because you tweaked a rubric, a verifier, or a judge, stop. The saved `generations.jsonl` is the input; re-score it. Regeneration is reserved for exactly one situation: the model, its weights, its decoding parameters, or the prompt template changed. Nothing else.
```

### Content addressing makes provenance unambiguous

The mechanism that makes "re-score, don't regenerate" auditable is content addressing. When you persist a set of generations, you hash it (SHA-256 over the canonicalized responses) and that hash *is* the identity of the generation set. Every score run then records which generation hash it consumed. Now a metric is not floating free; it is bound to (this exact set of responses, scored by this exact scorer version). Two score runs that cite the same generation hash are guaranteed comparable because they scored literally the same text. A re-score after fixing a verifier is a new score run citing the same generation hash, and the diff between old and new metrics is attributable entirely to the verifier change, with proof. This is the same discipline the suite freeze used in Chapter 3.9, applied to the hot path.

### The MLflow schema for eval runs

MLflow is the tracking spine for the whole book, and eval runs get a fixed schema so that a run from six months ago is legible today. The natural hierarchy is a **parent run per eval campaign** with **nested child runs per model** (or per phase, before/after training), which keeps a comparison's members grouped. The schema pins down what goes in each of MLflow's four buckets.

Params (the inputs that define the run): model name and revision, suite name plus version plus content hash, decoding parameters (temperature, top-p, max tokens, seed), the judge model plus version and token budget if a judge was used, and the generation hash the scoring consumed. Metrics (the outputs, never bare): the accuracy point estimate with its bootstrap CI bounds from `evalstats`, per-stratum accuracies, and for a comparison the paired difference with its CI and p-value. Tags (the searchable metadata): the git commit of the eval code, the suite git tag, the pipeline phase, and the scorer version. Artifacts (the files): `generations.jsonl`, `scores.jsonl`, `metrics.json`, and the `manifest.json` that ties them together by hash. The rule that makes this pay off is that a metric logged without its CI is a schema violation; Chapter 3.7 built the module precisely so that logging the interval is as easy as logging the point.

### Artifact layout on the working tier

Runs live on the 1TB NVMe working tier in a flat, dated, self-describing convention:

```
runs/
  2026-07-22_qwen3-14b-awq_suite-v1.0/
    generations.jsonl      # immutable raw model responses (the expensive artifact)
    scores.jsonl           # per-item score under a named scorer version
    metrics.json           # aggregate metrics + evalstats CIs
    manifest.json          # hashes tying it all together; provenance
    logs/                  # serving + scoring logs
```

The directory name encodes date, model, and suite version so it sorts chronologically and is identifiable at a glance. `generations.jsonl` is written once and never edited; `scores.jsonl` may have siblings (`scores.judge-qwen.jsonl`, `scores.verifier-v2.jsonl`) when the same generations are scored multiple ways. The `manifest.json` is the index: it records the generation hash, which scorer produced which scores file, and the MLflow run id, so the on-disk directory and the tracking server never drift apart.

### Archive policy: promotion to the NAS

The NVMe tier is for the working set; the 5TB NAS is the archive. The discipline from the hardware-baseline chapter becomes a concrete promotion rule here. A run is **promoted** when it is done: tagged, referenced in a result, and unlikely to be re-scored soon. Promotion moves the bulky, cold artifact (`generations.jsonl`, which dominates the byte count) to the NAS, leaves the small hot artifacts (`metrics.json`, `manifest.json`) on NVMe, and records the NAS path in the manifest so the generations are still findable. The metrics stay fast to query; the raw text stops eating the fast disk. Nothing in the hot path ever reaches across the network, because a re-score of an *active* run happens before promotion, while its generations are still on NVMe. The retention rule is simple and generous: generations are never deleted, only moved, because they are the expensive thing and disk is cheaper than a GPU-hour.

## Tooling

The conventions above are enforced by a thin `evalops` helper rather than left to discipline, because a convention you have to remember is a convention you will violate at 2am. It does four things: `record_generation` writes and hashes a generation set and logs it to MLflow; `record_scoring` scores an existing generation set (by hash) with a named scorer and logs the metrics with their `evalstats` intervals; `promote` moves a finished run's generations to the NAS and updates the manifest; and `resolve` finds a generation set's current location (NVMe or NAS) by hash. MLflow is the spine; the helper just guarantees the schema and the content-addressing are applied every time. The primary artifact of this chapter is the runbook that ties these into a sequence; the helper is the runbook made executable.

## Lab

We produce the eval-ops runbook (the artifact) plus the `evalops` helper it references, then walk the worked sequence that proves the discipline: generate once, score with a verifier, re-score with a fixed verifier *without regenerating*, and confirm both scorings cite the same generation hash in MLflow.

### Project setup

```bash title="setup.sh"
uv init evalops && cd evalops
uv add "mlflow>=2.14" "numpy>=1.26"
uv add --editable ../evalstats     # Chapter 3.7
uv add --editable ../thesis-suite  # Chapter 3.9 verifiers
```

### The helper

```python title="evalops/ops.py"
"""Enforce the eval-ops schema: content-addressed generations, re-scorable."""
import json, hashlib, os, shutil
from pathlib import Path
from datetime import date
import mlflow
import evalstats as es

WORKING = Path("runs")                 # NVMe working tier
ARCHIVE = Path("/mnt/nas/eval-archive")  # 5TB NAS archive tier

def _hash_generations(records) -> str:
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x["id"]):
        h.update(json.dumps({"id": r["id"], "response": r["response"]},
                            sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()

def run_dir(model: str, suite_version: str) -> Path:
    slug = model.split("/")[-1].lower()
    d = WORKING / f"{date.today().isoformat()}_{slug}_suite-v{suite_version}"
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d

def record_generation(records, model, revision, suite_version, suite_hash,
                      decode_params):
    """Persist raw responses ONCE, hash them, log to MLflow. The expensive step."""
    d = run_dir(model, suite_version)
    (d / "generations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records))
    gen_hash = _hash_generations(records)
    manifest = {"model": model, "revision": revision,
                "suite_version": suite_version, "suite_hash": suite_hash,
                "generation_sha256": gen_hash, "decode_params": decode_params,
                "scorings": {}, "generations_location": str(d / "generations.jsonl"),
                "promoted": False}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    mlflow.set_experiment("evals")
    with mlflow.start_run(run_name=f"gen-{d.name}") as run:
        mlflow.log_params({"model": model, "revision": revision,
                           "suite_version": suite_version, **decode_params})
        mlflow.set_tags({"generation_sha256": gen_hash, "phase": "generate"})
        mlflow.log_artifact(str(d / "generations.jsonl"))
        manifest["gen_run_id"] = run.info.run_id
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"generated {len(records)} responses -> {d} (hash {gen_hash[:12]})")
    return d, gen_hash

def record_scoring(run_path, scorer_fn, scorer_version, items_by_id, groups=None):
    """Score EXISTING generations (never regenerate). Log metrics + CIs."""
    d = Path(run_path)
    manifest = json.loads((d / "manifest.json").read_text())
    gens = [json.loads(l) for l in
            (d / "generations.jsonl").read_text().splitlines()]
    scores = []
    for g in gens:
        ok = scorer_fn(items_by_id[g["id"]], g["response"])
        scores.append({"id": g["id"], "score": int(ok)})
    scores_file = d / f"scores.{scorer_version}.jsonl"
    scores_file.write_text("\n".join(json.dumps(s) for s in scores))

    y = [s["score"] for s in scores]
    acc = es.bootstrap_mean(y, groups=groups, name="accuracy")   # never bare
    metrics = {"scorer_version": scorer_version,
               "generation_sha256": manifest["generation_sha256"],
               "accuracy": acc.to_dict()}
    (d / f"metrics.{scorer_version}.json").write_text(json.dumps(metrics, indent=2))
    manifest["scorings"][scorer_version] = scores_file.name
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))

    mlflow.set_experiment("evals")
    with mlflow.start_run(run_name=f"score-{scorer_version}-{d.name}"):
        # Same generation hash => provably same responses as any other scoring.
        mlflow.set_tags({"generation_sha256": manifest["generation_sha256"],
                         "scorer_version": scorer_version, "phase": "score"})
        mlflow.log_params({"suite_version": manifest["suite_version"]})
        mlflow.log_metrics(acc.as_mlflow())
        mlflow.log_artifact(str(scores_file))
    print(f"scored with {scorer_version}: acc={acc.point:.3f} "
          f"[{acc.ci_low:.3f}, {acc.ci_high:.3f}] on gen {manifest['generation_sha256'][:12]}")
    return acc

def promote(run_path):
    """Move cold generations to NAS; keep hot metrics on NVMe. Never delete."""
    d = Path(run_path)
    manifest = json.loads((d / "manifest.json").read_text())
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE / d.name
    dest.mkdir(exist_ok=True)
    shutil.move(str(d / "generations.jsonl"), str(dest / "generations.jsonl"))
    manifest["generations_location"] = str(dest / "generations.jsonl")
    manifest["promoted"] = True
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"promoted generations -> {dest} (metrics stay on NVMe)")

def resolve(run_path) -> str:
    """Find the generations for a run, on NVMe or NAS, by manifest."""
    manifest = json.loads((Path(run_path) / "manifest.json").read_text())
    loc = manifest["generations_location"]
    if not Path(loc).exists():
        raise FileNotFoundError(f"generations missing at {loc}; check NAS mount")
    return loc
```

### The worked sequence

```python title="evalops/demo.py"
"""Generate once, score twice (fix a verifier) without regenerating."""
import json
from pathlib import Path
from evalops.ops import record_generation, record_scoring, promote

# Pretend these came from a served model over the frozen suite (Ch 3.9).
GENERATIONS = [
    {"id": "sda-0002", "response": "Filtering... the count is 2."},
    {"id": "sda-0004", "response": "1,3"},   # no space: v1 exact-match fails, v2 set-check passes
]
ITEMS = {
    "sda-0002": {"verifier": "numeric", "target": "2"},
    "sda-0004": {"verifier": "set", "target": "1, 3"},
}

def verifier_v1(item, response):
    # v1 set-checker: naive split, space-sensitive -> WRONG on "1,3"
    if item["verifier"] == "numeric":
        import re
        n = re.findall(r"-?\d+", response); return bool(n) and n[-1] == item["target"]
    return response.strip() == item["target"]   # brittle exact string

def verifier_v2(item, response):
    # v2: use the audited suite verifiers (order- and space-insensitive set)
    from suite.verifiers import check
    return check(item, response)

if __name__ == "__main__":
    d, gh = record_generation(
        GENERATIONS, model="Qwen/Qwen3-14B-AWQ", revision="<sha>",
        suite_version="1.0", suite_hash="<suite-sha>",
        decode_params={"temperature": 0.7, "top_p": 0.95,
                       "max_tokens": 512, "seed": 0})
    # Score with the buggy v1 verifier.
    record_scoring(d, verifier_v1, "verifier-v1", ITEMS)
    # Fix the verifier and RE-SCORE the SAME generations. No regeneration.
    record_scoring(d, verifier_v2, "verifier-v2", ITEMS)
    # Campaign done: promote cold generations to the NAS.
    promote(d)
```

### Comparing two runs the right way

The payoff of the schema is that an A-versus-B comparison (BF16 versus INT4, before-training versus after) becomes a paired analysis over two scorings that share a suite, wrapped in a parent campaign run. Because both models were scored on the same frozen suite items, the correct test is McNemar or a paired-difference bootstrap from Chapter 3.7, never two marginal CIs eyeballed for overlap. The helper below reads two runs' score files, aligns them by item id, and logs the paired result under a parent run.

```python title="evalops/compare.py"
"""Paired A/B comparison over two runs that scored the same frozen suite."""
import json
from pathlib import Path
import mlflow
import evalstats as es

def load_scores(run_dir, scorer_version):
    d = Path(run_dir)
    rows = [json.loads(l) for l in
            (d / f"scores.{scorer_version}.jsonl").read_text().splitlines()]
    return {r["id"]: r["score"] for r in rows}

def compare(run_a, run_b, scorer_version, label_a, label_b):
    sa, sb = load_scores(run_a, scorer_version), load_scores(run_b, scorer_version)
    ids = sorted(set(sa) & set(sb))          # shared items only -> paired
    a = [sa[i] for i in ids]; b = [sb[i] for i in ids]
    delta = es.bootstrap_paired_diff(a, b, name="delta")   # Ch 3.7 paired CI
    mc = es.mcnemar(a, b)                                   # exact paired test
    mlflow.set_experiment("evals")
    with mlflow.start_run(run_name=f"compare-{label_a}-vs-{label_b}"):
        mlflow.log_params({"model_a": label_a, "model_b": label_b,
                           "scorer_version": scorer_version, "n_shared": len(ids)})
        mlflow.log_metrics({**delta.as_mlflow(), "mcnemar_p": mc.pvalue})
    print(f"delta(a-b)={delta.point:+.3f} [{delta.ci_low:+.3f},{delta.ci_high:+.3f}]"
          f"  McNemar p={mc.pvalue:.4f}  (n_shared={len(ids)})")
    return delta, mc
```

The parent-run/child-run shape falls out naturally: the two `gen`/`score` runs are the children (one per model), and the `compare` run is the parent that records the verdict. A reviewer opening MLflow six months later sees the two models, the exact suite version and hash they were scored on, the shared-item count, and a paired difference with its confidence interval and p-value, which is a defensible claim rather than two floating numbers.

### The runbook

```markdown title="RUNBOOK.md"
# Eval Ops Runbook

## The one rule
Never re-generate what you can re-score. Regenerate ONLY when the model, its
weights, decoding params, or prompt template change. A scorer/verifier/judge/
rubric change is a RE-SCORE over the saved generations, never a regeneration.

## Generate (expensive; do once)
1. Serve the model with vLLM (serve/save/swap; see Ch 3.6).
2. record_generation(...) -> writes runs/<date>_<model>_suite-vX/generations.jsonl,
   hashes it (SHA-256), logs an MLflow gen-run tagged generation_sha256.
3. generations.jsonl is now IMMUTABLE. Never edit it.

## Score / re-score (cheap; do often)
1. record_scoring(run_dir, scorer_fn, scorer_version, items) over the SAME
   generations. Writes scores.<version>.jsonl + metrics.<version>.json.
2. Every metric carries an evalstats bootstrap CI. Bare metrics are forbidden.
3. Every score run logs the SAME generation_sha256 => scorings are provably
   comparable (same responses). Compare A/B with evalstats.mcnemar / paired diff.

## MLflow schema
- Parent run per campaign; child run per model/phase.
- params: model, revision, suite_version, suite_hash, decode params, judge+budget.
- metrics: accuracy.point/ci_low/ci_high, per-stratum, paired delta + CI + p.
- tags: git commit, suite tag, phase (generate|score), scorer_version, generation_sha256.
- artifacts: generations.jsonl, scores.*.jsonl, metrics.*.json, manifest.json.

## Storage tiers
- Working (NVMe, runs/): active generations + all metrics. Hot path never hits NAS.
- Archive (NAS, /mnt/nas/eval-archive/): promoted generations.

## Archive policy
- Promote when a run is done: tagged, cited in a result, unlikely to re-score soon.
- promote(run_dir) moves generations.jsonl to NAS, keeps metrics/manifest on NVMe,
  records NAS path in manifest. Generations are MOVED, never DELETED.
- Re-score BEFORE promotion (generations still on NVMe). To re-score a promoted
  run, resolve(run_dir) locates the generations on the NAS first.

## Disaster check
- resolve(run_dir) must succeed for every cited result. If it raises, the NAS
  mount is down or a generation set was deleted (it never should be).
```

### The artifact and what you should see

The artifact is `RUNBOOK.md`, backed by the executable `evalops` helper and a worked demo. Running `demo.py`, you should see one generation event (the expensive step, run once, producing an immutable `generations.jsonl` with a recorded SHA-256) followed by two scoring events over that same file. The first scoring uses a buggy set-verifier that is space-sensitive and marks the correct answer `"1,3"` wrong; the second uses the audited Chapter 3.9 verifier and marks it right. The accuracy moves between the two scorings, and (this is the whole point) both MLflow score runs are tagged with the identical `generation_sha256`, so the improvement is provably attributable to the verifier fix and not to a fresh, different draw of generations, because the model was never re-run. Each accuracy prints with its bootstrap confidence interval, never bare. Finally `promote` moves the generations to the NAS and rewrites the manifest, leaving the metrics hot on NVMe, and `resolve` still finds the generations by hash. What you have built is the operating discipline that makes the whole loop cheap to iterate and impossible to confound: generate once, score forever, log everything with its uncertainty, keep the fast disk fast, and never throw away the expensive thing. Wall-clock and disk figures for a full-suite run are (measured on the baseline machine -- record value, date, driver).

```admonish read-along
[RLHF] and the loop's later parts assume you can cheaply re-score old generations against a new reward function, which is exactly the capability this chapter's discipline provides. When Part VII turns eval scorers into training rewards, "re-score, don't regenerate" is what lets you audit a reward change against a fixed set of policy samples without paying to sample again. Read this runbook as the operational precondition for treating evals as rewards: the two only compose cheaply if generation and scoring are already separated and content-addressed. **[AIE]** ch. 10 is the production mirror of this runbook (evaluation wired into the serving architecture, with user feedback as the ongoing signal); the pairing is looser than most in this book, but the operational mindset, that evaluation is a pipeline you run forever rather than a report you run once, is the same.
```

```admonish substack-seed
The most expensive habit in model evaluation is re-running the model every time you change how you score it. Generation costs GPU-hours and is only approximately reproducible; scoring costs milliseconds and is exact. Fuse them and every rubric tweak, verifier fix, or judge swap costs you a fresh (and subtly different) set of generations, which also quietly ruins your before/after comparison. Separate them, hash the generations, and the discipline pays off forever: you generate once, score a hundred ways, and every metric is provably taken on the same responses because they all cite the same content hash. Keep the raw model output like it is gold, because compared to the electricity that made it, it is. Never re-generate what you can re-score.
```
