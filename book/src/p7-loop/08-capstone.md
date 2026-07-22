# Capstone: the loop end-to-end

This chapter introduces nothing. That is the whole point of it. Every tool it uses was built in an earlier chapter, every statistic comes from a module already written, every measured slot is stamped the way the book has stamped measurements since the hardware baseline. What the capstone does is run the entire loop in one sitting, on the baseline machine, start to finish, so that the thesis stops being a collection of chapters that each work in isolation and becomes a single pipeline that produces a result. Serve the baseline, evaluate it, train on the loop, merge the adapter, re-serve, re-evaluate, make the figures, archive everything. When the archive lands on the NAS with its provenance intact, the thesis pipeline is real: not a claim about what could be measured, but an artifact on disk that was.

I want to be honest about what "one sitting" means on a single 16GB card. It does not mean fast. It means uninterrupted and reproducible: one orchestration script, one command, one archive at the end, no manual step that a later me could get wrong. The serving and training phases cannot both hold their heavyweight state in 16GB at once (that constraint has driven the whole book), so the loop is a sequence in time, not a thing running in parallel, and the orchestration exists precisely to sequence it correctly and reclaim VRAM between phases.

## Theory

There is no new theory. There is only the shape of the loop, which every prior chapter has been a piece of, assembled once so you can see the whole thing at altitude.

The loop is a directed sequence with one return edge, and each node is a chapter:

- **Serve $M_0$.** Bring up the baseline policy on the chapter 2.6 inference substrate, the same vLLM configuration every measurement in the book has used. This is the fixed serving substrate; nothing about how the model is served changes between the pre and post halves, so the model is the only thing that varies.
- **Evaluate $M_0$.** Run the frozen thesis task suite v1.0 (chapter 3.9) against the served baseline, logging per-item, per-sample scores in the chapter 3.7 format. This is the "pre" measurement of chapter 7.6.
- **Train.** Tear down the server to reclaim VRAM, then run GRPO on 16GB (chapter 7.2) with the Part III verifier scorers wired in as rewards (chapter 7.3), on the difficulty-banded, provenance-stamped, eval-deduped prompt set (chapter 7.5). Output is a LoRA adapter.
- **Merge.** Merge the adapter into the base weights to produce $M_1$, a standalone checkpoint the serving substrate can load exactly as it loads $M_0$. Merging matters because it makes the post model served identically to the pre model, closing off "you served them differently" as an alternative explanation for the delta.
- **Re-serve and re-evaluate $M_1$.** Bring the substrate back up on the merged $M_1$ and run the identical suite at the identical sampling budget, logging in the identical format. This is the "post" measurement.
- **Figures.** Run the chapter 7.6 delta report on the two logs to produce the paired delta, its bootstrap CI, the permutation $p$-value, the effect size, and the pass@k curves.
- **Archive.** Promote every artifact from working NVMe to the 5TB NAS with a manifest, so the whole run is reproducible from the archive alone.

The return edge is the delta: the loop's output is the measured difference between the re-evaluation and the evaluation, which is the dependent variable the entire thesis is organized to estimate. Running it once, cleanly, is the capstone.

## Tooling

The tool is an orchestrator, and it is deliberately thin: it calls the chapter tools in order, checks that each phase produced its expected artifact before starting the next, reclaims VRAM between the serve and train phases, and writes one capstone report at the end that stitches the provenance of every phase together. It introduces no new logic because introducing new logic would violate the premise. The right shape for it is a small task runner so that each phase is a named target with explicit inputs and outputs, and a failed phase stops the loop rather than corrupting the archive.

```bash title="shell: the capstone project (orchestration only)"
uv init capstone
cd capstone
uv add "pydantic>=2.7" "pyyaml>=6.0"
# the actual work lives in the chapter projects, invoked as subprocesses:
#   ../curriculum  (7.5)   ../analysis (7.6)   ../evalstats (3.7)
#   plus the 2.6 serving config, 7.2/7.3 trainer, 3.9 frozen suite
```

### The run configuration

One file names everything the loop needs, and every path points at an artifact an earlier chapter already produced. Nothing here is new; it is a table of contents for the book made executable.

```yaml title="capstone/loop.yaml"
# Capstone loop configuration. Every entry references an artifact from
# an earlier chapter. No new concepts -- only wiring.

machine: "MSI Aegis R2 (Core Ultra 9 285, RTX 5080 16GB, Ubuntu 24.04)"
driver: "record: e.g. 570-open"          # measured on the baseline machine
run_date: "record: ISO date of this run"

base_model: "Qwen/Qwen2.5-3B-Instruct"   # M0
serve:
  substrate: "vllm (chapter 2.6 config)"
  port: 8000
  endpoint: "http://localhost:8000/v1"

suite:
  path: "../thesis_suite_v1.jsonl"        # frozen 3.9 suite
  revision: "thesis-suite v1.0"
  n_samples: 16                            # matched across pre/post
  temperature: 0.8
  max_tokens: 1024

train:
  prompt_set: "../curriculum/out/sda_train_prompts.v1.jsonl"   # 7.5
  provenance: "../curriculum/out/provenance.json"              # 7.5
  reward: "full"                            # correctness + format (7.3)
  group_size_K: 8                           # 7.2
  kl_beta: 0.02
  seed: 0
  adapter_out: "work/adapter"

merge:
  merged_out: "work/M1_merged"

archive:
  nas_root: "/mnt/nas/thesis-runs"          # 5TB NAS archive tier
```

### The orchestrator

```python title="capstone/run_loop.py"
"""Run the whole thesis loop end-to-end, one sitting, no new concepts.

Each phase invokes a tool built in an earlier chapter and checks its
artifact before the next phase starts. Serving and training never hold
VRAM at once: the server is torn down before training and brought back
up after merge. Nothing here computes a statistic or defines a reward --
it only sequences chapters and records provenance.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import yaml

CFG = yaml.safe_load(Path("loop.yaml").read_text())
WORK = Path("work")
WORK.mkdir(exist_ok=True)


def run(cmd: list[str], phase: str) -> None:
    print(f"\n=== phase: {phase} ===\n$ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)   # a failed phase stops the loop
    dt = time.time() - t0
    (WORK / f"{phase}.timing").write_text(
        f"{phase}: {dt:.1f}s  # measured on the baseline machine, "
        f"{CFG['run_date']}, driver {CFG['driver']}\n"
    )


def require(path: str, phase: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"phase {phase} did not produce {path}; loop halted")
    return p


def serve_up(model: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["vllm", "serve", model, "--port", str(CFG["serve"]["port"])]
    )
    # wait for readiness (chapter 2.6 has the real health-check;
    # this is the thin orchestration version)
    time.sleep(30)
    return proc


def serve_down(proc: subprocess.Popen) -> None:
    proc.terminate()
    proc.wait(timeout=60)
    time.sleep(5)   # let VRAM actually free before the next phase


def eval_suite(endpoint: str, model: str, out: str, phase: str) -> Path:
    run([
        "uv", "run", "--project", "../evalstats", "python", "-m", "evalstats.run_eval",
        "--endpoint", endpoint, "--model", model,
        "--suite", CFG["suite"]["path"], "--suite-rev", CFG["suite"]["revision"],
        "--n-samples", str(CFG["suite"]["n_samples"]),
        "--temperature", str(CFG["suite"]["temperature"]),
        "--max-tokens", str(CFG["suite"]["max_tokens"]),
        "--driver", CFG["driver"], "--measured-at", CFG["run_date"],
        "--out", out,
    ], phase)
    return require(out, phase)


def main() -> None:
    # ---- 1. serve + evaluate M0 (pre) ----
    proc = serve_up(CFG["base_model"])
    pre_log = eval_suite(CFG["serve"]["endpoint"], CFG["base_model"],
                         str(WORK / "eval_M0.json"), "eval_pre")
    serve_down(proc)   # reclaim VRAM before training

    # ---- 2. train (GRPO on 16GB, scorers-as-rewards, 7.5 prompt set) ----
    run([
        "uv", "run", "python", "-m", "trainer.grpo",
        "--base", CFG["base_model"],
        "--prompts", CFG["train"]["prompt_set"],
        "--reward", CFG["train"]["reward"],
        "--group-size", str(CFG["train"]["group_size_K"]),
        "--kl-beta", str(CFG["train"]["kl_beta"]),
        "--seed", str(CFG["train"]["seed"]),
        "--adapter-out", CFG["train"]["adapter_out"],
    ], "train")
    require(CFG["train"]["adapter_out"], "train")

    # ---- 3. merge adapter -> M1 ----
    run([
        "uv", "run", "python", "-m", "trainer.merge",
        "--base", CFG["base_model"],
        "--adapter", CFG["train"]["adapter_out"],
        "--out", CFG["merge"]["merged_out"],
    ], "merge")
    require(CFG["merge"]["merged_out"], "merge")

    # ---- 4. re-serve + re-evaluate M1 (post), identical budget ----
    proc = serve_up(CFG["merge"]["merged_out"])
    post_log = eval_suite(CFG["serve"]["endpoint"], CFG["merge"]["merged_out"],
                          str(WORK / "eval_M1.json"), "eval_post")
    serve_down(proc)

    # ---- 5. figures: the 7.6 delta report ----
    run([
        "uv", "run", "--project", "../analysis", "python", "run_delta.py",
        "--pre", str(pre_log), "--post", str(post_log),
        "--out", str(WORK / "delta"),
    ], "figures")
    require(str(WORK / "delta" / "delta_report.md"), "figures")

    # ---- 6. capstone report + archive ----
    write_capstone_report()
    archive_run()


def write_capstone_report() -> None:
    delta = json.loads((WORK / "delta" / "delta_report.json").read_text())
    prov = json.loads(Path(CFG["train"]["provenance"]).read_text())
    report = f"""# Capstone report -- thesis loop end-to-end

**Machine.** {CFG['machine']}
**Driver / date.** {CFG['driver']} / {CFG['run_date']}
(all timings and deltas measured on the baseline machine; see .timing files)

## Pipeline
serve M0 -> eval (pre) -> GRPO train -> merge -> re-serve M1 -> eval (post)
-> delta report -> archive

## Provenance of inputs
- base model (M0): {CFG['base_model']}
- frozen suite: {CFG['suite']['revision']}, {delta['n_items']} items,
  {delta['n_samples']} samples/item (matched pre/post)
- training prompt set: {prov['prompt_set_name']}
  (source {prov['source']['name']} rev {prov['source']['revision']},
   {prov['n_final']} prompts, deduped vs {prov['dedup']['eval_suite']})
- reward: {CFG['train']['reward']}, K={CFG['train']['group_size_K']},
  KL beta={CFG['train']['kl_beta']}, seed={CFG['train']['seed']}

## Result (the return edge of the loop)
- pre mean solve rate:  {delta['mean_pre']:.4f}
- post mean solve rate: {delta['mean_post']:.4f}
- **paired delta:       {delta['delta_mean']:+.4f}**, 95% CI {delta['delta_ci95']}
- permutation p:        {delta['perm_p']:.4g}
- Cohen's d_z:          {delta['cohens_dz']:+.3f}
- flips w->r / r->w:    {delta['flips_wrong_to_right']} / {delta['flips_right_to_wrong']}
- pass@k (pre/post):    see delta/passk_curve.png

## Reproduce
    uv run python run_loop.py   # from this archive, all inputs pinned above
"""
    (WORK / "capstone_report.md").write_text(report)
    print(report)


def archive_run() -> None:
    nas = Path(CFG["archive"]["nas_root"]) / f"capstone-{CFG['run_date']}"
    nas.mkdir(parents=True, exist_ok=True)
    manifest = {
        "machine": CFG["machine"],
        "driver": CFG["driver"],
        "run_date": CFG["run_date"],
        "artifacts": sorted(str(p.name) for p in WORK.rglob("*") if p.is_file()),
    }
    subprocess.run(["cp", "-r", str(WORK) + "/.", str(nas)], check=True)
    (nas / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"archived to {nas}")


if __name__ == "__main__":
    main()
```

## Lab: run the whole loop

One command, one sitting. The frozen suite, the trained prompt set, and the 2.6 serving config must already exist (they are the artifacts of chapters 3.9, 7.5, and 2.6); the capstone does not rebuild them, it consumes them.

```bash title="shell: run the capstone loop"
# fill loop.yaml's driver and run_date, confirm the referenced artifacts
# exist, then:
uv run python run_loop.py
```

### What you should see

The console walks the phases in order, printing each phase banner and the command it ran, and it stops hard the moment any phase fails to produce its artifact, so a broken run never silently corrupts the archive. When it finishes, three things exist that did not before. Under `capstone/work/` there is the full set of intermediate artifacts: the pre and post eval logs in the chapter 3.7 format, the trained adapter, the merged $M_1$ checkpoint, the chapter 7.6 delta report with its markdown, its JSON, and its pass@k figure, and a `.timing` file per phase stamping the wall-clock cost of each on the baseline machine. There is `capstone_report.md`, which is the artifact of this chapter and of the whole part: a single page that names every pinned input (the base model, the frozen suite and its revision, the training prompt set with its provenance and dedup, the reward and hyperparameters) and then states the result, the paired delta with its CI, its permutation $p$-value, its effect size, and the flip counts, as the return edge of the loop. And under the NAS archive root there is a dated directory holding a copy of all of it plus a manifest, so the entire run reproduces from the archive alone.

Every number in that report is stamped as measured on the baseline machine, and until you run the loop for real they read `record value, date, driver`, which is the book keeping its one promise to the end: a measured quantity comes stamped or it comes marked as not yet measured, and there is no third category. The moment those slots fill with real values from a real run, the thesis pipeline is no longer a design. It is a result you can hand a committee, and hand a later version of yourself, and both can reproduce it from one archived directory with one command.

```admonish thesis-thread
This closes the thread. The SDA verifiable-reasoning example has traveled the whole book: it was a task whose responses could be checked (the preface's promise), it became items in the frozen suite (chapter 3.9), its scorer became a reward (chapter 7.3), its difficulty band shaped the training prompt set (chapter 7.5), and its reasoning delta was measured with paired statistics (chapter 7.6). Here it rides the loop end-to-end and lands in the capstone report as a single paired delta on the SDA items, with a bootstrap CI and a permutation p-value, measured on the baseline machine — record value, date, driver, and hardened by the control runs of chapter 4.5 against the charge that any training at all would have moved it. That was the entire ambition of the thread: to take one concrete problem someone cared enough to write a thesis about and carry it from first eval to trained, re-evaluated model without ever losing the chain of provenance. The chain is now unbroken from the frozen item to the archived delta, and the loop that connects them runs on one 16GB card in one sitting. The general book stands on its own; the thread was the proof that the general machinery composes on a real problem, and this report is that proof made into a file on disk.
```

```admonish substack-seed
The satisfying post here is the one about the whole thing fitting in one sitting on one ordinary GPU. Everyone assumes the loop that turns "measure a model" into "train a model" into "measure it again and prove it improved" is a datacenter ritual, and the reveal is that it is a single script you can run before dinner on a 16GB desktop card, provided you respect one constraint the whole way through: serving and training never hold the card at once, so the loop is a sequence in time, not a swarm in parallel. Show the seven phases as a chain, point out that the last edge is the number the whole thing exists to produce, and end on the archive: the payoff is not the delta, it is that the delta arrives with its entire provenance attached, reproducible from one directory by one command, so that a year from now you or anyone can check whether it was real. The hook: the difference between a demo and a result is a manifest, and the manifest costs you almost nothing to keep.
```
