# Containers and the tracking spine

**Goal.** Stand up the tracking spine that every later chapter logs into, and fix the schema for what a run is.

**Covers.** Docker plus the nvidia-container-toolkit; an MLflow server on localhost; the run / experiment / artifact schema the whole book reuses; and the five things that get logged on every single run (git SHA, uv lock hash, driver version, VRAM peak, seeds).

## Theory

### Why a tracking spine at all

Without a tracking spine, an experiment-heavy project degenerates into a graveyard of terminal scrollback and half-remembered numbers. You run an eval, see 61.4%, tweak something, see 63.1%, and three weeks later you cannot answer the only question that matters: what exactly was different between those two runs? Which model revision, which code, which dependencies, which seed? A reasoning-delta thesis lives or dies on that question, because the entire claim is "this number moved because of this change, and nothing else." The tracking spine is the infrastructure that makes "nothing else" checkable rather than hopeful.

MLflow is the spine I use. It gives me three nouns that map cleanly onto how I actually work. An **experiment** is a named bucket for related runs (say, "grpo-gsm8k-qwen7b"). A **run** is a single execution with its parameters, metrics, and artifacts. An **artifact** is any file a run produces and wants to keep: a config, a plot, a checkpoint pointer, an eval transcript. Every number in Part 9's figures will be pulled from MLflow runs, so the discipline of logging into it from day one is what lets the book assemble itself from data instead of from memory.

```admonish under-the-hood title="What MLflow stores, and where"
MLflow separates two stores. The **tracking store** holds structured run metadata: parameters, metrics (time-series scalars), tags, and run/experiment relationships. The **artifact store** holds files. In this setup the tracking store is a backend database (SQLite for a single-user localhost server, upgradable to Postgres if it ever grows) and the artifact store is a directory on the working tier, periodically exported to the NAS archive tier. A run is identified by a `run_id`; metrics can be logged step-by-step (so a training curve is a real time series), and artifacts are content you can re-open later. Keeping the tracking DB and the artifact dir on the fast working tier keeps logging cheap; promoting the artifact store to the NAS is how a run becomes permanent.
```

### Docker, and why the GPU needs a toolkit to enter a container

A couple of services in this book (MLflow here, and later some serving and burst workflows) are cleaner to run as containers than as host installs, because a container pins their entire environment and keeps them from polluting the host. Docker provides that. But there is a specific wrinkle for GPU work: a container is isolated from the host's devices by default, so a process inside it cannot see the NVIDIA GPU unless something bridges the driver across the isolation boundary. That something is the **nvidia-container-toolkit**. It hooks into the container runtime so that `--gpus all` (or a Compose `deploy.resources` reservation) injects the host's GPU devices and the matching driver libraries into the container, letting CUDA code inside the container talk to the real card.

The MLflow server itself does not need the GPU (it is a metadata and file service), so the very first container I stand up is a clean test of Docker and Compose without the GPU complication. But I install and verify the nvidia-container-toolkit in the same chapter, because later chapters will run GPU workloads in containers, and the canonical proof that the toolkit works is running `nvidia-smi` inside a CUDA container and seeing the same card the host sees.

```admonish gotcha title="`--gpus all` without the toolkit = 'could not select device driver'"
If you try to run a GPU container before installing and configuring the nvidia-container-toolkit, Docker fails with an error like `could not select device driver "" with capabilities: [[gpu]]`. Installing the toolkit is not enough on its own; you must also register it with the Docker runtime (`nvidia-ctk runtime configure`) and restart the Docker daemon so it knows about the NVIDIA runtime. The verification step - `nvidia-smi` inside a container - is the only proof that matters; a green `apt install` is not.
```

### The five things every run logs, no exceptions

Here is the heart of the chapter, and it is a policy, not a tool. Every run in this book, whether it is a serve benchmark, an eval, or a training step, logs the same five provenance fields, always, so that any number can be reconstructed. If a run cannot produce these five, it does not get to contribute a number to a figure.

1. **Git SHA of the code.** The exact commit of the loop repository that produced the run, including a dirty flag if the working tree had uncommitted changes. This answers "which code."
2. **uv lock hash.** A hash of the relevant `uv.lock` (serve's or train's, per Chapter 0.4), so the run is tied to the exact resolved dependency graph. This answers "which environment."
3. **Driver version.** The NVIDIA driver version from `nvidia-smi` (Chapter 0.3), so a run is tied to the exact GPU software substrate. This answers "which driver."
4. **VRAM peak.** The peak GPU memory the run reached, so I can see how close to the 16 GB ceiling it ran and catch a regression that would OOM. This answers "did it fit, and how tightly."
5. **Seeds.** Every RNG seed the run set (Python, NumPy, Torch, and any sampler seed), so stochastic results are reproducible. This answers "was it luck."

Together with the pinned model revision from Chapter 0.5, these five turn a run from an anecdote into a reproducible object. The git SHA and lock hash pin the software; the driver pins the substrate; the VRAM peak pins the resource envelope; the seeds pin the randomness. A reviewer with those six facts (five plus the model revision) can, in principle, reproduce the run exactly. That is the whole point of the spine.

```admonish derivation title="Why these five and not others"
Think of a run's output as a function of inputs: $\text{result} = f(\text{code}, \text{deps}, \text{driver}, \text{model}, \text{data}, \text{seed}, \text{hardware})$. Hardware is fixed by the baseline machine and the preflight (Chapters 0.2–0.3). Model and data are pinned by revision (Chapter 0.5). That leaves code, deps, driver, and seed as the free inputs a run must record to be reproducible - which is fields 1, 2, 3, and 5. The VRAM peak (field 4) is not an input but a witness: it is the cheapest early-warning that a change pushed the run's resource envelope, which on a 16 GB card is the difference between "runs" and "OOMs." So four fields pin the inputs, and one field guards the constraint.
```

### Params, metrics, tags, artifacts: using the four slots correctly

MLflow gives a run four kinds of slot, and using them consistently is what makes the store queryable later instead of a junk drawer. **Parameters** are the inputs I set: the model revision, the learning rate, the batch size, the number of GRPO groups. They are logged once and never change during a run, and they are what I filter and group by in Part 9. **Metrics** are the outputs I measure, and they are time-series: a training loss logged every step, an eval accuracy logged at each checkpoint, the VRAM peak logged at the end. Because metrics carry a step index, a training curve is a real series I can plot, not a single final number. **Tags** are metadata about the run itself rather than its inputs or outputs, which is exactly where the five provenance fields live: git SHA, lock hash, driver, and seeds are tags, so they annotate the run without pretending to be experimental variables. **Artifacts** are files: configs, plots, eval transcripts, and pointers to the checkpoint on the NAS.

The discipline is to always put a thing in the same slot. A learning rate is always a param, never a tag; an eval score is always a metric, never a param; provenance is always a tag. When that convention holds across every run in the book, Part 9 can do things like "pull every run in experiment `grpo-gsm8k-qwen7b` that shares this git SHA and model revision, and plot their eval-accuracy metric against training step," and get a clean answer, because the fields it filters on are reliably in the slots it expects. A store where the learning rate is sometimes a param and sometimes buried in a run name is a store you cannot query, only browse.

```admonish gotcha title="Logging a metric where a param belongs (and vice versa) breaks queries"
MLflow lets you log almost anything as almost anything, which is a trap. If you log the learning rate as a metric (because it was handy) instead of a param, it will not show up where the query in Part 9 expects to group by it, and you will get a plot that silently drops runs or lumps them wrong. The failure is quiet: nothing errors, the numbers are just subtly incomplete. The fix is the convention above, enforced by putting all the slot decisions in one place (the `start_run` helper handles tags; each script logs its own params and metrics deliberately) rather than deciding ad hoc per script.
```

## Tooling

The tools are Docker + Docker Compose, the nvidia-container-toolkit, the MLflow server, and a small Python helper that stamps the five provenance fields onto every run.

**Docker Compose** describes the MLflow service declaratively in a `docker-compose.yml`, so standing up the spine is one command and its configuration is version-controlled. **The nvidia-container-toolkit** is installed on the host and registered with Docker so later GPU containers work; I verify it here even though MLflow itself does not use the GPU. **The MLflow server** runs on `localhost` with a SQLite backend store and a filesystem artifact store on the working tier. **The provenance helper** is a tiny module every run imports, which reads the git SHA, hashes the lockfile, reads the driver version, tracks the VRAM peak, and logs the seeds, so the five-field policy is enforced by code rather than by my memory.

```admonish read-along title="localhost, single user, on purpose"
The MLflow server here binds to localhost and uses SQLite. That is not me being lazy about a 'real' deployment; it is the right size for a single-user, single-machine research loop. No auth to manage, no network surface to secure, no Postgres to babysit. If the run count ever outgrows SQLite, the Compose file swaps the backend store URI for a Postgres service and nothing else in the book changes, because everything talks to MLflow through its client API and a tracking URI, not through the database directly.
```

## Lab

I install and verify the nvidia-container-toolkit, stand up MLflow via Docker Compose, write the provenance helper, and log a dummy run that carries all five fields. The artifacts are the `docker-compose.yml` and an MLflow-conventions document.

### Step 1 - install and verify the nvidia-container-toolkit

```bash title="scripts/00-container-gpu.sh"
#!/usr/bin/env bash
# Install Docker's NVIDIA container toolkit and prove the GPU crosses the container boundary.
set -euo pipefail

# (Docker Engine assumed installed via the official apt repo.)
# Install the toolkit, register it with the Docker runtime, restart the daemon.
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# The only proof that matters: run nvidia-smi INSIDE a container.
sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

That final `nvidia-smi` inside the container should print the same RTX 5080 the host sees. If it errors with a device-driver message, the toolkit is not registered with the runtime; re-run the `nvidia-ctk` step and restart Docker.

### Step 2 - the MLflow service

```yaml title="tracking/docker-compose.yml (the artifact)"
# MLflow tracking spine - localhost, single user.
# Tracking store: SQLite. Artifact store: a dir on the working tier.
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: thesis-mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow/store/mlflow.db
      --artifacts-destination /mlflow/artifacts
    ports:
      - "127.0.0.1:5000:5000"          # bind to localhost only
    volumes:
      - /data/mlflow/store:/mlflow/store          # SQLite DB on the NVMe working tier
      - /data/mlflow/artifacts:/mlflow/artifacts  # artifacts on the NVMe working tier
    restart: unless-stopped
```

```bash title="bash - start the spine and create the working dirs"
mkdir -p /data/mlflow/{store,artifacts}     # on the NVMe working tier
cd tracking
docker compose up -d
# UI now at http://127.0.0.1:5000
curl -s http://127.0.0.1:5000/health && echo "  <- MLflow healthy"
```

### Step 3 - the provenance helper

```python title="tracking/provenance.py"
"""Stamp the five mandatory provenance fields onto every MLflow run.

Every run in the book imports start_run() from here. No run contributes a
number to a figure unless it carries: git SHA, uv lock hash, driver version,
VRAM peak, and seeds.
"""
import os
import sys
import json
import hashlib
import random
import subprocess
import contextlib

import mlflow

MLFLOW_URI = "http://127.0.0.1:5000"


def _git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
    return sha + ("-dirty" if dirty else "")


def _lock_hash(lock_path: str) -> str:
    with open(lock_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def set_all_seeds(seed: int) -> dict:
    random.seed(seed)
    seeds = {"python": seed, "seed_arg": seed}
    try:
        import numpy as np
        np.random.seed(seed)
        seeds["numpy"] = seed
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeds["torch"] = seed
    except ImportError:
        pass
    return seeds


@contextlib.contextmanager
def start_run(experiment: str, lock_path: str, seed: int, run_name: str | None = None):
    """Open an MLflow run pre-stamped with the five mandatory provenance fields."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)
    seeds = set_all_seeds(seed)
    with mlflow.start_run(run_name=run_name) as run:
        # Fields 1,2,3,5 up front (VRAM peak, field 4, is logged at the end).
        mlflow.set_tags({
            "git_sha": _git_sha(),                 # 1
            "uv_lock_hash": _lock_hash(lock_path), # 2
            "driver_version": _driver_version(),   # 3
            "seeds": json.dumps(seeds),            # 5
        })
        try:
            yield run
        finally:
            # Field 4: VRAM peak, if torch/CUDA is present in this env.
            try:
                import torch
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    mlflow.log_metric("vram_peak_gb", round(peak, 3))
            except Exception:
                pass
```

### Step 4 - log a dummy run

```python title="tracking/smoke_run.py"
"""Log a dummy run to prove the spine and the five-field policy work end to end.

Run with: uv run python smoke_run.py
"""
import mlflow
from provenance import start_run

# lock_path points at whichever env's uv.lock this run belongs to.
LOCK = "../serve/uv.lock"

with start_run(experiment="smoke", lock_path=LOCK, seed=1234, run_name="hello-spine"):
    mlflow.log_param("dummy_param", "hello")
    for step in range(5):
        mlflow.log_metric("dummy_metric", 0.5 + 0.1 * step, step=step)
    with open("note.txt", "w") as f:
        f.write("first artifact on the spine\n")
    mlflow.log_artifact("note.txt")
    print("Logged dummy run. Open http://127.0.0.1:5000 to see it.")
```

```bash title="bash - run the smoke test from a project with mlflow installed"
cd tracking
uv run --with mlflow python smoke_run.py
```

### Step 5 - the conventions document (artifact)

```markdown title="docs/mlflow-conventions.md (the artifact)"
# MLflow conventions - evals-as-rewards loop

## Server
- Localhost only: tracking URI http://127.0.0.1:5000 (see tracking/docker-compose.yml).
- Backend store: SQLite on /data/mlflow/store (NVMe working tier).
- Artifact store: /data/mlflow/artifacts (NVMe), exported to NAS to make permanent.

## Nouns
- **Experiment**: named bucket of related runs, e.g. `grpo-gsm8k-qwen7b`.
- **Run**: one execution; carries params, metrics (time-series), tags, artifacts.
- **Artifact**: any file a run keeps (config, plot, transcript, checkpoint pointer).

## The five mandatory fields (enforced by tracking/provenance.py)
Every run logs, no exceptions:
1. `git_sha`      - code commit (+ `-dirty` if the tree was modified)
2. `uv_lock_hash` - sha256[:16] of the relevant uv.lock (serve or train)
3. `driver_version` - from nvidia-smi
4. `vram_peak_gb` - torch.cuda.max_memory_allocated at run end
5. `seeds`        - JSON of every RNG seed set (python/numpy/torch/sampler)
Plus the pinned model **revision** (Chapter 0.5) logged as a param on any run
that loads a model.

## Naming
- Experiment names: `<method>-<task>-<model>`, lowercase, hyphenated.
- Run names: short and human, e.g. `baseline`, `grpo-step-200`, `ablation-no-kl`.

## Retention
- Artifact store lives on the working tier; export to
  `${ARCHIVE_ROOT}/mlflow-exports/` on the NAS to make a run permanent.
```

**What you should see.** `nvidia-smi` inside the CUDA container prints the same RTX 5080 as the host, proving the toolkit bridges the GPU. `curl http://127.0.0.1:5000/health` returns healthy, and the MLflow UI loads in a browser. After the smoke run, the UI shows an experiment named `smoke` with one run named `hello-spine`, and that run carries four tags (`git_sha`, `uv_lock_hash`, `driver_version`, `seeds`), a `vram_peak_gb` metric (which will read near zero for this non-GPU dummy; real GPU runs will show a real peak, measured on the baseline machine; record value, date, driver), a five-point `dummy_metric` curve, and one artifact `note.txt`. The two artifacts, `tracking/docker-compose.yml` and `docs/mlflow-conventions.md`, are committed. From here on, every chapter that produces a number logs it through `start_run`, so the five fields are never optional and never forgotten.

```admonish thesis-thread
For the committee, this chapter is where reproducibility stops being an aspiration and becomes a mechanism. The claim "the reasoning delta is real and attributable" reduces to: the before-run and after-run differ in exactly the intended variable and agree on all five provenance fields except where the change requires otherwise. Because `provenance.py` stamps those fields automatically, Part 9 can filter the MLflow store to runs that share a git SHA, lock hash, driver, and model revision, and legitimately compare their metrics. The spine is what lets the thesis assert control over confounds instead of merely hoping for it.
```

```admonish substack-seed
The post here writes itself around a single embarrassing question: 'what was different between the run that got 61% and the run that got 63%?' Most people cannot answer it, and that inability is the quiet reason so much ML tinkering never compounds into knowledge. The fix is unglamorous and total: log five things on every run - the code commit, the locked dependencies, the driver, the peak memory, and the seeds - and enforce it in code so you cannot forget. Frame it as the difference between doing experiments and merely having experiences, and you have a piece that lands with anyone who has ever lost a good result to their own scrollback.
```
