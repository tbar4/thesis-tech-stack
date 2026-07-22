# Storage tiers and cache discipline

**Goal.** Split working storage from archive storage on purpose, and make "which model" mean a revision hash rather than a name.

**Covers.** The NVMe working tier versus the NAS archive tier; `HF_HOME` and `hf_transfer`; a checkpoint retention policy; and what "reproducible" actually means for weights (pin revisions and commit hashes, not names).

## Theory

### Two tiers, two jobs

Storage on this machine has to do two jobs that pull in opposite directions, and the mistake is trying to make one device do both. The first job is **fast working access**: when I serve a model or resume a training run, the weights need to stream off disk quickly, and scratch files need to be written and deleted without ceremony. The second job is **durable retention**: the handful of things I actually want to keep (a final checkpoint, an acceptance report, a curated dataset) need to survive a cache purge, a reinstall, or a dead system drive. Fast working access wants the local NVMe. Durable retention wants somewhere I will not casually `rm -rf`. So I run two tiers.

The **working tier** is the 1 TB NVMe. It holds the Hugging Face cache, the currently active model weights, in-flight checkpoints, and scratch. Its defining property is that it is disposable: everything on it should be reconstructible from a revision hash plus a lockfile. If the NVMe died tomorrow I should lose time re-downloading, not lose results. That disposability is a discipline, not just a description: I never put the only copy of something I care about on the working tier.

The **archive tier** is the 5 TB NAS. It holds things whose loss would actually hurt: final checkpoints I have decided are worth keeping, acceptance reports, exported MLflow artifact stores, and curated datasets. It is reached over the network, so it is slower than local NVMe (the exact throughput is a machine-log entry; measured on the baseline machine; record value, date, driver), and that is fine, because I touch it deliberately and rarely. The NAS is where a result goes to become permanent.

```admonish vram-budget title="Disk is a budget too, just a slower one"
The 16 GB VRAM ceiling gets all the attention, but the 1 TB NVMe is also a budget, and it fills faster than you expect. A single 7B model in bf16 is ~14 GB on disk; keep a base model, a quantized copy, and a few checkpoints and you are into tens of gigabytes per experiment. The Hugging Face cache silently accumulates every revision you ever touched. Without a retention policy, the working tier fills, downloads start failing mid-stream, and you lose an afternoon to `du -sh` archaeology. The two-tier split is partly about durability and partly about keeping the fast disk lean enough to stay fast.
```

### What "reproducible" means for weights

Here is the idea that changes how you think about models: a model name is not an identity. "Qwen2.5-7B-Instruct" names a moving target. The repository behind that name can be updated: a tokenizer fix, a re-uploaded shard, a config tweak. If my thesis says "I evaluated Qwen2.5-7B-Instruct" and someone downloads that name six months later, they may get subtly different weights and get different numbers, and neither of us will know why. The name is a pointer, and pointers move.

The fix is to pin the **revision**: every Hugging Face model repository is a git repository, so every state of it has a commit hash. When I say "the model," I mean a repository at a specific commit, e.g. `Qwen/Qwen2.5-7B-Instruct` at revision `<40-hex-commit>`. That hash is immutable. Downloading that revision gives byte-identical files today, next year, and on someone else's machine. So the reproducibility rule for this book is: **pin revisions, not names**, everywhere a model or dataset is loaded, and record the pinned hash in the run log alongside the git SHA and the lock hash.

```admonish derivation title="Why a name can drift but a hash cannot"
A Hugging Face repo is git-backed, so its contents are content-addressed: a commit hash is a cryptographic digest of the exact tree of files at that commit. Change one byte of one shard and the commit hash changes. A branch or tag name (which is what a bare model name resolves to, usually `main`) is just a movable label pointing at whichever commit is latest. So `revision="main"` (the implicit default) means "whatever is latest when I happened to download it," while `revision="<hash>"` means "this exact tree, forever." Reproducibility is the difference between those two, and it costs exactly one extra argument at load time.
```

### The same discipline for datasets

Everything I just argued about model weights applies with equal force to datasets, and it is easy to forget because a dataset feels more static than a model. It is not. A Hugging Face dataset repository is git-backed exactly like a model repository, so "the GSM8K split I evaluated on" is only well-defined if I pin its revision too. A dataset owner can fix a label, add examples, or re-shuffle a split, and if my eval loaded the dataset by bare name it would silently pull the changed version on the next run, which would move my metric for a reason that has nothing to do with the model. So the pin-the-revision rule covers both nouns: every model load and every dataset load in this book carries an explicit `revision=<hash>`, and both hashes are recorded in the run log. The reasoning delta is "same model change, same eval data, measured twice," and "same eval data" is only true if the data is pinned as hard as the weights.

There is a subtler contamination angle here too, which Part 3 develops in full but which the storage layer enables. Pinning dataset revisions is also part of how I reason about test-set contamination: if I know the exact commit of the eval set I used, I can check whether that exact data could have leaked into a model's training corpus, and I can hold the eval data fixed while I vary the model. A dataset that drifts under its name makes contamination analysis impossible, because I can no longer say precisely which examples were in play. So pinning is not just reproducibility hygiene; it is a precondition for the contamination arguments the thesis will need to make.

### The cache is a shared resource, so control where it lives

Hugging Face libraries default to caching under `~/.cache/huggingface`, which on this machine would land on the system drive by default and mixes cached weights in with your home directory. I move the whole cache onto the working tier explicitly via `HF_HOME`, for three reasons. First, **placement**: the cache belongs on the fast disk I designated as the working tier, not wherever the default happens to fall. Second, **visibility**: one environment variable means both the serve and train environments share one cache, so I download a model once and both use it, and I always know where the big files are when the disk fills. Third, **portability of the decision**: `HF_HOME` is a single knob I set in a dotfile, so the storage layout is a documented choice rather than an accident of defaults.

`hf_transfer` is the other half of cache discipline. Model repos are large (tens of GB), and the default download path is not always the fastest. `hf_transfer` is a Rust-based accelerated download backend for the Hugging Face hub; enabling it (an env var plus the package) parallelizes and speeds up large downloads so the first pull of a model does not dominate my afternoon. The actual speedup is a machine-log entry; measured on the baseline machine; record value, date, driver.

### A retention policy, because checkpoints breed

Training produces checkpoints, and checkpoints breed faster than anything else on the working tier. A single training run might save the model state every few hundred steps, and each save of a 7B model is gigabytes. Left ungoverned, a week of experiments buries the NVMe under dozens of intermediate checkpoints, ninety percent of which I will never load again. So the storage policy needs a rule for what to keep, and the rule has to be simple enough that I actually follow it at 11pm when a run finishes.

My rule has three tiers of fate. Intermediate checkpoints live on the working tier only while their run is active, and exist so I can resume if the run crashes. When a run finishes, exactly two checkpoints earn keeping: the one that scored best on the held-out eval (the actual product of the run) and the final one (for continuing training later if I want). Everything else is deleted from the working tier the moment those two are safely copied to the NAS. This keeps the fast disk lean and pushes the small number of genuinely valuable artifacts onto durable storage, which is precisely the division of labor the two tiers exist for. The "best-by-eval" choice is deliberate: it ties the retention decision back to the loop's own measurement, so the checkpoint I keep is the one the evaluation actually preferred, not just the last one that happened to be written.

```admonish gotcha title="Never hand-`rm` a Hugging Face snapshot directory"
The Hugging Face cache stores each model as a tree of content-addressed blobs with per-revision snapshot directories full of symlinks into those blobs. If you `rm -rf` a snapshot directory to reclaim space, you can leave dangling references or orphan blobs that other revisions still point at, and in the worst case corrupt the cache so a later download half-fails in confusing ways. Reclaim space with `huggingface-cli delete-cache` (or `scan-cache` to see what is there first), which understands the blob-sharing and removes revisions safely. The working tier is disposable, but disposing of it wrong still costs you an afternoon.
```

## Tooling

The tools here are environment variables (`HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`), the `huggingface_hub` download API with a pinned `revision`, a NAS mount managed via `/etc/fstab`, and a written retention policy that I actually follow.

**`HF_HOME`** relocates the entire Hugging Face cache (models, datasets, tokens) to a directory I choose on the working tier. **`HF_HUB_ENABLE_HF_TRANSFER=1`** plus the `hf_transfer` package switches on the accelerated download backend. Both are set once in a dotfile so every shell and every `uv run` inherits them.

**The NAS mount** is a network filesystem (NFS or SMB) mounted at a stable path like `/mnt/nas`, declared in `/etc/fstab` so it comes back after a reboot. I mount it read-write for the archive workflow but treat it as append-mostly: things go there to be kept, not churned.

**Pinned downloads** use `huggingface_hub`'s `snapshot_download` (or `hf_hub_download`) with an explicit `revision=<hash>`. That single argument is the whole reproducibility story for weights.

```admonish under-the-hood title="What HF_HOME actually moves"
`HF_HOME` sets the root under which the hub libraries put everything: the `hub` cache of downloaded repos (each stored as a content-addressed blob tree with symlinked snapshots per revision), the datasets cache, and the stored auth token. Because snapshots are symlinked into a per-revision directory pointing at deduplicated blobs, keeping several revisions of the same model costs only the bytes that differ between them, not a full copy each. This is also why you should delete revisions through the cache tooling rather than by hand: naive `rm` on a snapshot can orphan or break the shared blob store.
```

## Lab

I set the env vars, mount the NAS, do a first pinned model download onto the working tier, and write the storage-policy doc. The artifacts are a storage-policy document plus the dotfile that encodes the layout.

### Step 1 - the dotfile that encodes the layout

```bash title="~/.config/thesis-loop/storage.env"
# Storage + cache discipline for the evals-as-rewards loop.
# Sourced from ~/.bashrc so every shell and every `uv run` inherits it.

# Working tier: the fast NVMe. All Hugging Face caches live here.
export HF_HOME="/data/hf"                 # NVMe working tier
export HF_HUB_ENABLE_HF_TRANSFER=1        # accelerated large-file downloads

# Archive tier: the NAS mount point (see /etc/fstab).
export NAS_ROOT="/mnt/nas"
export ARCHIVE_ROOT="${NAS_ROOT}/thesis-loop"   # where kept things go

# Fail loudly if the working tier is missing, rather than silently
# scattering a 14 GB download into the wrong place.
[ -d "$HF_HOME" ] || echo "WARN: HF_HOME $HF_HOME does not exist yet" >&2
```

```bash title="bash - wire the dotfile into the shell and create the working dir"
echo 'source ~/.config/thesis-loop/storage.env' >> ~/.bashrc

# /data is the NVMe working root. On a fresh machine it is owned by root, so I
# create it with sudo and then hand ownership to my user; otherwise the very
# first HF download fails writing into an unwritable /data.
sudo mkdir -p /data/hf                     # NVMe working tier
sudo chown -R "$USER:$USER" /data          # own the whole working root
source ~/.config/thesis-loop/storage.env
```

### Step 2 - mount the NAS archive tier

```bash title="bash - mount the NAS and make it persistent"
sudo apt install -y nfs-common    # NFS client; without it the fstab NFS mount fails
sudo mkdir -p /mnt/nas

# Example NFS entry; adjust host/export/protocol to your NAS.
# Appended to /etc/fstab so the mount survives reboot.
echo 'nas.local:/volume1/thesis  /mnt/nas  nfs  defaults,noatime,_netdev  0  0' | sudo tee -a /etc/fstab

sudo mount -a
mkdir -p /mnt/nas/thesis-loop/{checkpoints,reports,datasets,mlflow-exports}
df -h /mnt/nas    # confirm it mounted and shows the NAS capacity
```

### Step 3 - first pinned model download

This script downloads a model **by revision** onto the working tier, using `hf_transfer`, and records the exact hash it pinned. Run it from the `serve/` project so it uses that locked environment:

```python title="serve/scripts/fetch_model.py"
"""Download a model by pinned revision onto the working tier (HF_HOME).

Reproducibility rule: pin the commit hash, not the name. Record what we pinned.
Run with: uv run python scripts/fetch_model.py
"""
import os
import json
import datetime
from huggingface_hub import snapshot_download, HfApi

REPO = "Qwen/Qwen2.5-7B-Instruct"
# Pin an explicit commit. Resolve the current main -> hash once, then hard-code it.
# Replace with the exact 40-hex commit you intend to freeze on.
REVISION = "REPLACE_WITH_40_HEX_COMMIT"

def resolve_latest(repo: str) -> str:
    """Helper: print the current main commit so you can copy it into REVISION."""
    info = HfApi().repo_info(repo)
    return info.sha

if __name__ == "__main__":
    if REVISION.startswith("REPLACE"):
        print("Current main commit for", REPO, "->", resolve_latest(REPO))
        print("Copy that hash into REVISION and re-run to freeze the download.")
        raise SystemExit(0)

    path = snapshot_download(
        repo_id=REPO,
        revision=REVISION,               # the whole reproducibility story
        # lands under $HF_HOME/hub; hf_transfer accelerates it via the env var
    )

    record = {
        "repo": REPO,
        "revision": REVISION,
        "local_path": path,
        "hf_home": os.environ.get("HF_HOME"),
        "fetched_at": datetime.datetime.now().astimezone().isoformat(),
    }
    print(json.dumps(record, indent=2))
    # Append to a pins log so every frozen model is on the record.
    with open(os.path.expanduser("~/machine-log/model-pins.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
```

```bash title="bash - run the pinned fetch from the serve env"
cd serve
uv add hf_transfer                       # accelerated backend, into the locked env
# First run prints the current main commit; paste it into REVISION, then:
uv run python scripts/fetch_model.py
```

### Step 4 - the storage-policy document (artifact)

```markdown title="docs/storage-policy.md (the artifact)"
# Storage policy - evals-as-rewards loop

## Tiers
- **Working tier - 1 TB NVMe (`/data`, `HF_HOME=/data/hf`).**
  Fast, disposable. Holds the HF cache, active weights, in-flight checkpoints,
  scratch. Rule: nothing here is the only copy of anything I care about.
- **Archive tier - 5 TB NAS (`/mnt/nas/thesis-loop`).**
  Durable, slow, append-mostly. Holds kept checkpoints, acceptance reports,
  MLflow exports, curated datasets.

## Reproducibility rule for weights
- Always load models/datasets by **revision commit hash**, never by bare name.
- Every frozen model is appended to `~/machine-log/model-pins.jsonl`
  (repo, revision, path, HF_HOME, timestamp).

## Checkpoint retention policy
- **In-flight checkpoints** live on the working tier during a run.
- **Keep** only: the best-by-eval checkpoint of a run, plus the final one.
- **Promote** kept checkpoints to `${ARCHIVE_ROOT}/checkpoints/<run-id>/` on the NAS.
- **Delete** all other intermediate checkpoints from the working tier once the
  run's best/final are safely on the NAS.
- **Cache hygiene**: prune unused HF revisions with `huggingface-cli delete-cache`
  (never hand-`rm` snapshot dirs - it can orphan the shared blob store).

## Env (see ~/.config/thesis-loop/storage.env)
- `HF_HOME=/data/hf`
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- `NAS_ROOT=/mnt/nas`, `ARCHIVE_ROOT=/mnt/nas/thesis-loop`
```

**What you should see.** After sourcing the dotfile, `echo $HF_HOME` prints `/data/hf` and `df -h /mnt/nas` shows the NAS capacity, confirming both tiers exist. The first run of `fetch_model.py` prints the current `main` commit hash; once you paste that into `REVISION` and re-run, the model downloads into `/data/hf/hub` (visibly on the NVMe, not in your home directory), and a line lands in `~/machine-log/model-pins.jsonl` recording exactly which commit you froze. The download speed and the on-disk size are machine-log entries; measured on the baseline machine; record value, date, driver. The two artifacts, `docs/storage-policy.md` and `~/.config/thesis-loop/storage.env`, are committed. From here on, "the model" in this book always means a repo at a pinned commit sitting on the working tier, and anything worth keeping has a defined path on the NAS.

```admonish gotcha title="A bare model name in a script is a time bomb"
The most common reproducibility failure in this whole book is loading a model with just its name and letting it resolve to `main`. It works today and quietly returns different weights months later when the repo owner re-uploads a shard, and your carefully measured reasoning delta becomes uncomparable to your earlier runs for a reason you cannot see. Grep your scripts for model loads that lack a `revision=` argument before you trust any number. On this stack the rule is absolute: no pinned revision, no run.
```

```admonish thesis-thread
For the committee, the pinned-revision rule is what makes the thesis' central comparison legitimate. The reasoning delta is "same eval, before and after training." If the base model's weights can drift under its name between the before-run and a reviewer's replication, the comparison is not well-defined. Recording the commit hash in `model-pins.jsonl`, and logging it with every MLflow run (Chapter 0.6), pins the "before" model to an immutable object, so a reviewer downloading the same hash gets the same base model, and the delta they compute is the delta the thesis claims.
```

```admonish substack-seed
The reframe worth a post is "a model name is a pointer, and pointers move." People treat a Hugging Face model name like a permanent address, but it is git-backed, so the name usually resolves to whatever `main` is today, and `main` can change under you. The one-line fix - pass the commit hash - is the kind of small discipline that separates a result you can defend from an anecdote you happened to observe. Wrap it in the broader two-tier idea (fast disposable disk for work, slow durable disk for keeps) and you have a tidy piece on treating storage as a set of deliberate decisions instead of a pile of defaults.
```
