# Appendix C: Troubleshooting bestiary

A symptom-to-cause-to-fix catalogue for this exact stack: Blackwell RTX 5080 on
Ubuntu 24.04 with the 570-open driver, uv-managed serve and train environments,
vLLM for inference, and Unsloth for training. Each entry is a short field note, not
an essay. This appendix accretes: every genuinely new failure I hit while running
the loop gets a new entry with the date I hit it and the driver I hit it under, so
that future-me searching an error string lands on the fix and not a rediscovery.
The nastiest ones (the failures that cost me an evening) carry a `gotcha`.

Entries are grouped by where in the loop they bite: driver and boot, serving,
quantized loading, training, evaluation, and storage.

## Driver and boot

### nvidia-smi shows "No devices were found" or a driver/library mismatch

**Symptom.** `nvidia-smi` prints `No devices were found`, or
`Failed to initialize NVML: Driver/library version mismatch`.

**Cause.** Either the kernel module is not loaded (common after a kernel update
rebuilt the module against a new kernel but the old one is still in memory), or the
userspace libraries and the loaded kernel module are on different versions (a
partial upgrade). On Blackwell the module must be the **open** flavor.

**Fix.**
```bash
cat /proc/driver/nvidia/version          # what module is actually loaded
dpkg -l | grep -i nvidia                 # what packages are installed
nvidia-smi                               # confirm after each step
sudo modprobe nvidia                     # try loading the module
# mismatch after an upgrade -> reboot so userspace and kernel module realign
sudo reboot
```
If the module refuses to load, confirm you are on `nvidia-driver-570-open` (the
open-kernel-module branch), not the proprietary `-570`, which does not support this
card cleanly. See *Ubuntu 24.04 on Blackwell*.

```admonish gotcha
A `Driver/library version mismatch` after `apt upgrade` almost always means the DKMS
module rebuilt but the running kernel still holds the old one. You cannot fix it
live; a reboot is the fix, not `modprobe`. Do not chase it with reinstalls first.
```

### Secure Boot blocks the NVIDIA module

**Symptom.** Driver installs cleanly but `nvidia-smi` finds no device; `dmesg`
shows `Loading of unsigned module is rejected` or `module verification failed`.

**Cause.** Secure Boot refuses to load the unsigned out-of-tree NVIDIA kernel
module.

**Fix.** Either enroll a Machine Owner Key (MOK) so the module can be signed and
trusted, or disable Secure Boot in firmware.
```bash
mokutil --sb-state                       # is Secure Boot enabled?
sudo update-secureboot-policy --enroll-key   # enroll DKMS MOK, set a password
sudo reboot                              # complete MOK enrollment at the blue screen
```
On reboot, the MOK Manager (blue screen) prompts you to enroll the key with the
password you set. Miss it and the module stays blocked. The firmware route is
covered in *Firmware first: BIOS and the Windows send-off*.

## Serving (vLLM)

### CUDA out of memory at server startup

**Symptom.** `vllm serve` dies ~30 s after boot with
`torch.OutOfMemoryError: CUDA out of memory` or `No available memory for the cache
blocks`.

**Cause.** Weights + activation overhead + requested KV pool exceeds what
`--gpu-memory-utilization` claimed, or the model simply does not fit (Qwen3-8B BF16
at 15.3 GiB is at the edge). This is the *KV cache arithmetic* budget failing.

**Fix (levers, bluntest first).**
```bash
--max-model-len 4096          # smaller S_ctx -> smaller worst-case KV reservation
--kv-cache-dtype fp8          # halve per-token KV bytes
--gpu-memory-utilization 0.95 # claim more of the 16 GiB (careful: starves overhead)
--max-num-seqs 4              # cap concurrency
--quantization awq_marlin     # or switch to the 4-bit 14B to free weight bytes
```
If BF16 8B will not fit with any KV pool, that is expected: move to FP8 KV or the
AWQ 14B. Re-derive $M_{kv}$ from *Appendix A* before guessing.

### vLLM: KV cache won't fit / "not enough KV cache blocks"

**Symptom.** Boot log reports very few or zero GPU KV cache blocks; or requests
with long prompts are rejected with a context-length error.

**Cause.** After weights and overhead there is almost nothing left for KV, so the
pool cannot hold even one sequence at the requested `--max-model-len`.

**Fix.** Shrink the promised context or the weight footprint. The block count vLLM
prints × 16 tokens is your total cached-token budget; if that is smaller than one
`--max-model-len`, no sequence fits. Lower `--max-model-len`, switch to FP8 KV, or
quantize the weights. Reconcile against the hand calculation in *KV cache
arithmetic* (`predict.py`).

```admonish gotcha
gpt-oss-20b's sliding-window layers make its real KV need smaller than the naive
formula predicts, so trust the server's reported block count over the arithmetic
for that model. For the dense Qwen3 models the arithmetic and the block count
should agree within allocator slack; a wild disagreement means you read $H_{kv}$ or
$L$ from the wrong config revision.
```

### AWQ or MXFP4 model fails to load

**Symptom.** `ValueError: Unknown quantization method`, a Marlin kernel error, or
`unsupported dtype` on Blackwell.

**Cause.** Wrong `--quantization` value, a kernel that is not built for
Blackwell/sm_120 in the pinned vLLM, or a checkpoint whose packing does not match
the requested method.

**Fix.**
```bash
# AWQ: prefer the marlin kernel; fall back to plain awq if marlin errors
--quantization awq_marlin        # or: awq
# GPTQ checkpoints:
--quantization gptq_marlin
# gpt-oss ships MXFP4; let vLLM detect it, or name it explicitly
--quantization mxfp4
uv pip show vllm                 # confirm a Blackwell-capable version is pinned
```
If a Marlin kernel throws on sm_120, upgrade vLLM within the lock
(`uv lock --upgrade-package vllm`) to a build with Blackwell kernels, then
`uv sync`. Match the `--quantization` flag to how the checkpoint was actually
quantized; AWQ weights will not load as GPTQ.

## Training (Unsloth / QLoRA / GRPO)

### CUDA out of memory during training

**Symptom.** OOM at the first backward, or partway into GRPO generation.

**Cause.** Activations, generation KV, or optimizer state overran the budget. From
*Appendix A*, the training budget is base + adapters + optimizer + activations +
(for GRPO) generation KV.

**Fix (levers, mapped to budget terms).**
```python
# activations:
gradient_checkpointing = "unsloth"     # recompute activations in backward
per_device_train_batch_size = 1
gradient_accumulation_steps = 8        # same effective batch, less peak memory
max_seq_length = 2048                  # shorter sequences -> fewer activations
# GRPO generation KV (see GRPO on 16GB):
num_generations = 4                    # group size G; linear in generation KV
max_completion_length = 768            # S_gen; linear in generation KV
# optimizer:
optim = "adamw_8bit"                   # 8-bit Adam moments; smaller than FP32 m,v
```
Order of impact for GRPO OOM: group size $G$, then generation length, then batch,
then checkpointing. See the GRPO lever list in *Appendix A*.

### Unsloth version-pin conflicts

**Symptom.** `uv add unsloth` fails to resolve, or import errors like
`cannot import name ... from transformers`, or a torch/xformers/triton ABI mismatch
at import.

**Cause.** Unsloth pins tight, sometimes pre-release, versions of torch,
transformers, trl, xformers, and triton; these collide with whatever vLLM wants,
which is exactly why serve and train are separate uv projects.

**Fix.**
```bash
# keep train/ isolated from serve/ (the two-environment doctrine)
cd train
uv add "unsloth" "unsloth-zoo" "trl" "transformers"   # let unsloth drive the pins
uv lock                                                # inspect the resolution
uv tree | grep -E "torch|transformers|trl|xformers|triton"
```
If resolution fails, pin the versions Unsloth's release notes call for explicitly,
lock, and `uv sync --frozen`. Never install Unsloth into the serve env; that is the
conflict you are trying to avoid.

```admonish gotcha
The most confusing Unsloth failures are silent import-time ABI mismatches (torch
built against one CUDA, triton expecting another) that look like bugs in your
training code. When a fresh `uv sync` of the train env suddenly breaks after an
upstream release, suspect the pins before your script. Freeze a known-good
`train/uv.lock` and only bump it deliberately.
```

## Evaluation

### Tokenizer or chat-template mismatch

**Symptom.** Eval scores are inexplicably low; model outputs look truncated,
double-wrapped in special tokens, or ignore the instruction; or the served model id
does not match what the eval requests (404).

**Cause.** The prompt was not wrapped in the model's chat template, or was wrapped
twice, or a different tokenizer/template than training was used. Reasoning models
are especially sensitive to their `<think>` scaffolding.

**Fix.**
```bash
# Inspect: confirm the served model id matches
curl http://localhost:8000/v1/models
# lm_eval: apply the chat template explicitly when comparing to chat-tuned numbers
--apply_chat_template --fewshot_as_multiturn
```
Verify the template by dumping one fully-rendered prompt (Inspect's log viewer or
lm-eval's `--log_samples`) and eyeballing that the special tokens appear exactly
once. A model served without `--served-model-name` advertises its repo path; the
eval must request that exact string. See *lm-evaluation-harness and comparability*
and *Judge models*.

### Non-determinism across eval runs

**Symptom.** The same eval on the same model gives different scores run to run.

**Cause.** Sampling temperature above 0, batching-order effects in the server,
floating-point non-associativity across different batch shapes, and genuine
Monte-Carlo variance in the eval itself. Some of this is irreducible; the fix is to
separate real variance from noise, not to pretend it is zero.

**Fix.**
```bash
--temperature 0.0        # greedy decoding removes sampling noise
--seed 0                 # pin the sampler seed where the tool exposes one
```
Even at temperature 0, exact scores can wobble by a sample or two because vLLM's
continuous batching changes reduction order. Do not chase the last decimal; instead
report a bootstrap confidence interval (from *The statistics of evals*) so a real
delta is distinguishable from run-to-run jitter. If you need bitwise repeatability,
fix batch size and disable chunked prefill, and accept the throughput cost.

## Storage

### NAS mount drops mid-run

**Symptom.** A training or eval job stalls or dies with `Input/output error`,
`Stale file handle`, or `Transport endpoint is not connected`, and `ls /mnt/nas`
hangs.

**Cause.** The network mount (NFS/SMB to the 5TB NAS) dropped, and a process
writing checkpoints or reading a dataset across it blocked or errored. Nothing in
the hot path should live on the NAS, but a checkpoint-archive step or a dataset
left there will take the run down with it.

**Fix.**
```bash
mount | grep nas                      # is it actually mounted?
sudo umount -l /mnt/nas               # lazy unmount a wedged handle
sudo mount -a                         # remount from /etc/fstab
```
Prevention is the real fix: keep the working set (active weights, dataset,
in-flight checkpoints) on the 1TB NVMe and only `rsync` to the NAS after a run
completes, per *Storage tiers and cache discipline*. Mount the NAS with `soft` and
a timeout so a drop returns an error instead of hanging forever, and write
checkpoints locally, then sync.

```admonish gotcha
A `hard` NFS mount will block a process *forever* on a dropped link, and that
process is often unkillable (`D` state), which can force a reboot and cost you the
run. Use `soft,timeo=,retrans=` for the archive mount so a drop fails fast, and
never point a checkpoint-writer directly at the NAS.
```

## Data pipeline and grounding

The failures in this group come from the parts of the loop that never touch the GPU:
the 3.9 SDA data pipeline (Airflow 3, space-track), the 3.10 SGP4 oracle, and the
8.1 RAG index. They bite just as hard as a CUDA OOM, and for the same reason, they
are silent until a number is wrong.

### Single-node Airflow 3 will not start, or a first unpause stampedes

**Symptom.** The scheduler or API server container crash-loops; DAGs never appear in
the UI; or the first `unpause` of a DAG immediately fires a run for every logical date
since `start_date`.

**Cause.** Airflow 3 is built for clusters and its defaults assume one. On a single
box the usual causes are: the one-time `airflow db migrate` (and admin-user create)
step was never run, so there is no metadata schema; an executor that wants
infrastructure you do not have (Celery/Kubernetes want a broker and workers); the
`data/` project is not mounted into the scheduler, so the `uv run` task modules are
not on the execution path; or `catchup` was left on and the DAG backfilled its whole
history on first unpause.

**Fix.**
```bash
cd data
docker compose -f docker-compose.airflow.yml up airflow-init   # ONCE: db migrate + admin
docker compose -f docker-compose.airflow.yml up -d
docker compose -f docker-compose.airflow.yml logs -f airflow-scheduler   # read the parse error
```
Use `LocalExecutor` (tasks run as subprocesses on the one box), keep Airflow's own
Postgres for metadata entirely separate from the data tiers, mount `..:/opt/project`
so the task modules and their `uv.lock` are visible, and set `catchup=False` per DAG
unless backfill is genuinely meaningful. Reproduce a broken task with
`uv run airflow tasks test <dag> <task> <date>`, which runs one task with no scheduler.

```admonish gotcha
The single most useful move when a DAG task fails is to ignore Airflow entirely and
run the underlying module by hand: `cd data && uv run python -m
sda_data.tasks.celestrak_tle --group active`. Because the DAGs are thin (they only
shell out to these modules), a plain-terminal traceback is the real error, not the
Airflow task-instance log four clicks deep. If the module runs clean but the DAG task
fails, the problem is the mount or the environment, not your code.
```

### space-track auth fails, or you get throttled / locked out

**Symptom.** `spacetrack` login raises an auth error; a whole logical date's task
fails; or repeated pulls start returning 429s and then the account is temporarily
blocked.

**Cause.** space-track is a session login (a cookie), not an API key, so a missing or
wrong credential fails the login outright. And it enforces strict per-minute and
per-hour rate limits with throttling and temporary blocks for abuse, so a naive loop
that fetches one object per request, or an unattended pipeline that re-fetches every
sample, will trip them.

**Fix.**
```bash
# credentials live in the environment, NEVER in the repo (the 3.9 boundary)
export SPACETRACK_IDENTITY=...   SPACETRACK_PASSWORD=...
```
Use the `spacetrack` library so you do not hand-roll the login and session, and let
its built-in throttling run (do not disable it). Batch queries (ask for many objects
in one request) instead of looping one object per request, cache by NORAD id plus
epoch, and back off on 429. For anything that repeats (an eval in a table), pin a 3.9
snapshot and read that, rather than re-fetching live.

```admonish gotcha
The redistribution line is a code invariant, not a promise: space-track output must
land only on the git-ignored, non-DVC `data/live/` tier, never on the shippable raw
tier. If you ever find a space-track file staged for commit, that is a bug, and the
Chapter 10.3 embargo check exists to fail the build when it happens. Never widen a
`git add` or `dvc add` glob to "just grab the data folder".
```

### SGP4 oracle: wrong frame, wrong units, or a stale-epoch propagation

**Symptom.** A conjunction miss distance is off by thousands of km; a correct-looking
model answer is graded wrong by exactly 1000x; or gold answers drift the further the
screening window sits from the TLE epoch.

**Cause.** Three things the `sgp4`/Skyfield libraries will not enforce for you (from
3.10). **Frame:** `sgp4` returns position and velocity in **TEME** (True Equator,
Mean Equinox), not ECEF/ECI-of-date and not lat/lon. Differencing two TEME vectors is
fine (both objects share the frame), but subtracting a TEME vector from a ground
site's lat/lon silently puts a conjunction off by thousands of km. **Units:**
everything the library returns is kilometers and km/s; if a model answers in meters
and the verifier does not fold the unit, a correct answer reads 1000x wrong. **Epoch:**
SGP4 accuracy degrades away from the TLE's epoch, so a window days from epoch is
propagating a stale fit; and the classic "minutes since epoch" convention is a trap if
you do not feed absolute Julian dates.

**Fix.**
```python
from sgp4.api import jday
jd, fr = jday(2026, 7, 22, 0, 0, 0)   # absolute JD, sidesteps "minutes since epoch"
# ground-site geometry: go through Skyfield's topocentric conversion, never
# subtract a TEME vector from a lat/lon (sat - site).at(t).altaz()
# units: fold the model's answer to km before comparing (m -> km) in the verifier
```
Keep the screening window near the snapshot's element epochs and record the epoch age,
because a task built on a week-old TLE has a gold answer with real error bars.
Km-versus-m, TEME-versus-ECEF, and epoch age are the three checks to run before
trusting any generated gold.

```admonish gotcha
Mixing TEME and ECEF is the silent one: no exception, no NaN, just a physically
impossible miss distance that looks plausible until you check it against a known
pass. When an oracle number smells wrong, verify the frame before you touch anything
else. The verifier and the gold generator share the same oracle code path on purpose
(3.10), so a frame or unit bug corrupts the answer key and the grade identically and
will not show up as a disagreement between them.
```

### LanceDB / embedding-model OOM from co-residency on 16GB

**Symptom.** The embedding vLLM server OOMs at boot when the generation server is
already up; or the generation server fails to allocate its KV pool after the embedding
server claimed the card; or `--gpu-memory-utilization` fractions that "should" fit
still race and one server dies.

**Cause.** Two vLLM servers on one 16GB card, each told to claim a fraction, whose
fractions sum too close to 1 (or a greedy embedder / a fat generation KV pool) so they
race for blocks (from 8.1). The embedding model itself is cheap (it grows no KV cache,
equation 1.6), but if you serve a 0.6B decoder-style embedder at a large
`--max-model-len`, or leave the generation KV pool wide, the sum overruns the ~15.5
GiB usable capacity.

**Fix.**
```bash
# co-resident split (8.1): fractions sum under 1, generation gets the lion's share
uv run vllm serve Qwen/Qwen3-8B --quantization fp8 \
  --gpu-memory-utilization 0.75              # ~11.6 GiB weights + KV
uv run vllm serve BAAI/bge-small-en-v1.5 --task embed --port 8001 \
  --gpu-memory-utilization 0.15              # ~0.07 GiB weights + activation + context
```
The honest fix when they still race is to shrink the generation KV pool or the
embedder, never to hope for a fit. Better still, do not hold both resident:
**time-share** the card (8.1 default). Serve the embedding model alone, embed the whole
corpus and every eval query in one offline pass, build the LanceDB index, tear it
down, then serve only the generation model reading the precomputed query vectors from
disk. Peak VRAM is then just whichever single model is up (see *Appendix A*).

```admonish gotcha
For a *fixed* eval suite the queries are known in advance, so the embedding model
never has to be resident at eval time at all: precompute every query vector in the
offline index-build pass and the eval loop reads them from disk. Co-residency
(equation 1.7 in *Appendix A*) is only worth the OOM risk when retrieval must be
*live* (8.2 / 8.3); for a batch eval, time-share and the whole problem disappears.
```

## How this appendix grows

New failures get appended under the group they belong to, each in the same
symptom-to-cause-to-fix shape, stamped with the date and driver they were hit
under. The value of this bestiary is entirely in it being *my* failures on *this*
machine, so the temptation to pad it with generic StackOverflow answers is the
thing to resist. If a fix required a real measurement (a memory figure, a
throughput cliff), record it as `(measured on the baseline machine — record value,
date, driver)` like everywhere else in the book.
