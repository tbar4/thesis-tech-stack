# Appendix B: CLI cheatsheets

One-screen references for the seven command-line tools the loop runs on. These are
not tutorials; each tool earns a full treatment in its chapter (uv in *uv and the
two-environment doctrine*, vLLM in *vLLM operations*, Inspect in *Inspect I* and
*Inspect II*, lm-eval in *lm-evaluation-harness and comparability*, MLflow in
*Containers and the tracking spine*). This is the copy-pasteable distillation you
reach for once you already know what each command does. Every command assumes the
baseline machine and the `uv`-everywhere doctrine: no global `pip`, no bare
`python`, every invocation prefixed with `uv run` so it resolves against the
project's locked environment.

## uv

The toolchain: Python version pinning, virtualenvs, lockfiles, and a runner. The
two-environment doctrine keeps `serve/` and `train/` as separate uv projects with
separate committed `uv.lock` files, because vLLM and Unsloth pin conflicting
versions of torch and friends.

```bash title="project lifecycle"
uv init serve                     # new project -> pyproject.toml + .python-version
cd serve
uv python pin 3.12                # write .python-version; uv fetches the interpreter
uv add "vllm>=0.6"                # add a dep, resolve, update uv.lock, sync .venv
uv add --dev pytest ruff          # dev-only deps
uv remove requests                # drop a dep
uv lock                           # re-resolve and rewrite uv.lock (no install)
uv lock --upgrade-package vllm    # bump one package within constraints
uv sync                           # make .venv exactly match uv.lock (CI-safe)
uv sync --frozen                  # sync but fail if the lock is stale (reproducible)
```

```bash title="running things"
uv run python train.py            # run inside the project env, no activation needed
uv run vllm serve ...             # any console script the deps installed
uv run --with matplotlib plot.py  # one-off extra dep, not added to the project
uvx ruff check .                  # run a tool in an ephemeral env (== uv tool run)
uv tree                           # show the resolved dependency graph
uv pip list                       # inspect the current env
```

```admonish gotcha
Commit `uv.lock`. `uv sync --frozen` in CI and on the burst box is the whole
reproducibility promise: it fails loudly if the lock and `pyproject.toml` have
drifted instead of silently resolving something new. Keep `serve/uv.lock` and
`train/uv.lock` as two independent files; never share one env between them.
```

## vllm serve

The OpenAI-compatible server. The repertoire commands below are the three configs
from the book, each annotated with the flags that matter on 16 GiB. Ports default
to 8000; the KV and memory flags are derived in *KV cache arithmetic*.

```bash title="Qwen3-8B, BF16 (edge of the card)"
uv run vllm serve Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 8192 \            # per-seq context ceiling S_ctx
  --gpu-memory-utilization 0.92 \   # fraction of 16 GiB vLLM may claim
  --kv-cache-dtype fp8 \            # halve B_tok to claw back KV pool
  --max-num-seqs 8                  # hard cap on concurrency N_seq
```

```bash title="Qwen3-14B, AWQ 4-bit (the workhorse)"
uv run vllm serve Qwen/Qwen3-14B-AWQ \
  --quantization awq_marlin \       # Marlin kernel; faster than plain awq
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16
```

```bash title="gpt-oss-20b, MXFP4 (MoE)"
uv run vllm serve openai/gpt-oss-20b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8
```

```bash title="flags you will actually tune"
--max-model-len N          # lowers worst-case KV reservation; admit more seqs
--max-num-seqs N           # min(this, what KV pool allows) sets concurrency
--gpu-memory-utilization F # sets M_kv; too high starves activations -> OOM
--kv-cache-dtype fp8       # halves per-token KV bytes
--quantization awq_marlin  # awq | awq_marlin | gptq_marlin | fp8 | mxfp4
--enable-chunked-prefill   # interleave prefill with decode; smooths latency
--served-model-name NAME   # the model id clients pass (decouple from repo path)
--api-key SECRET           # require a bearer token
```

```bash title="talk to it"
curl http://localhost:8000/v1/models          # is it up? what model id?
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}]}'
```

At boot, watch the log line reporting **GPU KV cache blocks**: blocks × block-size
(16 tok default) = total cached tokens = your real $N_{\text{seq}} \times
S_{\text{ctx}}$ budget. That is the number to reconcile against the hand
calculation.

## inspect (Inspect AI)

The evals framework: tasks are datasets + solvers + scorers. Point it at the local
vLLM server through its OpenAI-compatible endpoint.

```bash title="running evals"
export INSPECT_EVAL_MODEL=openai/Qwen/Qwen3-8B
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

uv run inspect eval tasks.py                    # run every task in the file
uv run inspect eval tasks.py@my_task            # one task by name
uv run inspect eval tasks.py \
  --model openai/Qwen/Qwen3-8B \                # override model
  --model-base-url http://localhost:8000/v1 \
  --limit 100 \                                 # first 100 samples (smoke test)
  --temperature 0.0 \                           # deterministic-ish decoding
  --max-connections 16 \                        # concurrent requests to the server
  --log-dir logs/                               # where .eval logs land
```

```bash title="viewing logs"
uv run inspect view --log-dir logs/             # local web viewer for .eval logs
uv run inspect log dump logs/2026-....eval      # dump one log as JSON
uv run inspect log list --log-dir logs/         # enumerate runs
```

```admonish gotcha
Inspect talks to vLLM as `openai/<model-id>`, where `<model-id>` must match what
`vllm serve` advertises (the repo path unless you set `--served-model-name`).
A mismatch is a 404 from the server, not an Inspect error, so check
`/v1/models` first. `--max-connections` should sit at or below the server's
`--max-num-seqs`, or requests queue and your wall-clock inflates.
```

## lm_eval (lm-evaluation-harness)

The comparability harness: standardized tasks (MMLU, GSM8K, ...) with fixed
few-shot prompts, used when a number needs to line up with published leaderboards.
Use `local-completions` to hit the local vLLM server.

```bash title="against the local vLLM server"
uv run lm_eval \
  --model local-completions \
  --model_args model=Qwen/Qwen3-8B,base_url=http://localhost:8000/v1/completions,num_concurrent=16,tokenized_requests=False \
  --tasks gsm8k,mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path results/ \
  --log_samples                     # persist per-sample outputs for auditing
```

```bash title="common knobs"
--tasks LIST            # comma-separated; `lm_eval --tasks list` to enumerate
--num_fewshot K         # shots in the prompt; MUST match the number you compare to
--limit N               # subsample for a smoke test
--apply_chat_template   # wrap prompts in the model's chat template
--fewshot_as_multiturn  # few-shot examples as prior turns, not one blob
--gen_kwargs temperature=0,max_gen_toks=512
```

```admonish gotcha
The single biggest source of "my MMLU doesn't match the paper" is `--num_fewshot`
and whether a chat template was applied. Comparability means fixing both to the
reference recipe; the harness will happily give you a different, internally
consistent number otherwise. See *lm-evaluation-harness and comparability*.
```

## mlflow

The tracking spine: experiment metadata, params, metrics, and artifacts, from
*Containers and the tracking spine*. Run the server against a local backing store
so every eval and training run is logged.

```bash title="server + ui"
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \       # run metadata
  --artifacts-destination ./mlartifacts \         # logged files
  --host 127.0.0.1 --port 5000
# UI is served at http://127.0.0.1:5000 by the same process
```

```python title="logging from a run"
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("grpo-4b")

with mlflow.start_run(run_name="r16-g8"):
    mlflow.log_params({"rank": 16, "group_size": 8, "lr": 1e-6})
    mlflow.log_metric("reward_mean", 0.42, step=100)
    mlflow.log_metric("pass_at_1", 0.31, step=100)
    mlflow.log_artifact("roofline.png")             # any file
    mlflow.set_tag("driver", "570-open")            # stamp the baseline
```

```bash title="cli"
uv run mlflow runs list --experiment-id 1
uv run mlflow artifacts download --run-id <id> --dst-path ./pulled
```

## nvidia-smi / nvtop

The GPU dashboard. `nvidia-smi` is the ground truth for memory and driver;
`nvtop` is the live top-like view.

```bash title="nvidia-smi"
nvidia-smi                                  # snapshot: mem, util, procs, driver
nvidia-smi -l 1                             # refresh every 1 s
watch -n 1 nvidia-smi                       # same, via watch
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm --format=csv -l 1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # who holds VRAM
nvidia-smi -q -d CLOCK                      # current vs max SM/mem clocks (throttling?)
nvidia-smi --gpu-reset                      # last resort after a wedged process (root)
```

```bash title="nvtop"
uv run --with nvitop nvitop    # nvitop is the pip-installable one (note the i)
sudo apt install nvtop; nvtop  # nvtop itself is an apt package, not on PyPI
```

```admonish gotcha
`nvidia-smi` reports memory the *driver* has handed out, which reads higher than
your summed parameter bytes because of the CUDA context (~few hundred MiB) and the
caching allocator's block rounding. When reconciling a VRAM budget, expect
`nvidia-smi` > arithmetic by that slack, not equal to it. If it shows "No devices
were found", that is the driver, not your code; see *Appendix C*.
```

## rsync

The burst-sync workhorse: move checkpoints and run artifacts between the NVMe
working tier, the 5TB NAS archive, and the Lambda burst box (from *Storage tiers
and cache discipline* and *The Lambda workflow*).

```bash title="the patterns"
# archive a finished run to the NAS (mirror, delete extras, preserve times)
rsync -avh --delete ./runs/grpo-4b/ /mnt/nas/archive/grpo-4b/

# pull weights up to the burst box before a job (resume-safe, show progress)
rsync -avhP ./models/ user@burst:/workspace/models/

# bring results back, only what changed, dry-run first
rsync -avhn user@burst:/workspace/runs/ ./runs/     # -n = dry run, preview only
rsync -avh  user@burst:/workspace/runs/ ./runs/     # then for real

# checkpoint sync mid-run without clobbering newer files
rsync -avh --update ./checkpoints/ /mnt/nas/checkpoints/
```

```bash title="flags decoded"
-a   archive: recurse + preserve perms, times, symlinks
-v   verbose;  -h human-readable sizes
-P   --partial --progress: resume interrupted transfers, show a bar
--delete   make dest an exact mirror of source (dangerous: deletes extras)
--update   skip files that are newer on the receiver
-n   dry run: print what would transfer, change nothing
-z   compress in transit (worth it over the network to burst; skip on LAN/NAS)
--exclude='*.tmp'   skip scratch files
```

```admonish gotcha
A trailing slash on the source (`./runs/`) copies the *contents*; no trailing slash
(`./runs`) copies the *directory itself* into the dest. This one character is the
difference between `dest/file` and `dest/runs/file`. Always `-n` a `--delete` first;
it is the command that eats archives.
```
