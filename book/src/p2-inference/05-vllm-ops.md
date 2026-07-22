# vLLM operations

The internals chapter explained the mechanisms; this chapter turns them into commands I can run at 2am without thinking. The goal is a serving runbook for the whole repertoire (Qwen3-8B at FP8, Qwen3-14B at AWQ, gpt-oss-20b at MXFP4) where every flag has a reason traceable to the arithmetic of the last four chapters, plus the operational scaffolding that makes serving durable: health checks, a systemd unit so the server survives a logout or reboot, and a clean model-swap workflow because a 16GB card holds exactly one of these models at a time. The deliverable is that runbook, committed to the repo, and its value is that six months from now, mid-experiment, I can bring any model in the repertoire up and swap between them without re-deriving a thing or rediscovering the same three failure modes. A runbook is a promise to my future self that the boring parts are already solved.

## Theory

There is not much new theory here, which is the point: operations is where theory becomes muscle memory. The chapters before this one earned the right to be terse now: I already know why each flag exists and what it trades off, so the job is to assemble the pieces into commands and scaffolding that run without me thinking about the arithmetic each time. That assembly is itself a skill worth writing down, because the failure modes of operations are not intellectual (I understand the KV budget fine) but procedural (I forgot the old process was still holding the card). Three operational facts frame everything.

First, **one model at a time.** Qwen3-8B in full BF16 alone needs ~15 GiB of the 16 GiB card for weights (chapter 2's budget) — so much that it leaves no usable KV pool and must itself be served in FP8 — so there is certainly no holding two models resident. Every serving session is a single `vllm serve` process bound to port 8000, and switching models means stopping one and starting another. This is why the swap workflow is a first-class artifact, not an afterthought.

Second, **persistence and swap safety are the real deliverables.** A `vllm serve` typed into a terminal dies the moment the SSH session drops or the machine reboots, which is fine for a five-minute test and useless for a substrate that eval and training runs depend on for hours. Wrapping the serve in a supervised service that restarts on failure and survives logout, and making the model swap a scripted sequence that cannot race itself, is what separates a demo from infrastructure. Everything below builds toward those two.

Third, **the flags are the exposed controls on the mechanisms.** `--gpu-memory-utilization` sets the memory vLLM claims (chapter 2's $M_{kv}$); `--max-model-len` sets per-sequence context $S_{\text{ctx}}$; `--max-num-seqs` caps concurrency $N_{\text{seq}}$; `--kv-cache-dtype` sets the KV byte width $b$; `--quantization` selects the weight kernel; `--max-num-batched-tokens` sets the chunked-prefill chunk budget. Choosing them is just plugging the KV arithmetic back in.

```admonish read-along title="Confirm the flags against your vLLM version before trusting a script"
Every serve command in this chapter is coupled to a vLLM version. Run `uv run vllm serve --help | less` on the machine and confirm the exact spelling and defaults of the six flags above before committing a runbook, because they drift: quantization backends get renamed, KV-dtype values change, and MXFP4 auto-detection has moved between releases. Then run `uv run vllm --version` and write that string at the top of the runbook. A serve command with no version stamp is not reproducible, and reproducibility is the entire reason this book exists.
```

## Tooling: the canonical serve commands, every flag justified

### Qwen3-8B at FP8

Full BF16 weights (~15.3 GiB) do not fit on this 16GB card with any usable KV pool (chapter 2's budget goes negative), so the reference model is served with FP8 *weights* via `--quantization fp8`. That halves the weight footprint to ~7.7 GiB and opens a real KV pool; FP8 KV on top keeps the per-token cost low. The served name stays `qwen3-8b` so downstream configs do not care that the weights are quantized.

```bash title="serve/qwen3-8b-fp8.sh"
#!/usr/bin/env bash
set -euo pipefail
uv run vllm serve Qwen/Qwen3-8B \
    --quantization fp8 \
    --max-model-len 8192 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --port 8000 \
    --served-model-name qwen3-8b
```

- `--quantization fp8`: serve the 8B with FP8 weights, because full BF16 (~15.3 GiB) leaves no KV pool on 16GB (chapter 2). FP8 halves the weight read, roughly doubling the decode ceiling as a bonus. Lowering `--gpu-memory-utilization` is *not* an alternative here: it gives less memory, not more, and BF16 cannot load at all.
- `--max-model-len 8192`: keeps the worst-case per-sequence KV reservation modest (8K, not the model's 32K native). With ~6 GiB of KV pool now available, raise it if a task needs the context and accept fewer concurrent seqs.
- `--kv-cache-dtype fp8`: halves KV bytes/token (chapter 2), stretching concurrency and context further. The accuracy cost is a fraction of a point; measure it.
- `--gpu-memory-utilization 0.92`: standard headroom now that FP8 weights leave a real pool; no need to push to 0.95.
- `--max-num-seqs 16`: the FP8-weight pool (~6 GiB) supports real concurrency at short contexts, unlike the BF16 case that could not load.
- `--enable-prefix-caching`: the eval workload shares long instruction scaffolds, so this is a free prefill saving.
- `--served-model-name qwen3-8b`: a stable client-facing name so the swap workflow does not break downstream configs.

### Qwen3-14B at AWQ (4-bit)

The workhorse. Quantized weights (~7.75 GiB) leave a comfortable ~6 GiB KV pool, so this config is generous: more concurrency, more context, native-precision KV for cleaner attention.

```bash title="serve/qwen3-14b-awq.sh"
#!/usr/bin/env bash
set -euo pipefail
uv run vllm serve Qwen/Qwen3-14B-AWQ \
    --quantization awq_marlin \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --port 8000 \
    --served-model-name qwen3-14b
```

- `--quantization awq_marlin`: selects the Marlin INT4 kernel that dequantizes in-register (chapter 3's under-the-hood), so decode genuinely reads 4-bit weights and the ceiling roughly doubles versus BF16.
- `--max-model-len 16384`: affordable here because the pool is ~6 GiB; at 16K, KV is 2.5 GiB/seq (chapter 2 table), so a couple of long sequences or many short ones fit.
- `--gpu-memory-utilization 0.92`: standard headroom; the weights are small enough that I do not need to push to 0.95.
- `--max-num-seqs 16`: the pool supports real concurrency now; 16 is a round cap that the KV budget can back at short contexts.
- `--max-num-batched-tokens 4096`: the chunked-prefill chunk size (chapter 4). 4K keeps a long prompt from freezing other streams while still being a big enough chunk for good prefill utilization.
- KV left at default BF16: the pool is comfortable, so I spend precision on attention rather than squeezing KV; switch to `fp8` only if I want to push concurrency further.

### gpt-oss-20b at MXFP4

Ships pre-quantized in MXFP4, an MoE with only ~3.6B active params, so it decodes fast (chapter 1's ceiling) and its weights (~12-13 GiB) still leave a workable pool. Its alternating sliding-window attention means its long-context KV grows more slowly than the naive formula suggests (chapter 2's gotcha).

```bash title="serve/gpt-oss-20b.sh"
#!/usr/bin/env bash
set -euo pipefail
uv run vllm serve openai/gpt-oss-20b \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 12 \
    --enable-prefix-caching \
    --port 8000 \
    --served-model-name gpt-oss-20b
```

- No explicit `--quantization`: the checkpoint is natively MXFP4 and vLLM detects it; on Blackwell the native FP4 path serves it directly.
- `--max-model-len 16384`: the sliding-window layers cap their own KV, so context is cheaper than the dense formula implies; 16K is comfortable and can go higher, verified against the boot-time block count rather than a hand estimate.
- `--max-num-seqs 12`: a moderate cap; the MoE's fast decode means even 12 concurrent streams stay responsive.
- `--gpu-memory-utilization 0.92`: standard; ~12-13 GiB of weights plus a few GiB pool fits with headroom.

```admonish gotcha
Flag names and defaults drift across vLLM releases (`awq` vs `awq_marlin`, KV-dtype spellings, MXFP4 auto-detection). Stamp the vLLM version in the runbook and re-verify the flags after any upgrade. A serve command is not a fact of nature; it is version-coupled.
```

### Health checks

A server that returns 200 on `/health` is not necessarily generating tokens correctly; a real health check does a tiny completion.

```bash title="serve/healthcheck.sh"
#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8000}"
# 1. liveness
curl -fsS "$BASE/health" >/dev/null && echo "health: OK"
# 2. model is loaded and named as expected
curl -fsS "$BASE/v1/models" | python3 -c 'import sys,json; \
    print("models:", [m["id"] for m in json.load(sys.stdin)["data"]])'
# 3. it actually generates
curl -fsS "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${2:-qwen3-14b}"'","prompt":"2+2=","max_tokens":3,"temperature":0}' \
    | python3 -c 'import sys,json; \
    print("gen:", json.load(sys.stdin)["choices"][0]["text"].strip())'
echo "healthcheck passed"
```

### A systemd unit for persistent serving

To keep a model serving across logout and reboot, wrap it in a user systemd service. This survives SSH disconnects and restarts on failure.

```ini title="serve/vllm-serve.service"
[Unit]
Description=vLLM inference server (repertoire)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/trevor/thesis-tech-stack/serve
Environment=HF_HOME=/home/trevor/.cache/huggingface
# The instance name selects the script: vllm@qwen3-14b-awq runs qwen3-14b-awq.sh.
ExecStart=/usr/bin/env bash /home/trevor/thesis-tech-stack/serve/%i.sh
Restart=on-failure
RestartSec=5
TimeoutStartSec=300

[Install]
WantedBy=default.target
```

Templated by model so I can start whichever I want:

```bash title="shell"
mkdir -p ~/.config/systemd/user
cp serve/vllm-serve.service ~/.config/systemd/user/vllm@.service
systemctl --user daemon-reload
# start the workhorse:
systemctl --user start vllm@qwen3-14b-awq
systemctl --user enable vllm@qwen3-14b-awq   # survive reboot
journalctl --user -u vllm@qwen3-14b-awq -f   # follow logs
```

### The model-swap workflow

Because only one model fits, swapping is: stop the current server, wait for VRAM to actually free, start the target, wait for health. Doing this by hand invites the classic failure where the new server OOMs because the old one has not released memory yet. Script it.

```bash title="serve/swap.sh"
#!/usr/bin/env bash
# Swap the resident model: ./swap.sh <target-unit-instance> <served-name>
set -euo pipefail
TARGET="${1:?usage: swap.sh <qwen3-8b-fp8|qwen3-14b-awq|gpt-oss-20b> <served-name>}"
SERVED="${2:?served model name for healthcheck}"

echo "stopping any running vllm@ unit..."
systemctl --user stop 'vllm@*' 2>/dev/null || true

echo "waiting for VRAM to free..."
for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "  VRAM used: ${USED} MiB"
    [ "$USED" -lt 1500 ] && break   # weights released; only driver overhead left
    sleep 2
done

echo "starting vllm@${TARGET}..."
systemctl --user start "vllm@${TARGET}"

echo "waiting for health..."
for i in $(seq 1 60); do
    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
        ./healthcheck.sh http://localhost:8000 "$SERVED"
        echo "swap to ${TARGET} complete."
        exit 0
    fi
    sleep 3
done
echo "ERROR: ${TARGET} did not become healthy in time" >&2
exit 1
```

The VRAM-drain wait loop is the load-bearing part: it polls `nvidia-smi` until the old weights are actually gone (used memory back under ~1.5 GiB of driver overhead) before launching the next server, which is exactly the race that bites people who just `stop; start`. Process teardown is not instantaneous: the Python process has to release its CUDA context, and until it does the memory it held is still counted as used, so a new server launched too eagerly sees a full card and OOMs at graph-capture time. The loop turns a flaky manual dance into a deterministic sequence I can put in a cron job or a training harness without babysitting it.

### Observability: what to watch while a model serves

A server that booted healthy can still degrade under load, and the window into that is the Prometheus `/metrics` endpoint (chapter 4). The three gauges worth a persistent eye are the running/waiting/swapped sequence counts, the KV cache usage fraction, and the preemption counter. (The swapped count is a V0 metric; on the V1 engine preemption is by recompute, so it stays 0 and the preemption counter is the one to watch.) A quick one-liner keeps them on screen:

```bash title="serve/watch.sh"
#!/usr/bin/env bash
# Poll the vLLM /metrics endpoint for the gauges that predict trouble.
BASE="${1:-http://localhost:8000}"
while true; do
    curl -fsS "$BASE/metrics" 2>/dev/null | grep -E \
      'vllm:num_requests_(running|waiting|swapped)|vllm:gpu_cache_usage_perc|vllm:num_preemptions' \
      | grep -v '#'
    echo "---"; sleep 5
done
```

The reading is diagnostic, not decorative. KV cache usage creeping toward 100% with a rising waiting count means the pool is saturated and I am about to start preempting; that is the signal to lower `--max-num-seqs` or shorten `--max-model-len` for this workload, both of which trace straight back to the chapter-2 hyperbola. A nonzero and climbing preemption counter means I am already thrashing and paying to swap or recompute cache, which is strictly worse than admitting fewer sequences in the first place. Watching these while the benchmark of the next chapter runs is how I learn the real operating envelope of each model on this specific card, as opposed to the envelope the arithmetic predicted, and the two should agree to within the overhead fudge factor.

### The three failure modes at boot, and their fixes

Almost every serving failure on this card is one of three things, and recognizing them by their symptom saves hours.

- **OOM during graph capture.** The server loads weights fine, then dies while capturing CUDA graphs or allocating the KV pool. Cause: `--gpu-memory-utilization` is too high for the real activation and graph overhead on the day. Fix: lower utilization a notch, shorten `--max-model-len`, or add `--kv-cache-dtype fp8` to shrink the pool's per-token cost. This is distinct from a model that cannot fit at all: Qwen3-8B in full BF16 exceeds the card no matter the utilization (lowering it gives *less* memory), which is why the reference is served with FP8 weights rather than tuned to fit.
- **OOM right after a swap.** The new server dies immediately even though its config fit before. Cause: the previous process had not released its CUDA context, so the card was still full. Fix: this is exactly what the swap script's VRAM-drain loop prevents; if it still happens, raise the drain threshold or the retry count.
- **Quantization backend mismatch.** The server refuses to load an AWQ or MXFP4 checkpoint. Cause: a `--quantization` value that this vLLM version spells differently, or a checkpoint whose format the installed kernels do not support. Fix: check `vllm serve --help` for the accepted values (the read-along), and confirm the Blackwell FP4/Marlin paths are present in this build.

None of these are mysterious once you have seen them, and the runbook's acceptance table exists precisely to catch them on the machine before they surprise me mid-experiment.

## Lab: stand up each model and script the swap

The artifact is `serve/RUNBOOK.md`, a markdown runbook committed to the repo that ties the scripts together and records the acceptance evidence.

### Set up

```bash title="shell"
cd ~/thesis-tech-stack
mkdir -p serve && cd serve
uv add "vllm>=0.6" requests   # into the serving env
chmod +x *.sh
```

### Exercise each model and the swap

```bash title="shell"
# bring up the workhorse and prove it
systemctl --user start vllm@qwen3-14b-awq
./healthcheck.sh http://localhost:8000 qwen3-14b

# swap to the reference model
./swap.sh qwen3-8b-fp8 qwen3-8b

# swap to the MoE
./swap.sh gpt-oss-20b gpt-oss-20b
```

### Write the runbook

```markdown title="serve/RUNBOOK.md"
# vLLM Serving Runbook: MSI Aegis R2 (RTX 5080 16GB)

Stamp: vLLM VERSION, NVIDIA 570-open, DATE. One model resident at a time (16GB).

## Models and their configs
| served name | checkpoint | precision | max ctx | max seqs | KV dtype | notes |
|---|---|---|---|---|---|---|
| qwen3-8b   | Qwen/Qwen3-8B      | FP8    | 8192  | 16 | fp8  | FP8 weights ~7.7 GiB (BF16 ~15.3 GiB won't fit) |
| qwen3-14b  | Qwen/Qwen3-14B-AWQ | INT4-g | 16384 | 16 | bf16 | workhorse; ~6 GiB KV pool |
| gpt-oss-20b| openai/gpt-oss-20b | MXFP4  | 16384 | 12 | bf16 | MoE, ~3.6B active, fast decode |

## Start a model
    systemctl --user start vllm@<unit>     # qwen3-8b-fp8 | qwen3-14b-awq | gpt-oss-20b
    ./healthcheck.sh http://localhost:8000 <served-name>

## Swap models (only one fits)
    ./swap.sh <target-unit> <served-name>
    # stops current, waits for VRAM drain (<1.5 GiB), starts target, waits for health

## Health
    ./healthcheck.sh                       # liveness + model list + 1 generation
    curl -s localhost:8000/metrics | grep -E 'running|waiting|kv_cache'

## Persistence
    systemctl --user enable vllm@<unit>    # start on boot
    journalctl --user -u vllm@<unit> -f    # logs

## Acceptance evidence (fill on the machine)
- qwen3-14b boot KV blocks: RECORD ; implied max seqs @16k: RECORD
- qwen3-8b (FP8) boot KV blocks: RECORD ; KV pool GiB at util=0.92: RECORD
- gpt-oss   boot KV blocks: RECORD
- swap round-trip time (14b -> 8b -> 20b): RECORD seconds
- date / driver: RECORD
```

### What you should see

Each `systemctl --user start` should bring a server to a healthy state within a couple of minutes (the first launch is slower because of weight download and CUDA-graph capture, both of which are one-time costs cached for subsequent boots), and `healthcheck.sh` should print the model list, a tiny correct generation, and "healthcheck passed." The generation step matters more than it looks: a server can pass a liveness probe while quietly serving a broken chat template or a mis-quantized checkpoint, and only an actual token round-trip catches that, which is why the health check ends by asking the model to compute two plus two rather than trusting a 200 response. The swap script is where the real proof is: after `swap.sh`, `nvidia-smi` should show VRAM dropping back to driver overhead before the next server claims it, and the whole round trip should take on the order of tens of seconds plus the target's startup time, with no OOM. Qwen3-8B is the one to watch: full BF16 weights (~15.3 GiB) simply do not fit on 16GB with any usable KV pool, so the reference is served with FP8 weights (`--quantization fp8`), which brings the weights to ~7.7 GiB and opens a real pool. If it still OOMs at boot the cause is graph-capture overhead, not the weights, and the fix is a slightly lower `--gpu-memory-utilization` or a shorter `--max-model-len` — never a hoped-for BF16 fit. The committed `serve/RUNBOOK.md`, with its acceptance table filled in from the real machine (block counts, swap timing, date, driver), is the artifact the rest of the thesis leans on whenever it needs a model up. It is deliberately boring, and boring is the whole objective of an operations chapter. The exciting version of serving infrastructure is the one that surprises you at midnight; the boring version is the one you can run half-asleep because every sharp edge was filed down in daylight and written into a script. I am aiming squarely for boring.

```admonish substack-seed
"One GPU, three models, zero drama." The unglamorous truth of single-GPU serving is that the hard part is not making a model fast, it is making the swap between models reliable. On a 16GB card you can hold exactly one of your models at a time, so your real infrastructure is the thirty-second choreography of stopping one server, waiting for the VRAM to actually drain (the race that OOMs everyone who skips it), and bringing up the next. This post is a love letter to operational boredom: a systemd unit, a health check that generates a token instead of trusting a 200, and a swap script whose most important line is a polling loop on nvidia-smi. Every serving flag gets exactly one sentence of justification traceable to arithmetic, and the reward is a runbook you can follow half-asleep.
```
