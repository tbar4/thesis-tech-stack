# Acceptance: proving the machine

**Goal.** Prove the machine is healthy with a scripted, repeatable acceptance suite before trusting any number it produces.

**Covers.** What a healthy card looks like at idle; memory-bandwidth and GEMM sanity tests; thermals under sustained load; and first tokens from PyTorch. The suite ends in an acceptance report archived to the NAS.

## Theory

### Why acceptance is a gate, not a vibe

Everything downstream in this book is a measurement, and a measurement is only as trustworthy as the instrument that produced it. The preflight in Chapter 0.2 proved the hardware was sound under the vendor's OS; the install in Chapter 0.3 proved the driver and link were correct. Acceptance is the third gate: it proves that the fully assembled Linux stack, driver plus CUDA plus PyTorch, produces sane, stable, in-spec numbers before I let it produce research numbers. It is deliberately a gate and not a feeling. Either the card hits a plausible fraction of its rated bandwidth, sustains a GEMM without errors, holds temperature under load, and emits correct tokens, or it does not, and if it does not, I fix the machine before I trust a single reasoning-delta figure.

The reason to script it, rather than eyeball `nvidia-smi` once and move on, is that acceptance is something I re-run. After a driver update, after a kernel bump, after moving the machine, after any "huh, that's slower than I remember" moment, I re-run the same suite and diff the report against the last known-good one. A scripted suite turns "the machine feels fine" into "the machine matches acceptance report v1 within tolerance," which is a claim I can actually check.

### What a healthy card looks like at idle

Before any load, a healthy RTX 5080 at idle has a recognizable signature in `nvidia-smi`: low power draw, low temperature, near-zero memory used, near-zero utilization, and the correct static facts (16 GB total memory, the 570-branch driver, PCIe Gen5 x16). The idle numbers are a baseline, not a benchmark; their job is to catch gross problems like a card stuck at high power draw, a runaway process already holding VRAM, or a link that renegotiated down since install. All the specific idle values are machine-log entries; measured on the baseline machine; record value, date, driver.

### Bandwidth and GEMM: the two numbers that predict everything

Two microbenchmarks predict almost all of the performance behavior this book cares about, and they map onto the two regimes I will formalize with the roofline model in Part 2.

The first is **memory bandwidth**. The RTX 5080 has roughly 960 GB/s of memory bandwidth (per NVIDIA's RTX 5080 product specification). A bandwidth test streams large buffers through the memory system and reports the achieved rate. It will never hit the theoretical peak (no real workload does), but it should reach a healthy fraction of it. This matters because the token-generation phase of LLM inference is memory-bound: each token's latency is dominated by reading the model weights and KV cache out of VRAM, so achieved bandwidth is a direct predictor of decode throughput. If the bandwidth test comes back far below the rated figure, every tokens-per-second number later will be quietly disappointing, and I want to know that is a hardware/driver issue, not a vLLM configuration issue.

The second is **GEMM throughput**. A general matrix multiply (GEMM) is the compute primitive at the heart of every transformer layer, and a large GEMM in the card's native low precision (bf16/fp16, and on Blackwell the newer low-precision tensor-core paths) exercises the compute-bound regime: prompt prefill and training are dominated by these matmuls. A GEMM sanity test does two jobs at once. It checks **correctness** (the result matches a reference within floating-point tolerance, which catches a miscompiled kernel or unstable memory), and it estimates **throughput** in FLOP/s, which I compare against the card's rated compute to confirm the tensor cores are actually engaged and not silently falling back to a slow path.

```admonish derivation title="Why bandwidth predicts decode and GEMM predicts prefill"
A matmul of an $m\times k$ by $k\times n$ matrix does about $2mnk$ floating-point operations and moves about $2(mk + kn + mn)$ bytes (at 2 bytes/element). Its arithmetic intensity is roughly $\frac{2mnk}{2(mk+kn+mn)}$ FLOP/byte. When one dimension is tiny - decode processes one token at a time, so $n = 1$ - the intensity collapses and the operation is bandwidth-bound: you spend all your time reading weights, not multiplying. When all dimensions are large - prefill and training process many tokens at once - the intensity is high and the operation is compute-bound: the tensor cores are the bottleneck. So the bandwidth test is a proxy for the decode regime and the GEMM test is a proxy for the prefill/training regime, which is exactly the roofline split Part 2 develops in full. Acceptance measures both ends so I know both regimes are healthy before I care which one dominates a given workload.
```

### Thermals: sustained load is a different test than peak load

A card can pass a two-second GEMM and still fail in practice, because a two-second burst does not heat the silicon the way a twenty-minute training run does. Thermal acceptance is about **sustained** load: I run the GPU flat out long enough for temperatures to reach steady state, and I watch two things. Does the temperature plateau below the throttling threshold, and do the clocks hold, or do they sag as the card thermal-throttles to protect itself? A machine that throttles under sustained load will produce results that silently degrade the longer a run goes, which is poison for a training loop that runs for an hour. All temperatures, clocks, and throttle observations here are machine-log entries; measured on the baseline machine; record value, date, driver.

### First tokens: the whole stack, end to end

The final acceptance check is the most holistic: load a small model in PyTorch and generate a few tokens. This exercises the entire software stack, PyTorch, CUDA, the driver, the GPU, and the tokenizer, in the same shape the real workloads use, and confirms it produces coherent output. It is not a benchmark; it is a smoke test that says "the thing that will run the loop can, in fact, turn a prompt into tokens on this GPU." If bandwidth, GEMM, and thermals all pass but first-tokens fails, the problem is in the Python/CUDA stack, not the hardware, and I have narrowed it before I even start.

### Tolerances, and what "healthy fraction" actually means

A gate needs a threshold, and a threshold needs a number, so it is worth being honest about where the acceptance tolerances come from. I do not have the card's internal design targets, so I anchor the tolerances to two things: the published spec (roughly 960 GB/s of memory bandwidth, and the card's rated low-precision compute from NVIDIA's RTX 5080 specification) and the first clean run of the suite itself, which becomes the reference. A microbenchmark never reaches theoretical peak, because the peak assumes perfect overlap and no overhead, so I expect the bandwidth test to land at some fraction of 960 GB/s and the GEMM test to land at some fraction of rated compute. The exact fractions I treat as healthy are recorded from the first good run; measured on the baseline machine; record value, date, driver. The point of writing them down is that "healthy fraction" stops being a vibe and becomes a stored number I can diff against.

Once that reference exists, the tolerance is relative, not absolute, which is the more useful form. I do not actually care whether the bandwidth test hits a specific GB/s figure to three digits; I care whether a later run drops meaningfully below the reference the machine established when it was known-good. A ten-or-fifteen-percent drop from the reference bandwidth is a red flag worth chasing (a thermal issue, a link renegotiation, a driver regression), even if the absolute number still looks superficially fine. So acceptance v1 is not just a pass/fail certificate, it is the baseline against which every future acceptance run is a relative comparison. That is why the report records the actual measured numbers and not just checkmarks: the numbers are the reference the machine will be held to for the rest of its working life.

```admonish under-the-hood title="Why a microbenchmark undershoots the spec, and that is fine"
The rated 960 GB/s assumes the memory bus is saturated with a perfectly-scheduled access pattern and no other traffic. A real copy kernel pays for launch overhead, imperfect overlap of reads and writes, and the fact that you are timing a finite loop rather than an infinite steady state. So landing below the rated figure is expected and not a fault; a healthy card reaches a solid fraction of peak, not peak itself. The same logic applies to the GEMM: the rated TFLOP/s assumes the tensor cores never stall, which a real kernel with finite tiles and memory traffic cannot quite achieve. The signal that matters is not "did it hit the spec" but "did it hit the fraction of spec this card hit when it was healthy," which is exactly what acceptance v1 records.
```

## Tooling

The suite is a bash driver that calls small, focused checks: `nvidia-smi` for idle and for load monitoring, a CUDA bandwidth measurement, a PyTorch GEMM correctness-and-throughput check, a sustained-load thermal watch, and a PyTorch generate check. Everything runs from the `train/` uv environment (it has PyTorch), and the whole run logs into MLflow through the Chapter 0.6 provenance helper so the acceptance report itself carries the five mandatory fields. The report is written to disk and archived to the NAS.

```admonish read-along title="Acceptance is itself a run"
It would be a little ironic to prove the machine with a suite that does not follow the book's own logging rules. So acceptance logs through `provenance.start_run`: the report is tagged with the git SHA, the `train/uv.lock` hash, the driver version, the VRAM peak the suite reached, and the seed. That means acceptance report v1 is not just a text file, it is an MLflow run I can diff future acceptance runs against, on exactly the provenance fields that would explain a regression.
```

## Lab

The suite is one bash script that orchestrates a Python check module, prints a human-readable report, and archives it to the NAS. The artifact is acceptance report v1 on the NAS. Every measured value in the report is a placeholder to be filled on the machine.

### The Python checks

```python title="acceptance/checks.py"
"""GPU acceptance checks: GEMM correctness+throughput, bandwidth, first tokens.

Run individual checks via the CLI, e.g.:  uv run python checks.py gemm
"""
import sys
import time
import json

import torch


def idle_summary() -> dict:
    """Static facts the report should confirm."""
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "total_mem_gb": round(p.total_memory / (1024**3), 2),
        "capability": f"{p.major}.{p.minor}",   # Blackwell reports sm_120
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def gemm_check(n: int = 8192, iters: int = 50, dtype=torch.bfloat16) -> dict:
    """Large bf16 GEMM: correctness vs fp32 reference + throughput estimate."""
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    # Correctness on a small tile against an fp32 reference.
    ref = (a[:512, :512].float() @ b[:512, :512].float())
    got = (a[:512, :512] @ b[:512, :512]).float()
    max_rel_err = ((got - ref).abs() / (ref.abs() + 1e-3)).max().item()

    # Throughput: 2*n^3 FLOP per matmul.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = (2 * n**3) / dt / 1e12

    return {
        "n": n, "dtype": str(dtype),
        "max_rel_err": max_rel_err,
        "sec_per_gemm": dt,
        "tflops_est": round(tflops, 1),   # compare to rated compute
        "correct": max_rel_err < 5e-2,
    }


def bandwidth_check(bytes_gb: float = 2.0, iters: int = 50) -> dict:
    """Streaming copy bandwidth: read+write a large buffer, report GB/s."""
    n = int(bytes_gb * (1024**3) / 4)     # fp32 elements
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)                        # reads n*4 bytes, writes n*4 bytes
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    moved_gb = (2 * n * 4) / (1024**3)    # read + write
    gbps = moved_gb / dt * (1024**3) / 1e9
    return {"gb_per_s_est": round(gbps, 1), "sec_per_iter": dt}   # compare to ~960 GB/s rated


def first_tokens(model_id: str, revision: str, n_new: int = 20) -> dict:
    """Whole-stack smoke test: load a small model by pinned revision, generate."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=n_new, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return {"prompt": prompt, "completion": text, "generated_ok": len(text) > len(prompt)}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    result = {}
    if which in ("idle", "all"):
        result["idle"] = idle_summary()
    if which in ("gemm", "all"):
        result["gemm"] = gemm_check()
    if which in ("bandwidth", "all"):
        result["bandwidth"] = bandwidth_check()
    if which in ("first_tokens", "all"):
        # Use a small model pinned by revision; replace the hash on the machine.
        result["first_tokens"] = first_tokens(
            "Qwen/Qwen2.5-0.5B-Instruct", revision="REPLACE_WITH_40_HEX_COMMIT"
        )
    print(json.dumps(result, indent=2))
```

### The thermal watch

```python title="acceptance/thermal_watch.py"
"""Sustained-load thermal check: hammer the GPU, sample temp/clock/power, report.

Run: uv run python thermal_watch.py 300   # seconds of sustained load
"""
import sys
import time
import subprocess

import torch


def sample() -> dict:
    q = "temperature.gpu,clocks.sm,power.draw,utilization.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]
    ).decode().strip()
    t, clk, pw, util = [s.strip() for s in out.split(",")]
    return {"temp_c": float(t), "sm_mhz": float(clk), "power_w": float(pw), "util_pct": float(util)}


def main(seconds: int):
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        for _ in range(50):
            c = a @ b                       # keep the tensor cores busy
        torch.cuda.synchronize()
        samples.append(sample())
        time.sleep(1)
    peak_t = max(s["temp_c"] for s in samples)
    min_clk = min(s["sm_mhz"] for s in samples)
    max_clk = max(s["sm_mhz"] for s in samples)
    # Throttling shows up as clocks sagging from their early peak under sustained heat.
    print({"peak_temp_c": peak_t, "clock_min_mhz": min_clk, "clock_max_mhz": max_clk,
           "clock_sag_mhz": round(max_clk - min_clk, 1), "n_samples": len(samples)})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
```

### The suite driver

```bash title="acceptance/run_acceptance.sh"
#!/usr/bin/env bash
# Scripted acceptance suite. Produces acceptance-report-vN and archives it to the NAS.
# Run from the train/ uv project (it has torch+transformers): bash run_acceptance.sh
set -euo pipefail

source ~/.config/thesis-loop/storage.env    # HF_HOME, ARCHIVE_ROOT
STAMP="$(date --iso-8601=seconds)"
VER="${1:-1}"
REPORT="acceptance-report-v${VER}-$(date +%Y%m%d).md"

{
  echo "# Acceptance report v${VER} - MSI Aegis R2 (A2NVV9-2218US)"
  echo "generated: ${STAMP}"
  echo
  echo "## Idle / static facts"
  echo '```'
  nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,power.draw --format=csv
  echo '```'
  echo
  echo "## GEMM, bandwidth, first-tokens (JSON)"
  echo '```json'
  uv run python checks.py all
  echo '```'
  echo
  echo "## Thermals under sustained load (300 s)"
  echo '```'
  uv run python thermal_watch.py 300
  echo '```'
  echo
  echo "## Pass/fail (fill in against tolerances)"
  echo "- [ ] card named RTX 5080, 16 GB total, driver on 570 branch, link Gen5 x16"
  echo "- [ ] GEMM correct (max_rel_err < 5e-2) and tflops_est within tolerance of rated"
  echo "- [ ] bandwidth gb_per_s_est a healthy fraction of ~960 GB/s rated"
  echo "- [ ] thermals: peak temp below throttle, clock_sag small (clocks held)"
  echo "- [ ] first_tokens: generated_ok true, completion coherent"
} | tee "$REPORT"

# Archive to the NAS to make it permanent.
mkdir -p "${ARCHIVE_ROOT}/reports"
cp "$REPORT" "${ARCHIVE_ROOT}/reports/"
echo "Archived $REPORT to ${ARCHIVE_ROOT}/reports/"
```

### Logging acceptance as an MLflow run

```python title="acceptance/log_acceptance.py"
"""Wrap the suite as an MLflow run so acceptance carries the five provenance fields.

Run: uv run python log_acceptance.py acceptance-report-v1-YYYYMMDD.md
"""
import sys
import mlflow
sys.path.insert(0, "../tracking")
from provenance import start_run   # noqa: E402

report_path = sys.argv[1]
with start_run(experiment="acceptance", lock_path="uv.lock", seed=0, run_name="v1"):
    mlflow.log_artifact(report_path)     # the report becomes an MLflow artifact too
    mlflow.set_tag("acceptance_version", "1")
    print("Logged acceptance report to MLflow.")
```

**What you should see.** The suite prints a report in which: the idle line names the card as an RTX 5080 with 16 GB total, a 570-branch driver, and a Gen5 x16 link; the GEMM check reports `correct: true` with a small `max_rel_err` and a `tflops_est` in a plausible fraction of the card's rated compute; the bandwidth check reports a `gb_per_s_est` that is a healthy fraction of the roughly 960 GB/s rated figure; the thermal watch reports a `peak_temp_c` below the throttling threshold with a small `clock_sag_mhz` (meaning clocks held under sustained load); and `first_tokens` reports `generated_ok: true` with a coherent completion of "The capital of France is." Every one of those numbers is a placeholder until you run it; measured on the baseline machine; record value, date, driver. The report is saved locally, copied to `${ARCHIVE_ROOT}/reports/` on the NAS, and logged as an MLflow run carrying the five mandatory fields. That archived `acceptance-report-v1` is the artifact and the machine's clean bill of health: from here on, when a later number looks wrong, the first move is to re-run the suite and diff against v1, and the whole rest of the book is allowed to trust the instrument.

```admonish gotcha title="A silent slow path looks like healthy hardware"
The scariest acceptance failure is not a crash, it is a pass that is quietly wrong. If PyTorch or CUDA falls back to a non-tensor-core code path, the GEMM check will still be *correct* but its `tflops_est` will be a fraction of what the card can do, and nothing errors. Likewise a link that renegotiated to x8 halves bandwidth with no crash. This is exactly why acceptance records throughput estimates and compares them to rated figures, not just correctness: on this stack, 'it ran and the answer was right' is necessary but not sufficient. Always read the throughput numbers, not just the pass/fail of correctness.
```

```admonish thesis-thread
For the committee, acceptance report v1 is the calibration certificate for the instrument that produces every subsequent measurement. It establishes, on a named date and driver, that the machine hits a known fraction of its rated bandwidth and compute, holds thermals under sustained load, and generates correctly. Re-running the suite after any environment change and diffing against v1 is what lets the thesis rule out 'the machine silently regressed' as an explanation for a shift in a reasoning-delta curve. Without this gate, a hardware or driver regression would be indistinguishable from a genuine training effect.
```

```admonish substack-seed
The idea worth a post is that the most dangerous failure in ML infrastructure is not the crash, it is the silent slow path: a GPU that runs, returns correct answers, and quietly delivers a third of its rated throughput because a kernel fell back or a PCIe link renegotiated down. Correctness tests will not catch it; only throughput tests compared against the spec will. Frame acceptance as calibrating an instrument before you trust its readings - you would not report measurements from a scale you had never checked against a known weight - and you have a piece that reframes 'it works' as the beginning of trust, not the end of it.
```
