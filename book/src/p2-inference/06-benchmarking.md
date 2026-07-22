# Benchmarking without lying to yourself

The runbook stands up a server; this chapter measures it honestly. That qualifier matters, because inference benchmarking is unusually easy to get wrong in ways that flatter your own stack. You warm nothing up and blame the model for a cold-start artifact. You report a single run and call the noise a result. You quote peak throughput at batch 64 and per-request latency in the same breath as if a user experiences both. The whole discipline of this chapter is measurement hygiene: defining metrics that mean something, controlling the conditions, quantifying variance, and producing a baseline report I would be willing to publish and, more to the point, willing to reuse as the fixed serving substrate for every training experiment later in the thesis. If the baseline is dishonest, every reward signal built on top of it inherits the lie.

The stakes are higher here than in a one-off benchmark blog post precisely because this number gets reused. Later parts of the thesis will hold the serving substrate fixed and vary something upstream (a quantization choice, a training checkpoint, a sampling setting) and ask whether throughput or latency moved. That comparison is only meaningful if the baseline was measured with enough rigor that I know its noise floor and its exact conditions. A baseline recorded sloppily is not just wrong once, it silently corrupts every downstream comparison that leans on it, and I will not know, because the corruption looks like a plausible number. So the discipline in this chapter is not perfectionism for its own sake; it is the minimum required for the baseline to be a foundation rather than a liability.

## Theory

### Latency and throughput are different axes, and they trade off

The two headline numbers answer different questions and pull in opposite directions.

**Latency** is how long one request takes, measured from the requester's side. It is what a single user feels. **Throughput** is how many tokens (or requests) the server completes per second across all concurrent work. It is what a batch eval job cares about. They trade off because the lever that raises throughput (bigger batches, more concurrency) also raises per-request latency: your request now shares the GPU with others and waits its turn in each scheduler step. You cannot summarize a server with one number, and any benchmark that reports only throughput or only latency is answering half the question. The honest artifact reports a curve, or at least both endpoints: latency at low load, throughput at saturation.

### The two latency metrics that matter: TTFT and TPOT

Because generation has two phases, latency has two components, and conflating them hides where time goes.

**Time to first token (TTFT)** is the wall-clock from sending the request to receiving the first output token. It is dominated by prefill (compute-bound, chapter 1) plus queueing delay if the server is busy. Long prompts and loaded servers inflate it. This is the "did it hang?" number a user judges responsiveness by.

**Time per output token (TPOT)**, also called inter-token latency, is the steady-state time between subsequent tokens during decode. It is dominated by the bandwidth-bound decode step (chapter 1) and, under batching, by how many sequences share each forward pass. Its reciprocal is the per-sequence decode tok/s.

```admonish derivation title="End-to-end latency decomposes cleanly"
For a request with prompt length $S$ and $N$ generated tokens, total latency is:

$$T_{\text{total}} = \underbrace{T_{\text{queue}}}_{\text{scheduler wait}} + \underbrace{T_{\text{prefill}}(S)}_{\approx\, \text{TTFT}} + \underbrace{(N-1)\cdot \text{TPOT}}_{\text{decode}}$$

This decomposition is why you must report TTFT and TPOT separately rather than a single latency: they scale with different inputs ($S$ for TTFT, $N$ for the decode term) and are governed by different hardware roofs (compute for prefill, bandwidth for decode). A change that improves one can worsen the other, e.g. chunked prefill lowers other requests' TPOT variance at the cost of a long prompt's own TTFT. A single blended latency number would hide that entirely.
```

So the minimal honest latency report is a distribution of TTFT and a distribution of TPOT, each summarized by median and a tail percentile, not a mean.

### Why percentiles, not means

Latency distributions are right-skewed with heavy tails: most requests are fast, a few are slow because they got queued behind a long prefill or hit a preemption (chapter 4). The mean is dragged around by the tail and hides the typical experience; the tail is exactly what you must not ignore because it is where users churn. Report the **median (p50)** for the typical case and **p95 or p99** for the tail. A mean alone is a way of lying to yourself with a true number.

### Warmup, and why the first runs are garbage

The first requests to a fresh server are systematically slow for reasons that have nothing to do with steady-state performance: CUDA kernels are being JIT-compiled or autotuned, CUDA graphs are being captured, the KV allocator is cold, caches are empty, and clock speeds may not have ramped. Including these in your statistics contaminates them. The fix is a **warmup phase**: send a batch of representative requests, discard their timings, and only then start measuring. Skipping warmup is the single most common way people accidentally slander their own stack.

Warmup has to be representative, not token. If my real workload sends 2K-token prompts and generates 256 tokens, my warmup should do the same, because the kernels autotuned and the graphs captured are shape-dependent; warming up with a ten-token prompt leaves the very code paths my benchmark exercises still cold. The rule I follow is to warm up with the same request distribution I am about to measure, run enough of it that the throughput number stabilizes across successive batches, and only then start recording. A cheap way to confirm warmup is sufficient is to plot the per-request latency against request index and check that it has flattened before the measurement window begins; if it is still descending, the card has not reached steady state and my first "measurements" are really extended warmup wearing a disguise.

### Load generation: open vs closed

How you generate load changes what you measure, and the distinction has a name. A **closed-loop** generator keeps a fixed number of concurrent clients, each sending a new request as soon as its previous one returns; it measures the server at a fixed concurrency and naturally finds the throughput ceiling. An **open-loop** generator sends requests at a fixed arrival *rate* regardless of whether prior ones finished, which models real traffic and exposes queueing collapse when the arrival rate exceeds service capacity. For a baseline I use closed-loop sweeps across concurrency levels to trace the latency-throughput curve; the rate-based open-loop test is how I later check a chosen operating point survives bursty eval traffic. Reporting which one you ran is part of the hygiene, because the same server produces different-looking numbers under each.

The distinction is not academic. A closed-loop test with 8 clients can never overload a server, because each client waits for its reply before sending again, so the offered load self-throttles to whatever the server can handle; it measures capacity but hides queueing collapse. An open-loop test at 20 requests per second against a server that can only clear 12 will show latency climbing without bound as the queue grows, which is the honest picture of what happens when real traffic exceeds capacity. For the thesis, the eval workload is closer to closed-loop (a harness fires the next prompt when the last returns), so the closed-loop sweep is the right baseline. But before I trust an operating point in production I run one open-loop test at the expected arrival rate to confirm the server has margin, because the failure mode I fear is the one closed-loop testing structurally cannot show me.

### Variance across runs

A single run is an anecdote. Thermal state, background processes, driver behavior, and scheduling nondeterminism all move the numbers between otherwise-identical runs. The honest move is to run the whole benchmark several times and report the **variance or a confidence interval** on the summary metrics, so a later comparison (say, does an FP8 KV cache change throughput?) can be judged against the noise floor rather than mistaken for signal. If two configs differ by less than the run-to-run spread, they do not differ.

### Comparing two configurations honestly

Most of the value of a baseline is that later I will change one thing and ask whether it helped. That question is a statistical one, not a matter of eyeballing two numbers. If configuration A throughputs at $92 \pm 4$ tok/s (mean plus-minus stdev over runs) and B at $95 \pm 5$, the three-token gap is inside the noise and I have learned nothing. The cheap discipline is to keep everything identical except the one variable, run each config several times, and compare the difference in means against the spread; a rough rule is that the difference should exceed roughly two standard deviations of the run-to-run variation before I treat it as real. When the effect is small and I care (a 3% throughput change is worth real money at scale), I run more repeats to shrink the standard error, because the uncertainty on a mean falls as $1/\sqrt{k}$ in the number of runs $k$. The Part III statistics chapter develops this properly for eval metrics; for inference the same instinct applies, and the harness below records the per-run spread precisely so this comparison is possible at all rather than being reconstructed from a single lucky number.

```admonish gotcha
Determinism traps to control before you trust a number: pin the driver and vLLM version (stamp them in the report); fix `temperature=0` and a seed if you want reproducible token counts, since sampling changes $N$ and thus the decode term; hold `max_tokens` fixed so throughput comparisons are on equal output lengths; disable other GPU consumers; and let the GPU reach thermal steady state (the first minute of a cold card runs at different clocks). None of these are optional if you plan to reuse the baseline as a reference substrate.
```

## Tooling

Two layers of tooling. vLLM ships a benchmark script (`vllm bench serve`, formerly `benchmarks/benchmark_serving.py`) that does closed and open-loop load with proper TTFT/TPOT accounting, and it is the right tool for the headline numbers; it also has the advantage of being the reference implementation the vLLM community quotes, so numbers from it are comparable to numbers others publish. But I also want a small harness I fully control and understand, because the baseline report needs to record exactly my conditions and feed my own plots. The harness below wraps the OpenAI-compatible streaming API, does explicit warmup, sweeps concurrency, records per-request TTFT and TPOT, repeats runs for variance, and reads the server's `/metrics` for cross-checking. It stays deliberately small so nothing is hidden.

## Lab: a benchmark harness and a baseline report

The artifact is `baseline/inference_baseline.json` plus a human-readable `baseline/REPORT.md`: the inference baseline that later parts of the thesis treat as the fixed serving substrate. It uses measured-number placeholders throughout, to be filled on the machine.

### Set up

```bash title="shell"
cd ~/thesis-tech-stack && mkdir -p baseline && cd baseline
uv init bench && cd bench
uv add openai numpy
```

### The harness

```python title="baseline/bench/harness.py"
"""Controlled latency/throughput harness against a streaming vLLM server.

Does explicit warmup, closed-loop concurrency sweep, per-request TTFT/TPOT,
and multi-run variance. Reports p50/p95, never bare means.
"""
import asyncio, time, argparse, json, statistics
from openai import AsyncOpenAI

PROMPTS = [
    "Explain why decode is memory-bound in three sentences.",
    "List five prime numbers and their squares.",
    "Summarize the roofline model for a colleague.",
    "What is 17 times 23? Show your working.",
]

async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    t_first = None
    n = 0
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, stream=True)
    async for chunk in stream:
        if not chunk.choices or chunk.choices[0].delta.content is None:
            continue
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        n += 1
    t_end = time.perf_counter()
    ttft = (t_first - t0) if t_first else float("nan")
    tpot = ((t_end - t_first) / (n - 1)) if n > 1 else float("nan")
    return {"ttft": ttft, "tpot": tpot, "out_tokens": n,
            "total": t_end - t0}

async def run_at_concurrency(client, model, conc, n_requests, max_tokens):
    sem = asyncio.Semaphore(conc)
    async def guarded(p):
        async with sem:
            return await one_request(client, model, p, max_tokens)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    wall0 = time.perf_counter()
    results = await asyncio.gather(*[guarded(p) for p in prompts])
    wall = time.perf_counter() - wall0
    total_out = sum(r["out_tokens"] for r in results)
    return {
        "concurrency": conc,
        "throughput_tok_s": total_out / wall,
        "ttft_p50": statistics.median(r["ttft"] for r in results),
        "ttft_p95": _pct([r["ttft"] for r in results], 95),
        "tpot_p50": statistics.median(r["tpot"] for r in results if r["tpot"] == r["tpot"]),
        "tpot_p95": _pct([r["tpot"] for r in results if r["tpot"] == r["tpot"]], 95),
        "n_requests": n_requests,
    }

def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[k]

async def main(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    # 1. Warmup: fire and discard.
    print("warmup...")
    await run_at_concurrency(client, args.model, conc=4,
                             n_requests=16, max_tokens=args.max_tokens)

    # 2. Concurrency sweep, repeated for variance.
    sweep = {}
    for conc in args.concurrency:
        runs = []
        for r in range(args.repeats):
            res = await run_at_concurrency(client, args.model, conc,
                                           args.requests, args.max_tokens)
            runs.append(res)
            print(f"  conc={conc} run={r} "
                  f"tp={res['throughput_tok_s']:.1f} tok/s "
                  f"ttft_p50={res['ttft_p50']*1000:.0f}ms "
                  f"tpot_p50={res['tpot_p50']*1000:.1f}ms")
        # aggregate across repeats with mean+stdev of the summary metrics
        sweep[conc] = _aggregate(runs)

    report = {
        "model": args.model,
        "conditions": {
            "max_tokens": args.max_tokens, "temperature": 0.0,
            "repeats": args.repeats, "requests_per_run": args.requests,
            "STAMP": "RECORD vLLM version / driver 570-open / date / GPU RTX 5080",
        },
        "sweep": sweep,
    }
    with open("../inference_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote baseline/inference_baseline.json")

def _aggregate(runs):
    def ms(key):
        vals = [r[key] for r in runs]
        return {"mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    return {
        "throughput_tok_s": ms("throughput_tok_s"),
        "ttft_p50_s": ms("ttft_p50"), "ttft_p95_s": ms("ttft_p95"),
        "tpot_p50_s": ms("tpot_p50"), "tpot_p95_s": ms("tpot_p95"),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-14b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    asyncio.run(main(ap.parse_args()))
```

### Run it against the workhorse

Start the 14B-AWQ server from the runbook, then:

```bash title="shell"
uv run python harness.py --model qwen3-14b \
    --concurrency 1 4 8 16 --requests 64 --repeats 3 --max-tokens 256
```

Cross-check throughput against vLLM's own tool and its metrics:

```bash title="shell"
uv run vllm bench serve --model qwen3-14b \
    --base-url http://localhost:8000 --num-prompts 128 --request-rate 8
curl -s http://localhost:8000/metrics | grep -E 'time_to_first_token|time_per_output_token'
```

### Write the baseline report

```markdown title="baseline/REPORT.md"
# Inference Baseline: MSI Aegis R2 (RTX 5080 16GB)

STAMP: vLLM VERSION, NVIDIA 570-open, Ubuntu 24.04, DATE. Reused downstream as the
fixed serving substrate; do not compare across differently-stamped runs.

## Method
- Closed-loop concurrency sweep {1,4,8,16}, 64 requests/run, 3 repeats.
- Explicit warmup (16 req discarded). temperature=0, max_tokens=256 fixed.
- Metrics: TTFT and TPOT as p50/p95; throughput as tok/s at saturation.
- Cross-checked against `vllm bench serve` and /metrics histograms.

## Results (fill from inference_baseline.json)
| model | conc | throughput tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
|---|---|---|---|---|---|---|
| qwen3-14b (AWQ) | 1  | RECORD | RECORD | RECORD | RECORD | RECORD |
| qwen3-14b (AWQ) | 8  | RECORD | RECORD | RECORD | RECORD | RECORD |
| qwen3-14b (AWQ) | 16 | RECORD | RECORD | RECORD | RECORD | RECORD |
| qwen3-8b  (BF16)| 1  | RECORD | RECORD | RECORD | RECORD | RECORD |
| gpt-oss-20b     | 1  | RECORD | RECORD | RECORD | RECORD | RECORD |

## Sanity vs theory (chapter 1 ceilings)
- qwen3-14b AWQ decode ceiling ~116 tok/s (single seq). Measured TPOT p50 at conc=1
  implies RECORD tok/s; ratio to ceiling = RECORD (expect 0.5-0.8).
- Throughput should RISE with concurrency (batching amortizes weight reads) while
  per-request TPOT DEGRADES; confirm both trends in the table.

## Variance
- Run-to-run stdev on throughput at conc=8: RECORD tok/s (noise floor for later A/Bs).
```

### What you should see

The concurrency sweep should show the textbook shape: throughput rises as concurrency climbs (batching amortizes the weight read across more tokens, the roofline's intensity boost made visible), while per-request TPOT degrades because each sequence now shares the forward pass. TTFT should stay low at concurrency 1 and climb at higher concurrency as requests queue. At concurrency 1, the measured decode rate (the reciprocal of TPOT p50) should land at roughly half to four-fifths of the chapter-1 ceiling for that model (about 116 tok/s for the 14B AWQ), the gap being kernel overhead, sampling, and KV reads; if it exceeds the ceiling, my byte or bandwidth accounting is wrong. The multi-run stdev columns give me a noise floor: any future change I test (FP8 KV, a different `--max-num-batched-tokens`) has to move a metric by more than that stdev before I am allowed to call it a real effect. The committed `inference_baseline.json` and `REPORT.md`, fully stamped with vLLM version, driver, and date, are the artifact the training chapters reuse whenever they need to know how fast the serving substrate is, so its honesty is load-bearing for everything downstream. Fill every RECORD from the real machine before treating the baseline as canonical.

```admonish substack-seed
"How to benchmark an LLM without fooling yourself." Most inference numbers you see are quietly dishonest, not through malice but through skipped hygiene: no warmup, so a cold CUDA-graph capture gets blamed on the model; a single run, so noise masquerades as a result; a mean latency that a heavy tail has dragged somewhere no real request lives; and the cardinal sin of quoting peak throughput and low latency together as if one user experienced both. This post is a field guide to measurement you can defend: separate TTFT from TPOT because they live on different hardware roofs, always report p50 and p95 instead of a mean, warm up and discard, sweep concurrency to draw the real latency-throughput curve, and repeat runs so you know your own noise floor before you claim an improvement. The deliverable is a benchmark you would stake a thesis on, which, as it happens, I am.
```
