# Sampling and decoding

The model gives you a probability distribution over the next token. It does not give you a token. The step that turns that distribution into an actual choice is *sampling*, and it is where a great deal of eval reproducibility is won or lost, because two people running "the same model on the same benchmark" can get different scores purely because one used greedy decoding and the other sampled at temperature 0.7. Sampling is a set of knobs that reshape the distribution before drawing from it: temperature, top-k, top-p, min-p, repetition penalties. Each one is a small, exact transformation, and I am going to derive the most important (temperature, as a Boltzmann rescaling) and specify the rest precisely, then argue hard that evals must pin every one of these plus the seed, and be honest about where even a pinned seed stops guaranteeing determinism. The lab runs one prompt through a sweep of sampler settings and shows the outputs diverge, which is the whole reproducibility argument made visible.

## Theory

### Greedy versus stochastic

The simplest decoder is **greedy**: at each step take the single highest-probability token, $t_i = \arg\max_v p_\theta(v \mid t_{<i})$. It is deterministic (same input, same output, always) and it is what you often want for evals where there is one right answer and you want to measure the model's best single guess. Its weakness is that locally-greedy is not globally-optimal (the highest-probability token now can lead into a low-probability continuation later, the classic garden-path problem), and it produces repetitive, flat text because it never takes a risk. **Stochastic** decoding instead *samples* from the distribution, $t_i \sim p_\theta(\cdot \mid t_{<i})$, which introduces diversity and, for reasoning models, is often necessary because the interesting solution path is not always the single most probable token at every step. The knobs below all operate on stochastic decoding, reshaping the distribution you sample from. Greedy is the limit of one of them (temperature to zero), which is the first thing to derive.

### Temperature as Boltzmann rescaling

Temperature is the master knob, and it comes straight from statistical physics, which is not a coincidence: the softmax *is* the Boltzmann distribution, and temperature is the same temperature.

```admonish derivation title="Temperature as a Boltzmann rescaling of the logits"
Recall the softmax over logits from the objective chapter, $p_v = e^{\ell_v} / \sum_u e^{\ell_u}$. In statistical mechanics, the probability of a state with energy $E_v$ at temperature $T$ is the Boltzmann distribution $p_v \propto e^{-E_v / (k_B T)}$. Identify the negative logit with energy ($\ell_v \leftrightarrow -E_v/k_B$) and the two are the same object. Temperature enters by dividing the logits by a scalar $\tau > 0$ before the softmax:

$$p_v(\tau) = \frac{e^{\ell_v / \tau}}{\sum_{u=1}^{V} e^{\ell_u / \tau}}. \tag{7.1}$$

Now read off the two limits, because they explain everything the knob does. As $\tau \to 0^+$, divide the logits by a vanishing number and the largest logit's exponential dominates all others infinitely, so the distribution collapses onto the single most probable token: $p(\tau) \to$ one-hot at $\arg\max_v \ell_v$. **Temperature zero is exactly greedy decoding**, which is why "temperature 0" and "greedy" are used interchangeably. As $\tau \to \infty$, divide by a huge number and every logit maps toward the same value, so $p_v(\tau) \to 1/V$, the uniform distribution, maximal randomness, the model's preferences erased. Between them, $\tau = 1$ leaves the distribution exactly as the model produced it (equation 7.1 with $\tau=1$ is the plain softmax).

To see the *sharpening* precisely, look at the ratio of any two probabilities:

$$\frac{p_a(\tau)}{p_b(\tau)} = e^{(\ell_a - \ell_b)/\tau}. \tag{7.2}$$

The log-odds between any pair of tokens is the fixed logit gap $\ell_a - \ell_b$ divided by $\tau$. Lowering $\tau$ multiplies every log-odds by $1/\tau > 1$, stretching the distribution's contrasts and concentrating mass on the front-runners; raising $\tau$ shrinks the log-odds toward zero, flattening. Temperature does not change the *ranking* of tokens (equation 7.2 keeps the same sign), only how sharply the probability mass is distributed across that ranking. That single fact, reranking-preserving, contrast-scaling, is why temperature is the right first knob and why calling it "creativity" is a reasonable shorthand for "how far down the ranked list the sampler is willing to wander."
```

### Truncation: top-k, top-p, min-p

Temperature reshapes the whole distribution but never zeroes anything out, so at high temperature there is always a small chance of sampling a genuinely bad token from the long tail. The truncation knobs fix that by *cutting the tail off* before sampling, then renormalizing what remains. They differ in how they decide where to cut.

**Top-k** keeps the $k$ highest-probability tokens and zeros the rest: sort the distribution, take the top $k$, renormalize over just those. It is simple but rigid, because a fixed $k$ is too permissive when the model is confident (the true answer is in the top 2, but you keep 50) and too restrictive when the model is genuinely uncertain (the mass is spread over 200 plausible tokens but you keep 50). 

**Top-p** (nucleus sampling) fixes the rigidity by cutting at a *cumulative probability mass* instead of a fixed count. Sort tokens by probability descending, then keep the smallest set whose cumulative probability first exceeds $p$ (say $0.9$), and renormalize:

$$\mathcal{N}_p = \text{smallest } S \text{ s.t.} \sum_{v \in S} p_v \ge p, \qquad \text{sample from } p \text{ restricted to } \mathcal{N}_p. \tag{7.3}$$

The kept set $\mathcal{N}_p$ (the "nucleus") is *adaptive*: when the model is confident, a couple of tokens already exceed $p$ and the nucleus is tiny; when it is uncertain, many tokens are needed to reach $p$ and the nucleus is large. That adaptivity is why top-p is the default in most serving stacks.

**Min-p** takes a different adaptive cut: keep every token whose probability is at least a fraction $p_{\min}$ of the *most probable* token's probability, $\{v : p_v \ge p_{\min} \cdot \max_u p_u\}$. It scales the threshold to the model's confidence directly, keeping more when the top token is only mildly favored and fewer when one token dominates, and it tends to behave better than top-p at high temperature because it is anchored to the peak rather than to a cumulative sum that a fat tail can inflate.

The order these are applied in matters and is a real source of cross-tool disagreement: some stacks apply temperature first then truncate, others truncate on the raw distribution then apply temperature to the survivors. The results differ, which is exactly why "temperature 0.7, top-p 0.9" is not a complete specification unless you also know the tool and its ordering.

### Repetition penalties

Language models left to sample can fall into loops, repeating a phrase because each repetition makes the next repetition more likely (a token that just appeared has high probability of appearing again). Repetition penalties fight this by down-weighting tokens that have already appeared. The common **repetition penalty** $r > 1$ divides (for positive logits) or multiplies (for negative) the logit of any previously-seen token by $r$ before the softmax, pushing its probability down. Variants like **frequency penalty** and **presence penalty** subtract a term proportional to how often (frequency) or whether at all (presence) a token has appeared. These are heuristic, not derived from a principle, and they interact with the truncation knobs, which is one more setting that has to be pinned to reproduce a result.

### Why evals must pin all of it, and the limits of seeds

Here is the argument that makes this chapter matter for the thesis. An eval score is a measurement of a model, and a measurement is meaningless if you cannot say what apparatus produced it. Every knob above changes the *distribution of outputs*, so two eval runs with different sampler settings are measuring different things even on the identical model and prompts. Greedy versus temperature-0.7 can swing a math benchmark by several points, because greedy commits to one reasoning path while sampling explores several; top-p 0.9 versus top-p 1.0 changes which tail tokens are reachable; the tool's truncation-ordering changes the numbers again. So a defensible eval **pins the entire decoding configuration**: decoder (greedy or sampled), temperature, top-k, top-p, min-p, repetition penalties, max tokens, stop sequences, and the seed, all logged next to the score. This is the sampler-side twin of the chat-template rule from the tokenization chapter: both are parts of the model's effective input/output contract that are invisible if you only log the model name.

The seed deserves its own honesty. Setting a random seed makes the *pseudo-random draws* reproducible, so on a single machine, single GPU, single library version, a seeded sampled run repeats. But a seed does **not** guarantee determinism across everything people assume it does. Floating-point reductions on a GPU are not associative (chapter 1, equation 1.8), and many CUDA kernels sum in a nondeterministic order for speed, so the same logits can come out a few ULPs different run to run, and near a truncation boundary or a close $\arg\max$ that tiny difference can flip which token is chosen and cascade into a completely different generation. Batching changes results too: a prompt decoded alone versus in a batch with others can hit different kernels and different numerics. And different hardware, driver, or library versions reorder reductions differently. So the practical rule is: seed for repeatability on a fixed rig, but do not claim bit-identical outputs across machines or batch sizes, and for evals prefer greedy (which is robust to small logit perturbations *unless* two tokens are near-tied) or report over multiple seeds with a confidence interval rather than trusting one seeded run. The statistics of that (how many samples, what interval) is a Part III chapter; the point here is that determinism has a boundary and you should know where it is.

## Tooling

The tools are two, at two altitudes. At the library level, Hugging Face `transformers` exposes all of this through `model.generate` and a `GenerationConfig`: `do_sample` (greedy versus stochastic), `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, `max_new_tokens`, and the seed via `transformers.set_seed` or a manual `torch.manual_seed`. The `GenerationConfig` is a serializable object, which is exactly what you want for reproducibility: you can save it next to an eval result as the record of the decoding apparatus. At the serving level, vLLM (Part II) exposes the same knobs through its `SamplingParams`, and the two do *not* always apply them in the same order or with identical defaults, which is a concrete reason a benchmark run through `transformers.generate` and the same benchmark through a vLLM endpoint can disagree slightly even with matching nominal settings. Knowing that the knobs are the same math (equations 7.1–7.3) but the *ordering and defaults* differ across tools is what lets you debug a "why did my score change when I switched serving backends" mystery instead of being baffled by it.

```admonish under-the-hood
`GenerationConfig` has defaults baked into each model's `generation_config.json` on the Hub, so `model.generate()` with no arguments does *not* mean greedy, it means whatever that checkpoint shipped as its default, which is frequently `do_sample=True` at some temperature. This is a classic footgun: you think you are measuring greedy accuracy and you are actually measuring a sampled run at the model author's chosen temperature. Always pass an explicit `GenerationConfig` (or explicit kwargs) in an eval, never rely on the checkpoint default, and log the config you passed. The default is convenience for chat, not a specification for measurement.
```

## Lab

The goal is to make sampler sensitivity undeniable: take one fixed prompt and one fixed model, run it through a sweep of decoding configurations (greedy, low temperature, high temperature, top-p, min-p), with a fixed seed for the stochastic ones, and record how the generated text diverges. The artifact is a JSON table of (config, output) pairs plus a diversity measure, which is the evidence behind "pin the sampler in your evals."

```bash
uv init labs/sampling
cd labs/sampling
uv add torch transformers
```

```python title="labs/sampling/sampler_sweep.py"
"""Run one prompt through a sweep of decoding configs and diff the outputs.

Artifact: artifacts/sampler_sweep.json with per-config generations.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Give me a one-sentence description of a city by the sea."
SEED = 0
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

CONFIGS = [
    {"name": "greedy",         "do_sample": False},
    {"name": "temp_0.3",       "do_sample": True, "temperature": 0.3},
    {"name": "temp_1.0",       "do_sample": True, "temperature": 1.0},
    {"name": "temp_1.5",       "do_sample": True, "temperature": 1.5},
    {"name": "top_p_0.9",      "do_sample": True, "temperature": 1.0, "top_p": 0.9},
    {"name": "top_k_20",       "do_sample": True, "temperature": 1.0, "top_k": 20},
    {"name": "min_p_0.1",      "do_sample": True, "temperature": 1.0, "min_p": 0.1},
]

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    results = []
    for cfg in CONFIGS:
        name = cfg.pop("name")
        gens = []
        # Two draws per config to show stochastic spread (greedy repeats itself).
        for run in range(2):
            set_seed(SEED + run)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=40, pad_token_id=tok.eos_token_id, **cfg
                )
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            gens.append(text.strip())
        results.append({"config": name, "params": cfg, "gen_seed0": gens[0], "gen_seed1": gens[1]})

    # Greedy should be identical across the two draws; sampling should differ.
    for r in results:
        r["two_draws_identical"] = (r["gen_seed0"] == r["gen_seed1"])

    (OUT / "sampler_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"{'config':12} {'identical draws?':16} first generation")
    print("-" * 80)
    for r in results:
        print(f"{r['config']:12} {str(r['two_draws_identical']):16} "
              f"{r['gen_seed0'][:48]!r}")
    print(f"\nArtifact: {(OUT / 'sampler_sweep.json').resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python sampler_sweep.py
```

```admonish gotcha
When `do_sample=False` (greedy), `transformers` will warn if you also pass `temperature`/`top_p` because those knobs are ignored in greedy mode, so the config dicts above only attach sampling knobs to sampling configs, which is why greedy has none. Also, `set_seed` before each `generate` is what makes the two same-seed draws comparable; note that even so, greedy's two draws are identical because it is deterministic regardless of seed, while the sampled configs' two draws (seed 0 vs seed 1) differ, which is the point. If you run on CPU the outputs are reproducible but slow; on GPU you get the honest reminder from the theory that a *third* run at the same seed may still differ by a token if a CUDA reduction reorders near a boundary.
```

**What you should see.** The script writes `artifacts/sampler_sweep.json` and prints a table where the **same prompt** produces visibly different continuations across the seven configs. Greedy's two draws are identical (`two_draws_identical: true`), the confirmation that temperature-zero decoding is deterministic. The low-temperature (0.3) output is safe and generic; temperature 1.5 is noticeably wilder and may start to wander or degrade, exactly what equation (7.2)'s log-odds-shrinking predicts; and the truncation configs (top-p, top-k, min-p) sit in between, coherent but varied. The sampled configs' two seeds produce different text, making the seed's role concrete. The whole table is one page of evidence for the chapter's thesis: nothing but the decoder changed, and the model's behavior changed with it, so an eval that does not pin this configuration is not a reproducible measurement. Keep the JSON as the receipt, and note in it the exact `transformers` version, since truncation ordering and defaults can shift between releases (measured on the baseline machine, record value, date, driver).

```admonish read-along
Read **[BRM]** ch. 2, which walks through decoding strategies for reasoning models specifically, greedy versus sampling, temperature, and the truncation methods, with the reasoning-model angle that the "best" single path is often not the greedy one, so sampling and inference-time search matter more here than for ordinary chat. Equation (7.1)'s temperature limits and the top-p adaptivity of equation (7.3) are the mechanics behind the strategies Raschka compares, and his chapter is the bridge from these knobs to the inference-time-scaling ideas that recur in Part VI. **[AIE]** ch. 2 covers the same knobs from the production side (sampling fundamentals and strategies, structured outputs, and why sampling makes model behavior probabilistic); read it as the practitioner's recap of the mechanics this chapter derives.
```

```admonish substack-seed
"Temperature in an LLM is literally temperature, the softmax is the Boltzmann distribution, and turning the knob to zero is how you get greedy decoding." A post that identifies the logit with negative energy, shows that dividing logits by tau scales every pairwise log-odds by 1/tau (so it sharpens contrasts without changing the ranking), and reads off the two limits: tau to zero collapses to the single best token, tau to infinity flattens to uniform. Then the practical sting for anyone benchmarking models: because every one of these knobs reshapes the output distribution, an eval that reports a score without reporting temperature, top-p, and seed is reporting a measurement with no units, and even the seed only buys you determinism until a nondeterministic GPU reduction flips a near-tied token. Physics, decoding, and reproducibility in one thread.
```
