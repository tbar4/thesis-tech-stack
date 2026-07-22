# uv and the two-environment doctrine

**Goal.** Establish one canonical way to manage Python, and separate the serving stack from the training stack so their dependencies never fight.

**Covers.** uv as a toolchain (Python pinning, virtual environments, lockfiles, `uv run`); why serve and train stay in separate projects; and a repository layout of two uv projects, `serve/` and `train/`, each with a committed `uv.lock`.

## Theory

### One tool for the whole Python surface

Python environment management has historically been a pile of tools that each do one slice: `pyenv` to pick an interpreter, `venv` to make an environment, `pip` to install, `pip-tools` or `poetry` to lock. Every seam between them is a place for "works on my machine" to hide. uv collapses that whole pile into one fast tool. It installs and pins Python interpreters, creates virtual environments, resolves and installs dependencies, writes a cross-platform lockfile, and runs commands inside the environment, all from one binary. For a book whose entire premise is that nothing should be a black box, having one tool with one mental model for "what Python, which packages, exactly which versions" is worth a lot.

The reason I standardize on uv, and forbid myself bare `pip` and `python -m venv` for the rest of the book, is reproducibility with teeth. A `requirements.txt` full of loose version ranges tells you what you asked for, not what you got. uv's lockfile records the exact resolved version and hash of every package in the dependency graph, transitively, so that `uv sync` on another day rebuilds the identical environment or fails loudly if it cannot. When a reasoning-delta number in Part 7 looks surprising, the first question is always "did the environment change," and a committed `uv.lock` answers it definitively instead of with a shrug.

```admonish under-the-hood title="What uv actually stores, and why it is fast"
uv resolves your declared dependencies (in `pyproject.toml`) into a fully pinned dependency graph and writes it to `uv.lock`, including exact versions and content hashes. `uv sync` then materializes a `.venv` from that lock, and because uv keeps a global content-addressed cache of downloaded wheels and hardlinks them into each environment, creating an environment is mostly filesystem links rather than downloads. The interpreter itself can be a uv-managed build pinned by `uv python pin`, so "which Python" is recorded too. The net effect: the tuple (`pyproject.toml`, `uv.lock`, pinned Python) fully determines the environment, and rebuilding it is a link-heavy, near-instant operation rather than a fresh compile-and-download.
```

### Why serve and train are two environments, not one

Here is the doctrine, and it falls directly out of the 16 GB constraint and out of dependency reality. The serving stack (vLLM) and the training stack (Unsloth, and the PyTorch/transformers/TRL/bitsandbytes constellation around it) are two large, fast-moving, deeply-pinned dependency trees. They both pin specific versions of PyTorch, CUDA-linked libraries, `transformers`, and low-level kernels, and those pins do not agree. Trying to satisfy both in one environment forces the resolver into a compromise that makes at least one of them unhappy, and the failure is often not a clean resolver error but a subtle runtime breakage: a kernel compiled against the wrong CUDA minor, an attention backend that silently falls back to a slow path, a `transformers` version that one side needs and the other rejects.

So I keep them apart, and the separation buys three distinct things. First, **clean resolution**: each project locks its own coherent dependency graph with no cross-contamination, so vLLM gets exactly the stack it wants and Unsloth gets exactly the stack it wants. Second, **temporal separation on the GPU**: the loop already serves and trains at different times (Chapter 0.1), never both at once on 16 GB, so there is no reason to co-resident their software either; two environments make "one heavy thing at a time" the path of least resistance. Third, **independent evolution**: I can bump vLLM to chase a serving improvement without touching the trainer, and re-lock only `serve/`, so a serving upgrade cannot silently perturb a training result. Each environment moves on its own clock, and each `uv.lock` timestamps that clock.

```admonish gotcha title="One env for both = a resolver stalemate or a silent kernel mismatch"
If you try to `uv add vllm` and `uv add unsloth` into the same project, the best case is a slow, thrashing resolution that ends in a conflict you have to hand-arbitrate; the worse case is a resolution that "succeeds" but pins a PyTorch or CUDA-linked library version that one side then mis-uses at runtime, giving you a slow attention path or an import-time CUDA error that looks like a hardware problem. Keep them in separate projects. The seam between serving and training in this book is an HTTP endpoint (Chapter 0.1), not a shared Python process, so there is never a technical need for them to share an interpreter.
```

### Why not conda, poetry, or bare pip

It is fair to ask why uv specifically, given that conda, poetry, pipenv, and plain pip-plus-venv all exist and all "work." The honest answer is a combination of speed, correctness, and scope. Conda solves the interpreter-and-native-library problem well but is slow to resolve and pulls packages from its own channels, which drifts from the PyPI reality that vLLM and Unsloth actually publish to. Poetry locks nicely but does not manage the Python interpreter itself and has historically been slow on large graphs. Bare pip does not lock at all in any reproducible way without bolting on `pip-tools`, and `python -m venv` leaves you managing interpreters by hand. uv does the whole span (interpreter, environment, resolution, locking, running) in one fast tool, with a lockfile that pins exact versions and hashes, which is precisely the reproducibility property this book needs. So the choice is not fashion; it is that uv is the one tool whose scope matches "pin everything and rebuild it identically," which is the only workflow a measurement-driven thesis can trust.

The corollary is the rule I enforce on myself for the rest of the book: no bare `pip install`, ever, and no `python -m venv`, ever. The moment I `pip install` something into an environment imperatively, that environment stops being a pure function of the committed lockfile and becomes a snowflake that only exists on this machine, at this moment, in a state no one can reproduce. Every dependency enters through `uv add` so it lands in `pyproject.toml` and `uv.lock`, and every command runs through `uv run` so it uses the locked environment. It is a small discipline that pays off the first time I need to reproduce a three-month-old result and find that `uv sync` on the old commit just works.

### The evaluator is a thin third thing, and that is deliberate

A fair question: if serve and train each get an environment, where does the evaluation harness (Inspect) live? The answer is that the evaluator talks to the model over the OpenAI-compatible HTTP endpoint, so it only needs a light client stack, not the heavy serving or training dependencies. In practice Inspect can live in its own small environment or ride along in whichever project is convenient, because it never imports vLLM or Unsloth; it makes HTTP calls. I will pin down its exact home when Part 3 builds the eval harness. The point for now is that the two heavyweight environments are the ones the doctrine is about, and the evaluator's independence from both is a feature of the endpoint seam, not an afterthought.

### The lockfile is a contract, and CI can enforce it

A lockfile is only worth committing if it is treated as a contract rather than a suggestion, and the discipline that makes it a contract is small: I never edit `uv.lock` by hand, I only ever regenerate it through uv, and I treat an unexpected change to it in a diff as a signal that something moved that I should understand before I commit. The payoff is that the pair (`pyproject.toml`, `uv.lock`) becomes a checkable claim. If I check out an old commit to reproduce an old result, `uv sync` rebuilds the exact environment that produced it, or it fails loudly because a wheel is no longer available, in which case I know immediately that the old result cannot be reproduced with today's package index rather than discovering it silently through a subtly different number.

This is also where the two-environment doctrine pays a second dividend. Because each environment is locked independently, I can verify each one in isolation, and I can bisect a problem to one side. If a serving benchmark regresses, I check whether `serve/uv.lock` changed since the last good run; the training stack is provably irrelevant because it is a different lockfile that was not touched. Without the split, every dependency change is a change to one giant shared environment, and I lose the ability to say "the serving stack is identical, so the regression is not in the serving dependencies." The separation turns "the environment changed" from an undifferentiated worry into a precise, attributable fact.

```admonish under-the-hood title="Why `uv sync` is safe to run anywhere, anytime"
`uv sync` is idempotent and total: it makes the `.venv` exactly match `uv.lock`, adding what is missing, removing what should not be there, and downgrading what drifted, without consulting `pyproject.toml`'s loose ranges at all. That means running it is never a gamble. It cannot silently upgrade you, because upgrades only happen when you deliberately re-resolve with `uv add` or `uv lock`. This is the property that makes the lockfile a contract rather than a starting point: the environment is a pure function of the committed lock, and `uv sync` is how you evaluate that function. It is why every chapter can open with `uv sync` and trust exactly what it gets.
```

## Tooling

The uv command surface I actually use is small and worth memorizing, because it recurs in every later chapter.

- `uv init <dir>` creates a new project: a `pyproject.toml`, a pinned Python, and the scaffolding for a `.venv`.
- `uv python pin <version>` records which Python this project uses, writing `.python-version`.
- `uv add <pkg>` adds a dependency, resolves it, updates `pyproject.toml`, and updates `uv.lock`.
- `uv sync` materializes the `.venv` to exactly match `uv.lock` (installing, removing, or downgrading as needed).
- `uv lock` re-resolves and rewrites `uv.lock` from `pyproject.toml` without changing the running environment.
- `uv run <cmd>` runs a command inside the project's environment, syncing first if needed, so you never manually `source .venv/bin/activate`.

The habit that makes this reproducible: I never install into an environment imperatively and hope. I declare dependencies in `pyproject.toml` (via `uv add`), commit the resulting `uv.lock`, and reconstruct anywhere with `uv sync`. `uv run` is how every script in this book is invoked, so the environment is always the locked one, never whatever happens to be on `PATH`.

```admonish read-along title="Why `uv run`, always"
Activating a virtual environment by hand is a stateful, forgettable act - the classic bug is running a script against the wrong `.venv` because you forgot which one was active. `uv run python train.py` binds the command to *this project's* locked environment every time, with no activation step to forget. Throughout the book, whenever you see a Python invocation, it is `uv run ...` for exactly this reason: the environment is a property of the project directory, not of your shell's mutable state.
```

Two extra flags matter for this hardware. Some GPU packages publish CUDA-specific wheels on their own package indexes rather than only on PyPI, so a project may declare an extra index (for the CUDA-matched PyTorch build, for instance) in `pyproject.toml` under a `[tool.uv]` section. And uv can pin the Python version per project, which matters because vLLM and Unsloth may support different Python version windows; each project pins its own.

## Lab

I create both projects, lock vLLM in `serve/` and Unsloth in `train/`, and commit both `pyproject.toml` + `uv.lock` pairs. The artifact is those two pairs, printed in full. The exact resolved versions are recorded from the lock on the machine; the `pyproject.toml` contents below are realistic declarations, and the resulting `uv.lock` files (large, hash-bearing, machine-generated) are committed but not reprinted here in full.

### Step 1 - repository layout

```text title="repo layout - two uv projects side by side"
thesis-loop/
├── serve/            # vLLM serving environment
│   ├── pyproject.toml
│   ├── uv.lock       # committed
│   └── .python-version
├── train/            # Unsloth training environment
│   ├── pyproject.toml
│   ├── uv.lock       # committed
│   └── .python-version
└── README.md
```

### Step 2 - create and lock the serve environment

```bash title="scripts/setup-serve.sh"
#!/usr/bin/env bash
# Create the serving environment: vLLM, locked.
set -euo pipefail

uv init serve
cd serve

# Pin a Python the serving stack supports (record the exact value on the machine).
uv python pin 3.12

# Add vLLM. uv resolves the full graph (torch, CUDA-linked libs, etc.) and writes uv.lock.
uv add vllm

# Add the light client bits used to poke the endpoint during bring-up.
uv add httpx

# Materialize the environment from the lock and smoke-test the import.
uv sync
uv run python -c "import vllm; print('vllm', vllm.__version__)"
```

```toml title="serve/pyproject.toml (realistic; exact versions resolved into uv.lock on the machine)"
[project]
name = "serve"
version = "0.1.0"
description = "vLLM serving environment for the evals-as-rewards loop"
requires-python = ">=3.12,<3.13"
dependencies = [
    "vllm>=0.8",     # Blackwell-capable serving engine; exact pin lands in uv.lock
    "httpx>=0.27",   # for poking the OpenAI-compatible endpoint during bring-up
]

[tool.uv]
# vLLM pulls a CUDA-matched torch build; if a dedicated index is needed for the
# Blackwell (sm_120) wheels, declare it here so resolution is reproducible.
# index-url / extra-index-url are recorded on the machine once the exact build is chosen.
```

```text title="serve/.python-version"
3.12
```

The generated `serve/uv.lock` is a large machine-written file that pins every transitive dependency with an exact version and a content hash (vLLM, torch, the CUDA-linked libraries, and everything they pull). It is committed verbatim. Its exact contents, including the resolved vLLM and torch versions, are recorded from the lock on the machine; record the resolved versions, date, and driver alongside it.

### Step 3 - create and lock the train environment

```bash title="scripts/setup-train.sh"
#!/usr/bin/env bash
# Create the training environment: Unsloth (+ the GRPO/LoRA stack), locked.
set -euo pipefail

uv init train
cd train

# Pin a Python the training stack supports (record the exact value on the machine).
uv python pin 3.12

# Add Unsloth and the post-training constellation. uv resolves torch/transformers/TRL/peft/bnb.
uv add unsloth
uv add trl peft bitsandbytes datasets

# Materialize and smoke-test.
uv sync
uv run python -c "import unsloth, trl, peft; print('unsloth stack imported OK')"
```

```toml title="train/pyproject.toml (realistic; exact versions resolved into uv.lock on the machine)"
[project]
name = "train"
version = "0.1.0"
description = "Unsloth training environment (GRPO/LoRA) for the evals-as-rewards loop"
requires-python = ">=3.12,<3.13"
dependencies = [
    "unsloth>=2025.1",   # 16GB-friendly training; exact pin lands in uv.lock
    "trl>=0.12",         # GRPO trainer lives here
    "peft>=0.13",        # LoRA/QLoRA adapters
    "bitsandbytes>=0.44",# 4-bit quantization for QLoRA
    "datasets>=3.0",     # dataset loading
]

[tool.uv]
# Unsloth pins a CUDA-matched torch; if a dedicated index is needed for the
# Blackwell (sm_120) wheels, declare it here. Recorded on the machine.
```

```text title="train/.python-version"
3.12
```

As with `serve/`, the generated `train/uv.lock` pins the entire graph (Unsloth, torch, transformers, TRL, peft, bitsandbytes, datasets, and all transitive dependencies) with exact versions and hashes, and is committed verbatim. The resolved versions are recorded from the lock on the machine; record the resolved versions, date, and driver.

### Step 4 - commit both pairs

```bash title="bash - commit the two locked projects"
cd thesis-loop
git add serve/pyproject.toml serve/uv.lock serve/.python-version
git add train/pyproject.toml train/uv.lock train/.python-version
git commit -m "Two locked uv projects: serve (vLLM) and train (Unsloth)"

# Record the resolved headline versions into the machine log for quick reference.
echo "serve: $(cd serve && uv run python -c 'import vllm; print(vllm.__version__)')" >> ~/machine-log/env-versions.txt
echo "train: $(cd train && uv run python -c 'import unsloth,trl,peft; print(\"unsloth+trl+peft OK\")')" >> ~/machine-log/env-versions.txt
```

**What you should see.** Two sibling directories, each with a `pyproject.toml` you wrote, a `.python-version` pinning the interpreter, and a `uv.lock` that uv generated and that you did not hand-edit. In `serve/`, `uv run python -c "import vllm; print(vllm.__version__)"` prints a version and exits cleanly. In `train/`, the Unsloth stack imports without error. Neither environment mentions the other's heavyweight packages: `serve/uv.lock` has no Unsloth, `train/uv.lock` has no vLLM, which is the whole point. The committed lockfiles are the artifact. From now on, any machine that runs `uv sync` in `serve/` gets byte-identical serving dependencies, and the same for `train/`, so a reasoning-delta result can never be quietly undermined by a dependency that drifted between the serve step and the train step. The exact resolved versions live in the lockfiles and in `~/machine-log/env-versions.txt`; record them with the date and driver in effect.

```admonish thesis-thread
For the committee, the two committed `uv.lock` files are the second pillar of run provenance (the first was the driver/link capture in Chapter 0.3). Chapter 0.6 will hash these lockfiles and log the hash with every run, so that a figure in Part 9 can be traced not just to a git SHA of the code but to the exact resolved dependency graph of both the serving and training stacks. The two-environment doctrine is what makes that hash meaningful: because serve and train are locked independently, the provenance record can attribute a change to the serving stack or the training stack specifically, rather than to an undifferentiated blob of "the environment."
```

```admonish substack-seed
The counterintuitive, shareable claim is "keep your inference and your training in separate environments on purpose, even on one machine." Most people's instinct is to build one big env with everything in it, and then spend an afternoon fighting a resolver or debugging a CUDA kernel mismatch that turns out to be two libraries disagreeing about which PyTorch to pin. The clean frame: serving and training are two dependency universes that happen to share a GPU in time but should never share a Python interpreter, and the seam between them is an HTTP endpoint, not an import. Add the reproducibility punchline - a committed lockfile is the difference between "what I asked for" and "what I actually got" - and you have a practical post that will save readers a specific, memorable headache.
```
