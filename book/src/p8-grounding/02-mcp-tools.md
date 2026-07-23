# MCP tools for live SDA data

The last chapter (8.1) retrieved *text* the model could not compute: news, NASA writeups, the qualitative context a language model has no way to derive from its weights. This chapter does the mirror-image job for *numbers*. A conjunction screen is not a fact to look up; it is a computation on data that is both live and, in the authoritative case, un-shippable. The two constraints point at the same architecture. Live, because a two-line element set (TLE) is only good for a few days around its epoch and the question I care about ("do these two objects pass within 5 km in the next 24 hours?") is a question about *now*, so any answer baked into a shipped dataset is stale the day I ship it. Un-shippable, because the authoritative catalog behind that question comes from space-track, and the 3.9 open-data boundary is unambiguous: space-track data is fetch-only, redistribution-restricted, never committed to the repo. I cannot turn it into a dataset even if I wanted to. Both constraints resolve to the same conclusion: the live numerical side of Space Domain Awareness belongs behind a *tool call the model makes at inference*, not behind a file I distribute. This chapter builds that tool surface as an MCP server over the 3.9 source clients and the 3.10 Skyfield oracle, drives it two ways (a thin vLLM tool-call loop and an Inspect eval), and scores the result with *verifiable tool-use*: because the tool returns oracle ground truth, I can check both that the model called it and that it used the number correctly, which is a stronger claim than grading the final answer alone.

## Theory

### Retrieval is for what you cannot compute; tools are for what you must not freeze

RAG (8.1) and tools (8.2) look like the same idea (give the model something at inference it did not have in its weights) but they answer different failure modes and the distinction is worth keeping sharp. Retrieval addresses *ignorance*: the model does not know that a particular satellite maneuvered last week, and no amount of reasoning recovers a fact it never saw, so I fetch the text. Tools address *computation and freshness*: the model could in principle reason about orbital mechanics, but it cannot propagate an SGP4 trajectory in its head to 6-digit precision, and even if it could, the *inputs* (current TLEs, the current catalog) are live and restricted. So the split is not "retrieval for hard things, tools for easy things"; it is "retrieval for what you cannot derive, tools for what you can derive but must not stale-freeze into a shipped artifact." Conjunction screening sits squarely on the tool side of that line, which is why it is the flagship here and not in 8.1.

### The open-data boundary is the reason, not a footnote

The 3.9 boundary table is the load-bearing argument of this chapter. celestrak and the spacedevs feeds are permissively licensed, so 8.1 can ship a snapshot of their text. space-track is not: an account login, strict rate limits, and terms that forbid redistribution. That single row forces the design. I cannot precompute a conjunction dataset from the authoritative catalog and hand it out, because the intermediate TLEs are exactly the thing I am not allowed to redistribute. What I *can* distribute is the *derived* answer computed by the oracle at eval time, and the *code* that fetches and computes. So the authoritative catalog lives behind a tool the model calls, the tool fetches under my credentials at run time, and nothing restricted is ever written to disk in the public artifact. The boundary that looked like a licensing annoyance in 3.9 is the thing that makes "put it behind a tool" the correct answer rather than a stylistic choice. This is a thesis beat, not plumbing: the reproducibility package (Part X) ships the server and the derived task instances, never the raw catalog.

### MCP: a typed tool contract, not bespoke glue

I could hand-wire each tool into each caller as ad-hoc function-calling JSON, but I would write that glue twice (once for the vLLM loop, once for Inspect) and keep the schemas in sync by hand. The Model Context Protocol (MCP) is the standard that removes that duplication: I declare each tool *once*, as a typed Python function on a server, and the protocol exposes its name, its JSON-schema signature, and its transport to any client. The official `mcp` SDK ships FastMCP, which derives the tool schema straight from the function's type hints and docstring, so the typed signature `screen_conjunction(tle_a: str, tle_b: str, window_h: int, threshold_km: float) -> dict` *is* the contract the model sees. One declaration, two consumers: the thin vLLM loop and the Inspect eval both talk to the same server. That is the whole reason to reach for MCP here rather than a dictionary of Python callables: the tools become a reusable, transport-agnostic surface that Part IX's burst work and Part X's assembly can point at without re-deriving the schemas.

### Verifiable tool-use: check the call and the use, not just the answer

Chapter 3.4 drew the line between a scorer with no opinion (verifiable) and a scorer with opinions (model-graded), and insisted on the former wherever the construct allows. Tool-use lets me push that discipline one notch stronger. An answer-only scorer checks that the final number is right; but a reasoning model can be right for the wrong reason (a lucky guess, a memorized answer, a contamination leak from 3.8). Because my tool returns *oracle ground truth* (the same Skyfield/sgp4 code path from 3.10 that generates the gold answer), I can inspect the transcript and verify three things: that the model *called* `screen_conjunction`, that it called it with the *right arguments*, and that its final answer is *consistent with the number the tool returned*. That is a claim about the model's *process*, not just its output, and it is exactly the property 8.3 needs to say "the +tools arm's gain is tool access being used correctly, not a lookup coincidence." Verifiable tool-use is the answer-verifiability of 3.4 applied to the trajectory through the tool, and it is only possible because the tool itself is grounded in the oracle.

### Transports: stdio for local, streamable-HTTP for the serving host

MCP separates the tool logic from how the client reaches it. For local development and for the Inspect eval running on the same box as the server, **stdio** is the right transport: the client launches the server as a subprocess and talks over its stdin/stdout, no ports, no network, no auth surface. For the case where the tool server lives on the serving host and the caller is elsewhere (or where I want one long-lived server many clients share), MCP offers **streamable-HTTP**: the server binds a port and speaks the protocol over HTTP with server-sent-event streaming. FastMCP exposes both from the same tool definitions with a one-word change at launch (`mcp.run()` versus `mcp.run(transport="streamable-http")`), so I write the tools once and choose the transport per deployment. On the single 16GB node the default is stdio (the server is a light IO-bound CPU process co-resident with the GPU-bound vLLM server); streamable-HTTP is the option I reach for only when a second machine enters the picture in Part IX.

```admonish gotcha title="Live feeds break reproducibility, and auth breaks unattended runs; pin a snapshot"
A tool that hits a live feed makes the eval non-reproducible by construction: the same task run today and next week screens against *different* TLEs, so the gold answer moves and two runs are no longer comparable, which is the exact comparability failure chapter 5 warns about. Two disciplines fix it. First, **snapshot-pinning**: the server runs in one of two modes. In `pinned` mode every tool reads from a DVC-pinned 3.9 snapshot (a content hash), so the eval is frozen and re-runnable forever; in `current` mode it hits the live feed, which I use only when the question is genuinely "what is happening now." An eval that goes in a table runs pinned, always. Second, **auth and rate-limit discipline**: space-track needs credentials (kept in the environment, never in the repo per the 3.9 boundary) and enforces strict rate limits, so the tool caches aggressively (a fetched TLE is valid for its epoch window) and a naive unattended run that re-fetches per sample will get throttled or locked out. Cache by NORAD id plus epoch, back off on 429, and prefer pinned mode for anything that repeats. The live path is for the demo and the "current" arm; the pinned path is for every measurement.
```

## Tooling

### The `mcp/` uv sub-project

Per the locked stack, the FastMCP server and its source clients live in their own `mcp/` uv sub-project (light, IO-bound, no GPU), separate from `data/` and `serve/`. It depends on the official SDK, `httpx` and `spacetrack` for the 3.9 fetch clients, and Skyfield/sgp4 for the 3.10 oracle.

```bash title="setup.sh"
cd ~/thesis-tech-stack
uv init mcp && cd mcp
uv add "mcp[cli]" httpx spacetrack skyfield sgp4
```

I keep the actual data-and-physics logic in a plain module, `sda_core.py`, that imports the 3.9 clients and the 3.10 oracle. The MCP server is then a thin typed façade over it, which matters for testing (I can exercise `sda_core` with no protocol in the loop) and for Inspect (which can call the same core in-process). The core is deliberately boring: fetch, or read the pinned snapshot, then hand off to the oracle.

```python title="mcp/sda_core.py"
"""Data + physics core the MCP tools wrap. No protocol here on purpose:
this is importable by the FastMCP server, the vLLM loop, and Inspect alike.

Two modes (see the snapshot-pinning gotcha):
  SDA_MODE=pinned  -> read TLEs from a DVC-pinned 3.9 snapshot (reproducible)
  SDA_MODE=current -> fetch live (space-track/celestrak); only for "now" queries
"""
from __future__ import annotations

import os
from functools import lru_cache

# 3.9 source clients (httpx + spacetrack) and the 3.10 Skyfield/sgp4 oracle.
from sda_pipeline.clients import fetch_tle_live, catalog_search        # 3.9
from sda_pipeline.snapshot import read_tle_pinned, catalog_search_pinned
from sda_oracle import propagate_state, closest_approach                # 3.10

_MODE = os.environ.get("SDA_MODE", "pinned")
_SNAPSHOT = os.environ.get("SDA_SNAPSHOT", "")   # DVC content hash when pinned


@lru_cache(maxsize=4096)
def get_tle(norad_id: int) -> dict:
    """Return the current TLE for a NORAD id as {name, line1, line2, epoch}.

    Cached by norad_id; in pinned mode the snapshot hash makes the cache exact.
    """
    if _MODE == "pinned":
        return read_tle_pinned(norad_id, snapshot=_SNAPSHOT)
    return fetch_tle_live(norad_id)              # auth'd, rate-limited (3.9)


def catalog_lookup(query: str, limit: int = 10) -> list[dict]:
    """Search the object catalog by name or designator; list of catalog rows."""
    if _MODE == "pinned":
        return catalog_search_pinned(query, limit=limit, snapshot=_SNAPSHOT)
    return catalog_search(query, limit=limit)


def propagate(tle: dict, t_iso: str) -> dict:
    """Propagate a TLE to ISO-8601 UTC time t; return ECI position/velocity km."""
    return propagate_state(tle["line1"], tle["line2"], t_iso)   # sgp4 via 3.10


def screen_conjunction(
    tle_a: dict, tle_b: dict, window_h: int = 24, threshold_km: float = 5.0
) -> dict:
    """Screen two objects for close approach over the next window_h hours.

    Ground truth from the 3.10 oracle: returns time of closest approach (TCA),
    miss distance (km), and whether it breaches threshold_km. Deterministic.
    """
    tca_iso, miss_km = closest_approach(
        tle_a["line1"], tle_a["line2"], tle_b["line1"], tle_b["line2"],
        window_h=window_h,
    )
    return {
        "tca_utc": tca_iso,
        "miss_km": round(miss_km, 3),
        "threshold_km": threshold_km,
        "conjunction": miss_km <= threshold_km,
    }
```

### The FastMCP server: typed tools, two transports

The server is the typed façade. `FastMCP` derives each tool's JSON schema from the type hints and the docstring, so the four functions below become four MCP tools whose contracts the model sees verbatim. Note the tools take NORAD ids and let the server fetch the TLEs, so the model never has to carry a raw element set around (which it would get wrong); `screen_conjunction` is exposed both as an id-pair convenience and, for the eval, as a raw-TLE form.

```python title="mcp/server.py"
"""FastMCP server exposing typed SDA tools over the 3.9 clients + 3.10 oracle.

Run locally over stdio (default):        uv run mcp/server.py
Run over streamable-HTTP on the host:     uv run mcp/server.py --http
"""
from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

import sda_core

mcp = FastMCP("sda-tools")


@mcp.tool()
def get_tle(norad_id: int) -> dict:
    """Fetch the two-line element set for a catalog object by NORAD id.

    Returns {name, line1, line2, epoch}. In pinned mode this reads the
    reproducible 3.9 snapshot; in current mode it fetches live.
    """
    return sda_core.get_tle(norad_id)


@mcp.tool()
def catalog_lookup(query: str, limit: int = 10) -> list[dict]:
    """Search the object catalog by name or international designator."""
    return sda_core.catalog_lookup(query, limit=limit)


@mcp.tool()
def propagate(norad_id: int, t_utc: str) -> dict:
    """Propagate an object to ISO-8601 UTC time t_utc.

    Returns ECI position (km) and velocity (km/s) from the SGP4 oracle.
    """
    return sda_core.propagate(sda_core.get_tle(norad_id), t_utc)


@mcp.tool()
def screen_conjunction(
    norad_id_a: int, norad_id_b: int, window_h: int = 24, threshold_km: float = 5.0
) -> dict:
    """Screen two catalog objects for a close approach in the next window_h hours.

    Returns time of closest approach (TCA, UTC), miss distance (km), the
    threshold used, and whether a conjunction is flagged. Ground truth from the
    3.10 Skyfield/sgp4 oracle: deterministic for a fixed snapshot.
    """
    tle_a = sda_core.get_tle(norad_id_a)
    tle_b = sda_core.get_tle(norad_id_b)
    return sda_core.screen_conjunction(
        tle_a, tle_b, window_h=window_h, threshold_km=threshold_km
    )


if __name__ == "__main__":
    if "--http" in sys.argv:
        # Streamable-HTTP for the serving host / multi-client use (8.2 theory).
        mcp.run(transport="streamable-http")
    else:
        # Stdio for local dev and the co-resident Inspect eval (the default).
        mcp.run()
```

For the model to *call* these tools, the vLLM server (Part II) has to be launched with tool-calling enabled and a parser that matches the model's tool syntax. For Qwen3 that is the Hermes parser.

```bash title="serve/qwen3-8b-tools.sh"
# vLLM must be told to emit and parse tool calls (extends the 05-vllm-ops runbook).
uv run vllm serve Qwen/Qwen3-8B --quantization fp8 \
    --served-model-name qwen3-8b --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes
```

### The model as MCP client: a thin vLLM tool-call loop

The "model as MCP client" is a small loop, and keeping it small is the point (it is the same transparency doctrine as the hand-rolled RAG loop in 8.1). The loop connects to the server over stdio, translates the MCP tool list into the OpenAI `tools` schema vLLM expects, then runs the standard tool-call cycle: send the messages plus tools, and if the model responds with `tool_calls`, dispatch each to the MCP session, append the results as `tool` messages, and call again until the model answers with content instead of a call.

```python title="mcp/loop.py"
"""Thin vLLM + MCP tool-call loop: the model as MCP client (8.2 theory).

Connects to server.py over stdio, exposes its tools to a vLLM-served model via
OpenAI-compatible tool-calling, and runs the dispatch cycle until the model
stops calling tools and answers.
"""
from __future__ import annotations

import asyncio
import json

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # vLLM


def _to_openai_tools(mcp_tools) -> list[dict]:
    """MCP tool list -> OpenAI function-tool schema. One contract, no hand-sync."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,   # FastMCP derived this from hints
            },
        }
        for t in mcp_tools
    ]


async def run(question: str, model: str = "qwen3-8b", max_turns: int = 6) -> str:
    params = StdioServerParameters(command="uv", args=["run", "mcp/server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = _to_openai_tools((await session.list_tools()).tools)

            messages = [
                {"role": "system", "content":
                 "You are an SDA analyst. Use the tools for any orbital number; "
                 "never guess a TLE or a miss distance. State the miss distance "
                 "and TCA the tool returns, then answer yes/no."},
                {"role": "user", "content": question},
            ]

            for _ in range(max_turns):
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=tools, temperature=0.0,
                )
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return msg.content            # model answered; done

                messages.append(msg)              # the assistant tool-call turn
                for call in msg.tool_calls:
                    args = json.loads(call.function.arguments)
                    result = await session.call_tool(call.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content[0].text,   # the oracle number
                    })
            return "tool-call budget exhausted"


if __name__ == "__main__":
    q = ("Do NORAD 25544 (ISS) and 48274 pass within 5 km in the next 24 hours? "
         "Screen them and answer with the miss distance and yes/no.")
    print(asyncio.run(run(q)))
```

```admonish under-the-hood title="The tool-call control flow, message by message"
The loop above is four messages and one branch. For the conjunction query, the transcript vLLM sees and produces is exactly this sequence, and reading it once removes all the mystery from "how does the model call a tool":

    1. system  : "You are an SDA analyst. Use the tools for any orbital number..."
       user    : "Do 25544 and 48274 pass within 5 km in the next 24h? ..."
                     |
                     v   client.chat.completions.create(messages, tools=[...])
    2. assistant: content=None, tool_calls=[
                     { id: "call_0", function: {
                         name: "screen_conjunction",
                         arguments: '{"norad_id_a":25544,"norad_id_b":48274,
                                      "window_h":24,"threshold_km":5.0}' } } ]
                     |
                     v   session.call_tool("screen_conjunction", {...})
                     |     -> sda_core -> 3.9 get_tle x2 -> 3.10 oracle
    3. tool     : tool_call_id="call_0",
                  content='{"tca_utc":"2026-07-23T18:41:07Z","miss_km":3.812,
                            "threshold_km":5.0,"conjunction":true}'
                     |
                     v   client.chat.completions.create(messages + tool result)
    4. assistant: content="The oracle screen gives a closest approach of
                  3.812 km at 2026-07-23T18:41:07Z, which is inside the 5 km
                  threshold. Yes, they conjunct."   (no tool_calls -> loop ends)

Message 2 is the model deciding *to call* and with *what arguments*; message 3 is the tool returning *oracle ground truth*, not the model's opinion; message 4 is the model *reasoning over the returned number*. The verifiable-tool-use scorer reads exactly this transcript: it confirms message 2 exists (the model called the tool), that its arguments name the right objects, and that message 4's stated `miss_km` matches message 3's. That is why grounding the answer in a tool result is a stronger check than grading message 4 alone: I can see the number came from the oracle and was carried through, not conjured.
```

### Wiring the tools into Inspect for the eval

The vLLM loop is good for a demo and for the deployed inference path, but the *eval* wants Inspect (3.4), which already owns agentic tool-use, sandboxing, and the log-as-audit-object. Inspect calls tools through its own `@tool` abstraction and the `use_tools` solver, so I wrap the same `sda_core` functions as Inspect tools. Because `sda_core` is protocol-free, this is a thin in-process wrapper (no subprocess, no transport), which is the fastest and most reproducible option for a batch eval on one box.

```python title="mcp/sda_inspect_tools.py"
"""Inspect tool wrappers over sda_core (same functions the MCP server exposes).

For the batch eval, calling sda_core in-process is simplest and most
reproducible. To instead drive the *running* MCP server, Inspect can bridge to
it with inspect_ai.tool.mcp_server_stdio (see the note below).
"""
from __future__ import annotations

from inspect_ai.tool import tool

import sda_core


@tool
def screen_conjunction():
    async def execute(
        norad_id_a: int, norad_id_b: int, window_h: int = 24, threshold_km: float = 5.0
    ) -> str:
        """Screen two catalog objects for a close approach in the next window_h hours.

        Args:
            norad_id_a: NORAD id of the first object.
            norad_id_b: NORAD id of the second object.
            window_h: screening horizon in hours.
            threshold_km: conjunction threshold in km.

        Returns:
            JSON with tca_utc, miss_km, threshold_km, conjunction (oracle truth).
        """
        import json
        a, b = sda_core.get_tle(norad_id_a), sda_core.get_tle(norad_id_b)
        r = sda_core.screen_conjunction(a, b, window_h=window_h, threshold_km=threshold_km)
        return json.dumps(r)

    return execute
```

To point Inspect at the *live protocol server* instead of the in-process core (useful when the tools run on another host over streamable-HTTP), Inspect provides an MCP bridge: `from inspect_ai.tool import mcp_server_stdio` (or `mcp_server_http`) builds a tool source from the same `server.py`, and it drops into `use_tools` exactly like the native tool above. I use the in-process wrapper for the reproducible batch eval and keep the bridge for the "current" live demo.

## Lab

The artifact is two things that share one core: the **FastMCP server** (`mcp/server.py` over `sda_core.py`) and a **tool-augmented conjunction-screening eval** wired through Inspect with a verifiable-tool-use scorer, run against a pinned snapshot for reproducibility. The eval is the payload: it proves the model calls the tool for real orbital data and then reasons over the returned number, and it produces the log 8.3 consumes as the +tools arm.

Lay it out under the `mcp/` sub-project:

```
mcp/
  pyproject.toml
  uv.lock
  sda_core.py                # data + physics core (above)
  server.py                  # FastMCP server (above)
  loop.py                    # thin vLLM MCP client loop (above)
  sda_inspect_tools.py       # Inspect tool wrappers (above)
  conjunction_task.py        # the eval task + verifiable-tool-use scorer
  data/conjunction_screen.jsonl
```

The task dataset is a small set of screening questions whose gold answers are the oracle's, generated from a pinned 3.9 snapshot by the 3.10 generator. Each record carries the two NORAD ids, the pinned gold verdict, and the gold miss distance, so the scorer can check the model's *use* of the tool result against ground truth.

```json title="mcp/data/conjunction_screen.jsonl"
{"norad_a": 25544, "norad_b": 48274, "window_h": 24, "threshold_km": 5.0, "gold_conjunction": true,  "gold_miss_km": 3.812}
{"norad_a": 43013, "norad_b": 44714, "window_h": 24, "threshold_km": 5.0, "gold_conjunction": false, "gold_miss_km": 41.6}
{"norad_a": 33591, "norad_b": 39084, "window_h": 24, "threshold_km": 5.0, "gold_conjunction": false, "gold_miss_km": 118.2}
```

The task uses the tool-wrapping solver and a scorer that grades *both* halves of verifiable tool-use. The scorer is programmatic and factored (extraction then verification, per 3.4): it reads the transcript for the tool call and reads the final answer for the model's verdict, then scores correct only when the model *called the tool with the right objects* and its *verdict matches the oracle*.

```python title="mcp/conjunction_task.py"
"""Tool-augmented conjunction-screening eval with verifiable-tool-use scoring.

Reproducible: runs against a pinned snapshot (SDA_MODE=pinned, SDA_SNAPSHOT=<hash>).
The scorer grades BOTH that the model called screen_conjunction on the right
objects AND that its verdict matches the 3.10 oracle ground truth.
"""
from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message, use_tools

from sda_inspect_tools import screen_conjunction

SYSTEM = (
    "You are an SDA analyst. For any orbital number you MUST call the "
    "screen_conjunction tool; never guess a miss distance. After the tool "
    "returns, state the miss distance it gave, then end with exactly "
    "'VERDICT: YES' or 'VERDICT: NO' for whether a conjunction is flagged."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def record_to_sample(r: dict) -> Sample:
    return Sample(
        input=(f"Do NORAD {r['norad_a']} and {r['norad_b']} pass within "
               f"{r['threshold_km']} km in the next {r['window_h']} hours?"),
        target="YES" if r["gold_conjunction"] else "NO",
        metadata={"norad_a": r["norad_a"], "norad_b": r["norad_b"],
                  "gold_miss_km": r["gold_miss_km"]},
    )


def _called_tool_on(state: TaskState, a: int, b: int) -> bool:
    """True if the transcript contains a screen_conjunction call naming a and b."""
    ids = {a, b}
    for m in state.messages:
        for call in (getattr(m, "tool_calls", None) or []):
            if call.function == "screen_conjunction":
                args = call.arguments or {}
                if {args.get("norad_id_a"), args.get("norad_id_b")} == ids:
                    return True
    return False


@scorer(metrics=[accuracy(), stderr()])
def verifiable_tool_use():
    async def score(state: TaskState, target: Target) -> Score:
        m = _VERDICT_RE.search(state.output.completion or "")
        verdict = m.group(1).upper() if m else None
        called = _called_tool_on(
            state, state.metadata["norad_a"], state.metadata["norad_b"]
        )
        if not called:
            return Score(value=INCORRECT, answer=verdict,
                         explanation="did not call screen_conjunction on the pair")
        if verdict is None:
            return Score(value=INCORRECT, answer=None,
                         explanation="tool called but no parseable VERDICT")
        ok = verdict == target.text.upper()
        return Score(
            value=CORRECT if ok else INCORRECT, answer=verdict,
            explanation=("tool called + verdict matches oracle" if ok
                         else f"tool called but verdict {verdict} != gold {target.text}"),
            metadata={"tool_called": True, "gold_miss_km": state.metadata["gold_miss_km"]},
        )
    return score


@task
def conjunction_screen() -> Task:
    return Task(
        dataset=json_dataset("data/conjunction_screen.jsonl", sample_fields=record_to_sample),
        solver=[system_message(SYSTEM), use_tools(screen_conjunction()), generate()],
        scorer=verifiable_tool_use(),
    )
```

Run it against a pinned snapshot so the eval is reproducible, with vLLM launched in tool-calling mode. Pinned mode is the only mode that goes in a table.

```bash title="shell: run the tool-augmented eval (pinned)"
# 1) vLLM up with tool-calling (see the Tooling serve command)
# 2) pin the snapshot so the oracle ground truth is frozen and re-runnable
export SDA_MODE=pinned
export SDA_SNAPSHOT=$(dvc get-url .  # the 3.9 snapshot content hash, e.g. a1b9f3c...)

# 3) run the tool-augmented eval through Inspect
uv run inspect eval conjunction_task.py@conjunction_screen \
    --model openai/qwen3-8b --log-dir logs

# 4) inspect the transcripts: the tool call, the oracle result, the verdict
uv run inspect view --log-dir logs
```

To see the same tools drive the thin vLLM loop instead of Inspect (the deployed inference path, and the "current" live mode), run the loop directly:

```bash title="shell: run the live tool loop"
# live "current" mode: hits the feed under my space-track credentials
SDA_MODE=current uv run mcp/loop.py
```

**What you should see.** The Inspect run prints `accuracy` and `stderr` as usual, but the log now carries the whole tool trajectory per sample: open `inspect view` and each sample shows the assistant's `screen_conjunction` call with its arguments, the `tool` message with the oracle's `miss_km` and `tca_utc`, and the model's final `VERDICT:` line reasoning over that number. The scorer's `explanation` distinguishes three failure modes that an answer-only scorer collapses into one: "did not call screen_conjunction" (the model guessed instead of using the tool), "tool called but no parseable VERDICT" (a format miss), and "tool called but verdict X != gold" (used the tool but reasoned over its output wrong). That three-way split *is* verifiable tool-use: I am grading process, not just outcome. On the pinned snapshot the run is deterministic and re-runnable, so the number is comparable across days; in `current` mode the same task screens against live TLEs and the gold moves, which is exactly why measurements run pinned. On the baseline machine, record the tool-augmented accuracy, the fraction of samples where the model called the tool at all, wall-clock, and tokens/sec, with the snapshot hash, date, and driver version, since this log is the +tools arm 8.3 compares against the parametric and +RAG arms. The artifact is the MCP server (`mcp/server.py` + `sda_core.py`, one typed tool surface over the 3.9 clients and 3.10 oracle, stdio and streamable-HTTP) and the tool-augmented eval (`conjunction_task.py`, verifiable-tool-use scored, snapshot-pinned) whose delta feeds the causal comparison next.

```admonish read-along
**[AIE] Huyen ch. 6 (agents and tools)** is the conceptual backing for this chapter: it frames tool-use as extending a model's action space beyond text generation and lays out the plan-act-observe loop that my thin vLLM loop implements in twenty lines. Read Huyen's treatment of tool selection and failure handling against the verifiable-tool-use scorer here: her agents chapter motivates *why* tools help, and this chapter's oracle-grounded scoring is how I *prove* a specific tool call helped for the right reason rather than by luck, which is the bridge into the causal arms of 8.3.
```

```admonish substack-seed
"Retrieval is for what your model can't know; tools are for what it can't compute or isn't allowed to keep." The cleanest way to decide whether a capability belongs in RAG or in a tool is to ask two questions about the data behind it: can the model derive this by reasoning, and am I allowed to ship it? Text about last week's launch fails the first test, so you retrieve it. A conjunction screen fails both: the model can't propagate an orbit in its head to kilometer precision, and the authoritative catalog it needs is redistribution-restricted, so you can't freeze it into a dataset even if you wanted to, which leaves exactly one option, a live tool call the model makes at inference. The post would use the space-track boundary as the worked example (a licensing constraint that turns out to *dictate* the architecture) and close on the payoff that made it worth the trouble: because the tool returns ground truth from the same physics oracle that writes the answer key, you can grade the model's *process*, confirming it called the tool, called it correctly, and carried the number through, which is a far stronger claim than checking the final answer and hoping it wasn't a lucky guess.
```
