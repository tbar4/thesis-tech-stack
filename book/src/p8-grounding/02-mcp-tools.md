# MCP tools for live SDA data

```admonish note
Status: drafting pending. Design spec:
`docs/superpowers/specs/2026-07-22-space-data-grounding-design.md` (chapter 8.2).
This stub exists so the book builds under `create-missing = false`.
```

**Goal.** Give the model live numerical SDA data (conjunction / catalog / orbital
queries) via MCP tools at inference, with verifiable tool-use.

**Covers.** A FastMCP server wrapping the 3.9 clients + Skyfield oracle; transports
(stdio / streamable-HTTP); the model as MCP client via vLLM tool-calling; wiring
tools into Inspect for the agentic eval (3.4); verifiable tool-use; caching and the
open-data boundary (tools fetch live, never redistribute).

## Theory

## Tooling

## Lab

```admonish substack-seed
Extractable post angle for this chapter, one paragraph, to be drafted.
```
