# System architecture

This page is the whole machine on one screen: how live space data becomes verifiable tasks, how those tasks measure a model, how the measurement becomes the reward that trains it, and how the loop closes on a single 16GB GPU. If you read only one page before diving into the parts, read this one. The diagram is the map; the prose under it is the *why*, because an architecture is a set of decisions, and the decisions are more useful than the boxes.

<style>
  /* Self-contained architecture diagram, inlined into the page (no iframe, so
     the content flows with the page and there is no height to sync). All tokens
     and rules are scoped under #arch-root; dark palettes ride mdbook's theme
     classes (coal / navy / ayu) on <html>. */
  #arch-root {
    --panel: #ffffff; --panel-2: #f4f7fc;
    --fg: #1a2236; --muted: #566078; --faint: #8894ab;
    --line: rgba(40, 60, 100, 0.14); --line-strong: rgba(40, 60, 100, 0.26);
    --signal: #1f97bd;   /* orbital cyan: the loop / control flow */
    --verify: #2f9b66;   /* emerald: ground truth, verifiers, oracle */
    --reward: #b9772a;   /* amber: reward, training energy */
    --infra: #566d92;    /* slate: hardware, tracking, storage */
    --product: #b85878;  /* rose: beyond-thesis / product */
    --radius: 12px;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Code", Menlo, Consolas, monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--fg); font-family: var(--sans); line-height: 1.5;
    border: 1px solid var(--line); border-radius: 16px;
    background: var(--panel-2); padding: 30px 26px 40px; margin: 20px 0 8px;
    -webkit-font-smoothing: antialiased;
  }
  html.coal #arch-root, html.navy #arch-root, html.ayu #arch-root {
    --panel: #121724; --panel-2: #171d2c;
    --fg: #e2e8f5; --muted: #8b96ac; --faint: #5c667c;
    --line: rgba(150, 170, 205, 0.16); --line-strong: rgba(150, 170, 205, 0.30);
    --signal: #4cc2e0; --verify: #57c08a; --reward: #e0a55e; --infra: #8aa0c4; --product: #e0849e;
  }

  #arch-root, #arch-root * { box-sizing: border-box; }
  #arch-root code { background: transparent; color: inherit; padding: 0; font-size: inherit; border: 0; font-family: var(--mono); }
  #arch-root p { margin: 0; }

  #arch-root header { position: relative; margin: 0 0 30px; }
  #arch-root .orbit { position: absolute; inset: -18px -8px auto auto; width: 200px; height: 200px; pointer-events: none; opacity: 0.5; }
  #arch-root .eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--signal); margin: 0 0 10px; }
  #arch-root h1 { font-size: clamp(22px, 3.4vw, 32px); line-height: 1.1; margin: 0 0 10px; letter-spacing: -0.015em; font-weight: 650; border: 0; padding: 0; }
  #arch-root h1 .sub { color: var(--muted); font-weight: 450; }
  #arch-root .lede { max-width: 62ch; color: var(--muted); margin: 0; font-size: 15px; }
  #arch-root .legend { display: flex; flex-wrap: wrap; gap: 10px 20px; margin: 16px 0 0; }
  #arch-root .legend span { display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--muted); font-family: var(--mono); }
  #arch-root .dot { width: 10px; height: 10px; border-radius: 3px; flex: none; }

  #arch-root .node {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px;
    margin: 20px 0 28px; padding: 12px 16px; border: 1px solid var(--line);
    border-radius: var(--radius); background: var(--panel);
    font-family: var(--mono); font-size: 12.5px; color: var(--muted);
  }
  #arch-root .node b { color: var(--fg); font-weight: 600; }
  #arch-root .node .spark { color: var(--signal); }

  #arch-root .flow { display: grid; gap: 0; }
  #arch-root .tier { position: relative; display: grid; grid-template-columns: 168px 1fr; gap: 20px; padding: 22px 0; }
  #arch-root .tier + .tier { border-top: 1px dashed var(--line); }
  #arch-root .rail .tag { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--faint); }
  #arch-root .rail .name { font-weight: 640; font-size: 15px; margin-top: 4px; line-height: 1.25; }
  #arch-root .rail .accent { display: block; width: 30px; height: 3px; border-radius: 2px; margin-top: 10px; }

  #arch-root .cards { display: flex; flex-wrap: wrap; gap: 10px; align-content: start; }
  #arch-root .card {
    border: 1px solid var(--line-strong); border-radius: 10px; background: var(--panel);
    padding: 11px 13px; min-width: 150px; flex: 1 1 auto; max-width: 260px;
    transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
  }
  #arch-root .card:hover { transform: translateY(-2px); border-color: var(--tier, var(--signal)); background: var(--panel-2); }
  #arch-root .card h3 { margin: 0; font-size: 13.5px; font-weight: 620; display: flex; align-items: center; gap: 8px; border: 0; padding: 0; }
  #arch-root .card h3 .d { width: 8px; height: 8px; border-radius: 2px; background: var(--tier, var(--signal)); flex: none; }
  #arch-root .card p { margin: 6px 0 0; font-size: 12px; color: var(--muted); }
  #arch-root .card .parts { margin-top: 8px; font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.03em; color: var(--faint); }

  #arch-root .loop { grid-column: 1 / -1; margin: 6px 0 2px; padding: 22px;
    border: 1px solid var(--line-strong); border-radius: 16px;
    background:
      radial-gradient(120% 140% at 50% -20%, color-mix(in srgb, var(--signal) 10%, transparent), transparent 60%),
      var(--panel); }
  #arch-root .loop-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  #arch-root .loop-head .tag { font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--signal); }
  #arch-root .loop-head h2 { margin: 0; font-size: 17px; font-weight: 660; border: 0; padding: 0; }
  #arch-root .loop-head .note { color: var(--muted); font-size: 12.5px; margin-left: auto; font-family: var(--mono); }
  #arch-root .ring { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; align-items: stretch; }
  #arch-root .stage { position: relative; border: 1px solid var(--line-strong); border-radius: 12px; background: var(--panel-2); padding: 14px; }
  #arch-root .stage .n { font-family: var(--mono); font-size: 11px; color: var(--signal); letter-spacing: 0.06em; }
  #arch-root .stage h4 { margin: 4px 0 6px; font-size: 15px; font-weight: 650; border: 0; padding: 0; }
  #arch-root .stage p { margin: 0; font-size: 11.5px; color: var(--muted); }
  #arch-root .stage .tools { margin-top: 9px; font-family: var(--mono); font-size: 10.5px; color: var(--faint); }
  #arch-root .stage::after { content: "\2192"; position: absolute; right: -13px; top: 50%; transform: translateY(-50%); color: var(--signal); font-size: 17px; z-index: 2; }
  #arch-root .stage:last-child::after { content: ""; }
  #arch-root .loop-return { display: flex; align-items: center; gap: 10px; margin-top: 12px; font-family: var(--mono); font-size: 11.5px; color: var(--signal); }
  #arch-root .loop-return .line { flex: 1; height: 1px; background: linear-gradient(90deg, var(--signal), transparent); }
  #arch-root .loop-feeds { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--line); font-size: 11.5px; color: var(--muted); }
  #arch-root .loop-feeds b { color: var(--verify); font-family: var(--mono); font-weight: 600; }
  #arch-root .loop-feeds .r { color: var(--reward); }

  #arch-root .tier.product .card { border-style: dashed; background: transparent; }
  #arch-root .fence { grid-column: 1 / -1; font-family: var(--mono); font-size: 11px; color: var(--product); letter-spacing: 0.04em; margin-bottom: 4px; }

  #arch-root .spine { margin-top: 26px; padding: 18px 20px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel); display: flex; flex-wrap: wrap; gap: 10px 26px; align-items: center; }
  #arch-root .spine .t { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--infra); }
  #arch-root .spine .i { font-size: 12.5px; color: var(--muted); }
  #arch-root .spine .i b { color: var(--fg); font-family: var(--mono); font-weight: 600; }

  #arch-root .metaline { margin-top: 20px; color: var(--faint); font-size: 12px; font-family: var(--mono); }

  @media (max-width: 720px) {
    #arch-root .tier { grid-template-columns: 1fr; gap: 12px; }
    #arch-root .ring { grid-template-columns: 1fr 1fr; }
    #arch-root .stage:nth-child(2)::after { content: ""; }
    #arch-root .stage:nth-child(1)::after, #arch-root .stage:nth-child(3)::after { content: "\2192"; }
    #arch-root .orbit { display: none; }
  }
  @media (prefers-reduced-motion: reduce) { #arch-root .card { transition: none; } }
</style>

<!-- NOTE: this whole <div id="arch-root"> block must contain NO blank lines. In
     CommonMark (mdbook), a blank line ends a raw-HTML block, after which the
     indented markup below is parsed as an indented code block and renders as
     literal source instead of the diagram. Keep it one contiguous block. -->
<div id="arch-root">
  <header>
    <svg class="orbit" viewBox="0 0 200 200" fill="none" aria-hidden="true">
      <ellipse cx="100" cy="100" rx="92" ry="52" stroke="var(--signal)" stroke-width="1" opacity="0.55" transform="rotate(-24 100 100)"/>
      <ellipse cx="100" cy="100" rx="70" ry="88" stroke="var(--verify)" stroke-width="1" opacity="0.4" transform="rotate(18 100 100)"/>
      <circle cx="100" cy="100" r="6" fill="var(--signal)"/>
      <circle cx="176" cy="83" r="3" fill="var(--verify)"/>
    </svg>
    <p class="eyebrow">System architecture</p>
    <h1>Evals as Rewards<span class="sub"> &mdash; a serve, evaluate, score, train loop on one GPU</span></h1>
    <p class="lede">Real space data becomes verifiable tasks; the eval that measures the model becomes
      the reward that trains it; and the whole loop closes on a single 16&nbsp;GB card. This is the
      system end to end, from live feeds to a trained, re-evaluated reasoning model.</p>
    <div class="legend">
      <span><i class="dot" style="background:var(--signal)"></i>control / the loop</span>
      <span><i class="dot" style="background:var(--verify)"></i>ground truth / verifiers</span>
      <span><i class="dot" style="background:var(--reward)"></i>reward / training</span>
      <span><i class="dot" style="background:var(--infra)"></i>infra / tracking</span>
      <span><i class="dot" style="background:var(--product)"></i>beyond the thesis</span>
    </div>
  </header>
  <div class="node">
    <span class="spark">&#9670; one node</span>
    <span><b>MSI Aegis R2</b> &middot; RTX 5080 16&nbsp;GB</span>
    <span><b>Ubuntu 24.04</b> on Blackwell</span>
    <span><b>uv</b> two-environment doctrine</span>
    <span><b>Docker Compose</b> + MLflow spine</span>
    <span>NVMe / NAS storage tiers</span>
  </div>
  <div class="flow">
    <section class="tier" style="--tier:var(--verify)">
      <div class="rail">
        <div class="tag">Part III &middot; 3.9&ndash;3.10</div>
        <div class="name">Ground truth &amp; data</div>
        <span class="accent" style="background:var(--verify)"></span>
      </div>
      <div class="cards">
        <div class="card"><h3><i class="d"></i>Live sources</h3><p>celestrak &middot; space-track &middot; api.nasa.gov &middot; thespacedevs &middot; Spaceflight News</p><div class="parts">httpx &middot; spacetrack</div></div>
        <div class="card"><h3><i class="d"></i>Ingest pipeline</h3><p>Airflow 3 assets, thin DAGs over <code>uv run</code> modules; Pydantic + pandera gates</p><div class="parts">chapter 3.9</div></div>
        <div class="card"><h3><i class="d"></i>Snapshots</h3><p>immutable, content-hashed Parquet; DuckDB queries; DVC-pinned</p><div class="parts">Parquet &middot; DuckDB &middot; DVC</div></div>
        <div class="card"><h3><i class="d"></i>Physics oracle</h3><p>Skyfield / SGP4 generates <em>and</em> grades: conjunction screening, elements, passes</p><div class="parts">chapter 3.10</div></div>
      </div>
    </section>
    <section class="tier" style="--tier:var(--verify)">
      <div class="rail">
        <div class="tag">Part III &middot; 3.11</div>
        <div class="name">The frozen instrument</div>
        <span class="accent" style="background:var(--verify)"></span>
      </div>
      <div class="cards">
        <div class="card" style="max-width:none"><h3><i class="d"></i>Thesis task suite v1.0</h3><p>content-hashed, difficulty-stratified, contamination-scanned, power-sized (about 300 paired items); the fixed instrument every result is measured against</p><div class="parts">tagged suite-v1.0 &middot; datasheet</div></div>
      </div>
    </section>
    <div class="loop">
      <div class="loop-head">
        <span class="tag">&#9670; the loop</span>
        <h2>Serve &#8594; Evaluate &#8594; Score &#8594; Train &#8594; re-evaluate</h2>
        <span class="note">Parts II &middot; III &middot; V &middot; VI &middot; VII</span>
      </div>
      <div class="ring">
        <div class="stage"><div class="n">01 &middot; serve</div><h4>Serve</h4><p>the model under test, on 16&nbsp;GB</p><div class="tools">vLLM &middot; FP8 / AWQ &middot; paged KV</div></div>
        <div class="stage"><div class="n">02 &middot; evaluate</div><h4>Evaluate</h4><p>run the frozen suite; log per-item, per-sample</p><div class="tools">Inspect &middot; lm-eval &middot; judges</div></div>
        <div class="stage"><div class="n">03 &middot; score</div><h4>Score</h4><p>the verifier's verdict, with uncertainty</p><div class="tools">evalstats &middot; bootstrap &middot; McNemar</div></div>
        <div class="stage"><div class="n">04 &middot; train</div><h4>Train</h4><p>the scorer becomes the reward</p><div class="tools">GRPO / RLVR &middot; TRL &middot; Unsloth &middot; LoRA</div></div>
      </div>
      <div class="loop-return"><span>&#8635; re-evaluate the trained checkpoint on the same instrument</span><span class="line"></span><span>chapter 7.6 &middot; the reasoning delta</span></div>
      <div class="loop-feeds">
        <span><b>the suite</b> feeds Evaluate &nbsp;&middot;&nbsp; <b>the oracle verifier</b> feeds Score &nbsp;&middot;&nbsp; <span class="r">the reward core</span> feeds Train &nbsp;&middot;&nbsp; every run logs to the MLflow spine</span>
      </div>
    </div>
    <section class="tier" style="--tier:var(--signal)">
      <div class="rail">
        <div class="tag">Part IV</div>
        <div class="name">Causal validity</div>
        <span class="accent" style="background:var(--signal)"></span>
      </div>
      <div class="cards">
        <div class="card"><h3><i class="d"></i>DAGs &amp; identification</h3><p>backdoor / front-door; confounders, colliders, mediators in eval pipelines</p><div class="parts">4.2&ndash;4.4</div></div>
        <div class="card"><h3><i class="d"></i>Interventions</h3><p>training as <code>do(train)</code>; interrupted time series; placebo &amp; negative controls</p><div class="parts">4.5</div></div>
        <div class="card"><h3><i class="d"></i>Causal audit</h3><p>a standing exhibit the methodology chapter cites verbatim</p><div class="parts">4.6</div></div>
      </div>
    </section>
    <section class="tier" style="--tier:var(--signal)">
      <div class="rail">
        <div class="tag">Part VIII &middot; 8.1&ndash;8.3</div>
        <div class="name">Grounding at inference</div>
        <span class="accent" style="background:var(--signal)"></span>
      </div>
      <div class="cards">
        <div class="card"><h3><i class="d"></i>RAG over space text</h3><p>vLLM-served embeddings, LanceDB index; retrieval metrics vs end-task</p><div class="parts">chapter 8.1</div></div>
        <div class="card"><h3><i class="d"></i>MCP tools</h3><p>FastMCP wraps the oracle + clients; live numerical data, verifiable tool-use</p><div class="parts">chapter 8.2</div></div>
        <div class="card"><h3><i class="d"></i>Augmentation arms</h3><p>base / RL-trained / +RAG / +tools at matched budget; what actually caused the gain</p><div class="parts">chapter 8.3</div></div>
      </div>
    </section>
    <section class="tier" style="--tier:var(--infra)">
      <div class="rail">
        <div class="tag">Parts IX&ndash;X</div>
        <div class="name">Scale &amp; assembly</div>
        <span class="accent" style="background:var(--infra)"></span>
      </div>
      <div class="cards">
        <div class="card"><h3><i class="d"></i>Burst</h3><p>containerize the stack; a Lambda GPU burst when 16&nbsp;GB is not enough</p><div class="parts">Part IX</div></div>
        <div class="card"><h3><i class="d"></i>Assembly</h3><p>logs to figures, the methodology chapter, the reproducibility package</p><div class="parts">Part X</div></div>
        <div class="card"><h3><i class="d"></i>Substack map</h3><p>an editorial calendar; the book is the reservoir, Substack the tap</p><div class="parts">10.4</div></div>
      </div>
    </section>
    <section class="tier product" style="--tier:var(--product)">
      <div class="fence">&#9622; outside the thesis &mdash; forward-looking product work, no thesis claims</div>
      <div class="rail">
        <div class="tag">Part XI</div>
        <div class="name">Beyond the thesis</div>
        <span class="accent" style="background:var(--product)"></span>
      </div>
      <div class="cards">
        <div class="card"><h3><i class="d"></i>Agentic RL</h3><p>train the policy to <em>decide</em> when to call its tools; an autonomous SDA agent</p><div class="parts">chapter 11.1</div></div>
      </div>
    </section>
  </div>
  <div class="spine">
    <span class="t">Tracking spine</span>
    <span class="i"><b>MLflow</b> logs every run: model, config, git SHA, uv lock, snapshot hash, metrics with CIs</span>
    <span class="i"><b>DVC</b> pins every data snapshot; content-addressed &amp; reproducible forever</span>
  </div>
  <p class="metaline">Evals as Rewards &middot; single-node reasoning-model loop &middot; measured numbers recorded on the baseline machine.</p>
</div>


## The one idea the whole thing is built around

The load-bearing claim of this book is an identity: **the function that measures a reasoning model is the same function that can reward it.** A scorer that decides whether an answer is correct, used one way, is an evaluation; used another way, it is a reinforcement-learning reward. Every structural choice on this page descends from taking that identity literally. It is why the evaluation stack (Part III) and the reinforcement-learning stack (Parts V through VII) are not two separate systems that happen to share a repo, but one loop with a shared verifier at its center. Build the scorer once, with enough care to stake a training run on it, and you get the measurement and the reward from the same object.

That is also why the middle band of the diagram, **Serve to Evaluate to Score to Train and back**, is drawn as the centerpiece rather than as one section among many. Everything above it exists to produce a trustworthy thing to measure against; everything below it exists to make the measurement defensible, cheaper, or larger. The loop is the product; the rest is what makes the loop honest.

## Why one GPU is a design input, not a limitation

The constraint at the top of the diagram, one MSI Aegis R2 with a 16GB RTX 5080, is not a footnote about budget. It is the input that forces almost every interesting decision downstream. A 16GB card cannot hold a served model, a training run, and a second judge model at once, so the architecture separates *serving* from *training* into two environments (the uv two-environment doctrine) and moves between them deliberately rather than running them concurrently. It cannot afford to regenerate expensive model outputs every time a verifier changes, so the eval-ops discipline is "generate once, score forever": responses are content-hashed and re-scored, never re-drawn. It cannot fit a full-precision teacher, so quantization (FP8, AWQ, QLoRA) is load-bearing rather than optional. A datacenter architecture would make none of these choices, because it would not have to. The single-node constraint is what gives this system its shape, and it is why the same shape transfers to anyone else working on one card.

## Why the ground truth is real space data

A verifiable reward is only as honest as its answer key, and most answer keys are written by humans, which means they carry human bias, error, and disagreement into the reward. The top tier of the diagram exists to escape that. Space Domain Awareness has a property almost no other domain does: its ground truth is a *law of physics*. Ask whether two objects pass within five kilometers tomorrow, and you do not consult an expert; you propagate both objects with the standard orbital model and read the miss distance off the geometry. The Skyfield/SGP4 oracle in chapter 3.10 both *writes* the gold answer and *grades* the model's answer through the same code path, so the answer key and the grader cannot disagree. That single fact is why the data pipeline (chapter 3.9) and the oracle (chapter 3.10) sit above the loop: they manufacture a stream of questions whose reward is free, exact, and impossible to sweet-talk. The pipeline is real data engineering (Airflow, immutable snapshots, DVC) precisely because a task is only reproducible if the snapshot it was built from is pinned forever.

## Reading the diagram top to bottom

The vertical order is the data flow. **Live sources** are pulled on a schedule and normalized into **immutable snapshots**. The **oracle** turns a snapshot into task instances with machine-checkable answers. Those instances are frozen into **the instrument**, thesis task suite v1.0, sized by a power analysis so it can actually resolve the effect I am claiming and content-hashed so "the suite" is a specific object rather than a description. The instrument feeds the **Evaluate** stage of the loop; the oracle's verifier feeds the **Score** stage; and the reward core built from that same verifier feeds the **Train** stage. The trained checkpoint is re-served and re-evaluated on the identical instrument, and the difference between the pre and post measurements, taken with paired statistics, is the reasoning delta the whole thesis is named after (chapter 7.6).

## Why causal validity is its own layer

A measured delta is not a caused delta. If the trained model scores higher, a committee will ask the obvious rival explanations: did something else move between the two measurements, would any training at all have nudged the number, is the gain an artifact of spending more inference compute. The **Causal validity** band (Part IV) exists so those questions have answers built into the design rather than patched on afterward. It treats training as a `do()` intervention, models the pre/post comparison as an interrupted time series, and runs control arms (a random-reward placebo, a negative-control task) whose only job is to rule out the alternatives. The causal audit is a standing exhibit the methodology chapter cites verbatim, so the claim "this reward caused this reasoning gain" is defensible, not merely asserted.

## Why grounding sits apart from training

Retrieval and tools are the *other* two ways to make a model answer better, and they are rival explanations for any gain. The **Grounding** band (Part VIII) builds both, RAG over the pipeline's text corpus (chapter 8.1) and live MCP tools over its numbers (chapter 8.2), but it deliberately keeps them at inference time and separate from the training loop. The reason is the four-arm comparison in chapter 8.3: base, RL-trained, plus-RAG, and plus-tools, run at a *matched* compute budget so no arm wins just by spending more tokens. Only by holding those augmentations apart can the architecture answer the sharpest question in the book, whether the gain was reasoning the model now carries in its weights, or lookup and tool access it was simply handed. Folding tools into training would confound exactly the thing the comparison is built to separate, which is why that step is fenced off into Part XI.

## The spine, and why reproducibility is structural

Underneath everything runs the tracking spine: **MLflow** logs every run with its model, config, git SHA, uv lock hash, snapshot hash, and metrics-with-intervals, and **DVC** pins every data snapshot by content hash. This is not bookkeeping for its own sake. Because a task instance carries the hash of the snapshot it was built from, and every run records the exact inputs it consumed, any number in the book resolves back to the precise bytes and code that produced it. That is what lets the reproducibility package (chapter 10.3) ship the whole pipeline, and it is what keeps the non-redistributable data (space-track) *out* of the package while still shipping the derived, oracle-computed answers. Reproducibility is drawn as a spine because it holds the rest of the skeleton together.

## The fence at the bottom

The last band, **Beyond the thesis** (Part XI), is drawn with a dashed border on purpose. It is where the research becomes a product: an agentic RL loop that trains the model to decide *when* to call its tools, an autonomous Space Domain Awareness agent rather than a model reasoning over tools a harness handed it. That capability is valuable and it belongs in the book so the research-to-product path is documented, but it is not a thesis contribution and it carries no thesis result. Parts 0 through X stand complete without it. Keeping it visibly fenced is the same discipline as everywhere else in this architecture: be explicit about what a thing is, and about what it is allowed to claim.
