# System architecture

This page is the whole machine on one screen: how live space data becomes verifiable tasks, how those tasks measure a model, how the measurement becomes the reward that trains it, and how the loop closes on a single 16GB GPU. If you read only one page before diving into the parts, read this one. The diagram is the map; the prose under it is the *why*, because an architecture is a set of decisions, and the decisions are more useful than the boxes.

<iframe id="arch-frame" src="architecture-diagram.html" title="Evals as Rewards system architecture diagram" loading="lazy" style="width:100%;border:1px solid rgba(128,128,128,0.25);border-radius:12px;min-height:1500px;background:transparent;"></iframe>

<script>
  // Auto-size the architecture iframe to its content and keep its light/dark
  // theme in sync with the book's theme toggle (mdbook sets coal/navy/ayu for
  // dark themes on the <html> element).
  (function () {
    var frame = document.getElementById("arch-frame");
    if (!frame) return;
    function bookTheme() {
      return /\b(coal|navy|ayu)\b/.test(document.documentElement.className || "") ? "dark" : "light";
    }
    function syncTheme() {
      try { frame.contentWindow.postMessage({ archTheme: bookTheme() }, "*"); } catch (e) {}
    }
    window.addEventListener("message", function (e) {
      if (e.data && e.data.type === "arch-height" && e.data.height) {
        frame.style.height = (e.data.height + 8) + "px";
      }
    });
    frame.addEventListener("load", syncTheme);
    new MutationObserver(syncTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  })();
</script>

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
