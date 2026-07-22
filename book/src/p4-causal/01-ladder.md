# The ladder of causation

**Goal.** Distinguish seeing, doing, and imagining, and place the claims I make about models on the right rung, so that an eval table and a thesis sentence stop being confused for the same kind of statement.

**Covers.** Seeing versus doing versus imagining; why an eval table is a rung-one object and a thesis claim is a rung-two object; observational versus interventional questions as they actually show up in ML practice.

## Theory

### Three kinds of question hiding in one sentence

Here is a sentence I have written into a results table without thinking twice: "the trained model scores 0.61 on the held-out reasoning split, up from 0.52." It reads like one fact. It is actually two very different kinds of claim wearing the same clothes, and the whole of Part IV exists because I kept letting the second one sneak in on the credibility of the first.

The number 0.61 is a thing I saw. I ran the eval, the scorer returned a mean, and 0.61 is a faithful summary of what came back. Nobody can argue with it except by re-running the eval. The phrase "up from 0.52" is also a thing I saw, a second observation. But the word I keep wanting to attach, "improved," is not something I saw at all. "The training improved the model" is a claim about what the model's score would have been had I not trained it, compared against what it was after I did. One of those two worlds is counterfactual. I only ever get to observe one of them per model, and turning a pair of observed numbers into a claim about the effect of an intervention is a leap the numbers alone do not license.

Judea Pearl organizes exactly this leap into a three-rung ladder, and Ness builds the whole of [CAI] on it. The rungs are seeing, doing, and imagining, and each rung asks a strictly harder question than the one below.

### The three rungs

Rung one is association: seeing. The questions here are about what I observe and how observations move together. "What is the model's accuracy on this split?" "Do longer responses tend to score higher?" "Is judge A's mean rating correlated with response verbosity?" These are questions about the joint distribution of things I can record, and I answer them with counting, averaging, and correlation. Formally, rung one lives entirely inside $P(y \mid x)$, the probability of observing outcome $y$ given that I observe feature $x$. Every cell of every eval table I will ever produce is a rung-one object. It is an estimate of a conditional probability or an expectation under the distribution my pipeline happened to sample from.

Rung two is intervention: doing. Now the question changes shape. "What happens to the score if I train the model?" "What happens to the judge's rating if I make the response shorter, holding its actual quality fixed?" The operative object is not $P(y \mid x)$ but $P(y \mid \operatorname{do}(x))$, the distribution of $y$ when I reach in and set $x$ myself rather than merely observing whichever $x$ nature handed me. The $\operatorname{do}$ operator is the notational heart of the whole subject: $\operatorname{do}(x)$ means "set $X$ to $x$ by intervention, severing whatever normally determines $X$." The reason $P(y \mid \operatorname{do}(x))$ and $P(y \mid x)$ can differ, sometimes wildly, is the entire content of chapters 4.3 and 4.4. For now the point is only that they are different questions, and no amount of rung-one data answers a rung-two question on its own. You need an assumption about the mechanism, which is what a causal model supplies.

Rung three is counterfactual: imagining. Here the question is about a specific unit in a world that did not happen. "This particular training run got the model to 0.61. Would this same run, on this same seed, have gotten there anyway without the reward signal, on nothing but KL-regularized drift?" That is a claim about the very run I observed, under a change to one of its inputs, holding everything else about that run fixed. Counterfactuals are the strongest and the most treacherous, because the world you are comparing against is by construction unobservable. The formal object is something like $P(y_{x'} \mid x, y)$: the probability that the outcome would have been $y_{x'}$ under treatment $x'$, given that I actually observed treatment $x$ and outcome $y$. Most of my honest thesis claims will try to live on rung two and stay off rung three, because rung two is hard enough and rung three needs assumptions I usually cannot defend on one GPU.

```admonish read-along title="Go deeper: [CAI] chapters 1 and 2"
Ness introduces the ladder in [CAI] chapter 1 and formalizes association versus intervention in chapter 2. Read chapter 1 for the intuition and the historical framing (Pearl's ladder, the do-operator as the thing statistics left out), and chapter 2 for the first careful statement of why $P(y \mid x)$ and $P(y \mid \operatorname{do}(x))$ come apart. Everything I do in the rest of Part IV is machinery for climbing from rung one to rung two without lying about it.
```

### Why an eval table is rung one and a thesis claim is rung two

I want to nail this down because it is the load-bearing idea of the part. An eval table is a rung-one artifact by construction. It is a set of estimated expectations, $\hat{E}[\text{score} \mid \text{model} = m, \text{task} = t]$, computed over whatever prompts, decoding settings, and judge my harness happened to run. It answers "what did I see when I ran this model under these conditions." That is genuinely valuable and genuinely limited. It does not, and cannot, tell me what would have happened under conditions I did not run.

A thesis claim is almost always a rung-two object dressed as a rung-one summary. "GRPO training improved the model's reasoning" is $P(\text{score} \mid \operatorname{do}(\text{train}))$ versus $P(\text{score} \mid \operatorname{do}(\text{no train}))$. "Quantization to 4-bit hurt reasoning" is $P(\text{score} \mid \operatorname{do}(\text{4-bit}))$ versus $P(\text{score} \mid \operatorname{do}(\text{bf16}))$. "The judge is biased toward verbosity" is $P(\text{rating} \mid \operatorname{do}(\text{longer}))$ with true quality held fixed. Every one of these is a doing question. Every one of them is the kind of sentence a committee will actually challenge. And every one of them is, on its face, unsupported by the eval table alone, because the table is a stack of seeing.

The gap between the two rungs is not pedantry. It is the exact place where I can fool myself. If the model I trained also happened to be served at a different temperature, or evaluated on a slightly different prompt template, or scored by a judge that shifted between the two runs, then the observed jump from 0.52 to 0.61 is a rung-one fact contaminated by everything that changed alongside the training. Reading it as "training caused the jump" is a rung-confusion error, and it is the single most common way eval results mislead. The fix is not more decimal places. It is a causal model that says which of the things that changed are allowed to affect the score, so I can decide whether the comparison identifies the effect I care about.

```admonish gotcha title="The delta is not the effect"
The most seductive rung-confusion in this whole book is treating the reasoning delta of chapter 7.6 (post-training score minus pre-training score) as the causal effect of training. It is not. It is an observed difference between two rung-one measurements. It equals the causal effect only if nothing else that moves the score changed between the two measurements, which is a claim I have to argue, not assume. Chapter 7.6 computes the delta; chapters 4.5 and the audit in 4.6 are what earn the right to call it an effect.
```

### Observational versus interventional, in ML terms

The clean way to tell which rung a question sits on is to ask: does answering it require me to have controlled something, or only to have recorded it? Observational questions ("do high-verbosity responses get higher judge scores in my logged data?") need only records. Interventional questions ("if I forced responses to be more verbose, would scores rise?") need control, either a real experiment where I set verbosity, or a causal model plus an identification argument that lets me borrow the interventional answer from observational data.

Machine learning practice is soaked in interventional questions answered with observational data, usually without saying so. Ablations are the honest cases: an ablation is a deliberate $\operatorname{do}$, where I set one component on or off and hold the rest fixed, which is exactly why ablations are the closest thing to a real experiment in the ML toolkit and why chapter 7.7 leans on them so hard. Leaderboard comparisons are the dishonest cases: two models trained by two teams on two data mixes with two eval harnesses, compared as if the only difference were the thing the leaderboard names. That comparison reads as $\operatorname{do}(\text{model} = A)$ versus $\operatorname{do}(\text{model} = B)$ but is estimated from wildly non-comparable observations. Recognizing which of those two situations I am in, every single time, is the skill this part is trying to build.

### Why data alone cannot climb the ladder

It is worth being blunt about the impossibility at the center of all this, because it is the reason the rest of Part IV is not optional. No amount of rung-one data, however large, however clean, answers a rung-two question by itself. This is not a practical limitation I could fix with a bigger eval budget; it is a logical one. Two completely different causal worlds can produce identical observational distributions, and if they produce the same $P(y, x)$ then no function of the data can distinguish them, yet they can disagree about $P(y \mid \operatorname{do}(x))$. The textbook example is a variable that both raises treatment and raises the outcome on its own: the observed correlation between treatment and outcome is the same whether the treatment works, does nothing, or actively hurts, depending on how strong the hidden common cause is. The data is silent on which world I am in. What breaks the tie is an assumption about the mechanism, and a causal model is just a disciplined way of writing that assumption down so it can be argued with.

This is why I keep insisting that a thesis claim smuggles in structure the eval table does not contain. When I say "training improved the model," I am not reporting the table; I am asserting a causal model in which the training is the thing that moved the score and the confounders did not, and then reading the effect off that model using the table as input. The honest version of the sentence always has two parts: here is what I observed (rung one, the table), and here is the causal model under which that observation identifies the effect I claim (rung two, the assumption). Part IV is the machinery for writing the second part explicitly, checking whether it is even the kind of model under which the effect can be recovered, and stress-testing it when it is. The eval table is necessary and never sufficient, and the gap between necessary and sufficient is exactly one causal model wide.

### The cost of getting the rung wrong

There is an asymmetry in the failure modes that is worth internalizing. Under-claiming (reporting a rung-two result as if it were only rung one, "the score was 0.61," and refusing to say why) is safe but useless: it wastes a real finding and tells a committee nothing about whether the method works. Over-claiming (reporting a rung-one observation as if it were rung two, "training caused the improvement," with no model behind it) is the dangerous direction, because it is persuasive and wrong in a way that survives casual review and collapses under a hostile one. The committee reader whose job is to find the skipped step will find it exactly here, at the unearned causal verb, and everything downstream of that verb becomes suspect. So the discipline is not to avoid causal claims; the loop of this book is a causal machine and its whole point is causal claims. The discipline is to make every causal claim carry its model in the open, so that the reader can attack the model rather than discovering, late and annoyed, that there was never a model at all.

## Tooling

There is no library for this chapter. The tool is a habit, and the habit is a two-column reading of every claim I write: what did I see, and what am I asserting about doing. So the tooling here is a small, explicit rubric I can apply mechanically to any sentence, and then encode as a classifier I run over real model-card claims in the lab.

The rubric has three tests, applied in order.

First, the **verb test**. Does the claim contain a causal verb (improve, hurt, cause, enable, degrade, boost, break) or a causal preposition (because, due to, thanks to)? If yes, it is reaching for at least rung two. A pure rung-one claim uses only descriptive verbs: scores, achieves, measures, correlates.

Second, the **counterfactual test**. Can I restate the claim as "compared to what would have happened otherwise"? If the restatement is natural and the "otherwise" world is not something I observed, the claim is rung two (or three). "0.61 on the split" has no natural otherwise; it is rung one. "Improved to 0.61" implies an otherwise (the untrained score) and is rung two.

Third, the **unit test**. Is the claim about a population or distribution (rung two) or about one specific realized unit under a change (rung three)? "Training improves models like this one on average" is rung two. "This run would have failed without the reward" is rung three, because it is about that run.

I will encode this as three boolean features per claim and a small rule that maps them to a rung. It is deliberately crude. The value is not a perfect classifier; the value is that running the rubric forces me to read each claim as a causal statement and notice when a published card is quietly standing on rung two while showing me rung-one evidence.

## Lab

The lab classifies ten model-card-flavored claims, assigns each a rung with the rubric above, and writes an annotated table to disk. The claims are paraphrased archetypes, not quotations, so the artifact is self-contained and reproducible without scraping anyone's card. No GPU is involved; this runs anywhere uv runs.

```python
# file: labs/p4-01-ladder/classify_claims.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2"]
# ///
"""Classify model-card claims by rung of the ladder of causation.

Rung 1 = association (seeing): a summary of observed data.
Rung 2 = intervention (doing): a claim about P(y | do(x)) over a population.
Rung 3 = counterfactual (imagining): a claim about a specific realized unit
         under a change to one of its inputs.

The classifier is intentionally a transparent rule over three boolean tests,
not an ML model. The point is to make the reading of each claim auditable.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

CAUSAL_VERBS = r"(improv|hurt|caus|enabl|degrad|boost|break|unlock|regress|damag|help)"
CAUSAL_PREPS = r"(because|due to|thanks to|as a result of|leads to|drives)"
# words that signal a claim about one specific realized run/unit
UNIT_MARKERS = r"(this run|this checkpoint|this seed|would have|had we not|without the)"

@dataclass
class Claim:
    text: str
    source_kind: str  # e.g. "capability card", "quant report", "judge study"

def verb_test(t: str) -> bool:
    return bool(re.search(CAUSAL_VERBS, t, re.I) or re.search(CAUSAL_PREPS, t, re.I))

def counterfactual_test(t: str) -> bool:
    # a natural "compared to otherwise" reading: comparative language paired
    # with a causal verb over an unobserved alternative condition
    comparative = re.search(r"(up from|compared to|versus|vs\.?|better than|worse than|relative to)", t, re.I)
    return bool(comparative and verb_test(t))

def unit_test(t: str) -> bool:
    return bool(re.search(UNIT_MARKERS, t, re.I))

def classify(t: str) -> tuple[int, str]:
    v, c, u = verb_test(t), counterfactual_test(t), unit_test(t)
    # NOTE: c (counterfactual_test) is a diagnostic feature, not a decision
    # input. It is defined as (comparative AND verb_test), so it can never flip
    # a rung that v has not already set; the rung is decided by u and v alone.
    # I report it to flag "compared-to-otherwise" phrasing, not to classify.
    if u:
        return 3, "counterfactual: 'would have'/'this run'/'without the' names a specific unhappened world"
    if v or c:
        return 2, "interventional: causal verb/preposition implies a do() question"
    return 1, "associational: descriptive summary of observed data"

CLAIMS = [
    Claim("The model scores 0.61 on the held-out reasoning split.", "capability card"),
    Claim("Post-training accuracy improved to 0.61, up from 0.52 before RL.", "capability card"),
    Claim("4-bit quantization degraded chain-of-thought accuracy by 3 points.", "quant report"),
    Claim("Longer responses are positively correlated with judge scores.", "judge study"),
    Claim("The instruction-tuned variant outperforms the base model because of preference data.", "capability card"),
    Claim("On MATH the model achieves 0.44 pass@1 at temperature 0.7.", "capability card"),
    Claim("Our GRPO recipe improves reasoning over the SFT-only baseline.", "method report"),
    Claim("Without the verifiable reward, this run would have plateaued at the SFT level.", "method report"),
    Claim("The judge assigns higher ratings than a blind human panel on the same items.", "judge study"),
    Claim("Enabling the scratchpad prompt boosts GSM8K relative to no scratchpad.", "method report"),
]

def main() -> None:
    out_dir = Path("labs/p4-01-ladder")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in CLAIMS:
        rung, why = classify(c.text)
        rows.append({**asdict(c), "rung": rung, "rationale": why,
                     "verb_test": verb_test(c.text),
                     "counterfactual_test": counterfactual_test(c.text),
                     "unit_test": unit_test(c.text)})
    art = out_dir / "claim_rungs.json"
    art.write_text(json.dumps(rows, indent=2))
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        print(f"[rung {r['rung']}] {r['text']}")
    print("\nrung counts:", df["rung"].value_counts().sort_index().to_dict())
    print(f"artifact written: {art.resolve()}")

if __name__ == "__main__":
    main()
```

Run it with uv, which resolves pandas into a throwaway pinned environment (pandas is here only to keep the artifact shape consistent with later chapters that read this JSON back into a DataFrame; the classifier itself is pure standard library):

```bash
uv run labs/p4-01-ladder/classify_claims.py
```

The artifact is `labs/p4-01-ladder/claim_rungs.json`: ten claims, each stamped with its three boolean test results, the assigned rung, and a one-line rationale. That file is the durable object. It is also the seed of a habit. From here on, whenever I paste a sentence into the thesis, I can run it through the same rubric and see immediately whether I am about to stand on rung two while pointing at rung-one evidence.

**What you should see.** Ten lines to stdout, one per claim, each tagged with a rung, then a count line reading `{1: 4, 2: 5, 3: 1}`. Four claims land at rung 1: the two pure measurements (the 0.61 split score and the MATH pass@1 line) and, more instructively, the two comparative-but-verbless claims (longer responses are "correlated" with scores, the judge "assigns higher ratings than" a panel), which describe observed differences without any causal verb and so stay associational. Five land at rung 2, the ones carrying an explicit causal verb or preposition (improved to 0.61, degraded by 3 points, outperforms because of preference data, improves over baseline, boosts GSM8K), because a causal verb sits on top of a comparison to an unobserved alternative. One lands at rung 3: "without the verifiable reward, this run would have plateaued," which names a specific realized run and a world where one of its inputs was changed. The exact split matters less than the reflex: every causal verb in a results section is a promissory note that the rest of Part IV has to pay off. The two verbless comparatives are the ones worth staring at, because they read as causal to a hurried reader but the `verb_test` field is `false`, which is the classifier telling you the causal reading was in your head, not on the page.

```admonish substack-seed
Every results table in machine learning is a photograph, and every abstract quietly captions it as a movie. The photograph says "the model scored 0.61." The caption says "training made it better." Those are not the same claim, and Judea Pearl's ladder of causation tells you exactly why: the table lives on rung one (seeing), and the word "better" lives on rung two (doing), the rung where you have to reason about what would have happened otherwise. This post is a 900-word field guide to spotting the jump in your own writing, a three-question rubric (is there a causal verb, is there an unobserved "otherwise," is it about one specific run) that tells you which rung a sentence is standing on, run over ten real-flavored model-card claims. The punchline is that the gap between rungs is not pedantry. It is the exact seam where an eval result turns into a lie you did not mean to tell.
```
