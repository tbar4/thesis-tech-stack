# The language-modeling objective

Everything up to here has been machinery: tensors, tokens, attention, the block. This chapter is the *goal* those parts are optimized toward. A language model is trained to do one thing, predict the next token, and the loss that measures how well it does that is cross-entropy, which is not an arbitrary choice but the unique consequence of maximum likelihood. Get this chapter right and a lot of later confusion dissolves, because the training loss, perplexity, the eval log-probabilities, and eventually the RL reward are all readings of the same next-token distribution. So I derive cross-entropy from maximum likelihood, derive perplexity as its exponential, nail down the off-by-one bookkeeping of label shifting, explain why loss curves have the shape they do, and then compute a loss and a perplexity by hand on a real paragraph and match the library to the decimal.

## Theory

### Next-token prediction as a probability model

A language model defines a probability distribution over sequences of tokens. By the chain rule of probability, any joint distribution over a sequence $t_1, t_2, \dots, t_T$ factors into a product of conditionals, each one the probability of the next token given all the previous:

$$p_\theta(t_1, \dots, t_T) = \prod_{i=1}^{T} p_\theta(t_i \mid t_1, \dots, t_{i-1}). \tag{5.1}$$

This factorization is exact, not an approximation, and it is why autoregressive next-token prediction is a complete model of language and not a simplification: if you can model every conditional $p_\theta(t_i \mid t_{<i})$, you have modeled the whole joint. The transformer's job is to produce each of those conditionals, and it does: for each position $i$, the network outputs a hidden vector, the unembedding turns it into a logit vector $\ell^{(i)} \in \mathbb{R}^{V}$, and the softmax turns that into the distribution over the next token,

$$p_\theta(t_{i} = w \mid t_{<i}) = \mathrm{softmax}(\ell^{(i)})_w = \frac{e^{\ell^{(i)}_w}}{\sum_{v=1}^{V} e^{\ell^{(i)}_v}}. \tag{5.2}$$

The causal mask from the attention chapter is what makes equation (5.2) honest: position $i$'s logits depend only on $t_{<i}$, never on $t_i$ or later, so the model genuinely predicts rather than peeks. And because a single forward pass produces the logits at *every* position simultaneously (that is what the sequence dimension is), one forward pass over a length-$T$ sequence yields all $T$ conditionals at once, which is what makes training efficient.

### Cross-entropy is maximum likelihood, derived

We have a model that assigns a probability to any sequence. Training means adjusting $\theta$ so the model assigns *high* probability to the real text in the training corpus. That principle is maximum likelihood, and cross-entropy loss is exactly what it becomes after a couple of monotone transformations.

```admonish derivation title="Cross-entropy loss from maximum likelihood"
Maximum likelihood says: choose $\theta$ to maximize the probability the model assigns to the observed data. For a single sequence, that is the joint of equation (5.1). Maximizing a product of many numbers in $(0,1)$ is numerically hopeless (the product underflows to zero), and maxima are unchanged by a monotone increasing transform, so take the logarithm, which turns the product into a sum:

$$\log p_\theta(t_1, \dots, t_T) = \sum_{i=1}^{T} \log p_\theta(t_i \mid t_{<i}). \tag{5.3}$$

Maximizing the log-likelihood is the same as minimizing its negative, so define the loss as the *negative* log-likelihood, and average over the $T$ positions to make it a per-token quantity that does not grow with sequence length:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{i=1}^{T} \log p_\theta(t_i \mid t_{<i}). \tag{5.4}$$

Now expand one term using the softmax of equation (5.2). For position $i$ with true next token $t_i$, the per-position loss is

$$\ell_i = -\log p_\theta(t_i \mid t_{<i}) = -\log \frac{e^{\ell^{(i)}_{t_i}}}{\sum_{v} e^{\ell^{(i)}_v}} = -\ell^{(i)}_{t_i} + \log \sum_{v=1}^{V} e^{\ell^{(i)}_v}. \tag{5.5}$$

This is exactly **cross-entropy** between the one-hot target distribution (all mass on the true token $t_i$) and the model's predicted distribution. To see the equivalence, cross-entropy between a target distribution $q$ and a prediction $p$ is $H(q, p) = -\sum_w q_w \log p_w$; with $q$ one-hot at $t_i$, every term vanishes except $w = t_i$, leaving $-\log p_{t_i}$, which is equation (5.5). So "cross-entropy loss" and "average negative log-likelihood of the true next token" are the same object, and equation (5.4) is what every LM trainer minimizes.

The gradient is clean and worth seeing because it is why this loss trains well. Differentiating equation (5.5) with respect to the logits gives

$$\frac{\partial \ell_i}{\partial \ell^{(i)}_v} = \mathrm{softmax}(\ell^{(i)})_v - \mathbb{1}[v = t_i] = p_v - q_v, \tag{5.6}$$

the predicted probability minus the target — a bounded, well-behaved gradient that is exactly zero only when the model puts all its mass on the true token, and largest when it is confidently wrong. This "prediction minus target" form is the same clean signal the softmax-plus-cross-entropy pairing always produces, and it is no accident that the two are computed together in one fused kernel.
```

The second form of equation (5.5), $-\ell_{t_i} + \log\sum_v e^{\ell_v}$, is worth pausing on because that second term, the **log-sum-exp**, is the log of the softmax denominator, i.e. the log-partition-function. It is what couples the true token's logit to *all* the others: to lower the loss you can either raise the true token's logit or lower everyone else's, and the log-sum-exp is what makes those two moves trade off. It is also the term that must be computed in a numerically stable way (subtract the max logit first), which is a detail the library handles and a hand computation must respect.

### Perplexity

The loss in equation (5.4) is in nats (natural-log units), which is fine for optimization but unintuitive as a report. Perplexity is the standard human-facing metric, and it is just the loss exponentiated back out of log-space.

```admonish derivation title="Perplexity as the exponential of cross-entropy"
Perplexity is defined as the exponential of the average per-token negative log-likelihood:

$$\mathrm{PPL} = \exp\!\left(-\frac{1}{T}\sum_{i=1}^{T} \log p_\theta(t_i \mid t_{<i})\right) = \exp(\mathcal{L}). \tag{5.7}$$

Undoing the log and the average recovers the geometric mean of the per-token probabilities, inverted:

$$\mathrm{PPL} = \left(\prod_{i=1}^{T} p_\theta(t_i \mid t_{<i})\right)^{-1/T}. \tag{5.8}$$

Equation (5.8) is the interpretation that makes perplexity worth reporting: it is the reciprocal geometric-mean probability the model assigned to the true tokens, which reads as an **effective branching factor** — the model is, on average, as uncertain as if it were choosing uniformly among $\mathrm{PPL}$ equally likely tokens at each step. A perplexity of 1 is a perfect model (it assigned probability 1 to every true token); a perplexity of $V$ (the vocabulary size) is a model no better than uniform guessing; real models land somewhere in between, and lower is better. Because $\exp$ is monotone, minimizing loss and minimizing perplexity are the same optimization, but perplexity is the number you quote. One caution the definition makes obvious: perplexity depends on the tokenizer, because $T$ (the token count) and the per-token probabilities both change with the vocabulary — so perplexities are only comparable across models that share a tokenizer, a point that connects straight back to the tokenization chapter.
```

### Label shifting: the off-by-one that everyone hits

There is a bookkeeping subtlety that trips up every from-scratch implementation, and it follows directly from equation (5.2): the logits at position $i$ predict the token at position $i+1$, not the token at position $i$. So to build the (prediction, target) pairs for the loss, you shift. Given an input sequence of token IDs `input_ids` of length $T$, the model produces logits of shape $(T, V)$, and the correct pairing is:

$$\text{prediction at position } i \;\longleftrightarrow\; \text{target } t_{i+1}, \qquad i = 1, \dots, T-1. \tag{5.9}$$

In practice this means the logits are sliced to drop the *last* position (it predicts a token beyond the sequence, which has no label) and the labels are sliced to drop the *first* token (nothing predicts it), so you compute the loss over $T-1$ pairs: `logits[:-1]` against `labels[1:]`. Getting this shift wrong by one position is the single most common silent bug in a custom training loop; it does not crash, it just trains the model to predict the *current* token from itself, which produces a suspiciously low loss that collapses to nonsense. The Hugging Face models do this shift internally when you pass `labels`, which is convenient but hides the mechanism, so the lab does it by hand to make equation (5.9) unavoidable. Padding tokens and, in instruction tuning, the prompt tokens are masked out of the loss by setting their label to a sentinel (`-100`) that the loss function skips, which is how you train only on the completion and not on the question.

### Why loss curves look the way they do

A training loss curve for a language model has a characteristic shape, and every part of it is explainable from the objective. It starts high, near $\log V$: an untrained model with random weights outputs roughly uniform logits, so by equation (5.5) the initial per-token loss is about $-\log(1/V) = \log V$, which for a vocabulary of $151{,}936$ is $\log(151936) \approx 11.9$ nats. That is a number you can predict *before training starts* and check on step 0, and it is a great sanity test: if your fresh model's loss is not near $\log V$, something (usually the label shift or the tokenizer) is wrong. The curve then drops fast at first, because the easiest wins are cheap: learning unigram frequencies (some tokens are just common) and basic local structure knocks the loss down quickly. Then it flattens into a long slow decline, because the remaining loss is the genuinely hard, long-range, semantic prediction that improves slowly, and because equation (5.6)'s gradient shrinks as predictions get better. The curve is often plotted on a log-x axis precisely because progress is roughly linear in log-steps over the long tail. Sharp spikes usually mean a bad batch or a learning rate too high (an optimizer step that overshot); a loss that plateaus far above $\log V$'s implied floor for the data means underfitting or a bug. Reading these curves fluently is a skill the whole training half of the book leans on, and it all comes back to equation (5.4).

## Tooling

The tool is `torch.nn.functional.cross_entropy` (and its object form `nn.CrossEntropyLoss`), which fuses three things the derivation kept separate: the softmax of equation (5.2), the log, and the negative-log-likelihood gather of equation (5.5), all in one numerically stable kernel. It expects *raw logits*, not probabilities, precisely so it can apply the stable log-sum-exp (subtracting the max logit) internally rather than taking the log of an already-softmaxed number that may have underflowed. It gathers the true-token logit via an integer label tensor (no one-hot ever built, the same gather-versus-one-hot efficiency from the embeddings chapter), supports the `-100` ignore index for masking, and returns the mean over unmasked positions by default, which is exactly equation (5.4). On the Hugging Face side, passing `labels=input_ids` to a causal LM's forward makes it do the label shift of equation (5.9) internally and return `.loss`; understanding that this convenience is *just* the shifted cross-entropy is the difference between trusting the number and being able to reproduce it, which is the point of the lab.

```admonish under-the-hood
`cross_entropy` reports its loss in **nats** (natural log), so exponentiating it directly gives perplexity per equation (5.7). If you ever see loss reported in bits, that is log base 2, and the two differ by a factor of $\ln 2 \approx 0.693$; perplexity from bits is $2^{\text{loss}}$ rather than $e^{\text{loss}}$. Mixing the two is a common way to compute a perplexity that is off by an exponential factor, so always know which log your loss is in. PyTorch is nats; equation (5.7) assumes nats.
```

## Lab

The goal is to compute a language-modeling loss and perplexity fully by hand from raw logits, honoring the label shift of equation (5.9) explicitly, and match the library's fused `cross_entropy` and the model's own `.loss` to many decimal places. The artifact is a JSON report showing the hand-computed and library values side by side, which proves the loss is exactly the shifted negative log-likelihood of the derivation and nothing more.

```bash
uv init labs/lm-objective
cd labs/lm-objective
uv add torch transformers
```

```python title="labs/lm-objective/loss_by_hand.py"
"""Compute LM loss and perplexity by hand, matched against the library.

Artifact: artifacts/loss_check.json with hand vs library values.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PARAGRAPH = (
    "The mitochondrion is the powerhouse of the cell. It generates most of "
    "the cell's supply of adenosine triphosphate, used as chemical energy."
)
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    ids = tok(PARAGRAPH, return_tensors="pt").input_ids  # (1, T)
    T = ids.shape[1]

    with torch.no_grad():
        logits = model(ids).logits  # (1, T, V)

    # --- By hand: shift (eq 5.9), then per-position NLL (eq 5.5), then mean. ---
    shift_logits = logits[0, :-1, :]   # positions 1..T-1 predict next token
    shift_labels = ids[0, 1:]          # tokens 2..T are the targets
    log_probs = F.log_softmax(shift_logits, dim=-1)          # eq 5.2 in log-space
    true_logp = log_probs[torch.arange(T - 1), shift_labels] # gather eq 5.5
    per_token_nll = -true_logp
    hand_loss = per_token_nll.mean().item()                 # eq 5.4
    hand_ppl = math.exp(hand_loss)                           # eq 5.7

    # --- Library: fused cross_entropy on the same shifted tensors. ---
    lib_loss = F.cross_entropy(shift_logits, shift_labels).item()

    # --- Library convenience: model computes the shift + loss itself. ---
    with torch.no_grad():
        model_loss = model(ids, labels=ids).loss.item()

    report = {
        "model": MODEL,
        "num_tokens": T,
        "num_predicted_positions": T - 1,
        "hand_loss_nats": hand_loss,
        "library_cross_entropy": lib_loss,
        "model_dot_loss": model_loss,
        "hand_perplexity": hand_ppl,
        "log_V_floor": math.log(model.config.vocab_size),
        "max_abs_diff_hand_vs_lib": abs(hand_loss - lib_loss),
    }
    (OUT / "loss_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nA random model would start near log(V) = "
          f"{report['log_V_floor']:.3f} nats; this trained model is far below.")
    print(f"Artifact: {(OUT / 'loss_check.json').resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python loss_by_hand.py
```

```admonish gotcha
The one place a hand computation drifts from the library is numerical: computing `log(softmax(logits))` as two separate steps can differ in the last few digits from `log_softmax`, which fuses them with the stable log-sum-exp. The lab uses `F.log_softmax` on purpose so my "by hand" version and the library's `cross_entropy` share the same stable path and agree to ~1e-6; if you instead wrote `torch.log(torch.softmax(...))` you would still match to about five decimals but see a slightly larger residual, which is itself the log-sum-exp stability lesson from the derivation. Also, the `model(ids, labels=ids).loss` value matches only because HF applies exactly the equation (5.9) shift internally — if it disagreed, that would mean a shift convention mismatch, which is precisely the bug this lab is designed to expose.
```

**What you should see.** The script writes `artifacts/loss_check.json` where three independently computed numbers agree: my by-hand loss, the library's `cross_entropy` on the same shifted tensors, and the model's own `.loss` from passing `labels`, all matching to roughly six decimal places (`max_abs_diff` around $10^{-6}$). The hand-computed perplitxy is the exponential of that loss per equation (5.7). And the report prints the $\log V$ floor (about 11.9 nats for Qwen3's vocabulary) next to the actual loss, which sits far below it, exactly as a trained model should — that gap between "random-init loss $\approx \log V$" and "trained loss" is the whole of language modeling compressed into two numbers. Keep the JSON as the receipt that the training loss is nothing but the shifted negative log-likelihood of equation (5.4), computed by hand and confirmed. Record your exact loss and perplexity values from the run, since they depend on the checkpoint revision.

```admonish read-along
Pair this with **[BLLM]** ch. 5, which implements the training loop, the cross-entropy loss, and perplexity from scratch and plots the loss curve on real text. Raschka's manual loss code is the long form of this lab's `loss_by_hand`, and his discussion of what the loss and perplexity numbers mean lines up directly with equations (5.4)–(5.8); read his training-loop section right after this derivation and the label shift of equation (5.9) will be muscle memory.
```

```admonish substack-seed
"Cross-entropy loss isn't a design choice — it's the only thing maximum likelihood can become." A post that derives the LM loss in one sitting: the chain rule of probability turns a sequence into a product of next-token conditionals, maximum likelihood says make that product big, taking the log turns it into a sum you can actually optimize, and negating it gives you cross-entropy — with the softmax's log-sum-exp term as the thing that couples the right answer to all the wrong ones. Then the kicker: a fresh random model's loss is predictable in advance at log(vocab-size), about 11.9 for a modern tokenizer, so if your training run doesn't start there, you have a bug, not a model. It's a derivation that doubles as a debugging checklist.
```
