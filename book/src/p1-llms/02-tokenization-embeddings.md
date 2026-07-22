# Tokenization and embeddings

A language model does not read text. It reads integers. Everything between the string you typed and the first matrix multiply inside the network is the job of the tokenizer and the embedding matrix, and this stage is where a surprising amount of eval reproducibility quietly lives or dies. If you run the same prompt through two different chat templates you can get two different integer sequences, two different loss values, and two different eval scores, without touching a single weight. So this chapter treats the input surface as a first-class object: how byte-pair encoding turns text into tokens, what the vocabulary size actually trades off, why a chat template is *itself a tokenization decision* and not a cosmetic string wrapper, and how the embedding matrix (and weight tying) turns those integers into the vectors the transformer actually consumes.

## Theory

### Why not just use characters, or words

There are two obvious ways to turn text into integers, and both are bad. Character-level tokenization gives a tiny vocabulary (a few hundred symbols) but makes sequences brutally long, and a transformer's compute scales with the square of sequence length, so long sequences are expensive exactly where you cannot afford it. Word-level tokenization gives short sequences but a vocabulary that is effectively unbounded (every proper noun, typo, and hashtag is a new word) and helpless in front of anything it never saw at training time, the dreaded out-of-vocabulary token. The whole field settled on a middle path: **subword** tokenization, where common words stay whole, rare words fracture into pieces, and truly novel strings fall back to bytes so that *nothing* is ever out-of-vocabulary. Byte-pair encoding is the dominant recipe for building that subword vocabulary, and it is what Qwen3, gpt-oss, and essentially every model I touch in this book use.

### Byte-pair encoding, derived from a greedy compression rule

BPE started life as a compression algorithm, and it is easiest to understand as one. You begin with the most granular possible alphabet (raw bytes, so 256 base symbols, which guarantees every possible input is representable) and you repeatedly find the most frequent adjacent pair of symbols in your training corpus and merge it into a new single symbol. Do that $K$ times and you have learned $K$ merges and a vocabulary of $256 + K$ tokens.

```admonish derivation title="The BPE training loop and its greedy objective"
Let the corpus, after the initial split into base symbols, be a sequence of tokens. For any ordered pair of adjacent symbols $(a, b)$, let $c(a, b)$ be its count across the whole corpus. One step of BPE training does exactly this:

$$(a^\star, b^\star) = \arg\max_{(a,b)} \; c(a, b), \qquad \text{then introduce a new symbol } z := ab \tag{2.1}$$

and rewrite every adjacent occurrence of $a^\star b^\star$ in the corpus as the single symbol $z$. Repeat for $K$ merges. The learned artifact is an *ordered* list of merges $\big[(a_1,b_1), (a_2,b_2), \dots, (a_K,b_K)\big]$ plus the resulting vocabulary.

Why the greedy max-count rule? Merging the most frequent pair is the locally optimal move for shrinking the encoded length. If pair $(a,b)$ occurs $c(a,b)$ times, replacing all its occurrences with one symbol removes $c(a,b)$ tokens from the corpus. So the greedy step is exactly "delete as many tokens as possible this round," a coordinate-descent step on total sequence length. It is not globally optimal (the merge order interacts, and a merge that is best now can foreclose a better pair later), but it is cheap, deterministic, and empirically excellent, which is the usual reason a greedy algorithm wins.

**Encoding** a new string at inference time replays the merges *in the order they were learned*. Start from bytes, then for the learned merges in order, apply every merge wherever its pair currently appears, before moving to the next merge. That "in learned order" is not optional: apply merges in a different order and you can get a different tokenization of the same string, which would break the correspondence between the tokenizer and the weights that were trained against it. The rank of a merge in the list *is* its priority.
```

The consequence of equation (2.1)'s greedy, frequency-driven rule is that the resulting vocabulary is a mirror of the training corpus. Frequent English words become single tokens; a chemistry paper's jargon fractures into many pieces; a language underrepresented in the corpus tokenizes into far more tokens per sentence than English does, which is a real and measurable cost (more tokens means more compute and, in a fixed context window, less room). This is why "tokens per word" (the **fertility** of a tokenizer) differs across languages and domains, and why the same eval prompt can cost noticeably more tokens under one model's tokenizer than another's.

### The vocabulary-size tradeoff, quantified

How big should the vocabulary be? This is a genuine tradeoff with costs on both sides, and it is worth making the two forces explicit rather than hand-waving "bigger is better up to a point."

```admonish derivation title="Two opposing costs of vocabulary size V"
**Cost that falls with V (sequence length).** A larger vocabulary means longer merges are available, so a fixed piece of text tokenizes into fewer tokens. Let $L(V)$ be the expected token count of a document under a vocabulary of size $V$; empirically $L$ decreases with $V$ (with diminishing returns, roughly logarithmically once $V$ is in the tens of thousands). Since a transformer's self-attention cost per layer scales as $O(L^2 d)$ and its feed-forward cost as $O(L d^2)$, shorter sequences are cheaper *quadratically* in the attention term. Fewer tokens per document is a direct compute and context-window win.

**Cost that rises with V (embedding and output).** The model must carry an embedding matrix $E \in \mathbb{R}^{V \times d}$ and (unless tied) an output matrix of the same size. Its parameter count and memory grow *linearly* in $V$:

$$\text{params}(E) = V \cdot d. \tag{2.2}$$

The output softmax over $V$ classes also costs $O(L \cdot V \cdot d)$ per forward pass, and the loss's normalization sums over all $V$ logits. So a huge vocabulary bloats both the parameter budget (equation 2.2) and the final projection.

The two forces meet at a $V$ where the marginal token-length savings stop being worth the marginal embedding/softmax cost. Modern models land this at roughly $32{,}000$ to $256{,}000$: Qwen3 uses about $151{,}936$, and gpt-oss about $201{,}088$. The move to six-figure vocabularies is largely about multilingual and code fertility (equation 2.1's frequency rule underserves anything rare, so you buy those tokens back with a bigger $V$) traded against the fact that on a 16GB card an embedding matrix of $V \times d$ at, say, $151{,}936 \times 1024$ in BF16 is already $151{,}936 \times 1024 \times 2 \approx 0.31$ GB of the budget just to hold the lookup table.
```

### Chat templates are tokenization, not decoration

Here is the point most tutorials skip and that matters most for evals. A modern instruction-tuned model was trained on conversations that were serialized into a *specific* string format using *special tokens* that mark the boundaries between roles. These special tokens (things like a begin-of-turn marker, role labels for system/user/assistant, and an end-of-turn marker) are real entries in the vocabulary with their own integer IDs, and the model learned that "an assistant reply begins right after this exact token sequence." The **chat template** is the (usually Jinja) function that takes your list of role-tagged messages and produces exactly that serialized string, special tokens and all, before BPE ever runs.

This means the chat template is part of the tokenization, and getting it wrong is not a style error, it is feeding the model a distribution it was never trained on. Concretely, a bare prompt like `Solve: 2+2` and the same content wrapped as `<|im_start|>user\nSolve: 2+2<|im_end|>\n<|im_start|>assistant\n` are different token sequences, and the model's next-token distribution after each is different, because only the second one puts the model in the state it was trained to answer from. Two families disagree on the marker strings, on whether a system message is injected by default, on whether a trailing "generation prompt" (the dangling `assistant\n` that cues the model to start replying) is appended, and on whether reasoning models wrap a hidden thinking segment in their own special tokens. Every one of those choices changes the integer sequence, and therefore the logits, and therefore any eval score computed from them. When I insist later that evals pin the chat template, this is why: it is as load-bearing as the sampler settings.

### From integers to vectors: the embedding matrix

Once text is a sequence of token IDs $[t_1, \dots, t_L]$ with each $t_i \in \{0, \dots, V-1\}$, the model's first learned operation is to look up a dense vector for each ID. That is the embedding matrix $E \in \mathbb{R}^{V \times d}$: row $t_i$ is the $d$-dimensional vector fed into the first transformer layer.

```admonish derivation title="Embedding lookup is a one-hot matrix product"
It is tempting to think of the lookup as "just indexing," and computationally it is, but it is worth seeing why it is a genuine linear layer, because that is what ties it to the output side. Represent token ID $t$ as a one-hot row vector $\mathbf{e}_t \in \mathbb{R}^{V}$ (all zeros except a $1$ in position $t$). Then

$$\mathbf{e}_t^\top E = E_{t,:} \tag{2.3}$$

is exactly row $t$ of $E$: the one-hot vector selects a row. So embedding lookup is the linear map $x \mapsto x E$ restricted to one-hot inputs, which is why frameworks implement it as a gather (indexing) rather than an actual $V$-wide matrix multiply, the one-hot product is $V-1$ multiplications by zero. But treating it as a linear layer is what makes the gradient clean: during backprop, only the rows of $E$ that were actually looked up receive a gradient (the adjoint flows back through equation 2.3 into row $t$ and nowhere else), so the embedding's gradient is sparse in exactly the tokens that appeared in the batch. That sparsity is why the embedding table, despite being one of the largest single parameters, updates cheaply.
```

At the *other* end of the network, the model produces a hidden vector $h \in \mathbb{R}^{d}$ for each position and must turn it into a distribution over the $V$ possible next tokens. It does so with an output projection (the "unembedding" or `lm_head`) $U \in \mathbb{R}^{V \times d}$, computing logits $\ell = U h \in \mathbb{R}^{V}$, one score per vocabulary entry, which the softmax then normalizes (that is the whole next chapter and the objective chapter). Both $E$ and $U$ are $V \times d$ tables that map between token-space and hidden-space, just in opposite directions.

### Weight tying

Since $E$ maps tokens to vectors and $U$ maps vectors to token-logits, and both are $V \times d$, a natural question is whether they should be the *same* matrix (up to transpose). **Weight tying** sets $U = E$, so the output logits are $\ell = E h$ and the input lookup is $\mathbf{e}_t^\top E$. The argument for it is partly parametric (it removes a full $V \times d$ matrix from the model, which for Qwen3-0.6B's $151{,}936 \times 1024$ is $\approx 0.15$ billion parameters saved, a big fraction of a small model) and partly semantic (a token's input representation and the direction you must point a hidden state to *predict* that token are plausibly the same geometry). Small models tie almost always, because the embedding is a huge fraction of their parameters and the savings are decisive; large models sometimes untie, because at scale the extra flexibility of a separate output matrix is affordable and slightly helpful. Qwen3's smaller variants tie; whether a given checkpoint ties is a flag in its `config.json` (`tie_word_embeddings`), which I will read directly in the model-zoo chapter. Knowing whether tying is on tells you immediately whether the embedding and the `lm_head` are one tensor or two on disk, which is a memory fact you can check.

## Tooling

The tool here is the Hugging Face `tokenizers` / `transformers` stack, and specifically the `AutoTokenizer` interface plus the `apply_chat_template` method. A loaded tokenizer bundles four things that map straight onto the theory: the merge rules and vocabulary (equation 2.1's learned artifact, usually a fast Rust-backed BPE), the special-token registry (the IDs for the role and boundary markers), the chat template (a Jinja string stored in `tokenizer_config.json`), and the decode direction (turning IDs back into text). The fast tokenizers are the same `tokenizers` library exposed to Python, which is why they are quick enough to sit in an eval inner loop.

The two calls that matter for reproducibility are `tokenizer(text)`, which runs plain BPE and returns `input_ids`, and `tokenizer.apply_chat_template(messages, ...)`, which first renders your role-tagged message list through the model's Jinja template into the exact training-time string and *then* tokenizes it. The important arguments are `add_generation_prompt` (whether to append the dangling assistant cue so the model knows to start replying, which you want for generation and not for scoring a completed conversation) and, for reasoning models, `enable_thinking` or a template-specific flag that controls whether the hidden reasoning segment's special tokens are emitted. Because the template lives *inside the checkpoint*, using `AutoTokenizer.from_pretrained(model_id)` is what guarantees you serialize conversations the way that specific model expects. Hand-rolling the string yourself is how you accidentally feed a Qwen model a Llama-shaped prompt and wonder why the eval regressed.

```admonish under-the-hood
`apply_chat_template` with `tokenize=False` returns the rendered *string* before BPE, which is the single best debugging tool in this whole area: you can literally read the special tokens the model will see. Do this once for any new model and you will catch template mismatches (a missing generation prompt, a doubled begin-of-turn token, a system message you did not ask for) before they silently cost you eval points. The special tokens are added by the template as literal text and then recognized by the tokenizer as atomic IDs, so a mismatch shows up as either the wrong ID or, worse, the marker getting shattered into ordinary sub-tokens because it was not registered as special.
```

## Lab

The goal is to make "chat templates are tokenization" undeniable by measuring it: I will take one fixed prompt, render and tokenize it under three different chat templates, and count exactly how the integer sequences differ. The artifact is a table on disk showing token counts and the specific special tokens each template injects, which is the evidence behind the reproducibility rule.

```bash
uv init labs/tokenization
cd labs/tokenization
uv add transformers torch
```

```python title="labs/tokenization/compare_templates.py"
"""Tokenize one prompt under several chat templates and diff the results.

Artifact: artifacts/template_diff.json plus a printed comparison table.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

# Three instruction models with genuinely different chat templates.
MODELS = [
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]

MESSAGES = [
    {"role": "system", "content": "You are a careful math tutor."},
    {"role": "user", "content": "What is 17 * 24? Think step by step."},
]

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def analyze(model_id: str) -> dict:
    tok = AutoTokenizer.from_pretrained(model_id)
    rendered = tok.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    ids = tok.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True
    )
    special_ids = set(tok.all_special_ids)
    specials_used = [
        tok.convert_ids_to_tokens(i) for i in ids if i in special_ids
    ]
    return {
        "model": model_id,
        "vocab_size": tok.vocab_size,
        "n_tokens": len(ids),
        "n_special": sum(1 for i in ids if i in special_ids),
        "specials_used": specials_used,
        "rendered_prefix": rendered[:200],
        "first_20_ids": ids[:20],
    }

def main() -> None:
    results = [analyze(m) for m in MODELS]
    (OUT / "template_diff.json").write_text(json.dumps(results, indent=2))

    print(f"{'model':38} {'vocab':>8} {'tokens':>7} {'special':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:38} {r['vocab_size']:>8} "
              f"{r['n_tokens']:>7} {r['n_special']:>8}")
    print()
    for r in results:
        print(f"{r['model']}")
        print(f"  special tokens injected: {r['specials_used']}")
        print(f"  rendered head: {r['rendered_prefix']!r}\n")

    counts = [r["n_tokens"] for r in results]
    print(f"Token-count spread for the SAME prompt: "
          f"min={min(counts)}, max={max(counts)}, "
          f"delta={max(counts) - min(counts)} tokens")
    print(f"Artifact: {(OUT / 'template_diff.json').resolve()}")

if __name__ == "__main__":
    main()
```

```bash
uv run python compare_templates.py
```

```admonish gotcha
Some of these checkpoints (Llama in particular) are gated on the Hub and need `huggingface-cli login` plus an accepted license before `from_pretrained` will fetch them; if a model 403s, swap in any open instruct model you already have (for example `Qwen/Qwen2.5-0.5B-Instruct` or `HuggingFaceTB/SmolLM2-360M-Instruct`), the lesson is in the *difference between templates*, not in these specific three. Also note `tokenizer.vocab_size` reports the base vocabulary and may exclude a handful of added special tokens, so `len(tokenizer)` can read slightly higher; that discrepancy is itself a small teaching moment about where special tokens live.
```

**What you should see.** The script writes `artifacts/template_diff.json` and prints a table where the *same two-message prompt* produces visibly different token counts across the three models, typically a spread of several tokens, driven entirely by how many special/boundary tokens each template injects and whether it prepends a default system frame. You will see genuinely different special-token strings in each row (Qwen's `<|im_start|>`/`<|im_end|>` family versus Llama's `<|start_header_id|>`/`<|eot_id|>` family versus Phi's `<|system|>`/`<|end|>` family), and the printed `rendered head` lets you read the exact serialized string each model expects. The closing line reports the token-count delta for identical content, which is the whole point: nothing about the *meaning* changed, yet the integer sequence the model sees did, and that is sufficient to move a loss or an eval number. Record the exact counts from your run (they depend on the checkpoint revisions you pulled) and keep the JSON as the receipt for why the eval config pins a tokenizer and template, not just a model name.

```admonish read-along
Pair this chapter with **[BLLM]** ch. 2, which builds a BPE tokenizer and an embedding layer from scratch. Raschka's from-scratch merge loop is the concrete code behind equation (2.1), and his embedding-layer section is equation (2.3) in PyTorch; read his byte-level BPE walkthrough right after this chapter's derivation and the greedy merge rule stops being abstract.
```

```admonish substack-seed
"You can change a model's eval score without touching a single weight, just change the wrapper around the prompt." A post that shows the same question tokenized three ways, counts the token difference, and explains that the special tokens marking 'the assistant starts here' are real vocabulary entries the model was trained to condition on. The hook is that a 'chat template' sounds like formatting and is actually part of the model's input distribution, so anyone benchmarking models who isn't pinning templates is comparing apples to differently-wrapped apples. Ends on the practical rule: log the rendered string, not just the message list.
```
