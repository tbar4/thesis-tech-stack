# How to read this book

Before you hit chapter one, spend a few minutes here. This book has a fixed rhythm, a small vocabulary of labeled boxes, a shorthand for the references I lean on, and a couple of non-negotiable rules about how the code runs. None of it is complicated, but knowing it up front means the conventions get out of your way instead of tripping you up later. Think of this chapter as the legend on a map.

## The Theory, Tooling, Lab rhythm

Most chapters move through three phases in the same order, and the order is the pedagogy.

Theory comes first. I try to give you the intuition for what we are about to do and why it should work before I write a single equation, because an equation you were handed cold is just notation, whereas an equation you were expecting is an insight. This is also the part that extracts most cleanly into a blog post, so it is written to stand more or less on its own.

Tooling comes second. Here the intuition earns its math, and the math earns its implementation. This is where I derive the thing properly (the loss, the estimator, the update rule) and then connect each symbol to the actual library, flag, or function that computes it. The goal of this phase is that you never have to take the tooling on faith: you can point at a line of output and say which term of which equation produced it.

Lab comes last, and every lab ends with something written to disk. You run the commands, you get an artifact (a metrics JSON, a set of scored responses, a checkpoint, a plot), and that artifact is the proof the loop closed. The reason theory precedes tooling precedes lab is simple: I want you to understand the thing, then see how it is built, then build it yourself, in that order, so that when the lab misbehaves you already have the model in your head to debug it. Reversing the order (code first, understanding later) is how you end up with a working script you cannot reason about, which is exactly the situation this whole book exists to get you out of.

Not every chapter uses all three phases with equal weight. A reference chapter may be almost all tooling; the preface and this chapter are all theory. But when a chapter is teaching a piece of the loop, expect the three-beat structure.

## The admonition vocabulary

Throughout the book I use a small set of labeled callout boxes (mdBook calls them admonitions). Each type means one specific thing, so you can decide at a glance whether a box is essential to your path or safely skippable. Here is the whole vocabulary.

| Admonition | What it is for |
|---|---|
| `derivation` | A result worked from definitions, step by step, rigorous enough for the thesis committee. Skip if you trust the result; read if you need to defend it. |
| `under-the-hood` | What the library or hardware is actually doing beneath the API, when the abstraction hides something worth seeing. |
| `vram-budget` | Explicit memory accounting: where the gigabytes went, and whether it fits in 16GB. The running constraint of the book. |
| `thesis-thread` | The recurring SDA-flavored verifiable-reasoning worked example. Follow it for a sustained single problem; skip it and the main line still stands. |
| `substack-seed` | A framing or hook written to travel: the shareable core of an idea, sized for a post. |
| `gotcha` | A specific trap I hit or expect you to hit, and how to get past it. Read these; they are cheap insurance. |
| `read-along` | A pointer into one of the reference books below, so you can go deeper on a topic from someone who wrote a whole book on it. |

If you read nothing but the prose and the `gotcha` boxes you will still be able to run every lab. If you read the `derivation` boxes too you will be able to defend every result. The rest are there to enrich, extend, or share.

## Reference short keys

I lean on seven books repeatedly, and rather than spell out full citations every time, I use short keys in the text and in `read-along` boxes. Here is the key.

| Key | Reference |
|---|---|
| [S&B] | Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition |
| [RLHF] | Lambert, *RLHF Book* |
| [BRM] | Raschka, *Build a Reasoning Model (From Scratch)* |
| [BLLM] | Raschka, *Build a Large Language Model (From Scratch)* |
| [MADL] | Chaudhury, *Math and Architectures of Deep Learning* |
| [GAIA] | Hurbans, *Grokking Artificial Intelligence Algorithms* |
| [CAI] | Ness, *Causal AI* |

When I write "see [S&B] on the policy gradient theorem" or drop a `read-along` box pointing at [BRM], those keys are what I mean. The intent is not to make you buy seven books. It is to be honest about where an idea came from and to give you a well-worn path to more depth than a single chapter can hold. Where two references cover the same ground differently (and [S&B] versus [RLHF] on reward often will), I will say which one I am following and why.

## uv everywhere

Every piece of Python in this book runs under [uv](https://github.com/astral-sh/uv). There are no bare `pip install` lines and no hand-rolled `python -m venv` incantations anywhere in the labs, and this is a rule, not a preference.

The reason is reproducibility, which is the promise I made to future me in the preface. `uv` gives every lab a pinned, lockfile-backed environment that resolves the same way tomorrow as it did today, and it does it fast enough that spinning up a fresh environment per lab is painless rather than a chore. So when a lab needs dependencies you will see `uv` managing them, when it runs a script you will see `uv run`, and when it pins versions you will see them in a lockfile that is itself an artifact. If you are used to `pip` and `venv`, everything you know transfers; `uv` is just the tool that makes the pinning automatic instead of aspirational. If you have never used it, the chapter-zero setup walks you through installing it once, and after that it is invisible.

## Every lab ends in an artifact

I said it under the rhythm and I will say it again on its own because it is the load-bearing rule of the whole book: a lab is not done when the script exits zero. A lab is done when it has written a durable artifact to disk. A metrics file, a table of scored responses, a checkpoint directory, a saved figure. Something you can come back to, diff against a later run, and point at in an argument.

This is what makes the serve, evaluate, score, train, re-evaluate loop auditable rather than anecdotal. "The model got better" is a claim. Two dated metrics JSONs, one before training and one after, with the estimator and its uncertainty recorded in each, are evidence. The artifacts are also what let future me re-run a lab and check whether the new hardware gives the old answer. So expect every lab to tell you exactly what it wrote and where, and expect me to treat that file, not the terminal output, as the real result.

## Running the book on the same hardware

This book is written to be run, not just read, and it is written against one specific machine, described in full in the next chapter. If you have that machine or something close to it, you can execute every lab as written and compare your numbers to mine directly. If your hardware differs, the labs still run; what changes is the measured numbers, and because every measured number in this book is stamped with the date and driver it came from, you will always be able to tell which figures are hardware-relative and recompute them on your own box.

Practically, that means the ideal way to read this book is with the machine on and a terminal open. Read the theory, follow the tooling, then actually run the lab and watch your artifact land on disk next to mine. The book will make sense from the armchair, but it was built for the workbench. The next chapter pins down exactly what that workbench is.
