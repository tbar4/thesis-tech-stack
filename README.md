# Evals as Rewards

**Measuring and training reasoning models on one GPU.**

An mdbook that walks one person, from a freshly unboxed machine, through every
layer of a serve → evaluate → score → train → re-evaluate loop for open-weight
reasoning models, with enough foundational theory that nothing in the stack is
a black box.

This repository currently holds the **scaffold**: `book.toml`, the complete
`SUMMARY.md`, and a titled stub for every chapter and appendix so the book
builds clean under `create-missing = false`. Chapters are drafted in dependency
order per the authoring workflow in the spec.

## Layout

```
book/            # the mdbook (book.toml, src/, theme/)
references/      # local copies of the seven reference books (gitignored)
.github/         # CI that builds the book with linkcheck
```

## Building locally

The book uses mdbook with `mdbook-katex`, `mdbook-mermaid`, `mdbook-admonish`,
`mdbook-toc`, and `mdbook-linkcheck`. Install them (via `cargo install`), then:

```sh
cd book
mdbook-admonish install .   # generates theme/mdbook-admonish.css
mdbook-mermaid install .    # generates theme/mermaid assets + book.toml entries
mdbook serve                # or: mdbook build
```

CI (`.github/workflows/book.yml`) runs those same steps on every push and pull
request, with linkcheck enabled.

## Licensing

- Code blocks: MIT (see `LICENSE`).
- Prose: CC BY-NC-SA.
- Thesis-specific results (final numbers, figures, the delta report) stay
  private until after the defense.
