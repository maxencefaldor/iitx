# Contributing to iitx

Thank you for considering a contribution. `iitx` holds itself to a strict standard —
clear, clean, lean, robust, zero technical debt — and this guide is how to meet it.

## Development setup

```sh
git clone https://github.com/maxencefaldor/iitx.git
cd iitx
uv sync --extra dev
uv run pytest tests/          # fast tier; add -m slow for the golden slow tier
uv run ruff check src tests && uv run ruff format --check src tests
uv run ty check src
```

## Before you write code

1. **Read the design documents.** `docs/design.md` fixes the core abstractions;
   `docs/glossary.md` fixes the vocabulary (use it identically in code, docstrings, and
   tests); `docs/generality.md` fixes the scope. A change that fights the design needs a
   design discussion first, not a workaround.
2. **Know the oracle contract.** PyPhi (pinned in `tests/oracle/generate/`) is the
   frozen oracle. If your change makes a number disagree with a fixture, do not adjust
   the code to match blindly — diagnose which side is right from the mathematics and
   record the finding in `docs/notes/oracle-findings.md`. A documented, justified
   disagreement is a result; a silent patch is a bug.

## Architecture rules of thumb

- **Static shapes always.** Combinatorial spaces are enumerated at build time as mask
  tables (`iitx.enumeration`); kernels `vmap`/`scan` over them. Never a data-dependent
  shape inside a jitted function; never `2**n` where `math.prod(shape)` is meant.
- **Full-shape embedding.** Repertoires are full state-space tensors with known
  constant factors on non-purview axes — and the whole and partitioned sides of any
  comparison must share one embedding.
- **Values by reduction, certificates by canonical order.** φ from masked min/max
  (differentiable); optimum identity from first-occurrence argmin over canonical
  tables, with quantize-then-compare tie semantics per measure.
- **Pure functions of `(system, state)`.** No global state, no config module; a
  measure's configuration is its dataclass fields. Every public function composes with
  `jit` and `vmap`; whether it composes with `grad` is part of its documented contract.
- Docstrings are Google style, state shape and dtype contracts, and cite the paper
  equation they implement.

## Tests

Every change comes with tests. The suite has three registers: golden fixtures (oracle
parity), hand-derived examples from the papers, and properties (normalization,
symmetry, jit ≡ eager, gradients vs finite differences). If your change affects
values, regenerate or extend the oracle fixtures via `tests/oracle/generate/`.

## Commits and pull requests

Imperative mood, and the message explains *why*. Keep `CHANGELOG.md` current. Run the
full local check before pushing:

```sh
uv run ruff check src tests && uv run ruff format src tests && uv run ty check src && uv run pytest tests/
```
