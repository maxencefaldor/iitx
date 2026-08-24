# cax engineering notes (for iitx)

Study of https://github.com/maxencefaldor/cax for engineering standard only — not domain design.

**Version studied:** `v0.3.3` era, commit `1af11859674142463c163f542aad9ac90ace3f1e`
("Refactor Lenia metrics and Leniabreeder evaluation API", 2026-08-05, `main`).
`pyproject.toml` declares `version = "0.3.3"`; latest tag on origin is `v0.3.3`
(tag `601f30ce…` → commit `5e36f581…`, a few commits behind the studied HEAD).

---

## 1. Packaging (`pyproject.toml`)

- **Build backend:** hatchling, minimal config — no plugins, no hatch-vcs:

  ```toml
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```

- **Version management:** **static**, hand-bumped `version = "0.3.3"` in `[project]`.
  No dynamic versioning, no `__version__` attribute anywhere in `src/` (checked —
  `src/cax/__init__.py` is a one-line docstring). CITATION.cff lags at `0.3.2`,
  a known cost of hand-bumping in two places.

- **Metadata style:** PEP 621 `[project]` table. Modern license fields
  (`license = "MIT"` + `license-files = ["LICENSE"]`, PEP 639). Rich `keywords`
  list mirrored verbatim in CITATION.cff. Trove classifiers list each supported
  Python minor explicitly (3.12/3.13/3.14) plus license and OS-independent.

- **Python floor:** `requires-python = ">=3.12"` — aggressive; recently bumped
  from 3.11 (commit "Drop Python 3.11 support, require >=3.12"). Source freely
  uses PEP 695 generics (`class ComplexSystem[State, Input](nnx.Module)`).

- **Dependency philosophy:** tiny runtime dep list, all **floor-pinned with `>=`,
  never upper-bounded**:

  ```toml
  dependencies = [
      "jax>=0.10.0",
      "flax>=0.12.6",
      "optax>=0.2.8",
      "pillow>=11.1.0",
  ]
  ```

  Floors are recent (tracks latest jax/flax rather than supporting old ones).
  `uv.lock` (459 KB) is committed for reproducible dev, but the published wheel
  stays permissive.

- **Optional extras:** three, cleanly separated by audience:
  - `dev`: `pytest>=8.3.0`, `pytest-cov>=5.0.0`, `pytest-xdist>=3.5.0`, `ruff>=0.15.0`, `ty>=0.0.38`
  - `docs`: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter` (unpinned)
  - `examples`: notebook-only heavyweight deps (`torchvision`, `evosax`, `mediapy`, `ipykernel`, `ipywidgets`, `tqdm`)

- **URLs:** just `Homepage` (GitHub) and `Documentation` (GitHub Pages).

- **Layout:** `src/` layout (`src/cax/...`), which hatchling picks up by
  convention with zero config. Package data (Lenia `patterns/*.pickle`) ships
  inside the package tree.

## 2. Tooling

- **Ruff is the only formatter+linter**, configured in `pyproject.toml` (no
  standalone `ruff.toml`):

  ```toml
  [tool.ruff]
  src = ["src", "tests"]
  target-version = "py314"
  line-length = 100

  [tool.ruff.lint]
  select = ["B", "C4", "D", "E", "F", "I", "N", "PERF", "PT", "RUF", "SIM", "UP", "W"]
  ignore = [
      "D203",  # incorrect-blank-line-before-class
      "D206",  # docstring-tab-indentation
      "D213",  # multi-line-summary-second-line
      "N803",  # non-lowercase-variable-in-function
      "N806",  # invalid-argument-name
      "W191",  # tab-indentation
  ]

  [tool.ruff.format]
  indent-style = "tab"
  ```

  Notable choices:
  - **Tabs, not spaces** — hence ignoring W191/D206; line length 100.
  - `D` (pydocstyle) enabled globally → **every module, class, and function has a
    docstring**, including tests and `__init__.py` files. D203/D213 ignored =
    Google-style blank-line conventions ("no blank line before class",
    "summary on first line").
  - `N803`/`N806` ignored so math-style capitals (`R`, `T`, kernel params) are legal.
  - Per-file ignores for `**/*.ipynb` (ruff lints the notebooks too): B007, B905,
    F811, RUF005, RUF059.
  - Every ignore carries an inline comment naming the rule.

- **Type checking: `ty`** (Astral's checker, pre-1.0), at maximum strictness with
  a pragmatic escape hatch:

  ```toml
  [tool.ty.environment]
  python-version = "3.14"

  [tool.ty.rules]
  all = "error"

  [tool.ty.analysis]
  # JAX's type stubs are imprecise — e.g., jnp.roll, jnp.mgrid, nnx.Param arithmetic
  replace-imports-with-any = ["flax.**"]
  ```

  Everything is an error, but *all of flax* is treated as `Any` because the stubs
  don't model NNX arithmetic. No mypy, no pyright. `ty` is a dev dep only —
  **not run in CI** (yet).

- **No pre-commit config, no justfile, no Makefile.** The workflow is
  `uv sync --all-extras --dev`, then raw `ruff check` / `ruff format` /
  `pytest tests/`, with CI as the enforcement backstop.

- **pytest config** turns deprecations into failures:

  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  filterwarnings = [
      "error::DeprecationWarning",
      "error::FutureWarning",
  ]
  ```

## 3. Testing

- **Framework:** plain pytest + fixtures + `@pytest.mark.parametrize`. No
  hypothesis, no chex, no coverage gate (pytest-cov/xdist are installed but not
  wired into CI flags).

- **Layout mirrors `src/` exactly:** `tests/test_core/test_perceive/test_conv_perceive.py`
  ↔ `src/cax/core/perceive/conv_perceive.py`; one test file per source module,
  `test_` prefix on every directory and file. No `conftest.py` — fixtures are
  defined per-file (`rngs` fixture returning `nnx.Rngs(0)` is re-declared in each).

- **What gets tested:**
  - *Core modules* (perceive/update/nn/utils): construction, output **shapes**,
    custom-parameter plumbing, forward-pass sanity (`jnp.any(perception != 0)`),
    parametrized over sizes/dims.
  - *Complex systems* (`tests/test_cs/`): the signature pattern is a
    **jit-instantiation smoke test** — every system must be constructible under
    `jax.jit`:

    ```python
    def test_life_jit_init() -> None:
        """Test that Life can be instantiated under jax.jit."""
        @jax.jit
        def init_life() -> Life:
            rngs = nnx.Rngs(0)
            ...
            return Life(birth=birth, survival=survival, rngs=rngs)
        try:
            init_life()
        except Exception as e:
            pytest.fail(f"Life instantiation failed under jit: {e}")
    ```
  - *Invariants where they exist*: sandpile mass conservation
    (`jnp.allclose(total_before, total_after)`), color-space round-trips.
  - **Tolerances:** default `jnp.allclose` mostly; explicit `atol=1e-6` for
    color conversions. No rtol discipline, no float64 testing.
  - **No explicit vmap tests**; jit coverage is via the init-under-jit pattern
    plus `nnx.jit` living on the library methods themselves.
  - Tests are fully type-annotated (`-> None`) and docstringed (ruff `D` applies
    to `tests/` via `src = ["src", "tests"]`).

- **CI: three small GitHub Actions workflows**, each path-filtered:
  - `.github/workflows/tests.yml` — on push/PR touching `**.py`, `pyproject.toml`,
    or the workflow itself; matrix `python-version: ["3.12", "3.13", "3.14"]`;
    steps: `astral-sh/setup-uv@v5` → `uv sync --all-extras --dev --python ${{ matrix.python-version }}`
    → `uv run pytest tests/`. Ubuntu only, CPU only, no coverage upload.
  - `.github/workflows/ruff.yml` — `astral-sh/ruff-action@v3` twice (lint, then
    `args: "format --check"`), with a comment linking the upstream issue that
    forces the two-invocation shape.
  - `.github/workflows/docs.yml` — on push to `main`: setup-python 3.13,
    `pip install -e .[docs]`, `mkdocs build --clean`, upload-pages-artifact →
    `deploy-pages@v4` with proper `permissions: pages: write, id-token: write`
    and a `concurrency: group: "pages"` guard.
  - **No release automation** — no PyPI publish workflow; releases are manual
    (tag + build + publish by hand).

## 4. Documentation

- **System:** MkDocs + Material theme + mkdocstrings(python) + mkdocs-jupyter,
  deployed to GitHub Pages. `strict: true` in `mkdocs.yml` so broken refs fail
  the docs build (and therefore CI).

- **Symlink trick — single source of truth:**
  - `docs/index.md -> ../README.md`
  - `docs/contributing.md -> ../.github/CONTRIBUTING.md`
  - `docs/examples -> ../examples`

  README *is* the docs landing page; notebooks *are* the tutorials; nothing is
  duplicated.

- **API pages are stubs** — one short `.md` per module containing a heading plus
  a mkdocstrings directive, e.g. `docs/api/cs/life.md` is two lines:
  `# Conway's Game of Life` / `::: cax.cs.life.cs.Life`. All prose lives in
  docstrings. mkdocstrings options: `docstring_style: google`,
  `show_root_heading: true`, `members_order: source`, `show_source: true`,
  `inherited_members: true`.

- **Docstring style: Google** (`Args:` / `Returns:` / `Attributes:`), enforced
  by ruff `D`. Docstrings are genuinely explanatory — they state shape
  conventions (`(..., *spatial_dims, channel_size)`), dtype contracts
  (`uint8`, `[0, 255]`), and semantics, not just parameter echoes.

- **Notebooks:** `examples/*.ipynb` with a **numeric taxonomy**: `00_` getting
  started, `1x_` discrete CA, `2x_` Lenia family, `3x_` particle systems,
  `4x_` NCA, `5x_` other/advanced. Each is Colab-runnable and rendered into docs
  via mkdocs-jupyter (`execute: false`).

- **README structure**, in order: title → centered logo → centered badge row
  (PyPI pyversions, PyPI version, arXiv paper, X/Twitter) → one-line pitch →
  Overview → Why (Rich / Flexible / Fast / Tested & Documented subsections) →
  "Zoo" table (system | reference | Colab badge, 20 rows) → Getting Started
  (one complete runnable snippet) → Installation (`uv pip install cax`, then
  `pip`) → Citing (bibtex) → Contributing. Emoji in every section header —
  a deliberate, consistent brand voice.

## 5. API style

- **Flax NNX throughout.** Base class `ComplexSystem[State, Input](nnx.Module)`
  (PEP 695 type parameters). Subclasses implement `_step` (single transition)
  and `render`; the base `__call__` is the only driver:

  ```python
  @nnx.jit(static_argnames=("num_steps", "input_in_axis", "sow"))
  def __call__(self, state, input=None, *, num_steps=1, input_in_axis=None, sow=False):
      ...
      state_axes = nnx.StateAxes({nnx.Intermediate: 0, ...: nnx.Carry})
      state = nnx.scan(step_fn, in_axes=(state_axes, nnx.Carry, input_in_axis),
                       out_axes=nnx.Carry, length=num_steps)(self, state, input)
  ```

  Trajectory capture uses `sow(nnx.Intermediate, "state", next_state)` +
  `nnx.pop(cs, nnx.Intermediate)` — not returned stacked arrays. Optional
  `remat: bool = False` class attr wraps the scan body with `nnx.remat`.

- **rngs handling:** constructors take keyword-only `*, rngs: nnx.Rngs` (even
  when unused, for interface uniformity); post-construction randomness uses
  explicit `key: Array` args (`Pool.sample(key, *, batch_size=...)`).

- **Data containers are `@nnx.dataclass` pytrees**, not modules:
  `class LeniaRuleParams(nnx.Pytree)` with `field: Array = nnx.data()` and
  `size: int = nnx.static()`; `@classmethod create(...)` factories; methods
  themselves decorated with `nnx.jit(static_argnames=...)`.

- **Naming:** `snake_case` modules, `CamelCase` classes with role suffixes
  (`ConvPerceive`, `NCAUpdate`, `LifePerceive`); `*_size` for dimensions
  (`channel_size`, `perception_size`), `*_dims` for shape tuples
  (`spatial_dims`), `*_fn` for callables, `state`/`input`/`perception` as the
  universal vocabulary (yes, it shadows the `input` builtin — accepted cost).
  Math symbols keep paper casing (`R`, `T`) thanks to the N803/N806 ignores.

- **`__init__.py` export style:** the top-level `cax/__init__.py` exports
  **nothing** (docstring only!) — users import from subpackages
  (`from cax.cs.life import Life`, `from cax.core.perceive import ConvPerceive`).
  Each subpackage `__init__.py` has a module docstring, explicit relative
  imports, and a **sorted `__all__`** listing exactly the public names. The
  public surface is kept small by re-exporting only curated names at the
  subpackage level; implementation modules stay reachable but undocumented
  unless listed.

- **File organization:** one class per file, tiny files. Framework in
  `core/` (`cs.py`, `perceive/`, `update/`); each concrete system is a
  directory `cs/<name>/` with a fixed internal grammar: `cs.py` (the system),
  `perceive.py`, `update.py`, plus optional `state.py`, `rule.py`, `kernel.py`,
  `growth.py`, `metrics.py`, `patterns/`. Cross-cutting helpers in `nn/`
  (Pool, Buffer, VAE) and `utils/` (render, emoji).

## 6. Repo hygiene

- **LICENSE:** MIT, "Copyright (c) 2024 Maxence Faldor".
- **CHANGELOG:** **none.** History lives in imperative-mood commit messages
  ("Add X", "Fix Y", "Drop Python 3.11 support, require >=3.12") and GitHub
  releases/tags (`v0.2.0` … `v0.3.3`).
- **CONTRIBUTING:** `.github/CONTRIBUTING.md` (symlinked into docs). Covers the
  uv-based fork/branch/PR workflow, bug-report anatomy, and — unusually — a
  "Designing Efficient CAX Architectures" section: an *architectural style
  guide* with code for a `CustomNCA`, best practices (vmap, nnx.Rngs, jit
  compatibility), and common JAX pitfalls with a link to "The Sharp Bits".
  Contains minor typos (`uv ruff check` should be `uv run ruff check`).
- **CITATION.cff:** thorough — software citation (ORCID, keywords, version,
  date-released) *plus* `preferred-citation` for the ICLR 2025 paper with full
  editor list. Version field lags pyproject (0.3.2 vs 0.3.3).
- **Issue/PR templates:** **none.** `.github/` holds only CONTRIBUTING.md and
  `workflows/`.
- **.gitignore:** short and curated with commented sections (Python-generated,
  venv, ruff, mypy, pytest caches, then "Custom": `.env`, `output/`, example
  data dirs, `site/`, `.cache/`). Notably matches the uv-generated template
  style (`*.py[oc]`, `wheels/`).
- **Committed lockfile:** `uv.lock` in the repo root.

---

## Checklist: to feel native alongside cax, iitx must …

- [ ] Use `src/iitx/` layout, hatchling backend, PEP 621 metadata, static
      hand-bumped version, `license`/`license-files` PEP 639 fields.
- [ ] Floor-pin runtime deps with `>=` only (no upper bounds); keep the runtime
      dep list minimal (jax + whatever is truly load-bearing); commit `uv.lock`.
- [ ] `requires-python = ">=3.12"` with 3.12/3.13/3.14 classifiers, and use
      modern syntax (PEP 695 generics, `X | None`).
- [ ] Provide `dev` / `docs` / `examples` extras with the same contents pattern
      (pytest+cov+xdist+ruff+ty; mkdocs stack; notebook deps).
- [ ] Configure ruff in `pyproject.toml`: line-length 100, **tab indentation**,
      `select = ["B","C4","D","E","F","I","N","PERF","PT","RUF","SIM","UP","W"]`,
      the same six documented ignores, per-file ignores for notebooks,
      `src = ["src", "tests"]`.
- [ ] Google-style docstrings on *every* module/class/function including tests
      and `__init__.py`s; docstrings state shape/dtype contracts explicitly.
- [ ] `ty` with `all = "error"` plus targeted `replace-imports-with-any` for
      stub-poor deps; pytest with `filterwarnings = error::DeprecationWarning/FutureWarning`.
- [ ] Mirror `tests/` to `src/` one-to-one; per-file fixtures; parametrize over
      sizes/dims; include an instantiate-under-`jax.jit` smoke test for every
      public top-level object; `jnp.allclose` with explicit `atol` where floats
      round-trip.
- [ ] Three path-filtered GitHub workflows: tests (uv, 3-version matrix),
      ruff (lint + format --check), docs (mkdocs build strict → GitHub Pages).
- [ ] MkDocs Material + mkdocstrings(google) + mkdocs-jupyter with
      `strict: true`; symlink `docs/index.md → README.md`,
      `docs/contributing.md → .github/CONTRIBUTING.md`, `docs/examples → examples/`;
      API pages as two-line `:::` stubs.
- [ ] Numbered example notebooks (`00_getting_started.ipynb`, then themed
      decades) with Colab badges, all rendered into docs unexecuted.
- [ ] README in the cax shape: logo, badge row, pitch, Why, feature/zoo table,
      runnable quickstart, uv-first install, bibtex citation, contributing link.
- [ ] MIT LICENSE, CITATION.cff (software + preferred-citation), CONTRIBUTING
      with a domain-specific architecture guide, short commented .gitignore.
- [ ] Empty (docstring-only) top-level `__init__.py`; curated sorted `__all__`
      per subpackage; one class per small file; a fixed per-subpackage file
      grammar analogous to cax's `cs.py`/`perceive.py`/`update.py`.
- [ ] Keep commits imperative-mood ("Add …", "Fix …", "Drop …"); tag releases
      `vX.Y.Z`; no CHANGELOG file (releases + commits carry history) — or
      deliberately improve on cax by adding one.

### Deliberate deviations to consider for iitx (pure functions over TPMs)

- **Plain JAX + pytrees instead of NNX modules** for the core: cax needs NNX
  because its systems carry learned parameters, `nnx.Rngs`, `sow`/`Intermediate`
  and `nnx.scan` training machinery. A TPM library's core objects are data, not
  models — `@nnx.dataclass(nnx.Pytree)` buys nothing over
  `jax.tree_util.register_dataclass` / plain frozen dataclasses, and it drags
  flax into the runtime deps (and forces `replace-imports-with-any = ["flax.**"]`
  in ty). Keep flax out of `dependencies` unless an `iitx.nn`-style optional
  layer genuinely trains something; if one appears, adopt cax's NNX idioms
  (`*, rngs: nnx.Rngs`, `nnx.jit`, sow) exactly there and only there.
- **Return trajectories from `lax.scan` directly** rather than the
  sow/`nnx.Intermediate`/`nnx.pop` pattern — that pattern exists to thread
  intermediates through NNX module state; pure functions don't need it.
- **Numerical testing must be stricter than cax's**: TPM math (stochastic-matrix
  invariants, stationary distributions, information quantities) warrants
  explicit rtol/atol policy, float64 (`jax.enable_x64`) test coverage, and
  property-style invariant tests (rows sum to 1, non-negativity, symmetry) —
  cax's shape-and-smoke level is not enough for a numerics-first library.
  Add explicit `jax.vmap` and grad tests, which cax skips.
- **Add what cax lacks** where it costs little: a `ty` CI job (deps already
  installed), a trusted-publisher PyPI release workflow triggered on tags,
  coverage reporting (pytest-cov is already in `dev`), and a CHANGELOG or
  GitHub-release discipline — none of these break the "native" feel.
- Everything else — tabs, ruff rule set, Google docstrings, symlinked docs,
  numbered notebooks, README shape, empty top-level `__init__.py` — should be
  copied verbatim.
