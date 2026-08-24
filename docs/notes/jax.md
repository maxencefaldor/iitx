# JAX engineering notes for `iitx`

Verified against live docs (docs.jax.dev, flax.readthedocs.io, ott-jax.readthedocs.io) and
empirically against **jax 0.11.1 / flax 0.12.9** on 2026-08-24. Doc URLs cited inline;
claims marked *(verified empirically)* were run locally.

Workload recap: n-unit discrete systems as TPMs; enumerate subsets (~2^n candidate
systems, mechanisms, purviews), partitions of small sets, and system cuts; per element
compute repertoires of shape `(2,)*n` (or products of unit alphabets), distances (EMD /
intrinsic difference), and min/max over the enumerated spaces. n is small (IIT is
exponential; n ≲ 10–12 in practice), so per-element arrays are tiny and the win comes
from batching the enumeration, not from big tensors.

---

## 1. Static shapes & combinatorial enumeration

JAX traces to a jaxpr keyed on **shape/dtype and static-arg values**; anything that
varies shape triggers recompilation
([jit-compilation](https://docs.jax.dev/en/latest/jit-compilation.html)).
Consequences for combinatorial enumeration:

**Enumerate in NumPy at trace time; compute in JAX over stacked masks.**
- Build enumeration tables (all subsets, all bipartitions/tripartitions of a k-set, all
  states) as **plain NumPy arrays at module/build time**. They are closed over by the
  jitted function and become compile-time constants — no tracing cost, no host→device
  transfer per call.
- Represent a subset of n units as a length-n boolean/0-1 mask row; a space of subsets is
  a `(num_subsets, n)` mask matrix. Then `vmap` the per-subset kernel over the leading
  axis. This gives one XLA program for the whole space instead of a Python loop that
  unrolls into a huge graph.
- Partitions: enumerate them in Python (e.g. all ways to split a mechanism/purview pair),
  encode each as fixed-width integer/mask arrays, stack, and `vmap` or `lax.scan` over
  the stack.

**Masking over ragged structures, always.** JAX has no ragged arrays; boolean indexing
that would produce a data-dependent shape is a compile error under jit
(`"Array boolean indices must be concrete"`,
[Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)).
So a "purview of size k ≤ n" is an n-wide mask, never a k-vector:
- min-reductions: pad excluded entries with `+inf` (`jnp.where(mask, x, jnp.inf)` then
  `jnp.min`); max with `-inf`; products with `1.0`; sums with `0.0`.
- *(verified empirically)* gradients flow correctly through the where-masked pattern:
  `grad(min(where(mask, x, inf)))` gives zero cotangent to masked-out entries and splits
  the rest among ties (see §4).
- Beware the where-of-where NaN trap: if the *branch not taken* can produce `nan`/`inf`
  under differentiation, the adjoint of `jnp.where` accumulates it. Standard fix is the
  "double where" — make the unused branch's *input* safe before the op
  ([FAQ, gradients with where](https://docs.jax.dev/en/latest/faq.html)). Relevant for
  `x * log(x/y)`-type intrinsic-difference terms: use
  `jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0) / q), 0.0)`.

**Cost model: few big masked shapes beat many small exact shapes.**
The jit cache is keyed on (function identity, input shapes/dtypes, static-arg values);
every distinct shape or static value is a fresh XLA compile, typically 10s–100s of ms
even for tiny kernels ([jit-compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
documents a 429 ms vs 3.1 ms cached example). With per-size exact shapes you'd compile
O(n) or O(2^n) variants; with masks you compile **one kernel per system size n** (the
`(2,)*n` repertoire shape still depends on n — unavoidable). Since a session touches only
a handful of n values, shape-specialized compilation *on n only* is fine; specialize on
nothing finer.
- Do **not** wrap kernels in fresh `partial`/`lambda` per call — that defeats the cache,
  which relies on function identity. Hold jitted functions at module scope, or memoize
  builders with `functools.lru_cache` keyed on `n`.
- `static_argnums`/`static_argnames` are appropriate exactly for `n`-like integers with
  few distinct values ([jit-compilation](https://docs.jax.dev/en/latest/jit-compilation.html)).

**Python loop vs `vmap` vs `scan` over the stack:**
- Default: `vmap` over stacked masks — fully parallel, one program.
- If the enumerated space is large enough that materializing all intermediates at once
  blows memory (e.g. all purviews × all partitions × repertoire), use `lax.scan` over
  chunks of the stacked masks: scan lowers to a single `WhileOp`, keeping the compiled
  program small, vs Python loops which "are unrolled, leading to large XLA computations"
  ([lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)). Carry must
  have fixed shape/dtype across iterations — running min/argmin accumulators fit this
  naturally.
- `scan(..., unroll=k)` trades compile time for runtime if the body is tiny.

## 2. Precision

- Defaults are 32-bit; `jax_enable_x64` flips defaults to `int64`/`float64`. Set via
  `jax.config.update("jax_enable_x64", True)` at startup or `JAX_ENABLE_X64=1`
  ([Default dtypes and the X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html),
  [Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)).
- The docs are explicit that the flag "is intended as a **global setting** that should
  have one value for your whole program", and that scoped enabling is deliberately not
  supported "within JAX's programming model, where code execution may happen in a
  different context than code compilation"
  ([default_dtypes](https://docs.jax.dev/en/latest/default_dtypes.html)). A
  `jax.enable_x64()` context manager exists
  ([jax.enable_x64](https://docs.jax.dev/en/latest/_autosummary/jax.enable_x64.html))
  *(verified: works in eager code, no deprecation warning; it supersedes the deprecated
  `jax.experimental.enable_x64`)*, but it has a long history of surprises under `jit`
  (jax-ml/jax #5982, #7336) — treat it as test-suite tooling only.
- **Policy for iitx**: the library must never flip the flag itself (it's global — a
  library mutating it breaks the host program). Instead: (a) write dtype-explicit code —
  take the dtype from the input TPM and preserve it; (b) document that matching a NumPy/
  PyPhi float64 oracle to ~1e-6 requires the user (or our test conftest) to enable x64;
  (c) enable x64 in `tests/conftest.py` before any JAX import-time array creation.
- Oracle matching: EMD/ID values are sums over ≤ 2^n terms of well-conditioned
  quantities; float64 matches a NumPy oracle to ~1e-13, and float32 typically to ~1e-6
  *per distance* — but **argmin selection near-degenerate φ values can flip** between
  float32 and float64. So: float32 is safe for gradients/optimization loops (where you
  differentiate through softened objectives anyway), float64 is required when the
  discrete certificate (which purview/partition is the MIP) must match the oracle.
- Accelerator float64: GPUs run f64 at 1/2 (data-center) to 1/32–1/64 (consumer) of f32
  throughput, and pathological cases exist (a reported ~400× slowdown for f64 scatter-add
  on GPU, jax-ml/jax #3270); **TPUs have no native f64** (TPU is bfloat16/f32-oriented;
  enabling x64 on TPU errors or falls back — jax-ml/jax discussion #9828). Plan: f64 on
  CPU for exactness, f32 on accelerators for scale/gradients.

## 3. Relevant primitives

- **`vmap` composes and nests freely**: `vmap(vmap(f))` for (system × mechanism),
  (mechanism × purview), etc. Inner structure stays a single program. `in_axes=None`
  broadcasts the shared TPM without copying. Prefer nesting vmaps over manually
  flattening index products — clearer and XLA fuses it the same.
- **`lax.scan`** for sequential reductions (running min over partitions when memory
  matters) and for anything that must not unroll; see §1.
- **`jax.checkpoint` / `jax.remat`**: docs now say it "often isn't needed for
  differentiated functions under a `jax.jit()`" — XLA already optimizes; it matters
  mainly around **staged control flow like `lax.scan`**, where you should remat the scan
  body if reverse-mode residuals over many iterations blow memory
  ([gradient-checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)).
  For iitx: only relevant if we differentiate through a scan over a large
  partition/purview space.
- **Conditioning on states**: a repertoire is `(2,)*n`; conditioning on a state of a
  subset = indexing some axes at fixed values. Under jit, use `jnp.take`/
  `jnp.take_along_axis` or precomputed **flat gather indices** (flatten to `(2**n,)`,
  gather with an integer index table built at trace time — that table is exactly the
  "enumerate states of the subset" table from §1). `jnp.take` has no NumPy `"raise"`
  mode; out-of-bounds is `mode="fill"` (default; NaN/sentinel, configurable
  `fill_value`) or `"clip"`
  ([jnp.take](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html)). Set
  `mode="clip"` (or better: guarantee validity by construction) so a masked/padded index
  never injects NaN. `unique_indices`/`indices_are_sorted` are perf hints — safe to pass
  when the index tables guarantee them.
- **Marginalization over partition blocks**: `jax.ops.segment_sum(data, segment_ids,
  num_segments=...)` — `num_segments` **must be static under jit** since it fixes the
  output shape ([segment_sum](https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html));
  out-of-range ids are silently dropped, which combines nicely with mask padding (send
  masked entries to id `num_segments`). `segment_min/max/prod` exist. For `(2,)*n`
  repertoires, an equivalent (often faster) pattern is reshape + `sum` over the
  marginalized axes; segment ops earn their keep for the state-space (flat `(2**n,)`)
  view with precomputed block ids.
- **Ties & determinism**: `jnp.argmin` documents "When the minimum value occurs more
  than once along a particular axis, the smallest index is returned"
  ([jnp.argmin](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html))
  *(verified: `argmin([2,1,1,1]) == 1` under jit)*. This is the determinism hook: order
  the enumeration tables canonically (e.g. lexicographic by subset bitmask, PyPhi order
  if oracle-matching), and first-occurrence argmin gives a reproducible MIP across
  backends. `jax.lax.top_k` now has `is_stable=True` **by default** ("equal elements
  preserve their relative order from the input",
  [lax.top_k](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html)); *(verified:
  `top_k([1,3,3,2], 2) → indices [1,2]`)*. `jnp.sort/argsort` default `stable=True`
  (`kind` is deprecated in favor of `stable=`,
  [jnp.argsort](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html)).
  For composite tie-breaks (min φ, then smallest purview, then lex order), fold the
  tie-break into the key: `key = phi * K + tiebreak_rank` with exact integer ranks, or
  lexicographic argmin via `jnp.lexsort`-style stable sorting. Floating-point equality
  ties from *independent* computations are still at the mercy of non-associative
  reduction order — where oracle-identical certificates matter, compare within an
  epsilon and resolve by canonical rank, not by raw float equality.
- NaN corner: `jnp.nanargmin` returns `-1` for all-NaN slices instead of raising
  ([Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)) —
  don't let all-masked slices reach argmin; keep the `+inf` padding so the min is finite.

## 4. Differentiability

- **`jnp.min`/`jnp.max` over an axis**: JVP rule is `_reduce_chooser_jvp_rule` in
  `jax/_src/lax/lax.py` — it computes tie indicators (`_eq_meet(operand, ans)`), counts
  them, and returns `sum(g * indicators) / counts`: **the (co)tangent is split equally
  among all tied extrema**. Elementwise `minimum/maximum` use `_balanced_eq` with the
  same even-split convention. *(verified empirically: `grad(min)([1,1,2]) == [0.5, 0.5, 0]`;
  `grad(minimum)(1,1) == (0.5, 0.5)`.)* So `min` gives a valid (sub)gradient element of
  the convex hull, and is differentiable-almost-everywhere; φ = min over partitions is
  therefore directly usable in `grad` — piecewise-smooth, with the even-split convention
  exactly at switching points.
- **`argmin` itself is non-differentiable** (integer output, zero/undefined gradient).
  Pattern: compute `phi = jnp.min(costs)` for the value (differentiable) and
  `idx = jnp.argmin(costs)` for the certificate (reporting only) — never gather-by-argmin
  when you want gradients, since `costs[argmin]` gives a one-hot gradient with
  *arbitrary-looking* (first-occurrence) tie selection instead of the balanced split.
- **Softmin relaxation** when a smooth, strictly-differentiable surrogate is wanted:
  `-tau * logsumexp(-costs/tau)` (use `jax.nn.logsumexp` with `b=mask` or `-inf`
  padding to respect masks). Anneal `tau → 0` recovers the hard min. Expose `tau` as an
  argument; hard min is `tau=0` semantics via a separate code path, not a limit.
- **Straight-through**: `hard + jax.lax.stop_gradient(soft_or_value - hard)`-style, or
  value-hard/gradient-soft: `soft + stop_gradient(hard - soft)` — forward pass returns
  the exact min, backward uses softmin's gradient. For anything fancier (e.g. custom
  tie conventions), `jax.custom_vjp`
  ([custom derivatives](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)).
- **EMD**: exact EMD is a linear program — not expressible in differentiable jaxpr form.
  Three tiers:
  1. *Exact, non-differentiable*: scipy/POT via `jax.pure_callback` (§7) — for IIT 3.0
     oracle parity on CPU. No gradients ("pure callbacks are not differentiable",
     [jax.pure_callback](https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html)).
     (The LP solution *is* differentiable a.e. via Danskin/envelope — the gradient of the
     objective w.r.t. the cost matrix is the optimal plan — so a `custom_vjp` around the
     callback returning the plan is a viable later upgrade.)
  2. *Entropic relaxation*: **`ott-jax`** — Sinkhorn with unrolling or **implicit
     differentiation** built in ([ott-jax docs](https://ott-jax.readthedocs.io/)).
     Histogram-to-histogram OT with an explicit cost matrix is supported via
     `ott.geometry.geometry.Geometry(cost_matrix=...)` +
     `ott.solvers.linear.sinkhorn.Sinkhorn` (docs note that with
     `max_iterations == min_iterations` it runs as a `lax.scan`, i.e. fixed-length and
     cleanly jittable/vmappable). Status: **0.6.0 released 2025-11-04** (PyPI; deps
     `jax>=0.4.0`, jaxopt, lineax, optax; `python>=3.9`), Apple-led team with active
     2026 development (Sinkhorn-variant PRs in May 2026), 1000+ commits — a healthy,
     reasonable **optional dependency** (`iitx[ot]`). Don't reimplement Sinkhorn.
  3. IIT 4.0's intrinsic difference is a pointwise max of KL-like terms — no LP,
     directly differentiable (with the double-where guard from §1), so the ott/callback
     machinery is only needed for the 3.0 EMD path.

## 5. Flax NNX in 2026 — and whether iitx needs it

- Status per [flax.readthedocs.io](https://flax.readthedocs.io/en/latest/): **NNX is the
  recommended API for new users**; Linen is *not* deprecated ("Flax Linen API is not
  going to be deprecated in the near future as most of Flax users still rely on this
  API") but is effectively in maintenance mode. NNX gives Pythonic reference semantics,
  mutable `nnx.Module`s, easier inspection/debugging — all aimed at **stateful neural
  network training** (params, optimizers, RNG state, mutation).
- **iitx has none of that.** A TPM analysis library is pure functions over arrays: no
  learned parameters, no mutable state, no RNG threading. NNX's reference semantics
  actively fight `jit`-cache friendliness and add a heavyweight dependency for zero
  benefit. **Recommendation: no Flax dependency in the core.**
- **Structured data instead**: frozen `dataclasses.dataclass` +
  **`jax.tree_util.register_dataclass`** — this is the in-core, recommended way to make
  a dataclass a pytree
  ([register_dataclass](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html)).
  Since jax v0.4.36, fields are inferred: annotate static fields with
  `dataclasses.field(metadata=dict(static=True))` (must be hashable — they become part
  of the treedef and hence the jit cache key), everything else is a data leaf.
  *(verified empirically: frozen dataclass with `tpm: jax.Array` data field and
  `n: int` static field round-trips through `jit` and `tree_leaves` correctly.)*

  ```python
  @jax.tree_util.register_dataclass
  @dataclasses.dataclass(frozen=True)
  class System:
  	tpm: jax.Array  # data leaf
  	state: jax.Array  # data leaf
  	n: int = dataclasses.field(metadata=dict(static=True))  # static / cache key
  ```
- `flax.struct.dataclass` / `flax.struct.PyTreeNode` (`pytree_node=False` for static
  fields, `.replace()` for updates,
  [flax.struct](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html))
  and equinox `Module`s are fine equivalents but each drags a dependency; stdlib
  `dataclasses.replace()` covers the update ergonomics. If a `train an optimal-TPM`
  example ever needs a NN, put it in `examples/` with NNX or equinox there — not in core.

## 6. Ecosystem conventions

- **jaxtyping** for shape/dtype annotations
  ([array API](https://docs.kidger.site/jaxtyping/api/array/)):
  `Float[Array, "states units"]`, `""` for scalars, symbolic dims shared across a
  signature (`"2**n"`-style expressions are supported as symbolic expressions), `*batch`
  variadic dims for vmap-polymorphic signatures, `#dim` for broadcastable. Runtime
  checking ([runtime docs](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/)):
  `@jaxtyped(typechecker=beartype.beartype)` per function, or codebase-wide
  `install_import_hook`; in tests, `pytest --jaxtyping-packages=iitx,beartype.beartype`
  (or the `pyproject.toml` equivalent). Caveats: **no
  `from __future__ import annotations`** in annotated modules; checks run at trace time
  only, so zero runtime cost under jit. Recommendation: annotations everywhere, hook
  enabled in CI, beartype as a test-only dependency.
- **chex** ([docs](https://chex.readthedocs.io/en/latest/),
  [README](https://github.com/google-deepmind/chex)) — DeepMind-maintained, active.
  Use: `chex.assert_trees_all_close` for oracle comparisons (tolerance kwargs),
  `chex.assert_shape`/`assert_tree_all_finite` as internal invariants,
  `chex.assert_max_traces(n=1)` to catch jit-cache regressions (the §1 cost model,
  enforced in CI). **`@chex.variants` requires the test class to inherit
  `chex.TestCase`**; then `self.variant(fn)` runs each test `with_jit`/`without_jit`
  (also `with_device`/`without_device`/`with_pmap`) — the cheapest way to guarantee
  eager and jitted paths agree. chex is absltest-flavored but chex.TestCase classes
  collect fine under pytest; complement with plain-pytest `parametrize` over
  `[f, jax.jit(f), jax.vmap(...)]` for function-level checks.
- Oracle tests: PyPhi (NumPy, float64) as reference fixtures; assert φ values with
  `assert_trees_all_close(..., atol=1e-6)` under x64, and assert MIP *certificates*
  (indices) exactly, relying on §3 canonical ordering.

## 7. Sharp bits refresher (this workload)

([Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) unless noted)

- **No boolean-mask indexing under jit** (dynamic result shape). Use `jnp.where` +
  padding (§1). `x[x > 0]` is the error to grep for in review.
- **Dynamic slice constraints**: slice *sizes* must be static inside jit/`while_loop`/
  `fori_loop`; start indices may be traced. OOB *reads* clamp to bounds; OOB *scatter
  updates are dropped* — both silent, so validate index tables at build time.
- **x64 is global** (§2): flag at startup; `jax.enable_x64()` context manager exists but
  is not reliable around jit; library code never touches the flag; be dtype-explicit.
- **Donation**: `jax.jit(..., donate_argnums=...)` reuses input buffers for outputs of
  matching shape/dtype; donated buffers are dead afterwards; keyword args unsupported
  ([buffer_donation](https://docs.jax.dev/en/latest/buffer_donation.html)). For iitx this
  is a micro-optimization at best (the big buffer is the read-only TPM, which must *not*
  be donated since every element reads it); ignore until profiling says otherwise.
- **jit cache** keyed on function identity + shape/dtype + static values; module-level
  jitted callables or `lru_cache`d builders keyed on `n`; `chex.assert_max_traces` in CI
  (§1, §6).
- **`jax.pure_callback`** — the exact-EMD escape hatch
  ([jax.pure_callback](https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html)):
  stable (non-experimental) API,
  `jax.pure_callback(fn, result_shape_dtypes, *args, vmap_method=...)`; requires
  `jax.ShapeDtypeStruct` outputs; callback must be pure (may be called multiple times or
  elided); **must set `vmap_method` explicitly** (`"sequential"` is the honest choice
  for a scipy LP; the unspecified default now raises); runs on CPU, forces
  device↔host sync, and blocks async dispatch — keep it out of hot vmapped inner loops
  and prefer batching one callback over the whole enumerated space. Not differentiable.
  `jax.experimental.io_callback` / `jax.debug.callback` are the impure/debug siblings.
- Reduction order is non-associative floating point: identical math on CPU vs GPU can
  differ in ulps, which matters only where it flips an argmin — handled by epsilon-aware
  canonical tie-breaking (§3), not by chasing bitwise equality.
