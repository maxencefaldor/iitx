# iitx design

`iitx` computes integrated information — the full cause-effect structure of IIT, and Φ —
for any finite discrete dynamical system, in JAX. This document fixes the core
abstractions, the data model, the API surface, the parallelization strategy, and the
scope of the first release. It is the contract for the implementation: everything here
was chosen from the mathematics (`docs/notes/iit-3.0.md`, `docs/notes/iit-4.0.md`), the
oracle's conventions (`docs/notes/pyphi.md`), and JAX's constraints
(`docs/notes/jax.md`). The vocabulary is fixed in `docs/glossary.md` and used
identically in code, docstrings, docs, and tests.

## 1. What the library must do

1. **Parallelize within a system.** One Φ computation is a large space of independent
   work — candidate systems, mechanisms, purviews, partitions, cuts, repertoires,
   distances. All of it becomes batched tensor work on the accelerator.
2. **Batch across systems.** Every public function is a pure function of
   `(system, state)` and composes with `jax.vmap` and `jax.jit`.
3. **Differentiate.** Gradients w.r.t. TPM entries flow through everything that is
   differentiable in principle (the IIT 4.0 path, almost everywhere), and fail loudly
   where they are not (exact EMD).
4. **Compose.** A measure of integrated information is a first-class object; IIT 3.0 and
   IIT 4.0 are two instances of one interface, and alternative measures plug in without
   touching the core.

Non-goals for 0.1 are listed in §13.

## 2. Design principles

These follow from the constraints and are applied everywhere; they are the reasons the
code looks the way it does.

- **P1 — State-by-state TPMs over heterogeneous alphabets are the common currency.**
  The core object is `p(next | prev)` over mixed-radix states with per-unit alphabet
  sizes `shape = (q_1, …, q_n)`. Nothing in the core assumes binary units or `2**n`
  anywhere. Conditional independence — required by mechanism-level IIT — is a *checked
  property* used to derive the factored (per-unit) view, never an unchecked assumption
  and never silently enforced by lossy conversion (PyPhi's sbs→sbn conversion silently
  destroys dependencies; iitx refuses instead).
- **P2 — Masks, not ragged structures.** A subset of units (candidate system, mechanism,
  purview, partition part) is a length-`n` mask. All combinatorial spaces are enumerated
  in NumPy at build time as stacked fixed-width mask/index tables that become
  compile-time constants; kernels `vmap`/`scan` over them. There is exactly **one
  compiled kernel per `(n, shape)` signature** — never per subset, never per size.
- **P3 — Full-shape embedding.** Every distribution-like array has the full state-space
  shape (`shape` as tensor dims, or its flattened size `Q = ∏ q_i`), regardless of which
  purview it concerns. Units outside the purview are carried as *known constant factors*
  (identity `1` for potentials, uniform for probabilities — §6.1), so shapes are uniform
  across an enumeration axis and `vmap` applies. Sums and normalizations are
  multiplicity-corrected using masked constants.
- **P4 — Values by reduction, certificates by canonical order.** φ values come from
  masked `min`/`max` reductions (differentiable, with exact even-split subgradients at
  ties); the *identity* of the optimum (which partition, which purview) comes from
  first-occurrence `argmin`/`argmax` over canonically ordered enumeration tables, with
  PyPhi-compatible composite tie keys and round-to-precision-before-compare semantics.
  Results are bit-for-bit reproducible on a fixed backend.
- **P5 — Struct-of-arrays results.** A cause-effect structure is not a list of Python
  objects; it is a pytree of stacked arrays (mechanism masks, purview masks, φ values,
  repertoires) with an existence mask. Batchable, differentiable, serializable; pretty
  `__repr__` for humans.
- **P6 — Dtype-polymorphic; the user owns x64.** The library never touches
  `jax_enable_x64`; it inherits the TPM's dtype. Oracle-parity tests enable x64 in their
  own conftest. Exact certificates that must match the oracle live on CPU/float64.
- **P7 — Exact first; approximations are named.** 0.1 computes exactly. Anything
  approximate arrives later under a name, type, and docstring that say so.
- **P8 — The core knows no provenance.** No dependency on cax, no grids, no
  neighbourhoods, no adapters in the core. Anything that builds a `System` from a CA, a
  circuit, or a network lives at the edge.

## 3. Data model

All containers are frozen `dataclasses` registered with
`jax.tree_util.register_dataclass`; array fields are pytree leaves, structural fields
(`shape`, table sizes) are static metadata and part of the jit cache key.

### 3.1 `System`

```python
@register_dataclass
@dataclass(frozen=True)
class System:
	tpm: Float[Array, "Q Q"]        # p(next | prev); rows = prev state, little-endian mixed radix
	cm: Bool[Array, "n n"]          # connectivity, cm[i, j] = unit i inputs to unit j
	shape: tuple[int, ...]          # static: per-unit alphabet sizes (q_1, …, q_n)
```

- `tpm[prev, next]` is the **interventional** transition probability
  `p(next | do(prev))`; rows sum to 1. Deterministic systems are the 0/1 special case.
- `shape` is static and equals the tensor shape of any full state-space array
  (`Q = math.prod(shape)`).
- `cm` defaults to all-ones (PyPhi's convention). It participates in results (strong
  connectivity short-circuits, purview reducibility pruning), so it is data, not
  decoration.
- Constructors: `System(tpm, shape, cm=None)` (state-by-state, the general form);
  `System.from_node_tpms(node_tpms, cm=None)` (tuple of per-unit conditionals
  `p(u_i' | prev)` with shapes `(*shape, q_i)` — conditionally independent by
  construction); `System.from_state_by_node(tpm, cm=None)` (binary `(Q, n)`
  convenience, PyPhi's most common input form).
- `node_tpms(system)` derives the factored view, validating conditional independence
  (`check_independence`); it raises at the Python boundary if the TPM does not
  factorize. Mechanism-level analysis consumes the factored view; transforms (§10) and
  system-level 4.0 quantities consume `tpm` directly.

### 3.2 States

A state is an `Int[Array, "n"]` vector of per-unit values (vmappable and
`Q`-independent); `ravel_state`/`unravel_state` convert to/from the little-endian
mixed-radix flat index (unit 0 is the fastest-varying digit — PyPhi's LOLI convention,
generalized). All state enumeration tables follow this order.

### 3.3 `Direction`

`Direction.CAUSE` / `Direction.EFFECT` (an `IntEnum`, static). Every quantity that
exists on both temporal sides is one function taking a `Direction`, so cause and effect
mirror each other structurally, not by parallel code paths.

### 3.4 Enumeration tables

Built once per `(n, shape)` in NumPy (module `iitx.enumeration`), closed over by jitted
kernels:

- `subsets(n)` — `(2**n, n)` bool, canonical order = increasing size, lexicographic
  within size (PyPhi's powerset order; certificates depend on it).
- `states(shape)` — `(Q, n)` int, little-endian mixed radix.
- `mechanism_partitions(scheme, n)` — padded part-assignment tables for `Θ(M, Z)`
  (IIT 4.0 `ALL`) and 3.0 bipartitions (`BI`), instantiated against `(M, Z)` masks at
  trace time; invalid/padding rows carry a validity mask.
- `system_partitions(scheme, n)` — 3.0 directed bipartitions (`2**n − 2` cuts); 4.0
  `SET_UNI/BI` set-partitions × per-part direction flags, with each partition also
  carrying its precomputed normalization (`Σ_i |S^(i)||X^(i)|` for 4.0).

Enumeration order replicates the oracle's generator order per scheme, so tie
certificates agree (see §7).

### 3.5 Results

Result pytrees, all mask-carrying struct-of-arrays (fields abridged):

- `SystemPhi` — `phi` (φ_s or Φ^3.0), `partition_index`, per-direction φ, and for 4.0
  the specified states `(cause_state, effect_state)` with their `ii`.
- `Distinctions` (4.0) / `Concepts` (3.0) — stacked per-mechanism: mechanism mask,
  cause/effect purview masks, specified states (4.0) or repertoires (3.0), φ values,
  existence mask.
- `PhiStructure` (4.0) — `SystemPhi` + `Distinctions` + `sum_phi_r` + `num_relations` +
  `big_phi`.
- `CauseEffectStructure` (3.0) — `Concepts` + expanded repertoires needed for Φ.
- `Complexes` — condensation output: per-complex subset masks, φ, order found.

## 4. Conventions (the oracle-compatibility contract)

Every convention that a reimplementation can silently get wrong, fixed here once:

1. **State order**: little-endian mixed radix everywhere (§3.2). PyPhi-identical for
   binary.
2. **Connectivity direction**: `cm[i, j] = 1` ⇔ edge i → j.
3. **Background conditions**: IIT 3.0 freezes external units at the current state for
   both directions (PyPhi 1.x / `iit3` preset). IIT 4.0 conditions the effect TPM on the
   current background state and builds the backward cause TPM by causal marginalization
   conditional on the current universe state (4.0 paper Eq. 4). Both are pure functions
   `(System, state, subset_mask) → conditioned tensors`.
4. **Mechanism marginalization**: non-purview inputs are uniformly marginalized *per
   unit factor before products* (the virtual-elements / product-probability doctrine of
   both theories); cause-side joint products are renormalized, effect-side products are
   already normalized.
5. **Precision-then-compare**: every distance/φ is quantized to the measure's precision
   (`1e-6` for IIT 3.0, `1e-13` for IIT 4.0) before any comparison; ties are defined at
   that precision, exactly as in PyPhi.
6. **Tie policy** (§7): documented composite keys per selection point, replicating the
   oracle's effective behaviour; where the oracle is nondeterministic, iitx defines a
   deterministic order and documents the divergence.
7. **EMD asymmetry (3.0)**: cause-side φ uses the full Hamming-metric EMD; effect-side φ
   uses the analytic per-unit form `Σ_i EMD_1(marginal_i)` — these are *not* equal on
   correlated repertoires, and PyPhi's numbers require the asymmetry.
8. **Unreachable states**: validation (`state` reachable given frozen background;
   probabilities well-formed) happens at the Python boundary, optional and on by
   default, never inside jitted kernels.

## 5. The measure interface

A **measure** is a frozen dataclass implementing the `Measure` protocol:

```python
class Measure(Protocol):
	def phi(self, system: System, state: Int[Array, "n"]) -> Float[Array, ""]: ...
	def analyze(self, system: System, state: Int[Array, "n"]) -> Any: ...   # full result pytree
```

`phi` is the headline scalar (φ_s for IIT 4.0, Φ for IIT 3.0 — each measure documents
which); `analyze` returns the full result. Both are jit/vmap/grad-transparent.

The two shipped measures are configurations over shared components:

```python
@register_dataclass
@dataclass(frozen=True)
class IIT4(Measure):
	mechanism_partitions: str = "all"       # Θ(M, Z), Eq. 38
	system_partitions: str = "set_uni_bi"   # Θ(S) with direction flags
	precision: float = 1e-13
	# comparator is the generalized intrinsic difference; fixed by the theory version

@register_dataclass
@dataclass(frozen=True)
class IIT3(Measure):
	mechanism_partitions: str = "bi"
	system_partitions: str = "directed_bi"
	precision: float = 1e-6
	# comparator is EMD at the mechanism level, XEMD at the system level
```

The pluggability seams — established by the measure survey
(`docs/notes/measures.md` §3) — are: **repertoire kernel** (which distributions are
built, with which conditioning convention), **partition scheme** (the cut lattice and
the semantics of a cut), **comparator** (the divergence), **aggregator/tie policy**
(min/max structure, precision, tie keys). Components live in registries keyed by name
(`iitx.enumeration` for partition schemes, `iitx.distances` for comparators); a new
scheme or distance is a new registry entry, not a core change. Two structural facts the
interface honours because the survey shows they are unavoidable:

- **Recursion**: Φ^3.0's system-level comparator consumes whole cause-effect
  structures (each built by the inner mechanism-level pipeline). The 3.0 measure is
  therefore internally two nested pipelines; the interface does not pretend otherwise.
- **Extra inputs**: measures needing stationary distributions or empirical covariances
  (Φ\*, Φ_G, ΦID…) take them as *declared additional inputs* via adapter front-ends in
  later releases; the core `Measure` signature stays `(System, state)`.

0.1 ships `IIT3` and `IIT4` in their canonical configurations only (§13); the registries
exist, but no non-canonical combination is implemented or tested.

## 6. The computation, and which axes parallelize

### 6.1 Repertoire algebra (shared kernel layer)

The primitive layer (`iitx.repertoires`) is mask-parameterized tensor algebra over
full-shape arrays:

- per-unit likelihood/conditional factors gathered from the factored TPM
  (`jnp.take` on precomputed little-endian gather tables, `mode="clip"`);
- `marginalize(x, mask)` — uniform mean over masked axes, keepdims (broadcast back);
- masked products, multiplicity-corrected sums and normalizations;
- two embeddings with explicit converters: **potential embedding** (non-purview dims
  constant `1`; used for products, 4.0 selectivities, and argmax over purview states —
  values exact, non-purview dims degenerate and masked out of certificates) and
  **probability embedding** (non-purview dims uniform; used for EMD and repertoire
  expansion — exact for EMD because W₁ with a per-unit-separable ground metric adds
  zero cost on identically-uniform independent dims);
- the double-`where` guard on every `p · log(p/q)` term so gradients never meet
  `0 · log 0` NaNs.

Everything downstream is composition of these primitives.

### 6.2 Axis-by-axis parallelization

| Axis | Cardinality | Treatment |
|---|---|---|
| purview/system states | `Q = ∏ q_i` | dense tensor dimensions — always parallel |
| mechanisms × purviews | `(2**n − 1)²` pairs per direction | `vmap` over mask tables, `scan`-chunked for memory |
| mechanism partitions | up to ~`2^{|M|+|Z|}` (3.0 BI) / super-exponential (4.0 ALL) | padded tables; `scan` with running (min, argmin) carry — nothing materialized per partition beyond the chunk |
| system partitions / cuts | `2**n − 2` (3.0) / set-partitions × flags (4.0) | `vmap` over cut tables; the whole downstream pipeline is a pure function of cut-conditioned factors |
| candidate systems | `2**n` subset masks | `vmap` over subset masks — shapes are n-wide by P3, so all candidates share one kernel |
| systems / states (user batch) | unbounded | user-side `jax.vmap`; leading batch axes on `System` leaves and `state` |
| relations (4.0) | up to `2^{2**n − 1}` | **never enumerated**: analytic Σφ_r and count (S3 Text sort/inclusion-exclusion formulas), `O(poly(|D|))` |

The memory frontier is (mechanisms × purviews × partitions × Q). The schedule is fixed:
`vmap` the widest cheap axes, `scan` the deep axis (partitions) with a running-min
carry, chunk the pair axis when `n` demands it. Chunk sizes are internal defaults tuned
by benchmarks, overridable via one documented argument.

### 6.3 IIT 4.0 pipeline (`iitx.measures.iit4`)

```
condition background → 𝒯_e, backward 𝒯_c                    [tensor ops]
system ii over all states → argmax → specified states s'      [dense + P4]
vmap over Θ(S): partitioned factors → φ_c, φ_e → φ_s(θ)/norm  [vmap + P4]
→ φ_s, MIP                                                    (SystemPhi)
mechanisms × purviews: π products → ii(m, z) → z'             [vmap]
  scan over Θ(M, Z): φ(m, Z) → max over Z → z*, φ_d           [scan + P4]
congruence filter (z* ⊆ s') → Distinctions
relations: analytic Σφ_r, count                               [sort/scan]
Φ = Σφ_d + Σφ_r                                               (PhiStructure)
complex search: vmap φ_s over subset masks; recursive condensation driver in Python
```

Everything except the Python condensation driver is one jit boundary per stage.
φ_s can be negative under GID (oracle golden: −0.38199…); negative φ_s means reducible,
as in PyPhi.

### 6.4 IIT 3.0 pipeline (`iitx.measures.iit3`)

```
freeze background → conditioned factored TPM
mechanisms × purviews × BI partitions → repertoires → EMD → φ  [vmap/scan + callback EMD]
max over purviews (tie keys) → MICE → Concepts (φ > 0)
expand repertoires to full space (probability embedding)
vmap over 2**n − 2 cuts: recompute the full CES under the cut
XEMD(CES, CES_cut) per cut → min → Φ, MIP cut                  (SystemPhi + CES)
complexes: vmap over candidate subset masks; condensation driver in Python
```

Two oracle-parity notes carried as open verification items into implementation:
(i) PyPhi restricts per-cut CES recomputation to prior concepts ∪ cut-split mechanisms
and treats this as exact — iitx recomputes all mechanisms (unconditionally exact) and
verifies equality on fixtures; a discrepancy would be a finding for `docs/notes/`.
(ii) PyPhi's XEMD has a "simple" path (one CES ⊆ the other: Σ φ·d(c, null)) and a
general unbalanced-transport path; both are implemented, matching its case split.

## 7. Determinism and ties

Selections happen at five points (mechanism MIP, specified state, purview/MICE, system
MIP, complex). At each: quantize values to the measure's precision, build a composite
key (value, then the oracle's documented secondary keys — e.g. MICE: `(φ, |mechanism|,
|purview|)` maximized with first-occurrence over powerset order; 4.0 MIP:
`(normalized φ, −φ)` minimized), take first-occurrence argmin/argmax over the canonical
table order. The full policy per selection point is documented in the API reference and
locked by certificate tests. Known oracle divergence, documented rather than replicated:
PyPhi 1.2.1's tied 3.0 system cuts are nondeterministic under its parallel evaluation;
iitx always returns the first tied cut in canonical order (Φ agrees regardless).
Gradients at exact ties follow `jnp.min`'s even-split subgradient (a valid element of
the subdifferential); this is documented where users see `grad`.

## 8. Differentiation

- **IIT 4.0**: differentiable almost everywhere by construction — every quantity is
  tensor algebra, `log` ratios (guarded), `|·|₊`, and min/max reductions with exact
  subgradients (Danskin: the gradient of a min over a finite enumerated set is the
  gradient of the active branch). `jax.grad(measure.phi)(system, state)` works, flowing
  into `system.tpm`. Nondifferentiable points: exact ties and `q → 0` boundaries;
  behaviour documented (even-split; guarded zeros).
- **IIT 3.0**: *not* differentiable in 0.1 — exact EMD is a host LP behind
  `jax.pure_callback`, which JAX correctly refuses to differentiate. This is explicit in
  the docs and the error the user sees. Principled relaxations (Sinkhorn via `ott-jax`,
  softmin temperatures) are a later release, shipped under names that say what they are
  (§13).
- Correctness of gradients is tested against finite differences on small systems, away
  from ties.

## 9. EMD strategy (IIT 3.0)

- **Ground metric**: Hamming distance between states (per-unit discrete metric,
  summed). For non-binary units — where the 3.0 paper defers to "an intrinsic property
  of the mechanisms" and no oracle exists — iitx declares the generalized Hamming metric
  (`d(a, b) = Σ_i 1[a_i ≠ b_i]`) as its documented convention.
- **Cause-side φ and all system-repertoire distances**: exact EMD via one batched
  `jax.pure_callback` (`vmap_method="sequential"` semantics, batched at the host
  boundary — one callback per enumeration stage, not per pair) to **POT**'s exact
  network-simplex solver (`ot.emd`), the same backend PyPhi `main` adopted. CPU;
  float64; not differentiable.
- **Effect-side φ**: analytic per-unit marginal form (valid because effect repertoires
  are products) — in-graph, cheap, and required for oracle parity (§4.7).
- **Constellation XEMD**: the unbalanced concept-transport problem (supplies/demands =
  φ values, null concept as sink, pairwise concept distances as costs) solved by the
  same host callback; concept-distance matrices are computed in-graph.
- POT is a runtime dependency: exact IIT 3.0 is core functionality, not an extra.

## 10. Macro transforms (`iitx.transforms`)

Coarse-graining and black-boxing are pure functions on state-by-state TPMs — no special
"macro" code path; their output is an ordinary `System` (this is what forced P1):

- `coarse_grain(system, partition, grouping, steps=1) → System` — the fiber-averaging
  transform `T_M = D⁻¹ Gᵀ Tᵗ G`, with per-macro-unit state groupings (identity-irrelevant
  for exchangeable micro-units); macro alphabets are generically non-binary; closed
  under composition.
- `black_box(system, partition, outputs, steps=1) → System` — hidden units noised at
  the initial step and across boxes during the τ-step window; macro state = projection
  onto output units at the window end; PyPhi-compatible semantics for the intermediate
  steps.
- Conditional independence of a transformed TPM is *checked* when a measure needs the
  factored view; a CI-violating macro TPM is a hard error naming the offending mapping,
  as in PyPhi.

Both transforms are jit/vmap-compatible (candidate mappings of equal shape batch) and
differentiable, which is what makes Φ-landscape and soft-grouping work possible later.
The *search* over mappings (`emergence`-style drivers) is out of scope (§13).

## 11. Oracle-first testing

The regression harness is built before the computation (process step 3):

1. **Fixture generation, offline.** `tests/oracle/generate/` contains scripts plus
   pinned uv environments (committed lockfiles) that run PyPhi and emit JSON fixtures:
   primary oracle **PyPhi `main` @ `ce2b2832`** with the `iit3` and `iit4_2023` presets
   (one modern environment, multi-valued units included); the published golden constants
   of 1.2.1 and `feature/iit-4.0` (already tabulated in `docs/notes/pyphi.md` §5) are
   committed directly as an independent second layer, and those two pinned refs remain
   available as diagnostic environments. Generated fixtures record the oracle ref,
   preset, and version. PyPhi is **never** a dependency of iitx's own environment.
2. **Cross-validation of the oracle.** Before `main`-generated fixtures are trusted, the
   harness asserts `main`'s presets reproduce the published constants of both pinned
   lineages. A mismatch is investigated from the mathematics and recorded in
   `docs/notes/` — it is a result, not an obstacle.
3. **Regression tests** compare iitx values (x64, CPU) against fixtures at the oracle's
   precision, and compare *certificates* (MIP identity, purviews, specified states)
   exactly under the documented tie policy.
4. **Property tests**: repertoires normalized and non-negative; φ ≥ 0 (3.0) /
   φ_s sign semantics (4.0); Φ = 0 for reducible systems; cause/effect symmetry on
   time-reversible fixtures; invariance of results under unit relabelling (up to
   relabelled certificates); transforms compose; CI checker accepts/rejects known cases.
5. **Transformation tests**: eager ≡ jit ≡ vmap (chex variants); one-trace-per-signature
   (`assert_max_traces`); gradient vs finite differences (§8); float32 vs float64
   tolerance envelope documented by test.
6. **Tiers**: fast (n ≤ 4, seconds, every CI run), `slow` (n = 5–6 goldens: `big`,
   `rule152`, fig16), `veryslow` (macro 9–12-unit examples) — opt-in markers, PyPhi
   style.

## 12. Engineering standard

Matches cax (see `docs/notes/cax-engineering.md` checklist) with the deliberate
deviations recorded there; per project decision, maximally modern:

- **Python ≥ 3.14 only** (`requires-python = ">=3.14"`, ruff/ty target py314, CI on
  3.14; PyPhi never shares this environment, so its constraints do not apply).
- uv-native; hatchling; `src/iitx/` layout; static hand-bumped version; PEP 639 license
  fields; MIT; committed `uv.lock`.
- Runtime dependencies (unversioned; `uv.lock` pins exactly): `jax`, `numpy`, `pot`. **No flax**:
  the core is pure functions + registered dataclasses (P5/P6); NNX buys nothing here.
- ruff only (line length 100, tabs, cax's rule set incl. `D`, Google docstrings
  everywhere including tests); `ty` with `all = "error"` — and a `ty` CI job.
- pytest with `filterwarnings = error`; chex available as a dev dependency; coverage
  uploaded; tests mirror `src/` one-to-one. Shape and dtype contracts live in
  docstrings (no jaxtyping — the repository stays lean, per project decision).
- Docs: MkDocs Material + mkdocstrings(google) + mkdocs-jupyter, `strict: true`,
  symlinked `docs/index.md → README.md`, two-line API stubs, numbered Colab notebooks
  (`00_getting_started`, `01_iit4_basics`, `02_reproducing_pyphi`,
  `03_batching_and_gradients`, …) reproducing literature results.
- README in the cax shape; CITATION.cff; CONTRIBUTING with an architecture guide;
  `CHANGELOG.md` (deliberate improvement over cax); imperative-mood commits explaining
  why; tag-triggered trusted-publisher PyPI release workflow.

Module layout:

```
src/iitx/
	__init__.py          # docstring only
	system.py            # System, node_tpms, check_independence, validation
	states.py            # ravel/unravel, state tables
	enumeration.py       # subset/partition/cut tables + registries (NumPy, build time)
	repertoires.py       # mask-parameterized repertoire algebra (§6.1)
	distances.py         # ID/GID, EMD (POT callback), analytic effect EMD, registry
	transforms.py        # coarse_grain, black_box
	measures/
		__init__.py      # Measure protocol
		iit3.py          # IIT3: φ, MICE, concepts, CES, Φ, complexes
		iit4.py          # IIT4: ii, φ_s, distinctions, relations, Φ-structure, complexes
```

## 13. Out of scope for 0.1

Each of these is deferred to a later release with its own design document — the
architecture reserves its seam, and nothing ships half-done:

1. Approximations of any kind: cut-one, Sinkhorn EMD (`ott-jax`), softmin-relaxed φ.
   When they come, they are named (`*_sinkhorn`, `temperature=`) and typed as
   approximations.
2. Alternative measures: Φ\*, Φ_G, Φ_MI/SI/WMS, Tegmark variants, ΦID/Φ_R — plus the
   stationary-distribution and time-series adapter front-ends they need.
3. The IIT 4.0 "2026" system measure (intrinsic-information cap) and tracking of PyPhi
   2.0 post-`ce2b2832` changes.
4. Emergence/grain search drivers (exhaustive or greedy) over macro mappings, and IIT
   4.0's "maximally irreducible within" unit-grain machinery.
5. Actual causation (Transitions, α).
6. Non-canonical configuration combinations of the shipped measures.
7. Relation *enumeration* (faces, individual φ_r lists) beyond the analytic sums and
   counts; concrete relation sets come with a dedicated design (they are the one
   doubly-exponential object).
8. Adapters from CAs/circuits/networks to `System` (edge packages, never core).

## 14. Open questions for review

1. **Scope call**: macro transforms (§10) are included in 0.1 on the argument that they
   force the non-binary core to be real and are cheap pure functions; the emergence
   search is excluded. Agreed?
2. **Naming**: `iitx.measures.iit3 / iit4` with `Measure.phi`/`Measure.analyze`, and no
   bare `phi` at package top level (φ_s vs Φ ambiguity). Agreed, or prefer top-level
   convenience functions?
3. **POT as a runtime dependency** (exact 3.0 EMD in core) vs an `iitx[emd]` extra with
   a lazy error. Design chooses runtime dependency for a working core out of the box.
4. **Oracle promotion**: PyPhi `main` @ `ce2b2832` as primary oracle (per project
   decision), cross-validated against the published constants of both pinned lineages
   before trust. Confirm.
