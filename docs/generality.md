# Generality: what PyPhi accepts, what it does not, and what `iitx` will accept

PyPhi's strength is that it knows nothing about where a system came from. This document
makes that boundary explicit — as a checklist, not an impression — so that "at least as
general as PyPhi" is verifiable, and so that every place `iitx` widens the boundary is a
deliberate, documented decision.

Three PyPhi lineages matter (see `docs/notes/pyphi.md` §0):

- **PyPhi `main` @ `ce2b2832`** (unreleased 2.0.0): a serious IIT 4.0-native rework with
  explicit `iit3` / `iit4_2023` presets, multi-valued units, and a POT-based EMD backend
  replacing the brittle pyemd. **Primary oracle for both IIT 3.0 and IIT 4.0**, pinned at
  that commit — and the only PyPhi that can check non-binary results.
- **PyPhi 1.2.1** (git tag `1.2.1`, commit `852b006a` = PyPI 1.2.0 + subprocess-config
  fix): the released IIT 3.0 implementation whose published golden numbers (Φ = 2.3125,
  …) are the independent ground truth for 3.0. Fallback/diagnostic oracle.
- **PyPhi `feature/iit-4.0`** (commit `b78d0e34`): the branch that produced the numbers
  in Albantakis et al. 2023. Its golden fixtures are the independent ground truth for
  4.0. Fallback/diagnostic oracle.

The published golden constants of the two pinned lineages (already extracted into
`docs/notes/pyphi.md` §5) are committed as fixtures in their own right; the harness
verifies that `main`'s presets reproduce them before `main`-generated fixtures are
trusted. If they ever disagree, the disagreement is diagnosed from the mathematics and
recorded in `docs/notes/` — never silently papered over.

## The table

"—" means not applicable; ✅/❌ mean supported/unsupported. `iitx 0.1` is the first
release; `iitx (later)` marks widening that is designed for but deliberately deferred to
a release with its own design document.

| Capability | PyPhi 1.2.1 | PyPhi `feature/iit-4.0` | iitx 0.1 | iitx (later) |
|---|---|---|---|---|
| **System model** | | | | |
| Any finite discrete dynamical system as a TPM | ✅ | ✅ | ✅ | |
| TPM input forms | state-by-state, state-by-node (2-D and multidimensional) | same | state-by-state (canonical) + node TPMs + binary state-by-node convenience | |
| State indexing convention | little-endian, binary | little-endian, binary | little-endian, **mixed-radix** | |
| Binary units | ✅ | ✅ | ✅ | |
| Non-binary units | ❌ (`[2]*n` hard-coded) | ❌ | ✅ **uniform and non-uniform per-unit alphabets** `(q_1, …, q_n)` | |
| Stochastic TPMs | ✅ | ✅ | ✅ | |
| Deterministic TPMs | ✅ (special case) | ✅ | ✅ (special case) | |
| Conditional independence | required; validated; silently destroyed by sbs→sbn conversion | required | required for mechanism-level analysis; **checked property** of state-by-state TPMs, never silently destroyed | |
| Connectivity matrix (optional, prunes/validates) | ✅ (`cm[i][j]` = i→j) | ✅ | ✅ (same convention) | |
| Self-loops | ✅ | ✅ | ✅ | |
| Candidate systems with frozen background | ✅ | ✅ (effect side) | ✅ | |
| Causally marginalized background (IIT 4.0 Eq. 4) | ❌ | ✅ (cause side, unconditional) | ✅ | |
| **Theory versions and measures** | | | | |
| IIT 3.0 (full: repertoires, φ, concepts, CES, Φ, complexes) | ✅ | ✅ (legacy path) | ✅ (canonical config) | |
| IIT 4.0 (ii, φ_s, distinctions, relations, Φ-structure, complexes) | ❌ | ✅ | ✅ (canonical `iit4_2023` config) | |
| IIT 4.0 "2026" variant (capped system measure, Eq. 23) | ❌ | ❌ | ❌ | own design doc |
| Measure as replaceable object | partial (config registry: distances, partition types) | partial | ✅ **first-class**: measures are objects; comparator, partition scheme, tie policy, precision are explicit components | |
| Alternative measures (Φ\*, Φ_G, Tegmark variants, ΦID) | ❌ | ❌ | ❌ (interface designed for them; see `docs/design.md` §5) | own design docs |
| Non-canonical config combinations (KLD/L1 distances, TRI/ALL for 3.0, CONCEPT_STYLE cuts, …) | ✅ (untested combinations) | ✅ | ❌ architecture admits them; only canonical configs are implemented and oracle-tested | as needed |
| Actual causation (Transitions, α) | ✅ | ✅ | ❌ | own design doc |
| **Macro-level analysis** | | | | |
| Coarse-graining (unit partition + state grouping) as a TPM transform | ✅ | ✅ | ✅ pure function, closed under composition, non-binary in and out | |
| Black-boxing (hidden units, outputs, temporal grain τ) as a TPM transform | ✅ | ✅ | ✅ pure function | |
| Emergence search over mappings/grains | ✅ (`emergence()`) | ✅ | ❌ (transforms are exposed; the search driver is deferred) | own design doc |
| Macro units feed back into the same analysis pipeline | ✅ (`MacroSubsystem` special path) | ✅ | ✅ **no special path** — a transformed TPM is an ordinary system | |
| **Computation** | | | | |
| Parallelism within one system | Python `multiprocessing` over cuts/concepts | Ray | ✅ accelerator-parallel: dense state dimensions + `vmap`/`scan` over enumeration tables | |
| Batching across systems/states in one call | ❌ | ❌ | ✅ every public function is `jax.vmap`-compatible over `(system, state)` | |
| CPU / GPU / TPU | CPU only | CPU (+Ray cluster) | ✅ (float64 oracle parity on CPU; float32 on accelerators) | |
| Gradients w.r.t. TPM entries | ❌ | ❌ | ✅ IIT 4.0 path (a.e., exact subgradients); ❌ IIT 3.0 path (exact EMD is a host LP — explicit, by design) | Sinkhorn/softmin relaxations, explicitly named |
| Exact computation | ✅ | ✅ | ✅ (exact before any approximation) | |
| Approximations (cut-one, …) | ✅ (off by default) | ✅ | ❌ | own design doc, named as approximations |
| Determinism of results incl. tie certificates | ❌ (parallel cut evaluation makes tied-MIP identity nondeterministic) | partial (`resolve_ties` config) | ✅ documented deterministic tie policy, canonical enumeration order | |

## Where iitx must match PyPhi exactly (oracle contract)

On every canonical example in the PyPhi documentation and test suite (fixtures listed in
`docs/notes/pyphi.md` §5), with canonical configuration:

- IIT 3.0 quantities (repertoires, φ, MICE, concept φ's, Φ, MIP cuts, complexes) match
  the canonical 3.0 configuration (`EMD`, `BI`, `3.0_STYLE`, `PRECISION=6`, frozen
  background — PyPhi `main`'s `iit3` preset ≡ 1.2.1 defaults) to its precision (1e-6).
- IIT 4.0 quantities (ii, specified states, φ_s, MIP, φ_d, Σφ_r, relation counts, Φ)
  match the canonical 4.0 configuration (GID, `ALL` mechanism partitions, `SET_UNI/BI`
  system partitions, `PRECISION=13`, backward cause TPM — PyPhi `main`'s `iit4_2023`
  preset ≡ `feature/iit-4.0` defaults) at 1e-10 or better.
- Non-binary systems match PyPhi `main` (the only oracle for them), plus hand-computed
  micro-examples.
- Tie *certificates* (which purview, which partition) match PyPhi's documented
  deterministic orderings; where PyPhi itself is nondeterministic (3.0 parallel cut
  ties), iitx defines a deterministic order and documents the divergence.

The oracle is trusted, not obeyed: a disagreement is diagnosed from the mathematics and
recorded in `docs/notes/` (see the known PyPhi quirks in `docs/notes/pyphi.md` §7).

## Where iitx exceeds PyPhi from release 0.1

1. **Non-binary, non-uniform unit alphabets** throughout the core — no `2**n` anywhere.
   The IIT 4.0 formalism is already stated over arbitrary finite alphabets; coarse-grained
   macro units *require* them. (For IIT 3.0's EMD on non-binary units the paper leaves
   the ground metric open; iitx declares the generalized Hamming metric — see
   `docs/design.md` §9 — as its convention, since no oracle exists.)
2. **Within-system parallelism on accelerators** — candidate systems, mechanisms,
   purviews, partitions, and cuts are exposed to the accelerator as batched tensor work.
3. **Batching across systems** — `vmap` over TPMs and states with one compiled kernel
   per `(n, alphabet)` signature.
4. **Differentiation** — the IIT 4.0 pipeline is differentiable almost everywhere as-is,
   with exact subgradients at min/max selections; non-differentiable paths (3.0 EMD) fail
   loudly rather than silently.
5. **Deterministic, documented tie policy** — PyPhi 1.2.1's tied system cuts are
   nondeterministic under its default parallelism; iitx results are reproducible bit-for-bit
   on a fixed backend.

## Explicitly out of scope for 0.1

Actual causation; emergence/grain search drivers; the IIT 4.0 "2026" system measure;
alternative measures (Φ\*, Φ_G, ΦID, Tegmark variants); time-series/stationary-distribution
front-ends; approximations of any kind (cut-one, Sinkhorn, softened minima); quantum
extensions. Each is a later release with its own design document; the measure interface
(`docs/design.md` §5) is shaped so they plug in without touching the core.
