# Macro-level analysis: coarse-graining, black-boxing, causal emergence

Sources:

- Marshall, Albantakis, Tononi (2018). *Black-boxing and cause-effect power.* PLoS Comput Biol 14(4): e1006114.
- Hoel, Albantakis, Tononi (2013). *Quantifying causal emergence shows that macro can beat micro.* PNAS 110(49): 19790–19795.
- Hoel, Albantakis, Marshall, Tononi (2016). *Can the macro beat the micro? Integrated information across spatiotemporal scales.* Neurosci. Consciousness 2016(1): niw012.
- PyPhi `pyphi.macro` (reference implementation; pins down several conventions the papers leave open).

Notation. Micro system `S_m` with units `U = {1..n}`, unit `i` has alphabet `Ω_i` (papers: binary). Micro state space `Ω = Π_i Ω_i`. Micro TPM `T` gives `p(s' | do(s))` for `s, s' ∈ Ω` — i.e. the transition distribution under perturbation (intervention), state-by-state form, shape `|Ω| × |Ω|`. Macro system `S_M` with units `{1..k}`, alphabets `Ω^M_j`.

---

## 1. Coarse-graining (Hoel 2013; Hoel 2016)

A (spatial) coarse-graining is a mapping `M : S_m → S_M` with two components:

1. **Unit partition.** A partition `P = {Z_1, …, Z_k}` of the micro units into disjoint, exhaustive, non-overlapping groups; group `Z_j` becomes macro unit `j`. Micro units left out of the analyzed system are *background conditions* (frozen at their current state, not marginalized).

2. **State grouping.** For each macro unit `j`, a surjective map `g_j : Ω_{Z_j} → Ω^M_j` from joint states of the micro units in `Z_j` onto macro states. The full state map is `G = (g_1, …, g_k) : Ω → Ω^M`, applied factor-wise. Every micro state maps to exactly one macro state (exhaustive + disjunctive), which is what makes the macro level **supervene** on the micro level: "given the micro elements of `S_m` and the causal relationships between them, all other members of {S} are fixed as well."

   *Identity-irrelevance constraint* (made explicit in Hoel 2016): admissible `g_j` may not distinguish micro states that differ only by which micro unit carries which value — "mappings of micro states into macro states are limited to those in which the identity of the individual micro elements within a macro element is irrelevant." For binary micro units this means `g_j` must factor through the count of ON units: `g_j(s_{Z_j}) = γ_j(Σ_{i∈Z_j} s_i)` for some map `γ_j : {0..|Z_j|} → Ω^M_j`. E.g. `{01, 10}` must map to the same macro state; distinguishing them would leak micro information into the macro level. PyPhi's `grouping` encodes exactly this: for each macro unit, a tuple of sets of ON-counts, one set per macro state.

3. **Temporal grain** (optional). A macro time step aggregates `τ` micro steps. For memoryless micro dynamics this is TPM composition: use `T^τ` (matrix power of the state-by-state TPM) before spatial grouping. Hoel 2013's temporal examples also cover second-order-Markov micro dynamics, where consecutive micro steps are grouped into a macro step so that the *macro* chain becomes first-order.

### Macro TPM derivation

Macro-level EI/Φ requires perturbing the *macro* system into all its states with equal probability. A macro perturbation `do(m)` is implemented as the uniform mixture over the fiber `G⁻¹(m)` — "one must set S into all micro states that are grouped into the corresponding macro state `s_M`, and average over the effects." Hence:

```
T_M(m → m') = (1 / |G⁻¹(m)|) · Σ_{s ∈ G⁻¹(m)} Σ_{s' ∈ G⁻¹(m')} T^τ(s → s')
```

In matrix form, with `G ∈ {0,1}^{|Ω| × |Ω^M|}` the fiber-membership (one-hot) matrix and `D = diag(Gᵀ1)` the fiber sizes:

```
T_M = D⁻¹ · Gᵀ · T^τ · G
```

This is a pure linear-algebra transform of the micro TPM. Note the **uniform weighting inside the fiber is a modeling choice** (the maxent-perturbation convention), not a mathematical necessity; the macro TPM is only defined relative to it.

### Validity conditions

- **Supervenience** is automatic (`G` is a function).
- **Exhaustive/disjoint** partition; **identity-irrelevance** of state groupings (above).
- **Conditional independence.** `T_M` as derived above is state-by-state. For it to define a causal *network* of macro units (as IIT's mechanism-level analysis requires), it must factorize over macro units: `p(m' | m) = Π_j p(m'_j | m)`. This is *not* guaranteed — averaging over fibers generically induces correlations between macro units (closely related to non-lumpability of Markov chains: the aggregated chain is exactly Markov/factorizable only when rows of `T^τ` within a fiber agree after aggregation). Hoel 2013's EI does not need factorization (EI is computed on the state-by-state TPM), but IIT analysis does. PyPhi's `CoarseGrain.macro_tpm(..., check_independence=True)` raises `ConditionallyDependentError` when the check fails; i.e. **PyPhi's policy is to reject such mappings**, not to project onto the nearest factorized TPM. The papers themselves do not state this condition; it is pinned down only by the implementation.

---

## 2. Black-boxing (Marshall 2018)

A black-box mapping is *not* an aggregation but a **projection plus temporal evolution**. A "black-box element is a physical macro element … constituted of several micro elements (spatial), operating over several micro time steps (temporal)."

Specification:

1. **Box partition.** Disjoint, non-overlapping boxes `B_1, …, B_k` covering the analyzed micro units (exclusion: "there cannot be any overlap between the micro elements of multiple black boxes"; leftovers are background conditions).
2. **Output designation.** Each box `B_j` has a designated output `O_j ⊂ B_j` (in the paper: a single output micro element per box; PyPhi's `Blackbox(partition, output_indices)` allows a set). `H_j = B_j \ O_j` are **hidden** units: "hidden from other black boxes … they do not directly contribute to the intrinsic cause-effect power of the system."
3. **Macro time step `τ`** with a designated *output time step*: the macro state of box `j` **is** the micro state of `O_j` read at the end of the `τ`-step window. The state mapping is the projection `π_O : Ω → Π_j Ω_{O_j}` — no averaging, no symmetry requirement, and the macro alphabet is inherited from the output units (binary outputs ⇒ binary macro units).

### Black-box TPM computation

To compute `p(o' | do(o))` for output states `o, o'`:

1. Set the outputs (and system inputs) into all possible states under maximum entropy — perturb `do(o)`.
2. **Noise the hidden initial states**: "the initial states of micro elements … other than its designated output element at the designated output time step, are noised during the perturbation analysis" — i.e. hidden units at `t = 0` are marginalized under the uniform distribution (their past cause-effect power is discounted).
3. **Evolve the micro dynamics unperturbed for `τ` micro steps**, with one structural restriction enforced throughout: any micro connection *leaving a box* other than through its designated output (at the designated output step) is noised/cut — other boxes may not read a box's hidden units.
4. Read the output units at step `τ`:

```
T_bb(o → o') = Σ_{h ∈ Ω_H} (1/|Ω_H|) · [ Π_over_τ_steps of the connection-masked micro TPM ] (s=(o,h) → states with π_O(s') = o')
```

PyPhi implements the iteration as `run_tpm(system, steps, blackbox)`, composing the full TPM once with `τ−1` applications of a "noised" TPM (`tpm · noise_tpm^(t−1)`) in which cross-box hidden connections are replaced by maxent inputs.

### Validity conditions (Marshall 2018's four requirements)

1. **Functionality** — each box "must have at least one input, one output, and two or more (macro) states that can be read from its output."
2. **Black-box condition** — internal micro interactions are hidden; they contribute only via the output.
3. **Integration** — every constituent micro element must contribute causally (irreducibly) to its box's output; a reducible micro system cannot yield an integrated macro system. This is a *state-dependent, causal* condition, checked by analysis, not by the mapping's syntax.
4. **Exclusion** — no unit belongs to two boxes.

### Coarse-graining vs black-boxing

| | Coarse-graining | Black-boxing |
|---|---|---|
| State map | Symmetric aggregation `G` (identity-irrelevant, e.g. counts) | Projection `π_O` onto output units at output time |
| Uses all micro units' states | Yes (all enter the macro state) | No (hidden states are discarded/noised) |
| Macro alphabet | Up to `|Z_j|+1` states per unit (can grow) | Inherited from outputs |
| Marginalization | Uniform over fibers of `G` | Uniform over hidden initial states + cut hidden cross-box links |
| Typical benefit | Reduces indeterminism/degeneracy (averaging) | Reveals higher-order mechanisms among specialized units; increases integration |
| Supervenience | Macro state and macro dynamics fixed by micro | Macro *state* is fixed (projection), but macro *dynamics* depend on marginalized hidden states — Markovianity of `T_bb` is an additional assumption/approximation |

Neither is a special case of the other; PyPhi composes them (black-box first, then optionally coarse-grain the outputs).

Key results (Marshall 2018): 3 XOR gates + COPY delays — micro Φ = 0.25 (first-order mechanisms only); black-boxed at τ = 2, Φ = 1.875 with genuine second-order mechanisms. 55-unit NOR network: micro Φ = 0.453; MAJORITY-gate black-boxing at τ = 4, Φ = 2.333; only ~0.22% of the 124,176 candidate macro systems had Φ > 0 (search is expensive, hits are sparse).

---

## 3. Effective information and causal emergence (Hoel 2013)

- **Effective information.** Perturb `S` into all `n = |Ω|` states with equal probability `1/n` (unconstrained repertoire `U_C`, maximum entropy `H_max`); let `U_E = (1/n) Σ_s p(·|do(s))` be the resulting effect distribution. Then

  ```
  EI(S) = MI(U_C ; U_E) = (1/n) Σ_s D_KL( p(·|do(s)) ‖ U_E )
  ```

  EI is the average, over maxent interventions, of the effect information of each state; equivalently the mutual information between all possible causes and their effects.

- **Effectiveness and decomposition.** `eff(S) = EI(S) / log₂ n ∈ [0,1]`, and

  ```
  eff = determinism_coef − degeneracy_coef
  ```

  determinism ↓ with noisy transitions (high `H(p(·|do(s)))`), degeneracy ↑ when many states transition to the same effects. `eff = 1` iff the TPM is a permutation matrix.

- **Causal emergence.** For a supervening macro `S_M` (obtained by the coarse-graining of §1, with EI computed from `T_M`):

  ```
  CE = EI(S_M) − EI(S_m)
  ```

  CE > 0 ⇔ the macro beats the micro. Decomposition: `CE = ΔI_eff + ΔI_size`, where `ΔI_size ≤ 0` (state space shrinks, capacity `log₂ n` drops) and `ΔI_eff ≥ 0` when averaging cancels noise (indeterminism) and/or collapses degenerate transitions. Examples: 4 noisy-AND micro units → 2 macro units: EI 1.15 → 1.55 bits (CE = 0.40); temporal grouping of a second-order-Markov system: 9-unit spatiotemporal example EI 0.59 → 3.51 bits (CE = 2.92).

- **Relation to IIT quantities.** EI is the *state-averaged*, *whole-system*, *partition-free* ancestor of IIT's cause/effect information (which is per-state `D_KL` against unconstrained repertoires). Hoel 2016 replaces EI with Φ: state-dependent, compositional (mechanisms over all subsets), with integration (minimum-information partition) and exclusion (Φ-maximal complex boundaries). EI cannot identify causal borders; Φ can — in Hoel 2016 Fig. 6, EI ranks micro > macro while Φ correctly identifies the macro complex. Hoel 2016 also decomposes φ into repertoire-size vs irreducible-selectivity terms: macro wins when selectivity gains (determinism/degeneracy reduction) outweigh the size loss. Empirically (their Fig. 7), Φ along the lattice of groupings is roughly unimodal — greedy/gradient ascent over groupings tends to find the global maximum.

---

## 4. Design implications for iitx

**The micro→macro mapping should be a first-class, declarative object**, separate from any network/subsystem class — a pure specification that transforms TPMs:

```python
@dataclass(frozen=True)  # pytree with static fields
class Macro:
	partition: tuple[tuple[int, ...], ...]  # micro units per macro unit / box
	grouping: tuple[...] | None  # per-unit state map g_j (coarse-grain)
	outputs: tuple[tuple[int, ...], ...] | None  # per-box output units (black-box)
	steps: int = 1  # temporal grain τ
```

with the two transforms as **pure functions on TPMs**:

- `coarse_grain(tpm_micro, mapping) -> (tpm_macro, macro_shape)` — implements `T_M = D⁻¹ Gᵀ T^τ G`. Inputs: state-by-state micro TPM (or state-by-node + a lift), the mapping. Outputs: state-by-state macro TPM plus the macro **state-space spec** `(k, (q_1..q_k))`, and the fiber map `G` (needed to translate the current micro state to the macro state and to interpret repertoires).
- `black_box(tpm_micro, mapping) -> (tpm_macro, macro_shape)` — builds the connection-masked TPM, composes `τ` steps, injects maxent over hidden initial states, projects onto outputs.
- `check_conditional_independence(tpm) -> residual` — CI must be a *checked property*, not an assumption. The state-by-state form is primary; state-by-node is a derived, sometimes-unavailable view.
- `macro_state(mapping, micro_state) -> macro_state` — deterministic in both transforms (aggregation resp. projection).

**"Macro units as first-class citizens"** means: after the transform, the result is an ordinary iitx system — same Φ pipeline, no special-cased "macro" code paths. That forces two ground-level commitments:

1. **Everything downstream must accept heterogeneous, non-binary alphabets** (see §5) — TPMs indexed by mixed radices `(q_1, …, q_k)`, repertoires with per-unit sizes, maxent = uniform over `Π q_j`, partitions/purviews over non-binary units.
2. **State-by-state TPMs are the common currency** of transforms; conversion to node form happens once, after a CI check, at the boundary to mechanism-level analysis.

**JAX specifics.** Both transforms are compositions of `matmul`, one-hot aggregation, masking, and normalization — jit-compatible, differentiable (e.g. for optimizing soft groupings), and `vmap`-able over candidate mappings *of the same shape*. The combinatorial search over mappings (`all_partitions × all_groupings × τ`, cf. PyPhi's `emergence()`) has data-dependent shapes and belongs *outside* jit: enumerate specs in Python, batch same-shaped candidates, `vmap` the transform + Φ. Since valid macro systems are sparse (0.22% in Marshall's NOR example) and the Φ landscape over groupings looks near-unimodal (Hoel 2016), provide both exhaustive and greedy search drivers.

---

## 5. Non-binary macro units

Coarse-graining **generically produces multi-valued macro units from binary micro units**: grouping `|Z_j|` binary units under identity-irrelevance yields up to `|Z_j| + 1` macro states (one per ON-count); `γ_j` may merge counts but need not merge down to 2. Restricting macro units to binary (as Hoel 2016 did "for tractability") silently discards most of the mapping space.

Implications for iitx:

- Unit alphabets `(q_1, …, q_k)` are part of the system spec everywhere: TPM shapes, repertoire shapes, maxent distributions, state enumeration (mixed-radix), and any `log₂`-based normalization (EI's `log₂ n` is `Σ_j log₂ q_j`).
- No `2**n` anywhere in the codebase; use `prod(shape)` and mixed-radix ravel/unravel utilities.
- Iterated coarse-graining (macro of macro, cf. Hoel 2016's grouping lattice) requires the transforms themselves to accept non-binary *inputs*, not just produce non-binary outputs — closure under composition.
- Black-boxing, by contrast, preserves output alphabets; a design that supports both must not conflate "macro" with "multi-valued".

---

## 6. Ambiguities / underspecified points

1. **Fiber weighting.** The uniform (maxent) distribution over `G⁻¹(m)` in the macro TPM is a convention from the perturbational EI framework; the papers never discuss alternatives (e.g. stationary-distribution weighting). `T_M` — and hence CE — depends on this choice.
2. **Conditional independence failure.** No paper states what to do when `T_M` does not factorize over macro units. Only PyPhi pins this down: `check_independence=True` (default) raises `ConditionallyDependentError`, i.e. the mapping is rejected. Alternatives (keep state-by-state and compute EI only; project to nearest product TPM) are unaddressed.
3. **Scope of admissible groupings.** The identity-irrelevance restriction is explicit only in Hoel 2016; Hoel 2013's examples obey it, but the general definition there is looser. PyPhi's `all_groupings` enumerates count-based groupings — and, per its comments, does not even enumerate *all* of those. What the "official" space of groupings is remains implementation-defined.
4. **Black-box temporal semantics.** Exactly which connections are noised at *intermediate* micro steps (hidden→other-box only? outputs readable mid-window?), whether a box's output may feed back into its own hidden units, and the composition order in `run_tpm` (`tpm · noise_tpm^(t−1)`) are fixed by PyPhi, not by the paper's prose. Marshall 2018 uses one output element per box read at one designated output step; PyPhi generalizes to output sets, without a published justification for the generalized semantics.
5. **Hidden-state marginalization vs conditioning.** Noising hidden initial states makes `T_bb` Markovian by construction, but the *true* output process is generally non-Markovian; the papers do not quantify or bound this approximation, nor say whether the black-box "integration" requirement is meant to guarantee it.
6. **Background conditions.** Whether units outside the candidate macro system are frozen at their observed state (IIT convention) or maxent-noised during macro TPM construction is inherited from the surrounding IIT analysis rather than stated in the macro formalism itself.
7. **State-dependence of the search.** Black-box validity (integration, functionality) and Φ-maximality are evaluated at a particular micro state; a mapping valid in one state may be invalid in another. Whether a "macro level" is a property of the system or of (system, state) is left implicit — PyPhi's `emergence(network, state, ...)` takes a state.
8. **Combining transforms.** Only PyPhi defines an order for combining black-boxing with coarse-graining (black-box, then coarse-grain the outputs) and with `time_scales`; the papers treat them separately.
