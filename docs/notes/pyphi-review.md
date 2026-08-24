# PyPhi code review — bugs, paper divergences, and iitx opportunities

Rigorous review of the PyPhi source as an oracle for `iitx`, 2026-08-24. Complements
(and in two places corrects) `oracle-findings.md`; the six findings there are **not**
restated. Every claim below cites file:line in the studied worktrees:

| tree | ref | role |
|---|---|---|
| `pyphi-1.2.1` | tag `1.2.1` @ `852b006a` | IIT 3.0 oracle |
| `pyphi-iit4` | `feature/iit-4.0` @ `b78d0e34` | IIT 4.0 oracle (2023 paper code) |
| `pyphi` (main) | `ce2b2832` | unreleased 2.0, used for cross-checks |

Paths below are relative to the given worktree. Numeric experiments live in
`scratchpad/cax/exp1_xemd.py` … `exp9_pyphifloat.py` (run with `scratchpad/jaxenv`;
pyemd 1.0.0 was installed there for solver-behavior tests — pyphi itself was never
installed). Classification: **BUG** (code wrong vs. its own spec), **PAPER-DIVERGENCE**
(code ≠ paper), **DESIGN-SMELL** (fragile/arbitrary/unproven), **VERIFIED-SOUND**
(suspected problem investigated and cleared — recorded so it is not re-litigated),
**IITX-OPPORTUNITY**.

---

## 1. IIT 3.0 (pyphi-1.2.1)

### R1. Analytic effect EMD is provably exact — for products only, and only products are reachable. [VERIFIED-SOUND]

`distance.py:147-163` (`effect_emd`) computes Σᵢ |P₁(nodeᵢ=OFF) − P₂(nodeᵢ=OFF)| in
place of the transport LP, for *any* purview size. This is exactly correct whenever
both arguments are product distributions, for any number of nodes:

- **Lower bound**: each coordinate projection is 1-Lipschitz w.r.t. Hamming cost, so
  W₁(P,Q) ≥ Σᵢ W₁(Pᵢ,Qᵢ); for a binary marginal W₁ = |p(0) − q(0)| (TV).
- **Upper bound**: the product of per-coordinate optimal couplings is a coupling of the
  two products with cost Σᵢ W₁(Pᵢ,Qᵢ) (cost is separable/additive over coordinates).

Verified numerically to 9e-16 over 300 random 2–4-node product pairs, and shown to be
*wrong* (0.0 vs. true 1.0) on a correlated pair (`exp8_effect_emd.py`). The formula's
precondition holds for every input PyPhi ever passes it: unpartitioned effect
repertoires (`subsystem.py:348-380`, product over purview nodes), partitioned ones
(products of parts, each itself a product), unconstrained and expanded ones, and — via
the conditional-independence projection of macro TPMs (see R8) — even
`MacroSubsystem` repertoires. `hamming_emd`'s Hamming matrix is endianness-invariant
(H[a,b] = popcount(a⊕b) under any consistent bit-order), so the order='F' flatten at
`distribution.py:130-151` is consistent by symmetry, not by coordination.
**iitx**: safe to use the analytic form on the effect side and full transport on the
cause side; assert product-ness in debug builds.

### R2. Extended EMD, part 1: the `max(distances)+1` blocking constant is inert under the actual solver — and insufficient under a transhipment solver. [DESIGN-SMELL]

`compute/distance.py:95-107` blocks the intra-CES diagonal blocks of the transport
matrix with `np.max(distances) + 1`, where `distances` are only the *pairwise*
C1×C2 concept distances — concept-to-null distances (up to 2n) can exceed this
constant. Does that corrupt Φ? Three experiments (`exp1_xemd.py`, `exp7_detour.py`):

- In the bipartite transportation reading, intra-CES entries pair a positive supply
  with a zero demand and are never used; pyemd's result equals the exact bipartite LP
  on the same matrix to 3e-15 across 700 random instances, including non-metric,
  nonzero-diagonal matrices.
- pyemd (Pele–Werman `emd_hat`) does **not** tranship: on an adversarial instance
  where a detour through a blocked edge plus a cheap null edge would cost 0.115, both
  pyemd and the bipartite LP return 3.005 (`exp7_detour.py`).
- However, the 1.2.1 CHANGELOG (lines 437-442) records that the blocking constant was
  added precisely because an earlier solver *did* take intra-constellation detours. If
  a transhipment-capable min-cost-flow backend were ever substituted, `max(pairwise)+1`
  is **not** a sufficient blocking value whenever some `d(c, null) > max(pairwise)+1` —
  the exact situation the constant is supposed to prevent would silently return.

**Verdict**: with pyemd, Φ is correct; the guard is dead code that documents a
solver assumption instead of enforcing it. **iitx**: formulate the CES distance as an
explicitly bipartite balanced transport (no blocked entries at all), which is immune
by construction.

### R3. Extended EMD, part 2: the negative-deficit case feeds pyemd a negative "histogram" — and works only by an undocumented solver behavior. [BUG (latent)]

`compute/distance.py:111-118`: `d2[-1] = sum(d1) - sum(d2)`. When the partitioned CES
carries more total φ among changed concepts than the unpartitioned one (possible —
cuts can create concepts and raise φ; the code comment at lines 64-68 acknowledges
it), the null-concept *demand* is **negative**. pyemd's documented contract requires
valid (non-negative) histograms; a negative demand has no meaning in the
transportation problem (scipy's LP declares it infeasible — `exp1_xemd.py`).

Empirically, pyemd 1.0.0 (same `emd_hat` C++ core as the 0.x releases 1.2.1 pins with
`pyemd >= 0.3.0`, unpinned) treats histograms via signed differences —
`emd(P,Q) = transport((P−Q)₊, (Q−P)₊)` (verified directly, `exp2_negdeficit.py`) — so
the negative entry acts as **null-concept supply**, which is precisely the natural
symmetric extension of Text S2's rule (ambiguity A6 in `iit-3.0.md`). Across 313
random negative-deficit instances pyemd matched the explicit null-as-supply LP to
2e-15. Confirmation that this is the *intended* semantics: main (2.0) rewrote the
construction to make it explicit and well-posed — `d1[-1] = total − sum1;
d2[-1] = total − sum2`, both non-negative (`pyphi/measures/ces.py:259-269` on main).

**Verdict**: 1.2.1's numbers are right, but by reliance on undocumented solver
behavior. **iitx**: implement the balanced construction (null gets each side's
deficit) explicitly; it reproduces 1.2.1 bit-for-bit in both regimes and is defensible.

### R4. Extended EMD, part 3: two different null concepts in one transport problem. [PAPER-DIVERGENCE]

`compute/distance.py:69-72` computes each concept's distance-to-null against **its own
subsystem's** null concept (`c.subsystem.null_concept`): uncut-system null for C1
concepts, *cut-system* null for partitioned-CES concepts (concepts get
`subsystem = cut_subsystem` in `compute/subsystem.py:54-62`). The cut changes the
unconstrained effect repertoire, so these are two different points in concept space
being treated as one transport node ("the" null). Same in `_ces_distance_simple`
(`compute/distance.py:38-49`): when the *partitioned* CES is the larger one (cut
created concepts, destroyed none), the new concepts' φ is priced at distance to the
**cut** null. The paper speaks of a single null concept p^uc of the candidate set.
main (2.0) changed this: one null concept from the **unpartitioned** system for both
sides (`pyphi/measures/ces.py:189,212` with `system=unpartitioned_system` at
`pyphi/formalism/iit3/__init__.py:318`). So the two oracle generations disagree
whenever mass moves between the partitioned CES and null. Effect is confined to
concept-creation / negative-deficit cases (C1→null pricing is uncut-null in both).
**iitx**: pick the uncut null (main's convention, closest to the paper), document that
1.2.1 differs in the creation corner.

### R5. `evaluate_cut`'s mechanism restriction: exactness asserted, not proved — but no counterexample found. [DESIGN-SMELL → empirically VERIFIED-SOUND]

`compute/subsystem.py:130-138` recomputes the partitioned CES only over (mechanisms
that were concepts uncut) ∪ (mechanisms split by the cut) — even with the
"non-approximate" default `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = False`. "Split"
(`models/cuts.py:59-89`) means *straddling* the cut (uses the dense cut matrix, not
the cm — conservative), so single-node mechanisms are never split, and an unsplit
mechanism's repertoires **do** change under the cut whenever the cut severs
mechanism→purview or purview→mechanism edges. The implicit theorem — a cut cannot
raise φ^Max of an unsplit mechanism from 0 — is stated in Mayner et al. 2018's
optimizations but not proved there.

Test: a from-scratch reimplementation of the 1.2.1 small-φ pipeline (BI partitions,
pyemd cause EMD, analytic effect EMD, PRECISION 6; validated to reproduce all seven
φ^Max values of `basic_network` exactly — `cax/mini_iit3.py`, `exp4_validate.py`)
searched 150 random 3–4-node networks (deterministic and stochastic), all 2ⁿ−2 cuts,
every unsplit zero-φ^Max mechanism: **zero counterexamples** (`exp5_cutsearch.py`;
~10⁴ mechanism×cut cases). Supporting argument (effect side): cutting only shrinks
each purview factor's mechanism-dependence set; a zero-EMD bipartition of the uncut
pair maps to a zero-EMD bipartition of the cut pair because the factor equalities are
preserved or re-groupable. The cause side (normalized joint) resists the same simple
argument. **iitx**: adopt the restriction as an optimization but keep a
verification mode (recompute the full CES for the winning cut, assert Φ unchanged);
GitHub issue #36 (open) asks for exactly this kind of transparency for the analogous
connectivity-based MICE optimization.

### R6. The SIA disk cache key omits result-changing config. [BUG]

`compute/subsystem.py:297-312` (`_sia_cache_key`) keys the joblib/db-cached `_sia` on
subsystem hash + {ASSUME_CUTS…, CUT_ONE…, MEASURE, PRECISION, VALIDATE_SUBSYSTEM_STATES,
SINGLE_MICRO…, PARTITION_TYPE}. Missing:

- `USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE` — switches the entire CES distance
  (`compute/distance.py:132-133`); changes Φ (golden: 2.3125 vs 1.083333 for the basic
  network).
- `PICK_SMALLEST_PURVIEW` — flips the MICE purview tie order
  (`models/mechanism.py:99`), changing concept repertoires and hence Φ.

Both are listed in the config docs' "settings that control the algorithms" block
(`conf.py:58-59`). With `CACHE_SIAS = True` (default False, `conf.py:440`) or the db
backend, changing either option silently returns stale Φ across runs (the joblib fs
cache persists). Low default exposure, real correctness bug when caching is on.
**iitx**: derive cache keys mechanically from the full set of semantics-affecting
options (single source of truth), or hash the entire resolved config.

### R7. `expand_repertoire`'s trailing normalization is a no-op that can only mask bugs. [VERIFIED-SOUND / DESIGN-SMELL]

`subsystem.py:432-470`: the expansion multiplies a normalized repertoire by the
normalized unconstrained repertoire of the *disjoint* complement — the product is
already normalized; `distribution.normalize` (`distribution.py:14-26`) is an identity
up to float error, and deliberately leaves an all-zero array all-zero (relevant only
for unreachable-state cause repertoires, which never reach expansion because concepts
require φ > 0). No divergence — but if a future purview-overlap bug ever produced an
unnormalized product, the normalize would hide it. iitx: expand without renormalizing
and assert Σ=1.

### R8. Macro pipeline: cut macro TPMs are silently projected onto conditional independence. [PAPER-DIVERGENCE, significant for all macro goldens]

`macro.py:288-301` (`_coarsegrain_space`) calls
`coarse_grain.macro_tpm(system.tpm, check_independence=(not is_cut))`;
`macro.py:545-568` then converts the coarse-grained state-by-state TPM to state-by-node
with `convert.state_by_state2state_by_node`. For the **uncut** system a conditionally
dependent macro TPM raises `ConditionallyDependentError` (`validate.py:73-87`), but for
**every evaluated system cut** the check is skipped and the conversion silently
replaces the cut macro TPM by its nearest conditionally-independent surrogate (per-node
marginals). Coarse-graining a cut micro system generically *does* create conditional
dependence, so essentially every macro Φ number (macro_network 0.597212, blackbox
goldens, `emergence()` results) embeds this undocumented projection. The theory offers
no license for it — the virtual-units doctrine says conditionally dependent TPMs are
not valid substrates at all. This is the macro-level analogue of oracle-finding #6 and
belongs in the same deferred-emergence spec: **iitx must decide** whether to reproduce
the projection (oracle fidelity) or reject/CI-model cut macro TPMs (theory fidelity);
the two give different numbers.

Related smaller items: `CoarseGrain.macro_state` (`macro.py:466-496`) hard-codes
*binary* macro states (`0 if … in grouping[i][0] else 1` — `grouping[i][1]` is never
consulted; `validate.coarse_grain` at `validate.py:204-217` enforces exactly two state
groups, so this is consistent but silently non-general). `validate.
conditionally_independent` uses `np.allclose` (rtol 1e-5!) — a far looser and
differently-shaped tolerance than `PRECISION`; borderline-dependent TPMs pass at
1e-6-scale dependence and are then projected by conversion.

### R9. Misc 1.2.1. [DESIGN-SMELL unless noted]

- `concept_distance` builds the union purview as `tuple(set(a + b))`
  (`compute/distance.py:29-30`) — deterministic only because CPython small-int hashing
  is identity; a sorted tuple would be self-evidently stable.
- `directional_emd`/`repertoire_distance` double-round (`distance.py:267,286`) —
  harmless, but shows rounding is sprinkled, not layered.
- `find_mip` returns the *first* zero partition as "the MIP" (`subsystem.py:590-592`);
  with `phi == 0` after rounding at 1e-6, solver noise decides which partition is
  reported (values unaffected). Replication requires matching enumeration order.
- Open GitHub issues relevant to correctness of environment, not math: #41 (s390x
  big-endian test failure — NumPy byte-order assumptions), #108/#129/#52 (modern
  Python breakage), #110 (1.2.1 never released to PyPI), #28 (CPU leak), #143
  (benchmark regression on main). No open issue documents a numerical bug beyond
  what this review covers.

---

## 2. IIT 4.0 branch (pyphi-iit4)

### R10. The GID pipeline matches the paper's equations — a verified mapping. [VERIFIED-SOUND]

Because so much of the review is negative, it is worth recording positively what was
checked line-by-line against Albantakis et al. 2023 and found to agree (modulo the
known missing |·|₊, oracle-finding #1):

| Paper | Code | Status |
|---|---|---|
| Eq. 3 (effect TPM, frozen background) | `subsystem.py:105-108` | ✓ |
| Eq. 4 (backward/cause TPM, causal marginalization weighted by p(w̄\|u)) | `tpm.py:675-709` (`backward_tpm`): weight = Σ_ŝ p(u\|ŝ,w̄)/Σ_û p(u\|û) exactly | ✓ (raises `StateUnreachableError` when the Eq. 4 denominator is 0) |
| Eqs. 5–9 (system ii; forward informativeness, backward cause selectivity) | `intrinsic_information` with mechanism=purview=S: selectivity = `repertoire(direction,…)` (effect repertoire / normalized backward product = Eq. 9), informativeness = forward vs. mean-filled unconstrained (`repertoire.py:102-117` = Eq. 8) | ✓ |
| Eqs. 17–18 (partitioned TPM = unit-wise uniform noising of cut inputs) | cut → cm → node TPMs marginalize severed inputs (`subsystem.py:126,142-149`) | ✓ |
| Eqs. 19–20 (φ_e/φ_c: selectivity at s′ × log forward ratio) | `evaluate_partition` GID branch `subsystem.py:745-764` + `new_big_phi.integration_value:199-228` | ✓ except no \|·\|₊ |
| Eqs. 28–33 (product purview probabilities, backward Bayes) | `_cause_repertoire:341-357` (normalized product = Eq. 33), `_effect_repertoire:415-432`, role-swap `forward_cause_probability` (`repertoire.py:41-58`) = Eq. 30 | ✓ |
| Eqs. 31–32 (unconstrained = mean over conditioning states) | `repertoire.py:85-117` | ✓ |
| Eqs. 34–37 (mechanism ii, state selection by ii before partitioning) | `intrinsic_information` (`subsystem.py:977-1056`), `find_mip` maps over tied states (`subsystem.py:893-914`) | ✓ |
| Eq. 38 (Θ(M,Z): M⁽ⁱ⁾=M ⇒ Z⁽ⁱ⁾=∅) | `all_partitions` (`partition.py:541-577`); the `parts[0]` check at :575 is sound because `set_partitions` (`combinatorics.py:283-296`) emits the one-block partition first with M at index 0. Partitions with several empty-mechanism parts are not enumerated, but they are φ- and normalization-equivalent to the single-empty-part form (Eq. 40's unconstrained is itself a product), so the enumeration is complete up to equivalence | ✓ |
| Eq. 16 semantics of ←/→/↔ flags | `_unidirectional_set_partitions` (`partition.py:769-791`) — the union cut-matrix reproduces X⁽ⁱ⁾ exactly | ✓ (see R14) |
| Eq. 48 (congruence: purview unit-states ⊆ s′) | `is_congruent` (`models/mechanism.py:110-116`) checks mutual units' states — equivalent since s′ covers all units | ✓ |
| Eq. 55 + S3 identity (φ_r = \|∩ (z_c ∪ z_e)\| · min φ_d/\|z_d\|) | `Relation.purview/phi` (`relations.py:138-154`), Unit(index,state) intersection builds congruence in, cause-vs-effect states of the same unit kept distinct | ✓ |
| S3 analytic Σφ_r and count | `sum_of_minimum_among_subsets` (`combinatorics.py:172-180`; counts 2^(K−j)−1 with ascending sort — matches), inclusion–exclusion in `AnalyticalRelations._num_relations` (`relations.py:324-337`) over `purview_inclusion` (`models/subsystem.py:66-79`) | ✓ |
| Relation existence ⟺ nonempty ∩ of purview unions | `_combinations_with_nonempty_congruent_overlap` (`relations.py:216-235`) — equivalent to face existence: if unit n is in every union, choosing per-distinction a purview containing n yields a valid face | ✓ |

### R11. Mechanism-level MIP search also short-circuits on the first φ ≤ 0 partition. [PAPER-DIVERGENCE — extends oracle-finding #2 downward]

`_find_mip_single_state` (`subsystem.py:824-830`) maps partitions with
`shortcircuit_func=utils.is_falsy`; `RepertoireIrreducibilityAnalysis.__bool__` is
`is_positive(phi)` (`models/mechanism.py:399-401`), and the serial `shortcircuit`
generator (`parallel/__init__.py:68-80`) stops at the first falsy result. With GID and
no |·|₊, partition φ can be negative, so for any reducible (mechanism, purview) the
*reported* MIP is the first partition in `all_partitions` order with φ ≤ 0 at
precision, possibly with negative φ — not the minimizer of the normalized key. φ_d and
distinction membership are unaffected (concept requires φ > 0), but reported MIP
certificates, `partitioned_repertoire`s, and tie sets for reducible pairs are
enumeration-order artifacts. Oracle-finding #2 documents this at the system level;
it is equally true at the mechanism level and must be replicated for
certificate-exact oracle tests.

### R12. `intrinsic_information` detects state ties by raw float equality — the only place in the pipeline without a tolerance. [BUG]

`subsystem.py:1039-1052`: `max()` over raw floats and
`if information == max_information` collect the tied specified states. In the GID path
the values come straight from `generalized_intrinsic_difference`
(`metrics/distribution.py:628-641`) — never rounded, never wrapped in `PyPhiFloat`
(wrapping happens *after* tie collection, `models/mechanism.py:61-62`). Everywhere
else ties are recognized at `PRECISION` (13). Consequence: two states whose ii differ
by 1e-15 due to summation order are *not* tied here (so `find_mip` never explores the
second state, and `resolve_congruence` at `models/mechanism.py:1028-1051` cannot swap
in a congruent alternative), while the same two values *are* "equal" for every later
PyPhiFloat comparison. Exactly-symmetric TPMs (the fixtures) produce bit-identical
values, which is why the goldens don't expose it; nearly-symmetric stochastic systems
will. **iitx**: one equality regime for tie detection, everywhere.

### R13. `PyPhiFloat` violates the eq/hash contract and is intransitive. [BUG]

`data_structures/pyphi_float.py:8-33`: `__eq__` is `math.isclose(rel_tol=1e-13,
abs_tol=1e-13)` (`utils.py:104-108`) but `__hash__` is `hash(round(x, 13))`.
Demonstrated concretely (`exp9_pyphifloat.py`): u = 0.8444218515250481,
v = 0.844421851525124 satisfy `u == v` yet `hash(u) != hash(v)`; `{u: 1, v: 2}` holds
**two** entries for "equal" keys. Since `SystemIrreducibilityAnalysis.__hash__`
includes `self.phi` (`new_big_phi/__init__.py:119-125`) and RIA/SIA objects go into
sets and dict-based caches, deduplication of tied objects can silently fail at
rounding boundaries. Independently, isclose-equality is not transitive
(a≈b, b≈c, a≉c — demonstrated), so `min`/`max`/`sorted` over PyPhiFloats are not
well-defined order statistics at tie boundaries; results depend on enumeration order.
This is the root cause behind several "ties are fragile" symptoms.
**iitx**: compare on a fixed integer grid (scale by 10^p and round once), giving a
genuine equivalence relation, a consistent hash, and order-independence.

### R14. SET_UNI/BI normalization: PyPhi's `cut_matrix.sum()` **equals** the paper's Σ|S⁽ⁱ⁾||X⁽ⁱ⁾| — oracle-finding #3's rationale is wrong. [CORRECTION to oracle-findings.md]

Oracle-finding #3 claims Eq. 23's Σᵢ|S⁽ⁱ⁾||X⁽ⁱ⁾| and the code's
`1/np.sum(cut_matrix)` (`models/cuts.py:293-295`) "differ whenever blocks' cut sets
overlap (e.g. a bipartition with flags (↔,↔) severs each cross edge once, not twice)".
This is not so: under Eq. 16, X⁽ⁱ⁾ enumerates the severed **inputs of part i**, so
every severed directed edge u→v is counted exactly once — in the term of the part
containing v — and the sum telescopes to the number of distinct severed ordered pairs,
i.e. `cut_matrix.sum()`. Verified exhaustively for every (set-partition, flag)
combination at n = 2, 3, 4 — 369 combos, zero mismatches (`exp3_norm.py`), including
(↔,↔) bipartitions (both give 2|S₁||S₂|). The *behavioral* content of finding #3
(divide by the union count) is correct and should stay; its mathematical rationale
should be corrected, because as stated it implies iitx would mismatch the paper —
it doesn't. The genuine divergences are narrower:

- the **complete** cut (`CompleteGeneralKCut`, all-ones matrix *including* diagonal,
  `models/cuts.py:335-343`) overrides the factor to 1/n while its own matrix sums to
  n²; but the complete cut is outside Θ(S) (Eqs. 14–16) and only appears for monads or
  with `SYSTEM_PARTITION_INCLUDE_COMPLETE=True` (`conf.py:866`) — a pure convention;
- the `GENERAL` scheme's arbitrary cut matrices have no Eq. 23 counterpart at all.

### R15. The mechanism-partition normalization has no published closed form — and Barbosa 2021 does not normalize at all. [PAPER-DIVERGENCE (provenance sharpened)]

`DISTINCTION_PHI_NORMALIZATION='NUM_CONNECTIONS_CUT'` divides by
`KPartition.num_connections_cut()` = Σᵢ |M⁽ⁱ⁾|·(|Z| − |Z⁽ⁱ⁾|)
(`models/cuts.py:502-510`, registry `models/mechanism.py:147-154`, with a
ZeroDivisionError→1 fallback). Eq. 43 of the 4.0 paper describes the denominator only
verbally ("maximum possible value … number of possible pairwise interactions
affected") and cites Barbosa et al. 2021 — but the Barbosa paper (checked: PMC8003304)
selects its MIP as a **plain argmin, with no normalization**. So the closed form
exists only in PyPhi; it is self-consistent (disjoint parts ⇒ per-pair count = union
count, so no R14-style ambiguity), but iitx should treat it as
implementation-defined, cite PyPhi as the source, and note that reproducing
Barbosa 2021 itself would require normalization NONE.

### R16. S1 Text tie resolution: implemented only at the bottom of the hierarchy. [PAPER-DIVERGENCE]

What the branch actually does versus the S1 Text procedure:

1. *System cause/effect state ties* (S1 step 1: pick state maximizing φ_s; step 2:
   then Φ; then disqualify): **not implemented** — `system_intrinsic_information`
   (`new_big_phi/__init__.py:32-62`) says "state ties are arbitrarily broken (for
   now)" and takes `ties[0]` in `itertools.product` enumeration order. This is not
   hypothetical: for the xor golden fixture (state (0,0,0)), the system cause state is
   *exactly* tied between (0,0,0) and (1,1,1) at ii_c = 1.0 (computed directly from
   Eqs. 5–9, `exp6_state_ties.py` — the global spin-flip is a TPM automorphism).
   PyPhi reports (0,0,0) because it enumerates it first. The choice propagates into
   Eq. 48 congruence filtering and every relation, i.e. into the entire golden
   Φ-structure. This also gives oracle-finding #5 (state enumeration order) a concrete
   blast radius: for a tie between, say, (0,1,1) and (1,0,0), PyPhi's
   last-index-fastest order picks (0,1,1) while a little-endian enumeration picks
   (1,0,0) — same φ_s, different reported cause states, different congruent
   distinction sets.
2. *System MIP ties* (S1 step 4): implemented — hardcoded key `(normalized_phi, −phi)`
   (`new_big_phi/__init__.py:290-294`), matching "largest unnormalized φ_s among tied
   normalized" (but bypassing the `MIP_TIE_RESOLUTION` config, as already noted in
   `pyphi.md`).
3. *Mechanism state/purview ties*: state ties carried and re-resolved by φ
   (`resolve_ties.states`, strategy PHI) and congruence substitution among ties
   (`models/mechanism.py:1028-1051`) — matches the S1 heuristics; the Φ-maximizing
   deep resolution is (unsurprisingly) not attempted; `next(filter(...))` takes the
   first congruent tie in enumeration order.
4. *Complex ties and the disqualification rule* (S1 steps 2–3: tied-in-Φ systems
   "fail the information postulate and are **not complexes**"): **entirely absent** —
   `maximal_complex` (`new_big_phi/__init__.py:578-582`) is a plain `max` (first-wins
   on ties), and nothing ever disqualifies a system for irresolvable ties.

**iitx**: implement 2 and the tie-carrying of 3 for oracle parity; implement 1 and 4
behind a "normative S1" flag as a differentiator (see R23).

### R17. Infinity handling in the GID is fragile but — inside the formalism — unreachable. [VERIFIED-SOUND / DESIGN-SMELL]

`pointwise_mutual_information_vector` (`metrics/distribution.py:664-666`) maps 0/0→0
but leaves ±inf to `np.nan_to_num`'s defaults, i.e. ±1.7977e308. A partition or
unconstrained reference with q = 0 while p > 0 would produce a φ of ~1.8e308.
Analysis shows this cannot arise from the pipeline's own quantities: every partitioned
or unconstrained probability is a mean over a state set *containing the actual
conditioning state*, hence ≥ p/K > 0 whenever p > 0 (mechanism partitions:
subset-products dominate full products; system partitions: unit marginalization
includes the intact configuration). States with p = 0 get selectivity 0 and
`0 × (−1.8e308) = −0.0`, which orders correctly below any positive ii. So no golden
can be corrupted — but the invariant is nowhere asserted, and any future measure
plugged into the registry (or a user-supplied reference distribution) breaks it
silently. Related nit: `intrinsic_difference`'s docstring
(`metrics/distribution.py:561-588`) claims p·log(p/q) := 0 when q = 0, but
`rel_entr` returns +inf there — the docstring describes a convention the code does
not implement. Also `repertoire_distance` (`metrics/distribution.py:730-739`) hides
genuine TypeErrors inside measure functions behind its three-level signature-probing
try/except.

### R18. Small iit-4.0 branch items. [DESIGN-SMELL unless noted]

- Dead `partitions == "GENERAL"` branch (`new_big_phi/__init__.py:350-360`) — known
  (oracle-findings context); note the additional consequence that the
  `is_disconnecting_partition` filter is unreachable for *every* scheme, so under
  `SYSTEM_PARTITION_TYPE="GENERAL"` non-disconnecting cut matrices are evaluated as
  MIP candidates, which is not obviously meaningful for Θ(S).
- `integration_value` (`new_big_phi/__init__.py:210-215`) indexes a
  `forward_cause_repertoire` built with `np.empty` and only one state filled
  (`repertoire.py:61-82`) — correct as used, but the array containing uninitialized
  memory flows into `StateSpecification.repertoire` in other call paths; any consumer
  iterating it reads garbage.
- `maximal_complex` mixes types: `max` over SIA objects with a `NullPhiStructure`
  default (`new_big_phi/__init__.py:578-582`).
- The legacy joblib SIA cache (and its R6 key bug) is gone on this branch — `sia`
  is uncached (`compute/subsystem.py:310-314`); nothing to replicate.
- `resolve` (`resolve_ties.py:87-97`) computes `extremum = operation(values,
  default=default)` where `default` is a *phi-object*, not a key tuple — harmless
  today (the zip is empty exactly when the default is used) but a type error waiting
  for a refactor.

---

## 3. Cross-cutting

### R19. Two incompatible rounding regimes across oracle generations. [DESIGN-SMELL, replication-critical]

1.2.1 rounds every distance to PRECISION=6 *at creation* and compares rounded values —
stable, order-independent, transitive; deliberately coarse to absorb pyemd noise
(main's iit3 preset keeps 6 for exactly that reason). The iit-4.0 branch mixes three
regimes: rounded-at-13 (`repertoire_distance`), raw floats (GID ii values, R12), and
isclose-at-1e-13 (`PyPhiFloat`, R13 — with a *relative* tolerance component, so
"equality" scales with magnitude). Any iitx comparison harness must emulate the right
regime per code path; a single global tolerance will both miss and invent ties
relative to the oracle. Round-then-compare *does* create false ties by design
(anything within 5e-7 in 1.2.1), but it never creates order-dependence; isclose does.

### R20. Conversions and config. [known items, one addition]

The silent sbs→sbn marginalization, the 1.2.1 endianness docstring bug, and CWD config
loading are documented in `pyphi.md` §1/§7. Addition: the only *validated* CI check
(`validate.py:73-87`) uses `np.allclose` with rtol=1e-5 — so a TPM can pass Network
validation while its round-trip differs by ~1e-5·p, an error 10× the 1.2.1 EPSILON;
CI validation and numeric tolerance are not the same scale anywhere.

---

## 4. IITX-OPPORTUNITIES (consolidated)

- **O1 — Well-posed CES distance.** Implement the extended EMD as an explicitly
  balanced bipartite transport with the null concept on *both* margins (main's
  `total − sumᵢ` construction) and a single (uncut) null point. Reproduces 1.2.1
  (via R3's equivalence) while being solver-independent and paper-faithful; no
  blocking constants needed (R2).
- **O2 — Exact, replicable EMD.** pyemd's approximation noise is the reason for
  PRECISION=6 and several tie artifacts. A small exact min-cost-flow (or LP) solver on
  Hamming ground metric — with the analytic product-form shortcut on the effect side
  (R1) — lets iitx run at much finer precision and *characterize* pyemd's deviation
  rather than inherit it, while a "oracle-compat" mode rounds at 6.
- **O3 — Lawful tie machinery.** Replace tolerance-equality with fixed-point grid
  comparison (transitive, hashable — fixes R12/R13), make every tie set explicit and
  serializable (PyPhi drops `_ties` on JSON), and implement S1 Text steps 1–4
  including the disqualification rule behind a flag (R16). This is a place where iitx
  can be strictly more normative than the reference implementation.
- **O4 — Verified optimizations.** Ship the `evaluate_cut` mechanism restriction (R5)
  and connectivity purview filters as *checked* optimizations with an audit mode; the
  randomized-search harness in `cax/` is reusable as a fuzzer against the JAX
  implementation.
- **O5 — Macro honesty.** Refuse (or explicitly model with virtual units) conditionally
  dependent cut macro TPMs instead of silently projecting (R8); expose the projection
  as a named, documented approximation when oracle parity is wanted.
- **O6 — Config-complete memoization.** Hash the full resolved semantic config into
  every persistent cache key (R6).
- **O7 — Normalization provenance.** Keep `cut_matrix.sum()` (= Eq. 23, R14) and
  `num_connections_cut` (implementation-defined, R15) but document each with its
  actual provenance; expose Barbosa-2021-style unnormalized mechanism MIP as an option.
- **O8 — Sign conventions as policy.** One config axis for {paper |·|₊ clipping,
  oracle no-clipping} at both mechanism and system levels, with the short-circuit
  semantics (finding #2 + R11) tied to the same axis, so "reducible" means one thing.

---

## 5. Corrections to existing notes

- **oracle-findings #3**: behavioral statement stands (divide by `cut_matrix.sum()`),
  but the rationale ("the two differ whenever blocks' cut sets overlap … (↔,↔) severs
  each cross edge once, not twice") is incorrect — Σᵢ|S⁽ⁱ⁾||X⁽ⁱ⁾| under Eq. 16 counts
  each severed edge exactly once (by destination part) and equals `cut_matrix.sum()`
  for every SET_UNI/BI partition (exhaustive check n ≤ 4, `exp3_norm.py`). The real
  divergence is confined to the out-of-Θ complete cut (1/n vs. n² matrix sum) and to
  schemes with no Eq. 23 counterpart.
- **oracle-findings #2** extends to the mechanism level (R11).
- **oracle-findings #5** has a concrete consumer: system-state tie breaking (R16.1) —
  the xor fixture's cause state is genuinely tied, and enumeration order is what picks
  (0,0,0).

---

## 6. Verdict on PyPhi as an oracle

PyPhi is a **strong value-oracle and a weak certificate-oracle**. Everywhere this
review could check the arithmetic against the papers' equations, the *values* (φ, Φ,
repertoires, relations) are computed as the papers say or as their documented,
deliberate divergences say — including several spots that look like bugs but turn out
sound (analytic effect EMD, GID equation mapping, Eq. 23 normalization, infinity
corner cases). The genuine defects cluster in three areas: (1) **edge-regime
machinery that works by accident** — the negative-deficit transport relies on
undocumented pyemd behavior, the blocking constant guards against a solver that is no
longer used; (2) **everything tie- and certificate-shaped** — which partition/state/
purview is *reported* is an artifact of enumeration order, short-circuiting,
inconsistent equality regimes, and a broken float type, so MIP certificates, tie
sets, and specified states should only be trusted after replicating PyPhi's exact
iteration order and rounding regime; (3) **the macro pipeline**, whose cut-level CI
projection is an undocumented approximation baked into every macro golden. For iitx:
trust the golden *numbers* (with PRECISION-6 slack on the 3.0 side), never trust a
golden *witness* without also fixing the enumeration order, and treat macro goldens
as specifications of PyPhi's algorithm rather than of the theory.
