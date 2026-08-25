# Metatheorems: what the definitional knobs force

Status: theory note, 2026-08-25, consolidating `ceiling-theorem.md`,
`maximum-theorem.md`, and e09. The subject is not IIT's phenomenological axioms but
the *formalization layer*: which properties of a φ-measure are forced by which
definitional choices, established by proof where possible and by whole-universe
enumeration (n = 3) where not.

## Desiderata (operationalized in e09)

D1 determinism lives (some deterministic system > 0) · D2 isolation rejected
(frozen-isolated states not maximal) · D2′ freezing rejected (no fixed point
maximal) · D3 information bound (deterministic max O(n)) · D4 non-vacuous on the
deterministic universe.

## Metatheorems

- **M1 (effect surprisal ⇒ ¬D1). Proven.** Any measure capped by the rectified
  surprisal of the specified effect state assigns 0 to every deterministic system:
  determinism makes that surprisal −log₂ 1 = 0. This is the single term
  responsible for 2026's deterministic extinction; no other ingredient of the 2026
  cap has this effect (an ii-only cap leaves deterministic systems alive).
- **M2 (capacity, knob-independent). Proven.** Any measure whose per-cut value is
  min(φ_c, φ_e) with factored effect repertoires obeys φ(cut) ≤ severed(cut) — one
  ibit per severed binary connection — because the partitioned repertoire contains
  the actual severed configuration with weight 2^(−k). Consequences: normalized
  selection ⇒ ceiling n(n−1) (2023); raw selection ⇒ ceiling = minimal severed
  count = n − 1 (linear — raw selection incidentally satisfies D3, e09-P3).
- **M3 (the frozen class is selection-invariant). By enumeration.** The 3,591
  capacity attainers under normalized selection are *exactly* the maximizers under
  raw selection (e09): the isolation pathology is created by delta-repertoire
  saturation, not by the severed-count normalization. Partial characterization:
  attainment *requires* unique cause (in-degree 1) and cross-minterm effect (each
  unit's output realized by exactly one cross-input configuration) — necessary,
  proven by enumeration with zero exceptions; sufficiency needs an additional
  per-cut cause-saturation condition, exact form open. The class contains both
  frozen-isolated states (3,584) and deterministic transit states (7), unifying
  with e02's stochastic transit needles.
- **M4 (occupancy fixes isolation, not freezing). By enumeration (e02).**
  Occupation weighting collapses the isolated plateau (≤ 1.5) but its optimum
  (3.0) is still a globally attracting fixed point: D2 recovered, D2′ not.
- **M5 (the repair: cause-only cap ⇒ full battery pass). By enumeration (e09).**
  Capping 2023-φ_s by the *cause* surprisal alone — min(φ_s, −log₂ p_c), p_c the
  specified cause state's backward probability — passes D1–D4 on the entire
  deterministic universe at n = 3: max 1.0000 attained by 136,602 ordinary
  in-degree-2 states; all-OR → 0, xor → 1. Rationale: every deterministic
  transition has a delta *effect* repertoire (so effect-side sharpness penalties
  cannot separate pathology from health), but only frozen-isolated states have
  delta *cause* repertoires — the cause side is where isolation is visible.
  Conjectured ceiling: max over in-degree m of min((1/m)·log₂(Q/m), log₂ m)
  (= 1 at n = 3, both terms binding at m = 2, matching the winners exactly);
  proof open.

## The satisfiability map (n = 3, whole-universe)

| measure | D1 | D2 | D2′ | D3 | D4 |
|---|---|---|---|---|---|
| 2023 (fixed state) | ✓ | ✗ | ✗ | ✗ | ✓ |
| 2026 (fixed state) | ✗ | ✓ | ✓ | ✓ | ✗ |
| 2023, raw selection | ✓ | ✗ | ✗ | ✓ | ✓ |
| 2023, occupation-weighted | ✓ | ✓ | ✗ | ✓ | ✓ |
| **2023, cause-capped** | **✓** | **✓** | **✓** | **✓** | **✓** |

## Corollary: does the theory require indeterminism? Only per level

M1 says a 2026-conscious substrate must have a stochastic TPM *at its own level*.
It does not follow that the physics must be indeterministic: coarse-graining
deterministic micro dynamics generically yields stochastic macro TPMs with
positive macro φ_s(2026) — measured up to **0.70** (70% of C(3)) for random
deterministic 9-unit rules coarse-grained to three 4-symbol row units; only
special orders (majority freezing, pure shifts) stay at zero. Hence, under
IIT-2026: (i) consciousness requires stochasticity at the substrate's level, from
*any* source — quantum, thermal, or averaging; (ii) in a strictly deterministic
world, consciousness is necessarily emergent (the deterministic micro level scores
exactly 0, so exclusion pushes every complex to a coarse-grained level); (iii) "no
noiseless system is conscious" is a theorem; "consciousness needs quantum events"
is not — it would additionally require rejecting the theory's own macro doctrine
*and* a collapse interpretation of quantum mechanics.

## Open problems

1. Prove the cause-capped ceiling formula and its n-scaling (its n = 3 value
   equals C(3) = 1 — same crossing structure as the 2026 stochastic optimum;
   coincidence or common theorem?).
2. Close the attainment sufficiency condition (M3).
3. Prove C(n) attainability for n ≥ 3 (currently numerical certificates).
4. An impossibility candidate for the family of *fixed-state, cap-free* measures:
   is delta-repertoire saturation always maximal (so D2 requires either a cap or
   occupancy)? M3's selection-invariance is the first evidence.
5. Re-run the battery at n = 4 (spot checks; enumeration is gone above n = 3).
