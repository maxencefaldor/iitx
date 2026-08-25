# The in-degree ladder

Status: probe result, 2026-08-25, from the e01/e09 sweeps (whole-universe, n = 3);
the centerpiece question for e11. Derivation open.

## The observation

Conditioning the deterministic universe on m = in-degree of the analyzed state
(its number of preimages), the conditional maxima of φ_s(2023) form a ladder:

| m | count | max φ₂₃ | m · max | max cause-capped |
|---|---|---|---|---|
| 0 | 5,764,801 | 0 | — | 0 |
| 1 | 6,588,344 | 6.000 | **6 = n(n−1)** | 0 |
| 2 | 3,294,172 | 3.000 | **6 = n(n−1)** | 1.000 |
| 3 | 941,192 | 1.000 | **3 = n** | 1.000 |
| 4 | 168,070 | 0.750 | **3** | 0.750 |
| 5 | 19,208 | 0.600 | **3** | 0.600 |
| ≥6 | 1,429 | 0 | 0 | 0 |

Two plateaus of m·g(m) — n(n−1) for m ≤ 2, n for 3 ≤ m ≤ 5 — and a cliff to zero
at m = 6. The cause-capped conditional maxima are exactly min(g(m), log₂ m),
which is why the repaired measure's global max (1.0) is attained from both m = 2
and m = 3.

## The anatomy (per-m winners; all select the complete cut and are fixed points)

The specified-state informations follow the in-degree exactly:
ii_c = (1/m)·log₂(Q/m) and ii_e = log₂(Q/m) (fixed point ⇒ the successor's
unconstrained probability is m/Q). The φ values decode as
**g(m) = (1/m)·log₂(1/L*)**, where L* is the maximal achievable suppression of the
specified cause state's partitioned likelihood under the complete cut:

- m ≤ 2: L* = 2^(−n(n−1)) — the full severed capacity of `maximum-theorem.md`;
- 3 ≤ m ≤ 5: L* = 1/Q exactly — suppression saturates at the state count;
- m ≥ 6: reducibility (φ = 0; with ≥ Q−2 preimages the units are near-constant).

## Open (e11)

1. Prove the threshold: why does the achievable suppression drop from 2^(−severed)
   to 1/Q at m = 3 (= Q/2 − 1 at n = 3? = 3 absolutely? the n = 4 ladder decides).
2. Prove the cliff (m ≥ 6 ⇒ reducible) and its n-scaling.
3. The n = 4 ladder by targeted per-m search: **e11's first attempt failed its
   own anchor test** (search g̃(1) = 4.245 vs the known g(1) = 12 from all-OR):
   minterm-saturated rung winners are unreachable by random + single-bit hill
   climbing. Requeued with constructive seeding from the exact characterization
   (build per-m candidates satisfying the double-minterm conditions as nearly as
   m allows).
4. Consequence for the repaired measure: its deterministic ceiling is
   max_m min(g(m), log₂ m); with the empirical g this is exactly 1 at n = 3.
   Its growth in n follows from g's.
