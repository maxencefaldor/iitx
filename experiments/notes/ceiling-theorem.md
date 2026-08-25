# The 2026 ceiling in closed form

Status: derivation note, 2026-08-25. Resolves the e03 backlog item "derive the n = 3
optimum in closed form". Every constant below is checked against e03's measured
optima.

## Theorem

For any system of n binary units in any state, under IIT 4.0 (2026),

**φ_s ≤ C(n), where C(n) is the unique positive root of t·2^t = n − t,**

and the bound is approached by systems whose specified cause state has backward
probability p_c = 2^(−C(n)). At n = 3 the root is rational: t = 1 solves
1·2¹ = 2 = 3 − 1, so **C(3) = 1 ibit exactly, at p_c = ½** — the value e03's ascent
attained to six decimals.

For units with arbitrary finite alphabets, replace n by log₂|Ω|, where |Ω| is the
number of joint states.

## Derivation

Write Q = 2ⁿ, and let L(u) = P(s | u) be the likelihood of reaching the current
state s from prior state u.

1. **The cause repertoire is the Bayes posterior under a uniform prior** (the
   theory's own maximum-entropy convention for causes): π_c(u) = L(u) / Σ_w L(w).
   Let p = π_c(s_c) be the specified cause state's probability.
2. **The cause informativeness collapses to a function of p alone.** The
   implementation (matching the oracle) computes
   ii_c = π_c(s_c) · log₂( L(s_c) / mean_w L(w) ), and since
   L(s_c)/mean(L) = π_c(s_c) · Q, this is **ii_c = p·log₂(pQ) = p(n + log₂ p).**
3. **The cause surprisal is −log₂ p** by definition.
4. The 2026 cap is bounded by the cause side alone:
   φ_s ≤ min(ii_c, surp_c) = min( p(n + log₂ p), −log₂ p ). The first term
   increases in p (for pQ > 1/e-ish territory), the second decreases; the max over
   p of the min is at their crossing. Substituting t = −log₂ p, p = 2^(−t), the
   crossing is p(n − t) = t, i.e. **t·2^t = n − t**. Uniqueness: t·2^t is
   increasing from 0, n − t is decreasing, so exactly one positive root. ∎

**Why the cause side, and only the cause side, gives a universal bound.** The
effect-side unconstrained repertoire is the prior-averaged product of unit
transitions — a system-dependent quantity — so ii_e is not a function of p_e alone
and admits no analogous system-free bound. The cause prior, by contrast, is uniform
*by the theory's definition of unconstrained causes*. This is why e03 found the
binding term to be the cause surprisal at every measured size: the cause side is
where the universal ceiling lives.

## Verification against e03

| n | C(n) (root) | p* = 2^(−C) | e03 measured φ | e03 measured p_c |
|---|---|---|---|---|
| 2 | 0.746806 | 0.595922 | 0.628540 | 0.6461 |
| 3 | **1.000000** | **0.500000** | **1.000000** | **0.5000** |
| 4 | 1.208250 | 0.432793 | 1.208219 | 0.4328 |
| 5 | 1.384635 | 0.382986 | 1.383296 | 0.3833 |

n = 4 agrees to 3×10⁻⁵ (the run was, in hindsight, essentially converged); n = 5 to
1.3×10⁻³ (the caveated under-convergence). The theorem also *explains* e03's
observations that the binding side is always the cause and that p_c marches down
0.646, 0.500, 0.433, 0.383 with n: those are the crossing points.

## Remarks

- **Integer ceilings.** C(n) = k exactly when n = k·(2^k + 1): C(3) = 1,
  C(10) = 2, C(27) = 3, C(68) = 4, … The "one-ibit ceiling" of e03 is the k = 1
  member of an exact integer family.
- **Asymptotics.** t·2^t = n − t gives C(n) = log₂ n − log₂ log₂ n + o(1): the 2026
  formalism's maximal integrated information grows **logarithmically** in system
  size (against n(n−1) under 2023 — a change of growth class, now provable rather
  than measured). The empirical "≈ 0.6·log₂ n" of e03 was this law seen through a
  4-point window.
- **Attainability.** C(n) is an upper bound; ascent attains it at n = 3, 4, 5
  (above). At n = 2 the best found is 0.6285 < C(2) = 0.7468, with *both*
  surprisals binding at the optimum (p_c ≈ p_e ≈ 0.646): with only two units, the
  effect side and the uncapped MIP apparently cannot reach the cause crossing.
  Characterizing the n = 2 optimum, and proving attainability for n ≥ 3, remain
  open.
- **Sharpening.** The bound uses only two of the cap's four terms. Whether ii_e /
  surp_e / the uncapped MIP force a strictly smaller ceiling at some sizes (as at
  n = 2) is the remaining question; empirically they do not for n = 3–5.
