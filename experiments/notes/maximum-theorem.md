# The 2023 maximum in closed form: φ_s ≤ n(n−1), attained

Status: derivation note, 2026-08-25. Companion to `ceiling-theorem.md` (which does
the same for 2026). Together they close both growth classes: the 2023 maximum is
exactly n(n−1); the 2026 maximum is at most C(n) ~ log₂ n.

## Theorem

For **any** system of n binary units — deterministic or stochastic — in any state,
under IIT 4.0 (2023):

**φ_s ≤ n(n−1),**

with equality for the all-OR system at the all-off state. For general alphabets the
bound is Σ log₂|A_i| over the ordered pairs (i → j) severed by the complete cut,
i.e. (n−1)·Σ_i log₂|A_i|.

## Proof

**Lemma (per-connection capacity).** For any cut severing k connections and any
effect state v, the partitioned effect repertoire satisfies q(v) ≥ 2^(−k)·p(v),
where p is the unpartitioned repertoire. *Proof:* the partitioned repertoire
replaces each severed input by its uniform average; for unit j with k_j severed
inputs, the average over the 2^(k_j) severed-input configurations includes the
actual configuration with weight 2^(−k_j), and probabilities are nonnegative, so
q_j(v_j) ≥ 2^(−k_j)·p_j(v_j). Multiplying over units and Σk_j = k gives the claim. ∎

**Corollary.** φ_e(cut) = selectivity · log₂(p/q) ≤ 1 · k = severed(cut), for every
cut and every effect state. Hence φ(cut) = min(φ_c, φ_e) ≤ severed(cut).

**The bound.** Whatever cut the minimum-information-partition selection returns,
φ_s equals that cut's φ, which by the corollary is at most its severed count, which
is at most the complete cut's n(n−1). ∎ (Note the bound needs *only* the effect
side, and holds for stochastic systems — it is not a deterministic-universe
statement. e01's exhaustive n = 3 maximum of 6.0 and the absence of any stochastic
ascent endpoint above 6 are both instances.)

**Attainment.** For the all-OR system at the all-off state, every inequality is
tight on every cut simultaneously: unpartitioned, each unit stays off with
probability 1 (selectivity 1); partitioned, an OR unit with k_j severed inputs
stays off with probability exactly 2^(−k_j) (it stays off iff every severed input
draws 0), so log₂(p/q) = k_j exactly; the cause side matches (the all-off state is
its own unique preimage). Thus **every cut has normalized φ exactly 1** — verified
numerically over all 22 / 150 / 1,061 cuts at n = 3 / 4 / 5 to 10⁻⁶ — every cut is
a minimum-information partition, and the tie resolution (maximal φ among normalized
ties) selects the complete cut: φ_s = n(n−1). ∎

## Remarks

- **Interpretation: φ_s(2023) is bounded by the wiring capacity of the cut.** One
  ibit per severable binary connection; the maximizer saturates every connection of
  every cut at once — it is a *perfect maximin equalizer*, which retroactively
  explains why e01 found the maximal plateau gapped away from the rest of the
  landscape.
- Numerical margin: over 400 random stochastic systems at n = 3, 4 (all cuts, all
  states probed), the largest observed normalized φ was 0.562 — random systems sit
  far from the capacity bound; only the frozen-isolation family reaches it.
- With `ceiling-theorem.md`, both formalisms' maxima are now closed-form: **2023:
  exactly n(n−1) (quadratic, attained by frozen isolation); 2026: at most C(n),
  root of t·2^t = n − t (logarithmic, attained empirically for n = 3, 4, 5 by
  calibrated-stochastic systems).** The growth-class separation between the two
  versions of the theory is fully proven on the 2023 side and proven-as-upper-bound
  on the 2026 side.
