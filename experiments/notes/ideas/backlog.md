# Experiment backlog

Candidates, roughly ordered. Each becomes a notebook when it is next.

- **Close the 2026 growth law** (from e03). The ceiling is cause-surprisal-bound at
  every measured n with φ = −log₂ p_c and p_c = 0.646, 0.500, 0.433, 0.383 for
  n = 2..5 (≈ 0.6·log₂ n). Three follow-ups: derive the n = 3 optimum in closed form
  (ascent attains exactly 1.000000 at p_c = ½); converge n = 4–6 with bigger budgets
  to pin the law; characterize what the 2026 optima *compute* (attractors, logic).
- **e03 (done) — the 1-ibit ceiling.** e02's 2026-objective ascent endpoints crowd
  φ_s(2026) = 1.0 at n = 3 (best 0.999996 of 1,024 seeds). Derive the bound from the
  rectified-surprisal cap (the optimum should balance informativeness against
  surprisal, p ≈ 1/2 territory) or refute it; establish its n-scaling. Since the
  2026 landscape is interior-only (all deterministic systems score 0 — oracle-
  confirmed), this is gradient-first territory by construction.
- **Transit states.** e02 showed ascent's 2023 needles are not soft-frozen: the
  analyzed state has ~0 dwell probability and ~0.87 in-flow — strongly caused,
  strongly causing, never staying. Characterize the architecture (it, not the frozen
  state, may be the theory-relevant 2023 maximizer at 4.885 ibits) and its
  Φ-structure.
- **Weighted ascent.** Ascend E_s[φ_s] under the occupation measure directly (the
  stationary distribution of a stochastic system is differentiable); compare against
  e02's weighted atlas winner (reachable frozen state, 3.0 ibits).

- **e02 — ascent phenomenology at n=4.** The mechanistic predictions from
  `docs/notes/phi-optimization.md` §3, each falsifiable: determinism–degeneracy plane
  trajectories; cut-spectrum compression (maximin theory predicts several cuts *tie*
  at convergence); active-MIP switch events as the landscape's phase transitions; the
  dissociation question (ascend φ_s only — do distinctions, relations, Φ follow?).
- **Descent under a fixed input–output map.** The unfolding argument (Doerig et al.
  2019) as an experiment: minimize φ_s while pinning behavior; the path from a high-Φ
  system to its low-Φ functional twin is a picture of what the theory counts.
- **Versions as adversaries.** Ascend 4.0 φ_s while monitoring 3.0 Φ and the 2026 cap;
  systems of maximal disagreement are the definition's fault lines. 3.0's exact EMD has
  no gradient (host callback) — 3.0 can be monitored exactly but only descended via the
  Sinkhorn surrogate.
- **Sensitivity maps vs lesions.** ∂φ_s/∂TPM as an exact integration map, compared to
  the mechanism-φ decomposition and to edge-removal deltas.
- **Constrained ascent.** Sparsity/wiring budgets; a parameterized physical family
  (Glauber–Ising ring) to test whether ascent walks to the critical surface
  (Aguilera 2019; Popiel et al. 2020).
- **Reach vs legibility, measured.** Same ascent under node-TPM logits (CI-native)
  vs joint state-by-state logits (correlated noise allowed): does the extra reach buy
  φ_s, and is anything found there readable?
- **Expected-φ over reachable states.** Ascending φ_s at a fixed state invites the
  optimizer to exploit unreachable-state causal structure; compare fixed-state ascent
  with E_s[φ_s] over the chain's stationary distribution and hunt that exploit.
