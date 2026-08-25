# Experiment backlog

Candidates, roughly ordered. Each becomes a notebook when it is next.

- **The MIP-surprisal crossing family** (e10 + e11): three optima now share one
  structure — the n = 2 2026 ceiling (0.64004660), the repaired measure's
  stochastic ceiling at n = 3 (≥ 1.3916), and possibly higher n — all sit where
  the uncapped MIP equals the cause surprisal. Derive the family: symbolic n = 2
  first (tractable), then the general crossing equations.
- **The n = 4 ladder, constructively seeded** (e11): build per-m rung candidates
  from the exact double-minterm characterization instead of random search; the
  anchor tests (g(1) = 12 via all-OR; g(2) vs the weighted search's [3.5, 6.0])
  are already in place.
- **The weighted-deterministic growth law, reopened** (e10). The "= n" conjecture
  was an n = 3 coincidence (n(n−1)/2 = n only there). Global attraction forces
  in-degree ≥ 2 at the attractor, so the ladder bounds the weighted max by
  g(2) = n(n−1)/2; e10's in-tree search at n = 4 found only 2.33 of that bound's
  6.0 — either the m = 2 rung is incompatible with global attraction (the n = 3
  rung winner contains a 2-cycle) or the search was weak. Needs a stronger search
  seeded by ladder-rung structures, and the n = 4 ladder itself.
- **The n = 2 closed form** (e10). At the n = 2 2026 optimum, MIP = surp_c =
  surp_e (triple point, verified to 2.5×10⁻⁴): with two units the MIP is
  symbolically tractable — derive φ*(2) = 0.64004660 from the triple-point
  equations.
- **Adversarial axiomatics II** (from e09 + `../metatheorems.md`): prove the
  cause-capped ceiling formula max_m min((1/m)log₂(Q/m), log₂ m) and its
  n-scaling (n = 3 value equals C(3) = 1 — same cause-surprisal crossing as the
  2026 stochastic optimum: coincidence or common theorem?); close the attainment
  sufficiency condition; the fixed-state cap-free impossibility candidate (is
  delta saturation always maximal?); battery spot-checks at n = 4; stochastic
  battery (the deterministic universe is only the enumerable slice).
- **"Only per level": determinism, stochasticity, and emergence** (from the
  quantum-events discussion): T4 + the macro computation (deterministic micro
  rules generically give macro φ_s(2026) up to 0.70) yield the corollary that
  IIT-2026 consciousness in a deterministic world is necessarily emergent; "no
  noiseless system is conscious" is a theorem, "needs quantum" is not. Write as a
  discussion subsection of the landscape paper or a standalone short note.
- **The behavioral φ-requirement as an object of study** (e08 established the
  floor is real, behavior-specific, 0–2.08 ibits, with a validated control; the
  ~1-ibit echo of e03 was coincidence). Next questions: what behavioral features
  predict the floor (hidden-memory depth? visible-entropy?); how floors scale with
  n and horizon K; floors under the 2026 objective.
- **(done as e08)** the ~1-ibit unfolding floor (from e04).
  Original item: Strictly behavior-pinned descent
  bottoms out at φ_s ≈ 1.0 at n = 4 and sits there for 900 steps. Real floor
  (this behavior *needs* ~1 ibit) or optimizer stuck? Test: restarts from the floor,
  basin-hopping, and pinning a *feedforward-realizable* behavior (whose floor should
  be 0). Note the value's echo of e03's 1-ibit ceiling is across different
  objectives (2023 vs 2026) — probably coincidence, but check.
- **(drafted as `paper/behavioral-requirement.md`)** paper 2 — "the behavioral price of integrated information." e04's three dissociations
  in one experiment: φ_s is behaviorally almost-free (90% removable at 0.5% error),
  architecturally invisible (input-dependence unchanged under ten-fold φ removal),
  and internally dissociated (big Φ *rises* 22.2 → 24.3 while φ_s falls 10.6 → 1.0;
  IIT 3.0 moves −7%). With e06's phenomenology this is a coherent second story,
  distinct from the landscape paper.

- **Close the 2026 growth law** (from e03; ceiling now PROVEN — see
  `../ceiling-theorem.md`: φ_s(2026) ≤ C(n), root of t·2^t = n − t, exact 1 at
  n = 3, C(n) ~ log₂n − log₂log₂n; verified to 3×10⁻⁵ at n = 4). Remaining:
  prove attainability for n ≥ 3 (empirical so far); characterize the n = 2 optimum
  (bound not attained, both surprisals bind at 0.63); check whether ii_e/surp_e/MIP
  ever force a strictly smaller ceiling above n = 2; what do the optima *compute*.
- **The gradient's radius of validity** (from e07). First-order Taylor fails to
  rank full edge lesions even on the graded basic network (Spearman 0.10): lesions
  cross MIP switches the local gradient cannot see. Find the perturbation scale
  lambda* below which gradient prediction recovers (sweep lambda down, watch the
  rank correlation), and relate it to the cell size of the piecewise landscape.
- **The tie gap vs the annealing floor** (from e06). The maximin ridge (several
  cuts tied at convergence) failed to appear at n = 4: the endpoint tie gap
  (0.0085) sits at the final annealing temperature's scale (0.005). Test: continue
  optimization past the schedule with exact subgradient ascent (τ = 0) and see
  whether the gap then closes (real ridge) or widens further (no ridge).
- **Distinction loss during ascent** (from e06). The median climbing seed loses a
  distinction while big Φ doubles: which mechanisms die, and is the loss the same
  funneling that raises degeneracy? Cross-reference with the transit-state
  architecture.
- **Macro criticality follow-ups** (from e05 Part II — the level-dependence
  result: micro φ23 saturates with order, macro φ peaks near the transition).
  Larger lattices (4×4 black-boxed harder) and alternative groupings/temporal
  grains (`steps > 1`); does the macro peak track the true critical coupling as
  size grows? Also: a better finite-size crossover estimator than the d⟨m²⟩/dK
  proxy (it pinned to the boundary at n = 2 and made P6 untestable).
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

- **(done as e06)** ascent phenomenology at n=4. The mechanistic predictions from
  `docs/notes/phi-optimization.md` §3, each falsifiable: determinism–degeneracy plane
  trajectories; cut-spectrum compression (maximin theory predicts several cuts *tie*
  at convergence); active-MIP switch events as the landscape's phase transitions; the
  dissociation question (ascend φ_s only — do distinctions, relations, Φ follow?).
- **(done as e04)** descent under a fixed input–output map. The unfolding argument (Doerig et al.
  2019) as an experiment: minimize φ_s while pinning behavior; the path from a high-Φ
  system to its low-Φ functional twin is a picture of what the theory counts.
- **(absorbed by e02/e04/e05)** versions as adversaries. Ascend 4.0 φ_s while monitoring 3.0 Φ and the 2026 cap;
  systems of maximal disagreement are the definition's fault lines. 3.0's exact EMD has
  no gradient (host callback) — 3.0 can be monitored exactly but only descended via the
  Sinkhorn surrogate.
- **(done as e07)** sensitivity maps vs lesions. ∂φ_s/∂TPM as an exact integration map, compared to
  the mechanism-φ decomposition and to edge-removal deltas.
- **(done as e05)** constrained ascent / Ising. Sparsity/wiring budgets; a parameterized physical family
  (Glauber–Ising ring) to test whether ascent walks to the critical surface
  (Aguilera 2019; Popiel et al. 2020).
- **Reach vs legibility, measured.** Same ascent under node-TPM logits (CI-native)
  vs joint state-by-state logits (correlated noise allowed): does the extra reach buy
  φ_s, and is anything found there readable?
- **Expected-φ over reachable states.** Ascending φ_s at a fixed state invites the
  optimizer to exploit unreachable-state causal structure; compare fixed-state ascent
  with E_s[φ_s] over the chain's stationary distribution and hunt that exploit.
