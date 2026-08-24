# Gradient-based optimization of Φ — literature review and experiment proposal

Status: research note, 2026-08-24. Question from project lead: *Have people already used
gradient-based optimization / backpropagation to increase Φ? Does it make sense? What
would we expect to see or visualize? Why is it interesting?*

Context: iitx makes the exact IIT 4.0 pipeline (φ_s, distinctions, relations, Φ)
differentiable almost everywhere w.r.t. the TPM (design.md §8; measures.md §3.3).
This note asks whether ascending those gradients is (a) novel, (b) sensible, (c) worth
a flagship notebook.

---

## 1. Prior work on *increasing* integrated information

### 1.1 Evolutionary / selection-based (the dominant tradition)

- **Edlund, Chandler, Hintze, Koch, Adami, Tononi (2011)**, "Integrated information
  increases with fitness in the evolution of animats", *PLoS Comput Biol* 7(10):e1002236.
  Genetic algorithm on Markov-brain animats in a maze task; Φ (an earlier
  state-averaged/atomic-partition variant) rises along the line of descent as fitness
  rises. Φ is never the objective — it increases as a *byproduct* of selection for a task
  under wiring constraints.
- **Albantakis, Hintze, Koch, Adami, Tononi (2014)**, "Evolution of integrated causal
  structures in animats exposed to environments of increasing complexity", *PLoS Comput
  Biol* 10(12):e1003966. Same paradigm, IIT 3.0 concepts/Φ: number of concepts and
  integrated conceptual information grow with adaptation, more so the more sequential
  memory the environment demands, under a fixed budget of internal elements.
- **Albantakis & Tononi (2015)**, "The intrinsic cause-effect power of discrete dynamical
  systems — from elementary cellular automata to adapting animats", *Entropy*
  17(8):5472–5502. Exhaustive small-system analysis: rich causal structures (many
  higher-order concepts, high Φ) require the right balance of integration and
  segregation — *specialized* (non-degenerate) units with heterogeneous input sets beat
  homogeneous/degenerate architectures at fixed size.
- **Sporns, Tononi, Edelman (2000)**, "Theoretical neuroanatomy: relating anatomical and
  functional connectivity in graphs and cortical connection matrices", *Cereb Cortex*
  10:127–141. The pre-IIT ancestor of "optimize an integration measure": graph selection
  (mutation + selection over ~10⁴–10⁵ generations, linear-Gaussian dynamics) to maximize
  TSE neural complexity C_N. Maximizers are clustered, modular, small-world graphs —
  the canonical "integration–segregation balance" result. Derivative-free throughout.
- **Garrido-Merchán & Sánchez-Cañizares (2022)**, "Optimizing integrated information with
  a prior guided random search algorithm", arXiv:2212.04589 (also *J. Consciousness
  Studies* work by the same authors). The only paper we found whose *objective is Φ
  itself* (PyPhi, IIT 3.0, EMD, ≤ 6 binary nodes). Method: prior-guided **random search**
  over TPM space. They explicitly argue the alternatives away: "the global optimization
  problem ω\* = argmax Φ(ω) is not smooth with respect to the smallest variations ∇ω";
  Φ "must have no known analytical expression, and hence we are not able to extract
  gradients"; the Lipschitz assumption of Bayesian optimization is "violated" because a
  small TPM change can make Φ jump "from ℝ to ∅" (system loses its complex / becomes
  reducible). Findings are modest: mean Φ of feasible random TPMs grows with n
  (n=3→5), and 72–92% of sampled TPMs are "infeasible" (no complex). Future-work
  section wishes for "a smooth, continuous space of conditional TPMs" — i.e., exactly
  the logit parameterization iitx supports.
- Adjacent evolutionary work: Joshi/Sporns-lineage animat complexity studies (e.g.
  "Fitness and neural complexity of animats exposed to environmental change", *BMC
  Neurosci* 2015, PMC4699123), morphological-computation animats (arXiv:2108.00904).
  All selection-based.

### 1.2 Gradient-based optimization of *related* causal/information measures

- **Effective information (EI), gradient-ascended**: Zhang & Liu (2023), "Neural
  information squeezer for causal emergence", *Entropy* 25(1):26, and Yang, Wang, Liu,
  Rong, Yuan, Zhang (2025), "Finding emergence in data by maximizing effective
  information", *Natl Sci Rev* 12(1):nwae279 (NIS/NIS+). They train invertible neural
  networks by SGD to find coarse-grainings whose macro-dynamics **maximize Hoel's EI** —
  the closest existing practice of "backprop through a causal-power measure". EI is a
  smooth function of a parameterized macro-TPM, so no piecewise machinery was needed.
- **Marrow, Michaud, Hoel (2020)**, "Examining the causal structures of deep neural
  networks using information theory", arXiv:2010.13871 / *Network Neurosci*. Tracks EI
  and its **determinism/degeneracy decomposition ("causal plane")** of DNN layers *during
  ordinary training* — observation, not objective, but it is the direct template for the
  trajectory visualizations proposed in §4.
- **Φ measured (not optimized) during learning**: a 2025 preprint, "Bridging integrated
  information theory and the free-energy principle in living neuronal networks",
  arXiv:2510.04084, reports Φ_R rising then plateauing in cultured neurons during
  learning; similar observational studies exist for RNNs/reservoirs.
- **Mutual-information gradient estimation** (MIGE etc., e.g. Wen et al. 2021,
  *Knowledge-Based Systems*) — generic machinery, never applied to Φ.
- **What we did *not* find, despite targeted search**: any published use of
  autodiff/backprop on Φ 3.0/4.0, φ_s, Φ_G, Φ\*, Φ_R, or on Sinkhorn-relaxed IIT
  quantities; no "differentiable phi" repo on GitHub; no gradient studies of TSE
  complexity (all selection-based). Note: "Maximal Algorithmic Caliber" (suggested lead)
  is Goertzel, arXiv:2005.04589, an AGI-theory paper unrelated to Φ-optimization; the
  actual Aguilera thread is §1.3.

### 1.3 Analytic / physics results on where Φ is maximal (no optimization loop)

- **Aguilera & Di Paolo (2019)**, "Integrated information in the thermodynamic limit",
  *Neural Netw* 114:136–146 (arXiv:1806.07879), and **Aguilera (2019)**, "Scaling
  behaviour and critical phase transitions in integrated information theory", *Entropy*
  21(12):1198. Kinetic Ising models: Φ-like measures peak — and in the thermodynamic
  limit *diverge* — exactly at the critical point; an adaptive agent maintains
  integration by staying on a critical surface (Aguilera & Di Paolo, ALIFE 2018,
  arXiv:1805.00393).
- **Popiel, Khajehabdollahi, Abeyasinghe, Riganello, Nichols, Owen, Soddu (2020)**, "The
  emergence of integrated information, complexity, and 'consciousness' at criticality",
  *Entropy* 22(3):339: Φ (and Φ susceptibility) maximal/transitioning at the critical
  temperature of the 2D Ising model. Kim & Lee (*Entropy* 2019) report the same
  criticality dependence in human-brain-derived networks. Mediano et al.
  (arXiv:1606.08313) find integrated information peaks at metastability in Kuramoto
  oscillators.
- **Marshall, Grasso, Mayner, Zaeemzadeh, Barbosa, Chastain, Findlay, Sasai, Albantakis,
  Tononi (2023)**, "System integrated information", *Entropy* 25(2):334 (the φ_s iitx
  implements): φ_s **increases with determinism, decreases with degeneracy, and is
  capped by connectivity "fault lines"** (weakly coupled subsystems). This is, in
  effect, an analytic prediction of what gradient ascent should discover.
- **Zaeemzadeh & Tononi (2024)**, "Upper bounds for integrated information", *PLoS
  Comput Biol* (mechanism integrated information): characterizes which mechanisms
  achieve the φ upper bound — again "what the maximum looks like", derived by hand.
- **Haun & Tononi (2019)**, "Why does space feel the way it does?", *Entropy*
  21(12):1160 (and Grasso, Haun, Tononi 2021, "Of maps and grids", *Neurosci
  Conscious*): 2D **grid-like lattices** of specialized units yield large, structured
  Φ-structures — the IIT-canonical "high-Φ architecture" at moderate n.
- **Hoel, Albantakis, Tononi (2013)**, "Quantifying causal emergence...", *PNAS*
  110:19790: EI = determinism − degeneracy decomposition used throughout §4.

### 1.4 Novelty verdict

**Exact-Φ backpropagation appears to be new.** The only direct Φ-maximization paper
(arXiv:2212.04589) treats Φ as a non-smooth black box and uses random search, stating
outright that gradients cannot be extracted — a statement iitx falsifies by
construction (the pipeline is tensor algebra + guarded logs + min/max reductions, with
exact Danskin subgradients; design.md §8). Gradient ascent has been applied to EI
(smooth, no CES machinery) but never to φ_s or Φ with their nested argmin/argmax
structure, and never with exact gradients through a full IIT 4.0 unfolding. Honest
caveats to print next to any novelty claim: (i) the *idea* is present as future work in
Garrido-Merchán & Sánchez-Cañizares and in Mediano/Seth/Barrett's differentiability
remarks; (ii) smooth proxies (Φ_G, Φ\*, Gaussian Φ_AR) have long been differentiable
in principle — nobody bothered, which is itself informative; (iii) at publishable claim
strength this needs one more sweep of 2024–2026 preprints before submission.

---

## 2. Does gradient ascent on Φ make sense? (theory)

Φ-type objectives are **piecewise smooth**: smooth within a "selection region" (fixed
system MIP, fixed specified cause/effect states, fixed purviews/MICE, fixed complex),
with nondifferentiable seams where the active selection switches. This is the standard
setting of nonsmooth/minimax optimization, and the literature there tells us what to
expect:

- **Min over cuts (Danskin)**: gradient ascent on `min_θ f(ω, θ)` is maximin
  optimization. Away from ties, the gradient is the active cut's gradient; ascent
  *raises the weakest cut* until a second cut ties, then follows the tie ridge. KKT
  conditions at a local max put maximizers **exactly on tie seams**: generically, several
  partitions are equally irreducible at an optimum. This is a feature, not a bug — it is
  the formal version of "ascent removes fault lines" (Marshall et al. 2023) and yields a
  crisp, visualizable prediction: *the φ-per-cut spectrum compresses during ascent*
  ("no weakest link" systems). Plain subgradient ascent may chatter across a ridge;
  fixes are standard (small steps, averaging, or a softmin temperature on the cut
  reduction — measures.md §3.3 already plans softmin as an Aggregator option).
- **Max over purviews / specified states (argmax switching)**: an inner max is fine for
  ascent (Danskin again), but switching events make the objective's gradient
  discontinuous → trajectories with kinks and possible hysteresis. Log the active
  selection at every step; treat switches as data (they are the "phase transitions" of
  the ascent).
- **Ties**: measure-zero for random TPMs, but ascent *attracts* to tie seams (above), and
  symmetric parameterizations sit on them at init. iitx's even-split subgradient at exact
  ties (design.md §7) is a valid subdifferential element; still, seed asymmetrically and
  document that quantization/tie policy interacts with optimization.
- **Determinism boundary**: φ_s grows with determinism, so ascent pushes probabilities to
  {0,1}. Parameterize by logits (`p = σ(L)`): the boundary is then at infinity, gradients
  vanish as σ saturates, and ascent slows asymptotically instead of crashing into
  `log 0`. Add weight decay or an entropy floor if you want to keep trajectories in the
  interior; report `q → 0` guard behavior (design.md §8).
- **Negative φ_s under GID**: φ_s can be negative (oracle golden −0.38199…,
  oracle-findings.md) — negative means reducible, and PyPhi treats such systems as
  having no complex. For ascent this is *good*: unlike a hard `Φ = 0 if reducible` rule
  (the "ℝ to ∅" jump that killed Bayesian optimization in arXiv:2212.04589), GID gives a
  smooth, informative slope *out of* the reducible region. Plot the φ_s = 0 crossing.
- **Flat/zero-gradient regions**: exactly factorized TPMs give zero gradient through the
  cut term; random inits avoid them almost surely.
- **Local maxima**: the selection-region decomposition implies a multi-basin landscape
  (combinatorially many smooth cells). Expect strong seed dependence; that is a result
  (a Φ-landscape atlas), not a failure. Multi-seed + basin clustering is the method.
- **One tension to surface**: Tegmark (2016) notes fully state-conditioned KL measures
  vanish for deterministic systems, while IIT 4.0's intrinsic information *rewards*
  determinism (informativeness term). Ascending different measures from the same init
  should therefore diverge qualitatively — a designed-in cross-measure experiment
  (measures.md makes measure choice first-class for exactly this reason).

Conclusion: yes, it makes sense — φ_s (IIT 4.0) is the right objective because it has
no LP inside (unlike 3.0's EMD), its nonsmoothness is of the benign
enumerated-min/max kind, and its known maximizer phenomenology (determinism ↑,
degeneracy ↓, fault lines ↓) gives falsifiable expectations.

---

## 3. What to expect and visualize under gradient ascent

Expected attractor phenomenology (predictions, each mapped to a plot):

1. **Determinism ascent**: trajectories drift toward deterministic TPMs (logit norms
   grow; effectiveness ↑). *Plot*: trajectory in the **determinism–degeneracy plane**
   (Hoel's decomposition; à la Marrow-Michaud-Hoel causal plane), colored by φ_s.
2. **Specialization / anti-degeneracy**: units differentiate (distinct input sets,
   distinct logic); degenerate rows of the TPM (many states mapping to one) are
   penalized. *Plot*: TPM heatmap evolution (animation); per-unit input-weight
   divergence; degeneracy coordinate.
3. **Fault-line healing and cut-tie compression**: the φ-per-partition spectrum
   narrows; at convergence several system cuts tie (maximin equalization). *Plot*:
   line-per-cut φ curves over ascent steps ("cut spectrum"); identity of the active MIP
   as a raster (switch events = phase-transition-like kinks in the φ_s curve).
4. **Motif convergence at small n**: compare endpoints against the Albantakis–Tononi
   (2015) motif taxonomy and Haun–Tononi grids — do n=4–6 ascents find specialized
   majority/parity-like logic, cycles, or grid-like coupling? Do copy-loops appear as
   local maxima that specialization escapes? *Plot*: endpoint gallery (digraph + logic
   table + φ_s), clustered across seeds.
5. **Structure growth**: distinctions (# and Σφ_d) and relations should grow even when
   only φ_s is ascended — or *not*, which would dissociate system-level from
   structure-level integration (a genuinely open empirical question). *Plot*:
   distinction count, Σφ_d, relation count, and Φ along a φ_s-ascent; the Φ-structure
   ("unfolding") rendered at checkpoints.
6. **Criticality signature (stretch)**: for TPMs constrained to a parameterized physical
   family (e.g. Glauber dynamics of an Ising ring with couplings/temperature as the only
   parameters), ascent on φ_s should walk toward the critical surface
   (Aguilera 2019; Popiel et al. 2020). *Plot*: parameter-space trajectory over the
   phase diagram with the known critical line.
7. **Measure disagreement**: same init, same optimizer, objective ∈ {φ_s, Φ (4.0), EI,
   Φ_H, smoothed 3.0 proxy}: cross-evaluation matrix (each row an ascent, each column a
   measure evaluated along it). Mediano/Seth/Barrett (2019, *Entropy* 21:17) showed the
   measures disagree *statically*; nobody has shown they disagree *dynamically* (climb
   one, and do the others rise or fall?).
8. **Landscape geometry**: φ_s along random 2-plane slices of logit space (piecewise-
   smooth cells visible as facets); histogram of endpoint φ_s across seeds (basin
   structure); fraction of seeds ending reducible (φ_s ≤ 0) vs integrated.

---

## 4. Why this is scientifically interesting

- **It answers a question the field has only asked with random search and evolution.**
  The IIT literature's structural claims (specialization, determinism, fault lines,
  grids, criticality) were derived from hand-built examples and exhaustive small-n
  enumeration. Exact gradient ascent turns them into *dynamical* claims that can be
  confirmed or refuted mechanically, and can discover unforeseen maximizers —
  including potential *pathological* high-φ systems relevant to critiques of IIT.
- **Evolution vs gradient.** Edlund/Albantakis show Φ rises under task selection;
  Sporns-Tononi-Edelman show complexity selection yields modular small-worlds. Comparing
  gradient flows of φ_s with those evolutionary trajectories asks whether selection acts
  like a noisy gradient on integration — connecting IIT to fitness-landscape and
  artificial-life questions.
- **Criticality.** If ascent within physical model families converges to critical
  surfaces, that links two literatures (IIT-maximization and
  criticality-of-consciousness) with a mechanism rather than a correlation.
- **ML relevance.** A differentiable φ_s is a prototype integration regularizer
  (cf. NIS+ maximizing EI); even a negative result (φ_s regularization does nothing for
  task performance) is publishable, given ongoing speculation about Φ in AI systems.
- **What iitx uniquely enables**: exact, oracle-verified IIT 4.0 values *and* exact
  a.e.-gradients from one code path — exact Φ backprop at small n where the claims are
  checkable against PyPhi, plus the measure registry to run the same ascent under proxy
  measures at larger n. No other tool (PyPhi, random-search pipelines, Gaussian-proxy
  code) offers this combination.

---

## 5. Proposed flagship experiment / notebook: "Ascending Φ"

**Setup.** `L ∈ ℝ^{2ⁿ×n}` logits; `tpm = σ(L)` (conditionally independent units —
iitx's native `System`). n = 4 (headline; full Φ-structure affordable), with n = 3 and
n = 5–6 (φ_s only) as appendices. Objective (choose per section):
`φ_s(system, s)` at a fixed state; `E_s[φ_s]` over reachable states (avoids optimizing
into an unreachable state's causal structure); and one section ascending big Φ at n = 3.
Optimizer: Adam (lr ~1e-2), 50–200 seeds, optional softmin temperature β on the cut
min annealed β: 10 → ∞, with the exact (hard-min) φ_s always reported.

**Core figures** (numbered as in §3): (1) φ_s vs step, seed fan + median; (2)
determinism–degeneracy plane trajectories; (3) cut-spectrum compression + active-MIP
raster with switch events marked on the φ_s curve; (4) endpoint motif gallery vs
Albantakis-Tononi 2015 / copy-loop / grid references; (5) distinctions, relations, Σφ_d,
Φ along a φ_s-only ascent; (7) cross-measure ascent matrix; (8) endpoint-φ_s histogram
and a 2-plane landscape slice. Sanity panels: finite-difference gradient check away from
ties (already a test-suite item, design.md §8); a tie-crossing zoom showing the
even-split subgradient behavior; φ_s = 0 crossing from a reducible init (the GID
"negative φ_s is a slope, not a wall" point, contrasted explicitly with
arXiv:2212.04589's infeasibility cliff).

**Pitfall handling.** Logit weight decay 1e-3 (or entropy floor) to delay saturation;
report both raw and softmin runs; document tie policy interaction; seed with
i.i.d. N(0, 1) logits (asymmetric); never quantize inside the optimization loop.

**Headline claims the notebook can support if predictions hold**: "Exact backprop
through IIT 4.0's φ_s (first demonstration); gradient ascent spontaneously produces
deterministic, specialized, fault-line-free systems whose minimum cuts tie — and
[does / does not] drag big Φ, EI, and Φ_H upward with it."

---

## 6. Bibliography (key items)

- Edlund JA, Chandler N, Hintze A, Koch C, Adami C, Tononi G (2011). *PLoS Comput Biol*
  7(10):e1002236. doi:10.1371/journal.pcbi.1002236
- Albantakis L, Hintze A, Koch C, Adami C, Tononi G (2014). *PLoS Comput Biol*
  10(12):e1003966. doi:10.1371/journal.pcbi.1003966
- Albantakis L, Tononi G (2015). *Entropy* 17(8):5472–5502. doi:10.3390/e17085472
- Sporns O, Tononi G, Edelman GM (2000). *Cereb Cortex* 10(2):127–141.
- Garrido-Merchán EC, Sánchez-Cañizares J (2022). arXiv:2212.04589.
  Code: github.com/EduardoGarrido90/iit_opt
- Zhang J, Liu K (2023). *Entropy* 25(1):26. doi:10.3390/e25010026
- Yang M, Wang Z, Liu K, Rong Y, Yuan B, Zhang J (2025). *Natl Sci Rev* 12(1):nwae279.
  (arXiv:2308.09952)
- Marrow S, Michaud EJ, Hoel E (2020). arXiv:2010.13871; *Network Neurosci* 4(4).
- Hoel EP, Albantakis L, Tononi G (2013). *PNAS* 110(49):19790–19795.
- Marshall W, Grasso M, Mayner WGP, Zaeemzadeh A, Barbosa LS, Chastain E, Findlay G,
  Sasai S, Albantakis L, Tononi G (2023). *Entropy* 25(2):334. doi:10.3390/e25020334
  (arXiv:2212.14537)
- Zaeemzadeh A, Tononi G (2024). Upper bounds for integrated information.
  *PLoS Comput Biol.*
- Haun A, Tononi G (2019). *Entropy* 21(12):1160. doi:10.3390/e21121160;
  Grasso M, Haun A, Tononi G (2021). *Neurosci Conscious* 2021(2):niab022.
- Aguilera M, Di Paolo EA (2019). *Neural Netw* 114:136–146 (arXiv:1806.07879);
  Aguilera M (2019). *Entropy* 21(12):1198. doi:10.3390/e21121198;
  Aguilera M, Di Paolo EA (2018). ALIFE 2018 (arXiv:1805.00393).
- Popiel NJM, Khajehabdollahi S, Abeyasinghe PM, Riganello F, Nichols ES, Owen AM,
  Soddu A (2020). *Entropy* 22(3):339. doi:10.3390/e22030339
- Mediano PAM, Seth AK, Barrett AB (2019). *Entropy* 21(1):17. doi:10.3390/e21010017
- Mediano PAM et al. (2016). Integrated information and metastability in systems of
  coupled oscillators. arXiv:1606.08313
- Bridging IIT and the free-energy principle in living neuronal networks (2025).
  arXiv:2510.04084
- Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G (2018). PyPhi.
  *PLoS Comput Biol* 14(7):e1006343.
- Albantakis L et al. (2023). IIT 4.0. *PLoS Comput Biol* 19(10):e1011465.
- Goertzel B (2020). Maximal algorithmic caliber. arXiv:2005.04589 — *checked and
  found unrelated* to Φ-optimization (noted to close a lead from the research brief).
