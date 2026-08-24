# Survey of integrated-information measures — signatures for the `iitx` measure interface

Status: design notes, 2026-08-24. Sources: Oizumi/Tsuchiya/Amari PLoS CB 2016 (Φ\*);
Oizumi/Amari et al. PNAS 2016 (Φ_G, unified geometric framework); Tegmark PLoS CB 2016
(taxonomy of 420 measures); Mediano/Seth/Barrett Entropy 2019 (candidate comparison);
Mediano/Rosas et al. arXiv:1909.02297 (ΦID, Φ_R); Albantakis et al. PLoS CB 2023 (IIT 4.0);
Oizumi/Albantakis/Tononi PLoS CB 2014 (IIT 3.0).

Notation: system `S` with `n` units; dynamics given either as a transition probability
matrix/function `T = p(x_t | x_{t-1})` or as a stationary joint `p(X, Y)` with
`X = X_{t-τ}` (past) and `Y = X_t` (present/future); `π` a partition of `S` into parts
`{M^1..M^K}`; `s` a current state.

---

## 1. Per-measure mathematical signatures

### 1.1 Φ^3.0 (Oizumi–Albantakis–Tononi 2014)

- **Inputs**: full TPM (with conditional independence of units given the past) **and a
  current state `s`**. Strictly state-dependent; no stationary distribution needed
  (perturbational: uses maximum-entropy interventional priors, not observed statistics).
- **Objects built**: for every mechanism `M ⊆ S` and every purview `Z ⊆ S`, a *cause
  repertoire* `p_c(Z | M = s_M)` (Bayes inversion over maxent prior) and an *effect
  repertoire* `p_e(Z | M = s_M)`. Unconstrained units are noised (virtualized).
- **What is partitioned**: *both* levels. (a) Mechanism partitions: each mechanism/purview
  pair is cut over factorizations `{(M_1,Z_1),(M_2,Z_2)}`; small `φ^MIP` = EMD between whole
  and partitioned repertoire, minimized over mechanism partitions, then **maximized over
  purviews** (`φ^max`, "core cause/effect"). (b) System level: unidirectional cuts of `S`
  (connections from one part to the rest are noised).
- **What is minimized over**: system cuts; Φ = min over cuts of **XEMD** (extended
  earth-mover's distance) between the whole *cause–effect structure* (constellation of all
  concepts, each a point in 2·2^n-dim "qualia space" weighted by φ^max) and the cut
  constellation.
- **Distance**: EMD (Wasserstein-1) at the repertoire level; XEMD at the constellation
  level (transports φ mass between concepts, cost = concept-to-concept EMD).
- **Needs the full cause–effect structure**: yes — Φ is a distance between *sets of
  concepts*, not between two system-level distributions. This is the defining structural
  feature.
- **Continuous vs discrete**: discrete-state only in practice (EMD over discrete
  repertoires; maxent perturbation ill-defined for continuous without extra choices).
- **Differentiability**: EMD is an LP → piecewise-linear in repertoire entries; nested
  argmax (purviews) / argmin (mechanism partitions, cuts) → piecewise-smooth with dense
  nondifferentiable seams; outer structure combinatorial (super-exponential in `n`).
  Smooth in TPM entries only within a fixed selection region.

### 1.2 Φ^4.0 / φ_s (Albantakis et al. 2023)

- **Inputs**: TPM + current state `s` (same perturbational setting as 3.0).
- **System level (φ_s)**: intrinsic information `ii = π(s'|s) · log[π(s'|s)/π_cut(s'|s)]`
  evaluated at the *maximally selective* cause/effect state `s'` (informativeness ×
  selectivity), for both cause and effect sides; φ_s = min over **directional system
  partitions** θ (each part tagged ←, →, or ↔ for which connections are severed) of the
  (normalized) intrinsic difference; overall min of cause and effect sides.
- **Distance**: **intrinsic difference** ID(p, q) = max_s' p(s') log(p(s')/q(s')) — a
  pointwise, state-selective divergence (not a sum over states). Replaces EMD everywhere.
- **Mechanism level**: distinctions with φ_d (min over mechanism partitions, max over
  purviews via congruence with system cause–effect states), plus *relations* among
  distinctions; **Φ (big Phi) = Σ φ_d over distinctions + relations** in the Φ-structure.
- **Needs full CES**: φ_s alone is system-level (cheap-ish); Φ needs the full structure of
  distinctions and relations (doubly exponential).
- **Continuous/discrete**: discrete. **Differentiability**: ID is smooth in probabilities
  *except* the argmax over the selected state and the argmin over partitions →
  piecewise-smooth; better behaved than EMD (no LP inside), still combinatorial outside.

### 1.3 Φ_MI (mutual-information-based; Tononi–Sporns lineage)

- **Signature**: Φ_MI(π) = Σ_k I(M^k_X ; M^k_Y complement structure) — in the
  unified/geometric framing: KL projection of `p(X,Y)` onto the manifold where the parts
  are *fully independent* (both same-time and across-time dependencies between parts
  removed). Equivalently total correlation-type quantity between parts.
- **Inputs**: stationary joint `p(X, Y)` (or just `p(X)` in the original 1994 static
  version). State-averaged.
- **Partition/minimization**: system bipartitions (or K-partitions), min over π with
  normalization. **Distance**: KL. **Continuous**: yes, closed form for Gaussians.
- **Differentiability**: smooth (entropies of covariance blocks) inside a partition
  choice; min over partitions → piecewise-smooth. Overestimates integration (also counts
  same-time correlation), upper-bounds Φ_G.

### 1.4 Φ_H / stochastic interaction (Ay; Barrett–Seth Φ_H)

- **Signature**: SI(π) = Σ_k H(M^k_Y | M^k_X) − H(Y | X); geometrically the KL projection
  onto the manifold where the *transition* factorizes: q(y|x) = Π_k q(y_k | x_k) **and**
  the parts' pasts are treated as independent.
- **Inputs**: stationary `p(X,Y)`. State-averaged. Nonnegative but violates the upper
  bound Φ ≤ I(X;Y) (can be > 0 for totally disconnected parts with correlated noise/inputs).
- Smooth per partition; Gaussian closed form; min over partitions piecewise.

### 1.5 Φ_E / Φ_WMS (Barrett–Seth 2011 "empirical" whole-minus-sum)

- **Signature**: Φ_E(π) = I(X;Y) − Σ_k I(M^k_X ; M^k_Y), min over (normalized)
  bipartitions. "Φ_E" = computed from the **empirical stationary distribution** (vs
  Balduzzi–Tononi's maxent-perturbed version); Φ_AR = Gaussian/autoregressive estimator
  from time-lagged covariances.
- **Inputs**: stationary joint / lagged covariance `Σ(X), Σ(X,Y)`. State-averaged.
- **Distance**: difference of mutual informations (KL-based, but *not* a divergence
  between p and any q — a signed difference). **Can be negative** (redundancy-dominated
  systems) — the defect ΦID repairs.
- Smooth per partition; trivial Gaussian closed form; continuous-friendly.

### 1.6 Φ\* (Oizumi–Tsuchiya–Amari 2016, decoding perspective)

- **Signature**: Φ\*(π) = I(X;Y) − I\*(X;Y), where I\* = max_β of the mismatched-decoding
  information using the partitioned model q(y|x) = Π_k q(y_k|x_k) as a *decoder* of the
  true joint. Quantifies information lost when decoding as if parts were independent.
- **Inputs**: stationary joint `p(X_{t-τ}, X_t)` — in practice empirical time-lagged
  covariances; stationarity assumed. **State-averaged**, τ-dependent.
- **Partition/minimization**: min over partitions (atomic partition used in practice as
  upper bound; MIP search exponential; continuous-variable normalization problematic
  because differential entropy can be negative).
- **Bounds**: 0 ≤ Φ\* ≤ I(X;Y) — satisfies both (unlike Φ_E lower, Φ_H upper).
- **Continuous**: yes; Gaussian case has near-closed form (β from a quadratic equation).
- **Differentiability**: the β-optimization is an inner smooth max (envelope theorem →
  differentiable); smooth per partition; min over partitions piecewise. **Structural
  quirk**: the "distance" is not `D(p, q)` — it is a decoding-gap functional
  `I(p) − I*(p, q, β*)` with an inner optimization; q enters as a decoder, not a comparand.

### 1.7 Φ_G (Oizumi–Amari et al. 2016, information geometry)

- **Signature**: Φ_G(π) = min_{q ∈ M_D} KL(p(X,Y) ‖ q(X,Y)) where the *disconnected
  manifold* M_D = {q : q(y_i | x) = q(y_i | x_i)} — only causal cross-influences are
  severed; same-time correlations and within-part dynamics are left free. Unifies the
  family: Φ_MI, transfer entropy, stochastic interaction are the same KL projection with
  progressively stronger manifold constraints; hierarchy gives Φ_G ≤ Φ_SI, Φ_G ≤ Φ_MI.
- **Inputs**: stationary joint `p(X,Y)`. State-averaged.
- **Partition/minimization**: min over partitions of an **inner constrained
  minimization** (e-projection). **No closed form even for Gaussians**: the disconnected
  model constrains the regression matrix to block structure but leaves noise covariance
  free → iterative/numerical projection required.
- **Continuous**: yes (Gaussian via numerical optimization). **Differentiability**: the
  inner projection is a smooth convex-ish program → value differentiable in p by envelope
  arguments (implicit-function differentiation through the KKT system, or unrolled
  iterations); min over partitions piecewise. **Structural quirk**: the cut does not
  produce a distribution to compare — it produces a *constraint set*; distance and
  partition fuse into one variational problem.

### 1.8 Tegmark's taxonomy (2016) and endorsed variants

- **Taxonomy = product of independent axes** (5 × 4 × 3 × 7 = 420): (A) how the
  factorized/cut model is built — noising with uniform noise `n`, marginal noising `m`,
  optimal factorized approximation `o`, state-dependent optimal `x`, state-averaged
  optimal `a`; (B) which distributions are compared — two-time joint `t`, future marginal
  `f`, one subsystem `a`, time-reversed/past `p`; (C) conditioning on the current state —
  unknown/ensemble `u`, separable `s`, known `k`; (D) distance — KL `k`, L1, L2, Hilbert,
  Jensen–Shannon, EMD, mismatched decoding `m`. This product structure is the strongest
  existing evidence that a *factorized* plug-in interface is the right design.
- **Endorsed**: φ_M = φ_otu = I(X_A; X_B | X_0) ("Markov measure": KL to best separable
  approximation of the two-time joint); φ_2.5 = min{φ_nak, φ_npk} (vanishes for both
  afferent and efferent pathways); φ_om (defined for continuous variables); φ_MD
  (mismatched decoding ≈ Φ\*); notes φ_3.0 as expensive; φ_B = Barrett–Seth (can be
  negative). Warns that fully state-conditioned (`k`) measures **vanish for deterministic
  systems** (future certain given state).
- **Inputs**: Markov TPM *or* two-time joint; Gaussian case reduced to O(n³) covariance
  formulas (e.g., φ_M = ½ log(|C_A||C_B|/|C_AB|)); proposes graph-theoretic approximation
  for the partition search.
- **Differentiability**: KL/L2 axes smooth; L1/EMD axes piecewise-linear; all get
  piecewise structure from the min over cuts; `x`/`a` options add inner optimizations.

### 1.9 ΦID-based Φ_R (Mediano, Rosas et al. 2019/2021)

- **Framework**: Integrated Information Decomposition — a **double redundancy lattice**
  with atoms indexed by pairs (α → β) of antichains over past parts × future parts (16
  atoms for 2 variables). "Integration" is revealed as an aggregate of distinct
  phenomena (transfer, synergy, redundancy conversion, ...). Individual atoms/combinations
  serve as *tailored* measures.
- **Φ_R** := Φ_WMS + double-redundancy correction (Φ_WMS double-counts redundancy, which
  is why it goes negative; adding back the redundancy–redundancy atom yields Φ_R ≥ 0).
- **Inputs**: joint `p(X_past, X_future)` over the parts **plus a user-chosen
  double-redundancy function** (e.g., MMI: min over part pairs of I(x_i; y_j)). Degrees of
  freedom: redundancy function, τ, partition.
- **Continuous**: yes with Gaussian redundancy functions (MMI has closed form).
- **Differentiability**: Φ_WMS part smooth; MMI redundancy introduces a min over pairwise
  MIs → piecewise-smooth even before the partition min. **Structural quirk**: the measure
  is parameterized by *another pluggable functional* (the redundancy function) — an
  interface within the interface.

### 1.10 Neighbors worth one line each

- **Φ^2.0** (Balduzzi–Tononi 2008): state-dependent effective information
  ei(X→X^π; s) with maxent perturbation, KL distance, min over normalized partitions —
  the (TPM, state) ancestor of 3.0/4.0 without the CES machinery.
- **Causal density (CD)**: mean pairwise conditional Granger causality / transfer
  entropy; stationary time series; no partition-minimization at all (aggregation is a
  mean over ordered pairs).
- **ψ (PID synergy)**: synergistic predictive information about the future whole beyond
  parts; needs a PID redundancy choice; Mediano et al. 2019 (Entropy) find no two of
  {Φ_WMS, Φ*, Φ_G, Φ_CIS/SI, CD, ψ} agree consistently across analyses — measure choice
  is empirically load-bearing, hence first-class in `iitx`.

---

## 2. Comparison table

| Measure | Inputs | State-dep? | What is partitioned | Minimized over | Distance/divergence | Needs full CES? | Continuous? | Differentiability |
|---|---|---|---|---|---|---|---|---|
| Φ^2.0 | TPM + state (maxent perturb.) | yes | system (normalized parts) | system partitions | KL | no | no | smooth per cut; piecewise (min); combinatorial |
| Φ^3.0 | TPM + state (maxent perturb.) | yes | mechanisms×purviews AND system | mech. partitions (min), purviews (max), system cuts (min) | EMD / XEMD over constellations | **yes** | no | piecewise-linear (LP) + nested argmin/argmax; combinatorial |
| φ_s (4.0) | TPM + state | yes | system, directional partitions θ | directional partitions (normalized min); cause vs effect min | Intrinsic Difference (pointwise max_s' p log p/q) | no (φ_s); Φ = Σφ_d needs CES | no | smooth except state-argmax + partition argmin; piecewise-smooth |
| Φ (4.0 big) | TPM + state | yes | mechanisms, purviews, relations | per-distinction MIPs | ID | **yes** | no | as φ_s, doubly exponential outer sum |
| Φ_MI | stationary p(X,Y) | no | system | partitions (normalized) | KL (projection: full independence) | no | yes (Gaussian closed form) | smooth per cut; piecewise (min) |
| Φ_H / SI | stationary p(X,Y) | no | system (transition factorized) | partitions | KL (projection: factorized transition) | no | yes (closed form) | smooth per cut; piecewise |
| Φ_E / Φ_WMS | stationary p(X,Y) / lagged cov. | no | system | bipartitions (normalized) | signed MI difference (can be < 0) | no | yes (closed form) | smooth per cut; piecewise |
| Φ\* | stationary p(X,Y) / lagged cov. | no | system (q as decoder) | partitions; inner max over β | decoding gap I − I\*(β) | no | yes (β analytic for Gaussian) | inner smooth max; piecewise over partitions |
| Φ_G | stationary p(X,Y) | no | system (manifold constraint q(y_i\|x)=q(y_i\|x_i)) | partitions; **inner KL projection** | KL to e-projection | no | yes (numerical only) | envelope-smooth inner opt; piecewise; no closed form |
| Tegmark φ_M etc. | TPM or two-time joint (+ state for `k`) | axis choice | system (bipartitions) | cuts ("cruelest cut"); some axes add inner opt | axis choice: KL/L1/L2/JS/EMD/MD | no | yes (O(n³) Gaussian formulas) | depends on distance axis; always piecewise from min |
| Φ_R (ΦID) | joint p(parts_past, parts_future) + redundancy fn | no | system parts (lattice over all subsets) | partition min (opt.); min inside MMI | Φ_WMS + double-redundancy atom | no (but full lattice for ΦID atoms) | yes (MMI Gaussian closed form) | piecewise-smooth (mins in redundancy) |
| CD | stationary time series | no | ordered pairs (no cut-min) | — (mean, not min) | conditional TE | no | yes | smooth |

---

## 3. Design conclusion: the plug-in interface

### 3.1 The natural factorization

Tegmark's 5×4×3×7 product plus the geometric unification (all classical measures =
divergence from a *disconnected model class*) suggest a five-stage pipeline:

```
SystemModel → RepertoireKernel → PartitionScheme → Comparator → Aggregator
```

1. **SystemModel** (input adapter). Canonical core object: `(TPM, state)`. Adapters
   produce whichever of these a measure consumes: (a) conditional repertoire generator
   `p(purview | mechanism-state)` with a *conditioning convention* (maxent perturbation,
   stationary marginals, marginal noising — Tegmark axes A/C); (b) a two-time joint
   `p(X, Y)` (requires a state prior — see §4); (c) Gaussian parameterization `(A, Σ_ε)`
   or lagged covariances.
2. **RepertoireKernel**: which distributions get built and compared (cause side, effect
   side, two-time joint, future marginal — Tegmark axis B; cause/effect repertoires for
   IIT proper). Owns virtualization/noising of background units.
3. **PartitionScheme**: the lattice of cuts *and the semantics of a cut* — i.e., the map
   `cut ↦ disconnected model`. Variants: factorized transition (SI), noised
   cross-connections (3.0 unidirectional cuts), directional θ-partitions (4.0), decoder
   q (Φ\*), **constraint manifold** (Φ_G), mechanism-level factorization (3.0/4.0 inner
   loop). Includes normalization policy and the search strategy (exhaustive, Queyranne,
   spectral, learned).
4. **Comparator**: `D(whole, cut) → ℝ`. KL, EMD, ID, L1/L2/JS, decoding gap (with inner
   β-max), redundancy-corrected difference (with pluggable redundancy function).
5. **Aggregator**: min over cuts (MIP), max over purviews, sum over
   distinctions+relations, mean over pairs (CD), expectation over states, min over
   cause/effect sides.

**Independently varying**: Comparator × PartitionScheme × conditioning convention ×
Aggregator are nearly orthogonal (Tegmark's taxonomy is literally this product; swapping
EMD→ID inside IIT, or KL→L1 inside Φ_M, is well defined). This is the interface's core
claim and it holds for ~all system-level measures.

### 3.2 Measures that break the factorization

- **Φ_G and Φ\***: the cut does *not* yield a distribution to hand to a comparator. For
  Φ_G the cut defines a constraint manifold and the "distance" is an inner argmin
  (projection); for Φ\* the cut model is a *decoder* inside a gap functional with an
  inner max over β. Fix: generalize Comparator's second argument from "distribution" to
  "model class / functional", i.e. `Comparator: (p, CutObject) → ℝ` where `CutObject` may
  be a distribution, a manifold, or a decoder family. The pipeline order survives; the
  type of the intermediate does not.
- **Φ^3.0 / Φ^4.0-big-Φ**: the comparand is a *cause–effect structure* (set of
  φ-weighted concepts/distinctions + relations), not a distribution — the whole
  mechanism-level pipeline (kernel→partition→comparator→aggregator) is *nested inside*
  the system-level comparator. Fix: make the pipeline recursive — a Measure can be a
  Comparator for an outer Measure. This is the single biggest architectural demand.
- **State-averaged vs state-dependent**: two different signatures,
  `μ(model, state) → ℝ` vs `μ(model) → ℝ`. Unify by making every measure state-dependent
  in principle and adding a `StateWeighting` aggregator (δ at the observed state ↔ pure
  state-dependent; stationary/maxent expectation ↔ state-averaged). Caveat (Tegmark):
  fully state-conditioned KL-type measures vanish for deterministic systems — the
  interface should surface, not hide, this choice.
- **Time-series measures**: Φ\*, Φ_G, Φ_E, Φ_R as *practiced* start from empirical lagged
  covariances with a stationarity assumption — the "model" is an estimated object, so
  estimation (plus τ selection) is a front-end stage, not part of the measure.
- **Φ_R**: imports a second pluggable functional (redundancy). Treat redundancy functions
  as their own registry that Φ_R's Comparator is parameterized by.

### 3.3 Differentiability note for JAX

Every measure with a partition-min is at best piecewise-smooth; `argmin` selections give
valid subgradients (danger only exactly at ties), so `min` over an enumerated cut set is
JAX-differentiable in practice. EMD (3.0) is itself an LP → piecewise-linear; use
Sinkhorn-smoothed EMD as a differentiable surrogate. ID (4.0) and decoding-gap (Φ\*) have
inner argmax/max: envelope theorem gives clean gradients. Φ_G needs
implicit-function/unrolled differentiation through the projection. Softmin temperature on
the partition search gives a fully smooth relaxation of any measure in the family —
worth exposing as an Aggregator option.

---

## 4. Scope split for `iitx` core

**Computable from (TPM, state) alone — core scope:**
Φ^2.0, Φ^3.0, φ_s and Φ^4.0, Tegmark's `k`-conditioned and maxent variants (φ_2.5,
φ_oak/φ_opk), and any state-averaged measure *if* the user supplies a state prior
(uniform/maxent), since `p(X,Y) = prior(X) · T(Y|X)`. The perturbational (maxent) prior is
the IIT-native convention and keeps everything a pure function of the TPM.

**Needing extra inputs — adapters, not core:**
- *Stationary distribution*: Φ_MI, Φ_H/SI, Φ_E/WMS, Φ\*, Φ_G, Φ_R, Tegmark `u`-variants.
  Extra input: stationary `p(X)` (or solve the TPM's fixed point — existence/uniqueness
  caveats for reducible/periodic chains must surface in the API).
- *Empirical time series / covariances*: practical Φ\*, Φ_G, Φ_AR, Φ_R, CD, ψ on data.
  Extra inputs: lagged covariance `(Σ_XX, Σ_XY, Σ_YY)` or raw series + τ + estimator;
  Gaussian model class `(A, Σ_ε)` as a first-class SystemModel.
- *Auxiliary functional choices*: Φ_R (redundancy function), Φ\* (β search domain),
  Tegmark (distance axis), all (normalization + partition search strategy).

The unifying deliverable: a `Measure` protocol taking a `SystemModel` (TPM+state | joint |
Gaussian) and a config of pluggable `(kernel, partition_scheme, comparator, aggregator)`
components, with the recursive escape hatch for constellation-based Φ and generalized
comparators for projection/decoding-type measures.
