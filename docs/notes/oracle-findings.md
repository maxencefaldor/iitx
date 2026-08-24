# Oracle findings

Facts about the reference implementation that differ from, or are absent from, the
published theory — discovered while implementing against the oracle's source and golden
fixtures. Each is a convention `iitx` reproduces deliberately; none is guessable from the
papers. (PyPhi refs: `feature/iit-4.0` @ `b78d0e34`, `main` @ `ce2b2832`.)

## 1. IIT 4.0 φ clipping: the two oracles disagree, and main follows the paper

Eqs. 19-20 and 41/44 of Albantakis et al. (2023) write the integrated-information
log-ratio inside a positive part, `|·|₊`. The `feature/iit-4.0` branch (the 2023
paper's code) does **not** clip — its published golden for
`basic_noisy_selfloop_network` is φ_s = −0.38198987262266504 — while PyPhi `main`
(2.0) restores the clamp (`SystemIrreducibilityAnalysis.__post_init__` applies
`positive_part`, keeping the raw value as `signed_phi`; its fixture for the same
network is φ_s = 0.0). Diagnosed from the mathematics: the paper's notation is
explicit, so `main` is right and the branch diverged. `iitx` follows the paper and
`main`: `SystemPhi.phi` is clamped, `SystemPhi.signed_phi` keeps the raw value
(negative signed φ still means reducible, and still carries gradient signal).

## 2. Reducibility short-circuits the system MIP search

`new_big_phi.sia` maps `evaluate_partition` over the partitions with
`shortcircuit_func=utils.is_falsy`: the first partition (in enumeration order, under the
default serial evaluation) whose φ ≤ 0 at precision **stops the search and is reported as
the minimum partition**, with its possibly-negative φ. The normalized-φ minimization key
`(normalized φ, −φ)` only governs the choice among partitions when every partition has
φ > 0. Getting this wrong changes both the reported φ_s and the certificate for every
reducible system (the fixture above selects cut #0, φ = −0.382, over the
normalized-minimal cut, φ = −0.372).

## 3. System-partition normalization: `cut_matrix.sum()` equals the paper's formula

The implementation (`GeneralKCut.normalization_factor`) divides by `cut_matrix.sum()`,
the number of severed connections. An earlier version of this note claimed this
diverges from the paper's Σᵢ |S⁽ⁱ⁾||X⁽ⁱ⁾| when blocks' cut sets overlap — **that claim
was wrong** (corrected by the review in `pyphi-review.md`, finding R14): Eq. 16 defines
X⁽ⁱ⁾ as the units whose inputs *to part i* are severed, so each severed edge is counted
exactly once, by its destination part, and the two quantities are equal for every
SET_UNI/BI partition (verified exhaustively for n ≤ 4). The genuine deviations are only
the complete cut (normalized by `n`, an out-of-Θ special case) and the GENERAL scheme.
iitx's `system_cuts` severed-edge count is correct under either reading.

## 4. Two different "unconstrained effect" conventions across theory versions

IIT 3.0's unconstrained effect repertoire is the *product* over purview units of
independently perturbed marginals (virtual elements). IIT 4.0's unconstrained effect
probability (Eq. 6) is the uniform average over interventions of the **joint**
conditional — a correlated mixture, not a product. Both are correct in their own
measure; conflating them silently shifts every ii value in 4.0 (or every EMD in 3.0).

## 5. Specified-state ties: resolved by maximal φ_s, then PyPhi's state order

Ties in the intrinsic information of candidate system states are not broken
arbitrarily: per the S1 Text cascade (and PyPhi `main`'s behaviour, exposed by the
`rule110` fixture, where the cause side ties at ii = 1.0 between (0,0,0) and (1,1,1)
and only the latter yields φ_s = 2.0), the tied pair with **maximal φ_s** wins.
Residual ties follow PyPhi's state-iteration order — `itertools.product`, the *last*
unit varying fastest — which differs from iitx's canonical little-endian order, so the
tie rank table `_oracle_rank` maps between them. Mechanism-level state ties follow the
same cascade with congruence preference.

## 6. PyPhi's macro Φ cuts at the micro level

`MacroSubsystem.cut_indices` returns *micro* indices: when PyPhi computes Φ of a
coarse-grained or black-boxed system (the `emergence` golden numbers, e.g. macro_network
micro 0.113889 → macro 0.597212), the system cuts sever micro connections and the
transform is re-applied to each cut micro TPM. Analyzing the transformed TPM as an
ordinary system — cutting macro connections — gives a different number (0.86905 for the
same coarse-graining, which is itself the golden Φ of PyPhi's `macro` fixture network:
the transforms agree exactly; the *analysis modes* differ). iitx 0.1 ships the
transforms and the ordinary-system analysis; micro-cut macro analysis belongs to the
deferred emergence-search design, where this note becomes its specification.
