# Oracle findings

Facts about the reference implementation that differ from, or are absent from, the
published theory — discovered while implementing against the oracle's source and golden
fixtures. Each is a convention `iitx` reproduces deliberately; none is guessable from the
papers. (PyPhi refs: `feature/iit-4.0` @ `b78d0e34`, `main` @ `ce2b2832`.)

## 1. IIT 4.0 φ has no positive-part clipping

Eqs. 19-20 and 41/44 of Albantakis et al. (2023) write the integrated-information
log-ratio inside a positive part, `|·|₊`. The reference implementation
(`metrics/distribution.py::generalized_intrinsic_difference`) does **not** clip: φ is
selectivity × raw log-ratio, and the published golden fixture for
`basic_noisy_selfloop_network` has φ_s = −0.38198987262266504. Negative φ_s means
reducible (`SystemIrreducibilityAnalysis.__bool__` is `phi > 0` at precision). `iitx`
follows the oracle and documents the divergence from the paper's notation.

## 2. Reducibility short-circuits the system MIP search

`new_big_phi.sia` maps `evaluate_partition` over the partitions with
`shortcircuit_func=utils.is_falsy`: the first partition (in enumeration order, under the
default serial evaluation) whose φ ≤ 0 at precision **stops the search and is reported as
the minimum partition**, with its possibly-negative φ. The normalized-φ minimization key
`(normalized φ, −φ)` only governs the choice among partitions when every partition has
φ > 0. Getting this wrong changes both the reported φ_s and the certificate for every
reducible system (the fixture above selects cut #0, φ = −0.382, over the
normalized-minimal cut, φ = −0.372).

## 3. System-partition normalization counts distinct severed connections

The paper (Eq. 23) writes the normalization as Σᵢ |S⁽ⁱ⁾||X⁽ⁱ⁾|. The implementation
(`GeneralKCut.normalization_factor`) divides by `cut_matrix.sum()` — the number of
**distinct** severed connections, i.e. the union of the blocks' cuts. The two differ
whenever blocks' cut sets overlap (e.g. a bipartition with flags (↔, ↔) severs each
cross edge once, not twice). The complete cut normalizes by `n`, not `n²`.

## 4. Two different "unconstrained effect" conventions across theory versions

IIT 3.0's unconstrained effect repertoire is the *product* over purview units of
independently perturbed marginals (virtual elements). IIT 4.0's unconstrained effect
probability (Eq. 6) is the uniform average over interventions of the **joint**
conditional — a correlated mixture, not a product. Both are correct in their own
measure; conflating them silently shifts every ii value in 4.0 (or every EMD in 3.0).

## 5. State enumeration order differs between PyPhi and iitx

PyPhi iterates candidate states via `itertools.product`, varying the **last** unit
fastest; `iitx` enumerates in little-endian order, varying unit 0 fastest. Values are
unaffected; the choice among exactly-tied specified states can differ. No golden fixture
exercises this so far; if one does, the divergence is documented here rather than
contorting the canonical order.

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
