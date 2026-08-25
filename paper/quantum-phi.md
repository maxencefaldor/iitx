# A Quantum Integrated Information Algorithm

*Draft — Maxence Faldor. Third paper of the iitx program (companions:
`phi-landscape.md`, `behavioral-requirement.md`). Empirical companion notebook:
`experiments/e12_quantum_phi.ipynb`. Written to be readable by two audiences at
once: IIT researchers who have never touched a qubit, and quantum-algorithms
people who have never computed a φ. Each side's prerequisites are one section
long.*

## Abstract

Computing integrated information is famously expensive — "super-exponential" in
folklore. We first make the folklore precise: for a system of n binary units with
an explicit transition probability matrix (TPM, size N = n·2ⁿ), the system-level
quantity φ_s costs Θ(K_n·4ⁿ) where K_n, the number of directional system
partitions, empirically tracks the Fubini numbers (measured K_2..8 = 3, 22, 150,
1061, 7896, 61888, 510313; ratio to Fubini(n) within [0.93, 2.0] over the range) — so φ_s is
**2^Θ(n log n): quasi-polynomial in the TPM size**, neither polynomial nor
exponential, while the full Φ-structure adds a second 2^Θ(n log n) axis (mechanism
partitions) and the relations axis, naively doubly exponential, is already
collapsed analytically. We then construct a **Quantum Integrated Information
Algorithm (QIIA)**: after O(n·3ⁿ) classical preprocessing (an Oracle Lemma:
recursive-halving subset-marginal tables make every per-cut value computable by an
O(poly(n, b))-gate reversible circuit), nested Dürr–Høyer minimum/maximum finding
computes the **exact** φ_s — 2023 or 2026 formalism, ties resolved in the reference
implementation's order — with Õ(2ⁿ·√K_n) oracle queries against the classical
Θ(K_n·4ⁿ): a full quadratic speedup, which we prove **optimal in the query model**
(Ω(√K_n) is necessary), so no quantum algorithm sees the cut structure faster
without exploiting it analytically. We simulate the algorithm with real amplitudes
on real systems (statevector Grover dynamics over iitx's exact per-cut values):
it returns the exact minimizing cut in 99–100% of runs with query counts fitting
K^0.56 against the classical K¹ (mean 143 queries at K = 1061). Finally we bound the ambition honestly: the partition axes remain
2^Θ(n log n) under a square root; for succinctly specified systems (units given as
circuits — the physically relevant regime) we conjecture NP-hardness, so an
exponential quantum speedup would require NP ⊆ BQP; and in the white-box regime
quantum gradient methods cannot beat classical backpropagation, which already
yields exact φ-gradients at constant overhead. The honest conclusion: quantum
computation halves the exponent of Φ, provably cannot do much more, and the
mathematics (analytic collapses, structure, gradients) remains the stronger lever.

## 1. Introduction: two communities, one computation

Integrated information theory (IIT) assigns to a system in a state a number, φ_s,
meant to quantify how irreducibly the state constrains the system's own past and
future. The computation is a nested optimization: an *outer minimization* over the
ways of partitioning the system (each partition severs some directed connections;
φ measures what the least-destructive severing destroys), with *inner selections*
(maximizations) of the "specified" past and future states. Everything is defined
on probability distributions over the system's 2ⁿ joint states. The reference
implementation is PyPhi; every number in this paper is computed by iitx, a JAX
implementation verified against PyPhi to 10⁻⁹ across 36 golden fixtures.

Quantum computation, meanwhile, offers exactly one broadly-applicable primitive
relevant here: **unstructured search and its relatives run quadratically faster**.
Grover's algorithm finds a marked item among D with O(√D) oracle queries;
Dürr–Høyer minimum-finding finds the minimum of an unsorted list of D values with
O(√D) queries; both are provably optimal — no quantum algorithm does better
without exploiting structure. The popular image of quantum computers
exponentially accelerating everything is false; where they do achieve exponential
speedups (factoring, simulating quantum dynamics), it is by exploiting very
specific algebraic structure.

The question of this paper: what does that one primitive buy for Φ? The answer
has three parts, and being precise about all three is the contribution:

1. **What exactly is the classical cost?** (§3 — sharper than the folklore in
   both directions.)
2. **What is the best quantum algorithm we can construct and prove correct?**
   (§4–5 — a quadratic speedup, exact, tie-faithful.)
3. **What can be ruled out?** (§6 — quadratic is optimal in the query model, and
   the exponential dream almost certainly dies on NP.)

## 2. The two primers

### 2.1 IIT for quantum readers, in one paragraph

A system is n units with finite alphabets (binary here); its dynamics is a
conditional distribution T(v | u) over next joint states given current ones,
factorizing over units (conditional independence): T(v|u) = Π_j p_j(v_j|u). Fix
the current state s. The *effect repertoire* is T(·|s); the *cause repertoire* is
the Bayes posterior over previous states under a uniform prior, π_c(u) ∝ T(s|u).
A *system partition* θ severs a set of directed unit-to-unit connections; the
partitioned dynamics replaces each severed input by an independent uniform
variable. For each partition, IIT computes how much the severing changes what the
state specifies about causes and effects — a value φ(θ, u, v) at candidate
cause/effect states (u, v), built from pointwise log-ratios of repertoire
probabilities. The reported φ_s composes a *minimax*: over candidate state pairs,
the *minimum-information partition* (MIP) is found (a min over θ of a normalized
φ), and the specified pair maximizes the resulting value; ties are resolved in a
fixed canonical order. The 2026 revision caps the value by "intrinsic
information" terms — also pointwise log-ratios. Everything in this paper concerns
this exact computation; the full "Φ-structure" (per-mechanism decompositions) is
handled in §5.3.

### 2.2 Quantum computation for IIT readers, in three paragraphs

A register of q qubits holds a unit vector in ℂ^(2^q) — an *amplitude* for each
classical index. Algorithms apply reversible linear operations; a *measurement*
samples an index with probability |amplitude|². Any classical circuit of G gates
can be run reversibly at O(G) cost, so a quantum computer can always evaluate a
classical function "in superposition": one application touches all indices'
amplitudes at once. This is not yet a speedup — extracting an answer still costs
measurements.

**Grover search** turns superposition into a speedup for *search*: given an
oracle circuit that flips the sign of "marked" indices, alternating the oracle
with a fixed "diffusion" operation rotates amplitude into the marked subspace, so
that after O(√(D/M)) rounds (D indices, M marked) a measurement yields a marked
index with high probability. The BBHT schedule handles unknown M;
**Dürr–Høyer minimum-finding** wraps this in a descending-threshold loop: repeat
"search for any index with value below the current best" until none exists —
expected O(√D) oracle queries total to find the exact minimum. These are
*provably optimal*: Ω(√D) oracle queries are necessary for search and hence for
minimum-finding.

The oracle is the whole game. If evaluating one candidate classically costs C,
the quantum algorithm costs O(√D · C) against the classical O(D · C): the
speedup is real only if the *per-candidate* cost C stays small inside a
reversible circuit. Sections 4–5 are the engineering of exactly that for φ.

## 3. The classical complexity of Φ, precisely

Fix n binary units, Q = 2ⁿ states, TPM given explicitly (input size N = nQ).

**The cut census.** The 2023 system partitions (the "SET" scheme: directional,
possibly multi-part) number K_n; iitx's enumeration gives **K_2..8 = 3, 22, 150,
1061, 7896, 61888, 510313**. These are not geometric — the successive ratios grow
(7.3, 6.8, 7.1, 7.4, 7.8, 8.2) — and they track the **Fubini numbers**
(ordered set partitions: 3, 13, 75, 541, 4683, 47293, 545835) within a small
constant factor, which is no accident: SET partitions are directional ordered
partitions with degenerate cases removed.

**Conjecture 1 (cut census).** K_n = Θ(Fubini(n)) = 2^Θ(n log n).
*(Status: measured for n ≤ 8 with ratio K_n/Fubini(n) ∈ [0.93, 2.0], non-monotone;
a bijective proof modulo degenerate partitions is expected to be routine and is
left open.)*

**Theorem (classical cost of φ_s).** With the pair-resolved semantics of the
reference implementation, φ_s costs Θ(K_n·Q²) value evaluations (each O(n) after
preprocessing, §4), i.e. **2^Θ(n log n)** under Conjecture 1 — *quasi-polynomial*
in the input size N: more than any polynomial, less than any exponential 2^(N^ε).
The folklore "super-exponential" is thus correct *in n* but misleading *in the
input*: the dominant exponential is the TPM itself.

**The structure level.** The full Φ-structure evaluates, for each of the 2ⁿ−1
mechanisms and each purview pair (≤ 4ⁿ), a minimum over mechanism partitions
whose count is again ordered-partition-flavored (measured: ~6.8×10⁴ padded
partitions per mechanism–purview pair at n = 5, ~2.2×10⁶ at n = 6): a second
independent 2^Θ(n log n) axis. The *relations* axis is naively doubly exponential
(subsets of distinctions) but is computed analytically in closed form (the sort
formula with inclusion–exclusion) by PyPhi and iitx — a standing reminder that
mathematical collapse beats hardware on these axes.

**The succinct regime.** Real substrates (brains, chips, cellular automata) are
not given as explicit TPMs but as *rules*: each unit's conditional is a small
circuit, input size poly(n). Every count above is then exponential-or-worse in
the input size. This is the physically honest regime, and the one where hardness
lives (§6).

## 4. The Oracle Lemma

The quantum algorithm needs, inside a reversible circuit: "given (θ, u, v),
output φ(θ, u, v)" — cheaply. Two facts make this possible.

**Lemma 1 (pointwise products).** At fixed current state s, every unpartitioned
and partitioned repertoire probability needed by φ is a product of n per-unit
factors: the effect side because T(·|s) and its severed versions factorize over
units outright; the cause side because the likelihood L(u) = T(s|u) = Π_j
p_j(s_j|u) is a product of per-unit lookups (its normalization Σ_u L(u) is
precomputed classically once). Hence any single repertoire probability is n
multiplications — or n *additions* if per-unit log-tables are stored.

**Lemma 2 (subset marginals by recursive halving).** The severed factor of unit j
under partition θ is its conditional with some subset S of inputs uniformly
averaged. All 2ⁿ subset-marginal tables of one unit occupy Σ_S 2^(n−|S|)·2 = O(3ⁿ)
numbers and are computed by *recursive halving* — averaging over one input at a
time, each entry the mean of two parent entries — in O(n·3ⁿ) time total for all
units. *(Verified exactly in e12: worst deviation from brute-force marginalization
2.2×10⁻¹⁶ — machine epsilon — over all subsets of random systems at n = 3, 4.)*

**Theorem 1 (Oracle Lemma).** After O(n·3ⁿ) classical preprocessing stored in
quantum-readable memory (QRAM, or compiled into the circuit), there is a
reversible circuit of O(poly(n, b)) gates (b-bit fixed point) computing any value
the φ_s minimax consumes — per-cut, per-state-pair φ for the 2023 semantics, and
the intrinsic-information and surprisal cap terms of 2026 — exactly to b bits.
*Proof sketch:* by Lemmas 1–2 every needed probability is n table lookups
(indexed by (unit, severed subset, retained configuration) — the severed subset
is a function of θ) combined by n additions in log-space; the φ expressions are
O(1) further arithmetic operations on these; classical reversible compilation
gives the gate bound. ∎

The preprocessing cost O(n·3ⁿ) is asymptotically negligible against the classical
φ_s cost K_n·Q² = 2^Θ(n log n) and against the quantum cost below — the point of
Lemma 2 is precisely that the oracle does not smuggle the classical computation
into itself.

## 5. The algorithm

### 5.1 QIIA for φ_s

**Theorem 2 (QIIA).** There is a quantum algorithm that, given the preprocessing
of Theorem 1, outputs the exact φ_s (2023 or 2026 semantics, b-bit arithmetic,
ties resolved in the reference order) with probability ≥ 2/3, using
**Õ(Q·√K_n · poly(n, b))** oracle gates — against the classical
Θ(K_n·Q²·poly(n, b)). Under Conjecture 1 this is 2^((1/2)·n·log n·(1+o(1))): the
exponent is halved.

*Proof sketch.* Write the reference semantics as a minimax: for each candidate
state pair (u, v) ∈ Q², the MIP value is a minimum over the K_n cuts of a
quantity from Theorem 1's oracle (with the normalization and the nonpositive
short-circuit folded into the comparison key, and the tie order realized by
comparing lexicographic pairs (key, index) — Dürr–Høyer finds the *canonical*
minimizer, not just a minimizer); φ_s is then a maximum over the Q² pairs of the
resulting (capped, for 2026) values. Run Dürr–Høyer maximum-finding over the Q²
pairs, whose value oracle is itself Dürr–Høyer minimum-finding over the K_n cuts.
The inner routine errs with bounded probability; amplifying it to error
O(1/poly) with O(log) repetitions and using the standard composition of Grover
search with bounded-error oracles preserves the outer √ complexity up to
logarithmic factors. Query count: Õ(√(Q²)·√K_n) = Õ(Q·√K_n). Correctness: at
every step the compared keys are exact b-bit values, so the returned pair and cut
are exactly the reference implementation's selections. ∎

Two remarks. First, *exactness*: because Lemma 1 makes probabilities pointwise
computable, no amplitude estimation and no sampling error enter — the quantum
part is pure search, and the output is the exact value, matching iitx bit for
bit. Second, the same construction applies unchanged to the 2026 cap (its terms
are pointwise by Theorem 1) and to candidate-subset sweeps (an outer Grover over
2ⁿ candidates adds another square root).

### 5.2 Simulation (e12)

We simulate the algorithm's quantum core with explicit statevectors: the index
register over cuts, the sign-flip oracle on iitx's exact per-cut values,
real Grover diffusion, measurement by sampling |amplitude|², and the BBHT/DH
threshold schedule. On random stochastic systems at n = 3, 4, 5 (K = 22, 150,
1061), the simulated algorithm returns the exact canonical minimizing cut in 99–100%
of runs within the query budget, with mean query counts 16 / 52 / 143 fitting
queries ~ K^0.56 (classical: K^1; already a 7.4× saving at K = 1061). The quadratic
separation is visible in a three-point log-log line — small, but real amplitudes,
real φ values, no idealization.

### 5.3 The structure level

**Theorem 3 (structure).** For each mechanism, the distinction's purview
selection (≤ 4ⁿ pairs) and partition minimization (B_n,z partitions) compose as
in Theorem 2 into Õ(√(4ⁿ·B)) queries per mechanism against the classical
Θ(4ⁿ·B); the sum over 2ⁿ−1 mechanisms is not an optimization and gains nothing.
Total: Õ(2ⁿ·√(4ⁿ·B)) = Õ(4ⁿ·√B) against Θ(8ⁿ·B). The mechanism-partition axis —
the practical wall at n = 6 — keeps its 2^Θ(n log n) character under the square
root. ∎

## 6. Limits: what quantum computation cannot do for Φ

**Theorem 4 (optimality in the query model).** Any quantum algorithm that
computes φ_s for all value assignments consistent with the oracle interface must
make Ω(√K_n) oracle queries. *Proof:* minimum-finding over K values reduces to
unstructured search (deciding whether any value lies below a threshold), whose
quantum query complexity is Θ(√K) [BBBV lower bound]; φ_s determines, for
suitable constructed instances, the outcome of that decision. ∎ Hence Theorem 2's
cut axis is optimal: **quadratic is not a preliminary result awaiting
improvement; it is the ceiling for any algorithm that treats the cut values as a
black box.** Exponential gains, if any exist, must come from *analytic structure*
of φ across cuts — which is mathematics, not hardware (the analytic relations
sum, and the closed-form theorems of the companion paper, are instances of
exactly that).

**Conjecture 2 (succinct hardness).** For systems whose unit conditionals are
given as circuits (input size poly(n)), deciding φ_s > 0 is NP-hard.
*(Sketch: reducibility detection contains strong-connectivity and
constraint-satisfaction flavors; embedding a SAT instance in a unit's conditional
makes the specified-state selection locate satisfying assignments. A full
reduction must respect the minimax semantics and is left open — hence a
conjecture, not a theorem.)* Under Conjecture 2, an exponential quantum speedup
for succinct Φ would imply NP ⊆ BQP, which is widely disbelieved.

**Remark (gradients — where quantum does *not* help).** The companion program's
central tool is the exact gradient of φ_s, computed classically by reverse-mode
differentiation at O(1)× the cost of one forward evaluation — *all* n·2ⁿ partial
derivatives at once. Quantum gradient estimation (Jordan; Gilyén–Arunachalam–
Wiebe) obtains d-dimensional gradients in Õ(√d) *queries*, which only beats
classical methods when the function is a black box queried by oracle. With an
explicit TPM, φ_s is white-box, and backpropagation is already optimal up to
constants: **there is no quantum advantage for Φ-gradients in the regime where
gradients are actually used.** In the succinct regime the query-model advantage
applies, with the same NP-shaped ceiling as above.

**Corollary (fast paths survive).** Classical shortcuts pass through unchanged:
deterministic input ⇒ φ_s(2026) = 0 in O(N) time (the determinism theorem of the
companion paper) — no algorithm, quantum or classical, should search a landscape
a theorem already values.

## 7. What a QIIA is for, honestly

Assembling the ledger: the QIIA halves the exponent of an exact, tie-faithful
φ_s computation (n ≈ 12 quantum ≈ n = 6 classical, *in query counts*), is
optimal in its model, extends to the 2026 cap and the structure level, and
requires QRAM plus fault-tolerant hardware that does not yet exist — while
classical progress on the same problem in the companion papers came from
vectorization (10⁵ exact φ_s/s on a laptop), analytic collapse of a
doubly-exponential axis, closed-form maxima theorems that *replace* search, and
gradients that replace enumeration. The honest ranking for the foreseeable
future: mathematics first, classical parallelism second, quantum quadratics
third. The value of writing the algorithm down anyway is twofold: it fixes the
true quantum ceiling for a computation whose difficulty is often gestured at
rather than measured, and it marks precisely where any genuine quantum surprise
would have to enter — through analytic structure across partitions, not through
search.

We note finally that this paper is orthogonal to "quantum IIT" (Zanardi et al.
2018; Albantakis et al. 2023), which generalizes the *definition* of φ to quantum
substrates; here the substrate and definition are classical and the *computation*
is quantum. Combining the two — a QIIA for quantum-φ of density matrices — is an
open direction with the same query-model ceiling.

## 8. Reproducibility

iitx 0.1.0, Python 3.14. Cut censuses from `iitx.enumeration.system_cuts`;
oracle-lemma verification, statevector Dürr–Høyer simulation, and Fubini
comparison in `experiments/e12_quantum_phi.ipynb` (pre-registered predictions in
its first cell). Classical cost figures from `docs/notes/performance.md`.

## References

*(to be formatted)* — Grover 1996; Boyer, Brassard, Høyer, Tapp 1998 (BBHT);
Dürr & Høyer 1996; Bennett, Bernstein, Brassard, Vazirani 1997 (lower bound);
Høyer, Mosca, de Wolf 2003 (bounded-error oracle composition); Jordan 2005;
Gilyén, Arunachalam, Wiebe 2019 (quantum gradients); Albantakis et al. 2023
(IIT 4.0); Marshall et al. 2023 (φ_s); Mayner et al. 2018 (PyPhi); Zanardi,
Tomasi, Tononi 2018 (quantum IIT); Albantakis, Prentner, Durham 2023 (quantum
mechanism); companion drafts `phi-landscape.md`, `behavioral-requirement.md`.
