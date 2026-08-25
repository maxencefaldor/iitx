# The computational-blindness trichotomy

Status: derivation note, 2026-08-25, prompted by the project lead's question:
*can we prove that for any computable measure φ, either "any computer is
conscious" or "computational power says nothing about φ"?* Answer: yes, as a
trichotomy, with one axiom doing all the work — and the finite-resource
refinement is e08's behavioral φ-requirement.

## Setup

Systems: finite networks of conditionally independent stochastic units (the iitx
class), of unbounded size. **Behavior** B(S) of a system: its visible
input–output profile (operationally, the visible-marginal k-step transition
family of e04/e08; any finite-horizon extensional notion works). A measure Φ is
any computable map from (system, state) to ℝ≥0.

**Axiom R (irreducibility grounding).** Φ = 0 on systems that are not strongly
connected (in particular on feedforward systems). This is the defining commitment
of *integrated* information — integration is irreducibility — and it holds for
every measure in this program: IIT 3.0, IIT 4.0 (2023 and 2026), the
occupation-weighted and cause-capped variants (all implement the
strong-connectivity null).

**Lemma (unrolling).** Every realizable finite-horizon behavior has a feedforward
witness: unroll the recurrent system into k + 1 layers, each layer a copy of the
units, layer t + 1 computing the one-step dynamics from layer t. The unrolled
system reproduces the k-step visible behavior exactly (including stochasticity —
each layer applies the same per-unit conditionals), is acyclic, hence not
strongly connected. (This is the unfolding construction of Doerig et al. 2019,
here used axiomatically.) Under Axiom R the witness has Φ = 0.

## Theorem (trichotomy)

Every computable measure Φ on this class satisfies at least one of:

1. **Φ is trivial** (identically 0 on all realizable behaviors);
2. **Φ is not behavior-determined** — there exist behavior-equivalent systems
   with different Φ, so behavioral and computational facts underdetermine Φ
   ("computational power says nothing about Φ", extensionally); or
3. **Φ violates Axiom R** — it assigns positive value to some feedforward
   system, i.e., it is not a measure of integration.

*Proof.* Suppose Φ satisfies Axiom R and is behavior-determined. Take any
realizable behavior b; the unrolling lemma provides a feedforward witness w with
B(w) = b and Φ(w) = 0. Behavior-determination forces Φ = 0 on every system with
behavior b; since b was arbitrary, Φ is trivial. ∎

**Reading.** "Any computer is conscious" (Φ readable off computational/behavioral
power) can only live in branch 3 — measures that score feedforward pipelines,
which are by construction not integration measures (branch 3 is inhabited: e.g.,
input–output mutual information). Every *nontrivial integration* measure lands in
branch 2: **computational content places no lower bound on Φ**. All three
branches are inhabited, so the trichotomy is tight.

## Corollaries and refinements

- **Double orthogonality (per-measure, already proven for 2023).** No lower
  bound: above. No upper bound from computational triviality either: the all-OR
  system computes a constant-flavored function yet attains the capacity maximum
  n(n−1). For φ_s(2023), computational content and Φ are unconstrained in both
  directions. Empirical exclamation point (e14): rule 110, computationally
  universal, has φ_s = 0 at 31 of 32 states.
- **The bounded-resource escape, quantified (e08).** The unrolling lemma spends
  units freely (k + 1 layers). At *fixed* size the construction is unavailable,
  and the computation–Φ coupling revives exactly as the behavioral
  φ-requirement R(b) = min Φ over fixed-size realizations of b — measured in e08
  to range over [0, 2.08] ibits at n = 4, with a certified control. The complete
  statement: **unbounded resources ⇒ Φ and computation fully orthogonal
  (theorem); bounded resources ⇒ the entire coupling is R(b), a measurable
  quantity.**
- **Epistemic addendum (unbounded substrates).** For unbounded-tape settings,
  behavior-equivalence itself is undecidable (Rice), so branch-2
  underdetermination acquires an in-principle-undecidable character; for the
  finite systems of this program everything is decidable and the theorem is
  purely structural.
- **Place in the program.** This upgrades paper 2's empirical story ("what the
  measures cannot see") with an axiomatic capstone: the invisibility of φ from
  behavior is not an accident of the 2023 formalism — it is forced by the
  irreducibility axiom itself, for every measure that will ever deserve the name
  "integrated." What remains measure-specific — and measurable only with
  gradients — is the finite-size residue R(b).
