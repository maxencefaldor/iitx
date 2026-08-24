"""IIT 4.0: system-level integrated information (Albantakis et al. 2023).

The system level of the 4.0 pipeline: condition the candidate system on its background
(effect side: background clamped at the current state; cause side: causal
marginalization conditional on the current state, Eq. 4), find the **maximal
cause-effect state** the system specifies (max intrinsic information, Eqs. 5-13), then
find the **minimum partition** over the directional system partitions (Eqs. 14-23):
``phi_s`` is the intrinsic information about the maximal states lost under that
partition, minimized across the cause and effect sides.

Everything here is pure tensor algebra plus min/max reductions — no linear programs —
so ``system_phi`` composes with ``jit``, ``vmap`` (batching over systems and states),
and ``grad`` (differentiable almost everywhere, with exact subgradients at ties).

Conventions inherited from the oracle rather than the paper are recorded in
``docs/notes/oracle-findings.md``; the load-bearing ones here: the reported φ is clamped
by the paper's ``|·|₊`` while *selection* runs on signed values (``signed_phi`` is kept,
not least for gradients); the partition normalization is ``1 / (number of distinct
severed connections)``; a partition with φ ≤ 0 short-circuits the search; and ties of the
specified state resolve by maximal φ_s, then the oracle's state-iteration order.
"""

import dataclasses
import math
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from iitx.direction import Direction
from iitx.enumeration import mechanism_partitions, subsets, system_cuts
from iitx.measures.common import quantize, strongly_connected
from iitx.repertoires import condition, purview_distribution, repertoire, sever
from iitx.states import all_states, radix_weights
from iitx.system import System, connectivity, is_strongly_connected, node_tpms

__all__ = [
	"CauseEffectState",
	"Complex",
	"Distinctions",
	"PartitionPhis",
	"PhiStructure",
	"Relations",
	"SystemPhi",
	"cause_effect_state",
	"complexes",
	"distinctions",
	"partition_phis",
	"phi_structure",
	"relations",
	"system_phi",
]

PRECISION = 1e-13
"""Quantization applied to values before any comparison (PyPhi's 4.0 ``PRECISION``)."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CauseEffectState:
	"""The maximal cause-effect state a system specifies (Eqs. 12-13).

	Attributes:
		cause_state: The past state the system specifies most, shape ``(n,)``. Units
			outside the candidate system carry no information; their entries are the
			first-occurrence tie convention, not a specification.
		effect_state: The future state the system specifies most, shape ``(n,)``.
		phi_cause: Intrinsic cause information ``ii_c`` of the cause state, in ibits.
		phi_effect: Intrinsic effect information ``ii_e`` of the effect state, in ibits.

	"""

	cause_state: jax.Array
	effect_state: jax.Array
	phi_cause: jax.Array
	phi_effect: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SystemPhi:
	"""System integrated information and its minimum partition (Eqs. 19-23).

	Attributes:
		phi: System integrated information ``phi_s = min(phi_cause, phi_effect)`` at the
			minimum partition, clamped at zero by the paper's ``|·|₊`` (as the primary
			oracle does), unnormalized, in ibits. Zero means reducible.
		signed_phi: The unclamped value — negative when a partition *increases* the
			specified probabilities. Selection uses signed values; keep this for
			gradients, which the clamp would zero out in the reducible region.
		normalized_phi: Clamped ``phi`` divided by the number of connections the minimum
			partition severs (normalization is used only to select the partition).
		phi_cause: Cause-side φ at the minimum partition (signed).
		phi_effect: Effect-side φ at the minimum partition (signed).
		cut_index: Index of the minimum partition in the canonical cut table
			(:func:`iitx.enumeration.system_cuts` for this candidate).
		cause_effect_state: The maximal cause-effect state the φ values refer to.

	"""

	phi: jax.Array
	signed_phi: jax.Array
	normalized_phi: jax.Array
	phi_cause: jax.Array
	phi_effect: jax.Array
	cut_index: jax.Array
	cause_effect_state: CauseEffectState


def cause_effect_state(
	system: System,
	state: jax.Array,
	candidate: jax.Array | None = None,
) -> CauseEffectState:
	"""Find the maximal cause-effect state a candidate system specifies.

	The intrinsic information of each candidate past/future state is its selectivity
	times its informativeness (Eqs. 5-9); the specified state maximizes it, with ties
	broken by first occurrence in canonical (little-endian) state order.

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Mask of the candidate system's units; ``None`` means all units.

	Returns:
		The maximal cause-effect state with its intrinsic information on both sides.

	"""
	if candidate is None:
		candidate = jnp.ones(system.n, dtype=bool)
	factors = node_tpms(system, check_independence=False)
	effect_factors = condition(factors, state, candidate)
	cause_factors = _backward_factors(factors, state, candidate)

	effect_state, phi_effect = _specify(
		_effect_information(effect_factors, state, candidate), system.shape
	)
	cause_state, phi_cause = _specify(
		_cause_information(cause_factors, state, candidate), system.shape
	)
	return CauseEffectState(
		cause_state=cause_state,
		effect_state=effect_state,
		phi_cause=phi_cause,
		phi_effect=phi_effect,
	)


def system_phi(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None = None,
) -> SystemPhi:
	"""Compute the system integrated information φ_s of a candidate system.

	Evaluates every directional system partition (the canonical ``SET_UNI/BI`` scheme):
	for each, the φ lost about the maximal cause-effect state on each side (Eqs. 19-20,
	without clipping, per the oracle), minimized across sides (Eq. 21); the minimum
	partition is selected by quantized ``(normalized φ, -φ)`` with first occurrence
	winning ties (Eqs. 22-23). A partition with φ ≤ 0 proves the system reducible, and —
	matching the oracle's short-circuit — the first such partition in enumeration order
	is reported as the minimum partition, with its (possibly negative) φ.

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Units of the candidate system (static, since it fixes the partition
			table); ``None`` means all units.

	Returns:
		The system φ analysis at the minimum partition.

	"""
	(
		pair_phi,
		phi_cause_z,
		phi_effect_z,
		severed,
		cut_choice,
		cause_index,
		effect_index,
		ces,
		strong,
	) = _system_tables(system, state, candidate)

	index = cut_choice[cause_index, effect_index]
	phi = pair_phi[index, cause_index, effect_index]
	normalized = phi / severed[index]
	phi_cause = phi_cause_z[index, cause_index]
	phi_effect = phi_effect_z[index, effect_index]

	signed = jnp.where(strong, phi, 0.0)
	return SystemPhi(
		phi=jnp.maximum(signed, 0.0),
		signed_phi=signed,
		normalized_phi=jnp.maximum(jnp.where(strong, normalized, 0.0), 0.0),
		phi_cause=jnp.where(strong, phi_cause, 0.0),
		phi_effect=jnp.where(strong, phi_effect, 0.0),
		cut_index=index,
		cause_effect_state=ces,
	)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PartitionPhis:
	"""Per-partition signed φ values at the resolved specified states.

	The raw material behind the minimum-partition selection, exposed for relaxations
	and landscape analysis (:mod:`iitx.relax`).

	Attributes:
		phi: Signed ``min(phi_cause, phi_effect)`` of every partition, shape ``(P,)``.
		phi_cause: Cause-side signed φ per partition, shape ``(P,)``.
		phi_effect: Effect-side signed φ per partition, shape ``(P,)``.
		severed: Connections severed by each partition (the normalization), shape
			``(P,)``.
		cause_effect_state: The resolved maximal cause-effect state.

	"""

	phi: jax.Array
	phi_cause: jax.Array
	phi_effect: jax.Array
	severed: jax.Array
	cause_effect_state: CauseEffectState


def partition_phis(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None = None,
) -> PartitionPhis:
	"""Evaluate every system partition at the resolved specified states.

	The partitions are the canonical table of :func:`iitx.enumeration.system_cuts` for
	this candidate; the specified states are resolved exactly as in
	:func:`system_phi` (ii-maximal, then φ-maximal, then oracle order).

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Units of the candidate system (static); ``None`` means all units.

	Returns:
		The per-partition signed φ values, sides, and normalizations.

	"""
	(
		pair_phi,
		phi_cause_z,
		phi_effect_z,
		severed,
		_cut_choice,
		cause_index,
		effect_index,
		ces,
		_strong,
	) = _system_tables(system, state, candidate)
	return PartitionPhis(
		phi=pair_phi[:, cause_index, effect_index],
		phi_cause=phi_cause_z[:, cause_index],
		phi_effect=phi_effect_z[:, effect_index],
		severed=severed,
		cause_effect_state=ces,
	)


def _system_tables(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None,
) -> tuple[
	jax.Array,
	jax.Array,
	jax.Array,
	jax.Array,
	jax.Array,
	jax.Array,
	jax.Array,
	CauseEffectState,
	jax.Array,
]:
	"""Compute the per-cut, per-state φ tables and resolve the specified states.

	Args:
		system: The system.
		state: Current state, shape ``(n,)``.
		candidate: Units of the candidate system (static), or ``None`` for all.

	Returns:
		``(pair_phi, phi_cause_z, phi_effect_z, severed, cut_choice, cause_index,
		effect_index, cause_effect_state, strong)`` — the per-cut φ table over
		specified-state pairs, the per-cut per-state side tables, the severed-edge
		counts, the per-pair minimum-partition choice, the resolved flat state indices,
		the resolved specification, and the strong-connectivity flag.

	"""
	units = tuple(range(system.n)) if candidate is None else tuple(sorted(candidate))
	candidate_mask = jnp.asarray(_static_mask(system.n, units))
	cuts, severed = system_cuts(system.n, units)
	cuts, severed = jnp.asarray(cuts), jnp.asarray(severed)

	factors = node_tpms(system, check_independence=False)
	effect_factors = condition(factors, state, candidate_mask)
	cause_factors = _backward_factors(factors, state, candidate_mask)
	shape = system.shape
	num_states = math.prod(shape)

	# Intrinsic information of every candidate cause/effect state, and per-cut,
	# per-state φ on each side (φ depends only on its own side's specified state, so
	# tie resolution over specified-state pairs needs the full per-state tables).
	information_effect = jnp.ravel(
		_effect_information(effect_factors, state, candidate_mask), order="F"
	)
	information_cause = jnp.ravel(
		_cause_information(cause_factors, state, candidate_mask), order="F"
	)
	selectivity_effect = jnp.ravel(
		purview_distribution(
			repertoire(effect_factors, state, candidate_mask, candidate_mask, Direction.EFFECT),
			candidate_mask,
		),
		order="F",
	)
	forward_effect = selectivity_effect
	selectivity_cause = jnp.ravel(
		purview_distribution(
			repertoire(cause_factors, state, candidate_mask, candidate_mask, Direction.CAUSE),
			candidate_mask,
		),
		order="F",
	)
	forward_cause = jnp.ravel(_likelihood(cause_factors, state, candidate_mask), order="F")

	def evaluate(cut: jax.Array) -> tuple[jax.Array, jax.Array]:
		partitioned_effect = jnp.ravel(
			purview_distribution(
				repertoire(
					sever(effect_factors, cut),
					state,
					candidate_mask,
					candidate_mask,
					Direction.EFFECT,
				),
				candidate_mask,
			),
			order="F",
		)
		partitioned_cause = jnp.ravel(
			_likelihood(sever(cause_factors, cut), state, candidate_mask), order="F"
		)
		phi_effect = selectivity_effect * _log2_ratio(forward_effect, partitioned_effect)
		phi_cause = selectivity_cause * _log2_ratio(forward_cause, partitioned_cause)
		return phi_cause, phi_effect

	phi_cause_z, phi_effect_z = jax.vmap(evaluate)(cuts)  # (num_cuts, Q) each

	# The minimum partition for every (cause state, effect state) pair: φ per cut is the
	# min across sides; a cut with φ ≤ 0 (at precision) proves reducibility and — as the
	# oracle short-circuits — the *first* such cut in enumeration order is the reported
	# minimum partition; otherwise the minimum by quantized (normalized φ, -φ).
	pair_phi = jnp.minimum(phi_cause_z[:, :, None], phi_effect_z[:, None, :])
	pair_normalized = pair_phi / severed[:, None, None]
	nonpositive = _q(pair_phi) <= 0.0
	key_normalized = _q(pair_normalized)
	minimal = key_normalized <= key_normalized.min(axis=0, keepdims=True)
	key_phi = jnp.where(minimal, _q(pair_phi), -jnp.inf)
	tied_cut = minimal & (key_phi >= key_phi.max(axis=0, keepdims=True))
	cut_choice = jnp.where(
		nonpositive.any(axis=0), jnp.argmax(nonpositive, axis=0), jnp.argmax(tied_cut, axis=0)
	)
	mip_phi = jnp.take_along_axis(pair_phi, cut_choice[None], axis=0)[0]  # (Q, Q)

	# Specified states: ii-maximal on each side, then — the oracle's tie cascade — the
	# pair with maximal φ_s, then the oracle's state-iteration order.
	tied_cause = _q(information_cause) >= _q(information_cause).max()
	tied_effect = _q(information_effect) >= _q(information_effect).max()
	allowed = tied_cause[:, None] & tied_effect[None, :]
	key_pair = jnp.where(allowed, _q(mip_phi), -jnp.inf)
	winners = allowed & (key_pair >= key_pair.max())
	rank = jnp.asarray(_oracle_rank(shape))
	pair_rank = rank[:, None] * num_states + rank[None, :]
	winner = jnp.argmin(jnp.where(winners, pair_rank, jnp.inf))
	cause_index, effect_index = winner // num_states, winner % num_states
	weights = jnp.asarray(radix_weights(shape))
	ces = CauseEffectState(
		cause_state=(cause_index // weights) % jnp.asarray(shape),
		effect_state=(effect_index // weights) % jnp.asarray(shape),
		phi_cause=information_cause[cause_index],
		phi_effect=information_effect[effect_index],
	)

	# A candidate that is not strongly connected is null by definition; a single-unit
	# candidate additionally needs a self-loop.
	strong = strongly_connected(connectivity(system), candidate_mask)
	if len(units) == 1:
		strong = strong & connectivity(system)[units[0], units[0]]

	return (
		pair_phi,
		phi_cause_z,
		phi_effect_z,
		severed,
		cut_choice,
		cause_index,
		effect_index,
		ces,
		strong,
	)


def _backward_factors(
	factors: tuple[jax.Array, ...],
	state: jax.Array,
	candidate: jax.Array,
) -> tuple[jax.Array, ...]:
	"""Build the backward (cause) conditionals of Eq. 4.

	The prior background state is unknown, so it is causally marginalized under its
	posterior given the current state of the whole system:
	``p_c(s_i | z) = sum_w p(s_i | z, w) p(w | u)``.

	Args:
		factors: Per-unit conditionals of the whole system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Mask of the candidate system's units.

	Returns:
		Backward factors, constant along background axes. When the current state is
		unreachable the posterior is undefined and the factors are zero; validate
		reachability at the library boundary.

	"""
	n = len(factors)
	# p(u | previous), for the whole current state u, as a function of the previous state.
	current = jnp.ones(factors[0].shape[:-1], dtype=factors[0].dtype)
	for i, factor in enumerate(factors):
		current = current * jnp.take(factor, state[i], axis=-1)

	# Posterior over the background's previous state: sum over candidate axes, normalize.
	background_marginal = current
	for axis in range(n):
		background_marginal = jnp.where(
			candidate[axis],
			background_marginal.sum(axis=axis, keepdims=True),
			background_marginal,
		)
	total = current.sum()
	weight = jnp.where(total > 0.0, background_marginal / jnp.where(total > 0.0, total, 1.0), 0.0)

	backward = []
	for factor in factors:
		out = factor * weight[..., None]
		for axis in range(n):
			out = jnp.where(candidate[axis], out, out.sum(axis=axis, keepdims=True))
		backward.append(out)
	return tuple(backward)


def _likelihood(
	factors: tuple[jax.Array, ...],
	state: jax.Array,
	candidate: jax.Array,
) -> jax.Array:
	"""Compute the forward probability of the candidate's current state, per prior state.

	``prod_{i in candidate} p(s_i | z)`` as a function of the prior state ``z`` — the
	forward cause probability whose ratio to its mean is the cause informativeness.

	Args:
		factors: Per-unit conditionals (backward factors on the cause side).
		state: Current state, shape ``(n,)``.
		candidate: Mask of the candidate system's units.

	Returns:
		Full-shape tensor over prior states.

	"""
	out = jnp.ones(factors[0].shape[:-1], dtype=factors[0].dtype)
	for i, factor in enumerate(factors):
		out = out * jnp.where(candidate[i], jnp.take(factor, state[i], axis=-1), 1.0)
	return out


def _effect_information(
	effect_factors: tuple[jax.Array, ...],
	state: jax.Array,
	candidate: jax.Array,
) -> jax.Array:
	"""Compute the intrinsic effect information of every candidate effect state (Eq. 5).

	Args:
		effect_factors: Background-clamped per-unit conditionals.
		state: Current state, shape ``(n,)``.
		candidate: Mask of the candidate system's units.

	Returns:
		Full-shape ``ii_e`` tensor (constant along background axes).

	"""
	constrained = purview_distribution(
		repertoire(effect_factors, state, candidate, candidate, Direction.EFFECT), candidate
	)
	return constrained * _log2_ratio(constrained, _unconstrained_effect(effect_factors, candidate))


def _unconstrained_effect(effect_factors: tuple[jax.Array, ...], candidate: jax.Array) -> jax.Array:
	"""Compute the unconstrained effect probability of every effect state (Eq. 6).

	This is the uniform average over interventions on the prior state of the *joint*
	conditional — a correlated mixture, not a product of per-unit means. (IIT 3.0's
	unconstrained effect repertoire is the product form; the two conventions differ, and
	each measure uses its own.)

	Args:
		effect_factors: Background-clamped per-unit conditionals.
		candidate: Mask of the candidate system's units.

	Returns:
		Full-shape tensor over effect states, with uniform factors on non-candidate axes.

	"""
	shape = tuple(factor.shape[-1] for factor in effect_factors)
	n = len(shape)
	# joint[prior..., next...] = prod_j p(next_j | prior); mean over the prior grid.
	joint = jnp.ones(shape + shape, dtype=effect_factors[0].dtype)
	for j, factor in enumerate(effect_factors):
		aligned = jnp.reshape(
			factor, factor.shape[:-1] + (1,) * j + (shape[j],) + (1,) * (n - 1 - j)
		)
		uniform = jnp.full_like(aligned, 1.0 / shape[j])
		joint = joint * jnp.where(candidate[j], aligned, uniform)
	return joint.mean(axis=tuple(range(n)))


def _cause_information(
	cause_factors: tuple[jax.Array, ...],
	state: jax.Array,
	candidate: jax.Array,
) -> jax.Array:
	"""Compute the intrinsic cause information of every candidate cause state (Eq. 7).

	Selectivity is the backward (Bayes) probability; informativeness is the forward
	ratio ``log2(p_c(s | z) / p_c(s))``.

	Args:
		cause_factors: Backward per-unit conditionals (Eq. 4).
		state: Current state, shape ``(n,)``.
		candidate: Mask of the candidate system's units.

	Returns:
		Full-shape ``ii_c`` tensor (constant along background axes).

	"""
	selectivity = purview_distribution(
		repertoire(cause_factors, state, candidate, candidate, Direction.CAUSE), candidate
	)
	forward = _likelihood(cause_factors, state, candidate)
	return selectivity * _log2_ratio(forward, jnp.mean(forward))


def _specify(information: jax.Array, shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
	"""Select the state maximizing an intrinsic-information tensor.

	Ties are broken by first occurrence in the oracle's state-iteration order (the last
	unit varying fastest), so tied specified states — common in symmetric deterministic
	systems, where the choice changes downstream φ — agree with the oracle.

	Args:
		information: Full-shape intrinsic-information tensor.
		shape: Per-unit alphabet sizes.

	Returns:
		The maximizing state vector and its intrinsic information.

	"""
	flat = jnp.ravel(information, order="F")
	tied = _q(flat) >= _q(flat).max()
	index = jnp.argmin(jnp.where(tied, jnp.asarray(_oracle_rank(shape)), jnp.inf))
	state = (index // jnp.asarray(radix_weights(shape))) % jnp.asarray(shape)
	return state, flat[index]


def _oracle_rank(shape: tuple[int, ...]) -> np.ndarray:
	"""Rank every state by the oracle's iteration order, indexed little-endian.

	PyPhi iterates candidate states with the *last* unit varying fastest
	(``itertools.product``); iitx's canonical flat order varies unit 0 fastest. This
	table maps each little-endian flat index to its rank in the oracle's order, so
	first-occurrence tie-breaking can follow the oracle exactly
	(``docs/notes/oracle-findings.md`` §5).

	Args:
		shape: Per-unit alphabet sizes.

	Returns:
		Integer rank table of shape ``(Q,)``, a build-time constant.

	"""
	states = all_states(shape)
	weights = np.ones(len(shape), dtype=np.int64)
	for i in range(len(shape) - 2, -1, -1):
		weights[i] = weights[i + 1] * shape[i + 1]
	return states @ weights


def _at(full: jax.Array, state: jax.Array, shape: tuple[int, ...]) -> jax.Array:
	"""Read a full-shape tensor at a state vector.

	Args:
		full: Full-shape tensor.
		state: State vector, shape ``(n,)``.
		shape: Per-unit alphabet sizes.

	Returns:
		The scalar value at that state.

	"""
	index = jnp.sum(state * jnp.asarray(radix_weights(shape)))
	return jnp.ravel(full, order="F")[index]


def _log2_ratio(p: jax.Array, q: jax.Array) -> jax.Array:
	"""Compute ``log2(p / q)`` with guarded gradients at ``p = 0``.

	Args:
		p: Numerator probabilities.
		q: Denominator probabilities.

	Returns:
		The elementwise log-ratio; ``0`` where ``p = 0`` (whose weight is zero anyway),
		``inf`` where ``p > 0`` and ``q = 0``.

	"""
	safe_p = jnp.where(p > 0.0, p, 1.0)
	safe_q = jnp.where(q > 0.0, q, 1.0)
	return jnp.where(p > 0.0, jnp.where(q > 0.0, jnp.log2(safe_p / safe_q), jnp.inf), 0.0)


def _q(x: jax.Array) -> jax.Array:
	"""Quantize to this measure's precision (see :func:`iitx.measures.common.quantize`).

	Args:
		x: Values to quantize.

	Returns:
		Values rounded to the nearest multiple of :data:`PRECISION`.

	"""
	return quantize(x, PRECISION)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Distinctions:
	"""The causal distinctions of a candidate system (Eqs. 27-48), as stacked arrays.

	Row ``k`` describes the mechanism whose mask is row ``k`` of
	``iitx.enumeration.subsets(n, nonempty=True)`` restricted to the candidate; rows for
	mechanisms not contained in the candidate, and mechanisms that specify no congruent
	irreducible cause-effect, have ``exists`` false and zeros elsewhere.

	Attributes:
		exists: Which mechanisms are distinctions (φ_d > 0 and congruent), shape ``(D,)``.
		phi: Distinction integrated information ``phi_d = min(phi_cause, phi_effect)``,
			shape ``(D,)``.
		mechanism: Mechanism masks, shape ``(D, n)``.
		cause_purview: Maximally irreducible cause purview masks, shape ``(D, n)``.
		effect_purview: Maximally irreducible effect purview masks, shape ``(D, n)``.
		cause_state: Specified cause states (values outside the purview are conventions,
			not specifications), shape ``(D, n)``.
		effect_state: Specified effect states, shape ``(D, n)``.
		phi_cause: Cause-side φ maxima, shape ``(D,)``.
		phi_effect: Effect-side φ maxima, shape ``(D,)``.

	"""

	exists: jax.Array
	phi: jax.Array
	mechanism: jax.Array
	cause_purview: jax.Array
	effect_purview: jax.Array
	cause_state: jax.Array
	effect_state: jax.Array
	phi_cause: jax.Array
	phi_effect: jax.Array


def distinctions(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None = None,
	specification: CauseEffectState | None = None,
) -> Distinctions:
	"""Unfold the causal distinctions of a candidate system.

	For every mechanism, on each side: the intrinsic information of every purview state
	selects candidate specified states (Eqs. 34-37); φ over the disintegrating partitions
	Θ(M, Z) (Eqs. 38-44, minimum by quantized normalized φ with the oracle's φ ≤ 0
	short-circuit) evaluates each purview; the maximally irreducible purview wins
	(Eqs. 45-46). Ties are searched for a representative congruent with the system's
	maximal cause-effect state (Eq. 48): among φ-tied states the congruent one is
	preferred, then among φ-tied purviews the one whose state is congruent; a mechanism
	with no congruent representative on either side is not a distinction.

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Units of the candidate system (static); ``None`` means all units.
		specification: The system's maximal cause-effect state to be congruent with.
			``None`` computes the ii-level specification; pass the φ-resolved one from
			:func:`system_phi` (as :func:`phi_structure` does) for oracle-exact
			congruence when the specification is tied.

	Returns:
		The distinctions, stacked over the nonempty mechanisms of the candidate.

	"""
	units = tuple(range(system.n)) if candidate is None else tuple(sorted(candidate))
	n, shape = system.n, system.shape
	static_candidate = _static_mask(n, units)
	candidate_mask = jnp.asarray(static_candidate)

	factors = node_tpms(system, check_independence=False)
	effect_factors = condition(factors, state, candidate_mask)
	cause_factors = _backward_factors(factors, state, candidate_mask)
	ces = (
		specification
		if specification is not None
		else cause_effect_state(system, state, candidate_mask)
	)

	# Master tables over co-part bitmasks (axis 1 ordered by bitmask value).
	bits = np.asarray([[bool(mask >> i & 1) for i in range(n)] for mask in range(2**n)], dtype=bool)
	smear_tables = tuple(
		jax.vmap(lambda keep, factor=factor: _smear_axes(factor, keep))(jnp.asarray(bits))
		for factor in effect_factors
	)
	conditional_tables = tuple(
		jax.vmap(lambda table_row: _at_prev(table_row, state, shape))(table)
		for table in smear_tables
	)
	likelihoods = tuple(
		jnp.take(factor, state[u], axis=-1) for u, factor in enumerate(cause_factors)
	)
	marginal_tables = jnp.stack(
		[
			jax.vmap(lambda keep, like=like: _smear_axes(like, keep))(jnp.asarray(bits))
			for like in likelihoods
		]
	)

	mechanism_table = subsets(n, nonempty=True)
	purview_table = subsets(n, nonempty=True)

	# One compiled body per mechanism-size class: mechanisms of equal size share their
	# partition-table shapes exactly, so a lax.scan over the class members reuses one
	# trace where a Python loop would unroll 2**n - 1 bodies.
	row_of = {tuple(row.tolist()): index for index, row in enumerate(mechanism_table)}
	results: list[tuple[jax.Array, ...] | None] = [None] * len(mechanism_table)
	inside = jnp.asarray(np.all(~purview_table | static_candidate, axis=1))
	purviews = jnp.asarray(purview_table)
	for size in range(1, n + 1):
		members = [
			row
			for row in mechanism_table
			if row.sum() == size and bool(np.all(~row | static_candidate))
		]
		if not members:
			continue

		stacked_masks = np.stack(members)
		stacked_bitmasks = (stacked_masks @ (1 << np.arange(n))).astype(np.int32)
		co_mechanism_class = []
		co_purview_class = []
		severed_class = []
		valid_class = []
		for member in members:
			mechanism_units = tuple(int(u) for u in np.flatnonzero(member))
			tables = [
				mechanism_partitions(
					mechanism_units, tuple(int(u) for u in np.flatnonzero(purview_row)), n
				)
				for purview_row in purview_table
			]
			max_partitions = max(len(entry[2]) for entry in tables)
			co_mechanism = np.zeros((len(purview_table), max_partitions, n), dtype=np.int32)
			co_purview = np.zeros_like(co_mechanism)
			severed = np.ones((len(purview_table), max_partitions), dtype=np.int32)
			valid = np.zeros((len(purview_table), max_partitions), dtype=bool)
			for k, (a, b, cut_count) in enumerate(tables):
				co_mechanism[k, : len(cut_count)] = a
				co_purview[k, : len(cut_count)] = b
				severed[k, : len(cut_count)] = cut_count
				valid[k, : len(cut_count)] = True
			co_mechanism_class.append(co_mechanism)
			co_purview_class.append(co_purview)
			severed_class.append(severed)
			valid_class.append(valid)

		def member_row(
			mechanism_row: jax.Array,
			mechanism_bitmask: jax.Array,
			co_mechanism: jax.Array,
			co_purview: jax.Array,
			severed: jax.Array,
			valid: jax.Array,
		) -> tuple[jax.Array, ...]:
			side = {}
			for direction in Direction:
				specified = ces.cause_state if direction is Direction.CAUSE else ces.effect_state

				def evaluate_purview(
					purview: jax.Array,
					co_m: jax.Array,
					co_p: jax.Array,
					ncc: jax.Array,
					ok: jax.Array,
					direction: Direction = direction,
					mechanism_row: jax.Array = mechanism_row,
					mechanism_bitmask: jax.Array = mechanism_bitmask,
					specified: jax.Array = specified,
				) -> tuple[jax.Array, jax.Array, jax.Array]:
					return _purview_phi(
						direction,
						mechanism_row,
						mechanism_bitmask,
						purview,
						co_m,
						co_p,
						ncc,
						ok,
						conditional_tables,
						marginal_tables,
						smear_tables,
						specified,
						shape,
					)

				phi_z, congruent, chosen_state = jax.vmap(
					evaluate_purview, in_axes=(0, 0, 0, 0, 0)
				)(purviews, co_mechanism, co_purview, severed, valid)
				phi_z = jnp.where(inside, phi_z, -jnp.inf)
				best = _q(phi_z) >= _q(phi_z).max()
				preferred = best & congruent
				purview_index = jnp.where(preferred.any(), jnp.argmax(preferred), jnp.argmax(best))
				side[direction] = (
					phi_z[purview_index],
					purviews[purview_index],
					chosen_state[purview_index],
					preferred.any(),
				)

			phi_cause, cause_purview, cause_state_row, cause_congruent = side[Direction.CAUSE]
			phi_effect, effect_purview, effect_state_row, effect_congruent = side[Direction.EFFECT]
			phi_d = jnp.minimum(phi_cause, phi_effect)
			exists = (_q(phi_d) > 0.0) & cause_congruent & effect_congruent
			return (
				exists,
				jnp.where(exists, phi_d, 0.0),
				mechanism_row,
				cause_purview,
				effect_purview,
				cause_state_row,
				effect_state_row,
				phi_cause,
				phi_effect,
			)

		def body(carry: None, xs: tuple[jax.Array, ...]) -> tuple[None, tuple[jax.Array, ...]]:
			return carry, member_row(*xs)

		_, outputs = jax.lax.scan(
			body,
			None,
			(
				jnp.asarray(stacked_masks),
				jnp.asarray(stacked_bitmasks),
				jnp.asarray(np.stack(co_mechanism_class)),
				jnp.asarray(np.stack(co_purview_class)),
				jnp.asarray(np.stack(severed_class)),
				jnp.asarray(np.stack(valid_class)),
			),
		)
		for position, member in enumerate(members):
			results[row_of[tuple(member.tolist())]] = tuple(leaf[position] for leaf in outputs)

	def zeros_row() -> tuple[jax.Array, ...]:
		return (
			jnp.asarray(False),
			jnp.asarray(0.0, dtype=system.tpm.dtype),
			jnp.zeros(n, dtype=bool),
			jnp.zeros(n, dtype=bool),
			jnp.zeros(n, dtype=bool),
			jnp.zeros(n, dtype=jnp.asarray(state).dtype),
			jnp.zeros(n, dtype=jnp.asarray(state).dtype),
			jnp.asarray(0.0, dtype=system.tpm.dtype),
			jnp.asarray(0.0, dtype=system.tpm.dtype),
		)

	strong = strongly_connected(connectivity(system), candidate_mask)
	rows = [row if row is not None else zeros_row() for row in results]
	stacked = [jnp.stack(column) for column in zip(*rows, strict=True)]
	return Distinctions(
		exists=stacked[0] & strong,
		phi=stacked[1],
		mechanism=stacked[2],
		cause_purview=stacked[3],
		effect_purview=stacked[4],
		cause_state=stacked[5],
		effect_state=stacked[6],
		phi_cause=stacked[7],
		phi_effect=stacked[8],
	)


def _static_mask(n: int, units: tuple[int, ...]) -> np.ndarray:
	"""Build the static candidate mask of a unit tuple.

	Args:
		n: Number of units.
		units: The candidate's units.

	Returns:
		Boolean NumPy mask of shape ``(n,)`` — static, since the candidate is part of
		the computation's structure, not its data.

	"""
	mask = np.zeros(n, dtype=bool)
	mask[list(units)] = True
	return mask


def _smear_axes(x: jax.Array, keep: jax.Array) -> jax.Array:
	"""Replace each state axis outside ``keep`` by its uniform mean, keeping dimensions.

	Args:
		x: Tensor whose first ``n`` axes are state axes (trailing axes untouched).
		keep: Mask of the state axes to keep, shape ``(n,)``.

	Returns:
		Tensor of the same shape, constant along the smeared axes.

	"""
	out = x
	for axis in range(keep.shape[0]):
		out = jnp.where(keep[axis], out, out.mean(axis=axis, keepdims=True))
	return out


def _at_prev(x: jax.Array, state: jax.Array, shape: tuple[int, ...]) -> jax.Array:
	"""Index the first ``n`` (state) axes of a tensor at a state vector.

	Args:
		x: Tensor whose first ``n`` axes are state axes.
		state: State vector, shape ``(n,)``.
		shape: Per-unit alphabet sizes.

	Returns:
		The tensor with the state axes consumed (trailing axes remain).

	"""
	out = x
	for i in range(len(shape)):
		out = jnp.take(out, state[i], axis=0)
	return out


def _purview_phi(
	direction: Direction,
	mechanism: jax.Array,
	mechanism_bitmask: jax.Array,
	purview: jax.Array,
	co_mechanism: jax.Array,
	co_purview: jax.Array,
	severed: jax.Array,
	ok: jax.Array,
	conditional_tables: tuple[jax.Array, ...],
	marginal_tables: jax.Array,
	smear_tables: tuple[jax.Array, ...],
	specified: jax.Array,
	shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Evaluate one mechanism-purview pair in one direction.

	Computes the intrinsic information of every purview state, the φ of every
	disintegrating partition at every state, the per-state minimum partition (with the
	oracle's φ ≤ 0 short-circuit), and the specified state — preferring, among φ-tied
	candidates, the state congruent with the system's specification.

	Args:
		direction: Temporal direction. Static.
		mechanism: Mechanism mask, shape ``(n,)``.
		mechanism_bitmask: The mechanism as a bitmask (scalar, traced).
		purview: Purview mask, shape ``(n,)``.
		co_mechanism: Per-partition co-part mechanism bitmasks of each purview unit.
		co_purview: Per-partition co-part purview bitmasks of each mechanism unit.
		severed: Connections severed by each partition (the φ normalization).
		ok: Which partition rows are real (the tables are padded).
		conditional_tables: Per unit ``j``: ``p(v | clamp A at the current state)`` for
			every bitmask ``A``, shape ``(2**n, q_j)``.
		marginal_tables: Per unit ``u``: the backward likelihood of ``u``'s current
			value, smeared to every keep-bitmask, shape ``(n, 2**n, *shape)``.
		smear_tables: Per unit ``j``: its conditional smeared to every keep-bitmask,
			shape ``(2**n, *shape, q_j)``.
		specified: The system's specified state for this direction, shape ``(n,)``.
		shape: Per-unit alphabet sizes.

	Returns:
		The φ of the maximally irreducible partition at the chosen state, whether the
		chosen state is the congruent one, and the chosen state.

	"""
	n = len(shape)

	if direction is Direction.EFFECT:
		# Selectivity = forward = the product effect repertoire over purview states.
		# Non-purview axes carry the identity factor 1, so every value is the true
		# purview probability, constant along non-purview axes.
		value = _outer(
			[
				jnp.where(
					purview[j],
					jnp.take(conditional_tables[j], mechanism_bitmask, axis=0),
					jnp.ones(shape[j]),
				)
				for j in range(n)
			],
			shape,
		)
		selectivity = value
		# Unconstrained: mixture over mechanism states of the joint conditional (Eq. 31).
		joint = jnp.ones(shape + shape, dtype=value.dtype)
		for j in range(n):
			aligned = jnp.reshape(
				jnp.take(smear_tables[j], mechanism_bitmask, axis=0),
				shape + (1,) * j + (shape[j],) + (1,) * (n - 1 - j),
			)
			joint = joint * jnp.where(purview[j], aligned, 1.0)
		unconstrained = joint.mean(axis=tuple(range(n)))
		information = selectivity * _log2_ratio(value, unconstrained)

		def partitioned(row: jax.Array) -> jax.Array:
			return _outer(
				[
					jnp.where(
						purview[j],
						jnp.take(conditional_tables[j], row[j], axis=0),
						jnp.ones(shape[j]),
					)
					for j in range(n)
				],
				shape,
			)

		partitioned_value = jax.vmap(partitioned)(co_mechanism)
	else:
		purview_bitmask = jnp.sum(jnp.where(purview, 1 << jnp.arange(n), 0))
		value = jnp.ones(shape, dtype=marginal_tables.dtype)
		for u in range(n):
			value = value * jnp.where(
				mechanism[u], jnp.take(marginal_tables[u], purview_bitmask, axis=0), 1.0
			)
		total = value
		for axis in range(n):
			total = jnp.where(purview[axis], total.sum(axis=axis, keepdims=True), total)
		selectivity = jnp.where(total > 0.0, value / jnp.where(total > 0.0, total, 1.0), 0.0)
		num_states = jnp.prod(jnp.where(purview, jnp.asarray(shape), 1))
		information = selectivity * _log2_ratio(value, total / num_states)

		def partitioned(row: jax.Array) -> jax.Array:
			out = jnp.ones(shape, dtype=marginal_tables.dtype)
			for u in range(n):
				out = out * jnp.where(
					mechanism[u], jnp.take(marginal_tables[u], row[u], axis=0), 1.0
				)
			return out

		partitioned_value = jax.vmap(partitioned)(co_purview)

	# φ of every partition at every purview state (Eqs. 41-44, unclipped).
	phi = selectivity[None] * _log2_ratio(value[None], partitioned_value)
	phi = jnp.where(ok.reshape((-1,) + (1,) * n), phi, jnp.inf)

	# Per state: the oracle's short-circuit at the first φ ≤ 0 partition, else the
	# minimum by quantized (normalized φ, -φ).
	flat_phi = phi.reshape((phi.shape[0], -1), order="F")
	nonpositive = (_q(flat_phi) <= 0.0) & ok[:, None]
	key_normalized = jnp.where(ok[:, None], _q(flat_phi / severed[:, None]), jnp.inf)
	minimal = key_normalized <= key_normalized.min(axis=0, keepdims=True)
	key_phi = jnp.where(minimal, _q(flat_phi), -jnp.inf)
	tied = minimal & (key_phi >= key_phi.max(axis=0, keepdims=True))
	choice = jnp.where(
		nonpositive.any(axis=0), jnp.argmax(nonpositive, axis=0), jnp.argmax(tied, axis=0)
	)
	mip_phi = jnp.take_along_axis(flat_phi, choice[None], axis=0)[0]

	# Specified state: ii-maximal states, then φ-maximal among them, preferring the
	# congruent state, then first occurrence in canonical order.
	flat_information = jnp.ravel(information, order="F")
	tied_information = _q(flat_information) >= _q(flat_information).max()
	key = jnp.where(tied_information, _q(mip_phi), -jnp.inf)
	candidates = tied_information & (key >= key.max())
	congruent_index = jnp.sum(specified * jnp.asarray(radix_weights(shape)))
	congruent = candidates[congruent_index]
	fallback = jnp.argmin(jnp.where(candidates, jnp.asarray(_oracle_rank(shape)), jnp.inf))
	index = jnp.where(congruent, congruent_index, fallback)
	state = (index // jnp.asarray(radix_weights(shape))) % jnp.asarray(shape)
	return mip_phi[index], congruent, state


def _outer(vectors: list[jax.Array], shape: tuple[int, ...]) -> jax.Array:
	"""Build the outer product of per-unit vectors as a full-shape tensor.

	Args:
		vectors: One vector per unit, lengths ``shape``.
		shape: Per-unit alphabet sizes.

	Returns:
		Full-shape tensor ``prod_j vectors[j][z_j]``.

	"""
	out = jnp.ones(shape, dtype=vectors[0].dtype)
	for j, vector in enumerate(vectors):
		out = out * vector.reshape((1,) * j + (-1,) + (1,) * (len(shape) - 1 - j))
	return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Relations:
	"""The causal relations of a distinction set, in aggregate (Eqs. 49-56).

	Relations are never enumerated — their number is doubly exponential — but their
	total φ and count have closed forms (S3 Text of Albantakis et al. 2023), which is
	all that Φ needs.

	Attributes:
		sum_phi: ``sum(phi_r)`` over all relations, in ibits.
		count: Number of relations (including self-relations with φ_r > 0).

	"""

	sum_phi: jax.Array
	count: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhiStructure:
	"""The Φ-structure of a candidate system: distinctions, relations, and Φ (Eqs. 57-59).

	Attributes:
		system: The system-level irreducibility analysis (φ_s and its partition).
		distinctions: The congruent causal distinctions.
		relations: The relations, in aggregate.
		big_phi: Structure integrated information ``Φ = sum(phi_d) + sum(phi_r)``.

	"""

	system: SystemPhi
	distinctions: Distinctions
	relations: Relations
	big_phi: jax.Array


def relations(found: Distinctions, ces: CauseEffectState, shape: tuple[int, ...]) -> Relations:
	"""Aggregate the relations among a set of distinctions, analytically.

	A relation binds distinctions whose cause/effect purviews overlap *congruently* —
	over the same units in the same specified states. Since every distinction is
	congruent with the system's cause-effect state, the overlap arithmetic reduces to
	sets of unit-state atoms: one atom per unit and side, merged when the system's cause
	and effect states agree on the unit.

	``sum(phi_r)`` uses the S3 Text sort identity (per atom, sort member distinctions by
	φ density and weight by the number of subsets whose minimum they are); the count
	uses inclusion-exclusion over atom subsets. Self-relations (a single distinction
	whose own cause and effect overlap) are added individually.

	Args:
		found: The distinctions (with existence mask).
		ces: The system's maximal cause-effect state (for the atom merging).
		shape: Per-unit alphabet sizes.

	Returns:
		The total relation φ and the relation count.

	"""
	n = len(shape)
	merged = ces.cause_state == ces.effect_state

	# Atom membership: slots 0..n-1 are cause-side atoms, n..2n-1 effect-side atoms;
	# when the system states agree on a unit its effect atom folds into its cause atom.
	cause_atoms = found.cause_purview | (merged[None] & found.effect_purview)
	effect_atoms = ~merged[None] & found.effect_purview
	atoms = jnp.concatenate([cause_atoms, effect_atoms], axis=1) & found.exists[:, None]

	size = atoms.sum(axis=1)
	density = jnp.where(size > 0, found.phi / jnp.where(size > 0, size, 1), 0.0)

	# Relations of two or more distinctions: per atom, sort members by density.
	member = atoms & found.exists[:, None]
	keys = jnp.where(member, density[:, None], jnp.inf)
	order = jnp.argsort(keys, axis=0, stable=True)
	sorted_density = jnp.take_along_axis(keys, order, axis=0)
	counts = member.sum(axis=0)
	ranks = jnp.arange(found.exists.shape[0])[:, None] + 1
	weights = jnp.where(ranks <= counts[None], 2.0 ** (counts[None] - ranks) - 1.0, 0.0)
	sum_higher = jnp.sum(jnp.where(weights > 0, sorted_density * weights, 0.0))

	# Count of relations with two or more distinctions: inclusion-exclusion over atom
	# subsets with nonempty common overlap.
	atom_subsets = jnp.asarray(subsets(2 * n, nonempty=True))
	missing = (~atoms).astype(jnp.int32) @ atom_subsets.T.astype(jnp.int32)
	contains = (missing == 0) & found.exists[:, None]
	num_containing = contains.sum(axis=0)
	sign = jnp.where(atom_subsets.sum(axis=1) % 2 == 1, 1.0, -1.0)
	count_higher = jnp.sum(sign * (2.0**num_containing - num_containing - 1.0)).astype(jnp.int64)

	# Self-relations: a distinction related to itself through the overlap of its own
	# cause and effect purviews (nonempty only over merged atoms).
	self_overlap = (merged[None] & found.cause_purview & found.effect_purview).sum(axis=1)
	self_phi = jnp.where(found.exists & (self_overlap > 0), self_overlap * density, 0.0)
	sum_self = self_phi.sum()
	count_self = jnp.sum(found.exists & (self_overlap > 0)).astype(jnp.int64)

	return Relations(sum_phi=sum_higher + sum_self, count=count_higher + count_self)


def phi_structure(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None = None,
) -> PhiStructure:
	"""Unfold the Φ-structure of a candidate system.

	Composes the system-level analysis, the distinctions, and the relations;
	``Φ = sum(phi_d) + sum(phi_r)`` (Eq. 59) — a sum over what exists, not a partitioned
	distance.

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Units of the candidate system (static); ``None`` means all units.

	Returns:
		The Φ-structure.

	"""
	analysis = system_phi(system, state, candidate)
	found = distinctions(system, state, candidate, specification=analysis.cause_effect_state)
	related = relations(found, analysis.cause_effect_state, system.shape)
	return PhiStructure(
		system=analysis,
		distinctions=found,
		relations=related,
		big_phi=jnp.where(found.exists, found.phi, 0.0).sum() + related.sum_phi,
	)


@dataclasses.dataclass(frozen=True)
class Complex:
	"""One complex of a condensed system.

	Attributes:
		units: The complex's units.
		analysis: Its system φ analysis.

	"""

	units: tuple[int, ...]
	analysis: SystemPhi


def complexes(system: System, state: jax.Array) -> list[Complex]:
	"""Condense a system into its complexes (recursive exclusion, Eqs. 24-26).

	Among all candidate systems, the one with maximal φ_s is a complex (the first
	complex is the maximal substrate); its units are removed and the search recurses on
	the remainder until no candidate has positive φ_s. Excluded units remain background
	conditions throughout. A Python driver by design (the recursion is data-dependent);
	each candidate's analysis is the jitted :func:`system_phi`.

	Ties among overlapping candidates resolve by first occurrence in powerset order
	(the S1 Text's Φ-comparison cascade for exact ties is deferred with the deeper tie
	machinery; the residual divergence is recorded in ``docs/notes/oracle-findings.md``).

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.

	Returns:
		The complexes, in order of discovery (decreasing φ_s across rounds).

	"""
	remaining = list(range(system.n))
	found: list[Complex] = []
	while remaining:
		best: Complex | None = None
		for size in range(1, len(remaining) + 1):
			for units in combinations(remaining, size):
				if not is_strongly_connected(system, units):
					continue
				analysis = system_phi(system, state, units)
				if float(analysis.phi) > 0.0 and (
					best is None or float(analysis.phi) > float(best.analysis.phi)
				):
					best = Complex(units=units, analysis=analysis)
		if best is None:
			break
		found.append(best)
		remaining = [u for u in remaining if u not in best.units]
	return found
