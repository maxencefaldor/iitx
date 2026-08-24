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

Two conventions are inherited from the oracle rather than the paper, both recorded in
``docs/notes/``:

- No positive-part clipping in φ (the paper's Eqs. 19-20 write ``|·|₊``; the reference
  implementation does not clip, and its published fixtures have **negative** φ_s for
  noisy systems — negative φ_s means reducible).
- The partition normalization is ``1 / (number of distinct severed connections)`` (the
  union of the blocks' cuts), and the minimum partition is selected by the key
  ``(normalized φ, -φ)`` with values quantized to the precision before comparison.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from iitx.direction import Direction
from iitx.enumeration import mechanism_partitions, subsets, system_cuts
from iitx.repertoires import purview_distribution, repertoire
from iitx.states import radix_weights
from iitx.system import System, node_tpms

__all__ = ["CauseEffectState", "SystemPhi", "cause_effect_state", "system_phi"]

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

	cause_state: Int[Array, " n"]
	effect_state: Int[Array, " n"]
	phi_cause: Float[Array, ""]
	phi_effect: Float[Array, ""]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SystemPhi:
	"""System integrated information and its minimum partition (Eqs. 19-23).

	Attributes:
		phi: System integrated information ``phi_s = min(phi_cause, phi_effect)`` at the
			minimum partition, unnormalized, in ibits. Negative means reducible.
		normalized_phi: ``phi`` divided by the number of connections the minimum
			partition severs (used only to select the partition).
		phi_cause: Cause-side φ at the minimum partition.
		phi_effect: Effect-side φ at the minimum partition.
		cut_index: Index of the minimum partition in the canonical cut table
			(:func:`iitx.enumeration.system_cuts` for this candidate).
		cause_effect_state: The maximal cause-effect state the φ values refer to.

	"""

	phi: Float[Array, ""]
	normalized_phi: Float[Array, ""]
	phi_cause: Float[Array, ""]
	phi_effect: Float[Array, ""]
	cut_index: Int[Array, ""]
	cause_effect_state: CauseEffectState


def cause_effect_state(
	system: System,
	state: Int[Array, " n"],
	candidate: Bool[Array, " n"] | None = None,
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
	effect_factors = _clamp_background(factors, state, candidate)
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
	state: Int[Array, " n"],
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
	units = tuple(range(system.n)) if candidate is None else tuple(sorted(candidate))
	candidate_mask = jnp.zeros(system.n, dtype=bool).at[jnp.asarray(units)].set(True)
	cuts, severed = system_cuts(system.n, units)
	cuts, severed = jnp.asarray(cuts), jnp.asarray(severed)

	factors = node_tpms(system, check_independence=False)
	effect_factors = _clamp_background(factors, state, candidate_mask)
	cause_factors = _backward_factors(factors, state, candidate_mask)

	ces = cause_effect_state(system, state, candidate_mask)

	# Uncut quantities at the specified states.
	effect_repertoire = purview_distribution(
		repertoire(effect_factors, state, candidate_mask, candidate_mask, Direction.EFFECT),
		candidate_mask,
	)
	selectivity_effect = _at(effect_repertoire, ces.effect_state, system.shape)
	forward_effect = selectivity_effect
	selectivity_cause = _at(
		purview_distribution(
			repertoire(cause_factors, state, candidate_mask, candidate_mask, Direction.CAUSE),
			candidate_mask,
		),
		ces.cause_state,
		system.shape,
	)
	forward_cause = _at(
		_likelihood(cause_factors, state, candidate_mask), ces.cause_state, system.shape
	)

	def evaluate(cut: Bool[Array, "n n"]) -> tuple[Float[Array, ""], Float[Array, ""]]:
		partitioned_effect = _at(
			purview_distribution(
				repertoire(
					_sever(effect_factors, cut),
					state,
					candidate_mask,
					candidate_mask,
					Direction.EFFECT,
				),
				candidate_mask,
			),
			ces.effect_state,
			system.shape,
		)
		partitioned_cause = _at(
			_likelihood(_sever(cause_factors, cut), state, candidate_mask),
			ces.cause_state,
			system.shape,
		)
		phi_effect = selectivity_effect * _log2_ratio(forward_effect, partitioned_effect)
		phi_cause = selectivity_cause * _log2_ratio(forward_cause, partitioned_cause)
		return phi_cause, phi_effect

	phi_cause, phi_effect = jax.vmap(evaluate)(cuts)
	phi = jnp.minimum(phi_cause, phi_effect)
	normalized = phi / severed

	# A partition with φ ≤ 0 (at precision) proves reducibility: the oracle short-circuits
	# and reports the *first* such partition in enumeration order. Otherwise the minimum
	# partition is selected by quantized (normalized φ, -φ), first occurrence among ties.
	nonpositive = _quantize(phi) <= 0.0
	key_normalized = _quantize(normalized)
	minimal = key_normalized <= key_normalized.min()
	key_phi = jnp.where(minimal, _quantize(phi), -jnp.inf)
	tied = minimal & (key_phi >= key_phi.max())
	index = jnp.where(nonpositive.any(), jnp.argmax(nonpositive), jnp.argmax(tied))

	return SystemPhi(
		phi=phi[index],
		normalized_phi=normalized[index],
		phi_cause=phi_cause[index],
		phi_effect=phi_effect[index],
		cut_index=index,
		cause_effect_state=ces,
	)


def _clamp_background(
	factors: tuple[Float[Array, "*shape q"], ...],
	state: Int[Array, " n"],
	candidate: Bool[Array, " n"],
) -> tuple[Float[Array, "*shape q"], ...]:
	"""Clamp non-candidate previous-state axes at the current state (effect TPM, Eq. 3).

	Args:
		factors: Per-unit conditionals of the whole system.
		state: Current state, shape ``(n,)``.
		candidate: Mask of the candidate system's units.

	Returns:
		Factors constant along background axes, equal to their value at the current
		background state.

	"""
	clamped = []
	for factor in factors:
		out = factor
		for axis in range(len(factors)):
			selected = jnp.take(out, state[axis], axis=axis)
			out = jnp.where(candidate[axis], out, jnp.expand_dims(selected, axis))
		clamped.append(out)
	return tuple(clamped)


def _backward_factors(
	factors: tuple[Float[Array, "*shape q"], ...],
	state: Int[Array, " n"],
	candidate: Bool[Array, " n"],
) -> tuple[Float[Array, "*shape q"], ...]:
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


def _sever(
	factors: tuple[Float[Array, "*shape q"], ...], cut: Bool[Array, "n n"]
) -> tuple[Float[Array, "*shape q"], ...]:
	"""Noise the connections a cut severs (Eqs. 17-18).

	Severed inputs are uniformly marginalized out of the receiving unit's conditional:
	unit ``j`` perceives each severed source as independent noise.

	Args:
		factors: Per-unit conditionals.
		cut: Cut matrix, shape ``(n, n)``; entry ``(i, j)`` severs the connection from
			unit ``i`` to unit ``j``.

	Returns:
		The partitioned factors.

	"""
	severed = []
	for j, factor in enumerate(factors):
		out = factor
		for axis in range(len(factors)):
			out = jnp.where(cut[axis, j], out.mean(axis=axis, keepdims=True), out)
		severed.append(out)
	return tuple(severed)


def _likelihood(
	factors: tuple[Float[Array, "*shape q"], ...],
	state: Int[Array, " n"],
	candidate: Bool[Array, " n"],
) -> Float[Array, "*shape"]:
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
	effect_factors: tuple[Float[Array, "*shape q"], ...],
	state: Int[Array, " n"],
	candidate: Bool[Array, " n"],
) -> Float[Array, "*shape"]:
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


def _unconstrained_effect(
	effect_factors: tuple[Float[Array, "*shape q"], ...], candidate: Bool[Array, " n"]
) -> Float[Array, "*shape"]:
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
	cause_factors: tuple[Float[Array, "*shape q"], ...],
	state: Int[Array, " n"],
	candidate: Bool[Array, " n"],
) -> Float[Array, "*shape"]:
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


def _specify(
	information: Float[Array, "*shape"], shape: tuple[int, ...]
) -> tuple[Int[Array, " n"], Float[Array, ""]]:
	"""Select the state maximizing an intrinsic-information tensor.

	Ties are broken by first occurrence in little-endian state order, the library's
	canonical order.

	Args:
		information: Full-shape intrinsic-information tensor.
		shape: Per-unit alphabet sizes.

	Returns:
		The maximizing state vector and its intrinsic information.

	"""
	flat = jnp.ravel(information, order="F")
	index = jnp.argmax(flat)
	state = (index // jnp.asarray(radix_weights(shape))) % jnp.asarray(shape)
	return state, flat[index]


def _at(
	full: Float[Array, "*shape"], state: Int[Array, " n"], shape: tuple[int, ...]
) -> Float[Array, ""]:
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


def _log2_ratio(p: Float[Array, ...], q: Float[Array, ...]) -> Float[Array, ...]:
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


def _quantize(x: Float[Array, ...]) -> Float[Array, ...]:
	"""Quantize values to the measure's precision before comparison.

	Ties are defined at :data:`PRECISION`, matching the oracle's round-then-compare
	semantics. Meaningful under float64; under float32 the quantization is a no-op.

	Args:
		x: Values to quantize.

	Returns:
		Values rounded to the nearest multiple of :data:`PRECISION`.

	"""
	return jnp.round(x / PRECISION) * PRECISION


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

	exists: Bool[Array, " D"]
	phi: Float[Array, " D"]
	mechanism: Bool[Array, "D n"]
	cause_purview: Bool[Array, "D n"]
	effect_purview: Bool[Array, "D n"]
	cause_state: Int[Array, "D n"]
	effect_state: Int[Array, "D n"]
	phi_cause: Float[Array, " D"]
	phi_effect: Float[Array, " D"]


def distinctions(
	system: System,
	state: Int[Array, " n"],
	candidate: tuple[int, ...] | None = None,
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

	Returns:
		The distinctions, stacked over the nonempty mechanisms of the candidate.

	"""
	units = tuple(range(system.n)) if candidate is None else tuple(sorted(candidate))
	n, shape = system.n, system.shape
	candidate_mask = jnp.zeros(n, dtype=bool).at[jnp.asarray(units)].set(True)

	factors = node_tpms(system, check_independence=False)
	effect_factors = _clamp_background(factors, state, candidate_mask)
	cause_factors = _backward_factors(factors, state, candidate_mask)
	ces = cause_effect_state(system, state, candidate_mask)

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

	results = []
	for mechanism_row in mechanism_table:
		mechanism_units = tuple(int(u) for u in np.flatnonzero(mechanism_row))
		mechanism_bitmask = int(sum(1 << u for u in mechanism_units))
		valid_mechanism = bool(np.all(~mechanism_row | np.asarray(candidate_mask)))
		if not valid_mechanism:
			results.append(None)
			continue

		# Padded partition tables across the purviews of this mechanism.
		partition_tables = [
			mechanism_partitions(
				mechanism_units, tuple(int(u) for u in np.flatnonzero(purview_row)), n
			)
			for purview_row in purview_table
		]
		max_partitions = max(len(entry[2]) for entry in partition_tables)
		co_mechanism = np.zeros((len(purview_table), max_partitions, n), dtype=np.int64)
		co_purview = np.zeros_like(co_mechanism)
		severed = np.ones((len(purview_table), max_partitions), dtype=np.int64)
		valid = np.zeros((len(purview_table), max_partitions), dtype=bool)
		for k, (a, b, s) in enumerate(partition_tables):
			co_mechanism[k, : len(s)] = a
			co_purview[k, : len(s)] = b
			severed[k, : len(s)] = s
			valid[k, : len(s)] = True

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
				mechanism_row: np.ndarray = mechanism_row,
				mechanism_bitmask: int = mechanism_bitmask,
				specified: jax.Array = specified,
			) -> tuple[jax.Array, jax.Array, jax.Array]:
				return _purview_phi(
					direction,
					jnp.asarray(mechanism_row),
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

			phi_z, congruent, chosen_state = jax.vmap(evaluate_purview, in_axes=(0, 0, 0, 0, 0))(
				jnp.asarray(purview_table),
				jnp.asarray(co_mechanism),
				jnp.asarray(co_purview),
				jnp.asarray(severed),
				jnp.asarray(valid),
			)
			# Restrict purviews to the candidate.
			inside = jnp.asarray(np.all(~purview_table | np.asarray(candidate_mask), axis=1))
			phi_z = jnp.where(inside, phi_z, -jnp.inf)
			best = _quantize(phi_z) >= _quantize(phi_z).max()
			# Among tied purviews, prefer the first whose state is congruent.
			preferred = best & congruent
			purview_index = jnp.where(preferred.any(), jnp.argmax(preferred), jnp.argmax(best))
			side[direction] = (
				phi_z[purview_index],
				jnp.asarray(purview_table)[purview_index],
				chosen_state[purview_index],
				preferred.any(),
			)

		phi_cause, cause_purview, cause_state_row, cause_congruent = side[Direction.CAUSE]
		phi_effect, effect_purview, effect_state_row, effect_congruent = side[Direction.EFFECT]
		phi_d = jnp.minimum(phi_cause, phi_effect)
		exists = (_quantize(phi_d) > 0.0) & cause_congruent & effect_congruent
		results.append(
			(
				exists,
				jnp.where(exists, phi_d, 0.0),
				jnp.asarray(mechanism_row),
				cause_purview,
				effect_purview,
				cause_state_row,
				effect_state_row,
				phi_cause,
				phi_effect,
			)
		)

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

	rows = [row if row is not None else zeros_row() for row in results]
	stacked = [jnp.stack(column) for column in zip(*rows, strict=True)]
	return Distinctions(
		exists=stacked[0],
		phi=stacked[1],
		mechanism=stacked[2],
		cause_purview=stacked[3],
		effect_purview=stacked[4],
		cause_state=stacked[5],
		effect_state=stacked[6],
		phi_cause=stacked[7],
		phi_effect=stacked[8],
	)


def _smear_axes(x: Float[Array, ...], keep: Bool[Array, " n"]) -> Float[Array, ...]:
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


def _at_prev(
	x: Float[Array, ...], state: Int[Array, " n"], shape: tuple[int, ...]
) -> Float[Array, ...]:
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
	mechanism: Bool[Array, " n"],
	mechanism_bitmask: int,
	purview: Bool[Array, " n"],
	co_mechanism: Int[Array, "P n"],
	co_purview: Int[Array, "P n"],
	severed: Int[Array, " P"],
	ok: Bool[Array, " P"],
	conditional_tables: tuple[Float[Array, "B q"], ...],
	marginal_tables: Float[Array, "n B *shape"],
	smear_tables: tuple[Float[Array, "B *shape q"], ...],
	specified: Int[Array, " n"],
	shape: tuple[int, ...],
) -> tuple[Float[Array, ""], Bool[Array, ""], Int[Array, " n"]]:
	"""Evaluate one mechanism-purview pair in one direction.

	Computes the intrinsic information of every purview state, the φ of every
	disintegrating partition at every state, the per-state minimum partition (with the
	oracle's φ ≤ 0 short-circuit), and the specified state — preferring, among φ-tied
	candidates, the state congruent with the system's specification.

	Args:
		direction: Temporal direction. Static.
		mechanism: Mechanism mask, shape ``(n,)``.
		mechanism_bitmask: The mechanism as a bitmask. Static.
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
					conditional_tables[j][mechanism_bitmask],
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
				smear_tables[j][mechanism_bitmask],
				shape + (1,) * j + (shape[j],) + (1,) * (n - 1 - j),
			)
			joint = joint * jnp.where(purview[j], aligned, 1.0)
		unconstrained = joint.mean(axis=tuple(range(n)))
		information = selectivity * _log2_ratio(value, unconstrained)

		def partitioned(row: Int[Array, " n"]) -> Float[Array, "*shape"]:
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

		def partitioned(row: Int[Array, " n"]) -> Float[Array, "*shape"]:
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
	nonpositive = (_quantize(flat_phi) <= 0.0) & ok[:, None]
	key_normalized = jnp.where(ok[:, None], _quantize(flat_phi / severed[:, None]), jnp.inf)
	minimal = key_normalized <= key_normalized.min(axis=0, keepdims=True)
	key_phi = jnp.where(minimal, _quantize(flat_phi), -jnp.inf)
	tied = minimal & (key_phi >= key_phi.max(axis=0, keepdims=True))
	choice = jnp.where(
		nonpositive.any(axis=0), jnp.argmax(nonpositive, axis=0), jnp.argmax(tied, axis=0)
	)
	mip_phi = jnp.take_along_axis(flat_phi, choice[None], axis=0)[0]

	# Specified state: ii-maximal states, then φ-maximal among them, preferring the
	# congruent state, then first occurrence in canonical order.
	flat_information = jnp.ravel(information, order="F")
	tied_information = _quantize(flat_information) >= _quantize(flat_information).max()
	key = jnp.where(tied_information, _quantize(mip_phi), -jnp.inf)
	candidates = tied_information & (key >= key.max())
	congruent_index = jnp.sum(specified * jnp.asarray(radix_weights(shape)))
	congruent = candidates[congruent_index]
	index = jnp.where(congruent, congruent_index, jnp.argmax(candidates))
	state = (index // jnp.asarray(radix_weights(shape))) % jnp.asarray(shape)
	return mip_phi[index], congruent, state


def _outer(vectors: list[Float[Array, " q"]], shape: tuple[int, ...]) -> Float[Array, "*shape"]:
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

	sum_phi: Float[Array, ""]
	count: Int[Array, ""]


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
	big_phi: Float[Array, ""]


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
	state: Int[Array, " n"],
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
	found = distinctions(system, state, candidate)
	related = relations(found, analysis.cause_effect_state, system.shape)
	return PhiStructure(
		system=analysis,
		distinctions=found,
		relations=related,
		big_phi=jnp.where(found.exists, found.phi, 0.0).sum() + related.sum_phi,
	)
