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
from jaxtyping import Array, Bool, Float, Int

from iitx.direction import Direction
from iitx.enumeration import system_cuts
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
