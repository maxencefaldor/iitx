"""IIT 3.0: concepts, cause-effect structures, and big Φ (Oizumi et al. 2014).

The 3.0 pipeline: freeze the background at the current state; for every mechanism and
purview compute cause and effect repertoires (Text S2, Eqs. S2-S9); small φ is the earth
mover's distance to the minimum-information bipartition (Eqs. 6-8), maximized over
purviews into the maximally irreducible cause and effect (MICE, Eq. 9); mechanisms with
φ > 0 are concepts, and together they form the cause-effect structure. Big Φ (Eq. 11) is
the extended EMD between the cause-effect structure and the structure recomputed under
the minimal unidirectional cut, transporting φ mass between concepts at
concept-distance cost, with destroyed mass absorbed by the null concept (Text S2).

The exact EMDs are host linear programs (:func:`iitx.distances.emd`), so this measure
composes with ``jit`` and ``vmap`` but **not** with ``grad`` — by design, and loudly
(``docs/design.md`` §8). Two oracle conventions matter: cause-side φ uses the full
Hamming EMD while effect-side φ uses the analytic per-unit form (they differ on
correlated repertoires), and every distance is rounded to the measure's precision (1e-6)
before any comparison, exactly as PyPhi rounds.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from iitx.direction import Direction
from iitx.distances import emd, hamming_matrix, marginal_emd
from iitx.enumeration import bipartitions, directed_bipartitions, subsets
from iitx.measures.common import quantize
from iitx.repertoires import condition, repertoire, sever
from iitx.system import System, node_tpms

__all__ = ["Concepts", "SystemPhi", "ces", "system_phi"]

PRECISION = 1e-6
"""Quantization applied to every distance before comparison (PyPhi's 3.0 ``PRECISION``)."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Concepts:
	"""A cause-effect structure: the concepts of a candidate system, as stacked arrays.

	Row ``k`` describes the mechanism whose mask is row ``k`` of
	``iitx.enumeration.subsets(n, nonempty=True)``; mechanisms with φ = 0 (or outside
	the candidate) have ``exists`` false.

	Attributes:
		exists: Which mechanisms are concepts (φ > 0 at precision), shape ``(D,)``.
		phi: Concept φ, ``min(phi_cause, phi_effect)``, shape ``(D,)``.
		mechanism: Mechanism masks, shape ``(D, n)``.
		cause_purview: Core cause purview masks, shape ``(D, n)``.
		effect_purview: Core effect purview masks, shape ``(D, n)``.
		cause_repertoire: Maximally irreducible cause repertoires, expanded over the
			full state space (uniform filler), shape ``(D, *shape)``.
		effect_repertoire: Maximally irreducible effect repertoires, expanded over the
			full state space (unconstrained-product filler), shape ``(D, *shape)``.
		phi_cause: Cause-side φ maxima, shape ``(D,)``.
		phi_effect: Effect-side φ maxima, shape ``(D,)``.

	"""

	exists: jax.Array
	phi: jax.Array
	mechanism: jax.Array
	cause_purview: jax.Array
	effect_purview: jax.Array
	cause_repertoire: jax.Array
	effect_repertoire: jax.Array
	phi_cause: jax.Array
	phi_effect: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SystemPhi:
	"""Big Φ and its minimal cut (Eq. 11).

	Attributes:
		phi: Integrated conceptual information Φ at the minimal unidirectional cut,
			rounded to the measure's precision (as the oracle reports it).
		cut_index: Index of the minimal cut in the canonical table
			(:func:`iitx.enumeration.directed_bipartitions`). Where the oracle's own
			tied-cut identity is nondeterministic under its parallel evaluation, iitx
			reports the first tied cut in canonical order.
		ces: The (uncut) cause-effect structure.

	"""

	phi: jax.Array
	cut_index: jax.Array
	ces: Concepts


def ces(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None = None,
) -> Concepts:
	"""Compute the cause-effect structure of a candidate system.

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Units of the candidate system (static); ``None`` means all units.

	Returns:
		The concepts, stacked over the nonempty mechanisms of the candidate.

	"""
	units = tuple(range(system.n)) if candidate is None else tuple(sorted(candidate))
	candidate_mask = _mask(system.n, units)
	factors = condition(
		node_tpms(system, check_independence=False), state, jnp.asarray(candidate_mask)
	)
	return _ces(factors, state, candidate_mask, system.shape)


def system_phi(
	system: System,
	state: jax.Array,
	candidate: tuple[int, ...] | None = None,
) -> SystemPhi:
	"""Compute big Φ of a candidate system.

	Evaluates every unidirectional cut: the cause-effect structure is recomputed under
	the cut and compared to the whole structure by the extended EMD; Φ is the minimum.

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Units of the candidate system (static); ``None`` means all units.

	Returns:
		The Φ analysis at the minimal cut, with the uncut cause-effect structure.

	"""
	units = tuple(range(system.n)) if candidate is None else tuple(sorted(candidate))
	n, shape = system.n, system.shape
	candidate_mask = _mask(n, units)
	factors = condition(
		node_tpms(system, check_independence=False), state, jnp.asarray(candidate_mask)
	)

	whole = _ces(factors, state, candidate_mask, shape)
	null_cause, null_effect = _null_repertoires(factors, shape)

	cuts = jnp.asarray(_candidate_cuts(n, units))

	def cut_distance(cut: jax.Array) -> jax.Array:
		partitioned = _ces(sever(factors, cut), state, candidate_mask, shape)
		return _ces_distance(whole, partitioned, null_cause, null_effect, shape)

	distances = quantize(jax.vmap(cut_distance)(cuts), PRECISION)
	index = jnp.argmin(distances)
	return SystemPhi(phi=distances[index], cut_index=index, ces=whole)


def _candidate_cuts(n: int, units: tuple[int, ...]) -> np.ndarray:
	"""Build the unidirectional cut matrices of a candidate system.

	Args:
		n: Number of units of the system.
		units: Units of the candidate system.

	Returns:
		Cut matrices of shape ``(num_cuts, n, n)`` severing connections among the
		candidate's units only.

	"""
	within = directed_bipartitions(len(units))
	cuts = np.zeros((len(within), n, n), dtype=bool)
	index = np.asarray(units)
	cuts[:, index[:, None], index[None, :]] = within
	return cuts


def _ces(
	factors: tuple[jax.Array, ...],
	state: jax.Array,
	candidate: np.ndarray,
	shape: tuple[int, ...],
) -> Concepts:
	"""Compute the cause-effect structure from background-conditioned factors.

	A pure function of the factors, so the same code evaluates the whole structure and
	every cut structure (under ``vmap`` over cut matrices).

	Args:
		factors: Background-conditioned (and possibly severed) per-unit conditionals.
		state: Current state, shape ``(n,)``.
		candidate: Mask of the candidate system's units (static, NumPy).
		shape: Per-unit alphabet sizes.

	Returns:
		The concepts.

	"""
	n = len(shape)
	cost = jnp.asarray(hamming_matrix(shape))
	mechanism_table = subsets(n, nonempty=True)
	purview_table = subsets(n, nonempty=True)
	inside = jnp.asarray(np.all(~purview_table | candidate, axis=1))

	# Unconstrained effect marginal of each unit (the expansion filler).
	unconstrained = tuple(factor.mean(axis=tuple(range(n))) for factor in factors)

	rows = []
	for mechanism_row in mechanism_table:
		mechanism_units = tuple(int(u) for u in np.flatnonzero(mechanism_row))
		valid_mechanism = bool(np.all(~mechanism_row | candidate))
		if not valid_mechanism:
			rows.append(None)
			continue

		# Padded bipartition tables across the purviews of this mechanism.
		tables = [
			bipartitions(mechanism_units, tuple(int(u) for u in np.flatnonzero(purview_row)), n)
			for purview_row in purview_table
		]
		max_partitions = max(len(entry[0]) for entry in tables)
		part_mechanism = np.zeros((len(purview_table), max_partitions, n), dtype=bool)
		part_purview = np.zeros_like(part_mechanism)
		valid = np.zeros((len(purview_table), max_partitions), dtype=bool)
		for k, (mask_m, mask_z) in enumerate(tables):
			part_mechanism[k, : len(mask_m)] = mask_m
			part_purview[k, : len(mask_m)] = mask_z
			valid[k, : len(mask_m)] = True

		side = {}
		for direction in Direction:

			def purview_phi(
				purview: jax.Array,
				first_mechanism: jax.Array,
				first_purview: jax.Array,
				ok: jax.Array,
				direction: Direction = direction,
				mechanism_row: np.ndarray = mechanism_row,
			) -> tuple[jax.Array, jax.Array]:
				mechanism = jnp.asarray(mechanism_row)
				whole = repertoire(factors, state, mechanism, purview, direction)

				def partition_phi(m1: jax.Array, z1: jax.Array, ok_one: jax.Array) -> jax.Array:
					part = _normalize(
						repertoire(factors, state, m1, z1, direction)
						* repertoire(factors, state, mechanism & ~m1, purview & ~z1, direction)
					)
					if direction is Direction.CAUSE:
						distance = emd(
							jnp.ravel(whole, order="F"), jnp.ravel(part, order="F"), cost
						)
					else:
						distance = marginal_emd(
							jnp.ravel(whole, order="F"), jnp.ravel(part, order="F"), shape
						)
					return jnp.where(ok_one, quantize(distance, PRECISION), jnp.inf)

				phis = jax.vmap(partition_phi)(first_mechanism, first_purview, ok)
				return jnp.min(phis), jnp.ravel(whole, order="F")

			phi_z, reps = jax.vmap(purview_phi)(
				jnp.asarray(purview_table),
				jnp.asarray(part_mechanism),
				jnp.asarray(part_purview),
				jnp.asarray(valid),
			)
			phi_z = jnp.where(inside, phi_z, -jnp.inf)
			# Core purview: maximal φ, ties to the larger purview, then first in order.
			sizes = jnp.asarray(purview_table.sum(axis=1))
			best = quantize(phi_z, PRECISION) >= quantize(phi_z, PRECISION).max()
			key_size = jnp.where(best, sizes, -1)
			tied = best & (key_size >= key_size.max())
			purview_index = jnp.argmax(tied)
			side[direction] = (
				phi_z[purview_index],
				jnp.asarray(purview_table)[purview_index],
				reps[purview_index],
			)

		phi_cause, cause_purview, cause_rep = side[Direction.CAUSE]
		phi_effect, effect_purview, effect_rep = side[Direction.EFFECT]

		# Expand the effect repertoire: replace the uniform filler on non-purview axes
		# with the unconstrained effect marginals (the cause filler is already uniform).
		effect_full = jnp.reshape(effect_rep, shape, order="F")
		for i in range(n):
			scale = (shape[i] * unconstrained[i]).reshape((1,) * i + (-1,) + (1,) * (n - 1 - i))
			effect_full = jnp.where(effect_purview[i], effect_full, effect_full * scale)
		effect_rep = jnp.ravel(effect_full, order="F")

		phi = jnp.minimum(phi_cause, phi_effect)
		rows.append(
			(
				quantize(phi, PRECISION) > 0.0,
				phi,
				jnp.asarray(mechanism_row),
				cause_purview,
				effect_purview,
				cause_rep,
				effect_rep,
				phi_cause,
				phi_effect,
			)
		)

	num_states = int(np.prod(shape))

	def zeros_row() -> tuple[jax.Array, ...]:
		return (
			jnp.asarray(False),
			jnp.asarray(0.0, dtype=factors[0].dtype),
			jnp.zeros(n, dtype=bool),
			jnp.zeros(n, dtype=bool),
			jnp.zeros(n, dtype=bool),
			jnp.zeros(num_states, dtype=factors[0].dtype),
			jnp.zeros(num_states, dtype=factors[0].dtype),
			jnp.asarray(0.0, dtype=factors[0].dtype),
			jnp.asarray(0.0, dtype=factors[0].dtype),
		)

	stacked = [
		jnp.stack(column)
		for column in zip(*[row if row is not None else zeros_row() for row in rows], strict=True)
	]
	return Concepts(
		exists=stacked[0],
		phi=jnp.where(stacked[0], stacked[1], 0.0),
		mechanism=stacked[2],
		cause_purview=stacked[3],
		effect_purview=stacked[4],
		cause_repertoire=stacked[5],
		effect_repertoire=stacked[6],
		phi_cause=stacked[7],
		phi_effect=stacked[8],
	)


def _null_repertoires(
	factors: tuple[jax.Array, ...], shape: tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
	"""Build the null concept's expanded repertoires.

	The null concept is the unconstrained cause-effect repertoire of the candidate:
	uniform on the cause side, the fully perturbed product on the effect side.

	Args:
		factors: Background-conditioned per-unit conditionals.
		shape: Per-unit alphabet sizes.

	Returns:
		Flat cause and effect null repertoires, shape ``(Q,)`` each.

	"""
	n = len(shape)
	num_states = int(np.prod(shape))
	cause = jnp.full(num_states, 1.0 / num_states, dtype=factors[0].dtype)
	effect_full = jnp.ones(shape, dtype=factors[0].dtype)
	for i, factor in enumerate(factors):
		marginal = factor.mean(axis=tuple(range(n)))
		effect_full = effect_full * marginal.reshape((1,) * i + (-1,) + (1,) * (n - 1 - i))
	return cause, jnp.ravel(effect_full, order="F")


def _ces_distance(
	whole: Concepts,
	partitioned: Concepts,
	null_cause: jax.Array,
	null_effect: jax.Array,
	shape: tuple[int, ...],
) -> jax.Array:
	"""Compute the extended EMD between two cause-effect structures (Text S2).

	The transport problem in concept space: φ mass moves between the structures'
	non-matching concepts at concept-distance cost (cause-side EMD plus effect-side EMD
	of the expanded repertoires), with within-structure moves blocked and the null
	concept absorbing the φ deficit on either side. Matching concepts (same mechanism, φ,
	and repertoires) transport for free and are excluded, as in the oracle.

	Args:
		whole: The uncut cause-effect structure.
		partitioned: The structure under a cut.
		null_cause: The null concept's cause repertoire, flat.
		null_effect: The null concept's effect repertoire, flat.
		shape: Per-unit alphabet sizes.

	Returns:
		The scalar distance (unquantized).

	"""
	cost = jnp.asarray(hamming_matrix(shape))
	num = whole.exists.shape[0]

	matched = (
		whole.exists
		& partitioned.exists
		& (quantize(whole.phi, PRECISION) == quantize(partitioned.phi, PRECISION))
		& jnp.all(jnp.abs(whole.cause_repertoire - partitioned.cause_repertoire) < 1e-9, axis=1)
		& jnp.all(jnp.abs(whole.effect_repertoire - partitioned.effect_repertoire) < 1e-9, axis=1)
	)
	unique_whole = whole.exists & ~matched
	unique_partitioned = partitioned.exists & ~matched

	def concept_distance(
		cause_a: jax.Array,
		effect_a: jax.Array,
		cause_b: jax.Array,
		effect_b: jax.Array,
	) -> jax.Array:
		return emd(cause_a, cause_b, cost) + emd(effect_a, effect_b, cost)

	pairwise = jax.vmap(
		lambda ca, ea: jax.vmap(lambda cb, eb, ca=ca, ea=ea: concept_distance(ca, ea, cb, eb))(
			partitioned.cause_repertoire, partitioned.effect_repertoire
		)
	)(whole.cause_repertoire, whole.effect_repertoire)
	to_null_whole = jax.vmap(lambda c, e: concept_distance(c, e, null_cause, null_effect))(
		whole.cause_repertoire, whole.effect_repertoire
	)
	to_null_partitioned = jax.vmap(lambda c, e: concept_distance(c, e, null_cause, null_effect))(
		partitioned.cause_repertoire, partitioned.effect_repertoire
	)

	# Distance matrix over [whole uniques | partitioned uniques | null]; within-structure
	# moves are blocked by a value above every pairwise distance, as in the oracle.
	pair_mask = unique_whole[:, None] & unique_partitioned[None, :]
	blocked = jnp.where(pair_mask, pairwise, 0.0).max() + 1.0
	size = 2 * num + 1
	matrix = jnp.full((size, size), blocked)
	matrix = matrix.at[:num, num : 2 * num].set(jnp.where(pair_mask, pairwise, blocked))
	matrix = matrix.at[num : 2 * num, :num].set(jnp.where(pair_mask.T, pairwise.T, blocked))
	matrix = matrix.at[-1, :num].set(jnp.where(unique_whole, to_null_whole, blocked))
	matrix = matrix.at[:num, -1].set(jnp.where(unique_whole, to_null_whole, blocked))
	matrix = matrix.at[-1, num : 2 * num].set(
		jnp.where(unique_partitioned, to_null_partitioned, blocked)
	)
	matrix = matrix.at[num : 2 * num, -1].set(
		jnp.where(unique_partitioned, to_null_partitioned, blocked)
	)
	matrix = matrix.at[-1, -1].set(0.0)

	phi_whole = jnp.where(unique_whole, whole.phi, 0.0)
	phi_partitioned = jnp.where(unique_partitioned, partitioned.phi, 0.0)
	deficit = phi_whole.sum() - phi_partitioned.sum()
	supply = jnp.concatenate([phi_whole, jnp.zeros(num), jnp.maximum(-deficit, 0.0)[None]])
	demand = jnp.concatenate([jnp.zeros(num), phi_partitioned, jnp.maximum(deficit, 0.0)[None]])
	total = supply.sum()
	scale = jnp.where(total > 0.0, total, 1.0)
	return jnp.where(total > 0.0, emd(supply / scale, demand / scale, matrix) * scale, 0.0)


def _mask(n: int, units: tuple[int, ...]) -> np.ndarray:
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


def _normalize(x: jax.Array) -> jax.Array:
	"""Normalize a non-negative tensor to sum 1, mapping the zero tensor to itself.

	Args:
		x: Non-negative tensor.

	Returns:
		``x / x.sum()``, or zeros when the total is zero.

	"""
	total = x.sum()
	return jnp.where(total > 0.0, x / jnp.where(total > 0.0, total, 1.0), 0.0)
