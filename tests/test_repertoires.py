"""Tests for the repertoire algebra.

Two anchors: the worked repertoires of Oizumi et al. (2014) for the OR/AND/XOR example
system, and a brute-force reference implementation (explicit loops over states, straight
from the marginalize-then-multiply definitions) cross-checked on random systems, binary
and non-binary.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.direction import Direction
from iitx.repertoires import purview_distribution, repertoire
from iitx.states import all_states
from iitx.system import System, node_tpms


def paper_system() -> System:
	"""Return the example system of Oizumi et al. (2014), Fig. 1: A=OR(B,C), B=AND(A,C), C=XOR(A,B).

	Returns:
		The three-unit binary system (PyPhi's ``fig4`` network).

	"""
	states = all_states((2, 2, 2))
	on = np.zeros((8, 3))
	for row, (a, b, c) in enumerate(states):
		on[row] = (int(b or c), int(a and c), a ^ b)
	return System.from_state_by_node(jnp.asarray(on))


def reference_repertoire(
	system: System,
	state: np.ndarray,
	mechanism: np.ndarray,
	purview: np.ndarray,
	direction: Direction,
) -> np.ndarray:
	"""Compute a repertoire by explicit loops, straight from the definitions.

	Args:
		system: The system.
		state: Current state, shape ``(n,)``.
		mechanism: Mechanism mask, shape ``(n,)``.
		purview: Purview mask, shape ``(n,)``.
		direction: Temporal direction.

	Returns:
		Full-shape repertoire (uniform on non-purview axes), as a NumPy array.

	"""
	shape = system.shape
	factors = [np.asarray(factor, dtype=np.float64) for factor in node_tpms(system)]
	states = all_states(shape)
	n, num_states = len(shape), len(states)

	if direction is Direction.CAUSE:
		joint = np.ones(num_states)
		for i in range(n):
			if not mechanism[i]:
				continue
			# Likelihood of unit i's observed state for every previous state...
			likelihood = np.asarray([factors[i][tuple(previous)][state[i]] for previous in states])
			# ...uniformly averaged over the non-purview coordinates before the product:
			# group previous states by their purview coordinates and average each group.
			smeared = np.zeros(num_states)
			for v, group_state in enumerate(states):
				members = [
					u
					for u, other in enumerate(states)
					if all(other[k] == group_state[k] for k in range(n) if purview[k])
				]
				smeared[v] = likelihood[members].mean()
			joint = joint * smeared
		if joint.sum() > 0.0:
			joint = joint / joint.sum()
		# Restore the uniform embedding over non-purview axes.
		outside = [k for k in range(n) if not purview[k]]
		full = joint.reshape(shape, order="F")
		for axis in outside:
			full = np.broadcast_to(full.mean(axis=axis, keepdims=True), shape).copy()
		return full

	perturbed = [
		u
		for u in range(num_states)
		if all(states[u][k] == state[k] for k in range(n) if mechanism[k])
	]
	result = np.ones(shape)
	for j in range(n):
		if purview[j]:
			factor = np.zeros(shape[j])
			for v in range(shape[j]):
				factor[v] = np.mean([factors[j][tuple(states[u])][v] for u in perturbed])
		else:
			factor = np.full(shape[j], 1.0 / shape[j])
		reshape = [1] * n
		reshape[j] = shape[j]
		result = result * factor.reshape(reshape, order="F")
	return result


def masks(
	n: int, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
	"""Build mechanism and purview masks from unit index tuples.

	Args:
		n: Number of units.
		mechanism: Indices of the mechanism units.
		purview: Indices of the purview units.

	Returns:
		The two boolean masks of shape ``(n,)``.

	"""
	mechanism_mask = (
		jnp.zeros(n, dtype=bool).at[jnp.asarray(mechanism, dtype=int)].set(True)
		if mechanism
		else jnp.zeros(n, dtype=bool)
	)
	purview_mask = (
		jnp.zeros(n, dtype=bool).at[jnp.asarray(purview, dtype=int)].set(True)
		if purview
		else jnp.zeros(n, dtype=bool)
	)
	return mechanism_mask, purview_mask


class TestPaperExamples:
	"""Worked repertoires from Oizumi et al. (2014), system A=OR, B=AND, C=XOR, state 100."""

	def test_unconstrained_effect_repertoire_of_each_unit(self) -> None:
		"""Test the unconstrained next-value distributions: OR 0.75, AND 0.25, XOR 0.5 ON.

		Text S2, Eqs. S11-S12: the unconstrained effect repertoire is the fully perturbed
		one-step distribution, not uniform.
		"""
		system = paper_system()
		factors = node_tpms(system)
		state = jnp.asarray([1, 0, 0])
		empty = jnp.zeros(3, dtype=bool)

		for j, p_on in [(0, 0.75), (1, 0.25), (2, 0.5)]:
			_, purview = masks(3, (), (j,))
			full = repertoire(factors, state, empty, purview, Direction.EFFECT)
			# Sum out the two uniform non-purview axes to read the unit's ON probability.
			on = float(np.moveaxis(np.asarray(full), j, 0)[1].sum())
			assert on == pytest.approx(p_on, abs=1e-6)

	def test_cause_repertoire_of_a_over_full_purview(self) -> None:
		"""Test p(ABC_past | A=1): uniform over the six states with B or C on.

		Text S2, Eq. S2: Bayes' rule under a uniform interventional prior. A=OR(B,C) is on
		exactly when its inputs were not both off, so the repertoire is 1/6 on those six
		states and 0 on (0,0,0) and (1,0,0).
		"""
		system = paper_system()
		state = jnp.asarray([1, 0, 0])
		mechanism, purview = masks(3, (0,), (0, 1, 2))

		full = repertoire(node_tpms(system), state, mechanism, purview, Direction.CAUSE)
		flat = np.asarray(full).ravel(order="F")

		expected = np.full(8, 1 / 6)
		expected[0] = 0.0  # (0,0,0): B=C=0
		expected[1] = 0.0  # (1,0,0): B=C=0
		np.testing.assert_allclose(flat, expected, atol=1e-6)

	def test_cause_repertoire_marginalizes_to_subpurview(self) -> None:
		"""Test p(BC_past | A=1): the full-purview repertoire with A summed out.

		Text S2, Eq. S3: sub-purview repertoires marginalize the excluded units under the
		uniform perturbation.
		"""
		system = paper_system()
		state = jnp.asarray([1, 0, 0])
		mechanism, purview = masks(3, (0,), (1, 2))

		full = repertoire(node_tpms(system), state, mechanism, purview, Direction.CAUSE)
		# Sum out unit A (axis 0): the distribution over (B, C).
		over_bc = np.asarray(full).sum(axis=0)
		np.testing.assert_allclose(over_bc, [[0.0, 1 / 3], [1 / 3, 1 / 3]], atol=1e-6)

	def test_higher_order_cause_repertoire_is_renormalized_product(self) -> None:
		"""Test p(ABC_past | AB=10): the normalized product of the elementary repertoires.

		Text S2, Eq. S5 with the renormalization the paper leaves implicit: A=1 allows the
		six states with B or C on; B=0 (AND) forbids the two states with A=C=1; the
		product is uniform on the remaining four states.
		"""
		system = paper_system()
		state = jnp.asarray([1, 0, 0])
		mechanism, purview = masks(3, (0, 1), (0, 1, 2))

		full = repertoire(node_tpms(system), state, mechanism, purview, Direction.CAUSE)
		flat = np.asarray(full).ravel(order="F")

		allowed = {2, 3, 4, 6}  # (0,1,0), (1,1,0), (0,0,1), (0,1,1) little-endian
		expected = np.asarray([0.25 if v in allowed else 0.0 for v in range(8)])
		np.testing.assert_allclose(flat, expected, atol=1e-6)

	def test_empty_mechanism_cause_repertoire_is_uniform(self) -> None:
		"""Test that the unconstrained cause repertoire is uniform (Text S2, Eq. S10)."""
		system = paper_system()
		state = jnp.asarray([1, 0, 0])
		mechanism, purview = masks(3, (), (0, 1, 2))

		full = repertoire(node_tpms(system), state, mechanism, purview, Direction.CAUSE)
		np.testing.assert_allclose(np.asarray(full), np.full((2, 2, 2), 1 / 8), atol=1e-6)


class TestAgainstReference:
	"""Cross-check against the brute-force reference on random systems."""

	@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 2), (2, 3, 2)])
	@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
	def test_random_systems(self, shape: tuple[int, ...], direction: Direction) -> None:
		"""Test every mechanism/purview pair of a random system against the reference."""
		rng = np.random.default_rng(hash(shape) % 2**32)
		factors = tuple(
			jnp.asarray(p / p.sum(axis=-1, keepdims=True))
			for p in (rng.random((*shape, q)) for q in shape)
		)
		system = System.from_node_tpms(factors)
		n = len(shape)
		state = jnp.asarray([rng.integers(q) for q in shape])

		subsets = [(0,), (1,), (0, 1), tuple(range(n))]
		for mechanism_units in [(), *subsets]:
			for purview_units in subsets:
				mechanism, purview = masks(n, mechanism_units, purview_units)
				ours = np.asarray(
					repertoire(node_tpms(system), state, mechanism, purview, direction)
				)
				reference = reference_repertoire(
					system,
					np.asarray(state),
					np.asarray(mechanism),
					np.asarray(purview),
					direction,
				)
				np.testing.assert_allclose(ours, reference, atol=1e-5)

	def test_repertoires_are_normalized(self) -> None:
		"""Test that full-shape repertoires sum to one on both sides."""
		system = paper_system()
		state = jnp.asarray([1, 0, 0])
		for direction in Direction:
			for mech in [(), (0,), (0, 2)]:
				for purv in [(1,), (0, 1, 2)]:
					mechanism, purview = masks(3, mech, purv)
					full = repertoire(node_tpms(system), state, mechanism, purview, direction)
					assert float(np.asarray(full).sum()) == pytest.approx(1.0, abs=1e-6)


class TestTransformations:
	"""jit and vmap composition over mask tables."""

	def test_vmap_over_mechanism_purview_pairs(self) -> None:
		"""Test that one kernel evaluates a table of (mechanism, purview) pairs."""
		system = paper_system()
		factors = node_tpms(system)
		state = jnp.asarray([1, 0, 0])

		mechanisms = jnp.asarray([[True, False, False], [True, True, False], [False, False, True]])
		purviews = jnp.asarray([[True, True, True], [False, True, True], [True, True, False]])

		batched = jax.jit(
			jax.vmap(repertoire, in_axes=(None, None, 0, 0, None)),
			static_argnums=4,
		)(factors, state, mechanisms, purviews, Direction.CAUSE)

		assert batched.shape == (3, 2, 2, 2)
		for k in range(3):
			single = repertoire(factors, state, mechanisms[k], purviews[k], Direction.CAUSE)
			np.testing.assert_allclose(np.asarray(batched[k]), np.asarray(single), atol=1e-6)

	def test_purview_distribution_recovers_marginal(self) -> None:
		"""Test that summing out the uniform axes recovers the purview distribution."""
		system = paper_system()
		state = jnp.asarray([1, 0, 0])
		mechanism, purview = masks(3, (0,), (1, 2))

		full = repertoire(node_tpms(system), state, mechanism, purview, Direction.CAUSE)
		bare = purview_distribution(full, purview)
		np.testing.assert_allclose(np.asarray(bare)[0], np.asarray(full).sum(axis=0), atol=1e-6)
		np.testing.assert_allclose(np.asarray(bare)[0], np.asarray(bare)[1], atol=1e-6)
