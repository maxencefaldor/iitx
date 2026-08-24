"""Tests for the system container and its factored view."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.states import all_states
from iitx.system import (
	System,
	connectivity,
	is_conditionally_independent,
	node_tpms,
	validate,
)


def basic_system() -> System:
	"""Return PyPhi's basic network, the canonical example of Oizumi et al. (2014).

	Three binary units wired so that unit 0 is ``OR(1, 2)``, unit 1 is ``COPY(2)``, and
	unit 2 is ``XOR(0, 1)``, in PyPhi's state-by-node form with rows in little-endian
	order.

	Returns:
		The three-unit binary system.

	"""
	tpm = jnp.asarray(
		[
			[0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0],
			[1.0, 0.0, 1.0],
			[1.0, 0.0, 0.0],
			[1.0, 1.0, 0.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 1.0],
			[1.0, 1.0, 0.0],
		]
	)
	cm = jnp.asarray([[0, 0, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
	return System.from_state_by_node(tpm, cm=cm)


def test_system_attributes() -> None:
	"""Test the derived size attributes of a system."""
	system = basic_system()
	assert system.n == 3
	assert system.shape == (2, 2, 2)
	assert system.num_states == 8
	assert system.tpm.shape == (8, 8)


def test_system_is_a_pytree_with_static_shape() -> None:
	"""Test that a system round-trips through jit with its shape as a static field."""
	system = basic_system()
	leaves, treedef = jax.tree_util.tree_flatten(system)
	assert len(leaves) == 2  # tpm and cm; shape is static

	@jax.jit
	def total(system: System) -> jax.Array:
		return system.tpm.sum() * system.n

	assert float(total(system)) == pytest.approx(8.0 * 3)
	assert jax.tree_util.tree_unflatten(treedef, leaves).shape == (2, 2, 2)


def test_systems_batch_under_vmap() -> None:
	"""Test that many systems are analyzed in one call by mapping over the TPM axis.

	Batching across systems is a core promise of the library, and it works because a
	system is a pytree whose alphabet sizes are static while its probabilities are leaves.
	"""
	systems = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), basic_system())

	@jax.jit
	@jax.vmap
	def row_sums(system: System) -> jax.Array:
		return system.tpm.sum(axis=1)

	np.testing.assert_allclose(np.asarray(row_sums(systems)), np.ones((2, 8)), atol=1e-6)


def test_from_state_by_node_matches_hand_computed_rows() -> None:
	"""Test the state-by-node conversion against hand-computed transitions.

	The basic network is deterministic, so every row of the state-by-state matrix is a
	one-hot vector on the successor state: unit 0 is ``OR(1, 2)``, unit 1 is ``COPY(2)``,
	and unit 2 is ``XOR(0, 1)``.
	"""
	system = basic_system()
	states = all_states(system.shape)
	tpm = np.asarray(system.tpm)

	for u, state in enumerate(states):
		a, b, c = (int(x) for x in state)
		successor = (int(b or c), c, a ^ b)
		v = int(np.flatnonzero((states == successor).all(axis=1))[0])
		np.testing.assert_allclose(tpm[u], np.eye(8)[v], atol=1e-6)


def test_node_tpms_round_trip() -> None:
	"""Test that the factored view reconstructs the system it came from."""
	system = basic_system()
	factors = node_tpms(system)
	assert [tuple(factor.shape) for factor in factors] == [(2, 2, 2, 2)] * 3
	np.testing.assert_allclose(
		np.asarray(System.from_node_tpms(factors).tpm), np.asarray(system.tpm), atol=1e-6
	)


def test_node_tpms_are_distributions() -> None:
	"""Test that each unit's conditional sums to 1 over its next values."""
	for factor in node_tpms(basic_system()):
		np.testing.assert_allclose(np.asarray(factor).sum(axis=-1), 1.0, atol=1e-6)


def test_conditionally_dependent_system_is_rejected() -> None:
	"""Test that a system whose units are not conditionally independent is refused.

	Two units that always transition together — never apart — cannot be described by
	independent per-unit conditionals, and marginalizing would silently replace the system
	with a different one.
	"""
	tpm = jnp.asarray(
		[
			[0.5, 0.0, 0.0, 0.5],
			[0.5, 0.0, 0.0, 0.5],
			[0.5, 0.0, 0.0, 0.5],
			[0.5, 0.0, 0.0, 0.5],
		]
	)
	system = System(tpm=tpm, shape=(2, 2))

	assert not is_conditionally_independent(system)
	with pytest.raises(ValueError, match="not conditionally independent"):
		node_tpms(system)

	# The check is what refuses; the marginals themselves are still computable.
	factors = node_tpms(system, check_independence=False)
	assert len(factors) == 2


def test_from_node_tpms_is_conditionally_independent_by_construction() -> None:
	"""Test that building from per-unit conditionals always yields an independent system."""
	rng = np.random.default_rng(0)
	shape = (3, 2)
	factors = tuple(
		jnp.asarray(probabilities / probabilities.sum(axis=-1, keepdims=True))
		for probabilities in (rng.random((*shape, q)) for q in shape)
	)
	system = System.from_node_tpms(factors)

	assert system.shape == shape
	assert is_conditionally_independent(system)
	np.testing.assert_allclose(np.asarray(system.tpm).sum(axis=1), 1.0, atol=1e-6)


def test_nonuniform_alphabets() -> None:
	"""Test a system whose units have different alphabet sizes.

	Coarse-graining produces multi-valued units even from binary micro-units, so
	heterogeneous alphabets are a first-class case, not an afterthought.
	"""
	shape = (3, 2)
	factors = (
		jnp.broadcast_to(jnp.asarray([0.2, 0.3, 0.5]), (*shape, 3)),
		jnp.broadcast_to(jnp.asarray([0.25, 0.75]), (*shape, 2)),
	)
	system = validate(System.from_node_tpms(factors))

	assert system.num_states == 6
	assert system.tpm.shape == (6, 6)
	# Independent units: the joint of the two marginals, in little-endian state order.
	np.testing.assert_allclose(
		np.asarray(system.tpm[0]),
		[0.2 * 0.25, 0.3 * 0.25, 0.5 * 0.25, 0.2 * 0.75, 0.3 * 0.75, 0.5 * 0.75],
		atol=1e-6,
	)


def test_connectivity_defaults_to_fully_connected() -> None:
	"""Test that an unspecified connectivity matrix means every unit inputs to every unit."""
	system = System(tpm=jnp.eye(4), shape=(2, 2))
	np.testing.assert_array_equal(connectivity(system), np.ones((2, 2), dtype=bool))
	np.testing.assert_array_equal(connectivity(basic_system()), np.asarray(basic_system().cm))


@pytest.mark.parametrize(
	("tpm", "shape", "match"),
	[
		(jnp.eye(8), (2, 2), "TPM of shape"),
		(jnp.zeros((4, 4)), (2, 2), "must sum to 1"),
		(jnp.full((4, 4), -0.25), (2, 2), "non-negative"),
		(jnp.eye(1), (), "at least one state"),
	],
)
def test_validate_rejects_malformed_systems(
	tpm: jax.Array, shape: tuple[int, ...], match: str
) -> None:
	"""Test that validation catches malformed systems at the library boundary."""
	with pytest.raises(ValueError, match=match):
		validate(System(tpm=tpm, shape=shape))


def test_validate_rejects_mismatched_connectivity() -> None:
	"""Test that connectivity must be square over the units."""
	system = System(tpm=jnp.eye(4), shape=(2, 2), cm=jnp.ones((3, 3), dtype=bool))
	with pytest.raises(ValueError, match="connectivity must have shape"):
		validate(system)


def test_validate_returns_the_system() -> None:
	"""Test that validation composes with construction."""
	system = basic_system()
	assert validate(system) is system
