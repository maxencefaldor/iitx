"""Tests for the mixed-radix state indexing convention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.states import all_states, radix_weights, ravel_state, unravel_state


@pytest.mark.parametrize(
	("shape", "expected"),
	[
		((2, 2, 2), [1, 2, 4]),
		((3, 2), [1, 3]),
		((2, 3, 4), [1, 2, 6]),
		((5,), [1]),
	],
)
def test_radix_weights(shape: tuple[int, ...], expected: list[int]) -> None:
	"""Test that place values are the cumulative products of the alphabet sizes."""
	np.testing.assert_array_equal(radix_weights(shape), expected)


def test_all_states_is_little_endian() -> None:
	"""Test that unit 0 varies fastest, matching PyPhi's LOLI convention."""
	np.testing.assert_array_equal(
		all_states((2, 2, 2)),
		[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
	)


def test_all_states_nonuniform_alphabets() -> None:
	"""Test enumeration over units with different alphabet sizes."""
	np.testing.assert_array_equal(
		all_states((3, 2)), [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
	)


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 2), (2, 3, 4), (5,)])
def test_ravel_unravel_round_trip(shape: tuple[int, ...]) -> None:
	"""Test that ravel and unravel invert each other over the whole state space."""
	states = all_states(shape)
	for index, state in enumerate(states):
		assert int(ravel_state(jnp.asarray(state), shape)) == index
		np.testing.assert_array_equal(unravel_state(jnp.asarray(index), shape), state)


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 4)])
def test_all_states_matches_tensor_axes(shape: tuple[int, ...]) -> None:
	"""Test that the state table indexes a full state-space tensor consistently.

	A tensor of shape ``shape`` flattened with C-order ``reshape`` must agree with the
	state table, so that a distribution's axes and its flat state indices are the same
	convention.
	"""
	tensor = np.arange(int(np.prod(shape))).reshape(shape, order="F")
	for index, state in enumerate(all_states(shape)):
		assert tensor[tuple(state)] == index


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 2)])
def test_state_conversions_are_jittable_and_vmappable(shape: tuple[int, ...]) -> None:
	"""Test that state conversions compose with jit and vmap."""
	states = jnp.asarray(all_states(shape))
	indices = jax.jit(jax.vmap(ravel_state, in_axes=(0, None)), static_argnums=1)(states, shape)
	np.testing.assert_array_equal(indices, np.arange(len(states)))

	round_trip = jax.jit(jax.vmap(unravel_state, in_axes=(0, None)), static_argnums=1)(
		indices, shape
	)
	np.testing.assert_array_equal(round_trip, states)
