"""States of a system, and the mixed-radix indexing convention.

A system of ``n`` units has a ``shape``: the tuple of per-unit alphabet sizes
``(q_0, ..., q_{n-1})``. A *state* assigns one value to each unit and is represented as an
integer vector of length ``n``. A *state index* is the position of that state in the
canonical enumeration of all ``Q = prod(shape)`` states.

The convention throughout `iitx` is **little-endian mixed radix**: unit 0 is the
fastest-varying digit. For binary units this is PyPhi's LOLI ("low-order bits correspond to
low-index nodes") convention, generalized to arbitrary alphabets::

	shape (2, 2, 2):  index 0 -> (0, 0, 0),  index 1 -> (1, 0, 0),  index 3 -> (1, 1, 0)
	shape (3, 2):     index 0 -> (0, 0),     index 1 -> (1, 0),     index 4 -> (1, 1)

The same convention orders the axes of every full state-space tensor: axis ``i`` is unit
``i``, and since unit 0 varies fastest, a distribution of shape ``shape`` flattens to a
distribution over state indices with ``jnp.ravel(p, order="F")`` — Fortran order is what
encodes little-endian indexing, as in PyPhi's ``convert`` module. The state index of a
state vector is :func:`ravel_state`.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["all_states", "radix_weights", "ravel_state", "unravel_state"]


def radix_weights(shape: tuple[int, ...]) -> np.ndarray:
	"""Return the place values of the little-endian mixed-radix encoding.

	The state index of a state ``s`` is ``sum(s[i] * weights[i])``.

	Args:
		shape: Per-unit alphabet sizes ``(q_0, ..., q_{n-1})``.

	Returns:
		Place values ``(1, q_0, q_0 * q_1, ...)`` of shape ``(n,)``, as a NumPy array of
		Python integers (these are static structure, not traced values).

	"""
	sizes = np.asarray(shape, dtype=np.int64)
	return np.concatenate([np.ones(1, dtype=np.int64), np.cumprod(sizes)[:-1]])


def ravel_state(state: jax.Array, shape: tuple[int, ...]) -> jax.Array:
	"""Convert a state vector to its little-endian mixed-radix state index.

	Args:
		state: State vector of shape ``(n,)``; ``state[i]`` is in ``range(shape[i])``.
		shape: Per-unit alphabet sizes.

	Returns:
		Scalar state index in ``range(prod(shape))``.

	"""
	return jnp.sum(state * jnp.asarray(radix_weights(shape)))


def unravel_state(index: jax.Array, shape: tuple[int, ...]) -> jax.Array:
	"""Convert a little-endian mixed-radix state index to its state vector.

	Args:
		index: Scalar state index in ``range(prod(shape))``.
		shape: Per-unit alphabet sizes.

	Returns:
		State vector of shape ``(n,)``.

	"""
	return (index // jnp.asarray(radix_weights(shape))) % jnp.asarray(shape)


def all_states(shape: tuple[int, ...]) -> np.ndarray:
	"""Enumerate every state of a system, in canonical (little-endian) order.

	Row ``k`` of the result is the state whose state index is ``k``, so this table is the
	inverse of :func:`ravel_state` and doubles as the gather table for conditioning
	distributions on states.

	Args:
		shape: Per-unit alphabet sizes ``(q_0, ..., q_{n-1})``.

	Returns:
		Array of shape ``(Q, n)`` with ``Q = prod(shape)``, as a NumPy array: state tables
		are enumerated once at build time and closed over as compile-time constants.

	"""
	indices = np.arange(int(np.prod(shape)), dtype=np.int64)
	return (indices[:, None] // radix_weights(shape)) % np.asarray(shape, dtype=np.int64)
