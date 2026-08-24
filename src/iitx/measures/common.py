"""Numerical conventions shared by the measures."""

import jax
import jax.numpy as jnp

__all__ = ["quantize", "strongly_connected"]


def quantize(x: jax.Array, precision: float) -> jax.Array:
	"""Quantize values to a measure's precision before comparison.

	Every selection (minimum partition, maximal purview, specified state) compares
	values rounded to the measure's precision, so ties are defined at that precision —
	the oracle's round-then-compare semantics. Reported values stay unrounded.
	Meaningful under float64; under float32 coarse precisions still apply and fine ones
	degrade to a no-op.

	Args:
		x: Values to quantize.
		precision: The measure's precision (1e-6 for IIT 3.0, 1e-13 for IIT 4.0).

	Returns:
		Values rounded to the nearest multiple of ``precision``.

	"""
	return jnp.round(x / precision) * precision


def strongly_connected(cm: jax.Array, candidate: jax.Array) -> jax.Array:
	"""Test strong connectivity of a candidate's units, in-graph.

	A candidate that is not strongly connected has a part with no causes or no effects
	in the rest, so its integrated information is null by definition — both theory
	versions short-circuit on it.

	Args:
		cm: Connectivity matrix, shape ``(n, n)`` boolean.
		candidate: Mask of the candidate's units, shape ``(n,)``.

	Returns:
		Boolean scalar: whether every candidate unit reaches every other along directed
		connections within the candidate.

	"""
	n = cm.shape[0]
	inside = candidate[:, None] & candidate[None, :]
	sub = (cm & inside) | jnp.eye(n, dtype=bool)
	reach = sub
	for _ in range(n):
		reach = reach | (jnp.matmul(reach, sub) > 0)
	return jnp.all(reach | ~inside)
