"""Numerical conventions shared by the measures."""

import jax
import jax.numpy as jnp

__all__ = ["quantize"]


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
