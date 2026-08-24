"""Repertoires: the distributions a mechanism specifies over a purview.

This is the shared kernel layer of the library (``docs/design.md`` §6.1): everything both
IIT versions compute is built from the repertoire algebra here, which in turn is nothing
but mask-parameterized tensor algebra over the factored view of a system
(:func:`iitx.system.node_tpms`).

**Full-shape embedding.** A repertoire over a purview ``Z`` is mathematically a
distribution over the states of ``Z`` alone, but `iitx` always represents it as a
full-shape tensor over the states of *all* units, with the units outside the purview
carried as an exactly uniform, independent factor (``docs/design.md`` §2, P3). Every
repertoire therefore has the same shape regardless of its purview, which is what lets one
compiled kernel ``vmap`` over tables of mechanism and purview masks. The embedding is
lossless — the purview distribution is recovered by summing out the uniform axes — and it
is transparent to the Hamming-metric EMD, which costs nothing for moving an identical
uniform factor onto itself.

**Semantics** (the conventions of both theories, fixed by the papers and the oracle):

- *Cause* (Bayes under a uniform interventional prior): each mechanism unit contributes
  the likelihood of its observed state as a function of the previous system state; each
  likelihood is uniformly marginalized over non-purview units *before* the product over
  mechanism units (the "virtual elements" / product-probability doctrine that severs
  common-input correlations); the product is then normalized.
- *Effect* (forward perturbation): each purview unit contributes its next-value
  distribution given the mechanism clamped to its observed state and every other unit
  uniformly perturbed; the repertoire is the product over purview units.
- The *unconstrained* repertoire is the same computation with an empty mechanism: uniform
  over the purview on the cause side, and the fully perturbed one-step distribution on
  the effect side (not uniform in general).
"""

import jax
import jax.numpy as jnp

from iitx.direction import Direction

__all__ = ["condition", "purview_distribution", "repertoire", "sever"]


def repertoire(
	node_tpms: tuple[jax.Array, ...],
	state: jax.Array,
	mechanism: jax.Array,
	purview: jax.Array,
	direction: Direction,
) -> jax.Array:
	"""Compute the repertoire a mechanism in a state specifies over a purview.

	Implements Eqs. S2-S9 of Oizumi et al. (2014) Text S2 — equivalently the product
	probabilities of Eqs. 29-30 of Albantakis et al. (2023) — in the marginalize-then-
	multiply form both theories mandate. An empty ``mechanism`` yields the unconstrained
	repertoire.

	Args:
		node_tpms: The factored view of the (already background-conditioned) system: one
			array of shape ``(*shape, q_i)`` per unit, ``p(u_i' = v | u)``.
		state: Current state of the system, shape ``(n,)``.
		mechanism: Mask of the mechanism units, shape ``(n,)``.
		purview: Mask of the purview units, shape ``(n,)``.
		direction: ``CAUSE`` for the distribution over previous states, ``EFFECT`` for the
			distribution over next states. Static.

	Returns:
		Full-shape tensor of shape ``shape``: the purview distribution with an exactly
		uniform factor on non-purview axes. Sums to 1, except for a cause repertoire whose
		mechanism state is unreachable within the purview, which is identically zero (the
		convention under which its φ is zero, matching PyPhi).

	"""
	shape = tuple(node_tpm.shape[-1] for node_tpm in node_tpms)
	if direction is Direction.CAUSE:
		joint = jnp.ones(shape, dtype=node_tpms[0].dtype)
		for i, node_tpm in enumerate(node_tpms):
			# Likelihood of unit i's observed state as a function of the previous state,
			# uniformly marginalized over non-purview axes *before* the product (virtual
			# elements); units outside the mechanism contribute the identity.
			likelihood = jnp.take_along_axis(
				node_tpm, state[i].reshape((1,) * len(shape) + (1,)), axis=-1
			).squeeze(-1)
			joint = joint * jnp.where(mechanism[i], _smear(likelihood, purview), 1.0)
		return _normalize(joint * _uniform(shape, purview, joint.dtype))

	joint = jnp.ones(shape, dtype=node_tpms[0].dtype)
	for j, node_tpm in enumerate(node_tpms):
		# Next-value distribution of purview unit j with the mechanism clamped and every
		# other unit uniformly perturbed; non-purview units contribute uniform.
		conditioned = node_tpm
		for i in range(len(shape)):
			weights = jnp.where(
				mechanism[i],
				jnp.arange(shape[i]) == state[i],
				jnp.full(shape[i], 1.0 / shape[i]),
			).astype(node_tpm.dtype)
			# The leading axis is always the lowest not-yet-contracted previous-state axis.
			conditioned = jnp.tensordot(conditioned, weights, axes=([0], [0]))
		factor = jnp.where(purview[j], conditioned, jnp.full(shape[j], 1.0 / shape[j]))
		joint = joint * factor.reshape((1,) * j + (-1,) + (1,) * (len(shape) - 1 - j))
	return joint


def purview_distribution(full: jax.Array, purview: jax.Array) -> jax.Array:
	"""Recover the bare purview distribution from a full-shape repertoire.

	Sums out the uniform non-purview axes (keeping dimensions, so the shape is stable
	under ``vmap``): the result is the distribution over purview states, broadcast along
	singleton-like constant axes.

	Args:
		full: Full-shape repertoire, shape ``shape``.
		purview: Mask of the purview units, shape ``(n,)``.

	Returns:
		Tensor of shape ``shape``, constant along non-purview axes, whose values are the
		purview-state probabilities.

	"""
	out = full
	for axis in range(full.ndim):
		out = jnp.where(purview[axis], out, out.sum(axis=axis, keepdims=True))
	return out


def _smear(x: jax.Array, mask: jax.Array) -> jax.Array:
	"""Replace each non-mask axis of ``x`` by its uniform mean, keeping dimensions.

	This is uniform marginalization in place: after smearing, ``x`` is constant along
	every axis outside ``mask``.

	Args:
		x: Full-shape tensor.
		mask: Mask of the axes to keep, shape ``(n,)``.

	Returns:
		Tensor of the same shape, constant along non-mask axes.

	"""
	out = x
	for axis in range(x.ndim):
		out = jnp.where(mask[axis], out, out.mean(axis=axis, keepdims=True))
	return out


def _uniform(shape: tuple[int, ...], purview: jax.Array, dtype: jnp.dtype) -> jax.Array:
	"""Build the uniform factor over non-purview axes (ones along purview axes).

	Args:
		shape: Per-unit alphabet sizes.
		purview: Mask of the purview units.
		dtype: Dtype of the result.

	Returns:
		Full-shape tensor equal to the product over non-purview axes of ``1 / q_i``.

	"""
	out = jnp.ones(shape, dtype=dtype)
	for axis, q in enumerate(shape):
		out = out * jnp.where(purview[axis], 1.0, 1.0 / q)
	return out


def _normalize(x: jax.Array) -> jax.Array:
	"""Normalize a non-negative tensor to sum 1, mapping the zero tensor to itself.

	Args:
		x: Non-negative tensor.

	Returns:
		``x / x.sum()``, or zeros when the total is zero (an unreachable mechanism state;
		the caller's φ is zero by convention).

	"""
	total = x.sum()
	return jnp.where(total > 0.0, x / jnp.where(total > 0.0, total, 1.0), 0.0)


def condition(
	node_tpms: tuple[jax.Array, ...],
	state: jax.Array,
	candidate: jax.Array,
) -> tuple[jax.Array, ...]:
	"""Clamp non-candidate previous-state axes of the factors at the current state.

	This is the frozen-background conditioning shared by both theory versions on the
	effect side (and by IIT 3.0 on the cause side): units outside the candidate system
	are fixed at their current state. The clamped factors are constant along background
	axes, so downstream uniform marginalization leaves them untouched.

	Args:
		node_tpms: Per-unit conditionals of the whole system.
		state: Current state of the whole system, shape ``(n,)``.
		candidate: Mask of the candidate system's units, shape ``(n,)``.

	Returns:
		Factors constant along background axes, equal to their value at the current
		background state.

	"""
	clamped = []
	for factor in node_tpms:
		out = factor
		for axis in range(len(node_tpms)):
			selected = jnp.take(out, state[axis], axis=axis)
			out = jnp.where(candidate[axis], out, jnp.expand_dims(selected, axis))
		clamped.append(out)
	return tuple(clamped)


def sever(node_tpms: tuple[jax.Array, ...], cut: jax.Array) -> tuple[jax.Array, ...]:
	"""Noise the connections a cut severs.

	Severed inputs are uniformly marginalized out of the receiving unit's conditional,
	so each unit perceives its severed sources as independent noise — the cut semantics
	of both theory versions (IIT 3.0 unidirectional cuts and IIT 4.0 directional
	partitions differ only in which cut matrices are enumerated).

	Args:
		node_tpms: Per-unit conditionals.
		cut: Cut matrix, shape ``(n, n)``; entry ``(i, j)`` severs the connection from
			unit ``i`` to unit ``j``.

	Returns:
		The partitioned factors.

	"""
	severed = []
	for j, factor in enumerate(node_tpms):
		out = factor
		for axis in range(len(node_tpms)):
			out = jnp.where(cut[axis, j], out.mean(axis=axis, keepdims=True), out)
		severed.append(out)
	return tuple(severed)
