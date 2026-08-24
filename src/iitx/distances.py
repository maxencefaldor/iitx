"""Distances between repertoires.

Two families of comparison live here, one per theory version:

- The **intrinsic difference** of IIT 4.0 (Barbosa et al. 2020) — a pointwise, state-
  selective divergence that is ordinary tensor algebra, differentiable almost everywhere.
- The **earth mover's distance** of IIT 3.0 — the Wasserstein-1 distance under the Hamming
  ground metric. Its exact value is the solution of a linear program, which no fixed
  tensor program can express, so it is solved on the host by POT's exact network-simplex
  solver behind :func:`jax.pure_callback` (the same backend PyPhi 2.0 uses). It jits and
  vmaps, but it does not differentiate — by design, and loudly (``docs/design.md`` §8 and §9).

PyPhi's IIT 3.0 numbers depend on an asymmetry that is easy to miss: cause-side φ uses the
full EMD, while effect-side φ uses the analytic per-unit form :func:`marginal_emd`, valid
because effect repertoires are products. The two disagree on correlated distributions, so
both are provided and the 3.0 measure uses each on its own side (``docs/design.md`` §4.7).
"""

import jax
import jax.numpy as jnp
import numpy as np
import ot

from iitx.states import all_states

__all__ = ["emd", "hamming_matrix", "intrinsic_difference", "marginal_emd"]


def intrinsic_difference(p: jax.Array, q: jax.Array) -> jax.Array:
	"""Compute the intrinsic difference between two distributions.

	The intrinsic difference is ``max_v p[v] * log2(p[v] / q[v])`` — the unique measure
	satisfying causality, intrinsicality, and specificity (Barbosa et al. 2020, Theorem 1).
	The maximand factors into *selectivity* (``p[v]``, how much probability the state
	concentrates) times *informativeness* (``log2(p[v] / q[v])``, how far above chance it
	is). Its unit is the ibit.

	States with ``p[v] = 0`` contribute zero (the continuous extension ``f(0, q) = 0``). A
	state with ``p[v] > 0`` and ``q[v] = 0`` is outside the measure's domain and yields
	``inf`` — mathematically honest, and possible in deterministic systems; callers that
	quantize to a precision handle it there.

	Args:
		p: Constrained distribution (flat, any shape — the maximum is over all elements).
		q: Reference ("chance") distribution, same shape as ``p``.

	Returns:
		Scalar intrinsic difference in ibits. Differentiable in ``p`` and ``q`` except at
		ties of the maximum and on the boundary ``p[v] = 0``.

	"""
	safe_p = jnp.where(p > 0.0, p, 1.0)
	safe_q = jnp.where(p > 0.0, q, 1.0)
	return jnp.max(jnp.where(p > 0.0, p * jnp.log2(safe_p / safe_q), 0.0))


def hamming_matrix(shape: tuple[int, ...]) -> np.ndarray:
	"""Build the generalized Hamming ground metric over a system's states.

	The distance between two states is the number of units whose values differ — the sum
	of per-unit discrete metrics. For binary units this is the Hamming distance PyPhi
	uses; for non-binary units, where the IIT 3.0 paper leaves the ground metric to "an
	intrinsic property of the mechanisms", this is `iitx`'s declared convention
	(``docs/design.md`` §9).

	Args:
		shape: Per-unit alphabet sizes ``(q_0, ..., q_{n-1})``.

	Returns:
		Cost matrix of shape ``(Q, Q)`` with ``Q = prod(shape)``, indexed by state index on
		both axes, as a build-time NumPy constant.

	"""
	states = all_states(shape)
	return (states[:, None, :] != states[None, :, :]).sum(axis=-1).astype(np.float64)


def emd(p: jax.Array, q: jax.Array, cost: jax.Array) -> jax.Array:
	"""Compute the exact earth mover's distance between two distributions.

	This is the Wasserstein-1 distance: the minimum cost of transporting the probability
	mass of ``p`` onto ``q``, where moving mass between states costs the ground metric
	``cost``. Exactness matters — the oracle's numbers are exact-EMD numbers — so the
	transportation linear program is solved on the host by POT's network simplex via
	:func:`jax.pure_callback`.

	Composes with ``jit`` and ``vmap`` (batch members are solved sequentially on the
	host). Does **not** compose with ``grad``: JAX raises on differentiating a pure
	callback, which is the designed behaviour — exact EMD has no tensor-program gradient,
	and the differentiable alternatives are explicitly named approximations
	(``docs/design.md`` §8).

	Args:
		p: Source distribution over state indices, shape ``(Q,)``; must sum to 1.
		q: Target distribution over state indices, shape ``(Q,)``; must sum to 1.
		cost: Ground metric between states, shape ``(Q, Q)`` (see :func:`hamming_matrix`).

	Returns:
		Scalar transport cost, in the dtype of ``p``.

	"""

	def solve(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> np.ndarray:
		return np.asarray(
			ot.emd2(
				np.ascontiguousarray(p, dtype=np.float64),
				np.ascontiguousarray(q, dtype=np.float64),
				np.ascontiguousarray(cost, dtype=np.float64),
			),
			dtype=p.dtype,
		)

	return jnp.asarray(
		jax.pure_callback(
			solve,
			jax.ShapeDtypeStruct((), p.dtype),
			p,
			q,
			cost,
			vmap_method="sequential",
		)
	)


def marginal_emd(p: jax.Array, q: jax.Array, shape: tuple[int, ...]) -> jax.Array:
	"""Compute the EMD between two *product* distributions from their unit marginals.

	For product distributions, the Hamming-metric EMD separates across units into a sum of
	one-unit transport problems, and one-unit transport under the discrete metric is the
	total variation distance ``sum_v |p_i(v) - q_i(v)| / 2``. For binary units this is
	exactly PyPhi's analytic effect-side EMD ``sum_i |P1(i OFF) - P2(i OFF)|``.

	This is an in-graph, differentiable formula — no linear program — but it equals
	:func:`emd` only when both arguments are product distributions (effect repertoires
	are; cause repertoires generally are not), which is why IIT 3.0 uses it on the effect
	side only.

	Args:
		p: Source distribution over state indices, shape ``(Q,)``.
		q: Target distribution over state indices, shape ``(Q,)``.
		shape: Per-unit alphabet sizes, to unflatten the state axis.

	Returns:
		Scalar transport cost: the sum over units of the total variation between the
		marginals of ``p`` and ``q``.

	"""
	total = jnp.zeros((), dtype=p.dtype)
	for i in range(len(shape)):
		axes = tuple(axis for axis in range(len(shape)) if axis != i)
		marginal_p = jnp.reshape(p, shape, order="F").sum(axis=axes)
		marginal_q = jnp.reshape(q, shape, order="F").sum(axis=axes)
		total += jnp.abs(marginal_p - marginal_q).sum() / 2.0
	return total
