"""Named, differentiable relaxations of integrated-information quantities.

Everything in this module is an **approximation** and says so (``docs/design.md`` §13):
where the exact quantities involve hard minima over enumerated partitions (piecewise
smooth, with subgradients at the seams), these surrogates replace the selection with a
temperature-controlled soft version — smooth everywhere, converging to the exact value
as the temperature goes to zero. Use them for optimization landscapes; report exact
values from the un-relaxed measures.
"""

import jax
import jax.numpy as jnp

from iitx.measures import iit4
from iitx.system import System

__all__ = ["soft_system_phi"]


def soft_system_phi(
	system: System,
	state: jax.Array,
	temperature: float = 0.1,
	candidate: tuple[int, ...] | None = None,
) -> jax.Array:
	"""Approximate IIT 4.0's signed φ_s with a smooth soft-minimum over partitions.

	The exact φ_s takes the φ of the partition minimizing normalized φ; this surrogate
	weights every partition by ``softmax(-normalized φ / temperature)`` and returns the
	weighted mean of the (signed, unclamped) φ values. As ``temperature`` → 0 it
	converges to the exact signed φ_s wherever the minimum partition is unique; at any
	positive temperature it is smooth everywhere — no subgradient seams at partition
	switches — which is what a gradient-ascent loop over TPMs wants.

	The φ evaluation itself (specified states, per-cut φ) is the exact pipeline of
	:func:`iitx.measures.iit4.system_phi`; only the partition *selection* is softened,
	and the reducibility short-circuit is not applied (the soft value can be negative,
	like the exact signed φ).

	Args:
		system: The system.
		state: Current state of the whole system, shape ``(n,)``.
		temperature: Softmin temperature, in ibits per severed connection. Zero is not
			allowed (use the exact measure for hard selection).
		candidate: Units of the candidate system (static); ``None`` means all units.

	Returns:
		Scalar smooth surrogate of the signed φ_s, in ibits.

	"""
	values = iit4.partition_phis(system, state, candidate)
	normalized = values.phi / values.severed
	weights = jax.nn.softmax(-normalized / temperature)
	return jnp.sum(weights * values.phi)
