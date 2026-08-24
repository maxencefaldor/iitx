"""The system under analysis: a transition probability matrix over discrete units.

A :class:`System` is the only input `iitx` accepts. It knows nothing about where it came
from — a cellular automaton, a logic circuit, a neural recording, a coarse-grained macro
level — which is what makes the library general (``docs/design.md`` §2, P8).

The canonical form is a **state-by-state** transition probability matrix over units with
arbitrary finite alphabets. Mechanism-level IIT additionally requires the units to be
*conditionally independent* given the previous state; that is a checked property here
(:func:`node_tpms`), never an unchecked assumption and never silently enforced by a lossy
conversion.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Num

from iitx.states import all_states

__all__ = ["System", "connectivity", "is_conditionally_independent", "node_tpms", "validate"]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class System:
	"""A finite discrete dynamical system, given by its transition probability matrix.

	The transition probability matrix is *interventional*: ``tpm[u, v]`` is
	``p(v | do(u))``, the probability of state ``v`` one step after the system is
	perturbed into state ``u``. It is not an observed frequency. Deterministic systems are
	the special case where every row is a one-hot vector.

	States are indexed in little-endian mixed-radix order (see :mod:`iitx.states`), so
	``tpm`` has shape ``(Q, Q)`` with ``Q = prod(shape)``.

	The constructor performs no validation, because it is also the pytree constructor and
	must accept tracers under ``jax.jit``. Call :func:`validate` at the boundary where a
	system enters the library from user code.

	Attributes:
		tpm: State-by-state transition probabilities, shape ``(Q, Q)``; rows are previous
			states, columns are next states, and each row sums to 1.
		shape: Per-unit alphabet sizes ``(q_0, ..., q_{n-1})``. Static: it is part of the
			jit cache key, since it fixes the shape of every distribution in the analysis.
		cm: Connectivity, shape ``(n, n)``, where ``cm[i, j]`` is true when unit ``i``
			inputs to unit ``j`` (PyPhi's convention). ``None`` means fully connected,
			including self-loops. Connectivity never changes a probability; it prunes
			purviews and detects reducible systems, so it is data, not decoration.

	"""

	tpm: Float[Array, "Q Q"]
	shape: tuple[int, ...] = dataclasses.field(metadata={"static": True})
	cm: Bool[Array, "n n"] | None = None

	@property
	def n(self) -> int:
		"""Number of units."""
		return len(self.shape)

	@property
	def num_states(self) -> int:
		"""Number of states of the whole system, ``Q = prod(shape)``."""
		return math.prod(self.shape)

	@classmethod
	def from_node_tpms(
		cls, node_tpms: tuple[Float[Array, "*shape q"], ...], cm: Bool[Array, "n n"] | None = None
	) -> System:
		"""Build a system from per-unit conditional distributions.

		This is the conditionally independent construction: the joint transition
		probability is the product of the units' conditionals, so the resulting system
		satisfies :func:`is_conditionally_independent` by construction.

		Args:
			node_tpms: One array per unit; unit ``i``'s array has shape ``(*shape, q_i)``
				and holds ``p(u_i' = v | u)`` over previous states ``u`` and next values
				``v``. Its last axis must sum to 1.
			cm: Connectivity, or ``None`` for fully connected.

		Returns:
			The system whose state-by-state matrix is the product of these conditionals.

		"""
		shape = tuple(int(node_tpm.shape[-1]) for node_tpm in node_tpms)
		if any(node_tpm.shape[:-1] != shape for node_tpm in node_tpms):
			msg = (
				"each node TPM must have shape (*shape, q_i) with shape the alphabet sizes "
				f"{shape}, got {[tuple(node_tpm.shape) for node_tpm in node_tpms]}"
			)
			raise ValueError(msg)

		num_states = math.prod(shape)
		states = all_states(shape)
		# tpm[u, v] = prod_i p(u_i' = states[v, i] | u), with u flattened little-endian.
		factors = [
			jnp.reshape(node_tpm, (num_states, shape[i]), order="F")[:, states[:, i]]
			for i, node_tpm in enumerate(node_tpms)
		]
		return cls(tpm=math.prod(factors[1:], start=factors[0]), shape=shape, cm=cm)

	@classmethod
	def from_state_by_node(
		cls, tpm: Float[Array, "Q n"], cm: Bool[Array, "n n"] | None = None
	) -> System:
		"""Build a binary system from a state-by-node matrix.

		The state-by-node form is PyPhi's most common input: ``tpm[u, i]`` is the
		probability that binary unit ``i`` is ON one step after the system is perturbed
		into state ``u``. It presumes conditional independence, since it specifies only
		the marginals.

		Args:
			tpm: Probabilities of each unit being ON, shape ``(Q, n)`` with ``Q = 2**n``;
				rows are previous states in little-endian order.
			cm: Connectivity, or ``None`` for fully connected.

		Returns:
			The equivalent system over ``n`` binary units.

		"""
		num_states, n = tpm.shape
		if num_states != 2**n:
			msg = (
				f"a state-by-node TPM over {n} binary units must have {2**n} rows, got {num_states}"
			)
			raise ValueError(msg)

		on = jnp.asarray(tpm)
		return cls.from_node_tpms(
			tuple(
				jnp.reshape(
					jnp.stack([1.0 - on[:, i], on[:, i]], axis=-1), (2,) * n + (2,), order="F"
				)
				for i in range(n)
			),
			cm=cm,
		)


def connectivity(system: System) -> Bool[Array, "n n"]:
	"""Return the system's connectivity matrix, resolving ``None`` to fully connected.

	Args:
		system: The system.

	Returns:
		Boolean array of shape ``(n, n)``; ``cm[i, j]`` is true when unit ``i`` inputs to
		unit ``j``.

	"""
	if system.cm is None:
		return jnp.ones((system.n, system.n), dtype=bool)
	return jnp.asarray(system.cm, dtype=bool)


def node_tpms(
	system: System, *, check_independence: bool = True, tolerance: float | None = None
) -> tuple[Float[Array, "*shape q"], ...]:
	"""Derive the per-unit conditional distributions of a system.

	This is the *factored view* that mechanism-level IIT requires: unit ``i``'s array holds
	``p(u_i' = v | u)``, obtained by marginalizing the state-by-state matrix over the other
	units' next states.

	The factored view describes the same dynamics as the state-by-state matrix only when
	the units are conditionally independent given the previous state. When they are not,
	marginalizing discards the dependencies, so `iitx` refuses rather than silently
	returning a different system — the failure PyPhi's ``convert`` module documents as a
	"danger". Model dependent units by adding hidden units, as both IIT papers prescribe.

	Args:
		system: The system.
		check_independence: Whether to verify that the units are conditionally
			independent. Disable only for systems already validated, and never with a
			state-by-state matrix of unknown provenance.
		tolerance: Absolute tolerance of the conditional-independence check, or ``None``
			to derive it from the TPM's dtype.

	Returns:
		One array per unit; unit ``i``'s array has shape ``(*shape, q_i)``, with the first
		``n`` axes indexing the previous state and the last axis the unit's next value.

	Raises:
		ValueError: If ``check_independence`` is set and the system's units are not
			conditionally independent.

	"""
	if check_independence and not is_conditionally_independent(system, tolerance=tolerance):
		msg = (
			"the units of this system are not conditionally independent given the previous "
			"state, so it has no factored (per-unit) form and mechanism-level IIT does not "
			"apply to it as given; model the dependencies with hidden units instead"
		)
		raise ValueError(msg)
	return _marginal_node_tpms(system)


def is_conditionally_independent(system: System, *, tolerance: float | None = None) -> bool:
	"""Test whether a system's units are conditionally independent given the previous state.

	Conditional independence means ``p(u' | u) = prod_i p(u_i' | u)`` for every pair of
	states. It is an assumption of both IIT 3.0 and IIT 4.0, and it is what allows every
	repertoire to be built from per-unit factors.

	Args:
		system: The system.
		tolerance: Absolute tolerance on each transition probability, or ``None`` to derive
			it from the TPM's dtype. Genuine conditional dependence is a gross effect, so
			the derived tolerance errs towards accepting rather than falsely refusing a
			system whose factorization is exact but for rounding.

	Returns:
		Whether the state-by-state matrix equals the product of its per-unit marginals.

	"""
	if tolerance is None:
		tolerance = _tolerance(system.tpm.dtype)
	states = all_states(system.shape)
	marginals = _marginal_node_tpms(system)
	num_states = system.num_states
	factors = [
		jnp.reshape(marginal, (num_states, system.shape[i]), order="F")[:, states[:, i]]
		for i, marginal in enumerate(marginals)
	]
	product = math.prod(factors[1:], start=factors[0])
	return bool(jnp.allclose(system.tpm, product, atol=tolerance, rtol=0.0))


def validate(system: System) -> System:
	"""Check that a system is well formed, and return it unchanged.

	Validation happens at the boundary between user code and the library, never inside a
	jitted kernel: the checks are data-dependent and raise Python exceptions.

	Args:
		system: The system to check.

	Returns:
		The same system, so this composes as ``system = validate(System(...))``.

	Raises:
		ValueError: If the alphabet sizes, matrix shapes, or probabilities are invalid.

	"""
	if not system.shape or any(q < 1 for q in system.shape):
		msg = f"every unit must have at least one state, got shape {system.shape}"
		raise ValueError(msg)

	expected = (system.num_states, system.num_states)
	if system.tpm.shape != expected:
		msg = (
			f"a system with shape {system.shape} needs a TPM of shape {expected}, "
			f"got {tuple(system.tpm.shape)}"
		)
		raise ValueError(msg)

	tpm = np.asarray(system.tpm)
	if not np.all(np.isfinite(tpm)) or tpm.min() < 0.0:
		msg = "transition probabilities must be finite and non-negative"
		raise ValueError(msg)
	if not np.allclose(tpm.sum(axis=1), 1.0, atol=_tolerance(tpm.dtype), rtol=0.0):
		msg = "each row of the TPM must sum to 1: rows are distributions over next states"
		raise ValueError(msg)

	if system.cm is not None and tuple(system.cm.shape) != (system.n, system.n):
		msg = f"connectivity must have shape {(system.n, system.n)}, got {tuple(system.cm.shape)}"
		raise ValueError(msg)

	return system


def _tolerance(dtype: np.dtype) -> float:
	"""Return the comparison tolerance appropriate to a floating-point dtype.

	`iitx` is dtype-polymorphic: it inherits the precision of the TPM it is given, so
	tolerances are derived rather than hard-coded. The square root of the machine epsilon
	is the standard choice — comfortably above the error accumulated by summing over a
	state space, and far below any genuine violation of the property being checked.

	Args:
		dtype: The dtype of the array being compared.

	Returns:
		Absolute tolerance: about ``3e-4`` for float32 and ``1e-8`` for float64.

	"""
	return float(np.sqrt(np.finfo(dtype).eps))


def _marginal_node_tpms(system: System) -> tuple[Num[Array, "*shape q"], ...]:
	"""Marginalize the state-by-state TPM into per-unit conditionals, without checking.

	Args:
		system: The system.

	Returns:
		One array of shape ``(*shape, q_i)`` per unit.

	"""
	states = all_states(system.shape)
	marginals = []
	for i, q in enumerate(system.shape):
		# Sum the next-state distribution over every next state in which unit i has value v.
		selector = jnp.asarray(states[:, i][:, None] == jnp.arange(q), dtype=system.tpm.dtype)
		marginals.append(jnp.reshape(system.tpm @ selector, (*system.shape, q), order="F"))
	return tuple(marginals)
