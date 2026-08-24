"""The measure interface: integrated information as a first-class, replaceable object.

A measure maps ``(system, state)`` to integrated-information results. Holding the
definition of Φ behind one protocol is what lets alternative definitions — theory
versions, distances, partition schemes — be expressed, compared, and swapped under one
interface (``docs/design.md`` §5).
"""

import dataclasses
from typing import Any, Protocol, runtime_checkable

import jax

from iitx.measures import iit3, iit4
from iitx.system import System

__all__ = ["IIT3", "IIT4", "Measure"]


@runtime_checkable
class Measure(Protocol):
	"""A measure of integrated information.

	Implementations are frozen dataclasses whose fields are the measure's configuration;
	both methods are pure and compose with ``jax.jit`` and ``jax.vmap`` (and, where the
	measure's mathematics allows, ``jax.grad``).
	"""

	def phi(self, system: System, state: jax.Array) -> jax.Array:
		"""Compute the measure's headline scalar for a system in a state.

		Args:
			system: The system.
			state: Current state, shape ``(n,)``.

		Returns:
			The scalar value (which quantity it is — φ_s, Φ — is the measure's to
			document).

		"""
		...

	def analyze(self, system: System, state: jax.Array) -> Any:
		"""Compute the measure's full result for a system in a state.

		Args:
			system: The system.
			state: Current state, shape ``(n,)``.

		Returns:
			The measure's result pytree.

		"""
		...


@dataclasses.dataclass(frozen=True)
class IIT4:
	"""IIT 4.0 in its canonical configuration (Albantakis et al. 2023).

	``phi`` is the system integrated information φ_s (existence); ``analyze`` unfolds
	the full Φ-structure (distinctions, relations, and Φ). Differentiable almost
	everywhere with respect to the TPM.

	Attributes:
		version: Theory version — ``"2023"`` (canonical) or ``"2026"`` (the Mayner et
			al. 2026 intrinsic-information cap on φ_s).

	"""

	version: str = "2023"

	def phi(self, system: System, state: jax.Array) -> jax.Array:
		"""Compute φ_s.

		Args:
			system: The system.
			state: Current state, shape ``(n,)``.

		Returns:
			System integrated information φ_s, in ibits (clamped at zero; see
			:class:`iitx.measures.iit4.SystemPhi`).

		"""
		return iit4.system_phi(system, state, version=self.version).phi

	def analyze(self, system: System, state: jax.Array) -> iit4.PhiStructure:
		"""Unfold the Φ-structure.

		Args:
			system: The system.
			state: Current state, shape ``(n,)``.

		Returns:
			The Φ-structure.

		"""
		return iit4.phi_structure(system, state, version=self.version)


@dataclasses.dataclass(frozen=True)
class IIT3:
	"""IIT 3.0 in its canonical configuration (Oizumi et al. 2014).

	``phi`` is big Φ — the integrated conceptual information at the minimal
	unidirectional cut; ``analyze`` returns the Φ analysis with the cause-effect
	structure. Exact EMDs are host linear programs, so this measure does not
	differentiate (it says so loudly under ``jax.grad``).
	"""

	def phi(self, system: System, state: jax.Array) -> jax.Array:
		"""Compute big Φ.

		Args:
			system: The system.
			state: Current state, shape ``(n,)``.

		Returns:
			Integrated conceptual information Φ, in bits.

		"""
		return iit3.system_phi(system, state).phi

	def analyze(self, system: System, state: jax.Array) -> iit3.SystemPhi:
		"""Compute the Φ analysis with its cause-effect structure.

		Args:
			system: The system.
			state: Current state, shape ``(n,)``.

		Returns:
			The Φ analysis.

		"""
		return iit3.system_phi(system, state)
