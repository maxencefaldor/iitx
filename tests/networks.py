"""Canonical example systems shared across the test suite.

These are the oracle's fixture networks (PyPhi's ``examples`` module and the papers'
worked systems), in `iitx` form. State-by-node rows are little-endian, as everywhere.
"""

import jax.numpy as jnp
import numpy as np

from iitx.states import all_states
from iitx.system import System


def basic_network() -> System:
	"""Return PyPhi's basic network: unit 0 = OR(1, 2), unit 1 = COPY(2), unit 2 = XOR(0, 1).

	The standard three-unit example (state ``(1, 0, 0)`` in most fixtures).

	Returns:
		The system.

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


def basic_noisy_selfloop_network() -> System:
	"""Return PyPhi's basic network with self-loops and ε = 0.1 input noise.

	The canonical stochastic fixture (golden IIT 4.0 φ_s is *negative* for state
	``(1, 0, 0)``).

	Returns:
		The system.

	"""
	tpm = jnp.asarray(
		[
			[0.271, 0.19, 0.244],
			[0.919, 0.19, 0.756],
			[0.919, 0.91, 0.756],
			[0.991, 0.91, 0.244],
			[0.919, 0.91, 0.756],
			[0.991, 0.91, 0.244],
			[0.991, 0.99, 0.244],
			[0.999, 0.99, 0.756],
		]
	)
	cm = jnp.asarray([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
	return System.from_state_by_node(tpm, cm=cm)


def xor_network() -> System:
	"""Return PyPhi's XOR network: each unit is the XOR of the other two.

	Golden IIT 4.0 values for state ``(0, 0, 0)``: φ_s = 1.5 at the singleton
	tripartition.

	Returns:
		The system.

	"""
	states = all_states((2, 2, 2))
	on = np.zeros((8, 3))
	for row, (a, b, c) in enumerate(states):
		on[row] = (b ^ c, a ^ c, a ^ b)
	cm = jnp.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
	return System.from_state_by_node(jnp.asarray(on), cm=cm)


def paper_network() -> System:
	"""Return the 2014-paper example: unit 0 = OR(1, 2), unit 1 = AND(0, 2), unit 2 = XOR(0, 1).

	PyPhi's ``fig4`` network — the system behind the worked IIT 3.0 numbers (state
	``(1, 0, 0)``) and the IIT 4.0 fixture ``fig4`` (state ``(1, 0, 1)``, φ_s = 0).

	Returns:
		The system.

	"""
	states = all_states((2, 2, 2))
	on = np.zeros((8, 3))
	for row, (a, b, c) in enumerate(states):
		on[row] = (int(b or c), int(a and c), a ^ b)
	cm = jnp.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
	return System.from_state_by_node(jnp.asarray(on), cm=cm)
