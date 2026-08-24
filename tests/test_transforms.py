"""Tests for the macro transforms, against PyPhi's macro fixtures.

The transforms are tested as TPM-to-TPM functions (hand-derivable) and end-to-end: the
coarse-grained system reproduces PyPhi's ``macro`` fixture golden Φ, and the
black-boxed propagation-delay network reproduces the basic network's golden Φ.

PyPhi's ``MacroSubsystem`` analyses (the ``emergence`` golden numbers) apply cuts at the
*micro* level — a different, deferred analysis mode (``docs/notes/oracle-findings.md``);
here a transformed system is an ordinary system.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from iitx.measures import iit3
from iitx.states import all_states
from iitx.system import System, validate
from iitx.transforms import black_box, black_box_state, coarse_grain, coarse_grain_state

ATOL = 2e-6
"""Golden tolerance (see ``tests/measures/test_iit3.py``)."""


def macro_example_network() -> System:
	"""Return PyPhi's ``macro_network``: four units that gain Φ when coarse-grained.

	Each unit turns ON with probability 1 if the other block's two units are both ON,
	else 0.3.

	Returns:
		The micro system.

	"""
	on = np.full((16, 4), 0.3)
	for row, state in enumerate(all_states((2, 2, 2, 2))):
		if state[0] and state[1]:
			on[row, 2:] = 1.0
		if state[2] and state[3]:
			on[row, :2] = 1.0
	return System.from_state_by_node(jnp.asarray(on))


def propagation_delay_network() -> System:
	"""Return PyPhi's 8-unit propagation-delay fixture.

	The basic OR/COPY/XOR system with COPY gates as one-step delays on every
	connection: unit 0 = OR(5, 7), 1 = COPY(0), 2 = XOR(1, 6), 3 = COPY(2),
	4 = COPY(3), 5 = COPY(4), 6 = COPY(4), 7 = COPY(2).

	Returns:
		The micro system.

	"""
	on = np.zeros((256, 8))
	for row, s in enumerate(all_states((2,) * 8)):
		on[row] = [s[5] | s[7], s[0], s[1] ^ s[6], s[2], s[3], s[4], s[4], s[2]]
	return System.from_state_by_node(jnp.asarray(on))


class TestCoarseGrain:
	"""Coarse-graining as a pure TPM transform."""

	def test_macro_network_tpm(self) -> None:
		"""Test the fiber-averaged macro TPM of PyPhi's macro network.

		Blocks (0,1) and (2,3), each mapped to ON when both micro-units are ON: a
		macro-unit fires its partner with probability 1 - 0.7**2 = 0.51 ... exactly the
		hand-computed rows below (e.g. both-off row: (1 - 0.09)**2 = 0.8281 to stay
		both-off).
		"""
		grouping = (0, 0, 0, 1)
		macro = coarse_grain(macro_example_network(), ((0, 1), (2, 3)), (grouping, grouping))

		assert macro.shape == (2, 2)
		np.testing.assert_allclose(
			np.asarray(macro.tpm),
			[
				[0.8281, 0.0819, 0.0819, 0.0081],
				[0.0, 0.0, 0.91, 0.09],
				[0.0, 0.91, 0.0, 0.09],
				[0.0, 0.0, 0.0, 1.0],
			],
			atol=1e-12,
		)

	def test_macro_beats_micro(self) -> None:
		"""Test the golden Φ values on both sides of the coarse-graining.

		The micro network has Φ = 0.113889; the coarse-grained system is exactly
		PyPhi's ``macro`` fixture network, whose golden Φ is 0.86905.
		"""
		micro = macro_example_network()
		state = jnp.asarray([0, 0, 0, 0])
		grouping = (0, 0, 0, 1)
		macro = coarse_grain(micro, ((0, 1), (2, 3)), (grouping, grouping))
		macro_state = coarse_grain_state(micro, ((0, 1), (2, 3)), (grouping, grouping), state)

		assert float(iit3.system_phi(macro, macro_state).phi) == pytest.approx(0.86905, abs=ATOL)
		assert float(iit3.system_phi(micro, state).phi) == pytest.approx(0.113889, abs=ATOL)

	def test_non_binary_macro_units(self) -> None:
		"""Test that count-preserving groupings yield multi-valued macro units."""
		grouping = (0, 1, 1, 2)
		macro = coarse_grain(macro_example_network(), ((0, 1), (2, 3)), (grouping, grouping))

		assert macro.shape == (3, 3)
		validate(macro)

	def test_temporal_grain(self) -> None:
		"""Test that ``steps`` composes the micro dynamics before aggregation."""
		micro = macro_example_network()
		grouping = (0, 0, 0, 1)
		two_step = coarse_grain(micro, ((0, 1), (2, 3)), (grouping, grouping), steps=2)
		validate(two_step)
		# The two-step macro TPM is generally NOT the square of the one-step macro TPM
		# (aggregation and composition do not commute) — but rows still sum to one and
		# the deterministic all-on fiber is preserved.
		np.testing.assert_allclose(np.asarray(two_step.tpm[-1]), [0, 0, 0, 1], atol=1e-12)

	def test_state_mapping(self) -> None:
		"""Test the micro-to-macro state projection."""
		micro = macro_example_network()
		grouping = (0, 0, 0, 1)
		state = coarse_grain_state(
			micro, ((0, 1), (2, 3)), (grouping, grouping), jnp.asarray([1, 1, 1, 0])
		)
		np.testing.assert_array_equal(np.asarray(state), [1, 0])

	@pytest.mark.parametrize(
		("partition", "groupings", "match"),
		[
			(((0, 1),), ((0, 0, 0, 1),), "cover every unit"),
			(((0, 1), (1, 2, 3)), ((0, 0, 0, 1), (0,) * 8), "cover every unit"),
			(((0, 1), (2, 3)), ((0, 0, 1), (0, 0, 0, 1)), "joint states"),
			(((0, 1), (2, 3)), ((0, 0, 0, 2), (0, 0, 0, 1)), "must cover"),
		],
	)
	def test_invalid_mappings_are_rejected(
		self,
		partition: tuple[tuple[int, ...], ...],
		groupings: tuple[tuple[int, ...], ...],
		match: str,
	) -> None:
		"""Test that malformed partitions and groupings raise."""
		with pytest.raises(ValueError, match=match):
			coarse_grain(macro_example_network(), partition, groupings)


class TestBlackBox:
	"""Black-boxing as a pure TPM transform."""

	def test_propagation_delay_reduces_to_basic(self) -> None:
		"""Test that black-boxing the delay network recovers the basic system.

		Boxes (0,5,7), (3,4), (1,2,6) with outputs 0, 4, 2, over two micro-steps: every
		cross-box connection runs through an output, so the window dynamics is exactly
		the basic OR/COPY/XOR logic over the outputs (in ascending order 0, 2, 4):
		``0' = OR(2, 4)``, ``2' = XOR(0, 4)``, ``4' = COPY(2)``.
		"""
		macro = black_box(
			propagation_delay_network(),
			((0, 5, 7), (3, 4), (1, 2, 6)),
			((0,), (4,), (2,)),
			steps=2,
		)
		assert macro.shape == (2, 2, 2)

		expected = np.zeros((8, 3))
		for row, (a, x, c) in enumerate(all_states((2, 2, 2))):
			expected[row] = [x | c, a ^ c, x]
		np.testing.assert_allclose(
			np.asarray(macro.tpm),
			np.asarray(System.from_state_by_node(jnp.asarray(expected)).tpm),
			atol=1e-12,
		)

	def test_blackboxed_system_reproduces_basic_phi(self) -> None:
		"""Test Φ of the black-boxed system at the state mimicking basic (1,0,0).

		The macro system is the basic network up to relabeling, so its Φ is the golden
		2.3125.
		"""
		macro = black_box(
			propagation_delay_network(),
			((0, 5, 7), (3, 4), (1, 2, 6)),
			((0,), (4,), (2,)),
			steps=2,
		)
		state = black_box_state(((0,), (4,), (2,)), jnp.asarray([1, 0, 0, 0, 0, 0, 0, 0]))
		np.testing.assert_array_equal(np.asarray(state), [1, 0, 0])
		assert float(iit3.system_phi(macro, state).phi) == pytest.approx(2.3125, abs=ATOL)

	@pytest.mark.parametrize(
		("partition", "outputs", "match"),
		[
			(((0, 5, 7), (3, 4)), ((0,), (4,)), "cover every unit"),
			(((0, 5, 7), (3, 4), (1, 2, 6)), ((0,), (4,), ()), "at least one output"),
			(((0, 5, 7), (3, 4), (1, 2, 6)), ((0,), (4,), (5,)), "within the box"),
		],
	)
	def test_invalid_boxes_are_rejected(
		self,
		partition: tuple[tuple[int, ...], ...],
		outputs: tuple[tuple[int, ...], ...],
		match: str,
	) -> None:
		"""Test that malformed boxes and outputs raise."""
		with pytest.raises(ValueError, match=match):
			black_box(propagation_delay_network(), partition, outputs)
