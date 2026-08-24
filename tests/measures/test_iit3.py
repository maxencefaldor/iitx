"""Tests for IIT 3.0 against the oracle's golden values.

Golden constants come from PyPhi 1.2.1's test suite and doctested documentation
(``docs/notes/pyphi.md`` §5). The oracle computed EMDs with pyemd, whose approximate
optima it rounds to 1e-6 — hence goldens like 1.874999 where the exact value is 1.875 —
so comparisons allow 2e-6.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.enumeration import subsets
from iitx.measures.iit3 import ces, system_phi
from networks import basic_network, paper_network, xor_network

ATOL = 2e-6
"""Golden tolerance: the oracle's 1e-6 rounding of approximate pyemd optima."""


def by_mechanism(concepts: object) -> dict[tuple[int, ...], int]:
	"""Index the existing concepts by their mechanism.

	Args:
		concepts: A ``Concepts`` structure.

	Returns:
		Mapping from mechanism unit tuples to row indices.

	"""
	return {
		tuple(int(u) for u in np.flatnonzero(np.asarray(concepts.mechanism)[k])): int(k)
		for k in np.flatnonzero(np.asarray(concepts.exists))
	}


class TestCes:
	"""Cause-effect structures (golden: PyPhi 1.2.1 fixtures)."""

	def test_basic_network_concepts(self) -> None:
		"""Test the CES of the basic network at (1,0,0).

		Golden: four concepts with φ = {(1,): 0.25, (2,): 0.5, (0,1): 0.333333,
		(0,1,2): 0.5}, with core cause purviews (1,)→(2,), (2,)→(0,1), (0,1)→(1,2),
		(0,1,2)→(0,1,2).
		"""
		concepts = ces(basic_network(), jnp.asarray([1, 0, 0]))
		index = by_mechanism(concepts)
		assert set(index) == {(1,), (2,), (0, 1), (0, 1, 2)}

		golden_phi = {(1,): 0.25, (2,): 0.5, (0, 1): 0.333333, (0, 1, 2): 0.5}
		golden_cause = {(1,): (2,), (2,): (0, 1), (0, 1): (1, 2), (0, 1, 2): (0, 1, 2)}
		for mechanism, k in index.items():
			assert float(concepts.phi[k]) == pytest.approx(golden_phi[mechanism], abs=ATOL)
			cause = tuple(int(u) for u in np.flatnonzero(np.asarray(concepts.cause_purview)[k]))
			assert cause == golden_cause[mechanism]

	def test_xor_network_concepts(self) -> None:
		"""Test the CES of the XOR network at (0,0,0).

		Golden: three concepts — the three pairs — each with φ = 0.5.
		"""
		concepts = ces(xor_network(), jnp.asarray([0, 0, 0]))
		index = by_mechanism(concepts)
		assert set(index) == {(0, 1), (0, 2), (1, 2)}
		for k in index.values():
			assert float(concepts.phi[k]) == pytest.approx(0.5, abs=ATOL)

	def test_paper_network_concepts(self) -> None:
		"""Test the CES of the 2014-paper system at (1,0,0).

		Golden (paper Fig. 10 / PyPhi doctests): six concepts with φ values
		{0.166667, 0.166667, 0.25, 0.25, 0.333334, 0.499999}.
		"""
		concepts = ces(paper_network(), jnp.asarray([1, 0, 0]))
		exists = np.asarray(concepts.exists)
		assert int(exists.sum()) == 6
		np.testing.assert_allclose(
			sorted(float(x) for x in np.asarray(concepts.phi)[exists]),
			[0.166667, 0.166667, 0.25, 0.25, 0.333334, 0.499999],
			atol=ATOL,
		)


class TestSystemPhi:
	"""Big Φ (golden: PyPhi 1.2.1 fixtures and the 2014 paper)."""

	def test_basic_network(self) -> None:
		"""Test Φ of the basic network at (1,0,0).

		Golden: Φ = 2.3125, minimal cut (1,2) → (0,).
		"""
		result = system_phi(basic_network(), jnp.asarray([1, 0, 0]))
		assert float(result.phi) == pytest.approx(2.3125, abs=ATOL)

		source = tuple(int(u) for u in np.flatnonzero(subsets(3, nonempty=True)[result.cut_index]))
		assert source == (1, 2)

	def test_xor_network(self) -> None:
		"""Test Φ of the XOR network at (0,0,0): golden Φ = 1.874999 (exactly 15/8)."""
		result = system_phi(xor_network(), jnp.asarray([0, 0, 0]))
		assert float(result.phi) == pytest.approx(1.875, abs=ATOL)

	def test_paper_network(self) -> None:
		"""Test Φ of the 2014-paper system at (1,0,0).

		Golden (paper: Φ = 1.92; PyPhi doctest: 1.916665), the number the theory's own
		worked example publishes.
		"""
		result = system_phi(paper_network(), jnp.asarray([1, 0, 0]))
		assert float(result.phi) == pytest.approx(1.916665, abs=ATOL)


class TestTransformations:
	"""jit composition (grad is refused by construction — the EMD is a host LP)."""

	def test_jit(self) -> None:
		"""Test that Φ is identical eager and jitted."""
		system, state = basic_network(), jnp.asarray([1, 0, 0])
		eager = system_phi(system, state)
		jitted = jax.jit(system_phi)(system, state)
		assert float(eager.phi) == pytest.approx(float(jitted.phi), abs=1e-12)
		assert int(eager.cut_index) == int(jitted.cut_index)

	def test_grad_is_refused(self) -> None:
		"""Test that differentiating Φ^3.0 fails loudly at the EMD callback."""
		system, state = basic_network(), jnp.asarray([1, 0, 0])

		with pytest.raises(ValueError, match="do not support JVP"):
			jax.grad(lambda tpm: system_phi(dataclasses.replace(system, tpm=tpm), state).phi)(
				system.tpm
			)
