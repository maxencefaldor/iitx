"""Tests for IIT 4.0 system-level quantities against the oracle's golden values.

The golden constants come from PyPhi ``feature/iit-4.0``'s committed fixtures (the code
behind Albantakis et al. 2023) as recorded in ``docs/notes/pyphi.md`` §5.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.enumeration import system_cuts
from iitx.measures.iit4 import cause_effect_state, distinctions, phi_structure, system_phi
from iitx.system import System
from networks import (
	basic_network,
	basic_noisy_selfloop_network,
	paper_network,
	xor_network,
)


def cut_edges(n: int, index: int) -> set[tuple[int, int]]:
	"""Return the severed edges of the cut at ``index`` in the canonical table.

	Args:
		n: Number of units.
		index: Cut index.

	Returns:
		The set of severed ``(source, target)`` connections.

	"""
	cuts, _ = system_cuts(n)
	return {(i, j) for i in range(n) for j in range(n) if cuts[index, i, j]}


class TestCauseEffectState:
	"""The maximal cause-effect state (golden: basic network)."""

	def test_basic_network_specification(self) -> None:
		"""Test the specified states and intrinsic information of the basic network.

		Golden (PyPhi fixture ``basic``, state (1,0,0)): cause state (1,1,0) with
		ii_c = 3.0 and effect state (0,0,1) with ii_e = 3.0.
		"""
		ces = cause_effect_state(basic_network(), jnp.asarray([1, 0, 0]))

		np.testing.assert_array_equal(np.asarray(ces.cause_state), [1, 1, 0])
		np.testing.assert_array_equal(np.asarray(ces.effect_state), [0, 0, 1])
		assert float(ces.phi_cause) == pytest.approx(3.0, abs=1e-10)
		assert float(ces.phi_effect) == pytest.approx(3.0, abs=1e-10)


class TestSystemPhi:
	"""System integrated information φ_s (golden fixtures)."""

	def test_basic_network(self) -> None:
		"""Test φ_s of the basic network at (1,0,0).

		Golden: φ_s = 0.41503749927884376 = log2(4/3), MIP = {[0], [1,2]} — the cut
		severing the two connections into unit 0.
		"""
		result = system_phi(basic_network(), jnp.asarray([1, 0, 0]))

		assert float(result.phi) == pytest.approx(0.41503749927884376, abs=1e-10)
		assert float(result.normalized_phi) == pytest.approx(0.41503749927884376 / 2, abs=1e-10)
		assert cut_edges(3, int(result.cut_index)) == {(1, 0), (2, 0)}

	def test_xor_network(self) -> None:
		"""Test φ_s of the XOR network at (0,0,0).

		Golden: φ_s = 1.5, MIP = the singleton tripartition {[0], [1], [2]}.
		"""
		result = system_phi(xor_network(), jnp.asarray([0, 0, 0]))

		assert float(result.phi) == pytest.approx(1.5, abs=1e-10)
		assert cut_edges(3, int(result.cut_index)) == {
			(0, 1),
			(0, 2),
			(1, 0),
			(1, 2),
			(2, 0),
			(2, 1),
		}

	def test_paper_network_is_reducible_at_101(self) -> None:
		"""Test φ_s of the 2014-paper network at (1,0,1).

		Golden (fixture ``fig4``): φ_s = 0.0, MIP = {[0,1], [2]}.
		"""
		result = system_phi(paper_network(), jnp.asarray([1, 0, 1]))
		assert float(result.phi) == pytest.approx(0.0, abs=1e-10)

	def test_noisy_selfloop_network_is_reducible(self) -> None:
		"""Test φ_s of the noisy self-loop network at (1,0,0).

		The 2023-paper branch published φ_s = -0.38198987262266504; the primary oracle
		(and the paper's |·|₊) clamps the reported φ at zero while keeping the signed
		value — reducibility with gradient signal intact.
		"""
		result = system_phi(basic_noisy_selfloop_network(), jnp.asarray([1, 0, 0]))
		assert float(result.phi) == pytest.approx(0.0, abs=1e-10)
		assert float(result.signed_phi) == pytest.approx(-0.38198987262266504, abs=1e-10)


class TestTransformations:
	"""jit, vmap, and grad composition."""

	def test_jit(self) -> None:
		"""Test that φ_s is identical eager and jitted."""
		system, state = basic_network(), jnp.asarray([1, 0, 0])
		eager = system_phi(system, state)
		jitted = jax.jit(system_phi)(system, state)
		assert float(eager.phi) == pytest.approx(float(jitted.phi), abs=1e-12)
		assert int(eager.cut_index) == int(jitted.cut_index)

	def test_vmap_over_states(self) -> None:
		"""Test batching φ_s over all current states of one system in a single call."""
		system = basic_noisy_selfloop_network()
		states = jnp.asarray([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])
		batched = jax.jit(jax.vmap(system_phi, in_axes=(None, 0)))(system, states)

		for k in range(8):
			single = system_phi(system, states[k])
			assert float(batched.phi[k]) == pytest.approx(float(single.phi), abs=1e-12)

	def test_vmap_over_systems(self) -> None:
		"""Test batching φ_s over a stack of systems in a single call."""
		stacked = jax.tree.map(
			lambda *leaves: jnp.stack(leaves), basic_network(), basic_noisy_selfloop_network()
		)
		state = jnp.asarray([1, 0, 0])
		batched = jax.vmap(system_phi, in_axes=(0, None))(stacked, state)

		assert float(batched.phi[0]) == pytest.approx(0.41503749927884376, abs=1e-10)
		assert float(batched.signed_phi[1]) == pytest.approx(-0.38198987262266504, abs=1e-10)

	def test_grad_with_respect_to_tpm(self) -> None:
		"""Test that φ_s differentiates w.r.t. the TPM, against finite differences.

		The noisy network sits away from ties and zero-probability boundaries, so the
		subgradient is a plain gradient there.
		"""
		system, state = basic_noisy_selfloop_network(), jnp.asarray([1, 0, 0])

		def phi_of(tpm: jax.Array) -> jax.Array:
			return system_phi(System(tpm=tpm, shape=system.shape, cm=system.cm), state).signed_phi

		gradient = jax.grad(phi_of)(system.tpm)
		assert bool(jnp.all(jnp.isfinite(gradient)))

		eps = 1e-6
		rng = np.random.default_rng(0)
		for _ in range(3):
			u, v = rng.integers(8), rng.integers(8)
			bump = jnp.zeros((8, 8)).at[u, v].set(eps)
			numeric = (float(phi_of(system.tpm + bump)) - float(phi_of(system.tpm - bump))) / (
				2 * eps
			)
			assert float(gradient[u, v]) == pytest.approx(numeric, abs=1e-4)


class TestDistinctions:
	"""Causal distinctions against golden fixture values."""

	def test_basic_network(self) -> None:
		"""Test the distinctions of the basic network at (1,0,0).

		Golden: 2 distinctions with total φ_d = 1.0.
		"""
		found = distinctions(basic_network(), jnp.asarray([1, 0, 0]))
		exists = np.asarray(found.exists)
		assert int(exists.sum()) == 2
		assert float(np.asarray(found.phi)[exists].sum()) == pytest.approx(1.0, abs=1e-10)

	def test_xor_network(self) -> None:
		"""Test the distinctions of the XOR network at (0,0,0).

		Golden: 4 distinctions, total φ_d = 2.5; mechanism (0,1) has cause purview
		(0,1,2) with φ_c = 0.5 and effect purview (2,) with φ_e = 1.0; the full
		mechanism has φ_c = 1.0 and φ_e = 2.0.
		"""
		found = distinctions(xor_network(), jnp.asarray([0, 0, 0]))
		exists = np.asarray(found.exists)
		assert int(exists.sum()) == 4
		assert float(np.asarray(found.phi)[exists].sum()) == pytest.approx(2.5, abs=1e-10)

		by_mechanism = {
			tuple(np.flatnonzero(np.asarray(found.mechanism)[k])): k for k in np.flatnonzero(exists)
		}
		pair = by_mechanism[(0, 1)]
		assert float(found.phi_cause[pair]) == pytest.approx(0.5, abs=1e-10)
		assert tuple(np.flatnonzero(np.asarray(found.cause_purview)[pair])) == (0, 1, 2)
		assert float(found.phi_effect[pair]) == pytest.approx(1.0, abs=1e-10)
		assert tuple(np.flatnonzero(np.asarray(found.effect_purview)[pair])) == (2,)
		full = by_mechanism[(0, 1, 2)]
		assert float(found.phi_cause[full]) == pytest.approx(1.0, abs=1e-10)
		assert float(found.phi_effect[full]) == pytest.approx(2.0, abs=1e-10)

	def test_paper_network(self) -> None:
		"""Test the distinctions of the 2014-paper network at (1,0,1).

		Golden (fixture ``fig4``): 4 distinctions, total φ_d = 1.7174433312179418.
		"""
		found = distinctions(paper_network(), jnp.asarray([1, 0, 1]))
		exists = np.asarray(found.exists)
		assert int(exists.sum()) == 4
		assert float(np.asarray(found.phi)[exists].sum()) == pytest.approx(
			1.7174433312179418, abs=1e-10
		)


class TestPhiStructure:
	"""Φ-structures: relations in aggregate and big Φ."""

	def test_basic_network_has_no_relations(self) -> None:
		"""Test the basic network's Φ-structure: 0 relations, Φ = Σφ_d = 1.0.

		No unit has the same specified cause and effect value, and the two
		distinctions' purviews share no unit-state atom, so nothing relates.
		"""
		structure = phi_structure(basic_network(), jnp.asarray([1, 0, 0]))
		assert int(structure.relations.count) == 0
		assert float(structure.relations.sum_phi) == pytest.approx(0.0, abs=1e-12)
		assert float(structure.big_phi) == pytest.approx(1.0, abs=1e-10)

	def test_xor_network(self) -> None:
		"""Test the XOR network's Φ-structure against the golden count and closed form.

		Golden: 15 relations (every nonempty subset of the 4 distinctions relates).
		By hand: every distinction's atom set is all three units, so the 11 subsets of
		size ≥ 2 each contribute 3 x min-density = 0.5, and the self-relations
		contribute 1/6 + 1/6 + 1/6 + 1 — a total of Σφ_r = 7.0 and Φ = 9.5.
		"""
		structure = phi_structure(xor_network(), jnp.asarray([0, 0, 0]))
		assert int(structure.relations.count) == 15
		assert float(structure.relations.sum_phi) == pytest.approx(7.0, abs=1e-10)
		assert float(structure.big_phi) == pytest.approx(9.5, abs=1e-10)

	def test_paper_network_relation_count(self) -> None:
		"""Test the 2014-paper network's relation count at (1,0,1).

		Golden (fixture ``fig4``): 15 relations.
		"""
		structure = phi_structure(paper_network(), jnp.asarray([1, 0, 1]))
		assert int(structure.relations.count) == 15
