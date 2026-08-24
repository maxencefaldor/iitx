"""Tests for the named relaxations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.distances import emd, hamming_matrix, sinkhorn_emd
from iitx.measures import iit4
from iitx.relax import soft_system_phi
from iitx.system import System
from networks import basic_network, basic_noisy_selfloop_network


class TestSinkhornEmd:
	"""The entropic approximation of the exact EMD."""

	def test_converges_to_exact(self) -> None:
		"""Test convergence to the exact EMD with sufficient iterations."""
		cost = jnp.asarray(hamming_matrix((2, 2, 2)))
		rng = np.random.default_rng(0)
		p = rng.random(8)
		q = rng.random(8)
		p, q = jnp.asarray(p / p.sum()), jnp.asarray(q / q.sum())

		exact = float(emd(p, q, cost))
		approx = float(sinkhorn_emd(p, q, cost, epsilon=0.05, iterations=500))
		assert approx == pytest.approx(exact, abs=1e-4)

	def test_differentiable(self) -> None:
		"""Test that the approximation has finite gradients — its reason to exist."""
		cost = jnp.asarray(hamming_matrix((2, 2)))
		q = jnp.full(4, 0.25)
		gradient = jax.grad(lambda p: sinkhorn_emd(p, q, cost))(jnp.asarray([0.4, 0.3, 0.2, 0.1]))
		assert bool(jnp.all(jnp.isfinite(gradient)))

	def test_handles_sparse_support(self) -> None:
		"""Test distributions with zero-probability states."""
		cost = jnp.asarray(hamming_matrix((2, 2)))
		p = jnp.asarray([1.0, 0.0, 0.0, 0.0])
		q = jnp.asarray([0.5, 0.5, 0.0, 0.0])
		assert float(sinkhorn_emd(p, q, cost, epsilon=0.05, iterations=500)) == pytest.approx(
			0.5, abs=1e-4
		)


class TestSoftSystemPhi:
	"""The smooth surrogate of signed φ_s."""

	def test_low_temperature_recovers_exact(self) -> None:
		"""Test convergence to the exact signed φ_s as the temperature vanishes."""
		system, state = basic_network(), jnp.asarray([1, 0, 0])
		exact = float(iit4.system_phi(system, state).signed_phi)
		soft = float(soft_system_phi(system, state, temperature=1e-4))
		assert soft == pytest.approx(exact, abs=1e-6)

	def test_negative_for_reducible_systems(self) -> None:
		"""Test that the surrogate keeps the signed (unclamped) behaviour."""
		system, state = basic_noisy_selfloop_network(), jnp.asarray([1, 0, 0])
		assert float(soft_system_phi(system, state, temperature=1e-4)) < 0.0

	def test_smooth_gradient(self) -> None:
		"""Test gradients through the surrogate against finite differences."""
		system, state = basic_noisy_selfloop_network(), jnp.asarray([1, 0, 0])

		def value(tpm: jax.Array) -> jax.Array:
			return soft_system_phi(System(tpm, system.shape, system.cm), state, temperature=0.05)

		gradient = jax.grad(value)(system.tpm)
		assert bool(jnp.all(jnp.isfinite(gradient)))

		eps = 1e-6
		numeric = (
			float(value(system.tpm + eps * jnp.eye(8, 8)))
			- float(value(system.tpm - eps * jnp.eye(8, 8)))
		) / (2 * eps)
		assert float(jnp.sum(gradient * jnp.eye(8, 8))) == pytest.approx(numeric, abs=1e-4)
