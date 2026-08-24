"""Tests for repertoire distances, against worked examples from the papers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iitx.distances import emd, hamming_matrix, intrinsic_difference, marginal_emd


class TestIntrinsicDifference:
	"""The intrinsic difference of Barbosa et al. (2020)."""

	def test_noiseless_channel_of_n_wires_carries_n_ibits(self) -> None:
		"""Test the expansion property: a fully selective state over 2**n states = n ibits.

		Barbosa et al. (2020): a noiseless one-wire channel carries 1 ibit and a noiseless
		eight-wire channel 8 ibits.
		"""
		for n in (1, 8):
			p = jnp.zeros(2**n).at[0].set(1.0)
			q = jnp.full(2**n, 1.0 / 2**n)
			assert float(intrinsic_difference(p, q)) == pytest.approx(n, abs=1e-6)

	def test_identical_distributions_have_zero_difference(self) -> None:
		"""Test the causality property: ID is zero iff there is no difference."""
		p = jnp.asarray([0.4, 0.35, 0.25])
		assert float(intrinsic_difference(p, p)) == pytest.approx(0.0, abs=1e-7)

	def test_hand_computed_value(self) -> None:
		"""Test a two-state example against the closed form.

		For p = (0.75, 0.25) against uniform, the maximum is at the first state:
		0.75 * log2(1.5).
		"""
		value = intrinsic_difference(jnp.asarray([0.75, 0.25]), jnp.asarray([0.5, 0.5]))
		assert float(value) == pytest.approx(0.75 * np.log2(1.5), abs=1e-6)

	def test_dilution_by_an_unconstrained_unit(self) -> None:
		"""Test the dilution property: adding a fully noisy unit divides the ID.

		Extending (1, 0) against uniform by an independent uniform two-state unit halves
		the intrinsic difference from 1 ibit to 0.5 ibit — the property KLD lacks.
		"""
		p = jnp.asarray([0.5, 0.0, 0.5, 0.0])
		q = jnp.full(4, 0.25)
		assert float(intrinsic_difference(p, q)) == pytest.approx(0.5, abs=1e-6)

	def test_zero_probability_states_contribute_zero(self) -> None:
		"""Test the continuous extension f(0, q) = 0, including at q = 0."""
		p = jnp.asarray([1.0, 0.0])
		q = jnp.asarray([1.0, 0.0])
		assert float(intrinsic_difference(p, q)) == pytest.approx(0.0, abs=1e-7)

	def test_impossible_reference_state_diverges(self) -> None:
		"""Test that p > 0 where q = 0 yields inf, the honest out-of-domain answer."""
		value = intrinsic_difference(jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0]))
		assert bool(jnp.isinf(value))

	def test_gradient_matches_finite_differences(self) -> None:
		"""Test the gradient away from ties against central finite differences."""
		p = jnp.asarray([0.6, 0.3, 0.1], dtype=jnp.float32)
		q = jnp.asarray([0.3, 0.5, 0.2], dtype=jnp.float32)

		gradient = jax.grad(intrinsic_difference)(p, q)
		eps = 1e-3
		for v in range(3):
			bump = jnp.zeros(3).at[v].set(eps)
			numeric = (
				float(intrinsic_difference(p + bump, q)) - float(intrinsic_difference(p - bump, q))
			) / (2 * eps)
			assert float(gradient[v]) == pytest.approx(numeric, abs=1e-3)

	def test_gradient_has_no_nan_at_zero_probability(self) -> None:
		"""Test the double-where guard: zero-probability states get zero gradient, not NaN."""
		p = jnp.asarray([1.0, 0.0])
		q = jnp.asarray([0.5, 0.5])
		gradient = jax.grad(intrinsic_difference)(p, q)
		assert bool(jnp.all(jnp.isfinite(gradient)))


class TestHammingMatrix:
	"""The generalized Hamming ground metric."""

	def test_binary_three_units(self) -> None:
		"""Test distances between binary states, e.g. d(000, 111) = 3 and d(010, 100) = 2."""
		cost = hamming_matrix((2, 2, 2))
		assert cost.shape == (8, 8)
		assert cost[0, 7] == 3  # 000 vs 111
		assert cost[2, 1] == 2  # 010 vs 100
		np.testing.assert_array_equal(cost, cost.T)
		np.testing.assert_array_equal(np.diag(cost), np.zeros(8))

	def test_nonuniform_alphabets(self) -> None:
		"""Test that only the number of differing units counts, not how much they differ."""
		cost = hamming_matrix((3, 2))
		# States in little-endian order: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1).
		assert cost[0, 2] == 1  # (0,0) vs (2,0): one unit differs
		assert cost[0, 5] == 2  # (0,0) vs (2,1): both units differ


class TestEmd:
	"""The exact earth mover's distance (host linear program)."""

	def test_iit3_supplement_toy_example(self) -> None:
		"""Test Text S2 Fig S2-3 of Oizumi et al. (2014).

		A repertoire concentrated on state 00, compared with mass spread to a state at
		Hamming distance 1 (EMD 0.5) versus distance 2 (EMD 1.0) — the example motivating
		EMD over KLD, which scores both as 1 bit.
		"""
		cost = jnp.asarray(hamming_matrix((2, 2)))
		p = jnp.asarray([1.0, 0.0, 0.0, 0.0])
		spread_near = jnp.asarray([0.5, 0.5, 0.0, 0.0])  # half the mass to 10
		spread_far = jnp.asarray([0.5, 0.0, 0.0, 0.5])  # half the mass to 11

		assert float(emd(p, spread_near, cost)) == pytest.approx(0.5, abs=1e-9)
		assert float(emd(p, spread_far, cost)) == pytest.approx(1.0, abs=1e-9)

	def test_identity_and_symmetry(self) -> None:
		"""Test that EMD is zero on identical inputs and symmetric in its arguments."""
		cost = jnp.asarray(hamming_matrix((2, 2)))
		p = jnp.asarray([0.4, 0.3, 0.2, 0.1])
		q = jnp.asarray([0.1, 0.2, 0.3, 0.4])

		assert float(emd(p, p, cost)) == pytest.approx(0.0, abs=1e-9)
		assert float(emd(p, q, cost)) == pytest.approx(float(emd(q, p, cost)), abs=1e-9)

	def test_jit_and_vmap(self) -> None:
		"""Test that the host callback composes with jit and vmap."""
		cost = jnp.asarray(hamming_matrix((2, 2)))
		p = jnp.asarray([1.0, 0.0, 0.0, 0.0])
		targets = jnp.asarray([[0.5, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.5]])

		batched = jax.jit(jax.vmap(emd, in_axes=(None, 0, None)))(p, targets, cost)
		np.testing.assert_allclose(np.asarray(batched), [0.5, 1.0], atol=1e-9)

	def test_grad_is_refused(self) -> None:
		"""Test that differentiating the exact EMD fails loudly, as designed."""
		cost = jnp.asarray(hamming_matrix((2,)))
		p = jnp.asarray([0.7, 0.3])
		q = jnp.asarray([0.5, 0.5])

		with pytest.raises(ValueError, match="do not support JVP"):
			jax.grad(lambda p: emd(p, q, cost))(p)


class TestMarginalEmd:
	"""The analytic per-unit EMD for product distributions."""

	def test_matches_pyphi_effect_formula_on_binary_products(self) -> None:
		"""Test agreement with sum_i |P1(i OFF) - P2(i OFF)| on binary product distributions."""
		p_on = np.asarray([0.75, 0.25, 0.5])
		q_on = np.asarray([0.5, 0.5, 0.9])
		p = jnp.asarray(
			np.kron(np.kron([1 - p_on[2], p_on[2]], [1 - p_on[1], p_on[1]]), [1 - p_on[0], p_on[0]])
		)
		q = jnp.asarray(
			np.kron(np.kron([1 - q_on[2], q_on[2]], [1 - q_on[1], q_on[1]]), [1 - q_on[0], q_on[0]])
		)

		expected = np.abs(p_on - q_on).sum()
		assert float(marginal_emd(p, q, (2, 2, 2))) == pytest.approx(expected, abs=1e-6)

	def test_equals_exact_emd_on_product_distributions(self) -> None:
		"""Test that the separable formula agrees with the linear program on products."""
		rng = np.random.default_rng(1)
		shape = (3, 2)
		marginals = [rng.random(q) for q in shape]
		marginals = [m / m.sum() for m in marginals]
		other = [rng.random(q) for q in shape]
		other = [m / m.sum() for m in other]

		p = jnp.asarray(np.kron(marginals[1], marginals[0]))
		q = jnp.asarray(np.kron(other[1], other[0]))
		cost = jnp.asarray(hamming_matrix(shape))

		assert float(marginal_emd(p, q, shape)) == pytest.approx(float(emd(p, q, cost)), abs=1e-6)

	def test_differentiable(self) -> None:
		"""Test that the analytic form has finite gradients, unlike the exact EMD."""
		p = jnp.asarray([0.4, 0.3, 0.2, 0.1])
		q = jnp.asarray([0.25, 0.25, 0.25, 0.25])
		gradient = jax.grad(lambda p: marginal_emd(p, q, (2, 2)))(p)
		assert bool(jnp.all(jnp.isfinite(gradient)))
