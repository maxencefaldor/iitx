"""Tests for the measure protocol."""

import jax.numpy as jnp
import pytest

from iitx.measures import IIT3, IIT4, Measure
from networks import basic_network


def test_both_measures_satisfy_the_protocol() -> None:
	"""Test that the shipped measures are Measure instances."""
	assert isinstance(IIT3(), Measure)
	assert isinstance(IIT4(), Measure)


def test_headline_scalars() -> None:
	"""Test each measure's phi on the basic network at (1,0,0)."""
	system, state = basic_network(), jnp.asarray([1, 0, 0])
	assert float(IIT4().phi(system, state)) == pytest.approx(0.41503749927884376, abs=1e-10)
	assert float(IIT3().phi(system, state)) == pytest.approx(2.3125, abs=2e-6)


def test_analyze_returns_full_results() -> None:
	"""Test that analyze returns the measures' result pytrees."""
	system, state = basic_network(), jnp.asarray([1, 0, 0])
	structure = IIT4().analyze(system, state)
	assert float(structure.big_phi) == pytest.approx(1.0, abs=1e-10)
	analysis = IIT3().analyze(system, state)
	assert int(analysis.ces.exists.sum()) == 4
