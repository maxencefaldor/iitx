"""Tests for the enumeration of combinatorial spaces."""

import numpy as np
import pytest

from iitx.enumeration import subsets


def test_subsets_powerset_order() -> None:
	"""Test that subsets are enumerated in PyPhi's powerset order.

	Increasing size, lexicographic by unit index within a size — the order that makes
	first-occurrence tie-breaking agree with the oracle.
	"""
	np.testing.assert_array_equal(
		subsets(3),
		[
			[False, False, False],  # ()
			[True, False, False],  # (0,)
			[False, True, False],  # (1,)
			[False, False, True],  # (2,)
			[True, True, False],  # (0, 1)
			[True, False, True],  # (0, 2)
			[False, True, True],  # (1, 2)
			[True, True, True],  # (0, 1, 2)
		],
	)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8])
def test_subsets_cardinality(n: int) -> None:
	"""Test that every subset appears exactly once."""
	masks = subsets(n)
	assert masks.shape == (2**n, n)
	assert len({tuple(mask) for mask in masks}) == 2**n


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_subsets_nonempty_drops_only_the_empty_subset(n: int) -> None:
	"""Test that the nonempty variant is the full powerset minus its first row."""
	np.testing.assert_array_equal(subsets(n, nonempty=True), subsets(n)[1:])
	assert subsets(n, nonempty=True).all(axis=1).any()


@pytest.mark.parametrize("n", [1, 3, 5])
def test_subsets_sizes_are_non_decreasing(n: int) -> None:
	"""Test that subset size never decreases along the table."""
	sizes = subsets(n).sum(axis=1)
	assert (np.diff(sizes) >= 0).all()


def test_subsets_rejects_negative_n() -> None:
	"""Test that a negative unit count is rejected."""
	with pytest.raises(ValueError, match="non-negative"):
		subsets(-1)
