"""Enumeration of the combinatorial spaces of an IIT analysis.

Every combinatorial space `iitx` searches — candidate systems, mechanisms, purviews,
partitions, cuts — is enumerated **once, in NumPy, at build time** as a stacked table of
fixed-width masks. Kernels then `vmap` or `scan` over the leading axis of a table, so a
whole space costs one compiled program rather than one program per element (see
``docs/design.md`` §2, P2).

Table order is part of the library's contract, not an implementation detail: `iitx` reports
the *identity* of an optimum (which purview, which partition) by first-occurrence
``argmin``/``argmax`` over these tables, so the order is what makes tie-breaking
deterministic and oracle-compatible. Subsets are enumerated in **powerset order** —
increasing size, and lexicographic by unit index within each size — matching PyPhi's
``utils.powerset``.
"""

from collections.abc import Iterator
from itertools import combinations, product

import numpy as np
from jaxtyping import Bool, Int

__all__ = ["set_partitions", "subsets", "system_cuts"]


def subsets(n: int, *, nonempty: bool = False) -> Bool[np.ndarray, "num_subsets n"]:
	"""Enumerate subsets of ``n`` units as masks, in powerset order.

	Powerset order is increasing subset size, and lexicographic by unit index within a
	size: for ``n = 3`` the subsets are ``(), (0,), (1,), (2,), (0,1), (0,2), (1,2),
	(0,1,2)``. This is PyPhi's enumeration order, which `iitx` reproduces so that ties
	broken by "first encountered" agree with the oracle.

	Args:
		n: Number of units.
		nonempty: If true, omit the empty subset. Mechanisms and purviews range over
			nonempty subsets; partition parts and candidate systems may be empty.

	Returns:
		Boolean array of shape ``(num_subsets, n)``, where ``num_subsets`` is ``2**n``, or
		``2**n - 1`` when ``nonempty`` is set. Row ``k`` is the mask of the ``k``-th subset:
		``mask[i]`` is true when unit ``i`` belongs to the subset.

	"""
	if n < 0:
		msg = f"number of units must be non-negative, got {n}"
		raise ValueError(msg)

	masks = np.zeros((2**n - nonempty, n), dtype=bool)
	rows = (
		indices for size in range(int(nonempty), n + 1) for indices in combinations(range(n), size)
	)
	for row, indices in enumerate(rows):
		masks[row, list(indices)] = True
	return masks


def set_partitions(units: list[int], *, nontrivial: bool = False) -> Iterator[list[list[int]]]:
	"""Enumerate the set partitions of a list of units, in PyPhi's recursive order.

	The recursion inserts the first unit into each block of every partition of the
	remaining units, then as a singleton block — the order of PyPhi's
	``combinatorics.set_partitions``, which the tie-certificate contract requires: for
	``[0, 1, 2]`` the partitions are ``[[0,1,2]]``, ``[[0],[1,2]]``, ``[[0,1],[2]]``,
	``[[1],[0,2]]``, ``[[0],[1],[2]]``.

	Args:
		units: The units to partition.
		nontrivial: If true, omit the single-block partition.

	Yields:
		Partitions as lists of blocks, each block a list of units.

	"""

	def recurse(collection: list[int]) -> Iterator[list[list[int]]]:
		if not collection:
			return
		if len(collection) == 1:
			yield [collection]
			return
		first = collection[0]
		for smaller in recurse(collection[1:]):
			for k, block in enumerate(smaller):
				yield [*smaller[:k], [first, *block], *smaller[k + 1 :]]
			yield [[first], *smaller]

	iterator = recurse(list(units))
	if nontrivial:
		next(iterator, None)
	yield from iterator


def system_cuts(
	n: int, candidate: tuple[int, ...] | None = None
) -> tuple[Bool[np.ndarray, "num_cuts n n"], Int[np.ndarray, " num_cuts"]]:
	"""Enumerate the IIT 4.0 system partitions of a candidate system, as cut matrices.

	This is PyPhi's ``SET_UNI/BI`` scheme (the canonical IIT 4.0 system partitions,
	Albantakis et al. 2023 Eqs. 14-16): every nontrivial set partition of the candidate's
	units, combined with a per-block direction flag — ``CAUSE`` (the block's inputs are
	severed), ``EFFECT`` (its outputs are severed), or ``BIDIRECTIONAL`` (both). Each
	combination reduces to an ``(n, n)`` **cut matrix** whose entry ``(i, j)`` marks the
	connection from unit ``i`` to unit ``j`` as severed (unit ``j`` then perceives ``i``
	as uniform noise). Distinct flag assignments can induce the same matrix; duplicates
	are dropped keeping the first occurrence, exactly as PyPhi's order-preserving
	deduplication does, so cut indices are oracle-compatible certificates.

	The severed-edge count of each cut is returned alongside: the minimum partition is
	selected by φ *normalized* by that count (the maximum possible φ of the cut; Marshall
	et al. 2023), though the reported φ is unnormalized.

	Args:
		n: Number of units of the system.
		candidate: Units of the candidate system being partitioned; defaults to all
			``n`` units. Other units are untouched by every cut.

	Returns:
		A pair: boolean cut matrices of shape ``(num_cuts, n, n)``, and the number of
		severed edges of each cut, shape ``(num_cuts,)``.

	"""
	units = list(range(n)) if candidate is None else sorted(candidate)
	directions = ("cause", "effect", "bidirectional")

	matrices: list[np.ndarray] = []
	seen: set[bytes] = set()
	for partition in set_partitions(units, nontrivial=True):
		for flags in product(directions, repeat=len(partition)):
			matrix = np.zeros((n, n), dtype=bool)
			for block, flag in zip(partition, flags, strict=True):
				rest = [unit for unit in units if unit not in block]
				if flag in ("cause", "bidirectional"):
					matrix[np.ix_(rest, block)] = True
				if flag in ("effect", "bidirectional"):
					matrix[np.ix_(block, rest)] = True
			key = matrix.tobytes()
			if key not in seen:
				seen.add(key)
				matrices.append(matrix)

	cuts = np.stack(matrices)
	return cuts, cuts.sum(axis=(1, 2)).astype(np.int64)
