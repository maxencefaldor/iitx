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

__all__ = ["set_partitions", "subsets", "system_cuts"]


def subsets(n: int, *, nonempty: bool = False) -> np.ndarray:
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


def system_cuts(n: int, candidate: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
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
	if len(units) == 1:
		# A single-unit candidate admits only the complete partition (its self-loop
		# severed), as in the oracle's monad handling.
		complete = np.zeros((1, n, n), dtype=bool)
		complete[0, units[0], units[0]] = True
		return complete, np.ones(1, dtype=np.int64)
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


def mechanism_partitions(
	mechanism: tuple[int, ...], purview: tuple[int, ...], n: int
) -> tuple[
	np.ndarray,
	np.ndarray,
	np.ndarray,
]:
	"""Enumerate the disintegrating partitions Θ(M, Z) of a mechanism-purview pair.

	This is PyPhi's ``ALL`` scheme — the canonical IIT 4.0 mechanism partitions
	(Albantakis et al. 2023 Eq. 38, from Barbosa et al. 2021): disjoint parts pairing a
	mechanism block with a purview block, jointly covering both, where a part containing
	the whole mechanism must have an empty purview (so every partition genuinely
	disintegrates the pair). Mechanism blocks follow PyPhi's set-partition recursion with
	an appended empty block; purview blocks are distributed over them in every distinct
	way.

	The partitioned probability factors per unit, so each partition is encoded by
	*co-part bitmasks*: for each purview unit, the bitmask of the mechanism block in its
	part (its only unsevered mechanism inputs), and for each mechanism unit, the bitmask
	of the purview block in its part. These index the master conditional tables directly.

	Args:
		mechanism: Units of the mechanism (sorted).
		purview: Units of the purview (sorted).
		n: Number of units of the system (bitmask width).

	Returns:
		A triple of arrays over the ``P`` partitions: ``co_mechanism`` of shape
		``(P, n)`` — for each purview unit ``j``, the bitmask of its part's mechanism
		block (rows for units outside the purview are zero); ``co_purview`` of shape
		``(P, n)`` — for each mechanism unit ``u``, the bitmask of its part's purview
		block; and the number of connections each partition severs,
		``sum_i |mechanism_i| * |purview outside part i|``.

	"""

	def bitmask(units: tuple[int, ...] | list[int]) -> int:
		return sum(1 << unit for unit in units)

	def purview_splits(units: list[int], k: int) -> Iterator[list[list[int]]]:
		# All ways to split `units` into k labeled, possibly-empty ordered blocks such
		# that nonempty blocks are disjoint; equivalent to PyPhi's k_partitions plus
		# distinct permutations with empty padding: every assignment of units to k
		# labeled blocks, deduplicated by the resulting labeled composition.
		seen: set[tuple[tuple[int, ...], ...]] = set()
		for assignment in product(range(k), repeat=len(units)):
			blocks: list[list[int]] = [[] for _ in range(k)]
			for unit, block in zip(units, assignment, strict=True):
				blocks[block].append(unit)
			key = tuple(tuple(block) for block in blocks)
			if key not in seen:
				seen.add(key)
				yield blocks

	co_mechanism_rows: list[np.ndarray] = []
	co_purview_rows: list[np.ndarray] = []
	severed: list[int] = []
	for blocks in set_partitions(list(mechanism)):
		mechanism_blocks = [*blocks, []]
		for purview_blocks in purview_splits(list(purview), len(mechanism_blocks)):
			# A part keeping the whole mechanism must have an empty purview block.
			if any(
				set(mech) == set(mechanism) and purv
				for mech, purv in zip(mechanism_blocks, purview_blocks, strict=True)
			):
				continue
			co_mechanism = np.zeros(n, dtype=np.int64)
			co_purview = np.zeros(n, dtype=np.int64)
			cut = 0
			for mech, purv in zip(mechanism_blocks, purview_blocks, strict=True):
				for j in purv:
					co_mechanism[j] = bitmask(mech)
				for u in mech:
					co_purview[u] = bitmask(purv)
				cut += len(mech) * (len(purview) - len(purv))
			co_mechanism_rows.append(co_mechanism)
			co_purview_rows.append(co_purview)
			severed.append(cut)

	return (
		np.stack(co_mechanism_rows),
		np.stack(co_purview_rows),
		np.asarray(severed, dtype=np.int64),
	)


def bipartitions(
	mechanism: tuple[int, ...], purview: tuple[int, ...], n: int
) -> tuple[np.ndarray, np.ndarray]:
	"""Enumerate the IIT 3.0 mechanism-purview bipartitions.

	PyPhi's ``BI`` scheme (the canonical IIT 3.0 mechanism partitions, Oizumi et al.
	2014): unordered bipartitions of the mechanism crossed with directed bipartitions of
	the purview, excluding assignments where one part is empty on both sides. Each
	partition is ``(M1, Z1) x (M2, Z2)`` with ``M2``, ``Z2`` the complements within the
	mechanism and purview; only the first part is returned.

	Args:
		mechanism: Units of the mechanism (sorted).
		purview: Units of the purview (sorted).
		n: Number of units of the system (mask width).

	Returns:
		A pair of boolean arrays of shape ``(P, n)``: the first part's mechanism masks
		and purview masks.

	"""
	# Unordered mechanism halves: each pair {A, M without A} once — enumerate the halves
	# not containing the mechanism's first unit (both halves, for an empty mechanism).
	if mechanism:
		mechanism_halves = [
			half for size in range(len(mechanism)) for half in combinations(mechanism[1:], size)
		]
	else:
		mechanism_halves = [()]

	part_mechanism: list[np.ndarray] = []
	part_purview: list[np.ndarray] = []
	for mechanism_half in mechanism_halves:
		other_half = tuple(u for u in mechanism if u not in mechanism_half)
		for size in range(len(purview) + 1):
			for purview_half in combinations(purview, size):
				purview_other = tuple(u for u in purview if u not in purview_half)
				if (not mechanism_half and not purview_half) or (
					not other_half and not purview_other
				):
					continue
				mask_m = np.zeros(n, dtype=bool)
				mask_m[list(mechanism_half)] = True
				mask_z = np.zeros(n, dtype=bool)
				mask_z[list(purview_half)] = True
				part_mechanism.append(mask_m)
				part_purview.append(mask_z)

	return np.stack(part_mechanism), np.stack(part_purview)


def directed_bipartitions(n: int) -> np.ndarray:
	"""Enumerate the IIT 3.0 system cuts as cut matrices.

	A cut severs the connections *from* one nonempty proper subset *to* its complement
	(unidirectional; Oizumi et al. 2014). There are ``2**n - 2`` cuts, ordered by the
	subset table.

	Args:
		n: Number of units.

	Returns:
		Boolean cut matrices of shape ``(2**n - 2, n, n)``; entry ``(i, j)`` severs the
		connection from unit ``i`` to unit ``j``.

	"""
	masks = subsets(n, nonempty=True)[:-1]  # nonempty proper subsets
	source = masks[:, :, None]
	target = ~masks[:, None, :]
	return source & target
