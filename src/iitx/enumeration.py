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

from itertools import combinations

import numpy as np
from jaxtyping import Bool

__all__ = ["subsets"]


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
