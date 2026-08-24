"""Macro-level transforms: coarse-graining and black-boxing.

Both transforms are pure functions from a :class:`~iitx.system.System` to a
:class:`~iitx.system.System` — the resulting macro system feeds the unchanged Φ
pipeline, with no special macro code path (``docs/design.md`` §10). This is what forces
the core's generality: coarse-graining produces multi-valued macro units even from
binary micro-units, and macro TPMs need not factorize (conditional independence stays a
*checked* property, verified when a measure derives the factored view).

- **Coarse-graining** (Hoel et al. 2013; Hoel et al. 2016) aggregates: a partition of
  the micro-units into macro-units, with a per-macro-unit state grouping, and the macro
  TPM as the fiber average ``D⁻¹ Gᵀ Tᵗ G`` of the (τ-step) micro TPM under maximum-
  entropy perturbation within each fiber.
- **Black-boxing** (Marshall et al. 2018) projects: boxes of micro-units with designated
  output units; the macro state is the output units' state at the end of a τ-step
  window; hidden initial states are uniformly noised and boxes cannot read each other's
  hidden units. Following the oracle, cross-box connections beyond the first micro-step
  are severed entirely, and the black-box TPM is the conditionally independent
  factorization (per-unit marginals) of the window dynamics — the Markovian projection
  Marshall et al. leave implicit and PyPhi makes by construction.

Both transforms are jittable, differentiable in the TPM entries, and batchable over
same-shaped candidate mappings; the mappings themselves (partitions, groupings,
outputs, τ) are static structure.
"""

import math

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from iitx.states import all_states, radix_weights
from iitx.system import System, node_tpms

__all__ = ["black_box", "black_box_state", "coarse_grain", "coarse_grain_state"]


def coarse_grain(
	system: System,
	partition: tuple[tuple[int, ...], ...],
	groupings: tuple[tuple[int, ...], ...],
	steps: int = 1,
) -> System:
	"""Coarse-grain a system into macro-units.

	The macro transition probability is the uniform ("maximum-entropy") average over
	each fiber of micro-states mapping to a macro-state:
	``T_M(m, m') = mean over s in fiber(m) of sum over s' in fiber(m') of T^steps(s, s')``.

	Args:
		system: The micro system.
		partition: Disjoint blocks of micro-units, jointly covering the system; block
			``j`` becomes macro-unit ``j``. Static.
		groupings: For each block, the macro value of each of its joint micro-states
			(indexed little-endian within the block, units in ascending order). Macro
			values must cover ``range(max + 1)``. For the grouping to respect IIT's
			macro-unit doctrine it should not distinguish micro-states that differ only
			by which unit carries which value; the transform itself is general. Static.
		steps: Temporal grain: the number of micro-steps per macro-step.

	Returns:
		The macro system (state-by-state; conditional independence is not implied — it
		is checked downstream when a measure derives the factored view).

	"""
	blocks = _validated_partition(system, partition)
	mapping, macro_shape = _grouping_map(system.shape, blocks, groupings)

	num_macro = math.prod(macro_shape)
	members = jnp.asarray(mapping[:, None] == np.arange(num_macro)[None, :], dtype=system.tpm.dtype)
	stepped = jnp.linalg.matrix_power(system.tpm, steps)
	# D^-1 G^T T^t G: average rows within each source fiber, sum columns per target fiber.
	fiber_sizes = members.sum(axis=0)
	macro_tpm = (members.T @ stepped @ members) / fiber_sizes[:, None]
	return System(tpm=macro_tpm, shape=macro_shape)


def coarse_grain_state(
	system: System,
	partition: tuple[tuple[int, ...], ...],
	groupings: tuple[tuple[int, ...], ...],
	state: Int[Array, " n"],
) -> Int[Array, " k"]:
	"""Map a micro state to its macro state under a coarse-graining.

	Args:
		system: The micro system.
		partition: The unit partition of the coarse-graining.
		groupings: The state groupings of the coarse-graining.
		state: Micro state, shape ``(n,)``.

	Returns:
		Macro state, shape ``(k,)`` for ``k`` macro-units.

	"""
	values = []
	for block, grouping in zip(partition, groupings, strict=True):
		units = sorted(block)
		weights = radix_weights(tuple(system.shape[u] for u in units))
		index = sum(state[u] * int(w) for u, w in zip(units, weights, strict=True))
		values.append(jnp.asarray(grouping)[index])
	return jnp.stack(values)


def black_box(
	system: System,
	partition: tuple[tuple[int, ...], ...],
	outputs: tuple[tuple[int, ...], ...],
	steps: int = 1,
) -> System:
	"""Black-box a system: hide within-box micro-units behind designated outputs.

	The macro state is the joint state of the output units at the end of a ``steps``-
	long micro window. Within the window, boxes may read other boxes' outputs only at
	the first micro-step (later steps sever all cross-box connections); hidden units'
	initial states are uniformly noised; and the resulting window dynamics is factorized
	per unit — all following the oracle's semantics for the conventions the papers leave
	open (``docs/notes/macro.md`` §6).

	Args:
		system: The micro system. Must be conditionally independent (the window is
			composed from its factored view).
		partition: Disjoint boxes of micro-units, jointly covering the system. Static.
		outputs: For each box, its designated output units (a nonempty subset of the
			box). Hidden units are the rest. Static.
		steps: Temporal grain: the number of micro-steps per macro-step.

	Returns:
		The macro system over the output units (in ascending micro-unit order), with
		their micro alphabets.

	"""
	blocks = _validated_partition(system, partition)
	box_of = {}
	for box_index, block in enumerate(blocks):
		for unit in block:
			box_of[unit] = box_index
	output_units = sorted(
		unit
		for box_index, box in enumerate(outputs)
		for unit in _validated_outputs(blocks, box_index, box)
	)
	hidden_units = [u for u in range(system.n) if u not in output_units]

	factors = node_tpms(system)

	def noised(sever_outputs: bool) -> System:
		severed = []
		for j, factor in enumerate(factors):
			out = factor
			for i in range(system.n):
				cross_box = box_of[i] != box_of[j]
				hidden = i not in output_units
				if cross_box and (hidden or sever_outputs):
					out = out.mean(axis=i, keepdims=True)
			severed.append(jnp.broadcast_to(out, factor.shape))
		return System.from_node_tpms(tuple(severed))

	window = noised(sever_outputs=False).tpm
	if steps > 1:
		window = window @ jnp.linalg.matrix_power(noised(sever_outputs=True).tpm, steps - 1)

	# Factorize the window dynamics per unit (the oracle's Markovian projection), noise
	# the hidden initial states, and keep the output units.
	window_factors = node_tpms(System(tpm=window, shape=system.shape), check_independence=False)
	macro_factors = []
	for j in output_units:
		factor = window_factors[j]
		for i in hidden_units:
			factor = factor.mean(axis=i, keepdims=True)
		# Constant along hidden axes now; drop them (in descending order).
		for i in sorted(hidden_units, reverse=True):
			factor = jnp.take(factor, 0, axis=i)
		macro_factors.append(factor)
	return System.from_node_tpms(tuple(macro_factors))


def black_box_state(
	outputs: tuple[tuple[int, ...], ...], state: Int[Array, " n"]
) -> Int[Array, " k"]:
	"""Map a micro state to its black-box macro state (the outputs' projection).

	Args:
		outputs: The per-box output units of the black-boxing.
		state: Micro state, shape ``(n,)``.

	Returns:
		Macro state: the state of the output units in ascending micro-unit order.

	"""
	units = sorted(unit for box in outputs for unit in box)
	return jnp.asarray(state)[jnp.asarray(units)]


def _validated_partition(system: System, partition: tuple[tuple[int, ...], ...]) -> list[list[int]]:
	"""Check that a partition's blocks are disjoint and cover the system.

	Args:
		system: The system being transformed.
		partition: The candidate partition.

	Returns:
		The blocks with sorted units.

	Raises:
		ValueError: If the blocks overlap or do not cover every unit.

	"""
	units = [unit for block in partition for unit in block]
	if sorted(units) != list(range(system.n)):
		msg = (
			f"the partition must cover every unit of the system exactly once, "
			f"got blocks {partition} for {system.n} units"
		)
		raise ValueError(msg)
	return [sorted(block) for block in partition]


def _validated_outputs(
	blocks: list[list[int]], box_index: int, box_outputs: tuple[int, ...]
) -> tuple[int, ...]:
	"""Check that a box's outputs are a nonempty subset of the box.

	Args:
		blocks: The boxes.
		box_index: Which box.
		box_outputs: Its declared output units.

	Returns:
		The output units.

	Raises:
		ValueError: If the outputs are empty or not contained in the box.

	"""
	if not box_outputs or not set(box_outputs) <= set(blocks[box_index]):
		msg = (
			f"each box needs at least one output unit from within the box; box "
			f"{blocks[box_index]} declared outputs {box_outputs}"
		)
		raise ValueError(msg)
	return box_outputs


def _grouping_map(
	shape: tuple[int, ...],
	blocks: list[list[int]],
	groupings: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, tuple[int, ...]]:
	"""Build the micro-state to macro-state index map of a coarse-graining.

	Args:
		shape: Micro per-unit alphabet sizes.
		blocks: The (sorted) unit blocks.
		groupings: Per-block macro values of each joint block-state.

	Returns:
		The macro state index of every micro state (shape ``(Q,)``), and the macro
		alphabet sizes.

	Raises:
		ValueError: If a grouping's length does not match its block's state count, or
			its values do not cover a full alphabet.

	"""
	states = all_states(shape)
	macro_shape = []
	macro_digits = []
	for block, grouping in zip(blocks, groupings, strict=True):
		table = np.asarray(grouping, dtype=np.int64)
		block_states = math.prod(shape[u] for u in block)
		if table.shape != (block_states,):
			msg = (
				f"block {tuple(block)} has {block_states} joint states, but its grouping "
				f"has shape {table.shape}"
			)
			raise ValueError(msg)
		if sorted(set(table.tolist())) != list(range(table.max() + 1)):
			msg = f"grouping values must cover range({table.max() + 1}), got {sorted(set(table))}"
			raise ValueError(msg)
		weights = radix_weights(tuple(shape[u] for u in block))
		block_index = states[:, block] @ weights
		macro_digits.append(table[block_index])
		macro_shape.append(int(table.max()) + 1)

	mapping = np.zeros(len(states), dtype=np.int64)
	weights = radix_weights(tuple(macro_shape))
	for digit, weight in zip(macro_digits, weights, strict=True):
		mapping += digit * int(weight)
	return mapping, tuple(macro_shape)
