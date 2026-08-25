# Integrated Information Theory accelerated in JAX ⚡

<div align="center">
	<a href="https://pypi.org/project/iitx/"><img src="https://img.shields.io/pypi/v/iitx.svg" alt="PyPI"></a>
	<a href="https://github.com/maxencefaldor/iitx/actions/workflows/tests.yml"><img src="https://github.com/maxencefaldor/iitx/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
	<a href="https://maxencefaldor.github.io/iitx"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="docs"></a>
	<a href="https://github.com/maxencefaldor/iitx/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license"></a>
	<a href="https://github.com/maxencefaldor/iitx"><img src="https://img.shields.io/badge/python-3.14%2B-blue.svg" alt="python"></a>
</div>

`iitx` computes integrated information — φ_s, big Φ, and the full cause-effect structure
of IIT 3.0 and 4.0 — for any finite discrete dynamical system given as a transition
probability matrix. It reproduces the reference implementation
[PyPhi](https://github.com/wmayner/pyphi)'s numbers on the canonical examples, and does
four things PyPhi cannot:

- 🚀 **Parallelize within a system** — candidate systems, mechanisms, purviews,
  partitions, and cuts are batched tensor work on CPU, GPU, or TPU, not Python loops.
- 📦 **Batch across systems** — every analysis is a pure function of `(system, state)`
  that composes with `jax.vmap`: ten thousand systems are one call. On a laptop CPU,
  that is ~100,000 exact φ_s evaluations per second at n = 3 — enough to evaluate
  *every* deterministic 3-unit system, exhaustively, in three minutes.
- 🧠 **Differentiate** — the IIT 4.0 pipeline is differentiable with respect to the
  system's dynamics, almost everywhere and with exact subgradients: gradient ascent on
  integrated information itself, a first.
- 🧩 **Treat the measure as a choice** — IIT 3.0, IIT 4.0 (2023), and IIT 4.0 (2026)
  are pluggable measures over one substrate, so the same system can be scored under
  competing formalizations in one line.

## ✨ Overview

- **General.** Binary or multi-valued units, uniform or heterogeneous alphabets,
  deterministic or stochastic dynamics. Coarse-graining and black-boxing are pure
  `System → System` transforms, so macro-level systems feed the unchanged pipeline.
- **Faithful.** PyPhi (pinned) is the frozen test oracle: the regression suite compares
  values against generated golden fixtures for both theory versions, including
  non-binary systems, and every deliberate divergence is a documented finding
  (`docs/notes/oracle-findings.md`), never a silent patch.
- **Honest about exactness.** Everything is exact; the IIT 3.0 earth mover's distances
  are host linear programs (so 3.0 jits and vmaps but refuses `grad` loudly), while the
  IIT 4.0 intrinsic-difference pipeline is pure tensor algebra end to end. Named,
  clearly-labeled relaxations (`iitx.relax`) exist for optimization landscapes.

## 📦 Installation

```sh
pip install iitx
```

Requires Python ≥ 3.14. Dependencies: `jax`, `numpy`, and `POT` (the exact
earth-mover's-distance backend for IIT 3.0).

## 🚀 Getting started

```python
import jax
import jax.numpy as jnp

from iitx.measures import iit4
from iitx.system import System

# The standard 3-unit example (OR, COPY, XOR), as a state-by-node TPM.
tpm = jnp.asarray(
	[
		[0.0, 0.0, 0.0],
		[0.0, 0.0, 1.0],
		[1.0, 0.0, 1.0],
		[1.0, 0.0, 0.0],
		[1.0, 1.0, 0.0],
		[1.0, 1.0, 1.0],
		[1.0, 1.0, 1.0],
		[1.0, 1.0, 0.0],
	]
)
system = System.from_state_by_node(tpm)
state = jnp.asarray([1, 0, 0])

# System integrated information and the full Φ-structure.
analysis = iit4.system_phi(system, state)
structure = iit4.phi_structure(system, state)
print(analysis.phi, structure.big_phi)  # 0.415... and 1.0

# Batch over all 8 states of the system in one call.
states = jnp.stack([jnp.asarray([s >> i & 1 for i in range(3)]) for s in range(8)])
batched = jax.vmap(iit4.system_phi, in_axes=(None, 0))(system, states)

# Differentiate φ_s with respect to the dynamics.
gradient = jax.grad(
	lambda tpm: iit4.system_phi(System(tpm, system.shape, system.cm), state).signed_phi
)(system.tpm)
```

The full documentation lives at
[maxencefaldor.github.io/iitx](https://maxencefaldor.github.io/iitx).

## 📓 Examples

| Notebook | What it shows |
| --- | --- |
| [Getting started](examples/00_getting_started.ipynb) | Systems, φ_s, Φ-structures, IIT 3.0, batching, non-binary units, macro levels |
| [Ascending Φ](examples/01_ascending_phi.ipynb) | Gradient ascent on integrated information itself — to our knowledge a first — with structure emerging along the way |

## 🔬 Research

`iitx` is also a lab. The
[`experiments/`](https://github.com/maxencefaldor/iitx/tree/main/experiments)
directory holds a running research program on the gradients and landscapes of Φ —
executed notebooks with pre-registered predictions, in reading order. Highlights so
far: an audit showing exact φ_s gradients match finite differences to 10⁻¹⁰
(refuting the field's non-differentiability claim), the **first exhaustive Φ
landscape** (all 16.7M deterministic 3-unit systems, oracle-verified), the discovery
that the 2023-formalism maximum is a frozen state reachable only from itself and
grows as n(n−1), and that the 2026 revision zeroes the entire deterministic
universe. A paper draft distilling the story lives in
[`paper/`](https://github.com/maxencefaldor/iitx/tree/main/paper).

## 📐 Design

The design documents are part of the repository: `docs/design.md` (the core
abstractions and why), `docs/generality.md` (the PyPhi-vs-iitx capability table), and
`docs/glossary.md` (one vocabulary, used identically in code and papers). Everything
learned the hard way lives in `docs/notes/`.

## 📖 Citing iitx

If you use `iitx` in your research, please cite it (see also
[`CITATION.cff`](https://github.com/maxencefaldor/iitx/blob/main/CITATION.cff)):

```bibtex
@software{faldor2026iitx,
	author = {Faldor, Maxence},
	title = {iitx: Integrated Information Theory accelerated in JAX},
	url = {https://github.com/maxencefaldor/iitx},
	year = {2026},
}
```
