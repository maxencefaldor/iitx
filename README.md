# iitx

<div align="center">
	<a href="https://github.com/maxencefaldor/iitx/actions/workflows/tests.yml"><img src="https://github.com/maxencefaldor/iitx/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
	<a href="https://github.com/maxencefaldor/iitx/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license"></a>
	<a href="https://github.com/maxencefaldor/iitx"><img src="https://img.shields.io/badge/python-3.14%2B-blue.svg" alt="python"></a>
</div>

**Integrated Information Theory, accelerated in [JAX](https://github.com/jax-ml/jax).** ⚡

`iitx` computes integrated information — φ_s, big Φ, and the full cause-effect structure
of IIT 3.0 and 4.0 — for any finite discrete dynamical system given as a transition
probability matrix. It reproduces the reference implementation
[PyPhi](https://github.com/wmayner/pyphi)'s numbers on the canonical examples, and does
three things PyPhi cannot:

- **Parallelize within a system** 🚀 — candidate systems, mechanisms, purviews,
  partitions, and cuts are batched tensor work on CPU, GPU, or TPU, not Python loops.
- **Batch across systems** 📦 — every analysis is a pure function of `(system, state)`
  that composes with `jax.vmap`: ten thousand systems are one call.
- **Differentiate** 🎯 — the IIT 4.0 pipeline is differentiable with respect to the
  system's dynamics, almost everywhere and with exact subgradients: gradient ascent on
  integrated information itself.

## Overview

- **General.** Binary or multi-valued units, uniform or heterogeneous alphabets,
  deterministic or stochastic dynamics. Coarse-graining and black-boxing are pure
  `System → System` transforms, so macro-level systems feed the unchanged pipeline.
- **Faithful.** PyPhi (pinned) is the frozen test oracle: the regression suite compares
  values against generated golden fixtures for both theory versions, including
  non-binary systems, and every deliberate divergence is a documented finding
  (`docs/notes/oracle-findings.md`), never a silent patch.
- **Honest about exactness.** Everything is exact; the IIT 3.0 earth mover's distances
  are host linear programs (so 3.0 jits and vmaps but refuses `grad` loudly), while the
  IIT 4.0 intrinsic-difference pipeline is pure tensor algebra end to end.

## Getting started

```python
import jax
import jax.numpy as jnp

from iitx.measures import iit4
from iitx.system import System

# The standard 3-unit example (OR, COPY, XOR), as a state-by-node TPM.
tpm = jnp.asarray([
	[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
	[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0],
])
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

## Installation

```sh
uv pip install git+https://github.com/maxencefaldor/iitx.git
```

Requires Python ≥ 3.14. Dependencies: `jax`, `numpy`, and `POT` (the exact
earth-mover's-distance backend for IIT 3.0).

## Examples

| Notebook | What it shows |
| --- | --- |
| [Getting started](examples/00_getting_started.ipynb) | Systems, φ_s, Φ-structures, IIT 3.0, batching, non-binary units, macro levels |
| [Ascending Φ](examples/01_ascending_phi.ipynb) | Gradient ascent on integrated information itself — to our knowledge a first — with structure emerging along the way |

## Design

The design documents are part of the repository: `docs/design.md` (the core
abstractions and why), `docs/generality.md` (the PyPhi-vs-iitx capability table), and
`docs/glossary.md` (one vocabulary, used identically in code and papers). Everything
learned the hard way lives in `docs/notes/`.

## Citing iitx

Citation information will accompany the first release.

## Contributing

Contributions are welcome — see [CONTRIBUTING](contributing.md). Start from the
design documents; the tests define correctness.

## License

[MIT](https://github.com/maxencefaldor/iitx/blob/main/LICENSE)
