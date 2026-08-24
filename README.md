# iitx

Integrated Information Theory accelerated in [JAX](https://github.com/jax-ml/jax).

`iitx` computes integrated information (Φ) and the full cause-effect structure of IIT —
versions 3.0 and 4.0 — for any finite discrete dynamical system given as a transition
probability matrix: binary or multi-valued units, deterministic or stochastic dynamics,
micro or macro (coarse-grained, black-boxed) levels. Unlike the reference implementation
[PyPhi](https://github.com/wmayner/pyphi), which it reproduces number-for-number on the
canonical examples, `iitx` exposes the computation to the accelerator: the combinatorial
axes of one system are batched tensor work, many systems are one `jax.vmap` call, and
the IIT 4.0 pipeline is differentiable with respect to the system's dynamics.

Under construction — see `docs/design.md` for the design and `CHANGELOG.md` for status.

## Installation

```sh
uv pip install iitx
```

## Citing iitx

Citation information will accompany the first release.
