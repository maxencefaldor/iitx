# Changelog

All notable changes to `iitx` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/) once released.

## [Unreleased]

## [0.1.0] - 2026-08-25

First release: the exact core, verified against PyPhi.

### Added

- Study notes on IIT 3.0, IIT 4.0, the intrinsic difference measure, macro-level
  analysis, alternative measures, JAX engineering constraints, and the PyPhi reference
  implementation (`docs/notes/`), plus a rigorous PyPhi review and a literature review
  on gradient-based Φ optimization.
- Design documents: generality checklist (`docs/generality.md`), core design
  (`docs/design.md`), and glossary (`docs/glossary.md`).
- Package scaffold: Python ≥ 3.14, hatchling, ruff, ty, pytest; runtime dependencies
  `jax`, `numpy`, `pot`.
- Core: `System` (state-by-state TPMs over heterogeneous alphabets), little-endian
  mixed-radix state indexing, canonical enumeration tables, the repertoire algebra with
  full-shape embedding, and the intrinsic-difference and exact-EMD distances.
- IIT 4.0 (`iitx.measures.iit4`): φ_s with the directional system partitions,
  distinctions over the disintegrating partitions, analytic relations, Φ-structures,
  and recursive complex condensation — differentiable almost everywhere, with golden
  parity against PyPhi at 1e-9.
- IIT 3.0 (`iitx.measures.iit3`): concepts, cause-effect structures, big Φ via the
  extended EMD, and the major complex — with golden parity at the oracle's own
  precision.
- Macro transforms (`iitx.transforms`): coarse-graining and black-boxing as pure
  `System → System` functions.
- The `Measure` protocol with `IIT3` and `IIT4` measures.
- Oracle regression harness: pinned PyPhi `main` environment, fixture generator, and
  golden fixtures for both theory versions including non-binary systems
  (`tests/oracle/`).
- Documentation: MkDocs Material site with API reference, README, contributing guide.
- The IIT 4.0 (2026) variant (`version="2026"`): the intrinsic-information cap on φ_s,
  applied inside tie resolution as the oracle does, with its own 11 oracle fixtures.
- Named relaxations (`iitx.relax`, `iitx.distances.sinkhorn_emd`): differentiable
  approximations, explicitly named, converging to the exact quantities.
- Example notebooks: `00_getting_started` and `01_ascending_phi` — gradient ascent on
  integrated information itself, with emergent Φ-structure.

[Unreleased]: https://github.com/maxencefaldor/iitx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/maxencefaldor/iitx/releases/tag/v0.1.0
