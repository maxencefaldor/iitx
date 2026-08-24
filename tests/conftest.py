"""Test-suite configuration.

Enables float64 before any array is created: oracle golden numbers are float64
quantities compared at the oracle's precision (down to 1e-13 for IIT 4.0), and the
round-then-compare tie semantics are only meaningful at double precision. The library
itself never touches this flag (``docs/design.md`` §2, P6).
"""

import jax

jax.config.update("jax_enable_x64", True)
