"""The two temporal directions of causal analysis.

Every quantity that exists on both temporal sides — repertoires, integrated information,
purviews — is one function taking a :class:`Direction`, so cause and effect mirror each
other structurally rather than by parallel code paths (``docs/glossary.md``, naming
rule 1).
"""

import enum

__all__ = ["Direction"]


class Direction(enum.Enum):
	"""Temporal direction of a causal quantity.

	``CAUSE`` looks one step into the past (what states could have led here); ``EFFECT``
	looks one step into the future (what states this leads to). A direction is static
	configuration — pass it as a Python value, never as a traced array.
	"""

	CAUSE = "cause"
	EFFECT = "effect"
