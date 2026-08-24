"""Measures of integrated information.

A measure is a first-class object mapping ``(system, state)`` to integrated-information
results (``docs/design.md`` §5). The shipped measures are the canonical configurations of
IIT 4.0 (:mod:`iitx.measures.iit4`) and IIT 3.0 (:mod:`iitx.measures.iit3`); alternative
definitions of Φ plug in beside them without touching the core.
"""

from iitx.measures import iit3, iit4
from iitx.measures.measure import IIT3, IIT4, Measure

__all__ = ["IIT3", "IIT4", "Measure", "iit3", "iit4"]
