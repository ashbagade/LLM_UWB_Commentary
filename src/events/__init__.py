"""
Event abstractions and detection logic.

- schema: core EventType enum and UmpireEvent dataclass.
- detector: utilities to convert noisy window-level predictions
            into discrete umpire events.
"""

from .schema import EventType, UmpireEvent  # noqa: F401