"""
State tracking for cricket matches.

Currently includes:

- cricket_state: MatchState dataclass and helpers for updating
  state from detected UmpireEvents.
"""

from .cricket_state import MatchState, apply_event_to_state, build_state_from_events  # noqa: F401
