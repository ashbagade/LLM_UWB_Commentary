from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.events.schema import EventType, UmpireEvent


@dataclass
class MatchState:
    """
    Minimal cricket match state used for commentary.

    This is intentionally approximate and focuses on the pieces that
    are most useful for LLM commentary rather than full cricket rules.
    """

    innings: int = 1
    total_runs: int = 0
    wickets: int = 0
    balls_bowled: int = 0  # legal deliveries only (no-balls / wides excluded)
    sixes: int = 0
    fours: int = 0
    events: List[UmpireEvent] = field(default_factory=list)

    def overs_as_float(self) -> float:
        """
        Convenience: represent overs as X.Y where X is complete overs
        and Y is balls in the current over (0-5).
        """
        complete_overs = self.balls_bowled // 6
        balls_in_over = self.balls_bowled % 6
        return float(f"{complete_overs}.{balls_in_over}")

    def __str__(self) -> str:
        return (
            f"MatchState(innings={self.innings}, runs={self.total_runs}, "
            f"wickets={self.wickets}, balls={self.balls_bowled}, "
            f"4s={self.fours}, 6s={self.sixes})"
        )


# ----------------------------------------------------------------------
# Approximate scoring logic
# ----------------------------------------------------------------------

# Approximate run values for each umpire signal.
# NOTE: This is intentionally simplified and does not attempt to model
# every nuance of cricket law.
RUN_VALUES = {
    EventType.BOUNDARY4: 4,
    EventType.BOUNDARY6: 6,
    EventType.CANCELCALL: 0,   # cancel previous signal; we treat as no-op here
    EventType.DEADBALL: 0,
    EventType.LEGBYE: 1,       # in reality can be 1+; we approximate as 1
    EventType.NOBALL: 1,       # extra run; ball is not counted as legal delivery
    EventType.OUT: 0,
    EventType.PENALTYRUN: 5,   # often 5 penalty runs; approximation
    EventType.SHORTRUN: 0,     # in reality reduces prior runs; we treat as 0 here
    EventType.WIDE: 1,         # extra run; ball is not counted as legal delivery
}

# Whether an event corresponds to a legal ball that should increment balls_bowled.
# We consider legal deliveries to be those that count towards the over:
# boundaries, leg byes, outs, etc.
LEGAL_DELIVERY = {
    EventType.BOUNDARY4: True,
    EventType.BOUNDARY6: True,
    EventType.CANCELCALL: False,
    EventType.DEADBALL: False,
    EventType.LEGBYE: True,
    EventType.NOBALL: False,       # no-balls do not count as legal balls
    EventType.OUT: True,
    EventType.PENALTYRUN: False,   # penalty runs can be independent of a ball
    EventType.SHORTRUN: True,      # approximate
    EventType.WIDE: False,         # wides do not count as legal balls
}


def apply_event_to_state(state: MatchState, event: UmpireEvent) -> MatchState:
    """
    Update the given MatchState in-place based on a single UmpireEvent.

    Returns the same state instance (for chaining).
    """
    etype = event.event_type

    # 1) Runs
    runs = RUN_VALUES.get(etype, 0)
    state.total_runs += runs

    # 2) Fours / Sixes
    if etype == EventType.BOUNDARY4:
        state.fours += 1
    elif etype == EventType.BOUNDARY6:
        state.sixes += 1

    # 3) Wickets
    if etype == EventType.OUT:
        state.wickets += 1

    # 4) Balls bowled (legal deliveries only)
    if LEGAL_DELIVERY.get(etype, False):
        state.balls_bowled += 1

    # 5) Record event in history
    state.events.append(event)

    # NOTE: We treat CANCELCALL as a no-op for the scoreboard. In a more
    # detailed implementation, you might want to "undo" the last event.
    return state


def build_state_from_events(events: List[UmpireEvent]) -> MatchState:
    """
    Reconstruct a MatchState from a list of events, applying them in
    chronological order.
    """
    state = MatchState()
    for e in sorted(events, key=lambda ev: ev.timestamp):
        apply_event_to_state(state, e)
    return state


if __name__ == "__main__":
    from src.events.schema import EventType, UmpireEvent

    # Scenario: 2 × boundary6 + 1 × out → 12 runs, 1 wicket, 3 balls.
    evs = [
        UmpireEvent(timestamp=0.0, event_type=EventType.BOUNDARY6, confidence=1.0),
        UmpireEvent(timestamp=1.0, event_type=EventType.BOUNDARY6, confidence=0.95),
        UmpireEvent(timestamp=2.0, event_type=EventType.OUT, confidence=0.99),
    ]

    s = build_state_from_events(evs)
    print("Self-test state:", s)

    assert s.total_runs == 12, f"Expected 12 runs, got {s.total_runs}"
    assert s.wickets == 1, f"Expected 1 wicket, got {s.wickets}"
    assert s.balls_bowled == 3, f"Expected 3 balls, got {s.balls_bowled}"

    print("Simple self-test passed.")
