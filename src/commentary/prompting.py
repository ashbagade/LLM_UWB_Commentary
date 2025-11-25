from __future__ import annotations

from typing import List, Dict

from src.state.cricket_state import MatchState
from src.events.schema import UmpireEvent


def _format_event_brief(e: UmpireEvent) -> str:
    """
    Short human-readable description of an event, for use in the prompt.
    """
    label = e.event_type.value
    return f"t={e.timestamp:.1f}s: {label} (conf={e.confidence:.2f})"


def build_commentary_prompt(state: MatchState, event: UmpireEvent) -> List[Dict[str, str]]:
    """
    Build a single user message instructing the LLM to generate live commentary
    for the NEW event, given the current match state and recent history.

    Returns
    -------
    messages : list[dict]
        A list of chat-style messages (usually just one user message) to be
        appended after the system message and prior assistant messages.
    """
    # Score summary
    overs_float = state.overs_as_float()
    score_line = (
        f"Innings: {state.innings}, Score: {state.total_runs}/{state.wickets} "
        f"after {overs_float} overs (balls={state.balls_bowled}). "
        f"4s: {state.fours}, 6s: {state.sixes}."
    )

    # Recent history: use prior events, excluding the current one
    prior_events = [e for e in state.events if e.timestamp < event.timestamp]
    recent_events = prior_events[-5:]  # last 5 events for context

    if recent_events:
        history_lines = "\n".join(
            f"- {_format_event_brief(e)}" for e in recent_events
        )
    else:
        history_lines = "(no prior events)"

    # New event description
    new_event_desc = _format_event_brief(event)

    # High-level user instruction
    instructions = (
        "You are a lively cricket commentator calling a live match. "
        "Given the current score, recent events, and the NEW umpire signal, "
        "produce 1–2 sentences of live commentary for ONLY the new event. "
        "Do not explain the rules; speak as if on a broadcast. "
        "You may be enthusiastic, but stay concise and avoid emojis."
    )

    content = (
        f"{instructions}\n\n"
        f"Current match state:\n"
        f"- {score_line}\n\n"
        f"Recent events:\n"
        f"{history_lines}\n\n"
        f"New event to comment on:\n"
        f"- {new_event_desc}\n"
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]
