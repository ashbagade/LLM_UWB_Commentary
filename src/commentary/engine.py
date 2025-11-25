from __future__ import annotations

from typing import List, Dict

from src.commentary.llm_client import LLMClient
from src.commentary.prompting import build_commentary_prompt
from src.state.cricket_state import MatchState
from src.events.schema import UmpireEvent


class CommentaryEngine:
    """
    High-level interface for generating commentary for detected events.

    It maintains a chat-style history with the LLM so that the model can
    build up context and personality over the course of a match.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        history_limit: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.history_limit = history_limit

        # Initialize conversation history with system prompt
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def generate_for_event(self, state: MatchState, event: UmpireEvent) -> str:
        """
        Generate commentary for a single new event, given the current match state.

        The generated text is appended to the conversation history so that
        future calls can build on prior commentary.
        """
        # Build user message for this event
        new_messages = build_commentary_prompt(state, event)

        # Compose full message list: system + prior history + new user message(s)
        messages = self.history + new_messages

        # Call LLM client (Gemini under the hood)
        text = self.llm_client.generate(messages)

        # Append assistant reply to history and trim
        self.history.append({"role": "assistant", "content": text})
        if len(self.history) > self.history_limit:
            # Always keep the initial system prompt at index 0
            trimmed = self.history[1:][-(self.history_limit - 1):]
            self.history = [self.history[0]] + trimmed

        return text
