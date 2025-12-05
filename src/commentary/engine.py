from __future__ import annotations

from typing import List, Dict

from src.commentary.llm_client import LLMClient
from src.commentary.prompting import build_commentary_prompt
from src.state.cricket_state import MatchState
from src.events.schema import UmpireEvent


class CommentaryEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        history_limit: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.history_limit = history_limit

        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def generate_for_event(self, state: MatchState, event: UmpireEvent) -> str:
        """
        Generate commentary for a single new event, given the current match state.

        The generated text is appended to the conversation history so that
        future calls can build on prior commentary.
        """
        new_messages = build_commentary_prompt(state, event)

        messages = self.history + new_messages

        text = self.llm_client.generate(messages)

        self.history.append({"role": "assistant", "content": text})
        if len(self.history) > self.history_limit:
            trimmed = self.history[1:][-(self.history_limit - 1):]
            self.history = [self.history[0]] + trimmed

        return text
