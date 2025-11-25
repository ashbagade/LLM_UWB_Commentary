from __future__ import annotations

import os
from typing import List, Dict, Optional

from google import genai
from google.genai import types

# Default to a cheap, stable text model.
# You can override this in code or via GEMINI_MODEL_NAME env var.
DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")


class LLMClient:
    """
    Thin wrapper around the Gemini client.

    Usage:
        llm = LLMClient(system_prompt="You are a cricket commentator")
        text = llm.generate(
            [{"role": "user", "content": "Say hello"}]
        )
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.system_prompt = system_prompt.strip()
        self.model_name = model_name
        self.debug = debug

        # Prefer explicit key; otherwise let the SDK read GEMINI_API_KEY/GOOGLE_API_KEY.
        if api_key is not None:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def _flatten_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat-style messages into a single text prompt.

        We keep this simple: prepend ROLE: and join with newlines.
        """
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 80,
        temperature: float = 0.8,
    ) -> str:
        """
        Call Gemini and return plain text commentary.

        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}.
        """
        convo_text = self._flatten_messages(messages)

        if self.debug:
            print("\n[LLMClient] === Prompt sent to Gemini ===")
            print(convo_text)
            print("[LLMClient] === End of prompt ===\n")

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=convo_text,
            config=config,
        )

        # Primary path: use the convenience .text property.
        text = (getattr(response, "text", None) or "").strip()

        # Fallback: manually reconstruct from candidates if .text is empty.
        if not text:
            text_parts: List[str] = []
            candidates = getattr(response, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text_parts.append(part_text)
            text = " ".join(text_parts).strip()

        if self.debug:
            print("[LLMClient] Raw response object:", repr(response))
            print("[LLMClient] Extracted text:", repr(text))
            print()

        # If it's still empty, just return empty string.
        # The caller (CommentaryEngine) will still update match state;
        # you’ll see debug output so you can diagnose Gemini issues.
        return text
