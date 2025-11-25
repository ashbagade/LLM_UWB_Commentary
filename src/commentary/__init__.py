"""
Commentary generation components.

- llm_client: Gemini-backed LLM client abstraction.
- prompting: prompt construction utilities.
- engine: CommentaryEngine that combines state, events, and the LLM.
"""

from .llm_client import LLMClient  # noqa: F401
from .engine import CommentaryEngine  # noqa: F401
