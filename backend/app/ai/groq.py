"""Groq adapter (OpenAI-compatible chat completions with image input). Used as a generative fallback after Gemini; all logic is in ai/openai_compat.py.
NOT registered in ai/registry.py (admins cannot select it as the main provider); the workflow calls `build()`. The key is never logged."""
import logging
from collections.abc import Callable

from app.ai.openai_compat import OpenAICompatProvider
from app.core.config import Settings


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    display_name = "Groq"
    env_name = "GROQ_API_KEY"
    log = logging.getLogger("agroai.ai.groq")

    def _key(self) -> str:
        return self._s.groq_api_key.get_secret_value()

    def _model_name(self) -> str:
        return self._s.groq_model

    def _base_url(self) -> str:
        return self._s.groq_base_url


_factory: Callable[[Settings], GroqProvider] = lambda s: GroqProvider(s)


def build(settings: Settings) -> GroqProvider:
    return _factory(settings)
