"""Pollinations adapter (OpenAI-compatible chat completions with image input). Used as a generative fallback after Gemini; all logic is in ai/openai_compat.py.
NOT registered in ai/registry.py (admins cannot select it as the main provider); the workflow calls `build()`. The key is never logged."""
import logging
from collections.abc import Callable

from app.ai.openai_compat import OpenAICompatProvider
from app.core.config import Settings


class PollinationsProvider(OpenAICompatProvider):
    name = "pollinations"
    display_name = "Pollinations"
    env_name = "POLLINATIONS_API_KEY"
    log = logging.getLogger("agroai.ai.pollinations")

    def _key(self) -> str:
        return self._s.pollinations_api_key.get_secret_value()

    def _model_name(self) -> str:
        return self._s.pollinations_model

    def _base_url(self) -> str:
        return self._s.pollinations_base_url


_factory: Callable[[Settings], PollinationsProvider] = lambda s: PollinationsProvider(s)


def build(settings: Settings) -> PollinationsProvider:
    return _factory(settings)
