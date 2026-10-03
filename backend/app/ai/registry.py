"""Provider registry. To add a provider: implement AIProvider and call `register("name", factory)` here."""
from collections.abc import Callable

from app.ai.base import AIProvider
from app.ai.gemini import GeminiProvider
from app.core.config import Settings

_FACTORIES: dict[str, Callable[[Settings], AIProvider]] = {"gemini": lambda s: GeminiProvider(s)}


def register(name: str, factory: Callable[[Settings], AIProvider]) -> None:
    _FACTORIES[name] = factory


def unregister(name: str) -> None:          # tests
    _FACTORIES.pop(name, None)


def names() -> list[str]:
    return sorted(_FACTORIES)


def build(name: str, settings: Settings) -> AIProvider:
    if name not in _FACTORIES:
        raise KeyError(name)
    return _FACTORIES[name](settings)


def describe(settings: Settings) -> list[dict]:
    out = []
    for n in names():
        p = build(n, settings)
        out.append(dict(name=n, display_name=p.display_name, configured=p.is_configured(), model=p.model))
    return out
