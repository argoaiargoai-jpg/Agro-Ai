"""Which specialist providers exist and which of them are configured. Tests register fakes with `register_*`."""
from collections.abc import Callable

from app.ai.specialists.base import DiseaseDiagnoser, PlantIdentifier
from app.ai.specialists.kindwise import KindwiseProvider
from app.ai.specialists.plantix import PlantixProvider
from app.ai.specialists.plantnet import PlantNetProvider
from app.core.config import Settings

_IDENTIFIERS: dict[str, Callable[[Settings], PlantIdentifier]] = {"plantnet": lambda s: PlantNetProvider(s)}
_DIAGNOSERS: dict[str, Callable[[Settings], DiseaseDiagnoser]] = {"plantix": lambda s: PlantixProvider(s), "kindwise": lambda s: KindwiseProvider(s)}


def register_identifier(name: str, factory: Callable[[Settings], PlantIdentifier]) -> None:
    _IDENTIFIERS[name] = factory


def register_diagnoser(name: str, factory: Callable[[Settings], DiseaseDiagnoser]) -> None:
    _DIAGNOSERS[name] = factory


def unregister(name: str) -> None:          # tests
    _IDENTIFIERS.pop(name, None)
    _DIAGNOSERS.pop(name, None)


def identifiers(settings: Settings) -> list[PlantIdentifier]:
    return [p for p in (f(settings) for f in _IDENTIFIERS.values()) if p.is_configured()]


def diagnosers(settings: Settings) -> list[DiseaseDiagnoser]:
    return [p for p in (f(settings) for f in _DIAGNOSERS.values()) if p.is_configured()]
