"""Provider-neutral shapes and errors for the specialist providers. Detail strings are for server logs only and must never contain secrets."""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PlantMatch:
    name: str                      # best human-readable name (common name if known, else scientific)
    score: float                   # 0..1
    scientific: str | None = None


@dataclass
class DiseaseMatch:
    name: str
    probability: float             # 0..1 (approximate for providers that return qualitative likelihoods)


@dataclass
class Diagnosis:
    is_plant: bool | None = None
    crops: list[PlantMatch] = field(default_factory=list)
    diseases: list[DiseaseMatch] = field(default_factory=list)
    healthy_probability: float | None = None


class SpecialistError(Exception):
    """kind: not_configured | unauthorized | rate_limited | unavailable | timeout | bad_response"""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind, self.detail = kind, detail


def clamp01(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def clean(text, n: int = 80) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:n]


class PlantIdentifier(ABC):
    name = ""

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def identify(self, image: bytes, mime: str) -> list[PlantMatch]:
        """Best matches first. Empty list = nothing identified. Raises SpecialistError."""


class DiseaseDiagnoser(ABC):
    name = ""

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def diagnose(self, image: bytes, mime: str) -> Diagnosis:
        """Raises SpecialistError."""
