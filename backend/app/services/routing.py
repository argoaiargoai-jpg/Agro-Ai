"""Which specialist providers run BEFORE Gemini, chosen by our own model's confidence (the model itself is unchanged).

  confidence <= threshold (0.80):  Kindwise;  if Kindwise is unavailable -> Pl@ntNet
  confidence >  threshold:         Pl@ntNet, then Kindwise (a failing one is skipped, the other still runs)
  nothing available:               straight to Gemini

Gemini then always gets the original image plus whatever evidence was gathered. Unavailable means quota (429), 5xx, timeout, connection failure,
or not configured. Credential/format errors are logged as configuration problems and the provider is skipped the same way (the analysis never
fails because of a specialist). Customers never see this: only generic step names ("identify", "disease", "guidance") reach them.
"""
import logging
from dataclasses import dataclass, field

from app.ai.specialists import registry
from app.ai.specialists.base import DiseaseMatch, PlantMatch, SpecialistError
from app.core.config import Settings

log = logging.getLogger("agroai.routing")
PLANTNET, KINDWISE = "plantnet", "kindwise"
AVAILABILITY_FAILURES = {"rate_limited", "unavailable", "timeout"}          # quota/429, 5xx/connection, timeout
LOW, HIGH = "LOW", "HIGH"


def decide(ml: dict, settings: Settings) -> str:
    """LOW (<= threshold, Kindwise first) or HIGH (> threshold, Pl@ntNet first). Confidence is used for nothing else."""
    return HIGH if float(ml.get("confidence") or 0.0) > settings.route_confidence_threshold else LOW


def planned_steps(route: str, settings: Settings) -> list[str]:
    """Customer-safe steps that will start (providers without a key are not promised). Fallbacks may add one later."""
    pn, kw = registry.get_identifier(PLANTNET, settings), registry.get_diagnoser(KINDWISE, settings)
    if route == HIGH:
        steps = (["identify"] if pn else []) + (["disease"] if kw else [])
    else:
        steps = ["disease"] if kw else (["identify"] if pn else [])
    return steps + ["guidance"]


def any_specialist_configured(settings: Settings) -> bool:
    return bool(registry.get_identifier(PLANTNET, settings) or registry.get_diagnoser(KINDWISE, settings))


@dataclass
class Evidence:
    plants: list[PlantMatch] = field(default_factory=list)
    diseases: list[DiseaseMatch] = field(default_factory=list)
    healthy_probability: float | None = None
    steps_done: list[str] = field(default_factory=list)
    providers: list[dict] = field(default_factory=list)          # internal diagnostics: who was tried, and how it went (never a secret)
    latest: dict | None = None                                  # the most recent SUCCESSFUL specialist result (what we fall back to)

    def internal(self) -> dict:
        return {"providers": self.providers, "plants": [{"name": p.name, "score": round(p.score, 3)} for p in self.plants],
                "diseases": [{"name": d.name, "probability": round(d.probability, 3)} for d in self.diseases], "steps_done": self.steps_done,
                "latest": self.latest and {"provider": self.latest["provider"], "kind": self.latest["kind"]}}


def _failed(ev: Evidence, name: str, step: str, exc: SpecialistError) -> None:
    ev.providers.append({"provider": name, "step": step, "status": exc.kind})
    if exc.kind in AVAILABILITY_FAILURES:
        log.warning("%s (%s) unavailable: %s; falling back", name, step, exc.kind)
    else:
        log.error("%s (%s) failed with %s: this looks like a configuration or request problem (check its key/URL); skipping it", name, step, exc.kind)


def _identify(ev: Evidence, settings: Settings, image: bytes, mime: str, on_stage) -> bool:
    p = registry.get_identifier(PLANTNET, settings)
    if p is None:
        ev.providers.append({"provider": PLANTNET, "step": "identify", "status": "not_configured"})
        return False
    if on_stage:
        on_stage("identify")
    try:
        matches = p.identify(image, mime)
    except SpecialistError as exc:
        _failed(ev, PLANTNET, "identify", exc)
        return False
    ev.providers.append({"provider": PLANTNET, "step": "identify", "status": "ok", "results": len(matches)})
    ev.plants = matches or ev.plants
    ev.steps_done.append("identify")
    ev.latest = {"provider": PLANTNET, "kind": "identify", "plants": matches, "diseases": [], "healthy_probability": None}
    return True


def _diagnose(ev: Evidence, settings: Settings, image: bytes, mime: str, on_stage) -> bool:
    p = registry.get_diagnoser(KINDWISE, settings)
    if p is None:
        ev.providers.append({"provider": KINDWISE, "step": "disease", "status": "not_configured"})
        return False
    if on_stage:
        on_stage("disease")
    try:
        d = p.diagnose(image, mime)
    except SpecialistError as exc:
        _failed(ev, KINDWISE, "disease", exc)
        return False
    ev.providers.append({"provider": KINDWISE, "step": "disease", "status": "ok", "results": len(d.diseases)})
    ev.diseases = sorted(d.diseases, key=lambda m: -m.probability)[:5]
    ev.plants = ev.plants or d.crops
    ev.healthy_probability = d.healthy_probability
    ev.steps_done.append("disease")
    ev.latest = {"provider": KINDWISE, "kind": "disease", "plants": d.crops, "diseases": ev.diseases, "healthy_probability": d.healthy_probability}
    return True


def gather(route: str, image: bytes, mime: str, settings: Settings, on_stage=None) -> Evidence:
    ev = Evidence()
    if route == HIGH:
        _identify(ev, settings, image, mime, on_stage)           # Pl@ntNet, then Kindwise: both run, a failure of the first changes nothing
        _diagnose(ev, settings, image, mime, on_stage)
    elif not _diagnose(ev, settings, image, mime, on_stage):     # Kindwise first; Pl@ntNet only if Kindwise is unavailable
        _identify(ev, settings, image, mime, on_stage)
    return ev
