"""Decides which analysis path an image takes, from our own model's result, and gathers specialist evidence for it.

  A  DISEASE/HEALTHY, confidence >= 0.80          image + our result            -> Gemini
  B  DISEASE/HEALTHY, 0.50 <= confidence < 0.80   disease specialists           -> Gemini (image + specialist results + our result)
  C  DISEASE/HEALTHY, confidence < 0.50           plant identification, disease -> Gemini
  D  UNKNOWN (any confidence)                     plant identification, disease -> Gemini
  E  NO_PLANT                                     Gemini checks the image itself; plant identification too when our certainty is below 0.80

A specialist that is not configured, or that fails, is skipped: the path degrades to the next best one and never fabricates anything.
Customer-facing step names stay generic ("identify", "disease", "guidance"); provider names stay in the internal `specialists` record.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.ai.specialists import registry
from app.ai.specialists.base import Diagnosis, DiseaseMatch, PlantMatch, SpecialistError
from app.core.config import Settings

log = logging.getLogger("agroai.routing")


@dataclass(frozen=True)
class Route:
    code: str            # A | B | C | D | E
    identify: bool
    disease: bool


def decide(ml: dict, settings: Settings) -> Route:
    kind, conf = ml["classification_type"], float(ml.get("confidence") or 0.0)
    if kind == "UNKNOWN":
        return Route("D", True, True)
    if kind == "NO_PLANT":
        return Route("E", conf < settings.route_high_confidence, False)
    if conf >= settings.route_high_confidence:
        return Route("A", False, False)
    if conf >= settings.route_mid_confidence:
        return Route("B", False, True)
    return Route("C", True, True)


def planned_steps(route: Route, settings: Settings) -> list[str]:
    """Customer-safe list of the steps that WILL run (a step whose providers are not configured is not promised)."""
    steps = []
    if route.identify and registry.identifiers(settings):
        steps.append("identify")
    if route.disease and registry.diagnosers(settings):
        steps.append("disease")
    return steps + ["guidance"]


@dataclass
class Evidence:
    plants: list[PlantMatch] = field(default_factory=list)
    diseases: list[DiseaseMatch] = field(default_factory=list)
    healthy_probability: float | None = None
    is_plant: bool | None = None
    steps_done: list[str] = field(default_factory=list)
    providers: list[dict] = field(default_factory=list)          # internal: which provider answered / failed (never any secret)

    def empty(self) -> bool:
        return not (self.plants or self.diseases)

    def internal(self) -> dict:
        return {"providers": self.providers, "plants": [{"name": p.name, "score": round(p.score, 3)} for p in self.plants],
                "diseases": [{"name": d.name, "probability": round(d.probability, 3)} for d in self.diseases], "steps_done": self.steps_done}


def gather(route: Route, image: bytes, mime: str, settings: Settings, on_stage=None) -> Evidence:
    ev = Evidence()
    if route.identify:
        ids = registry.identifiers(settings)
        if ids:
            if on_stage:
                on_stage("identify")
            for p in ids:
                try:
                    matches = p.identify(image, mime)
                except SpecialistError as exc:
                    log.warning("plant identification via %s failed: %s", p.name, exc.kind)
                    ev.providers.append({"provider": p.name, "step": "identify", "status": exc.kind})
                    continue
                ev.providers.append({"provider": p.name, "step": "identify", "status": "ok", "results": len(matches)})
                ev.plants = matches
                if "identify" not in ev.steps_done:
                    ev.steps_done.append("identify")
                break
    if route.disease:
        diag = registry.diagnosers(settings)
        if diag:
            if on_stage:
                on_stage("disease")

            def one(p):
                try:
                    return p, p.diagnose(image, mime), None
                except SpecialistError as exc:
                    return p, None, exc
            with ThreadPoolExecutor(max_workers=len(diag)) as pool:
                for p, d, exc in pool.map(one, diag):
                    if exc is not None:
                        log.warning("disease analysis via %s failed: %s", p.name, exc.kind)
                        ev.providers.append({"provider": p.name, "step": "disease", "status": exc.kind})
                        continue
                    assert isinstance(d, Diagnosis)
                    ev.providers.append({"provider": p.name, "step": "disease", "status": "ok", "results": len(d.diseases)})
                    if "disease" not in ev.steps_done:
                        ev.steps_done.append("disease")
                    ev.diseases += d.diseases
                    if not ev.plants and d.crops:
                        ev.plants = d.crops
                    if d.healthy_probability is not None:
                        ev.healthy_probability = max(ev.healthy_probability or 0.0, d.healthy_probability)
                    if d.is_plant is not None:
                        ev.is_plant = d.is_plant if ev.is_plant is None else (ev.is_plant or d.is_plant)
            ev.diseases.sort(key=lambda m: -m.probability)
            ev.diseases = ev.diseases[:5]
    return ev
