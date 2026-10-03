"""Deterministic AI test doubles. No test ever calls a real AI service."""
from app.ai import registry
from app.ai.base import AIProvider, AIRequest, AIResponse, ProviderError


def payload(**over) -> dict:
    """A valid AI answer (healthy plant). Override any field."""
    base = dict(plant_present=True, plant="Tomato", crop="Tomato", identification_confidence="high", health_status="healthy", disease=None,
                symptoms=[], severity="none", affected_percentage=None, immediate_actions=[], treatment=[],
                prevention=["Water at the base of the plant", "Keep good airflow"], spread_risk=dict(level="low", explanation=None),
                warnings=[], monitoring=["Re-check weekly"], ml_consistency=None, image_quality="good", ai_notes="Leaf surface looks uniform.")
    base.update(over)
    return base


def diseased(disease="Late Blight", **over) -> dict:
    d = dict(health_status="diseased", disease=disease, symptoms=["Dark water-soaked lesions"], severity="moderate", affected_percentage=30,
             immediate_actions=["Remove affected leaves"], treatment=["Use a labelled fungicide suitable for tomato"],
             spread_risk=dict(level="high", explanation="Spreads in cool, wet weather."))
    d.update(over)
    return payload(**d)


NO_PLANT = payload(plant_present=False, plant=None, crop=None, health_status="not_a_plant", severity="unknown", ai_notes="This is a photo of a keyboard.", prevention=[], monitoring=[])


class Scenario:
    """Script what the fake provider does. `queue` items are payload dicts or exceptions; the last item repeats."""

    def __init__(self):
        self.configured = True
        self.queue: list = [payload()]
        self.requests: list[AIRequest] = []

    def respond(self, *items):
        self.queue = list(items)
        return self


class FakeProvider(AIProvider):
    name, display_name = "fake", "Fake provider"

    def __init__(self, scenario: Scenario):
        self.sc = scenario

    @property
    def model(self) -> str:
        return "fake-model-1"

    def is_configured(self) -> bool:
        return self.sc.configured

    def analyze(self, request: AIRequest) -> AIResponse:
        self.sc.requests.append(request)
        item = self.sc.queue.pop(0) if len(self.sc.queue) > 1 else self.sc.queue[0]
        if isinstance(item, Exception):
            raise item
        return AIResponse(data=item, model="fake-model-1", latency_ms=1.0)

    def ping(self) -> dict:
        item = self.sc.queue[0]
        if isinstance(item, ProviderError):
            raise item
        return {"ok": True, "latency_ms": 1, "model": "fake-model-1"}


def install(scenario: Scenario):
    registry.register("fake", lambda s: FakeProvider(scenario))
