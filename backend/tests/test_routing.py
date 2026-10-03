"""Specialist routing by our model's confidence, then Gemini; Gemini unavailable -> the latest specialist result. All providers are fakes or mocked HTTP."""
import io
import re
from datetime import timedelta

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from app.ai import registry
from app.ai.base import ProviderUnavailable
from app.ai.gemini import GeminiProvider
from app.ai.specialists import registry as specialists
from app.ai.specialists.base import Diagnosis, DiseaseDiagnoser, DiseaseMatch, PlantIdentifier, PlantMatch, SpecialistError
from app.core.config import Settings
from app.services import analysis_service, analysis_workflow, ml_service, routing
from tests import ml_fixture
from tests.ai_fakes import FakeProvider, NO_PLANT, Scenario, diseased, install, payload
from tests.test_gemini_keys import K1, K2, Wire

V = "/api/v1"
RED, GRAY = (255, 0, 0), (128, 128, 128)
S = Settings(_env_file=None)


def ml(kind="DISEASE", conf=0.9, crop="Tomato", disease="Early Blight"):
    return dict(classification_type=kind, confidence=conf, crop=crop if kind in ("DISEASE", "HEALTHY") else None,
                disease=disease if kind == "DISEASE" else None, message="m", supported_crops=["Tomato"], model_version="t", scores={})


def png(color=RED):
    b = io.BytesIO(); Image.new("RGB", (320, 240), color).save(b, "PNG"); return b.getvalue()


# --------------------------------------------------------------------------------------------- fakes that record the call order
class Calls:
    def __init__(self):
        self.order: list[str] = []
        self.images: list[bytes] = []


class FakePlantNet(PlantIdentifier):
    name = "plantnet"

    def __init__(self, calls, error=None, matches=None):
        self.calls, self.error, self.matches = calls, error, matches

    def is_configured(self): return True

    def identify(self, image, mime):
        self.calls.order.append("plantnet"); self.calls.images.append(image)
        if self.error:
            raise self.error
        return self.matches if self.matches is not None else [PlantMatch("Tomato", 0.93, "Solanum lycopersicum")]


class FakeKindwise(DiseaseDiagnoser):
    name = "kindwise"

    def __init__(self, calls, error=None, diseases=None, healthy=None, crops=None):
        self.calls, self.error, self.diseases, self.healthy, self.crops = calls, error, diseases, healthy, crops

    def is_configured(self): return True

    def diagnose(self, image, mime):
        self.calls.order.append("kindwise"); self.calls.images.append(image)
        if self.error:
            raise self.error
        return Diagnosis(diseases=self.diseases if self.diseases is not None else [DiseaseMatch("Early Blight", 0.81)],
                         crops=self.crops if self.crops is not None else [PlantMatch("Tomato", 0.95)], healthy_probability=self.healthy)


class OrderedGemini(FakeProvider):
    def __init__(self, scenario, calls):
        super().__init__(scenario); self.calls = calls

    def analyze(self, request):
        self.calls.order.append("gemini"); self.calls.images.append(request.image)
        return super().analyze(request)


@pytest.fixture
def world(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    saved = (dict(specialists._IDENTIFIERS), dict(specialists._DIAGNOSERS))
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    calls, sc = Calls(), Scenario()
    yield calls, sc
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear(); specialists._IDENTIFIERS.update(saved[0]); specialists._DIAGNOSERS.update(saved[1])
    ml_service.reset_ml_service(); registry.unregister("fake")


def register(calls, pn=True, kw=True, pn_error=None, kw_error=None, **kw_args):
    if pn:
        specialists.register_identifier("plantnet", lambda s: FakePlantNet(calls, pn_error))
    if kw:
        specialists.register_diagnoser("kindwise", lambda s: FakeKindwise(calls, kw_error, **kw_args))


def run(world_, ml_result, stages=None, **reg):
    calls, sc = world_
    register(calls, **reg)
    return analysis_workflow.run_guidance(OrderedGemini(sc, calls), ml_result, png(), S, stages.append if stages is not None else None)


# --------------------------------------------------------------------------------------------- the decision
@pytest.mark.parametrize("conf,route", [(0.81, "HIGH"), (0.9999, "HIGH"), (0.80, "LOW"), (0.7999, "LOW"), (0.50, "LOW"), (0.0, "LOW")])
def test_threshold_is_strictly_greater_than_80_percent_for_pl_ntnet_first(conf, route):
    assert routing.decide(ml(conf=conf), S) == route


@pytest.mark.parametrize("kind", ["DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"])
def test_the_route_depends_only_on_confidence_not_on_the_outcome(kind):
    assert routing.decide(ml(kind, 0.9), S) == "HIGH" and routing.decide(ml(kind, 0.4), S) == "LOW"


# --------------------------------------------------------------------------------------------- 1-3: the normal paths
def test_1_confidence_081_goes_plantnet_then_kindwise_then_gemini(world):
    calls, sc = world
    stages = []
    sc.respond(diseased("Early Blight"))
    ai, report, case, info = run(world, ml(conf=0.81), stages)
    assert calls.order == ["plantnet", "kindwise", "gemini"] and stages == ["identify", "disease", "guidance"]
    assert info["route"] == "HIGH" and info["plan_done"] == ["identify", "disease", "guidance"] and ai is not None and report.status == "DISEASE"
    assert all(i and i[:2] == b"\xff\xd8" for i in calls.images)                    # the same real JPEG bytes reached all three providers
    prompt = sc.requests[0].prompt
    assert "REFERENCE" in prompt and "Possible plant: Tomato" in prompt and "Possible conditions: Early Blight" in prompt and sc.requests[0].image


@pytest.mark.parametrize("conf", [0.80, 0.50, 0.10])
def test_2_3_confidence_up_to_80_percent_goes_kindwise_then_gemini_only(world, conf):
    calls, sc = world
    sc.respond(diseased("Early Blight"))
    stages = []
    _, _, _, info = run(world, ml(conf=conf), stages)
    assert calls.order == ["kindwise", "gemini"] and stages == ["disease", "guidance"]               # Pl@ntNet is not called at all
    assert info["route"] == "LOW" and info["plan_done"] == ["disease", "guidance"]


# --------------------------------------------------------------------------------------------- 4-6: provider unavailable -> next
@pytest.mark.parametrize("kind", ["rate_limited", "unavailable", "timeout"])
def test_4_high_confidence_plantnet_failure_still_runs_kindwise_then_gemini(world, kind):
    calls, sc = world
    sc.respond(diseased("Early Blight"))
    _, _, _, info = run(world, ml(conf=0.95), pn_error=SpecialistError(kind))
    assert calls.order == ["plantnet", "kindwise", "gemini"]
    assert info["plan_done"] == ["disease", "guidance"] and {"provider": "plantnet", "step": "identify", "status": kind} in info["specialists"]["providers"]


@pytest.mark.parametrize("kind", ["rate_limited", "unavailable", "timeout"])
def test_5_low_confidence_kindwise_failure_falls_back_to_plantnet_then_gemini(world, kind):
    calls, sc = world
    sc.respond(diseased("Early Blight"))
    stages = []
    _, _, _, info = run(world, ml(conf=0.4), stages, kw_error=SpecialistError(kind))
    assert calls.order == ["kindwise", "plantnet", "gemini"] and stages == ["disease", "identify", "guidance"]
    assert info["plan_done"] == ["identify", "guidance"] and "Possible plant: Tomato" in sc.requests[0].prompt


@pytest.mark.parametrize("conf", [0.95, 0.4])
def test_6_both_specialists_unavailable_continues_directly_to_gemini(world, conf):
    calls, sc = world
    sc.respond(diseased("Early Blight"))
    ai, report, _, info = run(world, ml(conf=conf), pn_error=SpecialistError("rate_limited"), kw_error=SpecialistError("timeout"))
    assert calls.order[-1] == "gemini" and set(calls.order) <= {"plantnet", "kindwise", "gemini"}
    assert ai is not None and report.status == "DISEASE" and info["plan_done"] == ["guidance"] and "REFERENCE" not in sc.requests[0].prompt
    assert sc.requests[0].image


def test_6b_unconfigured_specialists_are_skipped_without_being_called(world):
    calls, sc = world
    sc.respond(payload())
    _, _, _, info = run(world, ml(conf=0.9), pn=False, kw=False)
    assert calls.order == ["gemini"] and info["plan_done"] == ["guidance"]
    assert {p["status"] for p in info["specialists"]["providers"] if p["provider"] != "fake"} == {"not_configured"}


def test_configuration_errors_are_logged_as_such_and_skipped_not_called_quota_failures(world, caplog):
    calls, sc = world
    sc.respond(payload())
    run(world, ml(conf=0.4), kw_error=SpecialistError("unauthorized"), pn=False)
    assert any(r.levelname == "ERROR" and "configuration or request problem" in r.getMessage() for r in caplog.records)
    assert "unauthorized" not in " ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "unavailable" in r.getMessage())


# --------------------------------------------------------------------------------------------- 7-9: Gemini unavailable
def test_7_gemini_unavailable_after_kindwise_returns_the_kindwise_result(world):
    calls, sc = world
    sc.respond(ProviderUnavailable("503"))
    ai, report, case, info = run(world, ml(conf=0.5))
    assert calls.order == ["kindwise", "gemini"] and ai is None and case == "SPECIALIST_ONLY"
    assert report.status == "DISEASE" and report.headline == "Tomato — Early Blight" and report.disease == "Early Blight" and report.disease_source == "specialist"
    assert info["gemini_error"] == "unavailable" and info["specialists"]["latest"] == {"provider": "kindwise", "kind": "disease"}
    assert {"provider": "fake", "step": "guidance", "status": "unavailable"} in info["specialists"]["providers"]


def test_8_gemini_unavailable_after_plantnet_and_kindwise_returns_the_kindwise_result(world):
    calls, sc = world
    sc.respond(ProviderUnavailable("503"))
    ai, report, _, info = run(world, ml(conf=0.95))
    assert calls.order == ["plantnet", "kindwise", "gemini"] and ai is None
    assert info["specialists"]["latest"]["provider"] == "kindwise" and report.disease == "Early Blight"          # the LATEST successful one


def test_8b_if_only_plantnet_succeeded_its_plant_identification_is_the_result(world):
    calls, sc = world
    sc.respond(ProviderUnavailable("503"))
    ai, report, _, info = run(world, ml(conf=0.95), kw_error=SpecialistError("rate_limited"))
    assert ai is None and info["specialists"]["latest"]["provider"] == "plantnet"
    assert report.status == "UNCERTAIN" and report.plant == "Tomato" and report.disease is None            # identified, but no condition is claimed


def test_8c_a_weak_or_healthy_specialist_verdict_is_not_shown_as_a_disease(world):
    calls, sc = world
    sc.respond(ProviderUnavailable("503"))
    weak = run(world, ml(conf=0.5), diseases=[DiseaseMatch("Early Blight", 0.3)])[1]
    assert weak.status == "UNCERTAIN" and weak.disease is None
    sc.respond(ProviderUnavailable("503"))
    healthy = run(world, ml(conf=0.5), diseases=[DiseaseMatch("Early Blight", 0.2)], healthy=0.9)[1]
    assert healthy.status == "HEALTHY" and healthy.headline == "Tomato looks healthy"


def test_9_gemini_unavailable_on_the_direct_route_keeps_the_existing_behaviour(world):
    calls, sc = world
    sc.respond(ProviderUnavailable("503"))
    with pytest.raises(ProviderUnavailable):                                   # no specialist result to return: the error propagates...
        run(world, ml(conf=0.9), pn_error=SpecialistError("rate_limited"), kw_error=SpecialistError("rate_limited"))


# --------------------------------------------------------------------------------------------- through the API
@pytest.fixture
def api(world, client, admin_auth):
    calls, sc = world
    install(sc)
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake", "enabled": True}).status_code == 200
    return calls, sc


def analyze(client, headers, color=RED):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def test_9b_api_direct_route_gemini_failure_is_the_existing_preliminary_result_with_retry(client, user_auth, api):
    calls, sc = api
    register(calls, pn_error=SpecialistError("rate_limited"), kw_error=SpecialistError("rate_limited"))
    sc.respond(ProviderUnavailable("down"), diseased("Early Blight"))
    aid, r = analyze(client, user_auth)
    b = r.json()
    assert r.status_code == 200 and b["status"] == "partial" and b["result"]["final"] is None and b["result"]["ml"]["classification_type"] == "DISEASE"
    assert b["result"]["ai_error"]["retryable"] is True
    again = client.post(f"{V}/analyses/{aid}/analyze", headers=user_auth).json()
    assert again["status"] == "completed" and again["result"]["final"]["status"] == "DISEASE"


def test_api_gemini_failure_after_a_specialist_completes_the_analysis_with_the_specialist_result(client, user_auth, admin_auth, api):
    calls, sc = api
    register(calls)
    sc.respond(ProviderUnavailable("down"))
    aid, r = analyze(client, user_auth)
    b = r.json()
    assert r.status_code == 200 and b["status"] == "completed" and b["result"]["stage"] == "specialist_only" and "ai" not in b["result"]
    f = b["result"]["final"]
    assert f["status"] == "DISEASE" and f["headline"] == "Tomato — Early Blight" and f["ai_notes"].startswith("Detailed guidance couldn't be generated")
    assert b["result"]["plan"] == ["identify", "disease"]                              # the fixture model is confident (> 80%): Pl@ntNet then Kindwise
    admin_id, _ = analyze(client, admin_auth)
    full = client.get(f"{V}/analyses/{admin_id}", headers=admin_auth).json()
    assert full["ai_status"] == "unavailable" and full["result"]["route"] in ("LOW", "HIGH")
    names = {p["provider"] for p in full["result"]["specialists"]["providers"]}
    assert {"kindwise", "fake"} <= names                                          # admin diagnostics: who was tried and who failed


def test_api_unconfigured_gemini_with_a_specialist_still_gives_a_result(client, user_auth, api, settings, monkeypatch):
    calls, sc = api
    register(calls)
    sc.configured = False
    aid, r = analyze(client, user_auth)
    assert r.status_code == 200 and r.json()["status"] == "completed" and r.json()["result"]["final"]["disease"] == "Early Blight"


def test_api_stage_plan_gets_an_unannounced_fallback_step_inserted(client, user_auth, api):
    calls, sc = api
    register(calls, kw_error=SpecialistError("rate_limited"))
    sc.respond(diseased("Early Blight"))
    _, r = analyze(client, user_auth)
    assert r.json()["result"]["plan"] == ["identify", "guidance"]                        # Kindwise (announced) failed; Pl@ntNet ran instead


# --------------------------------------------------------------------------------------------- 10: customers see none of it
FORBIDDEN = re.compile(r"plantnet|pl@ntnet|kindwise|plantix|gemini|mobilenet|routing|fallback|specialist|disagree|\bml\b|quota|rate.?limit", re.I)


@pytest.mark.parametrize("gemini_answer", [diseased("Early Blight"), ProviderUnavailable("503")])
def test_10_customer_response_contains_no_provider_or_routing_terminology(client, user_auth, api, gemini_answer):
    calls, sc = api
    register(calls)
    sc.respond(gemini_answer)
    _, r = analyze(client, user_auth)
    body = r.json()
    f = body["result"]["final"]
    text = " ".join(str(f.get(k) or "") for k in ("headline", "rejection_reason", "disclaimer", "ai_notes", "plant", "crop", "disease")) + " " + \
        " ".join(x for k in ("symptoms", "immediate_actions", "treatment", "prevention", "warnings", "monitoring") for x in f.get(k, [])) + " " + \
        str((body["result"].get("ai_error") or {}).get("message", "")) + " " + " ".join(body["result"]["plan"])
    assert not FORBIDDEN.search(text), text
    assert not [k for k in ("route", "specialists", "ai", "ai_case", "prompt_version") if k in body["result"]]
    assert not [k for k in ("disagreement", "disease_source", "ml_state") if k in f] and not body["ai_provider"] and not body["ai_error_code"]
    assert "plantnet" not in r.text.lower() and "kindwise" not in r.text.lower()


# --------------------------------------------------------------------------------------------- Gemini two-key rotation still works on top
def test_rotation_and_failover_still_apply_underneath_the_routing(client, user_auth, world, admin_auth, settings, monkeypatch):
    calls, _ = world
    register(calls)
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr(K1)); monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(K2))
    w = Wire({K1: ["ok", 503]})                                                  # request 1 -> K1 ok; request 3 -> K1 fails -> K2
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(w.handler)))
    try:
        for _ in range(3):
            assert analyze(client, user_auth)[1].json()["result"]["final"]
        assert w.calls == ["K1", "K2", "K1", "K2"]                               # 1:K1, 2:K2, 3:K1 failed -> same request on K2
        assert calls.order.count("kindwise") == 3                               # the specialist ran once per analysis, not once per Gemini attempt
        _, r4 = analyze(client, user_auth)
        assert w.calls[-1] == "K2" and r4.json()["result"]["final"]               # request 4 continues the persistent sequence on K2
    finally:
        registry.register("gemini", lambda s: GeminiProvider(s))


def test_every_specialist_secret_is_still_redacted_from_logs():
    s = Settings(_env_file=None, plantnet_api_key="pn-" + "x" * 10, kindwise_api_key="kw-" + "x" * 10)
    assert {"pn-" + "x" * 10, "kw-" + "x" * 10} <= set(s.secret_values())
