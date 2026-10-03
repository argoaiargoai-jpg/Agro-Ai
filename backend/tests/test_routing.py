"""Confidence-based routing (A-E): which specialists run, what Gemini receives, and how the customer-facing result is formed."""
import io
from datetime import timedelta

import pytest
from PIL import Image

from app.ai import registry
from app.ai.specialists import registry as specialists
from app.ai.specialists.base import Diagnosis, DiseaseDiagnoser, DiseaseMatch, PlantIdentifier, PlantMatch, SpecialistError
from app.core.config import Settings
from app.services import analysis_service, analysis_workflow, ml_service, routing
from tests import ml_fixture
from tests.ai_fakes import FakeProvider, NO_PLANT, Scenario, diseased, install, payload

V = "/api/v1"
RED, GREEN, GRAY, BLUE = (255, 0, 0), (0, 255, 0), (128, 128, 128), (0, 0, 255)           # fixture ML: DISEASE / HEALTHY / UNKNOWN / NO_PLANT
S = Settings(_env_file=None)


def ml(kind, conf, crop="Tomato", disease="Early Blight"):
    return dict(classification_type=kind, confidence=conf, crop=crop if kind in ("DISEASE", "HEALTHY") else None,
                disease=disease if kind == "DISEASE" else None, message="m", supported_crops=["Tomato"], model_version="t", scores={})


# ------------------------------------------------------------------------------------------------ the decision table
@pytest.mark.parametrize("kind,conf,code,identify,disease", [
    ("DISEASE", 0.95, "A", False, False), ("HEALTHY", 0.80, "A", False, False), ("DISEASE", 0.80, "A", False, False),
    ("DISEASE", 0.7999, "B", False, True), ("HEALTHY", 0.50, "B", False, True), ("DISEASE", 0.65, "B", False, True),
    ("DISEASE", 0.4999, "C", True, True), ("HEALTHY", 0.10, "C", True, True),
    ("UNKNOWN", 0.99, "D", True, True), ("UNKNOWN", 0.85, "D", True, True), ("UNKNOWN", 0.30, "D", True, True), ("UNKNOWN", 0.0, "D", True, True),
    ("NO_PLANT", 0.99, "E", False, False), ("NO_PLANT", 0.79, "E", True, False), ("NO_PLANT", 0.50, "E", True, False),
])
def test_decision_table(kind, conf, code, identify, disease):
    r = routing.decide(ml(kind, conf), S)
    assert (r.code, r.identify, r.disease) == (code, identify, disease)


def test_thresholds_are_configurable_not_hardcoded():
    s = Settings(_env_file=None, route_high_confidence=0.9, route_mid_confidence=0.6)
    assert routing.decide(ml("DISEASE", 0.85), s).code == "B" and routing.decide(ml("DISEASE", 0.55), s).code == "C"


# ------------------------------------------------------------------------------------------------ fakes
class Calls:
    def __init__(self):
        self.log: list[tuple[str, str, bytes]] = []


class FakeId(PlantIdentifier):
    def __init__(self, calls, matches=None, error=None, name="fakeid"):
        self.name, self.calls, self.matches, self.error = name, calls, matches, error

    def is_configured(self): return True

    def identify(self, image, mime):
        self.calls.log.append(("identify", self.name, image))
        if self.error:
            raise self.error
        return self.matches if self.matches is not None else [PlantMatch("Rose", 0.91, "Rosa")]


class FakeDx(DiseaseDiagnoser):
    def __init__(self, calls, diseases=None, error=None, name="fakedx"):
        self.name, self.calls, self.diseases, self.error = name, calls, diseases, error

    def is_configured(self): return True

    def diagnose(self, image, mime):
        self.calls.log.append(("disease", self.name, image))
        if self.error:
            raise self.error
        return Diagnosis(diseases=self.diseases if self.diseases is not None else [DiseaseMatch("Black Spot", 0.74)], crops=[PlantMatch("Rose", 0.8)])


@pytest.fixture
def env(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    calls = Calls()
    sc = Scenario(); install(sc)
    # no real specialist is ever contacted: replace the production set by fakes registered per test
    saved = (dict(specialists._IDENTIFIERS), dict(specialists._DIAGNOSERS))
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    yield calls, sc
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.update(saved[1]); specialists._IDENTIFIERS.update(saved[0])
    ml_service.reset_ml_service(); registry.unregister("fake")


def png(color):
    b = io.BytesIO(); Image.new("RGB", (320, 240), color).save(b, "PNG"); return b.getvalue()


def settings_with_fake_ai(client, admin_auth):
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake", "enabled": True}).status_code == 200


def analyze(client, headers, color):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def run(ml_result, settings, scenario, stages=None):
    """Drive the workflow directly with a hand-made ML result (the fixture model cannot produce a 50-79% DISEASE)."""
    return analysis_workflow.run_guidance(FakeProvider(scenario), ml_result, png(RED), settings, stages.append if stages is not None else None)


# ------------------------------------------------------------------------------------------------ A: confident
def test_a_confident_result_skips_specialists_and_sends_image_plus_our_result(env, settings):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    stages = []
    ai, report, case, info = run(ml("DISEASE", 0.93), settings, sc.respond(diseased("Whatever", ml_consistency="consistent")), stages)
    assert calls.log == [] and info["route"] == "A" and info["plan_done"] == ["guidance"] and stages == ["guidance"]
    req = sc.requests[0]
    assert req.image and "ALREADY identified" in req.prompt and "Early Blight" in req.prompt and "REFERENCE" not in req.prompt
    assert report.status == "DISEASE" and report.disease == "Early Blight" and report.disease_source == "ml"      # confident: our identity stays


def test_a_healthy_confident_gets_context_but_is_inspected(env, settings):
    calls, sc = env
    ai, report, _, info = run(ml("HEALTHY", 0.95, crop="Grape"), settings, sc.respond(diseased("Black Rot", crop="Grape", plant="Grape")))
    assert calls.log == [] and "healthy Grape" in sc.requests[0].prompt and report.status == "DISEASE" and report.disease_source == "ai"


# ------------------------------------------------------------------------------------------------ B: 50-79.99%
def test_b_medium_confidence_calls_only_the_disease_specialists_then_gemini(env, settings):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls))
    specialists.register_diagnoser("dx1", lambda s: FakeDx(calls, name="dx1"))
    specialists.register_diagnoser("dx2", lambda s: FakeDx(calls, [DiseaseMatch("Early Blight", 0.6)], name="dx2"))
    stages = []
    ai, report, case, info = run(ml("DISEASE", 0.66), settings, sc.respond(diseased("Early Blight")), stages)
    assert sorted(c[1] for c in calls.log) == ["dx1", "dx2"] and all(c[0] == "disease" for c in calls.log)
    assert info["route"] == "B" and info["plan_done"] == ["disease", "guidance"] and stages == ["disease", "guidance"]
    prompt = sc.requests[0].prompt
    assert sc.requests[0].image and "REFERENCE" in prompt and "Black Spot" in prompt and "Early Blight" in prompt
    assert "NOT certain" in prompt                                                    # our own result is only a hint
    assert not any(n in prompt.lower() for n in ("dx1", "dx2", "plantix", "kindwise", "plantnet"))     # provider names never reach Gemini either
    assert all(c[2] and c[2][:2] == b"\xff\xd8" for c in calls.log)                     # actual JPEG bytes were sent to every specialist
    assert report.disease_source == "ai"                                              # Gemini decides, our mid-confidence guess is not final


# ------------------------------------------------------------------------------------------------ C: < 50%
def test_c_low_confidence_identifies_the_plant_first_then_diseases_then_gemini(env, settings):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    stages = []
    ai, report, _, info = run(ml("DISEASE", 0.31), settings, sc.respond(diseased("Black Spot", crop="Rose", plant="Rose")), stages)
    assert [c[0] for c in calls.log] == ["identify", "disease"] and stages == ["identify", "disease", "guidance"]
    assert info["route"] == "C" and info["plan_done"] == ["identify", "disease", "guidance"]
    assert "Possible plant: Rose" in sc.requests[0].prompt and "Possible conditions: Black Spot" in sc.requests[0].prompt
    assert report.headline == "Rose — Black Spot"


# ------------------------------------------------------------------------------------------------ D / E
@pytest.mark.parametrize("conf", [0.15, 0.55, 0.85, 0.99])
def test_d_unknown_always_takes_plant_identification_then_disease_then_gemini(env, settings, conf):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    _, _, _, info = run(ml("UNKNOWN", conf), settings, sc.respond(diseased("Black Spot", crop="Rose", plant="Rose")))
    assert [c[0] for c in calls.log] == ["identify", "disease"] and info["route"] == "D" and sc.requests[0].image


def test_e_no_plant_confident_goes_straight_to_gemini(env, settings):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls))
    _, report, _, info = run(ml("NO_PLANT", 0.97), settings, sc.respond(NO_PLANT))
    assert calls.log == [] and info["route"] == "E" and sc.requests[0].image
    assert report.status == "REJECTED" and report.headline == "No plant detected"


def test_e_no_plant_with_doubt_also_tries_plant_identification_and_continues_if_a_plant_is_found(env, settings):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    _, report, _, info = run(ml("NO_PLANT", 0.6), settings, sc.respond(diseased("Black Spot", crop="Rose", plant="Rose")))
    assert [c[0] for c in calls.log] == ["identify"] and "Possible plant: Rose" in sc.requests[0].prompt
    assert report.status == "DISEASE"                                                 # Gemini found a plant, so the analysis continued


def test_e_no_plant_is_final_only_when_gemini_agrees(env, settings):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls, matches=[]))
    _, report, _, _ = run(ml("NO_PLANT", 0.6), settings, sc.respond(NO_PLANT))
    assert report.status == "REJECTED"


# ------------------------------------------------------------------------------------------------ degrading gracefully
def test_unconfigured_specialists_are_not_promised_and_not_called(env, settings):
    calls, sc = env
    assert routing.planned_steps(routing.decide(ml("UNKNOWN", 0.4), settings), settings) == ["guidance"]
    _, _, _, info = run(ml("UNKNOWN", 0.4), settings, sc.respond(payload()))
    assert info["plan_done"] == ["guidance"] and sc.requests


def test_plan_lists_only_providers_that_exist(env, settings):
    calls, sc = env
    specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    assert routing.planned_steps(routing.decide(ml("UNKNOWN", 0.4), settings), settings) == ["disease", "guidance"]
    specialists.register_identifier("id", lambda s: FakeId(calls))
    assert routing.planned_steps(routing.decide(ml("UNKNOWN", 0.4), settings), settings) == ["identify", "disease", "guidance"]


@pytest.mark.parametrize("kind", ["rate_limited", "timeout", "unavailable", "unauthorized", "bad_response"])
def test_a_failing_specialist_never_blocks_the_analysis(env, settings, kind):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls, error=SpecialistError(kind)))
    specialists.register_diagnoser("dx", lambda s: FakeDx(calls, error=SpecialistError(kind)))
    _, report, _, info = run(ml("UNKNOWN", 0.4), settings, sc.respond(payload(health_status="uncertain")))
    assert info["plan_done"] == ["guidance"] and "REFERENCE" not in sc.requests[0].prompt          # nothing is claimed that did not happen
    assert {p["status"] for p in info["specialists"]["providers"]} == {kind} and report.status == "UNCERTAIN"


def test_one_specialist_failing_keeps_the_other(env, settings):
    calls, sc = env
    specialists.register_diagnoser("bad", lambda s: FakeDx(calls, error=SpecialistError("timeout"), name="bad"))
    specialists.register_diagnoser("good", lambda s: FakeDx(calls, name="good"))
    _, _, _, info = run(ml("DISEASE", 0.6), settings, sc.respond(diseased("Black Spot")))
    assert info["plan_done"] == ["disease", "guidance"] and "Black Spot" in sc.requests[0].prompt


def test_gemini_is_told_not_to_invent_certainty():
    from app.ai.prompts import SYSTEM
    assert "REFERENCE information" in SYSTEM and "It can be wrong" in SYSTEM
    assert "Never present a tool's guess as certain, and never invent a diagnosis" in SYSTEM
    assert "tools" in SYSTEM and "Never mention other models" in SYSTEM          # and never to name any provider/tool in the answer


# ------------------------------------------------------------------------------------------------ through the API
def test_api_route_d_unknown_image_records_plan_stage_and_hides_providers_from_customers(client, user_auth, admin_auth, env):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls, name="plantnet")); specialists.register_diagnoser("dx", lambda s: FakeDx(calls, name="plantix"))
    settings_with_fake_ai(client, admin_auth)
    sc.respond(diseased("Black Spot", crop="Rose", plant="Rose"))
    aid, r = analyze(client, user_auth, GRAY)                                          # fixture model: UNKNOWN
    b = r.json()
    assert r.status_code == 200 and b["result"]["ml"]["classification_type"] == "UNKNOWN" and b["result"]["final"]["headline"] == "Rose — Black Spot"
    assert b["result"]["plan"] == ["identify", "disease", "guidance"] and b["result"]["stage"] == "complete"
    text = r.text.lower()
    assert "route" not in b["result"] and "specialists" not in b["result"] and not any(n in text for n in ("plantnet", "plantix", "kindwise"))
    adm, _ = analyze(client, admin_auth, GRAY)
    full = client.get(f"{V}/analyses/{adm}", headers=admin_auth).json()["result"]
    assert full["route"] == "D" and {p["provider"] for p in full["specialists"]["providers"]} == {"plantnet", "plantix"}


def test_api_route_a_confident_disease_calls_no_specialist(client, user_auth, admin_auth, env):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    settings_with_fake_ai(client, admin_auth)
    aid, r = analyze(client, user_auth, RED)
    assert r.json()["result"]["ml"]["classification_type"] == "DISEASE" and calls.log == [] and r.json()["result"]["plan"] == ["guidance"]


def test_api_the_stage_is_committed_before_each_real_step(client, user_auth, admin_auth, env, monkeypatch):
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    settings_with_fake_ai(client, admin_auth)
    seen = []
    orig = FakeDx.diagnose

    def spy(self, image, mime):
        r = client.get(f"{V}/analyses", headers=user_auth).json()["items"][0]["result"]
        seen.append((r["stage"], r["plan"]))
        return orig(self, image, mime)
    monkeypatch.setattr(FakeDx, "diagnose", spy)
    analyze(client, user_auth, GRAY)
    assert seen == [("disease", ["identify", "disease", "guidance"])]                 # while the disease step runs, the stored stage says so


def test_api_gemini_failure_after_specialists_keeps_ml_and_plan_and_allows_retry(client, user_auth, admin_auth, env):
    from app.ai.base import ProviderUnavailable
    calls, sc = env
    specialists.register_identifier("id", lambda s: FakeId(calls)); specialists.register_diagnoser("dx", lambda s: FakeDx(calls))
    settings_with_fake_ai(client, admin_auth)
    sc.respond(ProviderUnavailable("down"), diseased("Black Spot", crop="Rose", plant="Rose"))
    aid, r = analyze(client, user_auth, GRAY)
    b = r.json()
    assert b["status"] == "partial" and b["result"]["stage"] == "guidance_failed" and b["result"]["plan"] == ["identify", "disease", "guidance"] and b["result"]["final"] is None
    r2 = client.post(f"{V}/analyses/{aid}/analyze", headers=user_auth)
    assert r2.json()["status"] == "completed" and r2.json()["result"]["ml"] == b["result"]["ml"]


def test_gemini_counter_sql_is_a_single_atomic_upsert_on_postgresql():
    """The production database is PostgreSQL: the exact statement must be one INSERT .. ON CONFLICT DO UPDATE .. RETURNING."""
    from sqlalchemy.dialects import postgresql

    from app.models.counter import Counter
    stmt = postgresql.insert(Counter).values(name="gemini_requests", value=1)
    stmt = stmt.on_conflict_do_update(index_elements=[Counter.name], set_={"value": Counter.value + 1}).returning(Counter.value)
    sql = str(stmt.compile(dialect=postgresql.dialect())).replace("\n", " ")
    assert "INSERT INTO counters" in sql and "ON CONFLICT (name) DO UPDATE SET value = (counters.value + %(value_1)s::BIGINT)" in sql and sql.rstrip().endswith("RETURNING counters.value")
