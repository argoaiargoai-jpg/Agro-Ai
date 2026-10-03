"""Every analysis: our model first, then the AI ALWAYS inspects the original image itself. Confidence never decides whether it is asked."""
import io
import re
from datetime import timedelta

import pytest
from PIL import Image

from app.ai import registry
from app.ai.base import ProviderUnavailable
from app.core.config import Settings
from app.services import analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.ai_fakes import FakeProvider, NO_PLANT, Scenario, diseased, install, payload

V = "/api/v1"
RED, GREEN, GRAY, BLUE = (255, 0, 0), (0, 255, 0), (128, 128, 128), (0, 0, 255)
S = Settings(_env_file=None)


def ml(kind="DISEASE", conf=0.9855, crop="Corn", disease="Northern Leaf Blight"):
    return dict(classification_type=kind, confidence=conf, crop=crop if kind in ("DISEASE", "HEALTHY") else None,
                disease=disease if kind == "DISEASE" else None, message="m", supported_crops=["Corn"], model_version="t", scores={})


def png(color=RED, size=(320, 240)):
    b = io.BytesIO(); Image.new("RGB", size, color).save(b, "PNG"); return b.getvalue()


def go(ml_result, scenario):
    return analysis_workflow.run_guidance(FakeProvider(scenario), ml_result, png(), S)


# ------------------------------------------------------------------------------------ A-C: confidence never decides
@pytest.mark.parametrize("conf", [0.9855, 0.55, 0.25])
def test_gemini_is_always_called_with_the_original_image_whatever_the_model_confidence(conf):
    sc = Scenario().respond(diseased("Northern Leaf Blight", crop="Corn", plant="Corn"))
    ai, report, case, info = go(ml("DISEASE", conf), sc)
    assert len(sc.requests) == 1                                              # called, even at 98.55%
    req = sc.requests[0]
    assert req.image and req.image[:2] == b"\xff\xd8" and req.image_mime == "image/jpeg"      # the image bytes, not just the prediction text
    assert "Corn with Northern Leaf Blight" in req.prompt and f"{conf:.2f}" in req.prompt      # our result is supporting context
    assert "Do NOT accept it blindly" in req.prompt
    assert info["plan_done"] == ["guidance"] and report.status == "DISEASE"          # no specialist key configured here: straight to Gemini


@pytest.mark.parametrize("kind,conf", [("HEALTHY", 0.99), ("HEALTHY", 0.6), ("UNKNOWN", 0.99), ("UNKNOWN", 0.2), ("NO_PLANT", 0.99), ("NO_PLANT", 0.4)])
def test_every_outcome_and_confidence_reaches_gemini(kind, conf):
    sc = Scenario().respond(payload())
    go(ml(kind, conf, crop="Tomato"), sc)
    assert len(sc.requests) == 1 and sc.requests[0].image


# ------------------------------------------------------------------------------------ D: Gemini wins when it disagrees
def test_ai_sees_rice_so_a_confident_corn_prediction_is_not_shown():
    """The example from the brief: Corn / Northern Leaf Blight at 98.55%, but the photo is rice."""
    sc = Scenario().respond(diseased("Rice Blast", crop="Rice", plant="Rice", ml_consistency="inconsistent"))
    ai, report, _, _ = go(ml("DISEASE", 0.9855), sc)
    assert report.status == "DISEASE" and report.headline == "Rice — Rice Blast" and report.plant == "Rice" and report.disease == "Rice Blast"
    shown = " ".join([report.headline, report.plant or "", report.crop or "", report.disease or ""] + report.symptoms + report.treatment + report.prevention)
    assert "Corn" not in shown and "Northern" not in shown
    assert report.disagreement and "Corn" in report.disagreement.ml_said and "Rice" in report.disagreement.ai_said        # kept internally


def test_ai_sees_a_healthy_rice_plant():
    sc = Scenario().respond(payload(crop="Rice", plant="Rice"))
    _, report, _, _ = go(ml("DISEASE", 0.9855), sc)
    assert report.status == "HEALTHY" and report.headline == "Rice looks healthy" and "Corn" not in report.headline and report.disagreement


def test_ai_unsure_gives_an_honest_uncertain_result_not_our_prediction():
    sc = Scenario().respond(payload(health_status="uncertain", crop=None, plant=None, identification_confidence="low", ai_notes="Cannot tell."))
    _, report, _, _ = go(ml("DISEASE", 0.9855), sc)
    assert report.status == "UNCERTAIN" and report.disease is None and report.crop is None and "Corn" not in report.headline


# ------------------------------------------------------------------------------------ E: agreement is the normal result
def test_ai_agrees_with_the_model():
    sc = Scenario().respond(diseased("Northern Leaf Blight", crop="Corn", plant="Corn", ml_consistency="consistent"))
    _, report, _, _ = go(ml("DISEASE", 0.9855), sc)
    assert report.status == "DISEASE" and report.headline == "Corn — Northern Leaf Blight" and report.disease_source == "ai" and report.disagreement is None
    assert report.symptoms and report.treatment and report.prevention and report.monitoring is not None


# ------------------------------------------------------------------------------------ F: failure is safe (through the API)
@pytest.fixture
def api(tmp_path, settings, monkeypatch, client, admin_auth):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    sc = Scenario(); install(sc)
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake", "enabled": True}).status_code == 200
    yield sc
    ml_service.reset_ml_service(); registry.unregister("fake")


def analyze(client, headers, color):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def test_gemini_failure_keeps_the_application_working_and_the_analysis_retryable(client, user_auth, api):
    api.respond(ProviderUnavailable("down"), diseased("Early Blight"))
    aid, r = analyze(client, user_auth, RED)
    b = r.json()
    assert r.status_code == 200 and b["status"] == "partial" and b["result"]["final"] is None and b["result"]["ml"]["classification_type"] == "DISEASE"
    assert b["result"]["ai_error"]["retryable"] is True and "ProviderUnavailable" not in r.text
    r2 = client.post(f"{V}/analyses/{aid}/analyze", headers=user_auth)
    assert r2.json()["status"] == "completed" and r2.json()["result"]["final"]["status"] == "DISEASE"
    assert client.get(f"{V}/health").status_code == 200


def test_through_the_api_every_state_reaches_gemini_with_the_uploaded_image(client, user_auth, api):
    for color, state in ((RED, "DISEASE"), (GREEN, "HEALTHY"), (GRAY, "UNKNOWN"), (BLUE, "NO_PLANT")):
        before = len(api.requests)
        api.respond(payload())
        _, r = analyze(client, user_auth, color)
        assert r.json()["result"]["ml"]["classification_type"] == state and len(api.requests) == before + 1 and api.requests[-1].image


# ------------------------------------------------------------------------------------ specialists are optional: unconfigured = straight to Gemini
def test_without_any_specialist_key_the_analysis_goes_straight_to_gemini(client, user_auth, api):
    api.respond(payload(health_status="uncertain"))
    _, r = analyze(client, user_auth, GRAY)
    assert r.status_code == 200 and r.json()["result"]["plan"] == ["guidance"] and api.requests and api.requests[-1].image


# ------------------------------------------------------------------------------------ G: customers never see the comparison
FORBIDDEN = re.compile(r"mobilenet|gemini|plantix|kindwise|pl@?ntnet|onnx|\bml\b|ml vs|disagree|confidence routing|fallback|model said|first-stage", re.I)
CUSTOMER_TEXT = ("headline", "rejection_reason", "disclaimer", "ai_notes", "plant", "crop", "disease")
CUSTOMER_LISTS = ("symptoms", "immediate_actions", "treatment", "prevention", "warnings", "monitoring")


@pytest.mark.parametrize("answer", [diseased("Rice Blast", crop="Rice", plant="Rice", ml_consistency="inconsistent"), payload(crop="Rice", plant="Rice"), NO_PLANT,
                                    payload(health_status="uncertain", crop=None, plant=None, identification_confidence="low")])
def test_customer_response_contains_no_model_or_provider_comparison(client, user_auth, api, answer):
    api.respond(answer)
    _, r = analyze(client, user_auth, RED)
    f = r.json()["result"]["final"]
    text = " ".join(str(f.get(k) or "") for k in CUSTOMER_TEXT) + " " + " ".join(x for k in CUSTOMER_LISTS for x in f.get(k, []))
    assert not FORBIDDEN.search(text), text
    assert not [k for k in ("disagreement", "disease_source", "ml_state") if k in f]            # internal fields are not even in the customer's JSON
    assert "ai" not in r.json()["result"] and not r.json()["ai_provider"]
