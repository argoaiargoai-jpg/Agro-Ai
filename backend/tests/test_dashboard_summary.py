"""Dashboard numbers are derived from real stored analyses only (never fabricated)."""
import io
from datetime import timedelta

import pytest
from PIL import Image

from app.ai import registry
from app.services import analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.ai_fakes import NO_PLANT, Scenario, diseased, install, payload

V = "/api/v1"
RED, GREEN, GRAY, BLUE = (255, 0, 0), (0, 255, 0), (128, 128, 128), (0, 0, 255)


@pytest.fixture(autouse=True)
def _env(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    yield
    ml_service.reset_ml_service(); registry.unregister("fake")


def _png(color):
    b = io.BytesIO(); Image.new("RGB", (320, 240), color).save(b, "PNG"); return b.getvalue()


def _run(client, headers, color):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", _png(color), "image/png")}).json()["id"]
    return client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def test_new_user_summary_is_all_zero_and_empty(client, user_auth):
    s = client.get(f"{V}/analyses/summary", headers=user_auth).json()
    assert s["total"] == 0 and s["outcomes"] == {"healthy": 0, "disease": 0, "unresolved": 0, "no_plant": 0}
    assert s["ai_assisted"] == 0 and s["top_diseases"] == [] and s["crops"] == [] and s["recent"] == []
    assert len(s["activity_14d"]) == 14 and sum(d["count"] for d in s["activity_14d"]) == 0


def test_summary_counts_match_the_stored_analyses(client, user_auth, admin_auth):
    sc = Scenario(); install(sc)
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake", "enabled": True}).status_code == 200
    sc.respond(diseased("Early Blight")); _run(client, user_auth, RED)              # ML DISEASE            -> disease
    sc.respond(diseased("Early Blight")); _run(client, user_auth, RED)              # again                 -> disease
    sc.respond(payload()); _run(client, user_auth, GREEN)                           # ML HEALTHY, AI agrees -> healthy
    sc.respond(diseased("Leaf Spot", crop="Rose", plant="Rose")); _run(client, user_auth, GRAY)    # UNKNOWN -> AI finds disease on a rose
    sc.respond(NO_PLANT); _run(client, user_auth, BLUE)                             # NO_PLANT, AI agrees   -> no_plant
    sc.respond(payload(health_status="uncertain", plant_present=True, identification_confidence="low", severity="unknown"))
    _run(client, user_auth, GRAY)                                                   # UNKNOWN, AI unsure    -> unresolved
    s = client.get(f"{V}/analyses/summary", headers=user_auth).json()
    assert s["total"] == 6
    assert s["outcomes"] == {"disease": 3, "healthy": 1, "no_plant": 1, "unresolved": 1}
    assert s["ai_assisted"] == 6
    assert {"name": "Early Blight", "count": 2} in s["top_diseases"] and {"name": "Leaf Spot", "count": 1} in s["top_diseases"]
    assert {"name": "Tomato", "count": 3} in s["crops"] and {"name": "Rose", "count": 1} in s["crops"]
    assert sum(d["count"] for d in s["activity_14d"]) == 6 and s["activity_14d"][-1]["count"] == 6
    assert len(s["recent"]) == 5


def test_summary_never_mixes_users_and_admin_stats_cover_everyone(client, user_auth, admin_auth):
    sc = Scenario(); install(sc)
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake", "enabled": True})
    sc.respond(diseased("Early Blight")); _run(client, admin_auth, RED)
    assert client.get(f"{V}/analyses/summary", headers=user_auth).json()["total"] == 0
    st = client.get(f"{V}/admin/stats", headers=admin_auth).json()
    assert st["analyses_total"] == 1 and st["outcomes"]["disease"] == 1 and st["ai_assisted"] == 1 and len(st["activity_14d"]) == 14


def test_ml_only_analyses_are_counted_but_not_as_ai_assisted(client, user_auth, admin_auth):
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": False})
    _run(client, user_auth, RED)
    s = client.get(f"{V}/analyses/summary", headers=user_auth).json()
    assert s["outcomes"]["disease"] == 1 and s["ai_assisted"] == 0
