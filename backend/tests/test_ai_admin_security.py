"""Admin configuration of the AI layer, and secret-handling / abuse audits."""
import io
import json
import logging
import pathlib
from datetime import timedelta

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import text

from app.ai import gemini, registry
from app.ai.base import ProviderTimeout
from app.ai.gemini import GeminiProvider
from app.db.session import engine
from app.main import SecretRedactingFilter
from app.services import ai_admin_service, analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.ai_fakes import Scenario, install, payload
from tests.conftest import register_and_verify

V = "/api/v1"
KEY = "AI" + "zaSy" + "LEAKCHECK_0123456789abcdefghijklmnop"      # runtime-built fake key (see test_ai_gemini.py)
ENDPOINTS = [("get", "/admin/ai", None), ("put", "/admin/ai", {"enabled": False}), ("post", "/admin/ai/test", {})]


@pytest.fixture(autouse=True)
def _env(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(KEY))
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0)
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    monkeypatch.setattr(gemini, "_SLEEP", lambda s: None)
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    yield
    ml_service.reset_ml_service(); registry.unregister("fake")


def call(client, method, path, headers=None, body=None):
    return getattr(client, method)(f"{V}{path}", headers=headers or {}, **({"json": body} if body is not None else {}))


# ------------------------------------------------------------------ authorization
@pytest.mark.parametrize("method,path,body", ENDPOINTS)
def test_ai_admin_endpoints_require_authentication(client, method, path, body):
    r = call(client, method, path, body=body)
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("method,path,body", ENDPOINTS)
def test_ai_admin_endpoints_forbidden_for_regular_users(client, user_auth, method, path, body):
    r = call(client, method, path, user_auth, body)
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"


def test_regular_user_cannot_change_ai_settings_through_the_generic_endpoint_either(client, user_auth):
    assert client.put(f"{V}/admin/settings", headers=user_auth, json={"values": {"ai_enabled": False}}).status_code == 403


# ------------------------------------------------------------------ admin functionality
def test_overview_reports_status_without_any_secret(client, admin_auth):
    r = client.get(f"{V}/admin/ai", headers=admin_auth)
    b = r.json()
    assert r.status_code == 200 and b["enabled"] is True and b["active_provider"] == "gemini" and b["state"] == "ready" and b["secret_source"] == "environment"
    g = next(p for p in b["providers"] if p["name"] == "gemini")
    assert g["configured"] is True and g["model"] and g["display_name"] == "Google Gemini"
    assert set(g) == {"name", "display_name", "configured", "model"}              # only a boolean about the secret, never the secret
    assert KEY not in r.text


def test_overview_states(client, admin_auth, settings, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(""))
    assert client.get(f"{V}/admin/ai", headers=admin_auth).json()["state"] == "not_configured"
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(KEY))
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": False})
    assert client.get(f"{V}/admin/ai", headers=admin_auth).json()["state"] == "disabled"


def test_update_provider_validates_against_the_registry_and_is_audited(client, admin_auth):
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "does-not-exist"}).status_code == 422
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": "yes"}).status_code == 422
    install(Scenario())
    r = client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake"})
    assert r.status_code == 200 and r.json()["active_provider"] == "fake"
    log = client.get(f"{V}/admin/audit-log", headers=admin_auth).json()
    assert any(e["action"] == "settings.update" and "ai_provider" in e["target"] for e in log)


def test_ai_settings_are_not_public_and_not_in_the_generic_system_listing(client, admin_auth):
    pub = client.get(f"{V}/config/public").json()
    assert "ai_enabled" not in pub and "ai_provider" not in pub
    assert "ai_enabled" not in client.get(f"{V}/admin/settings", headers=admin_auth).json()


def test_test_connection_ok_failing_and_unconfigured(client, admin_auth, settings, monkeypatch):
    sc = Scenario(); install(sc)
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "fake"})
    assert r.json()["ok"] is True and r.json()["state"] == "ready"
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0)
    sc.respond(ProviderTimeout("slow upstream"))
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "fake"})
    assert r.json()["ok"] is False and r.json()["state"] == "timeout"
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0); monkeypatch.setattr(settings, "gemini_api_key", SecretStr(""))
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "gemini"})
    assert r.json()["state"] == "not_configured" and r.json()["ok"] is False
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0)
    assert client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "nope"}).status_code == 422


def test_test_connection_is_rate_limited(client, admin_auth):
    install(Scenario())
    assert client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "fake"}).status_code == 200
    assert client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "fake"}).status_code == 429


# ------------------------------------------------------------------ the real Gemini adapter through the whole stack (mock HTTP)
def _use_mock_gemini(handler):
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(handler)))


@pytest.fixture
def real_gemini_stack(client):
    yield
    registry.register("gemini", lambda s: GeminiProvider(s))               # restore the production factory


def _png(color):
    b = io.BytesIO(); Image.new("RGB", (300, 200), color).save(b, "PNG"); return b.getvalue()


def _analyze(client, headers, color):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", _png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def test_full_stack_with_real_gemini_adapter_success(client, admin_auth, real_gemini_stack):
    seen = {}

    def handler(req):
        seen["key"], seen["url"], seen["body"] = req.headers.get("x-goog-api-key"), str(req.url), json.loads(req.content)
        ans = payload(health_status="diseased", disease="Early Blight", symptoms=["Concentric rings"], severity="mild", ml_consistency="consistent")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(ans)}]}, "finishReason": "STOP"}]})
    _use_mock_gemini(handler)
    aid, r = _analyze(client, admin_auth, (255, 0, 0))
    b = r.json()
    assert b["status"] == "completed" and b["ai_provider"] == "gemini" and b["result"]["final"]["disease_source"] == "ml"
    assert seen["key"] == KEY and KEY not in seen["url"] and "inlineData" in json.dumps(seen["body"]) and KEY not in r.text


def test_api_key_never_appears_in_any_response_log_or_table(client, user_auth, admin_auth, caplog, real_gemini_stack):
    caplog.set_level(logging.DEBUG)
    _use_mock_gemini(lambda req: httpx.Response(403, json={"error": {"message": f"Key {KEY} rejected"}}))          # provider echoes the key back
    aid, r = _analyze(client, user_auth, (255, 0, 0))
    bodies = [r.text, client.get(f"{V}/analyses/{aid}", headers=user_auth).text, client.get(f"{V}/analyses", headers=user_auth).text,
              client.get(f"{V}/analyses/summary", headers=user_auth).text, client.get(f"{V}/config/public").text, client.get(f"{V}/health").text,
              client.get(f"{V}/ml/info").text, client.get(f"{V}/admin/ai", headers=admin_auth).text, client.get(f"{V}/admin/settings", headers=admin_auth).text,
              client.get(f"{V}/admin/audit-log", headers=admin_auth).text, client.get(f"{V}/admin/stats", headers=admin_auth).text,
              client.get(f"{V}/users/me", headers=user_auth).text, client.get("/openapi.json").text]
    assert r.json()["ai_status"] == "misconfigured" and all(KEY not in b for b in bodies)
    assert KEY not in caplog.text
    with engine.connect() as c:                                                                                  # nothing secret persisted anywhere
        dump = ""
        for (t,) in c.execute(text("select name from sqlite_master where type='table'")).all():
            dump += json.dumps([[str(v) for v in row] for row in c.execute(text(f'select * from "{t}"')).all()])
    assert KEY not in dump


def test_redacting_filter_masks_secrets(settings):
    rec = logging.LogRecord("x", logging.WARNING, "f", 1, "boom %s", (KEY,), None)
    assert SecretRedactingFilter().filter(rec) and KEY not in rec.getMessage() and "***" in rec.getMessage()


def test_frontend_source_never_contains_provider_secrets_or_calls_the_provider_directly():
    src = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "src"
    if not src.exists():
        pytest.skip("frontend sources not present")
    blob = "".join(p.read_text(errors="ignore") for p in src.rglob("*") if p.suffix in (".js", ".jsx"))
    # Forbidden: key VALUES, the provider's URL/auth header (the browser must never talk to the provider), and any
    # VITE_-exposed key. (The admin help text may mention the env var NAME "GEMINI_API_KEY"; that is not a secret.)
    for needle in ("generativelanguage.googleapis.com", "x-goog-api-key", "VITE_GEMINI", "VITE_AI_KEY", "AIza"):
        assert needle not in blob, needle
    import re
    assert not re.search(r"GEMINI_API_KEY\s*[:=]\s*['\"`][^'\"`]+", blob), "a literal key assignment was found in the frontend"


def test_existing_auth_still_protects_analysis(client):
    assert client.post(f"{V}/analyses/00000000-0000-0000-0000-000000000000/analyze").status_code == 401
    other = register_and_verify(client, email="x@example.com")
    assert client.post(f"{V}/analyses/00000000-0000-0000-0000-000000000000/analyze", headers={"Authorization": f"Bearer {other['access_token']}"}).status_code == 404
