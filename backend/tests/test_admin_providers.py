"""Admin AI page: status of ALL providers the workflow uses (safe metadata only). No routing or behaviour change."""
import json

import httpx
import pytest
from pydantic import SecretStr

from app.ai import groq, pollinations
from app.ai.groq import GroqProvider
from app.ai.pollinations import PollinationsProvider
from app.services import ai_admin_service

V = "/api/v1"
SECRETS = {"gemini_api_key_1": "gem-one-SECRETVALUE-111111", "gemini_api_key_2": "gem-two-SECRETVALUE-222222", "kindwise_api_key": "kw-SECRETVALUE-333333",
           "plantnet_api_key": "pn-SECRETVALUE-444444", "groq_api_key": "gsk-SECRETVALUE-555555", "pollinations_api_key": "pk-SECRETVALUE-666666"}


@pytest.fixture
def all_keys(settings, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setattr(settings, k, SecretStr(v))
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0)


def by_id(body):
    return {p["id"]: p for p in body["workflow"]}


def test_admin_receives_all_five_providers_with_safe_metadata(client, admin_auth, all_keys):
    body = client.get(f"{V}/admin/ai", headers=admin_auth).json()
    w = by_id(body)
    assert list(w) == ["gemini", "kindwise", "plantnet", "groq", "pollinations"]
    assert w["gemini"] == {"id": "gemini", "name": "Google Gemini", "purpose": "AI verification & guidance (main AI step)", "configured": True, "model": "gemini-3.8-flash",
                           "keys": {"configured": 2, "of": 2}, "state": "primary", "testable": True}
    assert w["kindwise"]["configured"] is True and w["kindwise"]["state"] == "available" and "model" not in w["kindwise"] and w["kindwise"]["testable"] is False
    assert w["plantnet"]["configured"] is True and w["plantnet"]["state"] == "available" and w["plantnet"]["purpose"] == "Plant identification specialist"
    assert w["groq"]["configured"] is True and w["groq"]["model"] == "qwen/qwen3.8-27b" and w["groq"]["state"] == "fallback" and w["groq"]["testable"] is True
    assert w["pollinations"]["configured"] is True and w["pollinations"]["model"] == "openai/gpt-5.4-nano" and w["pollinations"]["state"] == "fallback"
    assert body["generative_ai_bypass_gemini"] is False and body["fallbacks"]["groq"]["configured"] is True


def test_api_key_values_are_never_returned(client, admin_auth, all_keys):
    text = client.get(f"{V}/admin/ai", headers=admin_auth).text
    for v in SECRETS.values():
        assert v not in text
    assert "SECRETVALUE" not in text and "authorization" not in text.lower()


def test_unconfigured_providers_are_reported_as_not_configured_and_a_partial_gemini_setup_is_visible(client, admin_auth, settings, monkeypatch):
    for k in SECRETS:
        monkeypatch.setattr(settings, k, SecretStr(""))
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert {p["state"] for p in w.values()} == {"not_configured"} and not any(p["configured"] for p in w.values())
    monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(SECRETS["gemini_api_key_2"]))
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert w["gemini"]["keys"] == {"configured": 1, "of": 2} and w["gemini"]["configured"] is True


def test_a_configured_key_alone_does_not_make_a_provider_primary(client, admin_auth, all_keys):
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert [p["id"] for p in w.values() if p["state"] == "primary"] == ["gemini"]           # only the selected main provider; the rest are merely available


def test_states_follow_the_ai_switch_and_the_gemini_bypass(client, admin_auth, all_keys):
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": False})
    assert by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())["gemini"]["state"] == "disabled"
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": True, "generative_ai_bypass_gemini": True})
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert w["gemini"]["state"] == "bypassed" and w["groq"]["state"] == "fallback" and w["kindwise"]["state"] == "available"
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"generative_ai_bypass_gemini": False})
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert w["gemini"]["state"] == "primary" and w["groq"]["state"] == "fallback"


def test_existing_gemini_status_fields_still_work(client, admin_auth, all_keys):
    body = client.get(f"{V}/admin/ai", headers=admin_auth).json()
    assert body["state"] == "ready" and body["active_provider"] == "gemini" and body["enabled"] is True
    assert [p["name"] for p in body["providers"]] == ["gemini"] and body["providers"][0]["configured"] is True and body["secret_source"] == "environment"
    assert {"limits", "usage_24h"} <= set(body)


@pytest.mark.parametrize("method,path,body", [("get", "/admin/ai", None), ("post", "/admin/ai/test", {"provider": "groq"})])
def test_only_admins_can_see_provider_configuration(client, user_auth, admin_auth, method, path, body):
    kw = {"json": body} if body is not None else {}
    assert getattr(client, method)(f"{V}{path}", **kw).status_code == 401
    assert getattr(client, method)(f"{V}{path}", headers=user_auth, **kw).status_code == 403


def test_public_and_customer_endpoints_carry_no_provider_information(client, user_auth, all_keys):
    for url in (f"{V}/config/public", f"{V}/users/me", f"{V}/health", f"{V}/ml/info"):
        h = {"headers": user_auth} if "users" in url else {}
        t = client.get(url, **h).text.lower()
        assert not any(w in t for w in ("groq", "pollinations", "kindwise", "plantnet", "gemini", "secretvalue")), url


@pytest.mark.parametrize("name,mod,cls,key", [("groq", groq, GroqProvider, "groq_api_key"), ("pollinations", pollinations, PollinationsProvider, "pollinations_api_key")])
def test_fallback_connection_test_uses_the_existing_endpoint_and_a_mocked_request(client, admin_auth, all_keys, monkeypatch, name, mod, cls, key):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"model": "vendor/x", "choices": [{"message": {"content": '{"ok": true}'}}]})
    monkeypatch.setattr(mod, "_factory", lambda s: cls(s, httpx.MockTransport(handler)))
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": name}).json()
    assert r["ok"] is True and r["state"] == "ready" and r["model"] == "vendor/x" and r["provider"] == name
    assert len(seen) == 1 and "image_url" not in json.dumps(seen[0]) and SECRETS[key] not in json.dumps(r)


@pytest.mark.parametrize("name,mod,cls,key", [("groq", groq, GroqProvider, "groq_api_key"), ("pollinations", pollinations, PollinationsProvider, "pollinations_api_key")])
def test_fallback_connection_failure_is_reported_without_the_key(client, admin_auth, all_keys, monkeypatch, name, mod, cls, key):
    monkeypatch.setattr(mod, "_factory", lambda s: cls(s, httpx.MockTransport(
        lambda req: httpx.Response(401, json={"error": {"code": 401, "message": "bad key " + SECRETS[key]}}))))
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": name})
    assert r.status_code == 200 and r.json()["ok"] is False and r.json()["state"] == "misconfigured" and SECRETS[key] not in r.text


def test_a_fallback_without_a_key_reports_not_configured(client, admin_auth, settings, monkeypatch):
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0)
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "groq"}).json()
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert r["ok"] is False and r["state"] == "not_configured" and w["groq"]["state"] == "not_configured" and w["pollinations"]["state"] == "not_configured"


def test_there_is_no_connection_test_for_the_specialists_because_it_would_spend_quota(client, admin_auth, all_keys):
    for name in ("kindwise", "plantnet"):
        ai_admin_service._last_test = 0.0                                                      # the endpoint throttles tests to one per 3 seconds
        assert client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": name}).status_code == 422
    assert by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())["kindwise"]["testable"] is False


def test_the_page_itself_makes_no_provider_calls(client, admin_auth, all_keys, monkeypatch):
    called = []
    monkeypatch.setattr(groq, "_factory", lambda s: GroqProvider(s, httpx.MockTransport(lambda r: called.append(r) or httpx.Response(500))))
    monkeypatch.setattr(pollinations, "_factory", lambda s: PollinationsProvider(s, httpx.MockTransport(lambda r: called.append(r) or httpx.Response(500))))
    client.get(f"{V}/admin/ai", headers=admin_auth)
    assert called == []
