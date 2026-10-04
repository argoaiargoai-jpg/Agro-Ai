"""Admin AI page: status of ALL providers the workflow uses (safe metadata only). No routing or behaviour change."""
import json

import httpx
import pytest
from pydantic import SecretStr

from app.ai import openrouter
from app.ai.openrouter import OpenRouterProvider
from app.services import ai_admin_service

V = "/api/v1"
SECRETS = {"gemini_api_key_1": "gem-one-SECRETVALUE-111111", "gemini_api_key_2": "gem-two-SECRETVALUE-222222", "kindwise_api_key": "kw-SECRETVALUE-333333",
           "plantnet_api_key": "pn-SECRETVALUE-444444", "openrouter_api_key": "sk-or-v1-SECRETVALUE-555555"}


@pytest.fixture
def all_keys(settings, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setattr(settings, k, SecretStr(v))
    monkeypatch.setattr(ai_admin_service, "_last_test", 0.0)


def by_id(body):
    return {p["id"]: p for p in body["workflow"]}


def test_admin_receives_all_four_providers_with_safe_metadata(client, admin_auth, all_keys):
    body = client.get(f"{V}/admin/ai", headers=admin_auth).json()
    w = by_id(body)
    assert list(w) == ["gemini", "kindwise", "plantnet", "openrouter"]
    assert w["gemini"] == {"id": "gemini", "name": "Google Gemini", "purpose": "AI verification & guidance (main AI step)", "configured": True, "model": "gemini-3.8-flash",
                           "keys": {"configured": 2, "of": 2}, "state": "primary", "testable": True}
    assert w["kindwise"]["configured"] is True and w["kindwise"]["state"] == "available" and "model" not in w["kindwise"] and w["kindwise"]["testable"] is False
    assert w["plantnet"]["configured"] is True and w["plantnet"]["state"] == "available" and w["plantnet"]["purpose"] == "Plant identification specialist"
    assert w["openrouter"]["configured"] is True and w["openrouter"]["model"] == "openrouter/free" and w["openrouter"]["state"] == "available" and w["openrouter"]["testable"] is True
    assert body["openrouter_test_mode"] is False


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


def test_states_follow_the_ai_switch_and_test_mode(client, admin_auth, all_keys):
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": False})
    assert by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())["gemini"]["state"] == "disabled"
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"enabled": True, "openrouter_test_mode": True})
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert w["gemini"]["state"] == "bypassed" and w["openrouter"]["state"] == "test_mode" and w["kindwise"]["state"] == "available"
    client.put(f"{V}/admin/ai", headers=admin_auth, json={"openrouter_test_mode": False})
    w = by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())
    assert w["gemini"]["state"] == "primary" and w["openrouter"]["state"] == "available"


def test_existing_gemini_status_fields_still_work(client, admin_auth, all_keys):
    body = client.get(f"{V}/admin/ai", headers=admin_auth).json()
    assert body["state"] == "ready" and body["active_provider"] == "gemini" and body["enabled"] is True
    assert [p["name"] for p in body["providers"]] == ["gemini"] and body["providers"][0]["configured"] is True and body["secret_source"] == "environment"
    assert {"limits", "usage_24h"} <= set(body)


@pytest.mark.parametrize("method,path,body", [("get", "/admin/ai", None), ("post", "/admin/ai/test", {"provider": "openrouter"})])
def test_only_admins_can_see_provider_configuration(client, user_auth, admin_auth, method, path, body):
    kw = {"json": body} if body is not None else {}
    assert getattr(client, method)(f"{V}{path}", **kw).status_code == 401
    assert getattr(client, method)(f"{V}{path}", headers=user_auth, **kw).status_code == 403


def test_public_and_customer_endpoints_carry_no_provider_information(client, user_auth, all_keys):
    for url in (f"{V}/config/public", f"{V}/users/me", f"{V}/health", f"{V}/ml/info"):
        h = {"headers": user_auth} if "users" in url else {}
        t = client.get(url, **h).text.lower()
        assert not any(w in t for w in ("openrouter", "kindwise", "plantnet", "gemini", "secretvalue")), url


def test_openrouter_connection_test_uses_the_existing_endpoint_and_a_mocked_request(client, admin_auth, all_keys, monkeypatch):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"model": "free/x", "choices": [{"message": {"content": '{"ok": true}'}}]})
    monkeypatch.setattr(openrouter, "_factory", lambda s: OpenRouterProvider(s, httpx.MockTransport(handler)))
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "openrouter"}).json()
    assert r["ok"] is True and r["state"] == "ready" and r["model"] == "free/x" and r["provider"] == "openrouter"
    assert len(seen) == 1 and "image_url" not in json.dumps(seen[0]) and SECRETS["openrouter_api_key"] not in json.dumps(r)


def test_openrouter_connection_failure_is_reported_without_the_key(client, admin_auth, all_keys, monkeypatch):
    monkeypatch.setattr(openrouter, "_factory", lambda s: OpenRouterProvider(s, httpx.MockTransport(
        lambda req: httpx.Response(401, json={"error": {"code": 401, "message": "bad key " + SECRETS["openrouter_api_key"]}}))))
    r = client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": "openrouter"})
    assert r.status_code == 200 and r.json()["ok"] is False and r.json()["state"] == "misconfigured" and SECRETS["openrouter_api_key"] not in r.text


def test_there_is_no_connection_test_for_the_specialists_because_it_would_spend_quota(client, admin_auth, all_keys):
    for name in ("kindwise", "plantnet"):
        ai_admin_service._last_test = 0.0                                                      # the endpoint throttles tests to one per 3 seconds
        assert client.post(f"{V}/admin/ai/test", headers=admin_auth, json={"provider": name}).status_code == 422
    assert by_id(client.get(f"{V}/admin/ai", headers=admin_auth).json())["kindwise"]["testable"] is False


def test_the_page_itself_makes_no_provider_calls(client, admin_auth, all_keys, monkeypatch):
    called = []
    monkeypatch.setattr(openrouter, "_factory", lambda s: OpenRouterProvider(s, httpx.MockTransport(lambda r: called.append(r) or httpx.Response(500))))
    client.get(f"{V}/admin/ai", headers=admin_auth)
    assert called == []
