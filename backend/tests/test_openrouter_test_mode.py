"""Admin-only OpenRouter Test Mode: Gemini is bypassed, specialists still run, OpenRouter gets exactly one attempt. OFF = unchanged production flow."""
import json
import logging
import re
from datetime import timedelta

import httpx
import pytest
from pydantic import SecretStr

from app.ai import openrouter, registry
from app.ai.gemini import GeminiProvider
from app.ai.openrouter import OpenRouterProvider
from app.ai.specialists import registry as specialists
from app.ai.specialists.base import DiseaseMatch
from app.services import analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.ai_fakes import diseased
from tests.test_gemini_keys import K1, K2, Wire
from tests.test_openrouter import ORKEY, Rec, analyze, chat
from tests.test_routing import Calls, FakeKindwise, FakePlantNet

V = "/api/v1"


@pytest.fixture
def world(tmp_path, settings, monkeypatch, client, admin_auth):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr(K1)); monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(K2))
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr(ORKEY))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0)); monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    saved = (dict(specialists._IDENTIFIERS), dict(specialists._DIAGNOSERS)); specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    calls = Calls()
    specialists.register_identifier("plantnet", lambda s: FakePlantNet(calls))
    specialists.register_diagnoser("kindwise", lambda s: FakeKindwise(calls, diseases=[DiseaseMatch("Early Blight", 0.8)]))
    gem = Wire()
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(gem.handler)))
    orr = Rec(chat(json.dumps(diseased("Rice Blast", crop="Rice", plant="Rice"))))
    monkeypatch.setattr(openrouter, "_factory", lambda s: OpenRouterProvider(s, orr.transport))
    yield {"calls": calls, "gemini": gem, "or": orr}
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear(); specialists._IDENTIFIERS.update(saved[0]); specialists._DIAGNOSERS.update(saved[1])
    ml_service.reset_ml_service(); registry.register("gemini", lambda s: GeminiProvider(s))


def set_mode(client, admin_auth, on):
    r = client.put(f"{V}/admin/ai", headers=admin_auth, json={"openrouter_test_mode": on})
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------------------------------------- A: OFF = the existing flow
def test_a_default_is_off_and_the_normal_flow_is_unchanged(client, admin_auth, user_auth, world):
    assert client.get(f"{V}/admin/ai", headers=admin_auth).json()["openrouter_test_mode"] is False
    _, r = analyze(client, user_auth)
    assert world["gemini"].calls == ["K1"] and world["or"].requests == []                          # Gemini answered; OpenRouter untouched
    assert world["calls"].order == ["plantnet", "kindwise"] and r.json()["result"]["final"]


def test_a_off_after_on_restores_the_normal_flow_including_the_openrouter_only_after_gemini_fails_rule(client, admin_auth, user_auth, world):
    set_mode(client, admin_auth, True); set_mode(client, admin_auth, False)
    world["gemini"].plan = {K1: [503], K2: [503]}
    _, r = analyze(client, user_auth)
    assert world["gemini"].calls == ["K1", "K2"] and len(world["or"].requests) == 1 and r.json()["result"]["final"]["headline"] == "Rice — Rice Blast"
    assert "test_mode" not in r.json()["result"]


# ----------------------------------------------------------------------------------------------- B, C: ON
def test_b_on_bypasses_gemini_keeps_specialist_routing_and_calls_openrouter_once(client, admin_auth, user_auth, world, caplog):
    caplog.set_level(logging.WARNING)
    assert set_mode(client, admin_auth, True)["openrouter_test_mode"] is True
    _, r = analyze(client, user_auth)
    assert world["gemini"].calls == []                                                            # Gemini was NOT called (no key slot used either)
    assert world["calls"].order == ["plantnet", "kindwise"]                                       # specialist routing unchanged (fixture model is confident: Pl@ntNet, Kindwise)
    assert len(world["or"].requests) == 1                                                         # exactly one OpenRouter request
    assert "OpenRouter test mode enabled: bypassing Gemini" in caplog.text
    sent = json.loads(world["or"].requests[0].content)["messages"][1]["content"]
    assert "Possible plant" in sent[0]["text"] and sent[1]["type"] == "image_url"                # same prompt (with the specialist evidence) and the image


def test_c_openrouter_success_is_the_final_ai_result_and_is_flagged_for_admins_only(client, admin_auth, user_auth, world):
    set_mode(client, admin_auth, True)
    _, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "completed" and b["result"]["final"]["headline"] == "Rice — Rice Blast" and b["result"]["final"]["status"] == "DISEASE"
    assert "test_mode" not in b["result"] and not b["ai_provider"]                                 # nothing for the customer
    aid, _ = analyze(client, admin_auth)
    full = client.get(f"{V}/analyses/{aid}", headers=admin_auth).json()
    assert full["result"]["test_mode"] == "openrouter" and full["ai_provider"] == "openrouter" and full["ai_status"] == "completed"
    provs = full["result"]["specialists"]["providers"]
    assert {"provider": "gemini", "step": "guidance", "status": "bypassed_test_mode"} in provs and {"provider": "openrouter", "step": "guidance", "status": "ok"} in provs


# ----------------------------------------------------------------------------------------------- D: failure
@pytest.mark.parametrize("failure", [httpx.Response(429, json={"error": {"code": 429, "message": "x"}}), httpx.Response(200, text="junk")])
def test_d_openrouter_failure_returns_the_specialist_result_and_never_calls_gemini(client, admin_auth, user_auth, world, failure):
    set_mode(client, admin_auth, True)
    world["or"].response = failure
    _, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "completed" and b["result"]["stage"] == "specialist_only" and b["result"]["final"]["headline"] == "Tomato — Early Blight"
    assert world["gemini"].calls == [] and len(world["or"].requests) == 1


def test_d_with_no_specialist_result_and_openrouter_down_the_existing_preliminary_result_with_retry_applies(client, admin_auth, user_auth, world):
    set_mode(client, admin_auth, True)
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    world["or"].response = httpx.Response(503, json={})
    _, r = analyze(client, user_auth)
    assert r.json()["status"] == "partial" and r.json()["result"]["ai_error"]["retryable"] is True and world["gemini"].calls == []


def test_d_without_an_openrouter_key_test_mode_still_never_calls_gemini(client, admin_auth, user_auth, world, settings, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr(""))
    set_mode(client, admin_auth, True)
    _, r = analyze(client, user_auth)
    assert world["gemini"].calls == [] and r.json()["result"]["stage"] == "specialist_only" and world["or"].requests == []


def test_the_openrouter_diagnostics_are_logged_in_test_mode(client, admin_auth, user_auth, world, caplog):
    caplog.set_level(logging.WARNING)
    set_mode(client, admin_auth, True)
    world["or"].response = httpx.Response(200, json={"model": "m:free", "choices": [{"finish_reason": "length", "message": {"content": "no json " + ORKEY}}]})
    analyze(client, user_auth)
    assert "OpenRouter answer unusable" in caplog.text and "model=m:free" in caplog.text and "finish=length" in caplog.text and "application/json" in caplog.text
    assert ORKEY not in caplog.text


# ----------------------------------------------------------------------------------------------- E: authorization
@pytest.mark.parametrize("method,path,body", [("get", "/admin/ai", None), ("put", "/admin/ai", {"openrouter_test_mode": True}),
                                              ("put", "/admin/settings", {"values": {"openrouter_test_mode": True}})])
def test_e_unauthenticated_and_normal_users_cannot_read_or_change_it(client, user_auth, admin_auth, method, path, body):
    kw = {"json": body} if body is not None else {}
    assert getattr(client, method)(f"{V}{path}", **kw).status_code == 401
    assert getattr(client, method)(f"{V}{path}", headers=user_auth, **kw).status_code == 403
    assert client.get(f"{V}/admin/ai", headers=admin_auth).json()["openrouter_test_mode"] is False       # still off


def test_e_the_setting_is_validated_server_side_and_is_not_public(client, admin_auth, user_auth):
    for bad in ("yes", 1, "true", None):                                                        # None = nothing provided: the existing "No settings provided" 422
        assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"openrouter_test_mode": bad}).status_code == 422, bad
    assert client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"openrouter_test_mode": "yes"}}).status_code == 422
    assert client.get(f"{V}/admin/ai", headers=admin_auth).json()["openrouter_test_mode"] is False
    assert "openrouter_test_mode" not in client.get(f"{V}/config/public").text and "openrouter" not in client.get(f"{V}/config/public").text.lower()
    assert "openrouter" not in client.get(f"{V}/users/me", headers=user_auth).text.lower()


def test_e_changes_are_audited_and_visible_in_the_status(client, admin_auth):
    out = set_mode(client, admin_auth, True)
    assert out["openrouter_test_mode"] is True and out["openrouter"] == {"configured": False, "model": "openrouter/free"}
    log = client.get(f"{V}/admin/audit-log", headers=admin_auth).json()
    items = log["items"] if isinstance(log, dict) else log
    assert any(e["action"] == "settings.update" and "openrouter_test_mode" in e["target"] for e in items)


# ----------------------------------------------------------------------------------------------- F: secrets
def test_f_no_key_in_any_admin_or_customer_response_or_log(client, admin_auth, user_auth, world, caplog):
    caplog.set_level(logging.DEBUG)
    set_mode(client, admin_auth, True)
    bodies = [client.get(f"{V}/admin/ai", headers=admin_auth).text, client.get(f"{V}/admin/settings", headers=admin_auth).text,
              client.get(f"{V}/admin/audit-log", headers=admin_auth).text, client.get(f"{V}/config/public").text]
    _, r = analyze(client, user_auth)
    aid, ra = analyze(client, admin_auth)
    bodies += [r.text, ra.text, client.get(f"{V}/analyses/{aid}", headers=admin_auth).text]
    assert all(ORKEY not in b and K1 not in b and K2 not in b for b in bodies) and ORKEY not in caplog.text
    assert not re.search(r"openrouter|gemini|kindwise|plantnet", r.text, re.I)                      # customer response stays provider-neutral


# ----------------------------------------------------------------------------------------------- regression: the branch must reach OpenRouter and report ITS outcome
def test_regression_success_records_openrouter_as_the_provider_with_the_routed_model_and_a_complete_stage(client, admin_auth, world):
    set_mode(client, admin_auth, True)
    aid, r = analyze(client, admin_auth)
    full = client.get(f"{V}/analyses/{aid}", headers=admin_auth).json()
    assert world["gemini"].calls == [] and len(world["or"].requests) == 1
    assert full["ai_status"] == "completed" and full["ai_provider"] == "openrouter" and full["ai_model"] == "free/vision-model"      # the model OpenRouter actually routed to
    assert full["result"]["stage"] == "complete" and full["result"]["ai"]["plant"] == "Rice" and full["result"]["ai_error"] is None
    assert full["result"]["final"]["headline"] == "Rice — Rice Blast" and full["result"]["test_mode"] == "openrouter"


@pytest.mark.parametrize("failure,status", [(httpx.Response(200, text="junk"), "bad_response"), (httpx.Response(429, json={"error": {"code": 429, "message": "x"}}), "rate_limited"),
                                            (httpx.Response(503, json={}), "unavailable"), (httpx.Response(401, json={"error": {"code": 401, "message": "k"}}), "misconfigured")])
def test_regression_failure_reports_openrouters_real_error_not_bypassed(client, admin_auth, world, failure, status):
    set_mode(client, admin_auth, True)
    world["or"].response = failure
    aid, r = analyze(client, admin_auth)
    full = client.get(f"{V}/analyses/{aid}", headers=admin_auth).json()
    assert full["status"] == "completed" and full["result"]["stage"] == "specialist_only" and full["result"]["final"]["headline"] == "Tomato — Early Blight"
    assert full["ai_status"] == status and full["result"]["ai_error"]["code"] == status and "bypassed" not in json.dumps(full["result"]["ai_error"])
    assert full["ai_provider"] == "openrouter"                                                  # what was actually tried, not Gemini
    assert world["gemini"].calls == [] and len(world["or"].requests) == 1                        # Gemini never called; OpenRouter called once
    provs = full["result"]["specialists"]["providers"]
    assert {"provider": "openrouter", "step": "guidance", "status": status} in provs and {"provider": "gemini", "step": "guidance", "status": "bypassed_test_mode"} in provs


def test_regression_a_missing_openrouter_key_is_reported_as_not_configured_and_logged(client, admin_auth, world, settings, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr(""))
    set_mode(client, admin_auth, True)
    aid, _ = analyze(client, admin_auth)
    full = client.get(f"{V}/analyses/{aid}", headers=admin_auth).json()
    assert full["ai_status"] == "not_configured" and full["ai_provider"] == "openrouter" and world["gemini"].calls == [] and world["or"].requests == []
    assert "OpenRouter test mode: OPENROUTER_API_KEY is not set on the server" in caplog.text


def test_regression_no_specialist_result_keeps_the_preliminary_result_with_the_real_error(client, admin_auth, user_auth, world):
    set_mode(client, admin_auth, True)
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    world["or"].response = httpx.Response(200, text="junk")
    _, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "partial" and b["result"]["final"] is None and b["result"]["ai_error"]["retryable"] is True and b["ai_status"] == "bad_response"
    assert world["gemini"].calls == [] and len(world["or"].requests) == 1


def test_regression_the_e73e282_diagnostics_are_still_logged_and_secrets_stay_out(client, admin_auth, world, caplog):
    caplog.set_level(logging.WARNING)
    set_mode(client, admin_auth, True)
    world["or"].response = httpx.Response(200, json={"model": "routed/model:free", "choices": [{"finish_reason": "length", "message": {"content": "nope " + ORKEY}}]})
    aid, ra = analyze(client, admin_auth)
    assert "OpenRouter answer unusable" in caplog.text and "HTTP 200" in caplog.text and "application/json" in caplog.text
    assert "model=routed/model:free" in caplog.text and "finish=length" in caplog.text and "nope" in caplog.text
    assert ORKEY not in caplog.text and ORKEY not in ra.text and "base64" not in caplog.text and "data:image" not in caplog.text
