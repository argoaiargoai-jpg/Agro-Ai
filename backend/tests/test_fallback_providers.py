"""Generative fallbacks after Gemini: Groq, then Pollinations, then the specialist result. All HTTP is mocked; no real key is used."""
import base64
import io
import json
import logging
import re
from datetime import timedelta

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from app.ai import groq, openai_compat, pollinations, registry
from app.ai.base import AIRequest, ProviderBadResponse, ProviderMisconfigured, ProviderNotConfigured, ProviderRateLimited, ProviderTimeout, ProviderUnavailable
from app.ai.gemini import GeminiProvider
from app.ai.groq import GroqProvider
from app.ai.pollinations import PollinationsProvider
from app.ai.specialists import registry as specialists
from app.ai.specialists.base import DiseaseMatch
from app.core.config import Settings
from app.services import analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.ai_fakes import diseased, payload
from tests.test_gemini_keys import K1, K2, Wire
from tests.test_routing import Calls, FakeKindwise, FakePlantNet

V = "/api/v1"
GKEY = "gsk_" + "TESTGROQKEY0123456789abcdef"
PKEY = "sk_" + "TESTPOLLKEY0123456789abcdef"
RED = (255, 0, 0)
IMG = b"\xff\xd8\xff\xe0FALLBACKIMG" * 10
REQ = AIRequest(system_instruction="sys", prompt="the prompt", json_schema={"type": "object", "properties": {"x": {"type": "string"}}}, image=IMG, image_mime="image/jpeg")

ADAPTERS = [
    pytest.param(GroqProvider, "groq_api_key", GKEY, "https://api.groq.com/openai/v1/chat/completions", "qwen/qwen3.8-27b", id="groq"),
    pytest.param(PollinationsProvider, "pollinations_api_key", PKEY, "https://gen.pollinations.ai/v1/chat/completions", "openai/gpt-5.4-nano", id="pollinations"),
]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(openai_compat, "_SLEEP", lambda s: None)


def S(field, key, **kw) -> Settings:
    return Settings(_env_file=None, environment="test", **{field: key}, **kw)


def chat(content, status=200, **extra):
    return httpx.Response(status, json={"model": "vendor/model-x", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 5, "completion_tokens": 7}, **extra})


class Rec:
    """Mock endpoint: `responses` are consumed in order (the last one repeats); an Exception instance is raised."""

    def __init__(self, *responses):
        self.requests, self.responses = [], list(responses)

    def __call__(self, req):
        self.requests.append(req)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    @property
    def transport(self):
        return httpx.MockTransport(self)


GOOD = chat(json.dumps(payload()))


# ------------------------------------------------------------------------------------------- the adapters
@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
def test_success_sends_image_prompt_schema_with_the_key_only_in_the_header(cls, field, key, url, model):
    rec = Rec(chat(json.dumps({"x": "y"})))
    r = cls(S(field, key), rec.transport).analyze(REQ)
    assert r.data == {"x": "y"} and r.model == "vendor/model-x" and r.usage["output_tokens"] == 7
    req = rec.requests[0]
    body = json.loads(req.content)
    assert str(req.url) == url and req.headers["authorization"] == f"Bearer {key}" and key not in req.content.decode() and key not in str(req.url)
    assert body["model"] == model and body["response_format"] == {"type": "json_object"}
    assert "the prompt" in json.dumps(body["messages"]) and '"x"' in body["messages"][0]["content"]                   # schema is in the system text
    part = body["messages"][1]["content"][1]["image_url"]["url"]
    assert part.startswith("data:image/jpeg;base64,") and base64.b64decode(part.split(",", 1)[1]) == IMG                # the original image bytes


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
@pytest.mark.parametrize("content", ['```json\n{"x": "y"}\n```', 'Sure! {"x": "y"} hope it helps', '<think>hmm</think>{"x": "y"}'])
def test_json_wrapped_in_fences_prose_or_thinking_is_still_extracted(cls, field, key, url, model, content):
    assert cls(S(field, key), Rec(chat(content)).transport).analyze(REQ).data == {"x": "y"}


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
@pytest.mark.parametrize("resp", [
    chat(""), chat(None), chat("   "), chat("I cannot help with that."), chat("[1, 2]"), chat("{}"), chat('{"x": '),
    httpx.Response(200, json={"choices": []}), httpx.Response(200, json={"nothing": 1}), httpx.Response(200, text="<html>gateway</html>"),
    httpx.Response(200, json={"choices": [{"message": {"content": '{"x": "y"}'}, "finish_reason": "length"}]}),
], ids=lambda r: "")
def test_http_200_with_an_unusable_answer_is_a_failure(cls, field, key, url, model, resp):
    with pytest.raises(ProviderBadResponse):
        cls(S(field, key), Rec(resp).transport).analyze(REQ)


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
def test_an_error_envelope_inside_http_200_is_an_error(cls, field, key, url, model):
    with pytest.raises(ProviderRateLimited):
        cls(S(field, key), Rec(httpx.Response(200, json={"error": {"code": 429, "message": "slow down"}})).transport).analyze(REQ)


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
def test_missing_key_means_not_configured_and_no_request(cls, field, key, url, model):
    rec = Rec(GOOD)
    p = cls(Settings(_env_file=None), rec.transport)
    assert not p.is_configured()
    with pytest.raises(ProviderNotConfigured):
        p.analyze(REQ)
    assert rec.requests == []


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
@pytest.mark.parametrize("status,exc", [(408, ProviderTimeout), (429, ProviderRateLimited), (500, ProviderUnavailable), (502, ProviderUnavailable),
                                        (503, ProviderUnavailable), (504, ProviderUnavailable)])
def test_transient_statuses_are_retried_with_bounded_exponential_backoff(cls, field, key, url, model, status, exc, monkeypatch):
    sleeps = []
    monkeypatch.setattr(openai_compat, "_SLEEP", lambda s: sleeps.append(s))
    rec = Rec(httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(exc):
        cls(S(field, key, ai_max_retries=2), rec.transport).analyze(REQ)
    assert len(rec.requests) == 3 and sleeps == [1.0, 2.0]                                   # 1 + 2 retries, then it gives up


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
def test_a_transient_failure_then_success_recovers(cls, field, key, url, model):
    rec = Rec(httpx.Response(503, json={}), httpx.Response(429, json={}), GOOD)
    assert cls(S(field, key, ai_max_retries=2), rec.transport).analyze(REQ).data["plant_present"] is True and len(rec.requests) == 3


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow")])
def test_timeouts_are_retried_then_reported(cls, field, key, url, model, exc):
    rec = Rec(exc)
    with pytest.raises(ProviderTimeout):
        cls(S(field, key, ai_max_retries=1), rec.transport).analyze(REQ)
    assert len(rec.requests) == 2


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
@pytest.mark.parametrize("status,exc", [(400, ProviderBadResponse), (401, ProviderMisconfigured), (403, ProviderMisconfigured), (404, ProviderMisconfigured)])
def test_permanent_errors_are_never_retried(cls, field, key, url, model, status, exc):
    rec = Rec(httpx.Response(status, json={"error": {"message": "bad key " + key}}))
    with pytest.raises(exc) as e:
        cls(S(field, key, ai_max_retries=2), rec.transport).analyze(REQ)
    assert len(rec.requests) == 1 and key not in str(e.value) and key not in e.value.detail


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
def test_the_retry_budget_stops_further_retries(cls, field, key, url, model):
    rec = Rec(httpx.Response(503, json={}))
    with pytest.raises(ProviderUnavailable):
        cls(S(field, key, ai_max_retries=5, ai_retry_budget_seconds=0), rec.transport).analyze(REQ)
    assert len(rec.requests) == 1


@pytest.mark.parametrize("cls,field,key,url,model", ADAPTERS)
def test_ping_is_text_only_single_attempt_and_never_exposes_the_key(cls, field, key, url, model):
    rec = Rec(chat('{"ok": true}'))
    out = cls(S(field, key), rec.transport).ping()
    assert out["ok"] is True and "image_url" not in rec.requests[0].content.decode() and key not in json.dumps(out)


def test_defaults_and_secrets():
    s = Settings(_env_file=None, groq_api_key=GKEY, pollinations_api_key=PKEY)
    assert s.groq_model == "qwen/qwen3.8-27b" and s.pollinations_model == "openai/gpt-5.4-nano" and s.ai_timeout_seconds == 120.0
    assert GKEY in s.secret_values() and PKEY in s.secret_values() and GKEY not in repr(s) and PKEY not in repr(s)
    assert "groq" not in registry.names() and "pollinations" not in registry.names()


# ------------------------------------------------------------------------------------------- in the workflow
@pytest.fixture
def world(tmp_path, settings, monkeypatch, client, admin_auth):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr(K1)); monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(K2))
    monkeypatch.setattr(settings, "groq_api_key", SecretStr(GKEY)); monkeypatch.setattr(settings, "pollinations_api_key", SecretStr(PKEY))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0)); monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    saved = (dict(specialists._IDENTIFIERS), dict(specialists._DIAGNOSERS)); specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    state = {"groq": Rec(chat(json.dumps(diseased("Rice Blast", crop="Rice", plant="Rice")))),
             "poll": Rec(chat(json.dumps(diseased("Wheat Rust", crop="Wheat", plant="Wheat")))), "calls": Calls()}
    monkeypatch.setattr(groq, "_factory", lambda s: GroqProvider(s, state["groq"].transport))
    monkeypatch.setattr(pollinations, "_factory", lambda s: PollinationsProvider(s, state["poll"].transport))
    state["gemini"] = Wire()
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(state["gemini"].handler)))
    yield state
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear(); specialists._IDENTIFIERS.update(saved[0]); specialists._DIAGNOSERS.update(saved[1])
    ml_service.reset_ml_service(); registry.register("gemini", lambda s: GeminiProvider(s))


def png(color=RED):
    b = io.BytesIO(); Image.new("RGB", (320, 240), color).save(b, "PNG"); return b.getvalue()


def analyze(client, headers, color=RED):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def gemini_down(world):
    world["gemini"].plan = {K1: [503], K2: [503]}


def with_specialist(world):
    specialists.register_diagnoser("kindwise", lambda s: FakeKindwise(world["calls"], diseases=[DiseaseMatch("Early Blight", 0.8)]))
    specialists.register_identifier("plantnet", lambda s: FakePlantNet(world["calls"]))


def provs(client, admin_auth, aid):
    full = client.get(f"{V}/analyses/{aid}", headers=admin_auth).json()
    return full, full["result"]["specialists"]["providers"]


def test_chain_1_gemini_succeeds_so_neither_fallback_is_called(client, user_auth, world):
    _, r = analyze(client, user_auth)
    assert r.json()["result"]["final"]["status"] == "HEALTHY" and world["groq"].requests == [] and world["poll"].requests == []


def test_chain_gemini_key_rotation_and_key_failover_come_before_any_fallback(client, user_auth, world):
    world["gemini"].plan = {K1: [503], K2: ["ok"]}
    _, r = analyze(client, user_auth)
    assert world["gemini"].calls == ["K1", "K2"] and r.json()["status"] == "completed" and world["groq"].requests == []


def test_chain_2_gemini_fails_so_groq_answers_and_pollinations_is_not_called(client, user_auth, admin_auth, world):
    gemini_down(world)
    aid, r = analyze(client, user_auth)
    assert r.json()["status"] == "completed" and r.json()["result"]["final"]["headline"] == "Rice — Rice Blast"
    assert len(world["groq"].requests) == 1 and world["poll"].requests == []
    sent = json.loads(world["groq"].requests[0].content)["messages"][1]["content"]
    assert base64.b64decode(sent[1]["image_url"]["url"].split(",", 1)[1])[:2] == b"\xff\xd8" and "INDEPENDENT_VERIFICATION" in sent[0]["text"]   # same image + prompt
    aid, _ = analyze(client, admin_auth)
    full, pv = provs(client, admin_auth, aid)
    assert full["ai_provider"] == "groq" and full["ai_status"] == "completed" and {"provider": "groq", "step": "guidance", "status": "ok"} in pv


@pytest.mark.parametrize("groq_failure", [httpx.Response(429, json={}), httpx.Response(503, json={}), chat(""), chat("not json"), httpx.Response(401, json={})])
def test_chain_3_groq_fails_so_pollinations_answers(client, user_auth, admin_auth, world, groq_failure):
    gemini_down(world); world["groq"].responses = [groq_failure]
    aid, r = analyze(client, admin_auth)
    assert r.json()["result"]["final"]["headline"] == "Wheat — Wheat Rust" and len(world["poll"].requests) == 1
    full, pv = provs(client, admin_auth, aid)
    assert full["ai_provider"] == "pollinations" and {"provider": "pollinations", "step": "guidance", "status": "ok"} in pv
    assert any(p["provider"] == "groq" and p["status"] != "ok" for p in pv)


def test_chain_3b_an_off_schema_groq_answer_is_not_accepted(client, user_auth, world):
    gemini_down(world); world["groq"].responses = [chat(json.dumps({"hello": "world"}))]
    _, r = analyze(client, user_auth)
    assert r.json()["result"]["final"]["headline"] == "Wheat — Wheat Rust"


def test_chain_4_everything_fails_so_the_latest_specialist_result_is_returned(client, user_auth, world):
    with_specialist(world); gemini_down(world)
    world["groq"].responses = [httpx.Response(503, json={})]; world["poll"].responses = [chat("")]
    _, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "completed" and b["result"]["stage"] == "specialist_only" and b["result"]["final"]["headline"] == "Tomato — Early Blight"
    assert len(world["groq"].requests) == 1 and len(world["poll"].requests) == 1


def test_chain_4b_everything_fails_with_no_specialist_keeps_the_preliminary_result_and_retry(client, user_auth, world):
    gemini_down(world)
    world["groq"].responses = [httpx.Response(503, json={})]; world["poll"].responses = [httpx.Response(503, json={})]
    aid, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "partial" and b["result"]["ml"]["classification_type"] == "DISEASE" and b["result"]["final"] is None and b["result"]["ai_error"]["retryable"] is True


def test_missing_fallback_keys_are_skipped_cleanly(client, user_auth, world, settings, monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", SecretStr("")); monkeypatch.setattr(settings, "pollinations_api_key", SecretStr(""))
    with_specialist(world); gemini_down(world)
    _, r = analyze(client, user_auth)
    assert r.json()["result"]["stage"] == "specialist_only" and world["groq"].requests == [] and world["poll"].requests == []


def test_a_missing_groq_key_still_lets_pollinations_answer(client, user_auth, world, settings, monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", SecretStr(""))
    gemini_down(world)
    _, r = analyze(client, user_auth)
    assert r.json()["result"]["final"]["headline"] == "Wheat — Wheat Rust" and world["groq"].requests == []


def test_a_fallback_alone_is_enough_when_gemini_has_no_keys(client, user_auth, world, settings, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr("")); monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(""))
    _, r = analyze(client, user_auth)
    assert r.json()["status"] == "completed" and r.json()["result"]["final"]["headline"] == "Rice — Rice Blast"


def test_specialist_routing_is_unchanged_underneath_the_chain(client, user_auth, world):
    with_specialist(world); gemini_down(world)
    analyze(client, user_auth)
    assert world["calls"].order == ["plantnet", "kindwise"] or world["calls"].order == ["kindwise"] or world["calls"].order[0] in ("kindwise", "plantnet")


# ---------------------------------------------------------------------------------------- Gemini bypass (admin switch)
def set_bypass(client, admin_auth, on):
    r = client.put(f"{V}/admin/ai", headers=admin_auth, json={"generative_ai_bypass_gemini": on})
    assert r.status_code == 200, r.text
    return r.json()


def test_bypass_defaults_off_and_gemini_is_used_normally(client, admin_auth, user_auth, world):
    assert client.get(f"{V}/admin/ai", headers=admin_auth).json()["generative_ai_bypass_gemini"] is False
    analyze(client, user_auth)
    assert world["gemini"].calls == ["K1"] and world["groq"].requests == []


def test_bypass_on_skips_gemini_keeps_specialists_and_goes_to_groq(client, admin_auth, user_auth, world, caplog):
    caplog.set_level(logging.WARNING)
    with_specialist(world)
    assert set_bypass(client, admin_auth, True)["generative_ai_bypass_gemini"] is True
    aid, r = analyze(client, admin_auth)
    assert world["gemini"].calls == [] and len(world["groq"].requests) == 1 and world["poll"].requests == []
    assert world["calls"].order[0] in ("kindwise", "plantnet") and "Gemini bypass enabled" in caplog.text
    full, pv = provs(client, admin_auth, aid)
    assert full["result"]["test_mode"] == "bypass_gemini" and full["ai_provider"] == "groq" and {"provider": "gemini", "step": "guidance", "status": "bypassed"} in pv


def test_bypass_on_groq_fails_so_pollinations_answers(client, admin_auth, user_auth, world):
    set_bypass(client, admin_auth, True); world["groq"].responses = [httpx.Response(503, json={})]
    aid, r = analyze(client, admin_auth)
    full, _ = provs(client, admin_auth, aid)
    assert world["gemini"].calls == [] and full["ai_provider"] == "pollinations" and r.json()["result"]["final"]["headline"] == "Wheat — Wheat Rust"


def test_bypass_on_all_fail_returns_the_specialist_result_and_never_calls_gemini(client, admin_auth, user_auth, world):
    with_specialist(world); set_bypass(client, admin_auth, True)
    world["groq"].responses = [chat("")]; world["poll"].responses = [httpx.Response(500, json={})]
    _, r = analyze(client, user_auth)
    assert world["gemini"].calls == [] and r.json()["result"]["stage"] == "specialist_only" and r.json()["result"]["final"]["headline"] == "Tomato — Early Blight"


def test_bypass_off_again_restores_gemini_first(client, admin_auth, user_auth, world):
    set_bypass(client, admin_auth, True); set_bypass(client, admin_auth, False)
    analyze(client, user_auth)
    assert world["gemini"].calls == ["K1"] and world["groq"].requests == []


# ---------------------------------------------------------------------------------------- customers see nothing provider-specific
FORBIDDEN = re.compile(r"groq|pollinations|gemini|kindwise|plantnet|pl@ntnet|mobilenet|routing|fallback|specialist|bypass", re.I)


def test_customer_responses_are_provider_neutral_in_every_branch(client, user_auth, world):
    for setup in (lambda: None, lambda: gemini_down(world)):
        setup()
        aid, r = analyze(client, user_auth)
        for text in (r.text, client.get(f"{V}/analyses/{aid}", headers=user_auth).text, client.get(f"{V}/analyses", headers=user_auth).text):
            assert not FORBIDDEN.search(text.replace("specialist_only", "")), FORBIDDEN.search(text)


def test_provider_keys_never_reach_logs_or_responses(client, user_auth, admin_auth, world, caplog):
    caplog.set_level(logging.DEBUG)
    gemini_down(world); world["groq"].responses = [httpx.Response(401, json={"error": {"message": "bad key " + GKEY}})]
    world["poll"].responses = [httpx.Response(500, json={"error": {"message": "boom " + PKEY}})]
    with_specialist(world)
    aid, r = analyze(client, admin_auth)
    blob = r.text + client.get(f"{V}/analyses/{aid}", headers=admin_auth).text + client.get(f"{V}/admin/ai", headers=admin_auth).text + caplog.text
    for secret in (GKEY, PKEY, K1, K2):
        assert secret not in blob
