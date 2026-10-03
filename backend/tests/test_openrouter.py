"""OpenRouter: one last attempt after BOTH Gemini keys fail; otherwise the existing specialist fallback. HTTP is mocked; no real key is used."""
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

from app.ai import openrouter, registry
from app.ai.base import AIRequest, ProviderBadResponse, ProviderMisconfigured, ProviderNotConfigured, ProviderRateLimited, ProviderTimeout, ProviderUnavailable
from app.ai.gemini import GeminiProvider
from app.ai.openrouter import OpenRouterProvider
from app.ai.specialists import registry as specialists
from app.ai.specialists.base import DiseaseMatch
from app.core.config import Settings
from app.services import analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.test_gemini_keys import K1, K2, Wire
from tests.test_routing import Calls, FakeKindwise, FakePlantNet
from tests.ai_fakes import diseased, payload

V = "/api/v1"
ORKEY = "sk-or-v1-" + "TESTKEY0123456789abcdef"
RED = (255, 0, 0)
IMG = b"\xff\xd8\xff\xe0OPENROUTERIMG" * 10
REQ = AIRequest(system_instruction="sys", prompt="the prompt", json_schema={"type": "object", "properties": {"x": {"type": "string"}}}, image=IMG, image_mime="image/jpeg")


def S(**kw) -> Settings:
    return Settings(_env_file=None, environment="test", openrouter_api_key=ORKEY, **kw)


def chat(content, status=200, **extra):
    return httpx.Response(status, json={"model": "free/vision-model", "choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}, **extra})


class Rec:
    def __init__(self, response=None, exc=None):
        self.requests, self.response, self.exc = [], response, exc

    def __call__(self, req):
        self.requests.append(req)
        if self.exc:
            raise self.exc
        return self.response

    @property
    def transport(self):
        return httpx.MockTransport(self)


# ------------------------------------------------------------------------------------------- the adapter
def test_request_carries_the_same_image_prompt_and_schema_and_the_key_only_in_the_header():
    rec = Rec(chat(json.dumps(payload())))
    r = OpenRouterProvider(S(), rec.transport).analyze(REQ)
    req = rec.requests[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions" and req.headers["authorization"] == f"Bearer {ORKEY}"
    assert ORKEY not in req.content.decode() and ORKEY not in str(req.url)
    assert body["model"] == "openrouter/free"
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "the prompt"} and user[1]["type"] == "image_url"
    url = user[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,") and base64.b64decode(url.split(",", 1)[1]) == IMG           # the actual image bytes
    assert "JSON Schema" in body["messages"][0]["content"] and '"x"' in body["messages"][0]["content"] and body["messages"][0]["content"].startswith("sys")
    assert r.data["plant_present"] is True and r.model == "free/vision-model" and r.usage == {"prompt_tokens": 5, "output_tokens": 7}


@pytest.mark.parametrize("content", ["```json\n" + json.dumps(payload()) + "\n```", "Here you go: " + json.dumps(payload()) + " done", [{"type": "text", "text": json.dumps(payload())}]])
def test_lenient_json_extraction(content):
    assert OpenRouterProvider(S(), Rec(chat(content)).transport).analyze(REQ).data["health_status"] == "healthy"


@pytest.mark.parametrize("status,cls", [(402, ProviderRateLimited), (429, ProviderRateLimited), (401, ProviderMisconfigured), (403, ProviderMisconfigured),
                                        (408, ProviderTimeout), (500, ProviderUnavailable), (502, ProviderUnavailable), (503, ProviderUnavailable), (400, ProviderBadResponse)])
def test_http_errors_map_to_neutral_provider_errors_without_the_key(status, cls):
    resp = httpx.Response(status, json={"error": {"code": status, "message": "boom " + ORKEY}}, headers={"retry-after": "7"})
    with pytest.raises(cls) as e:
        OpenRouterProvider(S(), Rec(resp).transport).analyze(REQ)
    assert ORKEY not in str(e.value) and ORKEY not in e.value.detail


def test_a_200_response_carrying_an_error_object_is_an_error():
    with pytest.raises(ProviderRateLimited):
        OpenRouterProvider(S(), Rec(httpx.Response(200, json={"error": {"code": 429, "message": "rate limited upstream"}})).transport).analyze(REQ)


@pytest.mark.parametrize("resp", [httpx.Response(200, text="<html>no</html>"), chat(""), chat("not json at all"), chat("[1, 2]"), httpx.Response(200, json={"choices": []})])
def test_unusable_answers_are_bad_responses(resp):
    with pytest.raises(ProviderBadResponse):
        OpenRouterProvider(S(), Rec(resp).transport).analyze(REQ)


@pytest.mark.parametrize("exc,cls", [(httpx.ReadTimeout("t"), ProviderTimeout), (httpx.ConnectError("c"), ProviderUnavailable)])
def test_timeouts_and_connection_failures(exc, cls):
    with pytest.raises(cls):
        OpenRouterProvider(S(), Rec(exc=exc).transport).analyze(REQ)


def test_no_key_means_not_configured_and_no_request():
    rec = Rec(chat("{}"))
    p = OpenRouterProvider(Settings(_env_file=None), rec.transport)
    assert not p.is_configured()
    with pytest.raises(ProviderNotConfigured):
        p.analyze(REQ)
    assert not rec.requests


def test_the_key_is_a_redacted_secret():
    s = S()
    assert ORKEY in s.secret_values() and ORKEY not in repr(s)


def test_it_is_not_a_selectable_admin_provider():
    assert "openrouter" not in registry.names()


# ------------------------------------------------------------------------------------------- in the workflow
@pytest.fixture
def world(tmp_path, settings, monkeypatch, client, admin_auth):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr(K1)); monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(K2))
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr(ORKEY))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0)); monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    saved = (dict(specialists._IDENTIFIERS), dict(specialists._DIAGNOSERS)); specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear()
    state = {"or": Rec(chat(json.dumps(diseased("Rice Blast", crop="Rice", plant="Rice")))), "calls": Calls()}
    monkeypatch.setattr(openrouter, "_factory", lambda s: OpenRouterProvider(s, state["or"].transport))
    yield state
    specialists._IDENTIFIERS.clear(); specialists._DIAGNOSERS.clear(); specialists._IDENTIFIERS.update(saved[0]); specialists._DIAGNOSERS.update(saved[1])
    ml_service.reset_ml_service(); registry.register("gemini", lambda s: GeminiProvider(s))


def png(color=RED):
    b = io.BytesIO(); Image.new("RGB", (320, 240), color).save(b, "PNG"); return b.getvalue()


def use_gemini(w):
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(w.handler)))


def analyze(client, headers, color=RED):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def test_1_gemini_succeeds_so_openrouter_is_not_called(client, user_auth, world):
    use_gemini(Wire())
    _, r = analyze(client, user_auth)
    assert r.json()["result"]["final"]["status"] == "HEALTHY" and world["or"].requests == []          # Gemini answered (the mocked answer is healthy)


def test_2_key1_fails_key2_succeeds_so_openrouter_is_not_called(client, user_auth, world):
    w = Wire({K1: [503]}); use_gemini(w)
    _, r = analyze(client, user_auth)
    assert w.calls == ["K1", "K2"] and r.json()["status"] == "completed" and world["or"].requests == []


def test_3_both_gemini_keys_fail_so_openrouter_is_called_once(client, user_auth, world):
    w = Wire({K1: [503], K2: [429]}); use_gemini(w)
    _, r = analyze(client, user_auth)
    assert w.calls == ["K1", "K2"] and len(world["or"].requests) == 1                      # exactly one attempt, only after both keys failed
    assert r.json()["status"] == "completed"


def test_4_openrouter_success_is_the_final_result_and_gets_the_same_image(client, user_auth, admin_auth, world):
    w = Wire({K1: [503], K2: [503]}); use_gemini(w)
    _, r = analyze(client, user_auth)
    f = r.json()["result"]["final"]
    assert f["headline"] == "Rice — Rice Blast" and f["status"] == "DISEASE" and r.json()["status"] == "completed"
    sent = json.loads(world["or"].requests[0].content)["messages"][1]["content"]
    assert base64.b64decode(sent[1]["image_url"]["url"].split(",", 1)[1])[:2] == b"\xff\xd8"          # the original image (re-encoded JPEG) went along
    assert "INDEPENDENT_VERIFICATION" in sent[0]["text"]                                         # the same prompt Gemini got
    aid, _ = analyze(client, admin_auth)
    full = client.get(f"{V}/analyses/{aid}", headers=admin_auth).json()
    assert full["ai_provider"] == "openrouter" and full["ai_status"] == "completed"
    assert {"provider": "openrouter", "step": "guidance", "status": "ok"} in full["result"]["specialists"]["providers"]


@pytest.mark.parametrize("failure", [httpx.Response(429, json={"error": {"code": 429, "message": "x"}}), httpx.Response(503, json={}), httpx.Response(200, text="junk")])
def test_5_openrouter_failure_returns_the_existing_specialist_result(client, user_auth, world, failure):
    world["or"].response = failure
    specialists.register_diagnoser("kindwise", lambda s: FakeKindwise(world["calls"], diseases=[DiseaseMatch("Early Blight", 0.8)]))
    specialists.register_identifier("plantnet", lambda s: FakePlantNet(world["calls"]))
    use_gemini(Wire({K1: [503], K2: [503]}))
    _, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "completed" and b["result"]["stage"] == "specialist_only" and b["result"]["final"]["headline"] == "Tomato — Early Blight"
    assert len(world["or"].requests) == 1                                                       # never retried


def test_5b_openrouter_fails_and_no_specialist_result_keeps_the_existing_preliminary_behaviour(client, user_auth, world):
    world["or"].response = httpx.Response(503, json={})
    use_gemini(Wire({K1: [503], K2: [503]}))
    aid, r = analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "partial" and b["result"]["final"] is None and b["result"]["ai_error"]["retryable"] is True
    assert len(world["or"].requests) == 1


def test_openrouter_alone_is_enough_when_gemini_has_no_keys(client, user_auth, world, settings, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr("")); monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(""))
    registry.register("gemini", lambda s: GeminiProvider(s))
    _, r = analyze(client, user_auth)
    assert r.json()["status"] == "completed" and r.json()["result"]["final"]["headline"] == "Rice — Rice Blast" and len(world["or"].requests) == 1


# ------------------------------------------------------------------------------------------- 8-10: rotation intact, customers see nothing, no key in logs
def test_8_gemini_rotation_is_unchanged_with_openrouter_configured(client, user_auth, world):
    w = Wire(); use_gemini(w)
    for _ in range(4):
        analyze(client, user_auth)
    assert w.calls == ["K1", "K2", "K1", "K2"] and world["or"].requests == []


FORBIDDEN = re.compile(r"openrouter|gemini|kindwise|plantnet|pl@ntnet|mobilenet|routing|fallback|specialist|disagree|\bml\b|quota|sk-or", re.I)


def test_9_customer_response_has_no_provider_terminology(client, user_auth, world):
    use_gemini(Wire({K1: [503], K2: [503]}))
    _, r = analyze(client, user_auth)
    b = r.json()
    f = b["result"]["final"]
    text = " ".join(str(f.get(k) or "") for k in ("headline", "rejection_reason", "disclaimer", "ai_notes", "plant", "crop", "disease")) + " " + \
        " ".join(x for k in ("symptoms", "immediate_actions", "treatment", "prevention", "warnings", "monitoring") for x in f.get(k, [])) + " " + " ".join(b["result"]["plan"])
    assert not FORBIDDEN.search(text), text
    assert not FORBIDDEN.search(json.dumps({k: v for k, v in b.items() if k != "result"}))     # ai_provider etc. are hidden from customers
    assert not [k for k in ("route", "specialists", "ai", "ai_case") if k in b["result"]]
    assert "openrouter" not in r.text.lower()


def test_10_the_openrouter_key_never_reaches_logs_or_responses(client, user_auth, admin_auth, world, caplog):
    caplog.set_level(logging.DEBUG)
    world["or"].response = httpx.Response(401, json={"error": {"code": 401, "message": "bad key " + ORKEY}})
    use_gemini(Wire({K1: [503], K2: [503]}))
    _, r = analyze(client, user_auth)
    aid, ra = analyze(client, admin_auth)
    assert ORKEY not in caplog.text and ORKEY not in r.text and ORKEY not in ra.text
    assert ORKEY not in client.get(f"{V}/analyses/{aid}", headers=admin_auth).text and K1 not in caplog.text and K2 not in caplog.text
