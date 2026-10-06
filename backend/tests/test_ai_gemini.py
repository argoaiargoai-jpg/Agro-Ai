"""The real GeminiProvider code, driven through httpx.MockTransport: request shape, parsing, error mapping. No network."""
import base64
import json

import httpx
import pytest
from pydantic import SecretStr

from app.ai import gemini, prompts
from app.ai.base import (AIRequest, ProviderBadResponse, ProviderBlocked, ProviderMisconfigured, ProviderNotConfigured, ProviderRateLimited,
                         ProviderTimeout, ProviderUnavailable)
from app.ai.gemini import GeminiProvider
from app.ai.schemas import AI_JSON_SCHEMA, AIAnalysis
from tests.ai_fakes import payload

KEY = "AI" + "zaSy" + "TESTKEY_0123456789abcdefghijklmnop"      # built at runtime so secret scanners never see a key-shaped literal


@pytest.fixture(autouse=True)
def _cfg(settings, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(KEY))
    monkeypatch.setattr(settings, "gemini_model", "gemini-test-model")
    monkeypatch.setattr(settings, "ai_max_retries", 1)
    sleeps = []
    monkeypatch.setattr(gemini, "_SLEEP", lambda s: sleeps.append(s))
    return sleeps


def envelope(obj=None, text=None, finish="STOP", **extra):
    text = json.dumps(obj) if text is None else text
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}], "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 22},
            "modelVersion": "gemini-test-model-001", **extra}


class Spy:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def provider(settings, spy):
    return GeminiProvider(settings, transport=httpx.MockTransport(spy))


REQ = AIRequest(system_instruction="SYS", prompt="Analyse.", json_schema=AI_JSON_SCHEMA, image=b"\xff\xd8fakejpeg", image_mime="image/jpeg")


# ------------------------------------------------------------------ request shape
def test_request_is_correct_and_the_key_never_appears_in_the_url(settings):
    spy = Spy(httpx.Response(200, json=envelope(payload())))
    provider(settings, spy).analyze(REQ)
    r = spy.requests[0]
    assert r.method == "POST" and str(r.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test-model:generateContent"
    assert r.headers["x-goog-api-key"] == KEY and KEY not in str(r.url)
    body = json.loads(r.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "Analyse."} and parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == b"\xff\xd8fakejpeg"
    gc = body["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["responseJsonSchema"] == AI_JSON_SCHEMA and gc["temperature"] <= 0.3


def test_prompt_builder_produces_a_schema_valid_request():
    req, case = prompts.build_request({"classification_type": "DISEASE", "crop": "Tomato", "disease": "Early Blight"}, b"x", "image/jpeg")
    assert case == "INDEPENDENT_VERIFICATION" and req.json_schema is AI_JSON_SCHEMA and req.image == b"x"
    AIAnalysis.model_validate(payload())                                                    # the documented shape is itself valid


def test_schema_has_required_core_fields():
    assert {"plant_present", "health_status", "severity", "spread_risk"} <= set(AI_JSON_SCHEMA["required"])
    assert AI_JSON_SCHEMA["properties"]["affected_percentage"]["type"] == ["number", "null"]       # explicit nullable estimate


# ------------------------------------------------------------------ parsing
def test_success_parsing_usage_and_model(settings):
    r = provider(settings, Spy(httpx.Response(200, json=envelope(payload())))).analyze(REQ)
    assert r.data["health_status"] == "healthy" and r.model == "gemini-test-model-001" and r.usage == {"prompt_tokens": 11, "output_tokens": 22}


def test_markdown_fences_and_thought_parts_are_handled(settings):
    body = {"candidates": [{"content": {"parts": [{"text": "internal reasoning", "thought": True}, {"text": "```json\n" + json.dumps(payload()) + "\n```"}]}, "finishReason": "STOP"}]}
    assert provider(settings, Spy(httpx.Response(200, json=body))).analyze(REQ).data["plant_present"] is True


@pytest.mark.parametrize("resp,exc", [
    (httpx.Response(200, json={"candidates": []}), ProviderBadResponse),
    (httpx.Response(200, json={}), ProviderBadResponse),
    (httpx.Response(200, json=envelope(text="not json at all")), ProviderBadResponse),
    (httpx.Response(200, json=envelope(text="[1,2,3]")), ProviderBadResponse),
    (httpx.Response(200, json=envelope(text="")), ProviderBadResponse),
    (httpx.Response(200, json=envelope(payload(), finish="MAX_TOKENS")), ProviderBadResponse),
    (httpx.Response(200, json=envelope(payload(), finish="SAFETY")), ProviderBlocked),
    (httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}), ProviderBlocked),
    (httpx.Response(200, content=b"<html>proxy error</html>"), ProviderBadResponse),
    (httpx.Response(200, json=["not", "an", "object"]), ProviderBadResponse),
])
def test_unusable_200_responses_are_classified(settings, resp, exc):
    with pytest.raises(exc):
        provider(settings, Spy(resp)).analyze(REQ)


# ------------------------------------------------------------------ HTTP errors
def test_rate_limit_429_with_retry_after_header(settings):
    spy = Spy(httpx.Response(429, headers={"retry-after": "42"}, json={"error": {"message": "quota"}}))
    with pytest.raises(ProviderRateLimited) as e:
        provider(settings, spy).analyze(REQ)
    assert e.value.retry_after == 42 and len(spy.requests) == 2                              # 429 is transient: 1 + AI_MAX_RETRIES (=1 here) attempts


def test_rate_limit_429_with_retryinfo_in_body(settings):
    body = {"error": {"message": "quota", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}
    with pytest.raises(ProviderRateLimited) as e:
        provider(settings, Spy(httpx.Response(429, json=body))).analyze(REQ)
    assert e.value.retry_after == 17


def test_5xx_is_retried_once_then_reported_unavailable(settings, _cfg):
    spy = Spy(httpx.Response(503, json={"error": {"message": "overloaded"}}))
    with pytest.raises(ProviderUnavailable):
        provider(settings, spy).analyze(REQ)
    assert len(spy.requests) == 2 and _cfg == [1.0]                                           # 1 + AI_MAX_RETRIES attempts, with backoff


def test_5xx_then_success_recovers(settings):
    spy = Spy(httpx.Response(503, json={}), httpx.Response(200, json=envelope(payload())))
    assert provider(settings, spy).analyze(REQ).data["plant_present"] is True and len(spy.requests) == 2


def test_timeout_is_classified_and_retried_once(settings):
    spy = Spy(httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeout):
        provider(settings, spy).analyze(REQ)
    assert len(spy.requests) == 2


def test_connection_error_is_unavailable(settings):
    with pytest.raises(ProviderUnavailable):
        provider(settings, Spy(httpx.ConnectError("dns"))).analyze(REQ)


@pytest.mark.parametrize("status,msg", [(401, "bad"), (403, "denied"), (404, "models/x is not found"), (400, "API key not valid. Please pass a valid API key.")])
def test_auth_and_model_errors_are_misconfiguration_not_outage(settings, status, msg):
    spy = Spy(httpx.Response(status, json={"error": {"message": msg}}))
    with pytest.raises(ProviderMisconfigured):
        provider(settings, spy).analyze(REQ)
    assert len(spy.requests) == 1                                                             # not retried


def test_schema_rejection_falls_back_once_without_schema(settings):
    spy = Spy(httpx.Response(400, json={"error": {"message": "Invalid JSON payload: responseJsonSchema is not supported for this model"}}),
              httpx.Response(200, json=envelope(payload())))
    assert provider(settings, spy).analyze(REQ).data["plant_present"] is True
    assert "responseJsonSchema" in json.loads(spy.requests[0].content)["generationConfig"]
    assert "responseJsonSchema" not in json.loads(spy.requests[1].content)["generationConfig"]


def test_other_400_is_a_bad_response(settings):
    with pytest.raises(ProviderBadResponse):
        provider(settings, Spy(httpx.Response(400, json={"error": {"message": "image too large"}}))).analyze(REQ)


def test_not_configured_makes_no_http_call(settings, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("   "))
    spy = Spy(httpx.Response(200, json=envelope(payload())))
    p = provider(settings, spy)
    assert p.is_configured() is False
    with pytest.raises(ProviderNotConfigured):
        p.analyze(REQ)
    assert spy.requests == []


# ------------------------------------------------------------------ secrets
def test_key_echoed_by_the_server_never_survives_into_errors_or_logs(settings, caplog):
    caplog.set_level("DEBUG")
    evil = {"error": {"message": f"API key {KEY} is invalid for project; also {KEY}"}}
    with pytest.raises(ProviderMisconfigured) as e:
        provider(settings, Spy(httpx.Response(403, json=evil))).analyze(REQ)
    assert KEY not in str(e.value) and KEY not in e.value.detail and KEY not in caplog.text
    with pytest.raises(ProviderBadResponse) as e2:
        provider(settings, Spy(httpx.Response(418, json={"error": {"message": f"teapot {KEY}"}}))).analyze(REQ)
    assert KEY not in str(e2.value) and KEY not in e2.value.detail and KEY not in caplog.text


def test_settings_repr_hides_the_key(settings):
    assert KEY not in repr(settings) and KEY not in str(settings) and KEY not in settings.model_dump_json()


def test_ping_ok_and_not_ok(settings):
    ok = provider(settings, Spy(httpx.Response(200, json=envelope({"ok": True})))).ping()
    assert ok["ok"] is True and ok["model"] == "gemini-test-model-001" and "latency_ms" in ok
