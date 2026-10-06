"""Two Gemini keys: persistent alternation (atomic DB counter) and one-shot automatic failover. All HTTP is mocked; no key in this file is real."""
import io
import json
import logging
import threading
from datetime import timedelta

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import select

from app.ai import gemini, keyring, registry
from app.ai.base import AIRequest, ProviderBlocked, ProviderError, ProviderMisconfigured, ProviderNotConfigured, ProviderRateLimited, ProviderTimeout, ProviderUnavailable
from app.ai.gemini import GeminiProvider
from app.core.config import Settings
from app.db.session import SessionLocal
from app.models.counter import Counter
from app.services import analysis_service, analysis_workflow, ml_service
from tests import ml_fixture
from tests.ai_fakes import payload

V = "/api/v1"
K1 = "AI" + "zaSy" + "KEY_ONE_0123456789abcdefghijklmnopqrs"
K2 = "AI" + "zaSy" + "KEY_TWO_0123456789abcdefghijklmnopqrs"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(gemini, "_SLEEP", lambda s: None)


def settings(k1=K1, k2=K2, legacy="", **kw) -> Settings:
    return Settings(_env_file=None, environment="test", gemini_api_key_1=k1, gemini_api_key_2=k2, gemini_api_key=legacy, **kw)


class Wire:
    """Mock Gemini endpoint. `plan` maps key -> list of outcomes consumed in order: 'ok', an int HTTP status, 'timeout', 'block'."""

    def __init__(self, plan=None):
        self.calls: list[str] = []                      # which key (as K1/K2 label) each HTTP call used
        self.plan = {K1: ["ok"], K2: ["ok"], **(plan or {})}

    def handler(self, req: httpx.Request) -> httpx.Response:
        key = req.headers["x-goog-api-key"]
        self.calls.append("K1" if key == K1 else "K2" if key == K2 else "?")
        outcomes = self.plan[key]
        o = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if o == "timeout":
            raise httpx.ReadTimeout("slow", request=req)
        if o == "block":
            return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
        if o == "ok":
            ping = "Return exactly" in req.content.decode()
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps({"ok": True} if ping else payload())}]}, "finishReason": "STOP"}]})
        body = {"error": {"message": "API key not valid" if o in (400, 401, 403) else "boom"}}
        return httpx.Response(o, json=body)

    def provider(self, s: Settings) -> GeminiProvider:
        return GeminiProvider(s, transport=httpx.MockTransport(self.handler))


REQ = AIRequest(system_instruction="s", prompt="p", json_schema={"type": "object"}, image=b"\xff\xd8img", image_mime="image/jpeg")


def counter_value() -> int:
    with SessionLocal() as db:
        c = db.scalar(select(Counter).where(Counter.name == keyring.GEMINI_COUNTER))
        return c.value if c else 0


# ------------------------------------------------------------------------------------------------- rotation
def test_requests_alternate_key1_key2_key1_key2():
    w = Wire(); p = w.provider(settings())
    for _ in range(4):
        p.analyze(REQ)
    assert w.calls == ["K1", "K2", "K1", "K2"] and counter_value() == 4


def test_the_sequence_is_persistent_not_per_provider_object():
    w = Wire()
    w.provider(settings()).analyze(REQ)                     # request 1 -> key 1
    w.provider(settings()).analyze(REQ)                     # a NEW provider object (like a restarted worker) continues: request 2 -> key 2
    assert w.calls == ["K1", "K2"]


def test_legacy_single_key_name_is_slot_one():
    s = settings(k1="", k2=K2, legacy=K1)
    assert s.gemini_keys() == [(1, K1), (2, K2)]
    w = Wire(); p = w.provider(s)
    p.analyze(REQ); p.analyze(REQ)
    assert w.calls == ["K1", "K2"]


# ------------------------------------------------------------------------------------------------- failover
@pytest.mark.parametrize("failure", [429, 503, 500, 401, 403, "timeout", 400])
def test_key1_fails_then_key2_answers_the_same_request(failure):
    w = Wire({K1: [failure]}); p = w.provider(settings())
    r = p.analyze(REQ)                                       # request 1: primary = key 1
    assert r.data["plant_present"] is True and w.calls == ["K1", "K2"]
    assert counter_value() == 1                              # the fallback did NOT increment the counter


def test_key2_fails_then_key1_answers():
    w = Wire({K2: [503]}); p = w.provider(settings())
    p.analyze(REQ)                                           # request 1 -> key 1 ok
    r = p.analyze(REQ)                                       # request 2 -> key 2 fails -> key 1
    assert r.data and w.calls == ["K1", "K2", "K1"] and counter_value() == 2


def test_counter_sequence_is_unchanged_by_a_fallback():
    w = Wire({K1: ["ok", 503, "ok"]}); p = w.provider(settings())
    p.analyze(REQ)                                           # #1 -> K1 ok
    p.analyze(REQ)                                           # #2 -> K2 ok
    p.analyze(REQ)                                           # #3 -> K1 fails -> K2
    p.analyze(REQ)                                           # #4 must still be K2 as primary (even)
    p.analyze(REQ)                                           # #5 -> K1
    assert w.calls == ["K1", "K2", "K1", "K2", "K2", "K1"] and counter_value() == 5


def test_both_keys_fail_makes_exactly_two_attempts_and_raises():
    w = Wire({K1: [503], K2: [429]}); p = w.provider(settings())
    with pytest.raises(ProviderError) as e:
        p.analyze(REQ)
    assert w.calls == ["K1", "K2"] and isinstance(e.value, ProviderRateLimited)           # the fallback's error is reported
    assert K1 not in str(e.value) and K2 not in str(e.value)


def test_a_safety_block_is_not_retried_on_the_other_key():
    w = Wire({K1: ["block"]}); p = w.provider(settings())
    with pytest.raises(ProviderBlocked):
        p.analyze(REQ)
    assert w.calls == ["K1"]


def test_transient_failures_are_retried_with_backoff_on_each_key_before_failing_over():
    sleeps = []
    gemini._SLEEP = lambda s: sleeps.append(s)
    w = Wire({K1: ["timeout"], K2: ["timeout"]}); p = w.provider(settings(ai_max_retries=2))
    with pytest.raises(ProviderTimeout):
        p.analyze(REQ)
    assert w.calls == ["K1", "K1", "K1", "K2", "K2", "K2"]                       # bounded: 1 + 2 attempts per key, never more
    assert sleeps == [1.0, 2.0, 1.0, 2.0]                                          # exponential backoff between attempts


@pytest.mark.parametrize("failure", [408, 429, 500, 502, 503, 504, "timeout"])
def test_every_transient_failure_is_retried_then_succeeds(failure):
    w = Wire({K1: [failure, "ok"]}); p = w.provider(settings(ai_max_retries=2))
    assert p.analyze(REQ).data["plant_present"] is True and w.calls == ["K1", "K1"]       # recovered on the SAME key, no failover needed


@pytest.mark.parametrize("failure", [400, 401, 403])
def test_permanent_errors_are_never_retried_on_a_key(failure):
    w = Wire({K1: [failure, "ok"]}); p = w.provider(settings(ai_max_retries=2))
    p.analyze(REQ)
    assert w.calls == ["K1", "K2"]                                                  # straight to the other key


def test_the_retry_budget_stops_further_retries(monkeypatch):
    w = Wire({K1: ["timeout"], K2: ["timeout"]}); p = w.provider(settings(ai_max_retries=5, ai_retry_budget_seconds=0))
    with pytest.raises(ProviderTimeout):
        p.analyze(REQ)
    assert w.calls == ["K1", "K2"]


def test_the_default_timeout_is_40_seconds():
    assert Settings(_env_file=None).ai_timeout_seconds == 40.0


# ------------------------------------------------------------------------------------------------- one or no key
def test_only_key1_configured():
    w = Wire(); s = settings(k2="")
    p = w.provider(s)
    p.analyze(REQ); p.analyze(REQ)
    assert w.calls == ["K1", "K1"] and counter_value() == 0 and s.gemini_configured


def test_only_key2_configured():
    w = Wire(); s = settings(k1="")
    p = w.provider(s)
    p.analyze(REQ)
    assert w.calls == ["K2"] and s.gemini_keys() == [(2, K2)]


def test_a_single_key_keeps_the_old_retry_policy_and_never_calls_a_missing_key():
    w = Wire({K1: [503, "ok"]}); p = w.provider(settings(k2="", ai_max_retries=1))
    assert p.analyze(REQ).data and w.calls == ["K1", "K1"]


def test_both_keys_missing():
    s = settings(k1="", k2="")
    p = GeminiProvider(s)
    assert not p.is_configured() and not s.gemini_configured
    with pytest.raises(ProviderNotConfigured):
        p.analyze(REQ)


def test_one_key_configured_warns_in_production():
    base = dict(environment="production", secret_key="x" * 40, cookie_secure=True, database_url="postgresql://u:p@h/d", frontend_url="https://a.example",
                backend_url="https://b.example", cors_origins="https://a.example", admin_password="Str0ng-unique-pass9", email_backend="smtp", smtp_host="s")
    one = Settings(_env_file=None, gemini_api_key_1="k" * 12, **base).startup_warnings()
    two = Settings(_env_file=None, gemini_api_key_1="k" * 12, gemini_api_key_2="j" * 12, **base).startup_warnings()
    assert any("Only one Gemini key" in w for w in one) and not any("Gemini" in w for w in two)


# ------------------------------------------------------------------------------------------------- atomicity
def test_concurrent_requests_never_receive_the_same_counter_value():
    got, errors = [], []
    barrier = threading.Barrier(24)

    def worker():
        try:
            barrier.wait()
            got.append(keyring.next_number())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=worker) for _ in range(24)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors and sorted(got) == list(range(1, 25)) and counter_value() == 24


def test_concurrent_requests_keep_the_odd_even_split():
    picks, barrier = [], threading.Barrier(20)

    def worker():
        barrier.wait()
        picks.append(keyring.primary_index(2))
    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert picks.count(0) == 10 and picks.count(1) == 10


def test_a_counter_failure_never_blocks_an_analysis(monkeypatch):
    monkeypatch.setattr(keyring, "next_number", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert keyring.primary_index(2) == 0


# ------------------------------------------------------------------------------------------------- secrecy
def test_keys_never_reach_logs_or_errors(caplog):
    caplog.set_level(logging.DEBUG)
    w = Wire({K1: [401], K2: [403]}); p = w.provider(settings())
    with pytest.raises(ProviderMisconfigured) as e:
        p.analyze(REQ)
    assert K1 not in caplog.text and K2 not in caplog.text and K1 not in str(e.value) and K2 not in str(e.value)
    assert "slot 1" in caplog.text and "slot 2" in caplog.text            # the slot is logged, never the key


def test_failover_success_is_logged_without_keys(caplog):
    caplog.set_level(logging.INFO)
    Wire({K1: [503]}).provider(settings()).analyze(REQ)
    assert "fallback key (slot 2) succeeded" in caplog.text and K1 not in caplog.text and K2 not in caplog.text


def test_ping_checks_every_key_and_reports_which_one_is_broken():
    w = Wire({K2: [401]})
    r = w.provider(settings()).ping()
    assert r["ok"] is False and r["keys"] == [{"slot": 1, "ok": True}, {"slot": 2, "ok": False, "state": "misconfigured"}]
    assert K1 not in json.dumps(r) and K2 not in json.dumps(r) and counter_value() == 0


# ------------------------------------------------------------------------------------------------- through the whole backend
@pytest.fixture
def stack(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", str(ml_fixture.build(tmp_path / "ml")))
    monkeypatch.setattr(settings, "gemini_api_key_1", SecretStr(K1))
    monkeypatch.setattr(settings, "gemini_api_key_2", SecretStr(K2))
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    yield
    ml_service.reset_ml_service()
    registry.register("gemini", lambda s: GeminiProvider(s))


def _png(color):
    b = io.BytesIO(); Image.new("RGB", (320, 240), color).save(b, "PNG"); return b.getvalue()


def _analyze(client, headers, color=(255, 0, 0)):
    aid = client.post(f"{V}/analyses", headers=headers, files={"file": ("a.png", _png(color), "image/png")}).json()["id"]
    return aid, client.post(f"{V}/analyses/{aid}/analyze", headers=headers)


def test_full_stack_alternates_keys_per_analysis(client, user_auth, stack):
    w = Wire()
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(w.handler)))
    for _ in range(3):
        assert _analyze(client, user_auth)[1].status_code == 200
    assert w.calls == ["K1", "K2", "K1"]


def test_full_stack_failover_hides_the_failure_from_the_customer(client, user_auth, stack):
    w = Wire({K1: [503]})
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(w.handler)))
    aid, r = _analyze(client, user_auth)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "completed" and body["result"]["final"] and body["result"]["ai_error"] is None
    assert w.calls == ["K1", "K2"] and K1 not in r.text and K2 not in r.text


def test_full_stack_both_keys_fail_keeps_the_ml_result_and_allows_retry(client, user_auth, stack):
    w = Wire({K1: [503], K2: [503]})
    registry.register("gemini", lambda s: GeminiProvider(s, transport=httpx.MockTransport(w.handler)))
    aid, r = _analyze(client, user_auth)
    b = r.json()
    assert b["status"] == "partial" and b["result"]["ml"]["classification_type"] == "DISEASE" and b["result"]["final"] is None
    assert b["result"]["ai_error"]["retryable"] is True and K1 not in r.text and K2 not in r.text
    w.plan[K1], w.plan[K2] = ["ok"], ["ok"]
    r2 = client.post(f"{V}/analyses/{aid}/analyze", headers=user_auth)
    assert r2.json()["status"] == "completed" and r2.json()["result"]["ml"] == b["result"]["ml"]            # ML reused, guidance regenerated
