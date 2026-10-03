"""Email delivery backends. No real network call is ever made: httpx.post / smtplib are replaced."""
import smtplib

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import AppError
from app.services import email_service

KEY = "xkeysib-" + "unit-test-key-0000"      # built at runtime: not a real key


def _use(monkeypatch, **kw):
    s = Settings(_env_file=None, environment="test", **kw)
    monkeypatch.setattr(email_service, "get_settings", lambda: s)
    return s


class Recorder:
    def __init__(self, response=None, exc=None):
        self.calls, self.response, self.exc = [], response, exc

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.response


def _resp(status, body=None):
    return httpx.Response(status, json=body if body is not None else {}, request=httpx.Request("POST", "https://x"))


@pytest.fixture
def brevo(monkeypatch):
    _use(monkeypatch, email_backend="brevo", brevo_api_key=KEY, email_from="AGRO AI <noreply@agro.example>")


def test_brevo_sends_the_expected_https_request(monkeypatch, brevo):
    rec = Recorder(_resp(201, {"messageId": "<1@x>"}))
    monkeypatch.setattr(email_service.httpx, "post", rec)
    email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    (url, kw), = rec.calls
    assert url == "https://api.brevo.com/v3/smtp/email"
    assert kw["headers"]["api-key"] == KEY
    assert kw["timeout"] == 10
    body = kw["json"]
    assert body["sender"] == {"name": "AGRO AI", "email": "noreply@agro.example"}
    assert body["to"] == [{"email": "farmer@example.com"}]
    assert body["subject"] == "Verify your AGRO AI email"
    assert "123456" in body["textContent"] and "10 minutes" in body["textContent"]


def test_brevo_uses_the_reset_subject_and_a_bare_sender_address(monkeypatch):
    _use(monkeypatch, email_backend="brevo", brevo_api_key=KEY, email_from="noreply@agro.example")
    rec = Recorder(_resp(201))
    monkeypatch.setattr(email_service.httpx, "post", rec)
    email_service.send_otp_email("a@example.com", "654321", "reset_password")
    body = rec.calls[0][1]["json"]
    assert body["subject"] == "Reset your AGRO AI password" and body["sender"] == {"email": "noreply@agro.example"}


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_brevo_non_2xx_becomes_a_clean_503_without_leaking(monkeypatch, brevo, caplog, status):
    monkeypatch.setattr(email_service.httpx, "post", Recorder(_resp(status, {"code": "unauthorized", "message": "Key " + KEY + " invalid"})))
    with pytest.raises(AppError) as e:
        email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert e.value.status_code == 503 and e.value.code == "email_unavailable"
    assert KEY not in e.value.message and KEY not in caplog.text
    assert "farmer@example.com" not in caplog.text and "123456" not in caplog.text
    assert f"status={status}" in caplog.text


def test_brevo_non_json_error_body_is_handled(monkeypatch, brevo):
    bad = httpx.Response(502, text="<html>bad gateway</html>", request=httpx.Request("POST", "https://x"))
    monkeypatch.setattr(email_service.httpx, "post", Recorder(bad))
    with pytest.raises(AppError) as e:
        email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert e.value.code == "email_unavailable"


@pytest.mark.parametrize("exc", [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.ConnectError("c")])
def test_brevo_network_errors_and_timeouts_become_503(monkeypatch, brevo, exc):
    monkeypatch.setattr(email_service.httpx, "post", Recorder(exc=exc))
    with pytest.raises(AppError) as e:
        email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert e.value.status_code == 503 and e.value.code == "email_unavailable"


def test_brevo_without_a_key_is_not_configured_and_makes_no_request(monkeypatch):
    _use(monkeypatch, email_backend="brevo", brevo_api_key="", email_from="AGRO AI <noreply@agro.example>")
    rec = Recorder(_resp(201))
    monkeypatch.setattr(email_service.httpx, "post", rec)
    with pytest.raises(AppError) as e:
        email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert e.value.code == "email_unavailable" and not rec.calls


def test_brevo_key_is_a_secret_and_never_in_repr():
    s = Settings(_env_file=None, brevo_api_key=KEY)
    assert KEY not in repr(s) and KEY not in str(s.model_dump())


def test_console_backend_never_touches_the_network(monkeypatch):
    _use(monkeypatch, email_backend="console")
    rec = Recorder(_resp(201))
    monkeypatch.setattr(email_service.httpx, "post", rec)
    email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert not rec.calls


def test_smtp_backend_still_works_and_does_not_call_brevo(monkeypatch):
    _use(monkeypatch, email_backend="smtp", smtp_host="smtp.example.com", smtp_user="u", smtp_password="p")
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout): sent.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): sent.append("starttls")
        def login(self, u, p): sent.append("login")
        def send_message(self, m): sent.append(("msg", m["To"], m["Subject"]))

    rec = Recorder(_resp(201))
    monkeypatch.setattr(email_service.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_service.httpx, "post", rec)
    email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert sent == [("connect", "smtp.example.com", 587), "starttls", "login", ("msg", "farmer@example.com", "Verify your AGRO AI email")]
    assert not rec.calls


def test_smtp_failure_still_maps_to_503(monkeypatch):
    _use(monkeypatch, email_backend="smtp", smtp_host="smtp.example.com")

    def boom(*a, **k):
        raise smtplib.SMTPException("x")
    monkeypatch.setattr(email_service.smtplib, "SMTP", boom)
    with pytest.raises(AppError) as e:
        email_service.send_otp_email("farmer@example.com", "123456", "verify_email")
    assert e.value.code == "email_unavailable"


def test_startup_warns_when_brevo_is_selected_without_a_key():
    base = dict(environment="production", secret_key="x" * 40, cookie_secure=True, database_url="postgresql://u:p@h/d",
                frontend_url="https://a.example", backend_url="https://b.example", cors_origins="https://a.example",
                email_backend="brevo", gemini_api_key="k", admin_password="Str0ng-unique-pass9")
    assert any("BREVO_API_KEY" in w for w in Settings(_env_file=None, **base).startup_warnings())
    assert not any("BREVO_API_KEY" in w for w in Settings(_env_file=None, brevo_api_key=KEY, **base).startup_warnings())


def test_registration_flow_uses_brevo_end_to_end(monkeypatch, client, settings):
    """Real register -> OTP -> verify through the API with the Brevo backend (HTTP mocked)."""
    monkeypatch.setattr(settings, "email_backend", "brevo")
    monkeypatch.setattr(settings, "brevo_api_key", SecretStr(KEY))
    monkeypatch.setattr(settings, "expose_dev_otp", False)
    rec = Recorder(_resp(201))
    monkeypatch.setattr(email_service.httpx, "post", rec)
    r = client.post("/api/v1/auth/register", json={"email": "new@example.com", "password": "Farmer123", "full_name": "New Farmer"})
    assert r.status_code == 201 and r.json()["dev_otp"] is None
    words = [w.rstrip(".") for w in rec.calls[0][1]["json"]["textContent"].split()]
    code = next(w for w in words if w.isdigit() and len(w) == 6)
    assert client.post("/api/v1/auth/verify-otp", json={"email": "new@example.com", "code": code}).status_code == 200


def test_registration_returns_503_when_brevo_rejects(monkeypatch, client, settings):
    monkeypatch.setattr(settings, "email_backend", "brevo")
    monkeypatch.setattr(settings, "brevo_api_key", SecretStr(KEY))
    monkeypatch.setattr(email_service.httpx, "post", Recorder(_resp(401, {"code": "unauthorized"})))
    r = client.post("/api/v1/auth/register", json={"email": "new@example.com", "password": "Farmer123", "full_name": "New Farmer"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "email_unavailable" and KEY not in r.text
