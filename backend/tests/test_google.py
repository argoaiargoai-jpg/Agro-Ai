import time
from urllib.parse import parse_qs, urlparse

import jwt
import pytest

from app.core import security
from app.services import auth_service

A = "/api/v1/auth"


@pytest.fixture
def google_on(settings, monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "cid.apps.googleusercontent.com")
    monkeypatch.setattr(settings, "google_client_secret", "secret")


class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._d = status, data

    def json(self):
        return self._d


def start(client, nonce="nonce-for-this-browser"):
    """What a real browser has after /google/login: the nonce in its cookie, and the matching signed state in the URL."""
    client.cookies.set("agro_oauth", nonce, path="/api/v1/auth/google")
    return security.create_state_token(nonce)


def id_token(profile, **over):
    claims = {"iss": "https://accounts.google.com", "aud": "cid.apps.googleusercontent.com", "sub": profile.get("sub", "g"),
              "email": profile.get("email"), "exp": int(time.time()) + 600}
    claims.update(over)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, "google-signing-key-not-checked-locally-0123456789", algorithm="HS256")


def fake_google(monkeypatch, profile, token_status=200, idt="auto"):
    class Client:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, data=None):
            assert url == auth_service.GOOGLE_TOKEN_URL and data["grant_type"] == "authorization_code"
            assert data["client_secret"] == "secret" and data["redirect_uri"].endswith("/api/v1/auth/google/callback")
            return FakeResp(token_status, {"access_token": "gtok", "id_token": id_token(profile) if idt == "auto" else idt})
        def get(self, url, headers=None):
            assert headers["Authorization"] == "Bearer gtok"
            return FakeResp(200, profile)

    monkeypatch.setattr(auth_service.httpx, "Client", Client)


def test_status_disabled_by_default(client):
    assert client.get(f"{A}/google/status").json() == {"enabled": False}


def test_login_unconfigured_returns_clear_error(client):
    r = client.get(f"{A}/google/login", follow_redirects=False)
    assert r.status_code == 503 and r.json()["error"]["code"] == "google_not_configured"


def test_login_redirects_to_google_with_state(client, google_on):
    r = client.get(f"{A}/google/login", follow_redirects=False)
    assert r.status_code == 302
    u = urlparse(r.headers["location"])
    q = parse_qs(u.query)
    assert u.netloc == "accounts.google.com"
    assert q["client_id"] == ["cid.apps.googleusercontent.com"]
    assert q["redirect_uri"] == ["http://localhost:8000/api/v1/auth/google/callback"]
    assert q["scope"] == ["openid email profile"]
    state = security.decode_token(q["state"][0], "oauth_state")
    cookie = r.headers["set-cookie"]
    assert f"agro_oauth={state['n']}" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/api/v1/auth/google" in cookie


def test_callback_cancelled(client, google_on):
    r = client.get(f"{A}/google/callback?error=access_denied", follow_redirects=False)
    assert r.status_code == 302 and "oauth_error=oauth_cancelled" in r.headers["location"]


def test_callback_bad_state(client, google_on):
    r = client.get(f"{A}/google/callback?code=abc&state=forged", follow_redirects=False)
    assert "oauth_error=oauth_state_invalid" in r.headers["location"]


def test_callback_creates_verified_user_and_session(client, google_on, monkeypatch):
    fake_google(monkeypatch, {"sub": "g-1", "email": "Gina@Gmail.com", "email_verified": True, "name": "Gina G", "picture": "http://x/p.png"})
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].endswith("/auth/google/done")
    assert "agro_refresh" in r.cookies
    me = client.post(f"{A}/refresh")
    assert me.status_code == 200
    u = me.json()["user"]
    assert u["email"] == "gina@gmail.com" and u["is_verified"] and u["auth_provider"] == "google" and not u["has_password"]


def test_callback_links_existing_password_account(client, google_on, monkeypatch):
    from tests.conftest import register_and_verify

    register_and_verify(client, email="farmer@example.com")
    client.cookies.clear()
    fake_google(monkeypatch, {"sub": "g-2", "email": "farmer@example.com", "email_verified": True, "name": "X"})
    client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    r = client.post(f"{A}/refresh")
    assert r.json()["user"]["full_name"] == "Test Farmer"  # linked, not duplicated


def test_callback_rejects_unverified_google_email(client, google_on, monkeypatch):
    fake_google(monkeypatch, {"sub": "g-3", "email": "a@b.co", "email_verified": False})
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert "oauth_error=oauth_email_unverified" in r.headers["location"]
    assert "agro_refresh" not in r.cookies


def test_callback_google_rejects_code(client, google_on, monkeypatch):
    fake_google(monkeypatch, {}, token_status=400)
    r = client.get(f"{A}/google/callback?code=bad&state={start(client)}", follow_redirects=False)
    assert "oauth_error=oauth_exchange_failed" in r.headers["location"]


def test_google_only_account_cannot_password_login(client, google_on, monkeypatch):
    fake_google(monkeypatch, {"sub": "g-4", "email": "only@gmail.com", "email_verified": True, "name": "O"})
    client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    r = client.post(f"{A}/login", json={"email": "only@gmail.com", "password": "Whatever123"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "use_google_signin"


def test_disabled_user_cannot_google_login(client, google_on, monkeypatch, admin_auth):
    fake_google(monkeypatch, {"sub": "g-5", "email": "dis@gmail.com", "email_verified": True, "name": "D"})
    client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    uid = client.post(f"{A}/refresh").json()["user"]["id"]
    client.cookies.clear()
    client.patch(f"/api/v1/admin/users/{uid}", json={"is_active": False}, headers=admin_auth)
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert "oauth_error=account_disabled" in r.headers["location"]


# ----------------------------------------------------------- hardening: browser-bound state, ID token claims, safe linking
GOOD = {"sub": "g-9", "email": "new@gmail.com", "email_verified": True, "name": "N"}


def test_callback_without_the_browser_cookie_is_rejected(client, google_on, monkeypatch):
    fake_google(monkeypatch, GOOD)
    state = security.create_state_token("someone-elses-nonce")             # e.g. a link an attacker prepared for the victim
    r = client.get(f"{A}/google/callback?code=abc&state={state}", follow_redirects=False)
    assert "oauth_error=oauth_state_invalid" in r.headers["location"] and "agro_refresh" not in r.cookies


def test_callback_with_a_different_browsers_cookie_is_rejected(client, google_on, monkeypatch):
    fake_google(monkeypatch, GOOD)
    state = security.create_state_token("attacker-nonce")
    client.cookies.set("agro_oauth", "victim-nonce", path="/api/v1/auth/google")
    r = client.get(f"{A}/google/callback?code=abc&state={state}", follow_redirects=False)
    assert "oauth_error=oauth_state_invalid" in r.headers["location"] and "agro_refresh" not in r.cookies


def test_state_cookie_is_single_use(client, google_on, monkeypatch):
    fake_google(monkeypatch, GOOD)
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert r.headers["location"].endswith("/auth/google/done")
    assert any(h.startswith("agro_oauth=") and ("Max-Age=0" in h or "expires" in h.lower()) for h in r.headers.get_list("set-cookie"))


@pytest.mark.parametrize("name,over", [
    ("wrong audience", {"aud": "someone-elses-client.apps.googleusercontent.com"}),
    ("wrong issuer", {"iss": "https://evil.example"}),
    ("expired", {"exp": 1}),
    ("no subject", {"sub": None}),
    ("subject differs from userinfo", {"sub": "other-subject"}),
    ("email differs from userinfo", {"email": "someone.else@gmail.com"}),
])
def test_callback_rejects_bad_id_token_claims(client, google_on, monkeypatch, name, over):
    fake_google(monkeypatch, GOOD, idt=id_token(GOOD, **over))
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert "oauth_error=oauth_exchange_failed" in r.headers["location"], name
    assert "agro_refresh" not in r.cookies


def test_callback_rejects_missing_id_token(client, google_on, monkeypatch):
    fake_google(monkeypatch, GOOD, idt=None)
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert "oauth_error=oauth_exchange_failed" in r.headers["location"]


def test_unexpected_google_failure_ends_on_our_login_page(client, google_on, monkeypatch):
    class Boom:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k): return FakeResp(200, {})          # no access_token / id_token at all
    monkeypatch.setattr(auth_service.httpx, "Client", Boom)
    r = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert r.status_code == 302 and "/login?oauth_error=" in r.headers["location"]


def test_preregistered_unverified_account_cannot_be_taken_over_by_its_password(client, google_on, monkeypatch):
    """An attacker registers victim@gmail.com with their own password but can't verify the mailbox. When the real owner
    signs in with Google, the attacker's password must stop working and no duplicate account may appear."""
    from sqlalchemy import func, select

    from app.db.session import SessionLocal
    from app.models import User

    assert client.post(f"{A}/register", json={"email": "victim@gmail.com", "password": "Attacker123", "full_name": "Attacker"}).status_code == 201
    client.cookies.clear()
    fake_google(monkeypatch, {"sub": "g-v", "email": "victim@gmail.com", "email_verified": True, "name": "Real Owner"})
    client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    r = client.post(f"{A}/login", json={"email": "victim@gmail.com", "password": "Attacker123"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "use_google_signin"
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(User).where(User.email == "victim@gmail.com")) == 1
        u = db.scalar(select(User).where(User.email == "victim@gmail.com"))
        assert u.is_verified and u.google_sub == "g-v" and not u.password_hash


def test_verified_password_account_keeps_its_password_after_linking(client, google_on, monkeypatch):
    from tests.conftest import PASSWORD, register_and_verify

    register_and_verify(client, email="both@example.com")
    client.cookies.clear()
    fake_google(monkeypatch, {"sub": "g-b", "email": "both@example.com", "email_verified": True, "name": "B"})
    client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    client.cookies.clear()
    assert client.post(f"{A}/login", json={"email": "both@example.com", "password": PASSWORD}).status_code == 200


def test_google_secret_never_appears_in_redirects_or_responses(client, google_on, monkeypatch):
    fake_google(monkeypatch, GOOD)
    r1 = client.get(f"{A}/google/login", follow_redirects=False)
    r2 = client.get(f"{A}/google/callback?code=abc&state={start(client)}", follow_redirects=False)
    assert "secret" not in (r1.headers["location"] + r2.headers["location"]).lower().replace("client_secret_", "")
    assert "client_secret" not in r1.text + r2.text + client.get("/api/v1/config/public").text
