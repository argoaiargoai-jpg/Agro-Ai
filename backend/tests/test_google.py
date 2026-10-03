from urllib.parse import parse_qs, urlparse

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


def fake_google(monkeypatch, profile, token_status=200):
    class Client:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, data=None):
            assert url == auth_service.GOOGLE_TOKEN_URL and data["grant_type"] == "authorization_code"
            return FakeResp(token_status, {"access_token": "gtok"})
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
    security.decode_token(q["state"][0], "oauth_state")


def test_callback_cancelled(client, google_on):
    r = client.get(f"{A}/google/callback?error=access_denied", follow_redirects=False)
    assert r.status_code == 302 and "oauth_error=oauth_cancelled" in r.headers["location"]


def test_callback_bad_state(client, google_on):
    r = client.get(f"{A}/google/callback?code=abc&state=forged", follow_redirects=False)
    assert "oauth_error=oauth_state_invalid" in r.headers["location"]


def test_callback_creates_verified_user_and_session(client, google_on, monkeypatch):
    fake_google(monkeypatch, {"sub": "g-1", "email": "Gina@Gmail.com", "email_verified": True, "name": "Gina G", "picture": "http://x/p.png"})
    r = client.get(f"{A}/google/callback?code=abc&state={security.create_state_token()}", follow_redirects=False)
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
    client.get(f"{A}/google/callback?code=abc&state={security.create_state_token()}", follow_redirects=False)
    r = client.post(f"{A}/refresh")
    assert r.json()["user"]["full_name"] == "Test Farmer"  # linked, not duplicated


def test_callback_rejects_unverified_google_email(client, google_on, monkeypatch):
    fake_google(monkeypatch, {"sub": "g-3", "email": "a@b.co", "email_verified": False})
    r = client.get(f"{A}/google/callback?code=abc&state={security.create_state_token()}", follow_redirects=False)
    assert "oauth_error=oauth_email_unverified" in r.headers["location"]
    assert "agro_refresh" not in r.cookies


def test_callback_google_rejects_code(client, google_on, monkeypatch):
    fake_google(monkeypatch, {}, token_status=400)
    r = client.get(f"{A}/google/callback?code=bad&state={security.create_state_token()}", follow_redirects=False)
    assert "oauth_error=oauth_exchange_failed" in r.headers["location"]


def test_google_only_account_cannot_password_login(client, google_on, monkeypatch):
    fake_google(monkeypatch, {"sub": "g-4", "email": "only@gmail.com", "email_verified": True, "name": "O"})
    client.get(f"{A}/google/callback?code=abc&state={security.create_state_token()}", follow_redirects=False)
    r = client.post(f"{A}/login", json={"email": "only@gmail.com", "password": "Whatever123"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "use_google_signin"


def test_disabled_user_cannot_google_login(client, google_on, monkeypatch, admin_auth):
    fake_google(monkeypatch, {"sub": "g-5", "email": "dis@gmail.com", "email_verified": True, "name": "D"})
    client.get(f"{A}/google/callback?code=abc&state={security.create_state_token()}", follow_redirects=False)
    uid = client.post(f"{A}/refresh").json()["user"]["id"]
    client.cookies.clear()
    client.patch(f"/api/v1/admin/users/{uid}", json={"is_active": False}, headers=admin_auth)
    r = client.get(f"{A}/google/callback?code=abc&state={security.create_state_token()}", follow_redirects=False)
    assert "oauth_error=account_disabled" in r.headers["location"]
