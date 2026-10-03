import pytest

from tests.conftest import PASSWORD, register_and_verify

A = "/api/v1/auth"


def reg(client, **over):
    body = {"email": "farmer@example.com", "password": PASSWORD, "full_name": "Test Farmer", **over}
    return client.post(f"{A}/register", json=body)


# ---- registration / validation
def test_register_returns_otp_in_dev(client):
    r = reg(client)
    assert r.status_code == 201
    assert len(r.json()["dev_otp"]) == 6


def test_dev_otp_hidden_when_not_exposed(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "expose_dev_otp", False)
    assert reg(client).json()["dev_otp"] is None


def test_dev_otp_never_exposed_in_production(settings, monkeypatch):
    monkeypatch.setattr(settings, "expose_dev_otp", True)
    monkeypatch.setattr(settings, "environment", "production")
    assert settings.dev_otp_visible is False


@pytest.mark.parametrize(
    "over,field",
    [
        ({"email": "not-an-email"}, "email"),
        ({"password": "short1"}, "password"),
        ({"password": "allletters"}, "password"),
        ({"password": "12345678"}, "password"),
        ({"password": "a1" * 40}, "password"),
        ({"full_name": "x"}, "full_name"),
    ],
)
def test_register_invalid_input(client, over, field):
    r = reg(client, **over)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert field in r.json()["error"]["fields"]


def test_register_missing_body(client):
    r = client.post(f"{A}/register", json={})
    assert r.status_code == 422 and len(r.json()["error"]["fields"]) == 3


def test_register_malformed_json(client):
    r = client.post(f"{A}/register", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_duplicate_verified_email_rejected(client):
    register_and_verify(client)
    r = reg(client)
    assert r.status_code == 409 and r.json()["error"]["code"] == "email_taken"


def test_email_is_case_insensitive(client):
    register_and_verify(client, email="Farmer@Example.com")
    r = client.post(f"{A}/login", json={"email": "FARMER@example.COM", "password": PASSWORD})
    assert r.status_code == 200


def test_reregister_unverified_updates_and_reissues_otp(client):
    first = reg(client).json()["dev_otp"]
    r = reg(client, password="Another123")
    assert r.status_code == 201
    if first != r.json()["dev_otp"]:  # (1-in-a-million collision otherwise)
        assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": first}).status_code == 400
    assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": r.json()["dev_otp"]}).status_code == 200
    assert client.post(f"{A}/login", json={"email": "farmer@example.com", "password": "Another123"}).status_code == 200


def test_registration_can_be_disabled(client, admin_auth):
    client.put("/api/v1/admin/settings", json={"values": {"registration_enabled": False}}, headers=admin_auth)
    r = reg(client)
    assert r.status_code == 403 and r.json()["error"]["code"] == "registration_disabled"


# ---- OTP
def test_verify_otp_success_issues_session_and_cookie(client):
    code = reg(client).json()["dev_otp"]
    r = client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": code})
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["is_verified"] is True and body["access_token"]
    assert "agro_refresh" in r.cookies
    assert "password_hash" not in r.text


def test_otp_wrong_code_counts_attempts_then_locks(client):
    code = reg(client).json()["dev_otp"]
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(5):
        assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": wrong}).status_code == 400
    r = client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": code})
    assert r.status_code == 429 and r.json()["error"]["code"] == "otp_locked"


def test_otp_cannot_be_reused(client):
    code = reg(client).json()["dev_otp"]
    assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": code}).status_code == 200
    assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": code}).status_code == 400


def test_otp_expired(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "otp_ttl_minutes", -1)
    code = reg(client).json()["dev_otp"]
    r = client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": code})
    assert r.status_code == 400 and r.json()["error"]["code"] == "otp_expired"


@pytest.mark.parametrize("code", ["12345", "abcdef", "1234567", ""])
def test_otp_format_validated(client, code):
    reg(client)
    assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": code}).status_code == 422


def test_otp_for_unknown_email(client):
    r = client.post(f"{A}/verify-otp", json={"email": "ghost@example.com", "code": "123456"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "otp_invalid"


def test_resend_invalidates_previous_code(client):
    first = reg(client).json()["dev_otp"]
    second = client.post(f"{A}/resend-otp", json={"email": "farmer@example.com"}).json()["dev_otp"]
    if first != second:
        assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": first}).status_code == 400
    assert client.post(f"{A}/verify-otp", json={"email": "farmer@example.com", "code": second}).status_code == 200


def test_resend_cooldown(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "otp_resend_cooldown_seconds", 60)
    reg(client)
    r = client.post(f"{A}/resend-otp", json={"email": "farmer@example.com"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "otp_cooldown"


def test_resend_does_not_reveal_account_existence(client):
    r = client.post(f"{A}/resend-otp", json={"email": "ghost@example.com"})
    assert r.status_code == 200 and r.json()["dev_otp"] is None


# ---- login / logout
def test_login_unverified_blocked(client):
    reg(client)
    r = client.post(f"{A}/login", json={"email": "farmer@example.com", "password": PASSWORD})
    assert r.status_code == 403 and r.json()["error"]["code"] == "email_not_verified"


def test_login_success_and_me(client):
    register_and_verify(client)
    client.cookies.clear()
    r = client.post(f"{A}/login", json={"email": "farmer@example.com", "password": PASSWORD})
    assert r.status_code == 200
    me = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "farmer@example.com" and me.json()["role"] == "user"


def test_login_wrong_password_and_unknown_email_same_response(client):
    register_and_verify(client)
    a = client.post(f"{A}/login", json={"email": "farmer@example.com", "password": "Wrong12345"})
    b = client.post(f"{A}/login", json={"email": "ghost@example.com", "password": "Wrong12345"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


def test_login_lockout_after_repeated_failures(client):
    register_and_verify(client)
    for _ in range(5):
        client.post(f"{A}/login", json={"email": "farmer@example.com", "password": "Wrong12345"})
    r = client.post(f"{A}/login", json={"email": "farmer@example.com", "password": PASSWORD})
    assert r.status_code == 429 and r.json()["error"]["code"] == "account_locked"


def test_login_invalid_payload(client):
    assert client.post(f"{A}/login", json={"email": "bad", "password": "x"}).status_code == 422
    assert client.post(f"{A}/login", json={"email": "a@b.co"}).status_code == 422


def test_logout_revokes_access_token_and_refresh(client):
    data = register_and_verify(client)
    h = {"Authorization": f"Bearer {data['access_token']}"}
    assert client.get("/api/v1/users/me", headers=h).status_code == 200
    assert client.post(f"{A}/logout").status_code == 200
    assert client.get("/api/v1/users/me", headers=h).status_code == 401
    assert client.post(f"{A}/refresh").status_code == 401


def test_logout_without_session_is_ok(client):
    assert client.post(f"{A}/logout").status_code == 200


# ---- refresh / tokens
def test_refresh_rotates_token(client):
    register_and_verify(client)
    old = client.cookies.get("agro_refresh")
    r = client.post(f"{A}/refresh")
    assert r.status_code == 200 and r.json()["access_token"]
    assert client.cookies.get("agro_refresh") != old


def test_refresh_reuse_after_grace_revokes_everything(client, monkeypatch):
    from app.services import auth_service

    monkeypatch.setattr(auth_service, "REFRESH_REUSE_GRACE_SECONDS", -1)
    register_and_verify(client)
    stolen = client.cookies.get("agro_refresh")
    assert client.post(f"{A}/refresh").status_code == 200  # legit rotation
    rotated = client.cookies.get("agro_refresh")
    client.cookies.set("agro_refresh", stolen, path="/api/v1/auth")
    assert client.post(f"{A}/refresh").status_code == 401  # replay detected
    # the legitimate rotated session is revoked too (theft response)
    client.cookies.set("agro_refresh", rotated, path="/api/v1/auth")
    assert client.post(f"{A}/refresh").status_code == 401


def test_refresh_without_cookie(client):
    r = client.post(f"{A}/refresh")
    assert r.status_code == 401 and r.json()["error"]["code"] == "no_session"


def test_refresh_with_garbage_cookie(client):
    client.cookies.set("agro_refresh", "garbage", path="/api/v1/auth")
    assert client.post(f"{A}/refresh").status_code == 401


# ---- protected routes
@pytest.mark.parametrize("path", ["/users/me", "/analyses", "/analyses/summary", "/admin/stats", "/admin/users"])
def test_protected_routes_require_auth(client, path):
    r = client.get(f"/api/v1{path}")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_garbage_and_tampered_tokens_rejected(client, user_auth):
    assert client.get("/api/v1/users/me", headers={"Authorization": "Bearer abc.def.ghi"}).status_code == 401
    tok = user_auth["Authorization"].split()[1]
    assert client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {tok[:-3]}xyz"}).status_code == 401


def test_expired_token_rejected(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "access_token_minutes", -1)
    data = register_and_verify(client)
    r = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "token_expired"


def test_refresh_token_cannot_be_used_as_access(client):
    register_and_verify(client)
    raw = client.cookies.get("agro_refresh")
    assert client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {raw}"}).status_code == 401


# ---- password flows
def test_forgot_and_reset_password(client):
    register_and_verify(client)
    r = client.post(f"{A}/forgot-password", json={"email": "farmer@example.com"})
    assert r.status_code == 200
    code = r.json()["dev_otp"]
    r = client.post(f"{A}/reset-password", json={"email": "farmer@example.com", "code": code, "new_password": "NewPass456"})
    assert r.status_code == 200
    assert client.post(f"{A}/refresh").status_code == 401  # sessions revoked
    assert client.post(f"{A}/login", json={"email": "farmer@example.com", "password": PASSWORD}).status_code == 401
    assert client.post(f"{A}/login", json={"email": "farmer@example.com", "password": "NewPass456"}).status_code == 200


def test_unverified_account_gets_no_reset_code(client):
    reg(client)
    # unverified accounts get no reset code
    r = client.post(f"{A}/forgot-password", json={"email": "farmer@example.com"})
    assert r.status_code == 200 and r.json()["dev_otp"] is None


def test_forgot_password_unknown_email_generic(client):
    r = client.post(f"{A}/forgot-password", json={"email": "ghost@example.com"})
    assert r.status_code == 200 and r.json()["dev_otp"] is None


def test_reset_password_weak_rejected(client):
    r = client.post(f"{A}/reset-password", json={"email": "a@b.co", "code": "123456", "new_password": "weak"})
    assert r.status_code == 422


def test_change_password(client, user_auth):
    r = client.post(f"{A}/change-password", headers=user_auth, json={"current_password": "Nope12345", "new_password": "Brand123new"})
    assert r.status_code == 400 and "current_password" in r.json()["error"]["fields"]
    r = client.post(f"{A}/change-password", headers=user_auth, json={"current_password": PASSWORD, "new_password": "Brand123new"})
    assert r.status_code == 200
    assert client.get("/api/v1/users/me", headers=user_auth).status_code == 401
    assert client.post(f"{A}/login", json={"email": "farmer@example.com", "password": "Brand123new"}).status_code == 200
