"""Production-readiness: config guards, CORS, cookies, security headers, error leakage, env documentation."""
import pathlib
import re
from datetime import timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.core.config import Settings
from app.main import create_app
from app.services import analysis_service
from tests.conftest import register_and_verify

V = "/api/v1"
PROD = dict(environment="production", secret_key="k" * 48, cookie_secure=True, cookie_samesite="none", frontend_url="https://app.netlify.app",
            backend_url="https://api.onrender.com", cors_origins="https://app.netlify.app", database_url="postgres://u:p@db.example.com/agro",
            admin_password="Str0ng-unique-pass9", email_backend="smtp", smtp_host="smtp.example.com", gemini_api_key_1="x" * 12, gemini_api_key_2="y" * 12)


def cfg(**over) -> Settings:
    return Settings(_env_file=None, **{**PROD, **over})


# ------------------------------------------------------------------ config guards
def test_a_correct_production_configuration_is_accepted():
    s = cfg()
    assert s.environment == "production" and s.cookie_secure and s.startup_warnings() == []


@pytest.mark.parametrize("over,needle", [
    ({"secret_key": "short"}, "SECRET_KEY"),
    ({"secret_key": ""}, "SECRET_KEY"),
    ({"cookie_secure": False, "cookie_samesite": "lax"}, "COOKIE_SECURE"),
    ({"admin_password": "ChangeMe123!"}, "ADMIN_PASSWORD"),
    ({"admin_password": "password"}, "ADMIN_PASSWORD"),
    ({"frontend_url": "http://app.netlify.app"}, "FRONTEND_URL"),
    ({"backend_url": "http://api.onrender.com"}, "BACKEND_URL"),
    ({"cors_origins": "http://app.netlify.app"}, "CORS_ORIGINS"),
    ({"database_url": "sqlite:///./agroai.db"}, "SQLite"),
])
def test_unsafe_production_settings_refuse_to_start(over, needle):
    with pytest.raises(ValueError) as e:
        cfg(**over)
    assert needle in str(e.value)


@pytest.mark.parametrize("env", ["development", "test", "production"])
def test_wildcard_cors_is_refused_in_every_environment(env):
    with pytest.raises(ValueError, match="explicit origins"):
        cfg(environment=env, cors_origins="*")
    with pytest.raises(ValueError, match="explicit origins"):
        cfg(environment=env, cors_origins="https://ok.example, *")
    with pytest.raises(ValueError, match="explicit origins"):
        cfg(environment=env, cors_origin_regex=".*")


def test_samesite_none_requires_secure_cookies():
    with pytest.raises(ValueError, match="COOKIE_SECURE"):
        Settings(_env_file=None, cookie_samesite="none", cookie_secure=False)


def test_database_url_and_origin_normalisation():
    s = cfg(database_url="postgres://u:p@h:5432/db", cors_origins="https://a.example/ , https://b.example//")
    assert s.database_url == "postgresql+psycopg://u:p@h:5432/db"
    assert Settings(_env_file=None, database_url="postgresql://u:p@h/db").database_url.startswith("postgresql+psycopg://")
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]
    assert Settings(_env_file=None, secret_key="  ").secret_key                        # empty dev secret falls back to the dev default


def test_startup_warnings_flag_risky_but_allowed_production_settings():
    w = cfg(email_backend="console", gemini_api_key_1="", gemini_api_key_2="", cors_origins="https://a.example,https://localhost:3000").startup_warnings()
    joined = " ".join(w)
    assert "EMAIL_BACKEND" in joined and "GEMINI_API_KEY" in joined and "localhost" in joined


def test_dev_otp_is_never_visible_in_production():
    assert cfg(expose_dev_otp=True).dev_otp_visible is False


def test_docs_are_not_served_in_production(monkeypatch, settings):
    monkeypatch.setattr(settings, "environment", "production")
    c = TestClient(create_app())
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_every_setting_is_documented_in_env_example():
    text = (pathlib.Path(__file__).resolve().parents[1] / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, re.M))
    internal = {"APP_NAME", "API_PREFIX"}
    missing = {n.upper() for n in Settings.model_fields} - documented - internal
    assert not missing, f"undocumented settings: {sorted(missing)}"
    assert not re.search(r"(?im)^(SECRET_KEY|GEMINI_API_KEY|ADMIN_PASSWORD|SMTP_PASSWORD|BREVO_API_KEY|GOOGLE_CLIENT_SECRET|GEMINI_API_KEY_[12]|PLANTNET_API_KEY|PLANTIX_API_KEY|KINDWISE_API_KEY|KINDWISE_PLANT_API_KEY)=\S", text), "an example file must not carry values for secrets"


def test_frontend_env_example_documents_every_variable_the_frontend_reads():
    root = pathlib.Path(__file__).resolve().parents[2] / "frontend"
    used = set(re.findall(r"import\.meta\.env\.(VITE_[A-Z_]+)", "".join(p.read_text() for p in (root / "src").rglob("*.js*"))))
    ex = (root / ".env.example").read_text()
    assert used and all(v in ex for v in used)


# ------------------------------------------------------------------ CORS behaviour
def _app_with(settings, monkeypatch, **fields):
    for k, v in fields.items():
        monkeypatch.setattr(settings, k, v)
    return TestClient(create_app())


def preflight(c, origin):
    return c.options(f"{V}/auth/login", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,authorization"})


def test_cors_allows_local_dev_and_the_configured_production_origin(settings, monkeypatch):
    c = _app_with(settings, monkeypatch, cors_origins="http://localhost:5173,https://agroai.netlify.app")
    for origin in ("http://localhost:5173", "https://agroai.netlify.app"):
        r = preflight(c, origin)
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin            # exact echo, never "*"
        assert r.headers["access-control-allow-credentials"] == "true"
        assert "authorization" in r.headers["access-control-allow-headers"].lower()
    simple = c.get(f"{V}/health", headers={"Origin": "https://agroai.netlify.app"})
    assert simple.headers["access-control-allow-origin"] == "https://agroai.netlify.app" and simple.headers.get("vary", "").lower().find("origin") >= 0


@pytest.mark.parametrize("origin", ["https://evil.example", "http://agroai.netlify.app", "https://agroai.netlify.app.evil.example", "null", "https://AGROAI.netlify.app.attacker.io"])
def test_cors_rejects_every_other_origin(settings, monkeypatch, origin):
    c = _app_with(settings, monkeypatch, cors_origins="https://agroai.netlify.app", cors_origin_regex="")
    r = preflight(c, origin)
    assert "access-control-allow-origin" not in r.headers
    assert "access-control-allow-origin" not in c.get(f"{V}/health", headers={"Origin": origin}).headers


def test_cors_regex_supports_netlify_deploy_previews_only(settings, monkeypatch):
    c = _app_with(settings, monkeypatch, cors_origins="https://agroai.netlify.app", cors_origin_regex=r"https://deploy-preview-\d+--agroai\.netlify\.app")
    assert preflight(c, "https://deploy-preview-12--agroai.netlify.app").headers["access-control-allow-origin"] == "https://deploy-preview-12--agroai.netlify.app"
    assert "access-control-allow-origin" not in preflight(c, "https://deploy-preview-12--other.netlify.app").headers


def test_cors_headers_are_present_on_early_rejections_too(settings, monkeypatch):
    c = _app_with(settings, monkeypatch, cors_origins="https://agroai.netlify.app")
    r = c.post(f"{V}/analyses", headers={"Origin": "https://agroai.netlify.app", "Content-Length": str(500 * 1024 * 1024)}, content=b"x")
    assert r.status_code == 413 and r.headers["access-control-allow-origin"] == "https://agroai.netlify.app"


# ------------------------------------------------------------------ cookies, tokens, headers
def set_cookie(r):
    return [h for h in r.headers.get_list("set-cookie") if h.startswith("agro_refresh=")]


def test_refresh_cookie_attributes_in_development(client):
    d = register_and_verify(client)
    c = set_cookie(client.post(f"{V}/auth/login", json={"email": "farmer@example.com", "password": "Farmer123"}))[0].lower()
    assert "httponly" in c and "path=/api/v1/auth" in c and "samesite=lax" in c and "secure" not in c and "max-age=1209600" in c
    assert d["access_token"]


def test_refresh_cookie_is_secure_and_cross_site_capable_when_configured(client, settings, monkeypatch):
    register_and_verify(client)
    monkeypatch.setattr(settings, "cookie_secure", True); monkeypatch.setattr(settings, "cookie_samesite", "none")
    login = client.post(f"{V}/auth/login", json={"email": "farmer@example.com", "password": "Farmer123"})
    c = set_cookie(login)[0].lower()
    assert "httponly" in c and "secure" in c and "samesite=none" in c
    out = set_cookie(client.post(f"{V}/auth/logout"))
    assert out and "max-age=0" in out[0].lower() and "samesite=none" in out[0].lower() and "secure" in out[0].lower()      # logout clears with the same attributes


def test_access_token_lifetime_matches_configuration(client, settings):
    tok = register_and_verify(client)["access_token"]
    p = jwt.decode(tok, options={"verify_signature": False})
    assert p["exp"] - p["iat"] == settings.access_token_minutes * 60 <= 15 * 60 and p["type"] == "access"


def test_logout_invalidates_refresh_and_access_immediately(client):
    d = register_and_verify(client)
    h = {"Authorization": f"Bearer {d['access_token']}"}
    client.post(f"{V}/auth/logout")
    assert client.get(f"{V}/users/me", headers=h).status_code == 401 and client.post(f"{V}/auth/refresh").status_code == 401


def test_security_headers_on_every_response(client):
    for r in (client.get(f"{V}/health"), client.get(f"{V}/nope"), client.post(f"{V}/auth/login", json={})):
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY" and r.headers["referrer-policy"] == "no-referrer"


def test_health_reports_status_without_leaking_configuration(client):
    r = client.get(f"{V}/health")
    b = r.json()
    assert r.status_code == 200 and set(b) == {"status", "app", "environment", "database", "ml_model", "commit"} and b["ml_model"] in ("installed", "missing")
    assert b["commit"] is None or re.fullmatch(r"[0-9a-f]{7}", b["commit"])            # the build id only: no configuration, URL or secret


# ------------------------------------------------------------------ error handling: nothing internal reaches the customer
SECRETS = ["hunter2", "/srv/secret.db", "10.0.0.5", "AIza", "Traceback", "RuntimeError", "sqlalchemy", "OperationalError"]


def _quiet_client():
    return TestClient(create_app(), raise_server_exceptions=False)


def test_unhandled_exception_returns_a_generic_message(monkeypatch, user_auth):
    def boom(*a, **k):
        raise RuntimeError("db password=hunter2 at /srv/secret.db host 10.0.0.5")
    monkeypatch.setattr(analysis_service, "list_for_user", boom)
    r = _quiet_client().get(f"{V}/analyses", headers=user_auth)
    assert r.status_code == 500 and r.json() == {"error": {"code": "server_error", "message": "Something went wrong on our side."}}
    assert not any(s in r.text for s in SECRETS)


def test_database_errors_are_not_exposed(monkeypatch, user_auth):
    def boom(*a, **k):
        raise OperationalError("SELECT * FROM users WHERE password='hunter2'", {}, Exception("connection to server at 10.0.0.5 refused"))
    monkeypatch.setattr(analysis_service, "list_for_user", boom)
    r = _quiet_client().get(f"{V}/analyses", headers=user_auth)
    assert r.status_code == 500 and not any(s in r.text for s in SECRETS)


def test_error_log_keeps_the_detail_server_side(monkeypatch, user_auth, caplog):
    monkeypatch.setattr(analysis_service, "list_for_user", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("internal-detail-xyz")))
    _quiet_client().get(f"{V}/analyses", headers=user_auth)
    assert "internal-detail-xyz" in caplog.text


@pytest.mark.parametrize("method,path,kw", [("get", "/nope", {}), ("put", "/health", {}), ("post", "/auth/login", {"content": "{bad json", "headers": {"content-type": "application/json"}}),
                                            ("get", "/analyses/not-a-uuid", {}), ("post", "/auth/register", {"json": {"email": 5}})])
def test_client_errors_use_the_safe_envelope(client, user_auth, method, path, kw):
    kw.setdefault("headers", {}); kw["headers"] = {**user_auth, **kw["headers"]}
    r = getattr(client, method)(f"{V}{path}", **kw)
    assert 400 <= r.status_code < 500 and set(r.json()) == {"error"} and {"code", "message"} <= set(r.json()["error"])
    assert not any(s in r.text for s in SECRETS)


@pytest.mark.parametrize("status", ["rate_limited", "timeout", "unavailable", "bad_response", "blocked", "misconfigured", "user_limit", "not_configured", "disabled"])
def test_customer_facing_guidance_messages_never_name_providers_or_internals(status):
    from app.ai import base
    msgs = {c.ai_status: c.user_message for c in vars(base).values() if isinstance(c, type) and issubclass(c, base.ProviderError) and c is not base.ProviderError}
    msgs.update(analysis_service.AI_MESSAGES)
    msgs["user_limit"] = "You've reached the limit of 30 guidance requests per hour. Please try again later."
    text = msgs[status].lower()
    assert not any(w in text for w in ("gemini", "google", "provider", "api key", "external", "model's", "server configuration", "http", "401", "403"))
