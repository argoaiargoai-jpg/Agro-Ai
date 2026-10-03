def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["database"] == "ok"


def test_public_config(client):
    r = client.get("/api/v1/config/public")
    assert r.status_code == 200
    body = r.json()
    assert body["registration_enabled"] is True and body["google_enabled"] is False
    assert "Tomato" in body["supported_crops"]


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_cors_allows_frontend_origin(client):
    r = client.options(
        "/api/v1/auth/login",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert r.headers["access-control-allow-credentials"] == "true"


def test_cors_blocks_unknown_origin(client):
    r = client.options(
        "/api/v1/auth/login",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


def test_ensure_admin_rejects_invalid_config(settings, monkeypatch, caplog):
    from app.db.session import SessionLocal
    from app.services import auth_service

    monkeypatch.setattr(settings, "admin_email", "admin@agroai.local")  # special-use TLD: can never log in
    with SessionLocal() as db:
        auth_service.ensure_admin(db)
        assert auth_service.get_user_by_email(db, "admin@agroai.local") is None
    assert "bootstrap admin NOT created" in caplog.text
