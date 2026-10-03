import os
import shutil
import tempfile

_tmp = tempfile.mkdtemp(prefix="agroai-test-")
os.environ.update(
    ENVIRONMENT="test",
    DATABASE_URL=f"sqlite:///{_tmp}/test.db",
    UPLOAD_DIR=f"{_tmp}/uploads",
    SECRET_KEY="test-secret-key-test-secret-key-test-secret-key",
    EXPOSE_DEV_OTP="true",
    EMAIL_BACKEND="console",
    ADMIN_EMAIL="admin@test.dev",
    ADMIN_PASSWORD="AdminPass123",
    BCRYPT_ROUNDS="4",
    OTP_RESEND_COOLDOWN_SECONDS="0",
    GOOGLE_CLIENT_ID="",
    GOOGLE_CLIENT_SECRET="",
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.main import app, init_db  # noqa: E402

PASSWORD = "Farmer123"


@pytest.fixture(autouse=True)
def fresh_db():
    shutil.rmtree(get_settings().upload_path, ignore_errors=True)
    get_settings().upload_path.mkdir(parents=True, exist_ok=True)
    Base.metadata.drop_all(engine)
    init_db()
    yield


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def register_and_verify(client, email="farmer@example.com", name="Test Farmer", password=PASSWORD):
    r = client.post("/api/v1/auth/register", json={"email": email, "password": password, "full_name": name})
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/auth/verify-otp", json={"email": email, "code": r.json()["dev_otp"]})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def user_auth(client):
    data = register_and_verify(client)
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture
def admin_auth(client):
    r = client.post("/api/v1/auth/login", json={"email": "admin@test.dev", "password": "AdminPass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _png(size=(64, 64), color=(60, 140, 60)) -> bytes:
    import io

    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", size, color).save(b, "PNG"); return b.getvalue()


# A small but valid, analyzable PNG (uploads smaller than 48px are rejected as too small to analyze)
PNG = _png()
