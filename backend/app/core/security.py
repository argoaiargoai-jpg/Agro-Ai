import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=get_settings().bcrypt_rounds)).decode()


def verify_password(password: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


# Constant-ish dummy hash so unknown-email logins cost the same as real ones.
DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt(rounds=12)).decode()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def hash_otp(user_id: uuid.UUID, code: str) -> str:
    key = get_settings().secret_key.encode()
    return hmac.new(key, f"{user_id}:{code}".encode(), hashlib.sha256).hexdigest()


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def create_access_token(user_id: uuid.UUID, role: str, session_id: uuid.UUID) -> str:
    s = get_settings()
    now = utcnow()
    payload = {
        "sub": str(user_id),
        "role": role,
        "sid": str(session_id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=s.access_token_minutes),
    }
    return jwt.encode(payload, s.secret_key, algorithm="HS256")


def decode_token(token: str, expected_type: str) -> dict:
    payload = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    if payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("wrong token type")
    return payload


def create_state_token(nonce: str | None = None) -> str:
    """Signed, short-lived OAuth `state` (CSRF protection). `nonce` is also kept in an HttpOnly cookie in the user's browser,
    so a state obtained by someone else cannot be replayed into another user's browser (login CSRF)."""
    s = get_settings()
    now = utcnow()
    return jwt.encode(
        {"type": "oauth_state", "n": nonce or secrets.token_urlsafe(16), "iat": now, "exp": now + timedelta(minutes=10)},
        s.secret_key,
        algorithm="HS256",
    )
