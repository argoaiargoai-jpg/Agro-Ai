import hmac
import logging
import secrets
import uuid
from datetime import timedelta
from urllib.parse import urlencode

import httpx
import jwt
from email_validator import EmailNotValidError, validate_email
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core import security
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.security import utcnow
from app.models import AuthSession, EmailOTP, Role, User
from app.models.user import DEFAULT_PREFERENCES
from app.schemas.auth import validate_password_strength
from app.services import email_service, settings_service

log = logging.getLogger("agroai.auth")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
REFRESH_REUSE_GRACE_SECONDS = 10


# ---------------------------------------------------------------- helpers
def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


def _invalid_credentials() -> AppError:
    return AppError(401, "invalid_credentials", "Incorrect email or password.")


# ------------------------------------------------------------------- OTP
def issue_otp(db: Session, user: User, purpose: str) -> str:
    """Create a fresh OTP (invalidating older ones) and email it. Returns the code."""
    s = get_settings()
    now = utcnow()
    latest = db.scalar(
        select(EmailOTP)
        .where(EmailOTP.user_id == user.id, EmailOTP.purpose == purpose)
        .order_by(EmailOTP.created_at.desc())
    )
    if latest and (now - latest.created_at).total_seconds() < s.otp_resend_cooldown_seconds:
        wait = int(s.otp_resend_cooldown_seconds - (now - latest.created_at).total_seconds()) + 1
        raise AppError(429, "otp_cooldown", f"Please wait {wait}s before requesting another code.")
    db.execute(
        update(EmailOTP)
        .where(EmailOTP.user_id == user.id, EmailOTP.purpose == purpose, EmailOTP.consumed_at.is_(None))
        .values(consumed_at=now)
    )
    code = security.generate_otp()
    db.add(
        EmailOTP(
            user_id=user.id,
            purpose=purpose,
            code_hash=security.hash_otp(user.id, code),
            expires_at=now + timedelta(minutes=s.otp_ttl_minutes),
        )
    )
    db.commit()
    email_service.send_otp_email(user.email, code, purpose)
    return code


def check_otp(db: Session, user: User, purpose: str, code: str) -> None:
    s = get_settings()
    now = utcnow()
    otp = db.scalar(
        select(EmailOTP)
        .where(EmailOTP.user_id == user.id, EmailOTP.purpose == purpose, EmailOTP.consumed_at.is_(None))
        .order_by(EmailOTP.created_at.desc())
    )
    if otp is None:
        raise AppError(400, "otp_invalid", "That code is invalid or has expired. Request a new one.")
    if otp.expires_at < now:
        raise AppError(400, "otp_expired", "That code has expired. Request a new one.")
    if otp.attempts >= s.otp_max_attempts:
        raise AppError(429, "otp_locked", "Too many incorrect attempts. Request a new code.")
    if not hmac.compare_digest(otp.code_hash, security.hash_otp(user.id, code)):
        otp.attempts += 1
        db.commit()
        left = s.otp_max_attempts - otp.attempts
        raise AppError(400, "otp_invalid", f"Incorrect code. {max(left, 0)} attempt(s) left.")
    otp.consumed_at = now
    db.commit()


def otp_response(user: User, code: str, message: str) -> dict:
    s = get_settings()
    return {
        "message": message,
        "email": user.email,
        "resend_in_seconds": s.otp_resend_cooldown_seconds,
        "dev_otp": code if s.dev_otp_visible else None,
    }


# --------------------------------------------------------- registration
def register(db: Session, email: str, password: str, full_name: str) -> dict:
    if not settings_service.get_value(db, "registration_enabled"):
        raise AppError(403, "registration_disabled", "New registrations are currently closed.")
    user = get_user_by_email(db, email)
    if user and user.is_verified:
        raise AppError(409, "email_taken", "An account with this email already exists. Try signing in.")
    if user is None:
        user = User(email=email.lower(), full_name=full_name, preferences=dict(DEFAULT_PREFERENCES))
        db.add(user)
    user.full_name = full_name
    user.password_hash = security.hash_password(password)
    db.commit()
    code = issue_otp(db, user, "verify_email")
    return otp_response(user, code, "Account created. We sent a 6-digit code to your email.")


def resend_verification(db: Session, email: str) -> dict:
    user = get_user_by_email(db, email)
    if user is None or user.is_verified:
        # Don't reveal whether an account exists.
        return {"message": "If that account needs verification, a new code is on its way.",
                "email": email, "resend_in_seconds": get_settings().otp_resend_cooldown_seconds, "dev_otp": None}
    code = issue_otp(db, user, "verify_email")
    return otp_response(user, code, "A new code has been sent.")


def verify_email(db: Session, email: str, code: str) -> User:
    user = get_user_by_email(db, email)
    if user is None:
        raise AppError(400, "otp_invalid", "That code is invalid or has expired. Request a new one.")
    if not user.is_active:
        raise AppError(403, "account_disabled", "This account has been disabled.")
    check_otp(db, user, "verify_email", code)
    user.is_verified = True
    db.commit()
    return user


# ------------------------------------------------------------ login
def authenticate(db: Session, email: str, password: str) -> User:
    s = get_settings()
    user = get_user_by_email(db, email)
    if user is None:
        security.verify_password(password, security.DUMMY_HASH)
        raise _invalid_credentials()
    now = utcnow()
    if user.locked_until and user.locked_until > now:
        mins = int((user.locked_until - now).total_seconds() // 60) + 1
        raise AppError(429, "account_locked", f"Too many failed attempts. Try again in {mins} minute(s).")
    if not user.has_password:
        raise AppError(400, "use_google_signin", "This account uses Google sign-in. Continue with Google.")
    if not security.verify_password(password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= s.max_failed_logins:
            user.locked_until = now + timedelta(minutes=s.lockout_minutes)
            user.failed_login_count = 0
        db.commit()
        raise _invalid_credentials()
    user.failed_login_count = 0
    user.locked_until = None
    db.commit()
    if not user.is_active:
        raise AppError(403, "account_disabled", "This account has been disabled. Contact support.")
    if not user.is_verified:
        raise AppError(403, "email_not_verified", "Please verify your email to continue.")
    return user


# --------------------------------------------------------- sessions
def create_session(db: Session, user: User, user_agent: str | None, ip: str | None) -> tuple[str, str]:
    s = get_settings()
    raw = security.new_refresh_token()
    sess = AuthSession(
        user_id=user.id,
        token_hash=security.sha256(raw),
        user_agent=(user_agent or "")[:300] or None,
        ip_address=ip,
        expires_at=utcnow() + timedelta(days=s.refresh_token_days),
    )
    db.add(sess)
    user.last_login_at = utcnow()
    db.commit()
    return security.create_access_token(user.id, user.role, sess.id), raw


def revoke_all_sessions(db: Session, user_id: uuid.UUID) -> None:
    db.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    db.commit()


def rotate_session(db: Session, raw_token: str | None, user_agent: str | None, ip: str | None) -> tuple[str, str, User]:
    if not raw_token:
        raise AppError(401, "no_session", "You are signed out.")
    sess = db.scalar(select(AuthSession).where(AuthSession.token_hash == security.sha256(raw_token)))
    now = utcnow()
    if sess is None or sess.expires_at < now:
        raise AppError(401, "session_expired", "Your session has expired. Please sign in again.")
    if sess.revoked_at is not None:
        # A rotated token was replayed: either a harmless race or theft.
        if (now - sess.revoked_at).total_seconds() > REFRESH_REUSE_GRACE_SECONDS:
            log.warning("Refresh token reuse detected for user %s; revoking all sessions", sess.user_id)
            revoke_all_sessions(db, sess.user_id)
        raise AppError(401, "session_expired", "Your session has expired. Please sign in again.")
    user = db.get(User, sess.user_id)
    if user is None or not user.is_active or not user.is_verified:
        raise AppError(401, "session_expired", "Your session has expired. Please sign in again.")
    sess.revoked_at = now
    db.commit()
    access, refresh = create_session(db, user, user_agent, ip)
    return access, refresh, user


def revoke_session_by_token(db: Session, raw_token: str | None) -> None:
    if not raw_token:
        return
    sess = db.scalar(select(AuthSession).where(AuthSession.token_hash == security.sha256(raw_token)))
    if sess and sess.revoked_at is None:
        sess.revoked_at = utcnow()
        db.commit()


# --------------------------------------------------- password reset/change
def request_password_reset(db: Session, email: str) -> dict:
    user = get_user_by_email(db, email)
    generic = {"message": "If an account exists for that email, we sent a reset code.",
               "email": email, "resend_in_seconds": get_settings().otp_resend_cooldown_seconds, "dev_otp": None}
    if user is None or not user.is_active or not user.is_verified:
        return generic
    try:
        code = issue_otp(db, user, "reset_password")
    except AppError as e:
        if e.code == "otp_cooldown":
            return generic
        raise
    out = otp_response(user, code, generic["message"])
    return out


def reset_password(db: Session, email: str, code: str, new_password: str) -> None:
    user = get_user_by_email(db, email)
    if user is None or not user.is_active:
        raise AppError(400, "otp_invalid", "That code is invalid or has expired. Request a new one.")
    check_otp(db, user, "reset_password", code)
    user.password_hash = security.hash_password(new_password)
    user.failed_login_count, user.locked_until = 0, None
    db.commit()
    revoke_all_sessions(db, user.id)


def change_password(db: Session, user: User, current: str | None, new: str) -> None:
    if user.has_password:
        if not current or not security.verify_password(current, user.password_hash):
            raise AppError(400, "invalid_credentials", "Current password is incorrect.", {"current_password": "Incorrect password."})
    user.password_hash = security.hash_password(new)
    db.commit()
    revoke_all_sessions(db, user.id)


# ------------------------------------------------------------ Google
def google_authorization_url() -> tuple[str, str]:
    """Returns (url, nonce). The nonce must be stored in a browser cookie and presented again at the callback."""
    s = get_settings()
    if not s.google_enabled:
        raise AppError(503, "google_not_configured", "Google sign-in is not configured on this server.")
    nonce = secrets.token_urlsafe(16)
    params = {
        "client_id": s.google_client_id,
        "redirect_uri": s.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": security.create_state_token(nonce),
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}", nonce


def _check_id_token(id_token: str | None, client_id: str) -> dict:
    """Claims of the ID token Google returned from the token endpoint (received directly over TLS with our client secret, so the signature
    check is skipped as Google documents). We still require: issuer, audience == our client id, not expired, a subject."""
    if not id_token:
        raise AppError(400, "oauth_exchange_failed", "Google did not return an identity.")
    try:
        claims = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False, "verify_exp": False})
    except jwt.PyJWTError as exc:
        raise AppError(400, "oauth_exchange_failed", "Google returned an unreadable identity.") from exc
    aud = claims.get("aud")
    if claims.get("iss") not in GOOGLE_ISSUERS or client_id not in ([aud] if isinstance(aud, str) else aud or []) \
            or not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < utcnow().timestamp() or not claims.get("sub"):
        log.warning("Google ID token rejected (issuer/audience/expiry mismatch)")
        raise AppError(400, "oauth_exchange_failed", "Google returned an invalid identity.")
    return claims


def google_complete(db: Session, code: str, state: str, nonce: str | None) -> User:
    s = get_settings()
    if not s.google_enabled:
        raise AppError(503, "google_not_configured", "Google sign-in is not configured on this server.")
    try:
        payload = security.decode_token(state, "oauth_state")
    except jwt.PyJWTError as exc:
        raise AppError(400, "oauth_state_invalid", "Sign-in expired or was tampered with. Please try again.") from exc
    if not nonce or not hmac.compare_digest(str(payload.get("n", "")), nonce):         # state must belong to THIS browser
        log.warning("Google callback rejected: state does not match the browser that started the sign-in")
        raise AppError(400, "oauth_state_invalid", "Sign-in expired or was tampered with. Please try again.")
    try:
        with httpx.Client(timeout=10) as client:
            tok = client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": s.google_client_id,
                    "client_secret": s.google_client_secret,
                    "redirect_uri": s.google_redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            if tok.status_code != 200:
                try:
                    reason = tok.json().get("error")           # e.g. redirect_uri_mismatch / invalid_grant / invalid_client: server log only
                except ValueError:
                    reason = None
                log.warning("Google token exchange failed: HTTP %s %s", tok.status_code, reason)
                raise AppError(400, "oauth_exchange_failed", "Google rejected the sign-in request.")
            tok_json = tok.json()
            claims = _check_id_token(tok_json.get("id_token"), s.google_client_id)
            info = client.get(
                GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {tok_json['access_token']}"}
            )
            if info.status_code != 200:
                raise AppError(400, "oauth_exchange_failed", "Could not read your Google profile.")
            profile = info.json()
    except httpx.HTTPError as exc:
        raise AppError(502, "oauth_unreachable", "Could not reach Google. Try again.") from exc

    email = (profile.get("email") or "").lower()
    if not email or not profile.get("email_verified"):
        raise AppError(400, "oauth_email_unverified", "Your Google email address is not verified.")
    sub = str(profile["sub"])
    if sub != str(claims["sub"]) or (claims.get("email") and str(claims["email"]).lower() != email):
        log.warning("Google identity mismatch between ID token and userinfo")
        raise AppError(400, "oauth_exchange_failed", "Google returned an inconsistent identity.")
    user = db.scalar(select(User).where(User.google_sub == sub)) or get_user_by_email(db, email)
    if user is None:
        if not settings_service.get_value(db, "registration_enabled"):
            raise AppError(403, "registration_disabled", "New registrations are currently closed.")
        user = User(
            email=email,
            full_name=(profile.get("name") or email.split("@")[0])[:120],
            is_verified=True,
            is_active=True,
            auth_provider="google",
            google_sub=sub,
            avatar_url=profile.get("picture"),
            preferences=dict(DEFAULT_PREFERENCES),
        )
        db.add(user)
    else:
        if user.google_sub and user.google_sub != sub:
            raise AppError(409, "oauth_conflict", "This email is linked to a different Google account.")
        if not user.is_verified:
            # Someone may have pre-registered this address with a password they chose but never proved they own the mailbox.
            # Google just proved the real owner: drop that password so it cannot be used to enter the account.
            user.password_hash = None
            revoke_all_sessions(db, user.id)
        user.google_sub = sub
        user.is_verified = True  # Google asserted ownership of the email
        user.avatar_url = user.avatar_url or profile.get("picture")
    if not user.is_active:
        raise AppError(403, "account_disabled", "This account has been disabled.")
    db.commit()
    return user


# ------------------------------------------------------------ bootstrap
def ensure_admin(db: Session) -> None:
    s = get_settings()
    if not (s.admin_email and s.admin_password):
        return
    try:
        email = validate_email(s.admin_email, check_deliverability=False).normalized.lower()
        security_pw = s.admin_password
        validate_password_strength(security_pw)
    except (EmailNotValidError, ValueError) as exc:
        log.error("ADMIN_EMAIL/ADMIN_PASSWORD are invalid, bootstrap admin NOT created: %s", exc)
        return
    user = get_user_by_email(db, email)
    if user is None:
        db.add(User(
            email=email, full_name=s.admin_name, role=Role.admin.value, is_verified=True,
            password_hash=security.hash_password(s.admin_password), preferences=dict(DEFAULT_PREFERENCES),
        ))
        db.commit()
        log.info("Bootstrap admin created: %s", email)
    elif user.role != Role.admin.value:
        user.role = Role.admin.value
        user.is_verified = True
        db.commit()
