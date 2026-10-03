import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import client_meta, get_current_user
from app.core.config import get_settings
from app.db.session import get_db
from app.models import User
from app.schemas.auth import (
    AuthOut, ChangePasswordIn, EmailIn, LoginIn, MessageOut, OtpSentOut,
    RegisterIn, ResetPasswordIn, UserOut, VerifyOtpIn,
)
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("agroai.auth")
COOKIE = "agro_refresh"
OAUTH_COOKIE = "agro_oauth"      # browser-side half of the OAuth state (short-lived, HttpOnly, only sent to the Google routes)


def _set_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(
        COOKIE, token, max_age=s.refresh_token_days * 86400, httponly=True,
        secure=s.cookie_secure, samesite=s.cookie_samesite, path=f"{s.api_prefix}/auth",
    )


def _clear_cookie(response: Response) -> None:
    s = get_settings()
    response.delete_cookie(COOKIE, path=f"{s.api_prefix}/auth", secure=s.cookie_secure, httponly=True, samesite=s.cookie_samesite)


def _auth_out(db: Session, user: User, request: Request, response: Response) -> AuthOut:
    ua, ip = client_meta(request)
    access, refresh = auth_service.create_session(db, user, ua, ip)
    _set_cookie(response, refresh)
    return AuthOut(access_token=access, user=UserOut.model_validate(user))


@router.post("/register", response_model=OtpSentOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    return auth_service.register(db, body.email, body.password, body.full_name)


@router.post("/verify-otp", response_model=AuthOut)
def verify_otp(body: VerifyOtpIn, request: Request, response: Response, db: Session = Depends(get_db)):
    user = auth_service.verify_email(db, body.email, body.code)
    return _auth_out(db, user, request, response)


@router.post("/resend-otp", response_model=OtpSentOut)
def resend_otp(body: EmailIn, db: Session = Depends(get_db)):
    return auth_service.resend_verification(db, body.email)


@router.post("/login", response_model=AuthOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    user = auth_service.authenticate(db, body.email, body.password)
    return _auth_out(db, user, request, response)


@router.post("/refresh", response_model=AuthOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    ua, ip = client_meta(request)
    try:
        access, new_refresh, user = auth_service.rotate_session(db, request.cookies.get(COOKIE), ua, ip)
    except Exception:
        _clear_cookie(response)
        raise
    _set_cookie(response, new_refresh)
    return AuthOut(access_token=access, user=UserOut.model_validate(user))


@router.post("/logout", response_model=MessageOut)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    auth_service.revoke_session_by_token(db, request.cookies.get(COOKIE))
    _clear_cookie(response)
    return {"message": "Signed out."}


@router.post("/forgot-password", response_model=OtpSentOut)
def forgot_password(body: EmailIn, db: Session = Depends(get_db)):
    return auth_service.request_password_reset(db, body.email)


@router.post("/reset-password", response_model=MessageOut)
def reset_password(body: ResetPasswordIn, db: Session = Depends(get_db)):
    auth_service.reset_password(db, body.email, body.code, body.new_password)
    return {"message": "Password updated. You can sign in now."}


@router.post("/change-password", response_model=MessageOut)
def change_password(
    body: ChangePasswordIn, response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    auth_service.change_password(db, user, body.current_password, body.new_password)
    _clear_cookie(response)
    return {"message": "Password changed. Please sign in again."}


# ---- Google OAuth -----------------------------------------------------
@router.get("/google/status")
def google_status():
    return {"enabled": get_settings().google_enabled}


def _oauth_cookie_path() -> str:
    return f"{get_settings().api_prefix}/auth/google"


@router.get("/google/login")
def google_login():
    s = get_settings()
    url, nonce = auth_service.google_authorization_url()
    resp = RedirectResponse(url, status_code=302)
    # Lax is enough (the callback is a top-level navigation back from Google) and is the stricter choice than the refresh cookie's "none".
    resp.set_cookie(OAUTH_COOKIE, nonce, max_age=600, httponly=True, secure=s.cookie_secure, samesite="lax", path=_oauth_cookie_path())
    return resp


@router.get("/google/callback")
def google_callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    db: Session = Depends(get_db),
):
    s = get_settings()
    login_url = f"{s.frontend_url.rstrip('/')}/login"

    def fail(code_: str):
        r = RedirectResponse(f"{login_url}?{urlencode({'oauth_error': code_})}", status_code=302)
        r.delete_cookie(OAUTH_COOKIE, path=_oauth_cookie_path())
        return r

    if error or not code or not state:
        return fail("oauth_cancelled")
    from app.core.errors import AppError

    try:
        user = auth_service.google_complete(db, code, state, request.cookies.get(OAUTH_COOKIE))
        ua, ip = client_meta(request)
        _, refresh = auth_service.create_session(db, user, ua, ip)
    except AppError as e:
        return fail(e.code)
    except Exception:  # noqa: BLE001  (a surprise in the Google exchange must end on our login page, never on a raw error page)
        log.exception("Unexpected error in the Google callback")
        return fail("oauth_exchange_failed")
    resp = RedirectResponse(f"{s.frontend_url.rstrip('/')}/auth/google/done", status_code=302)
    _set_cookie(resp, refresh)
    resp.delete_cookie(OAUTH_COOKIE, path=_oauth_cookie_path())          # single use
    return resp
