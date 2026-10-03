import uuid

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core import security
from app.core.errors import AppError
from app.db.session import get_db
from app.models import AuthSession, Role, User

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)
) -> User:
    if creds is None:
        raise AppError(401, "unauthorized", "Authentication required.")
    try:
        payload = security.decode_token(creds.credentials, "access")
        user_id, sid = uuid.UUID(payload["sub"]), uuid.UUID(payload["sid"])
    except jwt.ExpiredSignatureError:
        raise AppError(401, "token_expired", "Your session expired.") from None
    except (jwt.PyJWTError, ValueError, KeyError):
        raise AppError(401, "unauthorized", "Invalid authentication token.") from None
    sess = db.get(AuthSession, sid)
    if sess is None or sess.revoked_at is not None or sess.user_id != user_id:
        raise AppError(401, "unauthorized", "Session is no longer valid.")
    user = db.get(User, user_id)
    if user is None or not user.is_active or not user.is_verified:
        raise AppError(401, "unauthorized", "Account is not available.")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != Role.admin.value:
        raise AppError(403, "forbidden", "Administrator access required.")
    return user


def client_meta(request: Request) -> tuple[str | None, str | None]:
    return request.headers.get("user-agent"), request.client.host if request.client else None
