import uuid
from datetime import timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.core.security import utcnow
from app.services import analysis_service
from app.models import Analysis, AuditLog, Role, User
from app.services import auth_service


def stats(db: Session) -> dict:
    now = utcnow()
    count = lambda stmt: db.scalar(stmt) or 0  # noqa: E731
    since = now - timedelta(days=14)
    signups = db.execute(
        select(func.date(User.created_at), func.count()).where(User.created_at >= since).group_by(func.date(User.created_at))
    ).all()
    by_status = dict(db.execute(select(Analysis.status, func.count()).group_by(Analysis.status)).all())
    return {
        "users_total": count(select(func.count()).select_from(User)),
        "users_verified": count(select(func.count()).select_from(User).where(User.is_verified)),
        "users_active_7d": count(select(func.count()).select_from(User).where(User.last_login_at >= now - timedelta(days=7))),
        "admins": count(select(func.count()).select_from(User).where(User.role == Role.admin.value)),
        "analyses_total": count(select(func.count()).select_from(Analysis)),
        "analyses_by_status": by_status,
        "signups_14d": [{"date": str(d), "count": c} for d, c in sorted(signups)],
        **analysis_service.aggregate(db.execute(
            select(Analysis.created_at, Analysis.result, Analysis.ai_status, Analysis.crop_type)
            .order_by(Analysis.created_at.desc()).limit(analysis_service.SUMMARY_WINDOW)).all()),
    }


def list_users(db: Session, page: int, page_size: int, q: str | None, role: str | None):
    stmt = select(User)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(User.email.ilike(like), User.full_name.ilike(like)))
    if role:
        stmt = stmt.where(User.role == role)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(stmt.order_by(User.created_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return items, total


def update_user(db: Session, actor: User, user_id: uuid.UUID, role: str | None, is_active: bool | None) -> User:
    target = db.get(User, user_id)
    if target is None:
        raise AppError(404, "not_found", "User not found.")
    if target.id == actor.id and (role == "user" or is_active is False):
        raise AppError(400, "self_lockout", "You can't demote or deactivate your own account.")
    if role == "user" and target.role == Role.admin.value:
        admins = db.scalar(select(func.count()).select_from(User).where(User.role == Role.admin.value, User.is_active)) or 0
        if admins <= 1:
            raise AppError(400, "last_admin", "At least one active admin is required.")
    changes = {}
    if role is not None and role != target.role:
        target.role = role
        changes["role"] = role
    if is_active is not None and is_active != target.is_active:
        target.is_active = is_active
        changes["is_active"] = is_active
    if changes:
        db.add(AuditLog(actor_id=actor.id, action="user.update", target=str(target.id), meta=changes))
        db.commit()
        if changes.get("is_active") is False or "role" in changes:
            auth_service.revoke_all_sessions(db, target.id)
    return target


def audit_log(db: Session, limit: int = 50):
    rows = db.execute(
        select(AuditLog, User.email).outerjoin(User, User.id == AuditLog.actor_id).order_by(AuditLog.created_at.desc()).limit(limit)
    ).all()
    return [
        {"id": str(l.id), "actor": email, "action": l.action, "target": l.target, "meta": l.meta, "created_at": l.created_at}
        for l, email in rows
    ]
