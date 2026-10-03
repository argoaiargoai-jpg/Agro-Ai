import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.session import get_db
from app.models import User
from app.schemas.admin import AdminUserUpdateIn, AIConfigIn, AITestIn, SettingsUpdateIn, UserPage
from app.schemas.auth import UserOut
from app.services import admin_service, ai_admin_service, settings_service

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    return admin_service.stats(db)


@router.get("/users", response_model=UserPage)
def users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    q: str | None = Query(None, max_length=80),
    role: str | None = Query(None, pattern="^(user|admin)$"),
    db: Session = Depends(get_db),
):
    items, total = admin_service.list_users(db, page, page_size, q, role)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.patch("/users/{user_id}", response_model=UserOut)
def patch_user(user_id: uuid.UUID, body: AdminUserUpdateIn, actor: User = Depends(require_admin), db: Session = Depends(get_db)):
    return admin_service.update_user(db, actor, user_id, body.role, body.is_active)


@router.get("/settings")
def get_settings_(db: Session = Depends(get_db)):
    return settings_service.get_all(db)


@router.put("/settings")
def put_settings(body: SettingsUpdateIn, actor: User = Depends(require_admin), db: Session = Depends(get_db)):
    return settings_service.update(db, actor.id, body.values)


@router.get("/audit-log")
def audit(limit: int = Query(30, ge=1, le=200), db: Session = Depends(get_db)):
    return admin_service.audit_log(db, limit)


@router.get("/ai")
def ai_overview(db: Session = Depends(get_db)):
    return ai_admin_service.overview(db)


@router.put("/ai")
def ai_update(body: AIConfigIn, actor: User = Depends(require_admin), db: Session = Depends(get_db)):
    return ai_admin_service.update(db, actor.id, body.enabled, body.provider)


@router.post("/ai/test")
def ai_test(body: AITestIn | None = None, actor: User = Depends(require_admin), db: Session = Depends(get_db)):
    return ai_admin_service.test_connection(db, actor.id, body.provider if body else None)
