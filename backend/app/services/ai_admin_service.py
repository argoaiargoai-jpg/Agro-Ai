"""Admin view of the AI layer. Secrets are never returned: only whether they are configured."""
import threading
import time
import uuid
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai import registry
from app.ai.base import ProviderError
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.security import utcnow
from app.models import Analysis, AuditLog
from app.services import settings_service

_last_test = 0.0
_lock = threading.Lock()


def overview(db: Session) -> dict:
    s = get_settings()
    enabled = settings_service.get_value(db, "ai_enabled")
    active = settings_service.get_value(db, "ai_provider")
    providers = registry.describe(s)
    cur = next((p for p in providers if p["name"] == active), None)
    state = "disabled" if not enabled else "unknown_provider" if cur is None else "ready" if cur["configured"] else "not_configured"
    since = utcnow() - timedelta(hours=24)
    usage = dict(db.execute(select(Analysis.ai_status, func.count()).where(Analysis.ai_attempted_at >= since, Analysis.ai_status.is_not(None))
                            .group_by(Analysis.ai_status)).all())
    return {
        "enabled": enabled, "active_provider": active, "state": state, "providers": providers,
        "secret_source": "environment",          # keys are set in the server environment, never through the UI or database
        "limits": {"timeout_seconds": s.ai_timeout_seconds, "max_retries": s.ai_max_retries, "max_concurrency": s.ai_max_concurrency,
                   "user_hourly_limit": s.ai_user_hourly_limit, "max_image_side_px": s.ai_max_image_side},
        "usage_24h": usage,
    }


def update(db: Session, actor_id: uuid.UUID, enabled: bool | None, provider: str | None) -> dict:
    values = {}
    if enabled is not None:
        values["ai_enabled"] = enabled
    if provider is not None:
        values["ai_provider"] = provider
    settings_service.update(db, actor_id, values)
    return overview(db)


def test_connection(db: Session, actor_id: uuid.UUID, provider: str | None) -> dict:
    """One tiny text-only request. Safe to expose to admins: returns state + a sanitised reason, never a secret."""
    global _last_test
    with _lock:
        if time.monotonic() - _last_test < 3:
            raise AppError(429, "too_many_requests", "Please wait a few seconds between connection tests.")
        _last_test = time.monotonic()
    s = get_settings()
    name = provider or settings_service.get_value(db, "ai_provider")
    try:
        p = registry.build(name, s)
    except KeyError:
        raise AppError(422, "validation_error", "Unknown provider.", {"provider": "Unknown provider."}) from None
    if not p.is_configured():
        out = {"ok": False, "state": "not_configured", "message": "No API key is configured for this provider on the server.", "provider": name}
    else:
        try:
            r = p.ping()
            out = {"ok": bool(r.get("ok")), "state": "ready" if r.get("ok") else "bad_response", "latency_ms": r.get("latency_ms"), "model": r.get("model"),
                   "message": "Connection works." if r.get("ok") else "The provider answered but not as expected.", "provider": name}
        except ProviderError as e:
            out = {"ok": False, "state": e.ai_status, "message": e.user_message, "detail": e.detail[:200], "retry_after": e.retry_after, "provider": name}
    db.add(AuditLog(actor_id=actor_id, action="ai.test", target=name, meta={"ok": out["ok"], "state": out["state"]}))
    db.commit()
    return out
