"""Admin view of the AI layer. Secrets are never returned: only whether they are configured."""
import threading
import time
import uuid
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai import groq, pollinations, registry
from app.ai.base import ProviderError
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.security import utcnow
from app.models import Analysis, AuditLog
from app.services import settings_service

_last_test = 0.0
_lock = threading.Lock()


def _workflow(db: Session, enabled: bool, active: str, bypass: bool) -> list[dict]:
    """Status of EVERY provider the analysis workflow can use. Safe metadata only (counts and booleans, never a key). `state`:
    primary = the selected main AI provider; available = configured and used by the workflow when its turn comes; fallback = Groq / Pollinations
    (used only when the provider before them fails; Pollinations is first, also while Gemini is bypassed); bypassed = Gemini while the admin bypass switch is on;
    disabled = AI guidance switched off; not_configured = no key on the server."""
    s = get_settings()
    keys = len(s.gemini_keys())
    kindwise = bool(s.kindwise_api_key.get_secret_value().strip() or s.kindwise_plant_api_key.get_secret_value().strip())
    plantnet = bool(s.plantnet_api_key.get_secret_value().strip())
    groq_ok, poll_ok = groq.build(s).is_configured(), pollinations.build(s).is_configured()
    gemini_state = "not_configured" if not keys else "disabled" if not enabled else "bypassed" if bypass else "primary" if active == "gemini" else "available"
    return [
        {"id": "gemini", "name": "Google Gemini", "purpose": "AI verification & guidance (main AI step)", "configured": bool(keys), "model": s.gemini_model,
         "keys": {"configured": keys, "of": 2}, "state": gemini_state, "testable": True},
        {"id": "kindwise", "name": "Kindwise", "purpose": "Crop / plant health specialist", "configured": kindwise,
         "state": "available" if kindwise else "not_configured", "testable": False},
        {"id": "plantnet", "name": "Pl@ntNet", "purpose": "Plant identification specialist", "configured": plantnet,
         "state": "available" if plantnet else "not_configured", "testable": False},
        {"id": "pollinations", "name": "Pollinations", "purpose": "AI fallback 1 (after Gemini)", "configured": poll_ok, "model": s.pollinations_model,
         "state": "not_configured" if not poll_ok else "disabled" if not enabled else "fallback", "testable": True},
        {"id": "groq", "name": "Groq", "purpose": "AI fallback 2 (after Pollinations)", "configured": groq_ok, "model": s.groq_model,
         "state": "not_configured" if not groq_ok else "disabled" if not enabled else "fallback", "testable": True},
    ]


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
        "generative_ai_bypass_gemini": bool(settings_service.get_value(db, "generative_ai_bypass_gemini")),
        "workflow": _workflow(db, enabled, active, bool(settings_service.get_value(db, "generative_ai_bypass_gemini"))),
        "fallbacks": {"pollinations": {"configured": pollinations.build(s).is_configured(), "model": s.pollinations_model},          # never the keys
                      "groq": {"configured": groq.build(s).is_configured(), "model": s.groq_model}},
        "secret_source": "environment",          # keys are set in the server environment, never through the UI or database
        "limits": {"timeout_seconds": s.ai_timeout_seconds, "max_retries": s.ai_max_retries, "max_concurrency": s.ai_max_concurrency,
                   "user_hourly_limit": s.ai_user_hourly_limit, "max_image_side_px": s.ai_max_image_side},
        "usage_24h": usage,
    }


def update(db: Session, actor_id: uuid.UUID, enabled: bool | None, provider: str | None, generative_ai_bypass_gemini: bool | None = None) -> dict:
    values = {}
    if generative_ai_bypass_gemini is not None:
        values["generative_ai_bypass_gemini"] = generative_ai_bypass_gemini
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
        p = {"groq": groq.build, "pollinations": pollinations.build}[name](s) if name in ("groq", "pollinations") else registry.build(name, s)
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
