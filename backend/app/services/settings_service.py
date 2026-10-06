import uuid

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.models import AuditLog, SystemSetting

# AI settings live in the same table but are managed from the AI tab; they are NOT part of the public config or the generic System tab.
AI_DEFAULTS: dict[str, tuple[object, str]] = {
    "ai_enabled": (True, "Run external AI guidance after our ML model"),
    "ai_provider": ("gemini", "Active external AI provider"),
    "generative_ai_bypass_gemini": (False, "ADMIN TEST: skip Gemini and send the AI request to the fallback providers (Groq, then Pollinations)"),
}

# key -> (default, description, public?)
DEFAULTS: dict[str, tuple[object, str, bool]] = {
    "registration_enabled": (True, "Allow new users to sign up", True),
    "maintenance_mode": (False, "Show maintenance banner and block new analyses", True),
    "announcement": ("", "Banner message shown to all users", True),
    "max_upload_mb": (10, "Maximum image upload size (MB)", True),
    "supported_crops": (["Tomato", "Potato", "Maize", "Rice", "Wheat", "Cotton"], "Crops offered in the analyzer", True),
}

VALIDATORS = {
    "registration_enabled": lambda v: isinstance(v, bool),
    "maintenance_mode": lambda v: isinstance(v, bool),
    "announcement": lambda v: isinstance(v, str) and len(v) <= 300,
    "max_upload_mb": lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 25,
    "supported_crops": lambda v: isinstance(v, list) and 1 <= len(v) <= 50
    and all(isinstance(c, str) and 1 <= len(c.strip()) <= 40 for c in v),
}


def _ai_provider_ok(v) -> bool:
    from app.ai import registry
    return isinstance(v, str) and v in registry.names()


VALIDATORS["ai_enabled"] = lambda v: isinstance(v, bool)
VALIDATORS["generative_ai_bypass_gemini"] = lambda v: isinstance(v, bool)
VALIDATORS["ai_provider"] = _ai_provider_ok


def _default(key: str):
    return DEFAULTS[key][0] if key in DEFAULTS else AI_DEFAULTS[key][0]


def seed_defaults(db: Session) -> None:
    existing = {r.key for r in db.query(SystemSetting.key).all()}
    for key, (value, desc) in AI_DEFAULTS.items():
        if key not in existing:
            db.add(SystemSetting(key=key, value=value, description=desc))
    for key, (value, desc, _) in DEFAULTS.items():
        if key not in existing:
            db.add(SystemSetting(key=key, value=value, description=desc))
    db.commit()


def get_all(db: Session, public_only: bool = False) -> dict:
    rows = {r.key: r.value for r in db.query(SystemSetting).all()}
    out = {}
    for key, (default, _, public) in DEFAULTS.items():
        if public_only and not public:
            continue
        out[key] = rows.get(key, default)
    return out


def get_value(db: Session, key: str):
    row = db.get(SystemSetting, key)
    return row.value if row else _default(key)


def effective_max_upload_bytes(db: Session) -> int:
    cap = get_settings().max_upload_mb
    return min(int(get_value(db, "max_upload_mb")), max(cap, 1)) * 1024 * 1024


def update(db: Session, actor_id: uuid.UUID, values: dict) -> dict:
    if not values:
        raise AppError(422, "validation_error", "No settings provided.")
    errors = {}
    for key, val in values.items():
        if key not in DEFAULTS and key not in AI_DEFAULTS:
            errors[key] = "Unknown setting."
        elif not VALIDATORS[key](val):
            errors[key] = "Invalid value."
    if errors:
        raise AppError(422, "validation_error", "Some settings are invalid.", errors)
    for key, val in values.items():
        if key == "supported_crops":
            val = [c.strip() for c in val]
        row = db.get(SystemSetting, key)
        if row is None:
            row = SystemSetting(key=key, description=(DEFAULTS.get(key) or AI_DEFAULTS[key])[1])
            db.add(row)
        row.value = val
        row.updated_by = actor_id
    db.add(AuditLog(actor_id=actor_id, action="settings.update", target=",".join(values), meta=values))
    db.commit()
    return get_all(db)
