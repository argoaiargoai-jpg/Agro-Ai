from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.db.session import get_db
from app.services import ml_service, settings_service

router = APIRouter(tags=["system"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    s = get_settings()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        raise AppError(503, "db_unavailable", "Database is not reachable.") from exc
    return {"status": "ok", "app": s.app_name, "environment": s.environment, "database": "ok",
            "ml_model": "installed" if (s.ml_path / "agro_model.onnx").exists() else "missing"}


@router.get("/config/public")
def public_config(db: Session = Depends(get_db)):
    s = get_settings()
    return {**settings_service.get_all(db, public_only=True), "google_enabled": s.google_enabled}


@router.get("/ml/info")
def ml_info():
    """Which crops/conditions the installed model actually supports (honest scope for the UI)."""
    try:
        return ml_service.get_ml_service().info()
    except AppError:
        return {"available": False, "supported_crops": [], "supported_conditions": [], "outcomes": []}
