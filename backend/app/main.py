import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, analyses, auth, system, users
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.middleware import MaxUploadSizeMiddleware, SecurityHeadersMiddleware
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.services import auth_service, settings_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)      # request lines are noise and must never be able to leak anything
log = logging.getLogger("agroai")


class SecretRedactingFilter(logging.Filter):
    """Defence in depth: even if some code path tried to log the provider key, it is masked before it is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        from app.ai.safety import redact
        msg = record.getMessage()
        red = redact(msg, [get_settings().gemini_api_key.get_secret_value()])
        if red != msg:
            record.msg, record.args = red, ()
        return True


for _h in logging.getLogger().handlers:
    _h.addFilter(SecretRedactingFilter())


def init_db() -> None:
    import app.models  # noqa: F401  (register tables)

    if get_settings().environment != "production":
        Base.metadata.create_all(engine)  # dev/test: zero-setup. Production relies on `alembic upgrade head` (see README / render.yaml).
    with SessionLocal() as db:
        settings_service.seed_defaults(db)
        auth_service.ensure_admin(db)


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    s.upload_path.mkdir(parents=True, exist_ok=True)
    for w in s.startup_warnings():
        log.warning("CONFIG WARNING: %s", w)
    init_db()
    log.info("%s started (%s); ML model %s; AI guidance key %s", s.app_name, s.environment,
             "installed" if (s.ml_path / "agro_model.onnx").exists() else "NOT installed", "configured" if s.gemini_configured else "not configured")
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=f"{s.app_name} API", version="0.1.0", lifespan=lifespan,
        docs_url=None if s.environment == "production" else "/docs",
        redoc_url=None, openapi_url=None if s.environment == "production" else "/openapi.json",
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MaxUploadSizeMiddleware)
    app.add_middleware(      # added last = outermost, so CORS headers also appear on early rejections (e.g. 413)
        CORSMiddleware, allow_origins=s.cors_origin_list, allow_origin_regex=s.cors_origin_regex or None, allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type"],
    )
    register_error_handlers(app)
    for r in (system.router, auth.router, users.router, analyses.router, admin.router):
        app.include_router(r, prefix=s.api_prefix)
    return app


app = create_app()
