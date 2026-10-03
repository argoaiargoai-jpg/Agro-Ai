from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_DEFAULT_SECRET = "dev-insecure-secret-change-me-dev-insecure-secret"
KNOWN_DEFAULT_ADMIN_PASSWORDS = {"ChangeMe123!", "AdminPass123", "admin", "password", "Password123"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AGRO AI"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./agroai.db"
    secret_key: str = INSECURE_DEFAULT_SECRET
    access_token_minutes: int = 15
    refresh_token_days: int = 14
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"   # "none" (+ COOKIE_SECURE=true) is needed when the frontend and API are on different sites

    frontend_url: str = "http://localhost:5173"
    backend_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:5173"      # comma-separated EXACT origins, e.g. https://app.netlify.app (never "*")
    cors_origin_regex: str = ""                      # optional, for Netlify deploy previews, e.g. https://.*--agroai\.netlify\.app

    email_backend: Literal["console", "smtp", "brevo"] = "console"
    expose_dev_otp: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    brevo_api_key: SecretStr = SecretStr("")           # backend-only secret for EMAIL_BACKEND=brevo (HTTPS API; works where SMTP ports are blocked)
    brevo_base_url: str = "https://api.brevo.com"
    email_from: str = "AGRO AI <no-reply@agroai.local>"

    otp_ttl_minutes: int = 10
    otp_max_attempts: int = 5
    otp_resend_cooldown_seconds: int = 60

    bcrypt_rounds: int = 12
    max_failed_logins: int = 5
    lockout_minutes: int = 15

    google_client_id: str = ""
    google_client_secret: str = ""

    admin_email: str = ""
    admin_password: str = ""
    admin_name: str = "AGRO AI Admin"

    # --- External AI provider (backend-only secrets; NEVER sent to the frontend or stored in the DB) ---
    gemini_api_key: SecretStr = SecretStr("")           # legacy single key: used as slot 1 when GEMINI_API_KEY_1 is not set
    gemini_api_key_1: SecretStr = SecretStr("")         # two keys are used alternately (persistent DB counter) with automatic failover
    gemini_api_key_2: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.8-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    ai_timeout_seconds: float = 45.0
    ai_max_retries: int = 1                 # extra attempts on timeouts / 5xx only (never on 429 or 4xx)
    ai_max_concurrency: int = 3             # simultaneous provider calls
    ai_max_image_side: int = 1568           # longest side sent to the provider (metadata is stripped by re-encoding)
    ai_user_hourly_limit: int = 30          # provider calls per user per hour

    # --- Specialist plant/disease providers (all optional; each one that is not configured is simply skipped) ---
    plantnet_api_key: SecretStr = SecretStr("")         # plant identification (my.plantnet.org)
    plantnet_base_url: str = "https://my-api.plantnet.org"
    plantix_api_key: SecretStr = SecretStr("")          # crop disease image analysis (Plantix partner API)
    plantix_base_url: str = "https://api.plantix.net"
    kindwise_api_key: SecretStr = SecretStr("")         # crop.health (crop disease identification)
    kindwise_crop_base_url: str = "https://crop.kindwise.com"
    kindwise_plant_api_key: SecretStr = SecretStr("")   # plant.health (non-crop plants); separate Kindwise product, optional
    kindwise_plant_base_url: str = "https://plant.id"
    specialist_timeout_seconds: float = 20.0

    ml_model_dir: str = ""              # default: app/ml (model + metadata shipped with the app)
    ml_threads: int = 1                 # ONNX Runtime CPU threads (keep low on small free-tier machines)
    ml_max_concurrency: int = 2         # simultaneous inferences (bounds RAM)
    ml_max_pixels: int = 40_000_000     # reject decompression-bomb style images

    upload_dir: str = "./uploads"
    max_upload_mb: int = 10

    @model_validator(mode="after")
    def _normalise_and_check(self):
        if not self.secret_key.strip():
            self.secret_key = INSECURE_DEFAULT_SECRET      # empty value = dev default (production still refuses it below)
        # Render/Heroku-style URLs ("postgres://", "postgresql://") -> the psycopg driver we ship
        for prefix in ("postgres://", "postgresql://"):
            if self.database_url.startswith(prefix):
                self.database_url = "postgresql+psycopg://" + self.database_url[len(prefix):]
        self.cors_origins = ",".join(o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip())
        self.frontend_url, self.backend_url = self.frontend_url.rstrip("/"), self.backend_url.rstrip("/")
        origins = self.cors_origin_list
        if "*" in origins or self.cors_origin_regex.strip() in (".*", ".+"):
            raise ValueError("CORS_ORIGINS must list explicit origins: a wildcard cannot be combined with credentials (cookies)")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise ValueError("COOKIE_SAMESITE=none requires COOKIE_SECURE=true (browsers reject it otherwise)")
        if self.environment == "production":
            problems = []
            if self.secret_key == INSECURE_DEFAULT_SECRET or len(self.secret_key) < 32:
                problems.append("SECRET_KEY must be a random value of >=32 chars")
            if not self.cookie_secure:
                problems.append("COOKIE_SECURE must be true (HTTPS only)")
            if self.admin_password and self.admin_password in KNOWN_DEFAULT_ADMIN_PASSWORDS:
                problems.append("ADMIN_PASSWORD is a known default; choose a strong unique password")
            for name, url in (("FRONTEND_URL", self.frontend_url), ("BACKEND_URL", self.backend_url)):
                if not url.startswith("https://"):
                    problems.append(f"{name} must be an https:// URL")
            bad = [o for o in origins if not o.startswith("https://")]
            if bad:
                problems.append(f"CORS_ORIGINS must be https:// origins in production (got {', '.join(bad)})")
            if self.database_url.startswith("sqlite"):
                problems.append("DATABASE_URL points at SQLite; Render Free has an ephemeral disk, so use a hosted PostgreSQL database")
            if problems:
                raise ValueError("Unsafe production configuration: " + "; ".join(problems))
        return self

    def startup_warnings(self) -> list[str]:
        w = []
        if self.environment == "production":
            if self.email_backend == "console":
                w.append("EMAIL_BACKEND=console: verification codes are only printed to the log, so new users cannot register. Configure SMTP or Brevo.")
            if self.email_backend == "brevo" and not self.brevo_api_key.get_secret_value().strip():
                w.append("EMAIL_BACKEND=brevo but BREVO_API_KEY is not set: verification emails cannot be sent.")
            if not self.gemini_configured:
                w.append("No Gemini key is set (GEMINI_API_KEY_1 / GEMINI_API_KEY_2): analyses will show only the basic model result.")
            elif len(self.gemini_keys()) == 1:
                w.append("Only one Gemini key is configured: requests are not alternated and there is no failover key (set GEMINI_API_KEY_1 and GEMINI_API_KEY_2).")
            if any("localhost" in o or "127.0.0.1" in o for o in self.cors_origin_list):
                w.append("CORS_ORIGINS contains a localhost origin in production.")
        return w

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def google_redirect_uri(self) -> str:
        return f"{self.backend_url.rstrip('/')}{self.api_prefix}/auth/google/callback"

    @property
    def dev_otp_visible(self) -> bool:
        return self.expose_dev_otp and self.environment != "production"

    def gemini_keys(self) -> list[tuple[int, str]]:
        """Configured Gemini keys as (slot, key); slot 1 = GEMINI_API_KEY_1 (or the legacy GEMINI_API_KEY), slot 2 = GEMINI_API_KEY_2."""
        k1 = self.gemini_api_key_1.get_secret_value().strip() or self.gemini_api_key.get_secret_value().strip()
        k2 = self.gemini_api_key_2.get_secret_value().strip()
        return [(slot, k) for slot, k in ((1, k1), (2, k2)) if k]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_keys())

    def secret_values(self) -> list[str]:
        """Every provider secret, for log redaction. Never log or return these."""
        fields = (self.gemini_api_key, self.gemini_api_key_1, self.gemini_api_key_2, self.plantnet_api_key, self.plantix_api_key,
                  self.kindwise_api_key, self.kindwise_plant_api_key, self.brevo_api_key)
        return [v for v in (f.get_secret_value().strip() for f in fields) if len(v) >= 6]

    @property
    def ml_path(self) -> Path:
        return Path(self.ml_model_dir).resolve() if self.ml_model_dir else Path(__file__).resolve().parent.parent / "ml"

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
