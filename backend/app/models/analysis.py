import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UTCDateTime, _now


class AnalysisStatus(str, enum.Enum):
    uploaded = "uploaded"      # image stored, waiting for an analysis engine
    queued = "queued"
    partial = "partial"        # ML finished but the external-AI guidance failed transiently (retry possible)
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    crop_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload | camera
    image_path: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=AnalysisStatus.uploaded.value, index=True)
    # Filled by the analysis engine in a later phase.
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # --- Phase 3: external AI guidance (result JSON holds the structured payloads; these are the queryable facts) ---
    ai_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ai_model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    ai_status: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)   # completed|disabled|not_configured|rate_limited|timeout|unavailable|bad_response|blocked|misconfigured|user_limit
    ai_error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ml_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    ai_attempted_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True, index=True)
    ai_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, onupdate=_now)
