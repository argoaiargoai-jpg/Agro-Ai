import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    crop_type: str | None
    source: str
    original_filename: str | None
    content_type: str
    size_bytes: int
    notes: str | None
    status: str
    result: dict | None
    confidence: float | None
    error_message: str | None
    ai_provider: str | None = None
    ai_model: str | None = None
    ai_status: str | None = None
    ai_error_code: str | None = None
    ml_completed_at: datetime | None = None
    ai_completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AnalysisPage(BaseModel):
    items: list[AnalysisOut]
    total: int
    page: int
    page_size: int
