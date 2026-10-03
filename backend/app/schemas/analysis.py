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


INTERNAL_RESULT_KEYS = ("ai", "prompt_version", "ai_case")
INTERNAL_FINAL_KEYS = ("disagreement", "disease_source", "ml_state")


def present(analysis, is_admin: bool) -> "AnalysisOut":
    """The API view of an analysis. Administrators get the full record; customers get the unified result only
    (no provider output, ML-vs-AI disagreement, provider/model names or internal error codes)."""
    out = AnalysisOut.model_validate(analysis)
    if is_admin:
        return out
    result = dict(out.result) if out.result else out.result
    if result:
        for k in INTERNAL_RESULT_KEYS:
            result.pop(k, None)
        if isinstance(result.get("final"), dict):
            result["final"] = {k: v for k, v in result["final"].items() if k not in INTERNAL_FINAL_KEYS}
    out.result = result
    out.ai_provider = out.ai_model = out.ai_error_code = None
    return out
