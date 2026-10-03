"""Strict schemas: what we ask the AI for (`AIAnalysis`) and what we store/show (`AnalysisReport`).

Provider output is never trusted: it is validated here, lengths are capped, enums are closed, and anything
unparseable becomes a ProviderBadResponse upstream instead of reaching the user.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

MAX_ITEMS, MAX_ITEM_LEN, MAX_TEXT = 10, 300, 1200


def _clean_str(v):
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError("expected a string")
    v = " ".join(v.split())
    return v or None


def _clean_list(v):
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError("expected a list")
    out = []
    for x in v:
        if not isinstance(x, str):
            raise ValueError("list items must be strings")
        x = " ".join(x.split())
        if x:
            out.append(x[:MAX_ITEM_LEN])
    return out[:MAX_ITEMS]


Text = Annotated[str | None, BeforeValidator(_clean_str)]
Items = Annotated[list[str], BeforeValidator(_clean_list)]


class SpreadRisk(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)       # strict: provider output is untrusted, never "guess" a type
    level: Literal["low", "moderate", "high", "unknown"] = "unknown"
    explanation: Text = None


class AIAnalysis(BaseModel):
    """Exactly what the external AI must return."""
    model_config = ConfigDict(extra="ignore", strict=True)

    plant_present: bool
    plant: Text = None
    crop: Text = None
    identification_confidence: Literal["high", "medium", "low"] = "low"   # the AI's own qualitative certainty; never averaged with ML
    health_status: Literal["healthy", "diseased", "uncertain", "not_a_plant"]
    disease: Text = None
    symptoms: Items = []
    severity: Literal["none", "mild", "moderate", "severe", "unknown"] = "unknown"
    affected_percentage: float | None = None
    immediate_actions: Items = []
    treatment: Items = []
    prevention: Items = []
    spread_risk: SpreadRisk = SpreadRisk()
    warnings: Items = []
    monitoring: Items = []
    ml_consistency: Literal["consistent", "inconsistent", "cannot_assess"] | None = None
    image_quality: Literal["good", "fair", "poor"] | None = None
    ai_notes: Text = None

    @field_validator("affected_percentage", mode="before")
    @classmethod
    def _pct(cls, v):
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)) and 0 <= v <= 100:
            return float(v)
        return None                       # out of range / not numeric: "cannot be estimated" (never fabricate)

    @field_validator("ai_notes")
    @classmethod
    def _cap(cls, v):
        return v[:MAX_TEXT] if v else v

    @model_validator(mode="after")
    def _coherent(self):
        if not self.plant_present:
            self.health_status = "not_a_plant"
        return self


class Disagreement(BaseModel):
    ml_said: str
    ai_said: str
    message: str


class AnalysisReport(BaseModel):
    """The final analysis shown to the user (stored in analyses.result['final'])."""
    status: Literal["DISEASE", "HEALTHY", "UNCERTAIN", "REJECTED"]
    headline: str
    ml_state: Literal["DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"]
    plant: str | None = None
    crop: str | None = None
    disease: str | None = None
    disease_source: Literal["ml", "ai", "specialist"] | None = None
    symptoms: list[str] = []
    severity: Literal["none", "mild", "moderate", "severe", "unknown"] | None = None
    affected_percentage: float | None = None
    immediate_actions: list[str] = []
    treatment: list[str] = []
    prevention: list[str] = []
    spread_risk: SpreadRisk | None = None
    warnings: list[str] = []
    monitoring: list[str] = []
    ai_notes: str | None = None
    identification_confidence: Literal["high", "medium", "low"] | None = None
    disagreement: Disagreement | None = None
    rejection_reason: str | None = None
    disclaimer: str


# Hand-written JSON schema sent to the provider (compact, stable, nullable via type arrays).
_STR_LIST = {"type": "array", "items": {"type": "string"}, "maxItems": MAX_ITEMS}
AI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "plant_present": {"type": "boolean", "description": "true only if real plant/crop/leaf material is visible"},
        "plant": {"type": ["string", "null"], "description": "plant seen (common name), null if none or unsure"},
        "crop": {"type": ["string", "null"], "description": "agricultural crop, null if not a crop or unsure"},
        "identification_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "health_status": {"type": "string", "enum": ["healthy", "diseased", "uncertain", "not_a_plant"]},
        "disease": {"type": ["string", "null"], "description": "disease/disorder name only if visibly supported, else null"},
        "symptoms": {**_STR_LIST, "description": "visible observations only"},
        "severity": {"type": "string", "enum": ["none", "mild", "moderate", "severe", "unknown"]},
        "affected_percentage": {"type": ["number", "null"], "minimum": 0, "maximum": 100, "description": "visual estimate of affected area, null if it cannot be estimated"},
        "immediate_actions": _STR_LIST, "treatment": _STR_LIST, "prevention": _STR_LIST,
        "spread_risk": {"type": "object", "properties": {"level": {"type": "string", "enum": ["low", "moderate", "high", "unknown"]},
                                                         "explanation": {"type": ["string", "null"]}}, "required": ["level"]},
        "warnings": _STR_LIST, "monitoring": _STR_LIST,
        "ml_consistency": {"type": ["string", "null"], "enum": ["consistent", "inconsistent", "cannot_assess", None]},
        "image_quality": {"type": ["string", "null"], "enum": ["good", "fair", "poor", None]},
        "ai_notes": {"type": ["string", "null"], "description": "separate what is observed from what is inferred; state uncertainty"},
    },
    "required": ["plant_present", "health_status", "severity", "spread_risk"],
}
