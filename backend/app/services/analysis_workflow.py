"""Turns (our ML result + the AI provider's independent visual assessment) into the final report.

A plain mapping, NOT a decision engine: no voting, no confidence averaging, no routing by confidence. For EVERY analysis the AI receives the
original image and inspects it independently; our model's result is only supporting context. The customer sees the AI's visual assessment:
  plant not present        -> "No plant detected"
  diseased, disease named  -> that plant and disease (even if our model said something else)
  healthy                  -> that plant, healthy
  anything else / low certainty -> an honest "not a reliable conclusion"; our model's prediction is never shown as confirmed
Any difference between the two is kept in the internal `disagreement` field (administrators only).
"""
import logging
import threading

from pydantic import ValidationError

from app.ai import imaging, prompts, safety
from app.ai.base import AIProvider, ProviderBadResponse, ProviderRateLimited
from app.ai.schemas import AIAnalysis, AnalysisReport, Disagreement
from app.core.config import Settings

log = logging.getLogger("agroai.workflow")

DISCLAIMER = ("AGRO AI guidance is informational only. Verify any treatment against local agricultural guidance, "
              "your extension service and the product label before acting.")
_gate: threading.BoundedSemaphore | None = None
_gate_lock = threading.Lock()


def _provider_gate(settings: Settings) -> threading.BoundedSemaphore:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = threading.BoundedSemaphore(max(settings.ai_max_concurrency, 1))
        return _gate


def reset_gate() -> None:      # tests
    global _gate
    _gate = None


def _ml_label(ml: dict) -> str:
    t = ml["classification_type"]
    if t == "DISEASE":
        return f"{ml.get('crop')}: {ml.get('disease')}"
    if t == "HEALTHY":
        return f"Healthy {ml['crop']}" if ml.get("crop") else "Healthy plant"
    return "Unknown / unsupported" if t == "UNKNOWN" else "No plant detected"


def _guidance(ai: AIAnalysis) -> dict:
    out, removed = {}, 0
    for k in ("immediate_actions", "treatment", "prevention", "monitoring"):
        kept, n = safety.strip_dosages(getattr(ai, k))
        out[k], removed = kept, removed + n
    warnings = list(ai.warnings)
    if removed:
        warnings.append(safety.DOSAGE_NOTE)
    return dict(symptoms=ai.symptoms, severity=ai.severity, affected_percentage=ai.affected_percentage, spread_risk=ai.spread_risk,
                warnings=warnings, ai_notes=ai.ai_notes, identification_confidence=ai.identification_confidence, **out)


def _same_plant(a: str | None, b: str | None) -> bool:
    x, y = (a or "").strip().lower(), (b or "").strip().lower()
    return bool(x and y) and (x in y or y in x)


def _internal_disagreement(ml: dict, ai: AIAnalysis, status: str, shown_plant: str | None, shown_disease: str | None) -> Disagreement | None:
    """Admin-only record of how the AI's assessment differs from our model's. Never rendered for customers."""
    state = ml["classification_type"]
    differs = ai.ml_consistency == "inconsistent"
    if state in ("DISEASE", "HEALTHY"):
        differs = differs or status in ("REJECTED", "UNCERTAIN") or (status == "DISEASE") != (state == "DISEASE") \
            or (shown_plant is not None and not _same_plant(shown_plant, ml.get("crop"))) \
            or (status == "DISEASE" and not _same_plant(shown_disease, ml.get("disease")))
    elif state == "NO_PLANT":
        differs = differs or ai.plant_present
    elif state == "UNKNOWN":
        differs = differs or not ai.plant_present
    if not differs:
        return None
    seen = {"REJECTED": "No plant material visible", "UNCERTAIN": "No reliable conclusion",
            "HEALTHY": f"Healthy {shown_plant or 'plant'}", "DISEASE": f"{shown_plant or 'Plant'}: {shown_disease}"}[status]
    return Disagreement(ml_said=_ml_label(ml), ai_said=seen,
                        message="The AI's visual assessment differs from our first-stage model's result. The assessment shown is the AI's.")


def build_report(ml: dict, ai: AIAnalysis) -> AnalysisReport:
    state = ml["classification_type"]
    base = dict(ml_state=state, disclaimer=DISCLAIMER, **_guidance(ai))
    plant, crop = ai.plant, ai.crop                       # identity comes ONLY from the AI's own look at the image, never from our model

    if not ai.plant_present:
        return AnalysisReport(status="REJECTED", headline="No plant detected", disagreement=_internal_disagreement(ml, ai, "REJECTED", None, None),
                              rejection_reason="We couldn't find a plant, crop or leaf in this image. "
                                               "Please upload a clear, well-lit photo of a leaf or plant.", **base)

    conclusive = ai.identification_confidence != "low"
    if ai.health_status == "diseased" and ai.disease and conclusive:
        return AnalysisReport(status="DISEASE", headline=f"{crop or plant or 'Plant'} — {ai.disease}", plant=plant, crop=crop, disease=ai.disease,
                              disease_source="ai", disagreement=_internal_disagreement(ml, ai, "DISEASE", crop or plant, ai.disease), **base)
    if ai.health_status == "healthy" and conclusive:
        return AnalysisReport(status="HEALTHY", headline=f"{crop or plant or 'Plant'} looks healthy", plant=plant, crop=crop, disease=None,
                              disease_source=None, disagreement=_internal_disagreement(ml, ai, "HEALTHY", crop or plant, None), **base)
    return AnalysisReport(status="UNCERTAIN", headline="We couldn't reach a reliable conclusion", plant=plant, crop=crop, disease=None,
                          disagreement=_internal_disagreement(ml, ai, "UNCERTAIN", crop or plant, None), **base)


def run_guidance(provider: AIProvider, ml: dict, image: bytes, settings: Settings, on_stage=None) -> tuple[AIAnalysis, AnalysisReport, str, dict]:
    """ORIGINAL image + our model's result (as a hint) -> the AI provider -> validated assessment -> final report.
    Called for every analysis, whatever our model's confidence. Returns (ai, report, case, info). Raises ProviderError subclasses; never fabricates.
    `on_stage("guidance")` is called when the provider request starts."""
    jpeg, mime = imaging.prepare_for_provider(image, settings.ai_max_image_side, settings.ml_max_pixels)
    request, case = prompts.build_request(ml, jpeg, mime)
    if on_stage:
        on_stage("guidance")
    gate = _provider_gate(settings)
    if not gate.acquire(timeout=10):
        raise ProviderRateLimited("local provider concurrency limit reached", retry_after=10)
    try:
        response = provider.analyze(request)
    finally:
        gate.release()
    try:
        ai = AIAnalysis.model_validate(response.data)
    except ValidationError as exc:
        raise ProviderBadResponse("answer did not match the schema: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors()[:4])) from exc
    return ai, build_report(ml, ai), case, {"plan_done": ["guidance"]}
