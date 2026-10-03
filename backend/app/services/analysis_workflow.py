"""Turns (our ML result + the AI provider's independent/advisory output) into the final report.

This is a plain mapping, NOT a decision engine: no voting, no confidence averaging. Rules, per ML state:
  DISEASE  -> identity stays OURS (disease_source "ml"); the AI only adds explanation and advice.
  HEALTHY  -> the AI inspects independently. If it sees disease, the AI's disease is shown AND the disagreement is stated.
  UNKNOWN  -> the AI does a full independent analysis; "uncertain" is a valid, honest outcome.
  NO_PLANT -> the AI independently decides; only if it also finds no plant is the image rejected.
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


def build_report(ml: dict, ai: AIAnalysis) -> AnalysisReport:
    state = ml["classification_type"]
    g = _guidance(ai)
    base = dict(ml_state=state, disclaimer=DISCLAIMER, **g)
    ml_text = _ml_label(ml)

    if state == "DISEASE":                                           # identity is OURS; AI = explanation only
        crop, disease = ml.get("crop"), ml.get("disease")
        dis = None
        if ai.ml_consistency == "inconsistent":
            dis = Disagreement(ml_said=ml_text, ai_said="The AI reviewer could not visually confirm this",
                               message="The AI reviewer saw evidence that does not clearly match our model's identification. "
                                       "The identification shown comes from our model; please verify it before acting.")
        return AnalysisReport(status="DISEASE", headline=f"{crop} — {disease}", plant=crop, crop=crop, disease=disease,
                              disease_source="ml", disagreement=dis, **base)

    # HEALTHY / UNKNOWN / NO_PLANT: the AI analysed independently
    plant = ai.plant or (ml.get("crop") if state == "HEALTHY" else None)
    crop = ai.crop or (ml.get("crop") if state == "HEALTHY" else None)

    if not ai.plant_present:
        if state == "NO_PLANT":
            return AnalysisReport(status="REJECTED", headline="This image isn't suitable for plant analysis", disagreement=None,
                                  rejection_reason="We couldn't find a plant, crop or leaf in this image. "
                                                   "Please upload a clear, well-lit photo of a leaf or plant.", **base)
        dis = Disagreement(ml_said=ml_text, ai_said="No plant material visible",
                           message="The AI reviewer found no plant material in this image, which does not match the first-stage result.")
        if state == "UNKNOWN":
            return AnalysisReport(status="REJECTED", headline="This image isn't suitable for plant analysis", disagreement=dis,
                                  rejection_reason="We couldn't find plant material in this image. Please upload a clear photo of a leaf or plant.", **base)
        return AnalysisReport(status="UNCERTAIN", headline="We couldn't reach a reliable conclusion", disagreement=dis, **base)   # ML said healthy

    disagreement = None
    if state == "NO_PLANT":
        disagreement = Disagreement(ml_said=ml_text, ai_said=f"Plant material detected{f': {ai.plant}' if ai.plant else ''}",
                                    message="Our first-stage model did not detect a plant, but the AI reviewer did, so the analysis continued.")

    if ai.health_status == "diseased" and ai.disease:
        if state == "HEALTHY":
            disagreement = Disagreement(ml_said=ml_text, ai_said=ai.disease,
                                        message="The external AI detected a possible disease even though our first-stage model read this plant as healthy. "
                                                "Both views are shown; please inspect the plant and verify.")
        return AnalysisReport(status="DISEASE", headline=f"{crop or plant or 'Plant'} — {ai.disease}", plant=plant, crop=crop, disease=ai.disease,
                              disease_source="ai", disagreement=disagreement, **base)
    if ai.health_status == "healthy":
        return AnalysisReport(status="HEALTHY", headline=f"{crop or plant or 'Plant'} looks healthy", plant=plant, crop=crop, disease=None,
                              disease_source=None, disagreement=disagreement, **base)
    return AnalysisReport(status="UNCERTAIN", headline="We couldn't reach a reliable conclusion", plant=plant, crop=crop, disease=None,
                          disagreement=disagreement, **base)


def run_guidance(provider: AIProvider, ml: dict, image: bytes, settings: Settings) -> tuple[AIAnalysis, AnalysisReport, str]:
    """Image + ML result -> validated AI analysis + final report. Raises ProviderError subclasses; never fabricates."""
    jpeg, mime = imaging.prepare_for_provider(image, settings.ai_max_image_side, settings.ml_max_pixels)
    request, case = prompts.build_request(ml, jpeg, mime)
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
    return ai, build_report(ml, ai), case
