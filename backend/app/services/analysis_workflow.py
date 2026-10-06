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

from app.ai import groq, imaging, pollinations, prompts, safety
from app.ai.base import AIProvider, ProviderBadResponse, ProviderError, ProviderNotConfigured, ProviderRateLimited, ProviderUnavailable
from app.ai.schemas import AIAnalysis, AnalysisReport, Disagreement
from app.core.config import Settings
from app.services import routing

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


SPECIALIST_MIN_PROBABILITY = 0.5          # a specialist disease/health verdict below this is not shown as a finding


def specialist_report(ml: dict, ev: routing.Evidence) -> AnalysisReport | None:
    """The final result when Gemini is unavailable: the LATEST successful specialist result, in the same customer-facing shape
    (no guidance sections: they come from Gemini). None when no specialist produced anything."""
    latest = ev.latest
    if not latest:
        return None
    plants, diseases, healthy = latest["plants"], latest["diseases"], latest["healthy_probability"]
    plant = plants[0].name if plants else None
    base = dict(ml_state=ml["classification_type"], disclaimer=DISCLAIMER, plant=plant, crop=plant, disease_source="specialist",
                ai_notes="Detailed guidance couldn't be generated for this analysis, so this result is based on the automated analysis only.")
    top = diseases[0] if diseases else None
    if latest["kind"] == "disease" and top and top.probability >= SPECIALIST_MIN_PROBABILITY and top.probability >= (healthy or 0.0):
        return AnalysisReport(status="DISEASE", headline=f"{plant or 'Plant'} — {top.name}", disease=top.name, **base)
    if latest["kind"] == "disease" and healthy is not None and healthy >= SPECIALIST_MIN_PROBABILITY:
        return AnalysisReport(status="HEALTHY", headline=f"{plant or 'Plant'} looks healthy", **{**base, "disease_source": None})
    return AnalysisReport(status="UNCERTAIN", headline="We couldn't reach a reliable conclusion", **{**base, "disease_source": None})


def _validated(response) -> AIAnalysis:
    try:
        return AIAnalysis.model_validate(response.data)
    except ValidationError as exc:
        raise ProviderBadResponse("answer did not match the schema: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors()[:4])) from exc


def _fallback_chain(request, evidence: routing.Evidence, settings: Settings):
    """The generative fallbacks, in order: Groq, then Pollinations. Each gets the SAME request Gemini got (same image, same prompt) and is tried
    once (with its own bounded retries). A provider that is not configured, errors, or returns an unusable answer (empty / not JSON / not our
    schema) counts as failed and the next one is tried. Returns (AIAnalysis | None, provider, last error | None, model | None); `provider` is the
    one that answered, or the last one tried. Every outcome is recorded for the admin; nothing here ever contains a key."""
    from app.ai.safety import redact
    last_provider, last_err = None, None
    for build in (groq.build, pollinations.build):
        fb = build(settings)
        last_provider = fb
        if not fb.is_configured():
            evidence.providers.append({"provider": fb.name, "step": "guidance", "status": "not_configured"})
            last_err = ProviderNotConfigured(f"{fb.env_name} is not set on the server")
            log.info("%s is not configured; skipping", fb.display_name)
            continue
        try:
            response = fb.analyze(request)
            ai = _validated(response)
        except ProviderError as exc:
            evidence.providers.append({"provider": fb.name, "step": "guidance", "status": exc.ai_status})
            log.warning("%s fallback failed (%s): %s", fb.display_name, exc.ai_status, redact(exc.detail, settings.secret_values()))
            last_err = exc
            continue
        evidence.providers.append({"provider": fb.name, "step": "guidance", "status": "ok"})
        return ai, fb, None, response.model
    return None, last_provider, last_err, None


def run_guidance(provider: AIProvider, ml: dict, image: bytes, settings: Settings, on_stage=None, bypass_gemini: bool = False) -> tuple[AIAnalysis | None, AnalysisReport, str, dict]:
    """Our model's confidence picks the specialist path (routing.py); then Gemini ALWAYS gets the ORIGINAL image, our model's result as a hint and
    whatever evidence the specialists produced. If Gemini fails (both keys, after their retries), Groq is tried, then Pollinations (same request); if
    those fail too, the latest successful specialist result is returned as the final result (ai is None then); with no specialist result the
    ProviderError propagates (existing behaviour: preliminary result + retry). With `bypass_gemini` (admin switch) Gemini is skipped and the chain
    starts at Groq. Returns (ai, report, case, info); info is internal bookkeeping. `on_stage(name)` is called as each real step starts."""
    jpeg, mime = imaging.prepare_for_provider(image, settings.ai_max_image_side, settings.ml_max_pixels)
    route = routing.decide(ml, settings)
    evidence = routing.gather(route, jpeg, mime, settings, on_stage)         # the same re-encoded image bytes go to every visual provider
    request, case = prompts.build_request(ml, jpeg, mime, evidence.internal())
    info = {"route": route, "plan_done": list(evidence.steps_done), "specialists": evidence.internal()}
    if on_stage:
        on_stage("guidance")
    if bypass_gemini:                                # ADMIN SWITCH: Gemini is skipped; Groq -> Pollinations -> specialist result
        log.warning("Gemini bypass enabled: skipping Gemini")
        evidence.providers.append({"provider": provider.name, "step": "guidance", "status": "bypassed"})
        ai, fb, err, routed = _fallback_chain(request, evidence, settings)
        info.update(test_mode=True, specialists=evidence.internal())
        if ai is not None:
            info.update(plan_done=evidence.steps_done + ["guidance"], ai_used={"provider": fb.name, "model": routed or fb.model})
            return ai, build_report(ml, ai), case, info
        info["ai_attempted"] = {"provider": fb.name, "model": fb.model}                    # admin: what was actually tried (not Gemini)
        report = specialist_report(ml, evidence)
        if report is None:
            raise err                                                                      # the real fallback error: existing preliminary result + Retry
        return None, report, "SPECIALIST_ONLY", {**info, "gemini_error": err.ai_status, "gemini_error_class": type(err).__name__}
    try:
        if not provider.is_configured():
            raise ProviderNotConfigured("no key configured for the AI provider")
        gate = _provider_gate(settings)
        if not gate.acquire(timeout=10):
            raise ProviderRateLimited("local provider concurrency limit reached", retry_after=10)
        try:
            response = provider.analyze(request)
        finally:
            gate.release()
        ai = _validated(response)
    except ProviderError as exc:
        evidence.providers.append({"provider": provider.name, "step": "guidance", "status": exc.ai_status})
        ai, fallback, err, model = _fallback_chain(request, evidence, settings)                 # Gemini failed: Groq, then Pollinations
        if ai is not None:
            log.warning("AI guidance came from %s because Gemini failed (%s)", fallback.display_name, exc.ai_status)
            info.update(plan_done=evidence.steps_done + ["guidance"], specialists=evidence.internal(), ai_used={"provider": fallback.name, "model": model or fallback.model})
            return ai, build_report(ml, ai), case, info
        info["specialists"] = evidence.internal()
        report = specialist_report(ml, evidence)
        if report is None:
            raise
        log.warning("AI guidance unavailable (%s); returning the %s result instead", exc.ai_status, evidence.latest["provider"])
        return None, report, "SPECIALIST_ONLY", {**info, "gemini_error": exc.ai_status, "gemini_error_class": type(exc).__name__}
    evidence.providers.append({"provider": provider.name, "step": "guidance", "status": "ok"})
    info.update(plan_done=evidence.steps_done + ["guidance"], specialists=evidence.internal())
    return ai, build_report(ml, ai), case, info
