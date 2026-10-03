import logging
import re
import uuid
import warnings
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from datetime import timedelta

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.models import Analysis, AnalysisStatus, User
from app.ai import registry
from app.ai.base import ProviderError
from app.core.security import utcnow
from app.services import analysis_workflow, ml_service, settings_service

CHUNK = 1024 * 256
log = logging.getLogger("agroai.analysis")


def sniff_image(head: bytes) -> tuple[str, str] | None:
    """Return (content_type, extension) from magic bytes, ignoring client-supplied type."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp", "webp"
    return None


ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MIN_SIDE = ml_service.MIN_SIDE


def safe_filename(name: str | None) -> str | None:
    """The original name is only ever stored as text for display (files on disk use random names). Strip paths and control chars."""
    if not name:
        return None
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f\u202a-\u202e\u2066-\u2069]", "", name)     # control chars and bidi overrides (spoofed extensions)
    name = " ".join(name.split())[:255]
    return name or None


def validate_image_file(path: Path, expected_content_type: str, max_pixels: int) -> None:
    """Reject corrupt, mislabeled, tiny and decompression-bomb images at upload time. Raises AppError; caller deletes the file.

    The pixel count is read from the header BEFORE any pixel data is decoded, so a small file that claims to be
    50,000 x 50,000 pixels is refused without allocating memory for it."""
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                fmt, (w, h) = im.format, im.size
                if ALLOWED_FORMATS.get(fmt) != expected_content_type:
                    raise AppError(415, "unsupported_media", "Only JPEG, PNG or WebP images are supported.", {"file": "Unsupported image type."})
                if w * h > max_pixels:
                    raise AppError(413, "image_too_large", f"That image is too large ({w} x {h} px). Please use a smaller photo.", {"file": "Image dimensions too large."})
                if min(w, h) < MIN_SIDE:
                    raise AppError(422, "image_too_small", f"The image is too small to analyze (minimum {MIN_SIDE}px on the short side).", {"file": "Image too small."})
                im.load()                                    # full decode: catches truncated/corrupt data
    except AppError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:      # Pillow refuses these from the header, before decoding
        raise AppError(413, "image_too_large", "That image has too many pixels. Please use a smaller photo.", {"file": "Image dimensions too large."}) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, MemoryError) as exc:
        raise AppError(422, "invalid_image", "We couldn't read that image. It may be corrupt or incomplete.", {"file": "Unreadable image."}) from exc


def clean_crop_name(value: str | None) -> str | None:
    """The plant/crop hint is free text now (any plant may be analysed): trimmed, single-spaced, printable, at most 60 characters."""
    if not value:
        return None
    text = re.sub(r"\s+", " ", "".join(ch for ch in value if ch.isprintable() or ch.isspace())).strip()
    if len(text) > 60:
        raise AppError(422, "validation_error", "That plant name is too long.", {"crop_type": "Use at most 60 characters."})
    return text or None


def create(db: Session, user: User, file: UploadFile, crop_type: str | None, source: str, notes: str | None) -> Analysis:
    if settings_service.get_value(db, "maintenance_mode"):
        raise AppError(503, "maintenance", "AGRO AI is in maintenance mode. Please try again later.")
    if source not in ("upload", "camera"):
        raise AppError(422, "validation_error", "Invalid source.", {"source": "Must be upload or camera."})
    crop_type = clean_crop_name(crop_type)
    if notes and len(notes) > 1000:
        raise AppError(422, "validation_error", "Notes too long.", {"notes": "Max 1000 characters."})

    limit = settings_service.effective_max_upload_bytes(db)
    s = get_settings()
    folder = s.upload_path / str(user.id)
    folder.mkdir(parents=True, exist_ok=True)

    first = file.file.read(CHUNK)
    if not first:
        raise AppError(422, "empty_file", "The uploaded file is empty.", {"file": "File is empty."})
    sniffed = sniff_image(first)
    if sniffed is None:
        raise AppError(415, "unsupported_media", "Only JPEG, PNG or WebP images are supported.", {"file": "Unsupported image type."})
    content_type, ext = sniffed

    dest: Path = folder / f"{uuid.uuid4().hex}.{ext}"
    size = 0
    try:
        with dest.open("wb") as out:
            chunk = first
            while chunk:
                size += len(chunk)
                if size > limit:
                    raise AppError(413, "file_too_large", f"Image exceeds the {limit // (1024 * 1024)} MB limit.", {"file": "File too large."})
                out.write(chunk)
                chunk = file.file.read(CHUNK)
        validate_image_file(dest, content_type, s.ml_max_pixels)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    analysis = Analysis(
        user_id=user.id,
        crop_type=crop_type or None,
        source=source,
        image_path=str(dest.relative_to(s.upload_path)),
        original_filename=safe_filename(file.filename),
        content_type=content_type,
        size_bytes=size,
        notes=(notes or None),
    )
    db.add(analysis)
    db.commit()
    return analysis


def list_for_user(db: Session, user: User, page: int, page_size: int, status: str | None, q: str | None):
    stmt = select(Analysis).where(Analysis.user_id == user.id)
    if status:
        stmt = stmt.where(Analysis.status == status)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where((Analysis.crop_type.ilike(like)) | (Analysis.original_filename.ilike(like)))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(stmt.order_by(Analysis.created_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return items, total


def get_owned(db: Session, user: User, analysis_id: uuid.UUID) -> Analysis:
    a = db.get(Analysis, analysis_id)
    if a is None or a.user_id != user.id:
        raise AppError(404, "not_found", "Analysis not found.")
    return a


def delete(db: Session, user: User, analysis_id: uuid.UUID) -> None:
    a = get_owned(db, user, analysis_id)
    (get_settings().upload_path / a.image_path).unlink(missing_ok=True)
    db.delete(a)
    db.commit()


def image_file(a: Analysis) -> Path:
    path = (get_settings().upload_path / a.image_path).resolve()
    if get_settings().upload_path not in path.parents or not path.exists():
        raise AppError(404, "not_found", "Image file is missing.")
    return path


OUTCOME_OF_ML = {"DISEASE": "disease", "HEALTHY": "healthy", "UNKNOWN": "unresolved", "NO_PLANT": "no_plant"}
OUTCOME_OF_FINAL = {"DISEASE": "disease", "HEALTHY": "healthy", "UNCERTAIN": "unresolved", "REJECTED": "no_plant"}
SUMMARY_WINDOW = 1000          # newest analyses considered for the distribution charts (bounded work per request)


def outcome_of(result: dict | None) -> str | None:
    """Customer-level outcome of an analysis: healthy | disease | unresolved | no_plant (None until a result exists)."""
    if not result:
        return None
    final, ml = result.get("final"), result.get("ml")
    if final and final.get("status") in OUTCOME_OF_FINAL:
        return OUTCOME_OF_FINAL[final["status"]]
    if ml and ml.get("classification_type") in OUTCOME_OF_ML:
        return OUTCOME_OF_ML[ml["classification_type"]]
    return None


def _top(counter: dict, n: int = 6) -> list[dict]:
    return [{"name": k, "count": v} for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def aggregate(window) -> dict:
    """Distribution numbers from (created_at, result, ai_status, crop_type) rows. Real data only: nothing is padded or estimated."""
    outcomes = {"healthy": 0, "disease": 0, "unresolved": 0, "no_plant": 0}
    diseases: dict[str, int] = {}
    crops: dict[str, int] = {}
    ai_assisted = 0
    day_counts: dict[str, int] = {}
    for created, result, ai_status, crop_type in window:
        if created:
            day = created.date().isoformat()
            day_counts[day] = day_counts.get(day, 0) + 1
        out = outcome_of(result)
        if out is None:
            continue
        outcomes[out] += 1
        if ai_status == "completed":
            ai_assisted += 1
        final, ml = (result or {}).get("final") or {}, (result or {}).get("ml") or {}
        crop = final.get("crop") or final.get("plant") or ml.get("crop") or crop_type
        if crop and out in ("healthy", "disease"):
            crops[crop] = crops.get(crop, 0) + 1
        disease = final.get("disease") or ml.get("disease")
        if out == "disease" and disease:
            diseases[disease] = diseases.get(disease, 0) + 1
    today = utcnow().date()
    activity = [{"date": (today - timedelta(days=13 - i)).isoformat(), "count": day_counts.get((today - timedelta(days=13 - i)).isoformat(), 0)}
                for i in range(14)]
    return {"outcomes": outcomes, "ai_assisted": ai_assisted, "top_diseases": _top(diseases), "crops": _top(crops), "activity_14d": activity}


def summary_for_user(db: Session, user: User) -> dict:
    """Real numbers only, derived from this user's stored analyses (nothing is estimated or padded)."""
    rows = db.execute(
        select(Analysis.status, func.count()).where(Analysis.user_id == user.id).group_by(Analysis.status)
    ).all()
    by_status = {k: v for k, v in rows}
    recent = db.scalars(
        select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).limit(5)
    ).all()

    window = db.execute(
        select(Analysis.created_at, Analysis.result, Analysis.ai_status, Analysis.crop_type)
        .where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).limit(SUMMARY_WINDOW)
    ).all()
    agg = aggregate(window)
    return {"total": sum(by_status.values()), "by_status": by_status, "recent": recent, **agg}


# ----------------------------------------------------------------------------------------------------------------------
# Full analysis:  image -> our ML model -> DISEASE/HEALTHY/UNKNOWN/NO_PLANT -> selected AI provider -> final report
# ----------------------------------------------------------------------------------------------------------------------
STALE_PROCESSING = timedelta(minutes=5)     # a crashed worker must not lock an analysis forever
AI_RETRY_COOLDOWN = timedelta(seconds=10)   # blocks hammering the provider by re-clicking "retry"
AI_FORCE_COOLDOWN = timedelta(seconds=60)   # a FORCED re-run of the same analysis (costs a fresh provider call) is allowed at most once a minute

AI_MESSAGES = {
    "disabled": "Detailed guidance is switched off, so only the basic result is shown.",
    "not_configured": "Detailed guidance isn't available, so only the basic result is shown.",
}


def _claim(db: Session, a: Analysis) -> None:
    """Atomically move to 'processing'. Two simultaneous requests cannot both win (duplicate-request protection)."""
    stale = utcnow() - STALE_PROCESSING
    res = db.execute(update(Analysis).where(Analysis.id == a.id, or_(Analysis.status != AnalysisStatus.processing.value, Analysis.updated_at < stale))
                     .values(status=AnalysisStatus.processing.value, error_message=None, updated_at=utcnow()))
    db.commit()
    if res.rowcount == 0:
        raise AppError(409, "already_processing", "This image is already being analyzed.")
    db.refresh(a)


def _result(ml, ai=None, final=None, ai_error=None, stage="ml_done", case=None, plan=None, info=None) -> dict:
    """`plan` = customer-safe step names that will run or did run (always ["guidance"] after our model). `info` is kept for compatibility."""
    out = {"ml": ml, "ai": ai, "final": final, "ai_error": ai_error, "stage": stage, "plan": plan or ["guidance"],
           "prompt_version": analysis_workflow.prompts.PROMPT_VERSION, "ai_case": case}
    return out


def _planned(ml: dict) -> list[str]:
    return ["guidance"]                      # every analysis: our model, then the AI's independent look at the original image


def _ml_only(db: Session, a: Analysis, ml: dict, ai_status: str) -> Analysis:
    a.result = _result(ml, ai_error={"code": ai_status, "message": AI_MESSAGES[ai_status], "retryable": False}, stage="ml_only", plan=[])
    a.status, a.ai_status, a.ai_error_code = AnalysisStatus.completed.value, ai_status, None
    db.commit()
    return a


def _guidance_failed(db: Session, a: Analysis, ml: dict, ai_status: str, message: str, retryable: bool, retry_after: int | None, code: str) -> Analysis:
    a.result = _result(ml, ai_error={"code": ai_status, "message": message, "retryable": retryable, "retry_after": retry_after}, stage="guidance_failed",
                       plan=(a.result or {}).get("plan"))
    a.status, a.ai_status, a.ai_error_code = AnalysisStatus.partial.value, ai_status, code
    db.commit()
    return a


def _user_limit_reached(db: Session, user: User, a: Analysis, limit: int) -> int | None:
    """Returns seconds until the oldest counted call leaves the 1-hour window, or None if under the limit."""
    since = utcnow() - timedelta(hours=1)
    times = db.scalars(select(Analysis.ai_attempted_at).where(Analysis.user_id == user.id, Analysis.ai_attempted_at >= since,
                                                              Analysis.id != a.id).order_by(Analysis.ai_attempted_at)).all()
    if len(times) < limit:
        return None
    return max(int((times[0] + timedelta(hours=1) - utcnow()).total_seconds()), 1)


def _ai_stage(db: Session, user: User, a: Analysis, ml: dict, image: bytes, forced: bool = False) -> Analysis:
    settings = get_settings()
    if not settings_service.get_value(db, "ai_enabled"):
        return _ml_only(db, a, ml, "disabled")
    name = settings_service.get_value(db, "ai_provider")
    try:
        provider = registry.build(name, settings)
    except KeyError:
        return _guidance_failed(db, a, ml, "misconfigured", "Detailed guidance is temporarily unavailable.", False, None, "unknown_provider")
    a.ai_provider, a.ai_model = provider.name, provider.model
    if not provider.is_configured():
        return _ml_only(db, a, ml, "not_configured")
    now = utcnow()
    cooldown = AI_FORCE_COOLDOWN if forced else AI_RETRY_COOLDOWN
    if a.ai_attempted_at and now - a.ai_attempted_at < cooldown:
        wait = int((cooldown - (now - a.ai_attempted_at)).total_seconds()) + 1
        return _guidance_failed(db, a, ml, "rate_limited", f"Please wait {wait}s before retrying.", True, wait, "retry_cooldown")
    wait = _user_limit_reached(db, user, a, settings.ai_user_hourly_limit)
    if wait:
        return _guidance_failed(db, a, ml, "user_limit", f"You've reached the limit of {settings.ai_user_hourly_limit} guidance requests per hour. Please try again later.", True, wait, "user_hourly_limit")
    a.ai_attempted_at = now
    db.commit()
    a.result = _result(ml, plan=_planned(ml))                                    # (re)announce what will really run, for a retry as well
    db.commit()

    def on_stage(name: str) -> None:                                             # committed so the UI follows the REAL backend step
        a.result = {**(a.result or {}), "stage": name}
        db.commit()
    try:
        ai, report, case, info = analysis_workflow.run_guidance(provider, ml, image, settings, on_stage)
    except ProviderError as exc:
        log.warning("analysis %s: AI guidance failed (%s): %s", a.id, exc.ai_status, exc.detail)
        return _guidance_failed(db, a, ml, exc.ai_status, exc.user_message, exc.retryable, exc.retry_after, type(exc).__name__)
    a.result = _result(ml, ai=ai.model_dump(), final=report.model_dump(), stage="complete", case=case, plan=info["plan_done"], info=info)
    a.status, a.ai_status, a.ai_error_code, a.ai_completed_at = AnalysisStatus.completed.value, "completed", None, utcnow()
    db.commit()
    return a


def run_analysis(db: Session, user: User, analysis_id: uuid.UUID, force: bool = False) -> Analysis:
    a = get_owned(db, user, analysis_id)
    if settings_service.get_value(db, "maintenance_mode"):
        raise AppError(503, "maintenance", "AGRO AI is in maintenance mode. Please try again later.")
    if a.status == AnalysisStatus.completed.value and a.result and not force:
        return a
    prior_ml = (a.result or {}).get("ml") if a.status == AnalysisStatus.partial.value else None
    reuse_ml = bool(prior_ml) and not force                    # retrying failed guidance never re-runs (or changes) the ML result
    service = None if reuse_ml else ml_service.get_ml_service()  # 503 if the model isn't installed; the upload stays retryable
    _claim(db, a)
    try:
        data = image_file(a).read_bytes()
        if reuse_ml:
            ml = prior_ml
        else:
            ml = service.predict(data)
            a.result = _result(ml, plan=_planned(ml))           # stage 1 is committed first, so the UI can show a REAL stage change
            a.confidence, a.ml_completed_at = ml["confidence"], utcnow()
            a.ai_status = a.ai_error_code = a.ai_completed_at = None      # ai_attempted_at is kept: the cooldown must see earlier attempts
            db.commit()
    except AppError as exc:
        a.status, a.error_message = AnalysisStatus.failed.value, exc.message
        db.commit()
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("ML inference failed for analysis %s", a.id)
        a.status, a.error_message = AnalysisStatus.failed.value, "The analysis failed unexpectedly. Please try again."
        db.commit()
        raise AppError(500, "ml_failed", a.error_message) from exc
    try:
        return _ai_stage(db, user, a, ml, data, forced=force)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001  (a bug in the guidance stage must not lose the ML result)
        log.exception("Unexpected error in AI stage for analysis %s", a.id)
        return _guidance_failed(db, a, ml, "unavailable", "Detailed guidance is temporarily unavailable.", True, None, type(exc).__name__)
