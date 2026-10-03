import uuid

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import User
from app.models.user import Role
from app.schemas.analysis import AnalysisOut, AnalysisPage, present
from app.schemas.auth import MessageOut
from app.services import analysis_service

router = APIRouter(prefix="/analyses", tags=["analyses"])


def _admin(user: User) -> bool:
    return user.role == Role.admin.value


@router.post("", response_model=AnalysisOut, status_code=201)
def create_analysis(
    file: UploadFile = File(...),
    crop_type: str | None = Form(None),
    source: str = Form("upload"),
    notes: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return present(analysis_service.create(db, user, file, crop_type, source, notes), _admin(user))


@router.get("", response_model=AnalysisPage)
def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    status: str | None = Query(None, pattern="^(uploaded|queued|processing|partial|completed|failed)$"),
    q: str | None = Query(None, max_length=80),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = analysis_service.list_for_user(db, user, page, page_size, status, q)
    return {"items": [present(a, _admin(user)) for a in items], "total": total, "page": page, "page_size": page_size}


@router.get("/summary")
def summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = analysis_service.summary_for_user(db, user)
    data["recent"] = [present(a, _admin(user)) for a in data["recent"]]
    return data


@router.post("/{analysis_id}/analyze", response_model=AnalysisOut)
def analyze(
    analysis_id: uuid.UUID,
    force: bool = Query(False, description="Re-run even if a result already exists"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Full analysis: our ML model (DISEASE | HEALTHY | UNKNOWN | NO_PLANT), then the selected AI provider for guidance."""
    return present(analysis_service.run_analysis(db, user, analysis_id, force), _admin(user))


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return present(analysis_service.get_owned(db, user, analysis_id), _admin(user))


@router.get("/{analysis_id}/image")
def get_image(analysis_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = analysis_service.get_owned(db, user, analysis_id)
    return FileResponse(analysis_service.image_file(a), media_type=a.content_type,
                        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"})


@router.delete("/{analysis_id}", response_model=MessageOut)
def delete_analysis(analysis_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analysis_service.delete(db, user, analysis_id)
    return {"message": "Analysis deleted."}
