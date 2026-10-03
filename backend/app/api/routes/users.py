from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.errors import AppError
from app.core.security import verify_password
from app.db.session import get_db
from app.models import User
from app.schemas.auth import MessageOut, UserOut
from app.schemas.user import DeleteAccountIn, PreferencesIn, ProfileUpdateIn
from app.services import auth_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserOut)
def update_me(body: ProfileUpdateIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    for field, value in body.model_dump(exclude_unset=True).items():
        if field == "full_name":
            if value:
                user.full_name = value
        else:
            setattr(user, field, value or None)  # empty string clears the field
    db.commit()
    return user


@router.patch("/me/preferences", response_model=UserOut)
def update_preferences(body: PreferencesIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.preferences = {**user.preferences, **body.model_dump(exclude_none=True)}
    db.commit()
    return user


@router.post("/me/delete", response_model=MessageOut)
def delete_me(body: DeleteAccountIn, response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.has_password and not (body.password and verify_password(body.password, user.password_hash)):
        raise AppError(400, "invalid_credentials", "Password is incorrect.", {"password": "Incorrect password."})
    from app.services import analysis_service
    from app.models import Analysis

    for a in db.query(Analysis).filter(Analysis.user_id == user.id).all():
        (analysis_service.get_settings().upload_path / a.image_path).unlink(missing_ok=True)
    auth_service.revoke_all_sessions(db, user.id)
    db.delete(user)
    db.commit()
    return {"message": "Account deleted."}
