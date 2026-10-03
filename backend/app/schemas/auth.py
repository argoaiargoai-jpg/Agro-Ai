import re
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints, field_validator

Email = Annotated[EmailStr, Field(max_length=255)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=120)]
OtpCode = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\d{6}$")]


def validate_password_strength(value: str) -> str:
    if len(value) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if len(value.encode()) > 72:
        raise ValueError("Password is too long (max 72 bytes).")
    if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
        raise ValueError("Password must contain at least one letter and one number.")
    return value


class _Normalized(BaseModel):
    @field_validator("email", check_fields=False)
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower()


class RegisterIn(_Normalized):
    email: Email
    password: str
    full_name: Name

    _pw = field_validator("password")(validate_password_strength)


class VerifyOtpIn(_Normalized):
    email: Email
    code: OtpCode


class EmailIn(_Normalized):
    email: Email


class LoginIn(_Normalized):
    email: Email
    password: str = Field(min_length=1, max_length=200)


class ResetPasswordIn(_Normalized):
    email: Email
    code: OtpCode
    new_password: str

    _pw = field_validator("new_password")(validate_password_strength)


class ChangePasswordIn(BaseModel):
    current_password: str | None = None
    new_password: str

    _pw = field_validator("new_password")(validate_password_strength)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    auth_provider: str
    has_password: bool
    avatar_url: str | None = None
    phone: str | None = None
    farm_name: str | None = None
    region: str | None = None
    preferences: dict
    last_login_at: datetime | None = None
    created_at: datetime


class AuthOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class OtpSentOut(BaseModel):
    message: str
    email: str
    resend_in_seconds: int
    dev_otp: str | None = None  # only populated when EXPOSE_DEV_OTP and not production


class MessageOut(BaseModel):
    message: str
