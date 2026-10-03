from typing import Literal

from pydantic import BaseModel, StrictBool

from app.schemas.auth import UserOut


class UserPage(BaseModel):
    items: list[UserOut]
    total: int
    page: int
    page_size: int


class AdminUserUpdateIn(BaseModel):
    role: Literal["user", "admin"] | None = None
    is_active: bool | None = None


class SettingsUpdateIn(BaseModel):
    values: dict


class AIConfigIn(BaseModel):
    enabled: StrictBool | None = None
    provider: str | None = None


class AITestIn(BaseModel):
    provider: str | None = None
