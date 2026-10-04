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
    openrouter_test_mode: StrictBool | None = None            # ADMIN TEST SWITCH: bypass Gemini, send the AI request to OpenRouter


class AITestIn(BaseModel):
    provider: str | None = None
