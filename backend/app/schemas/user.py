from typing import Literal

from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated

Trimmed = Annotated[str, StringConstraints(strip_whitespace=True)]


class ProfileUpdateIn(BaseModel):
    full_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=120)] | None = None
    phone: Annotated[str, StringConstraints(strip_whitespace=True, max_length=32, pattern=r"^[0-9+()\-\s]*$")] | None = None
    farm_name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] | None = None
    region: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] | None = None


class PreferencesIn(BaseModel):
    language: Literal["en", "hi", "te", "es", "fr"] | None = None
    units: Literal["metric", "imperial"] | None = None
    email_notifications: bool | None = None
    weekly_summary: bool | None = None


class DeleteAccountIn(BaseModel):
    password: str | None = Field(default=None, max_length=200)
