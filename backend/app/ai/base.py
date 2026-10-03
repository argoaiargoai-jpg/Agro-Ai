"""The contract every external AI provider implements, plus provider-neutral errors.

The workflow never imports a vendor module; it asks the registry for an `AIProvider` and calls `analyze()`.
Errors carry a short, user-safe message. Internal detail (already stripped of secrets) is for server logs only.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AIRequest:
    system_instruction: str
    prompt: str
    json_schema: dict
    image: bytes | None = None        # already re-encoded/stripped of metadata by ai.imaging
    image_mime: str | None = None


@dataclass
class AIResponse:
    data: dict                        # parsed JSON object (NOT yet validated against our schema)
    model: str
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)


class ProviderError(Exception):
    ai_status = "unavailable"         # stored on the analysis; drives the UI
    retryable = True
    user_message = "Detailed guidance is temporarily unavailable."

    def __init__(self, detail: str = "", *, retry_after: int | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail          # internal only (logs); callers must never put secrets here
        self.retry_after = retry_after


class ProviderNotConfigured(ProviderError):
    ai_status, retryable = "not_configured", False
    user_message = "Detailed guidance isn't available right now."


class ProviderMisconfigured(ProviderError):
    """Key rejected / model not found: a server configuration problem, not an outage."""
    ai_status, retryable = "misconfigured", False
    user_message = "Detailed guidance is temporarily unavailable."


class ProviderTimeout(ProviderError):
    ai_status = "timeout"
    user_message = "Detailed guidance is taking longer than expected."


class ProviderRateLimited(ProviderError):
    ai_status = "rate_limited"
    user_message = "Detailed guidance is busy right now. Please try again shortly."


class ProviderUnavailable(ProviderError):
    ai_status = "unavailable"
    user_message = "Detailed guidance is temporarily unavailable."


class ProviderBadResponse(ProviderError):
    ai_status = "bad_response"
    user_message = "We couldn't generate usable guidance this time."


class ProviderBlocked(ProviderError):
    ai_status, retryable = "blocked", False
    user_message = "We couldn't generate guidance for this image."


class AIProvider(ABC):
    name: str = ""
    display_name: str = ""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def is_configured(self) -> bool:
        """True when the secrets/config needed to call the provider exist. Never exposes the secret itself."""

    @abstractmethod
    def analyze(self, request: AIRequest) -> AIResponse:
        """Send the request and return parsed JSON. Raise a ProviderError subclass on any failure."""

    @abstractmethod
    def ping(self) -> dict:
        """Tiny text-only health check used by the admin 'Test connection' button."""
