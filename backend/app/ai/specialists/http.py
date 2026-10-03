"""Shared HTTP plumbing: sanitised errors (URLs can carry API keys, so exception text is never forwarded)."""
import re

import httpx

from app.ai.specialists.base import SpecialistError


def call(method: str, url: str, *, timeout: float, transport: httpx.BaseTransport | None = None, ok=(200,), allow=(), **kw) -> httpx.Response:
    """Performs the request. `allow` = extra statuses returned to the caller (e.g. 404 = nothing found). Everything else maps to SpecialistError."""
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout), transport=transport) as c:
            r = c.request(method, url, **kw)
    except httpx.TimeoutException as exc:
        raise SpecialistError("timeout", type(exc).__name__) from None
    except httpx.TransportError as exc:
        raise SpecialistError("unavailable", type(exc).__name__) from None
    if r.status_code in ok or r.status_code in allow:
        return r
    if r.status_code in (401, 403):
        raise SpecialistError("unauthorized", f"HTTP {r.status_code}")
    if r.status_code == 429:
        raise SpecialistError("rate_limited", "HTTP 429")
    if r.status_code >= 500:
        raise SpecialistError("unavailable", f"HTTP {r.status_code}")
    snippet = re.sub(r"\s+", " ", r.text or "")[:160]                       # what the provider said, truncated; the caller redacts secrets before logging
    raise SpecialistError("bad_response", f"HTTP {r.status_code} {r.headers.get('content-type', '?').split(';')[0]} {snippet!r}")


def json_of(r: httpx.Response) -> dict:
    try:
        d = r.json()
    except ValueError:
        raise SpecialistError("bad_response", "not JSON") from None
    if not isinstance(d, dict):
        raise SpecialistError("bad_response", "unexpected JSON shape")
    return d
