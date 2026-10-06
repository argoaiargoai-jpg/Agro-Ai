"""Shared retry policy for the generative providers: bounded exponential backoff on transient failures only."""
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
BACKOFF_BASE, BACKOFF_CAP = 1.0, 20.0


def delay(attempt: int, retry_after: int | None = None) -> float:
    """Seconds to wait before retry number `attempt` (1-based): 1s, 2s, 4s ... capped; a provider's Retry-After is honoured up to the cap."""
    d = min(BACKOFF_BASE * 2 ** (attempt - 1), BACKOFF_CAP)
    return min(max(d, float(retry_after or 0)), BACKOFF_CAP)
