"""Pure-ASGI middleware: reject oversized uploads BEFORE they are buffered, and add basic security headers."""
import json

from app.core.config import get_settings


class _TooLarge(Exception):
    pass


async def _send_json(send, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class MaxUploadSizeMiddleware:
    """Starlette parses a whole multipart body before our route runs, so without this a huge upload would be written to
    temp storage first. We cut it off by Content-Length, and by counting bytes when there is no Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        s = get_settings()
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != f"{s.api_prefix}/analyses":
            return await self.app(scope, receive, send)
        limit = (s.max_upload_mb + 1) * 1024 * 1024            # +1 MB for multipart framing and the small form fields
        msg = f"Image exceeds the {s.max_upload_mb} MB limit."
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return await _send_json(send, 413, "file_too_large", msg)
        seen = 0
        overflow = False

        async def counting_receive():
            nonlocal seen, overflow
            m = await receive()
            if m["type"] == "http.request":
                seen += len(m.get("body", b""))
                if seen > limit:
                    overflow = True
                    raise _TooLarge()
            return m

        async def guarded_send(m):
            if not overflow:                    # FastAPI turns any body-parsing exception into its own 400; discard that, we answer 413
                await send(m)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _TooLarge:
            pass
        if overflow:
            await _send_json(send, 413, "file_too_large", msg)


class SecurityHeadersMiddleware:
    HEADERS = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"), (b"referrer-policy", b"no-referrer")]

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def wrapped(m):
            if m["type"] == "http.response.start":
                have = {k.lower() for k, _ in m["headers"]}
                m["headers"] = list(m["headers"]) + [h for h in self.HEADERS if h[0] not in have]
            await send(m)

        await self.app(scope, receive, wrapped)
