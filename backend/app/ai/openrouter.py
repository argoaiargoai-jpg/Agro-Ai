"""OpenRouter adapter (OpenAI-compatible chat completions), used ONLY as the last AI step after both Gemini keys have failed.

POST {base}/chat/completions, `Authorization: Bearer <key>`, the same prompt and the same original image (as a base64 data URL) that Gemini got; the
response JSON schema is appended to the system text because free routes do not all support a schema parameter. One attempt, no retries. It is NOT
registered in ai/registry.py (admins cannot select it as the active provider); the workflow calls `build()` directly. The key is never logged.
"""
import base64
import json
import logging
import re
import time
from collections.abc import Callable

import httpx

from app.ai import safety
from app.ai.base import (AIProvider, AIRequest, AIResponse, ProviderBadResponse, ProviderMisconfigured, ProviderNotConfigured, ProviderRateLimited,
                         ProviderTimeout, ProviderUnavailable)
from app.core.config import Settings

log = logging.getLogger("agroai.ai.openrouter")


class OpenRouterProvider(AIProvider):
    name = "openrouter"
    display_name = "OpenRouter"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._s, self._transport = settings, transport

    @property
    def model(self) -> str:
        return self._s.openrouter_model

    def is_configured(self) -> bool:
        return bool(self._s.openrouter_api_key.get_secret_value().strip())

    def _safe(self, text: str) -> str:
        return safety.redact(text, self._s.secret_values())[:300]

    def analyze(self, request: AIRequest) -> AIResponse:
        key = self._s.openrouter_api_key.get_secret_value().strip()
        if not key:
            raise ProviderNotConfigured("OPENROUTER_API_KEY is empty")
        system = request.system_instruction + "\n\nReturn ONLY one JSON object (no markdown, no commentary) that matches this JSON Schema:\n" + json.dumps(request.json_schema)
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        if request.image:
            url = f"data:{request.image_mime or 'image/jpeg'};base64,{base64.b64encode(request.image).decode()}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        body = {"model": self._s.openrouter_model, "temperature": 0.2, "max_tokens": 4096,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=httpx.Timeout(self._s.ai_timeout_seconds), transport=self._transport) as c:
                resp = c.post(f"{self._s.openrouter_base_url.rstrip('/')}/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"OpenRouter timeout ({type(exc).__name__})") from None
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"OpenRouter network error ({type(exc).__name__})") from None
        latency = (time.perf_counter() - t0) * 1000
        data = self._json(resp)
        err = data.get("error") if isinstance(data, dict) else None
        status = resp.status_code if resp.status_code != 200 else (int(err.get("code")) if isinstance(err, dict) and str(err.get("code", "")).isdigit() else 200)
        if status != 200 or err:
            self._raise(status if status != 200 else 502, err, resp)
        return self._parse(data, latency, resp)

    def _diag(self, resp: httpx.Response, data, text: str = "") -> str:
        """Short, redacted description of what OpenRouter actually answered (status, type, the model it routed to, why it stopped, a snippet).
        Never includes anything from the request (no image, no key)."""
        choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) and isinstance(data.get("choices"), list) and data.get("choices") else {}
        model = data.get("model") if isinstance(data, dict) else None
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        raw = text or (resp.text if data is None else json.dumps(data.get("error") if isinstance(data, dict) and data.get("error") else ""))
        snippet = re.sub(r"\s+", " ", str(raw or ""))[:240]
        return self._safe(f"HTTP {resp.status_code} {resp.headers.get('content-type', '?').split(';')[0]} model={model} finish={finish} body={snippet!r}")

    @staticmethod
    def _json(resp: httpx.Response):
        try:
            return resp.json()
        except ValueError:
            return None

    def _raise(self, status: int, err, resp: httpx.Response):
        msg = self._safe(str((err or {}).get("message", "")) if isinstance(err, dict) else "")
        log.warning("OpenRouter HTTP %s: %s | %s", status, msg, self._diag(resp, self._json(resp)))
        retry = resp.headers.get("retry-after")
        retry_after = min(int(retry), 3600) if retry and retry.isdigit() else None
        if status in (402, 429):
            raise ProviderRateLimited(f"HTTP {status}: {msg}", retry_after=retry_after)            # credits/quota or rate limit
        if status in (401, 403):
            raise ProviderMisconfigured(f"HTTP {status}: {msg}")
        if status == 408:
            raise ProviderTimeout(f"HTTP 408: {msg}")
        if status >= 500:
            raise ProviderUnavailable(f"HTTP {status}: {msg}")
        raise ProviderBadResponse(f"HTTP {status}: {msg}")

    def _parse(self, data, latency_ms: float, resp: httpx.Response) -> AIResponse:
        def bad(why: str, text: str = ""):
            detail = f"{why} | {self._diag(resp, data, text)}"
            log.warning("OpenRouter answer unusable: %s", detail)
            return ProviderBadResponse(detail)
        try:
            msg = data["choices"][0]["message"]
            text = msg["content"]
        except (TypeError, KeyError, IndexError):
            raise bad("no message in the response") from None
        if isinstance(text, list):                                       # some routes return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip()).strip()
        if not text:
            raise bad("empty answer (content was empty or null)", str(msg.get("reasoning") or "")[:240] if isinstance(msg, dict) else "")
        start, end = text.find("{"), text.rfind("}")
        try:
            parsed = json.loads(text[start:end + 1]) if start != -1 and end > start else None
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise bad("answer was not a JSON object", text)
        u = data.get("usage") or {}
        return AIResponse(data=parsed, model=str(data.get("model") or self._s.openrouter_model), latency_ms=latency_ms,
                          usage={"prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")})

    def ping(self) -> dict:
        """Tiny text-only request (free route) for the admin 'Test connection' button."""
        t0 = time.perf_counter()
        r = self.analyze(AIRequest(system_instruction="Reply with JSON only.", prompt='Return exactly {"ok": true}.',
                                   json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}))
        return {"ok": bool(r.data.get("ok")), "latency_ms": round((time.perf_counter() - t0) * 1000), "model": r.model}


_factory: Callable[[Settings], OpenRouterProvider] = lambda s: OpenRouterProvider(s)


def build(settings: Settings) -> OpenRouterProvider:
    return _factory(settings)
