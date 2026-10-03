"""Google Gemini adapter (REST `generateContent`). ALL Gemini-specific knowledge lives in this file.

* The API key is sent in the `x-goog-api-key` header (never in the URL, so it can't leak into access logs).
* The key is read from settings (a SecretStr) only here, and never put in an exception, log line or response.
* Errors are mapped to provider-neutral ProviderError subclasses.
"""
import base64
import json
import logging
import re
import time

import httpx

from app.ai import safety
from app.ai.base import (AIProvider, AIRequest, AIResponse, ProviderBadResponse, ProviderBlocked, ProviderMisconfigured,
                         ProviderNotConfigured, ProviderRateLimited, ProviderTimeout, ProviderUnavailable)
from app.core.config import Settings

log = logging.getLogger("agroai.ai.gemini")
BLOCKING_FINISH = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}
_SLEEP = time.sleep   # patched in tests


class GeminiProvider(AIProvider):
    name = "gemini"
    display_name = "Google Gemini"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._s = settings
        self._transport = transport

    @property
    def model(self) -> str:
        return self._s.gemini_model

    def is_configured(self) -> bool:
        return self._s.gemini_configured

    # ------------------------------------------------------------------ public
    def analyze(self, request: AIRequest) -> AIResponse:
        parts: list[dict] = [{"text": request.prompt}]
        if request.image:
            parts.append({"inlineData": {"mimeType": request.image_mime or "image/jpeg", "data": base64.b64encode(request.image).decode()}})
        return self._generate(request.system_instruction, parts, request.json_schema)

    def ping(self) -> dict:
        t0 = time.perf_counter()
        r = self._generate("Reply with JSON only.", [{"text": 'Return exactly {"ok": true}.'}],
                           {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}, max_tokens=64)
        return {"ok": bool(r.data.get("ok")), "latency_ms": round((time.perf_counter() - t0) * 1000), "model": r.model}

    # ------------------------------------------------------------------ internals
    def _secrets(self) -> list[str]:
        return [self._s.gemini_api_key.get_secret_value()]

    def _safe(self, text: str) -> str:
        return safety.redact(text, self._secrets())[:300]

    def _generate(self, system: str, parts: list[dict], schema: dict | None, max_tokens: int = 8192) -> AIResponse:
        key = self._s.gemini_api_key.get_secret_value().strip()
        if not key:
            raise ProviderNotConfigured("GEMINI_API_KEY is empty")
        url = f"{self._s.gemini_base_url.rstrip('/')}/v1beta/models/{self._s.gemini_model}:generateContent"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        use_schema = schema is not None
        attempts = 1 + max(self._s.ai_max_retries, 0)
        last: Exception | None = None
        i = 0
        while i < attempts:
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens, "responseMimeType": "application/json",
                                     **({"responseJsonSchema": schema} if use_schema else {})},
            }
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=httpx.Timeout(self._s.ai_timeout_seconds), transport=self._transport) as c:
                    resp = c.post(url, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                last = ProviderTimeout(f"timeout after {self._s.ai_timeout_seconds}s ({type(exc).__name__})")
            except httpx.TransportError as exc:
                last = ProviderUnavailable(f"network error ({type(exc).__name__})")
            else:
                latency = (time.perf_counter() - t0) * 1000
                status = resp.status_code
                if status == 200:
                    return self._parse(resp, latency)
                err_msg = self._error_message(resp)
                log.warning("Gemini HTTP %s: %s", status, self._safe(err_msg))
                if status == 429:
                    raise ProviderRateLimited(err_msg and self._safe(err_msg), retry_after=self._retry_after(resp))
                if status in (401, 403, 404) or (status == 400 and re.search(r"api key|API_KEY|permission|not found|not supported for generateContent", err_msg, re.I)):
                    raise ProviderMisconfigured(f"HTTP {status}: {self._safe(err_msg)}")
                if status == 400 and use_schema and re.search(r"schema|response_json_schema|responseJsonSchema|response_schema", err_msg, re.I):
                    use_schema = False            # model rejected our schema dialect: retry once relying on the prompt + our validation
                    log.warning("Gemini rejected the response schema; retrying without it")
                    continue
                if status >= 500:
                    last = ProviderUnavailable(f"HTTP {status}")
                else:
                    raise ProviderBadResponse(f"HTTP {status}: {self._safe(err_msg)}")
            i += 1
            if i < attempts:
                _SLEEP(0.5 * 2 ** (i - 1))
        assert last is not None
        raise last

    @staticmethod
    def _error_message(resp: httpx.Response) -> str:
        try:
            return str((resp.json().get("error") or {}).get("message") or "")
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _retry_after(resp: httpx.Response) -> int | None:
        h = resp.headers.get("retry-after")
        if h and h.isdigit():
            return min(int(h), 3600)
        try:
            for d in resp.json().get("error", {}).get("details", []):
                m = re.fullmatch(r"(\d+)(?:\.\d+)?s", str(d.get("retryDelay", "")))
                if m:
                    return min(int(m.group(1)), 3600)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _parse(self, resp: httpx.Response, latency_ms: float) -> AIResponse:
        try:
            body = resp.json()
        except ValueError as exc:
            raise ProviderBadResponse("response was not JSON") from exc
        if not isinstance(body, dict):
            raise ProviderBadResponse("response envelope was not an object")
        block = (body.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise ProviderBlocked(f"prompt blocked: {block}")
        cands = body.get("candidates")
        if not isinstance(cands, list) or not cands:
            raise ProviderBadResponse("no candidates in response")
        cand = cands[0] if isinstance(cands[0], dict) else {}
        finish = cand.get("finishReason")
        if finish in BLOCKING_FINISH:
            raise ProviderBlocked(f"finishReason {finish}")
        if finish == "MAX_TOKENS":
            raise ProviderBadResponse("answer truncated (MAX_TOKENS)")
        texts = [p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []) if isinstance(p, dict) and not p.get("thought")]
        text = "".join(t for t in texts if isinstance(t, str)).strip()
        if not text:
            raise ProviderBadResponse("empty answer")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ProviderBadResponse("answer was not valid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderBadResponse("answer JSON was not an object")
        u = body.get("usageMetadata") or {}
        return AIResponse(data=data, model=str(body.get("modelVersion") or self._s.gemini_model), latency_ms=latency_ms,
                          usage={"prompt_tokens": u.get("promptTokenCount"), "output_tokens": u.get("candidatesTokenCount")})
