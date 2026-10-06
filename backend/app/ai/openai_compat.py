"""Shared adapter for OpenAI-compatible chat-completions providers (Groq, Pollinations). Vendor modules only set names, settings and extras.

POST {base}/chat/completions with `Authorization: Bearer <key>`; the same prompt and the same original image (base64 data URL) Gemini gets; JSON
mode via response_format plus our JSON Schema in the system text. A provider answer counts as successful only when it is a non-empty JSON object
(the workflow then validates it against our schema). Transient failures (timeout, 408, 429, 5xx) are retried with exponential backoff; the key is never logged.
"""
import base64
import json
import logging
import re
import time

import httpx

from app.ai import retry, safety
from app.ai.base import (AIProvider, AIRequest, AIResponse, ProviderBadResponse, ProviderMisconfigured, ProviderNotConfigured, ProviderRateLimited,
                         ProviderTimeout, ProviderUnavailable)
from app.core.config import Settings

_SLEEP = time.sleep   # patched in tests


class OpenAICompatProvider(AIProvider):
    log = logging.getLogger("agroai.ai.compat")
    env_name = ""            # shown in "not configured" messages (a variable NAME, never a value)
    json_mode = True         # send response_format={"type": "json_object"}

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._s, self._transport = settings, transport

    # vendor hooks -----------------------------------------------------------------------------------------------
    def _key(self) -> str:
        raise NotImplementedError

    def _model_name(self) -> str:
        raise NotImplementedError

    def _base_url(self) -> str:
        raise NotImplementedError

    @property
    def model(self) -> str:
        return self._model_name()

    def is_configured(self) -> bool:
        return bool(self._key().strip())

    def _safe(self, text: str) -> str:
        return safety.redact(text, self._s.secret_values())[:300]

    # ------------------------------------------------------------------------------------------------------------
    def analyze(self, request: AIRequest, retries: int | None = None) -> AIResponse:
        key = self._key().strip()
        if not key:
            raise ProviderNotConfigured(f"{self.env_name} is empty")
        system = request.system_instruction + "\n\nReturn ONLY one JSON object (no markdown, no commentary) that matches this JSON Schema:\n" + json.dumps(request.json_schema)
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        if request.image:
            url = f"data:{request.image_mime or 'image/jpeg'};base64,{base64.b64encode(request.image).decode()}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        body = {"model": self._model_name(), "temperature": 0.2, "max_tokens": 4096,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        attempts = 1 + max(self._s.ai_max_retries if retries is None else retries, 0)
        started = time.monotonic()
        last: Exception | None = None
        for i in range(1, attempts + 1):
            wait = None
            try:
                return self._once(body, key)
            except (ProviderTimeout, ProviderUnavailable, ProviderRateLimited) as exc:        # the transient classes
                last, wait = exc, exc.retry_after
            if i == attempts or time.monotonic() - started >= self._s.ai_retry_budget_seconds:
                break
            self.log.warning("%s attempt %s/%s failed (%s); retrying", self.display_name, i, attempts, last.ai_status)
            _SLEEP(retry.delay(i, wait))
        assert last is not None
        raise last

    def _once(self, body: dict, key: str) -> AIResponse:
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=httpx.Timeout(self._s.ai_timeout_seconds), transport=self._transport) as c:
                resp = c.post(f"{self._base_url().rstrip('/')}/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"{self.display_name} timeout ({type(exc).__name__})") from None
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"{self.display_name} network error ({type(exc).__name__})") from None
        latency = (time.perf_counter() - t0) * 1000
        data = self._json(resp)
        if resp.status_code != 200:
            self._raise(resp.status_code, data, resp)
        err = data.get("error") if isinstance(data, dict) else None
        if err:                                                          # an error envelope inside HTTP 200
            code = err.get("code") if isinstance(err, dict) else None
            self._raise(int(code) if str(code).isdigit() else 502, data, resp)
        return self._parse(data, latency, resp)

    @staticmethod
    def _json(resp: httpx.Response):
        try:
            return resp.json()
        except ValueError:
            return None

    def _message(self, data) -> str:
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return self._safe(str(err.get("message", "")))
        return self._safe(str(err or ""))

    def _raise(self, status: int, data, resp: httpx.Response):
        msg = self._message(data)
        self.log.warning("%s HTTP %s: %s", self.display_name, status, msg)
        retry_h = resp.headers.get("retry-after")
        retry_after = min(int(retry_h), 3600) if retry_h and retry_h.isdigit() else None
        if status == 429:
            raise ProviderRateLimited(f"HTTP 429: {msg}", retry_after=retry_after)
        if status in (401, 403, 404):
            raise ProviderMisconfigured(f"HTTP {status}: {msg}")
        if status == 408:
            raise ProviderTimeout(f"HTTP 408: {msg}")
        if status >= 500:
            raise ProviderUnavailable(f"HTTP {status}: {msg}")
        raise ProviderBadResponse(f"HTTP {status}: {msg}")

    def _parse(self, data, latency_ms: float, resp: httpx.Response) -> AIResponse:
        def bad(why: str):
            detail = f"{why} | HTTP {resp.status_code}"
            self.log.warning("%s answer unusable: %s", self.display_name, detail)
            return ProviderBadResponse(detail)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg["content"]
        except (TypeError, KeyError, IndexError):
            raise bad("no message in the response") from None
        if isinstance(text, list):                                       # content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)          # reasoning models may inline their thinking
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
        if choice.get("finish_reason") == "length":
            raise bad("answer truncated (finish_reason=length)")
        if not text:
            raise bad("empty answer")
        start, end = text.find("{"), text.rfind("}")
        try:
            parsed = json.loads(text[start:end + 1]) if start != -1 and end > start else None
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict) or not parsed:
            raise bad("answer was not a JSON object")
        u = data.get("usage") or {}
        return AIResponse(data=parsed, model=str(data.get("model") or self._model_name()), latency_ms=latency_ms,
                          usage={"prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")})

    def ping(self) -> dict:
        """Tiny text-only request for the admin 'Test connection' button (single attempt)."""
        t0 = time.perf_counter()
        r = self.analyze(AIRequest(system_instruction="Reply with JSON only.", prompt='Return exactly {"ok": true}.',
                                   json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}), retries=0)
        return {"ok": bool(r.data.get("ok")), "latency_ms": round((time.perf_counter() - t0) * 1000), "model": r.model}
