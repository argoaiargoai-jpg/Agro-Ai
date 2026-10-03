"""Pl@ntNet plant identification: POST {base}/v2/identify/all?api-key=KEY, multipart `images` + `organs` (https://my.plantnet.org/doc/api/identify).
`organs` is sent ONCE, as a form field (one value for the one image). Sending it a second time in the query string makes the organs list longer than the
images list, which Pl@ntNet rejects with a 400 (our `bad_response`). The key travels in the query string (that is how Pl@ntNet authenticates), so
errors never echo the URL."""
import httpx

from app.ai.specialists import http
from app.ai.specialists.base import PlantIdentifier, PlantMatch, SpecialistError, clamp01, clean
from app.core.config import Settings


class PlantNetProvider(PlantIdentifier):
    name = "plantnet"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._s, self._transport = settings, transport

    def is_configured(self) -> bool:
        return bool(self._s.plantnet_api_key.get_secret_value().strip())

    def identify(self, image: bytes, mime: str) -> list[PlantMatch]:
        key = self._s.plantnet_api_key.get_secret_value().strip()
        if not key:
            raise SpecialistError("not_configured")
        r = http.call(
            "POST", f"{self._s.plantnet_base_url.rstrip('/')}/v2/identify/all", timeout=self._s.specialist_timeout_seconds, transport=self._transport,
            allow=(404,), params={"api-key": key, "nb-results": 3},
            files=[("images", ("plant.jpg", image, mime or "image/jpeg"))], data={"organs": "auto"},
        )
        if r.status_code == 404:                      # "species not found": nothing recognisable as a plant
            return []
        out = []
        for item in http.json_of(r).get("results") or []:
            sp = (item or {}).get("species") or {}
            sci = clean(sp.get("scientificNameWithoutAuthor"))
            common = clean((sp.get("commonNames") or [""])[0])
            if sci or common:
                out.append(PlantMatch(name=common or sci, scientific=sci or None, score=clamp01(item.get("score"))))
        return out[:3]
