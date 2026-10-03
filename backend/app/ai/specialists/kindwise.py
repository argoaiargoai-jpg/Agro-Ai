"""Kindwise: crop.health (POST {crop_base}/api/v1/identification) and plant.health (POST {plant_base}/api/v3/health_assessment).
Auth header `Api-Key`, JSON body {"images": [base64]}. crop.health is documented/verified; plant.health follows Kindwise's plant.id v3 conventions
and has NOT been exercised live. crop.health is tried first; plant.health only when it is configured and crop.health found no crop."""
import base64

import httpx

from app.ai.specialists import http
from app.ai.specialists.base import Diagnosis, DiseaseDiagnoser, DiseaseMatch, PlantMatch, SpecialistError, clamp01, clean
from app.core.config import Settings


def _parse(body: dict) -> Diagnosis:
    result = body.get("result") or {}
    d = Diagnosis()
    plant = result.get("is_plant") or {}
    if isinstance(plant, dict) and plant.get("binary") is not None:
        d.is_plant = bool(plant["binary"])
    for s in ((result.get("crop") or {}).get("suggestions") or [])[:3]:
        if s.get("name"):
            d.crops.append(PlantMatch(name=clean(s["name"]), score=clamp01(s.get("probability"))))
    for s in ((result.get("disease") or {}).get("suggestions") or [])[:5]:
        if s.get("name"):
            d.diseases.append(DiseaseMatch(name=clean(s["name"]), probability=clamp01(s.get("probability"))))
    healthy = result.get("is_healthy") or {}
    if isinstance(healthy, dict) and healthy.get("probability") is not None:
        d.healthy_probability = clamp01(healthy["probability"])
    return d


class KindwiseProvider(DiseaseDiagnoser):
    name = "kindwise"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._s, self._transport = settings, transport

    def _crop_key(self) -> str:
        return self._s.kindwise_api_key.get_secret_value().strip()

    def _plant_key(self) -> str:
        return self._s.kindwise_plant_api_key.get_secret_value().strip()

    def is_configured(self) -> bool:
        return bool(self._crop_key() or self._plant_key())

    def _post(self, url: str, key: str, image: bytes) -> Diagnosis:
        r = http.call("POST", url, timeout=self._s.specialist_timeout_seconds, transport=self._transport, ok=(200, 201),
                      headers={"Api-Key": key, "Content-Type": "application/json"}, json={"images": [base64.b64encode(image).decode()]})
        return _parse(http.json_of(r))

    def diagnose(self, image: bytes, mime: str) -> Diagnosis:
        if not self.is_configured():
            raise SpecialistError("not_configured")
        crop_err = None
        if self._crop_key():
            try:
                d = self._post(f"{self._s.kindwise_crop_base_url.rstrip('/')}/api/v1/identification", self._crop_key(), image)
                if d.crops or d.diseases or not self._plant_key():
                    return d
            except SpecialistError as exc:
                crop_err = exc
                if not self._plant_key():
                    raise
        if self._plant_key():
            try:
                return self._post(f"{self._s.kindwise_plant_base_url.rstrip('/')}/api/v3/health_assessment", self._plant_key(), image)
            except SpecialistError:
                if crop_err:
                    raise crop_err from None
                raise
        return Diagnosis()
