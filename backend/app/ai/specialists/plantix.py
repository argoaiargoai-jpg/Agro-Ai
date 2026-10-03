"""Plantix crop disease image analysis: POST {base}/v2/image_analysis, `Authorization: Bearer KEY`, multipart image.
From Plantix's public API reference we know the endpoint, the Bearer auth and that the answer carries `predicted_diagnoses`, each with a
`common_name` and a qualitative `diagnosis_likelihood` (very_likely .. very_unlikely). The multipart field name and any other fields are
NOT verified against the live API (partner access only), so parsing is deliberately tolerant."""
import httpx

from app.ai.specialists import http
from app.ai.specialists.base import Diagnosis, DiseaseDiagnoser, DiseaseMatch, PlantMatch, SpecialistError, clamp01, clean
from app.core.config import Settings

LIKELIHOOD = {"very_likely": 0.90, "likely": 0.70, "possible": 0.45, "unlikely": 0.20, "very_unlikely": 0.05}


class PlantixProvider(DiseaseDiagnoser):
    name = "plantix"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self._s, self._transport = settings, transport

    def is_configured(self) -> bool:
        return bool(self._s.plantix_api_key.get_secret_value().strip())

    def diagnose(self, image: bytes, mime: str) -> Diagnosis:
        key = self._s.plantix_api_key.get_secret_value().strip()
        if not key:
            raise SpecialistError("not_configured")
        r = http.call("POST", f"{self._s.plantix_base_url.rstrip('/')}/v2/image_analysis", timeout=self._s.specialist_timeout_seconds,
                      transport=self._transport, ok=(200, 201), headers={"Authorization": f"Bearer {key}"},
                      files={"image": ("plant.jpg", image, mime or "image/jpeg")})
        body = http.json_of(r)
        d = Diagnosis()
        for p in (body.get("predicted_diagnoses") or [])[:5]:
            if not isinstance(p, dict):
                continue
            name = clean(p.get("common_name") or p.get("name"))
            lk = p.get("diagnosis_likelihood")
            prob = LIKELIHOOD.get(str(lk), clamp01(p.get("probability", p.get("confidence", 0.0)))) if lk is not None else clamp01(p.get("probability", p.get("confidence", 0.0)))
            if name:
                d.diseases.append(DiseaseMatch(name=name, probability=prob))
            crop = clean(p.get("crop") or p.get("crop_name"))
            if crop and not any(c.name == crop for c in d.crops):
                d.crops.append(PlantMatch(name=crop, score=prob))
        return d
