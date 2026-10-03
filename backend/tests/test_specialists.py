"""Pl@ntNet / Kindwise / Plantix adapters against mocked HTTP: request shape, parsing, error mapping, and that no key ever leaks."""
import base64
import json

import httpx
import pytest

from app.ai.specialists.base import SpecialistError
from app.ai.specialists.kindwise import KindwiseProvider
from app.ai.specialists.plantix import PlantixProvider
from app.ai.specialists.plantnet import PlantNetProvider
from app.core.config import Settings

IMG = b"\xff\xd8\xff\xe0FAKEJPEGBYTES" * 20
KEY = "PRIVATEKEY-" + "0123456789abcdef"


def S(**kw) -> Settings:
    return Settings(_env_file=None, environment="test", **kw)


class Rec:
    def __init__(self, response=None, exc=None):
        self.requests, self.response, self.exc = [], response, exc

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if self.exc:
            raise self.exc
        return self.response

    @property
    def transport(self):
        return httpx.MockTransport(self)


def js(status, body):
    return httpx.Response(status, json=body)


# ------------------------------------------------------------------------------------------------ Pl@ntNet
PLANTNET_OK = {"bestMatch": "Rosa gallica L.", "results": [
    {"score": 0.91, "species": {"scientificNameWithoutAuthor": "Rosa gallica", "commonNames": ["French rose", "Gallic rose"]}},
    {"score": 0.05, "species": {"scientificNameWithoutAuthor": "Rosa canina", "commonNames": []}}]}


def test_plantnet_request_sends_the_image_bytes_and_parses_matches():
    rec = Rec(js(200, PLANTNET_OK))
    out = PlantNetProvider(S(plantnet_api_key=KEY), rec.transport).identify(IMG, "image/jpeg")
    req = rec.requests[0]
    assert req.method == "POST" and req.url.path == "/v2/identify/all" and req.url.params["api-key"] == KEY
    assert req.headers["content-type"].startswith("multipart/form-data") and IMG in req.content and b'name="images"' in req.content and b'name="organs"' in req.content
    assert [(m.name, m.scientific, m.score) for m in out] == [("French rose", "Rosa gallica", 0.91), ("Rosa canina", "Rosa canina", 0.05)]


def test_plantnet_404_means_nothing_identified():
    assert PlantNetProvider(S(plantnet_api_key=KEY), Rec(js(404, {"message": "Species not found"})).transport).identify(IMG, "image/jpeg") == []


@pytest.mark.parametrize("status,kind", [(401, "unauthorized"), (403, "unauthorized"), (429, "rate_limited"), (500, "unavailable"), (503, "unavailable"), (400, "bad_response")])
def test_plantnet_errors_are_mapped_and_never_echo_the_key_in_the_url(status, kind):
    with pytest.raises(SpecialistError) as e:
        PlantNetProvider(S(plantnet_api_key=KEY), Rec(js(status, {"message": "x"})).transport).identify(IMG, "image/jpeg")
    assert e.value.kind == kind and KEY not in str(e.value) and KEY not in e.value.detail


@pytest.mark.parametrize("exc,kind", [(httpx.ReadTimeout("t"), "timeout"), (httpx.ConnectError("c"), "unavailable")])
def test_plantnet_network_errors_never_leak_the_key(exc, kind):
    with pytest.raises(SpecialistError) as e:
        PlantNetProvider(S(plantnet_api_key=KEY), Rec(exc=exc).transport).identify(IMG, "image/jpeg")
    assert e.value.kind == kind and KEY not in str(e.value)


def test_plantnet_not_configured():
    p = PlantNetProvider(S())
    assert not p.is_configured()
    with pytest.raises(SpecialistError) as e:
        p.identify(IMG, "image/jpeg")
    assert e.value.kind == "not_configured"


# ------------------------------------------------------------------------------------------------ Kindwise
KW_OK = {"result": {"is_plant": {"binary": True, "probability": 0.99}, "crop": {"suggestions": [{"name": "tomato", "probability": 0.95}]},
                    "disease": {"suggestions": [{"name": "early blight", "probability": 0.81}, {"name": "septoria", "probability": 0.1}]}}}


def test_kindwise_crop_health_request_and_parsing():
    rec = Rec(js(201, KW_OK))
    d = KindwiseProvider(S(kindwise_api_key=KEY), rec.transport).diagnose(IMG, "image/jpeg")
    req = rec.requests[0]
    assert req.url.host == "crop.kindwise.com" and req.url.path == "/api/v1/identification" and req.headers["Api-Key"] == KEY and KEY not in str(req.url)
    assert base64.b64decode(json.loads(req.content)["images"][0]) == IMG                      # the actual image bytes (base64) are in the body
    assert d.is_plant is True and d.crops[0].name == "tomato" and [(x.name, x.probability) for x in d.diseases][:1] == [("early blight", 0.81)]


def test_kindwise_falls_back_to_plant_health_only_when_no_crop_was_found():
    seen = []

    def handler(req):
        seen.append(req.url.host)
        return js(200, {"result": {"crop": {"suggestions": []}, "disease": {"suggestions": []}}}) if "crop" in req.url.host else \
            js(200, {"result": {"is_plant": {"binary": True}, "is_healthy": {"binary": False, "probability": 0.2}, "disease": {"suggestions": [{"name": "rust", "probability": 0.6}]}}})
    p = KindwiseProvider(S(kindwise_api_key=KEY, kindwise_plant_api_key=KEY + "2"), httpx.MockTransport(handler))
    d = p.diagnose(IMG, "image/jpeg")
    assert seen == ["crop.kindwise.com", "plant.id"] and d.diseases[0].name == "rust" and d.healthy_probability == 0.2


def test_kindwise_with_only_crop_key_never_calls_plant_health():
    seen = []
    p = KindwiseProvider(S(kindwise_api_key=KEY), httpx.MockTransport(lambda r: (seen.append(r.url.host), js(200, {"result": {}}))[1]))
    p.diagnose(IMG, "image/jpeg")
    assert seen == ["crop.kindwise.com"]


@pytest.mark.parametrize("status,kind", [(401, "unauthorized"), (429, "rate_limited"), (500, "unavailable")])
def test_kindwise_errors(status, kind):
    with pytest.raises(SpecialistError) as e:
        KindwiseProvider(S(kindwise_api_key=KEY), Rec(js(status, {})).transport).diagnose(IMG, "image/jpeg")
    assert e.value.kind == kind and KEY not in str(e.value)


def test_kindwise_garbage_response_is_a_clean_error():
    with pytest.raises(SpecialistError) as e:
        KindwiseProvider(S(kindwise_api_key=KEY), Rec(httpx.Response(200, text="<html>nope</html>")).transport).diagnose(IMG, "image/jpeg")
    assert e.value.kind == "bad_response"


# ------------------------------------------------------------------------------------------------ Plantix
PX_OK = {"plantix_trace_id": "t-1", "predicted_diagnoses": [{"common_name": "Late blight", "diagnosis_likelihood": "very_likely"},
                                                           {"common_name": "Early blight", "diagnosis_likelihood": "possible"}]}


def test_plantix_request_and_parsing():
    rec = Rec(js(200, PX_OK))
    d = PlantixProvider(S(plantix_api_key=KEY), rec.transport).diagnose(IMG, "image/jpeg")
    req = rec.requests[0]
    assert req.url.host == "api.plantix.net" and req.url.path == "/v2/image_analysis" and req.headers["Authorization"] == f"Bearer {KEY}" and KEY not in str(req.url)
    assert IMG in req.content and req.headers["content-type"].startswith("multipart/form-data")
    assert [(x.name, x.probability) for x in d.diseases] == [("Late blight", 0.90), ("Early blight", 0.45)]


def test_plantix_tolerates_missing_or_unexpected_fields():
    d = PlantixProvider(S(plantix_api_key=KEY), Rec(js(200, {"predicted_diagnoses": [None, {"foo": 1}, {"common_name": "Rust", "probability": 0.7}]})).transport).diagnose(IMG, "image/jpeg")
    assert [x.name for x in d.diseases] == ["Rust"]
    assert PlantixProvider(S(plantix_api_key=KEY), Rec(js(200, {})).transport).diagnose(IMG, "image/jpeg").diseases == []


@pytest.mark.parametrize("status,kind", [(401, "unauthorized"), (429, "rate_limited"), (502, "unavailable"), (422, "bad_response")])
def test_plantix_errors(status, kind):
    with pytest.raises(SpecialistError) as e:
        PlantixProvider(S(plantix_api_key=KEY), Rec(js(status, {})).transport).diagnose(IMG, "image/jpeg")
    assert e.value.kind == kind and KEY not in str(e.value)


# ------------------------------------------------------------------------------------------------ secrecy
def test_every_provider_secret_is_in_the_log_redaction_list():
    s = S(gemini_api_key_1="g1-" + "x" * 10, gemini_api_key_2="g2-" + "x" * 10, plantnet_api_key="pn-" + "x" * 10, plantix_api_key="px-" + "x" * 10,
          kindwise_api_key="kw-" + "x" * 10, kindwise_plant_api_key="kp-" + "x" * 10, brevo_api_key="bv-" + "x" * 10)
    assert len(s.secret_values()) == 7 and not any(v in repr(s) for v in s.secret_values())
