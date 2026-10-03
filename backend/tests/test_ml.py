"""ML service + API tests.

These use a tiny test-fixture ONNX (see ml_fixture.py) so they check the plumbing and rules, NOT model accuracy.
Model accuracy is measured on held-out data in ml/ (reports/<run>/metrics.json).
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.errors import AppError
from app.services import ml_service
from tests import ml_fixture
from tests.conftest import register_and_verify

V = "/api/v1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))   # decision.py (the rules used for evaluation)


def img_bytes(color=(255, 0, 0), size=(300, 200), fmt="PNG", mode="RGB"):
    buf = io.BytesIO(); Image.new(mode, size, color).save(buf, fmt); return buf.getvalue()


@pytest.fixture
def model(tmp_path, settings, monkeypatch):
    d = ml_fixture.build(tmp_path / "ml")
    monkeypatch.setattr(settings, "ml_model_dir", str(d))
    ml_service.reset_ml_service()
    yield ml_service.get_ml_service()
    ml_service.reset_ml_service()


@pytest.fixture
def no_model(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", str(tmp_path / "empty"))
    ml_service.reset_ml_service()
    yield
    ml_service.reset_ml_service()


# ---------------------------------------------------------------- service: loading + outcomes
def test_model_loads_and_reports_scope(model):
    info = model.info()
    assert info["available"] and info["supported_crops"] == ["Tomato"]
    assert set(info["outcomes"]) == {"DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"}
    assert model.load_ms < 5000


def test_disease_result_schema(model):
    r = model.predict(img_bytes((255, 0, 0)))
    assert r["classification_type"] == "DISEASE" and r["crop"] == "Tomato" and r["disease"] == "Early Blight"
    assert 0.9 < r["confidence"] <= 1.0
    assert {"message", "model_version", "inference_ms", "scores", "supported_crops"} <= set(r)


def test_healthy_has_no_disease(model):
    r = model.predict(img_bytes((0, 255, 0)))
    assert r["classification_type"] == "HEALTHY" and r["crop"] == "Tomato" and r["disease"] is None


def test_non_plant_outcome(model):
    r = model.predict(img_bytes((0, 0, 255)))
    assert r["classification_type"] == "NO_PLANT" and r["crop"] is None and r["disease"] is None


def test_unknown_outcome_is_not_forced_into_a_class(model):
    r = model.predict(img_bytes((128, 128, 128)))       # flat evidence -> no class is believable
    assert r["classification_type"] == "UNKNOWN" and r["crop"] is None and r["disease"] is None
    assert r["confidence"] < 0.9


@pytest.mark.parametrize("fmt,mode,color", [("PNG", "RGBA", (255, 0, 0, 128)), ("PNG", "L", 80), ("PNG", "P", 3), ("JPEG", "RGB", (255, 0, 0)), ("WEBP", "RGB", (255, 0, 0))])
def test_image_modes_and_formats_are_handled(model, fmt, mode, color):
    r = model.predict(img_bytes(color, fmt=fmt, mode=mode))
    assert r["classification_type"] in {"DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"}


@pytest.mark.parametrize("size", [(64, 4000), (4000, 64), (224, 224), (50, 50)])
def test_non_square_and_extreme_aspect_ratios(model, size):
    assert model.preprocess(img_bytes(size=size)).shape == (1, 3, 224, 224)


def test_preprocess_shape_dtype_and_normalisation(model):
    x = model.preprocess(img_bytes((255, 255, 255), size=(300, 300)))
    assert x.dtype == np.float32 and x.shape == (1, 3, 224, 224)
    assert np.allclose(x, 1.0)    # fixture uses mean 0 / std 1 on [0,1] pixels


# ---------------------------------------------------------------- service: bad input
def test_corrupt_bytes_rejected(model):
    with pytest.raises(AppError) as e:
        model.predict(b"this is not an image")
    assert e.value.code == "invalid_image" and e.value.status_code == 422


def test_truncated_image_rejected(model):
    data = img_bytes(fmt="JPEG", size=(800, 800))
    with pytest.raises(AppError) as e:
        model.predict(data[: len(data) // 3])
    assert e.value.code == "invalid_image"


def test_empty_input_rejected(model):
    with pytest.raises(AppError) as e:
        model.predict(b"")
    assert e.value.code == "invalid_image"


def test_tiny_image_rejected(model):
    with pytest.raises(AppError) as e:
        model.predict(img_bytes(size=(20, 20)))
    assert e.value.code == "image_too_small"


def test_pixel_bomb_rejected(model, settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_max_pixels", 10_000)
    with pytest.raises(AppError) as e:
        model.predict(img_bytes(size=(300, 200)))
    assert e.value.status_code in (413, 422)


# ---------------------------------------------------------------- rejection thresholds
def test_confidence_threshold_controls_unknown(tmp_path, settings, monkeypatch):
    logits = np.array([1.5, 0.0, 0.0, 0.0], dtype=np.float32)            # softmax top prob = e^1.5/(e^1.5+3) ~ 0.60
    for t_conf, expected in ((0.5, "DISEASE"), (0.7, "UNKNOWN")):
        d = ml_fixture.build(tmp_path / f"m{t_conf}", t_conf=t_conf)
        monkeypatch.setattr(settings, "ml_model_dir", str(d)); ml_service.reset_ml_service()
        assert ml_service.get_ml_service().decide(logits)["classification_type"] == expected
    ml_service.reset_ml_service()


def test_non_plant_threshold(tmp_path, settings, monkeypatch):
    logits = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)            # non_plant prob = e^2/(e^2+3) ~ 0.71
    for t_np, expected in ((0.5, "NO_PLANT"), (0.9, "UNKNOWN")):   # above the threshold => not called NO_PLANT
        d = ml_fixture.build(tmp_path / f"n{t_np}", t_np=t_np, t_conf=0.5)
        monkeypatch.setattr(settings, "ml_model_dir", str(d)); ml_service.reset_ml_service()
        assert ml_service.get_ml_service().decide(logits)["classification_type"] == expected
    ml_service.reset_ml_service()


def test_service_rules_match_the_rules_used_for_evaluation(model):
    """Production (ml_service) and evaluation (ml/decision.py) must never disagree."""
    import decision
    cfg = dict(model.meta["decision"]); ids = model.ids
    rng = np.random.default_rng(0)
    for _ in range(400):
        z = rng.normal(0, rng.choice([0.5, 2, 6]), size=4).astype(np.float32)
        ours = model.decide(z)["classification_type"]
        theirs = decision.decide(z[None], ids, cfg)[0]["outcome"]
        assert ours == theirs, (z, ours, theirs)


def test_service_missing_model_is_a_503_not_a_fake_answer(no_model):
    with pytest.raises(AppError) as e:
        ml_service.get_ml_service()
    assert e.value.status_code == 503 and e.value.code == "ml_unavailable"


# ---------------------------------------------------------------- API
def upload(client, headers, data, name="leaf.png"):
    r = client.post(f"{V}/analyses", headers=headers, files={"file": (name, io.BytesIO(data), "image/png")})
    assert r.status_code == 201, r.text
    return r.json()


def test_analyze_requires_auth(client, model):
    assert client.post(f"{V}/analyses/00000000-0000-0000-0000-000000000000/analyze").status_code == 401


def test_analyze_end_to_end(client, user_auth, model):
    a = upload(client, user_auth, img_bytes((255, 0, 0), size=(400, 300)))
    assert a["status"] == "uploaded" and a["result"] is None
    r = client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth)
    assert r.status_code == 200, r.text
    body = r.json()
    ml = body["result"]["ml"]
    assert body["status"] == "completed" and body["confidence"] == ml["confidence"]
    assert ml["classification_type"] == "DISEASE" and ml["crop"] == "Tomato" and ml["disease"] == "Early Blight"
    # Phase 3 contract: with no AI provider configured the ML result is final and NO guidance is fabricated
    assert body["result"].get("ai") is None and body["result"]["final"] is None and body["result"]["stage"] == "ml_only"
    assert body["ai_status"] == "not_configured"
    lst = client.get(f"{V}/analyses?status=completed", headers=user_auth).json()
    assert lst["total"] == 1


def test_analyze_is_cached_unless_forced(client, user_auth, model):
    a = upload(client, user_auth, img_bytes((0, 255, 0)))
    first = client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth).json()
    again = client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth).json()
    assert again["result"]["ml"]["inference_ms"] == first["result"]["ml"]["inference_ms"]   # served from storage
    forced = client.post(f"{V}/analyses/{a['id']}/analyze?force=true", headers=user_auth)
    assert forced.status_code == 200 and forced.json()["result"]["ml"]["classification_type"] == "HEALTHY"


@pytest.mark.parametrize("color,expected", [((255, 0, 0), "DISEASE"), ((0, 255, 0), "HEALTHY"), ((0, 0, 255), "NO_PLANT"), ((128, 128, 128), "UNKNOWN")])
def test_api_returns_every_outcome(client, user_auth, model, color, expected):
    a = upload(client, user_auth, img_bytes(color))
    r = client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth).json()
    assert r["result"]["ml"]["classification_type"] == expected


def test_cannot_analyze_someone_elses_image(client, user_auth, model):
    a = upload(client, user_auth, img_bytes())
    other = register_and_verify(client, email="other@example.com")
    r = client.post(f"{V}/analyses/{a['id']}/analyze", headers={"Authorization": f"Bearer {other['access_token']}"})
    assert r.status_code == 404


def test_analyze_unknown_or_malformed_id(client, user_auth, model):
    assert client.post(f"{V}/analyses/00000000-0000-0000-0000-000000000000/analyze", headers=user_auth).status_code == 404
    assert client.post(f"{V}/analyses/not-a-uuid/analyze", headers=user_auth).status_code == 422


def test_corrupt_image_with_valid_magic_bytes_is_rejected_at_upload(client, user_auth, model):
    fake = b"\x89PNG\r\n\x1a\n" + b"garbage" * 50          # passes magic-byte sniffing, but is not a decodable image
    r = client.post(f"{V}/analyses", headers=user_auth, files={"file": ("a.png", io.BytesIO(fake), "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"
    assert client.get(f"{V}/analyses", headers=user_auth).json()["total"] == 0          # nothing was stored


def test_image_that_becomes_unreadable_after_upload_fails_cleanly(client, user_auth, model, settings):
    a = upload(client, user_auth, img_bytes())
    stored = next(p for p in settings.upload_path.rglob("*.png"))
    stored.write_bytes(b"\x89PNG\r\n\x1a\n" + b"disk corruption" * 20)                   # simulate the stored file being damaged later
    r = client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"
    got = client.get(f"{V}/analyses/{a['id']}", headers=user_auth).json()
    assert got["status"] == "failed" and got["error_message"] and got["result"] is None


def test_missing_model_is_503_and_upload_stays_retryable(client, user_auth, no_model):
    a = upload(client, user_auth, img_bytes())
    r = client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth)
    assert r.status_code == 503 and r.json()["error"]["code"] == "ml_unavailable"
    got = client.get(f"{V}/analyses/{a['id']}", headers=user_auth).json()
    assert got["status"] == "uploaded" and got["result"] is None          # no fabricated result


def test_maintenance_mode_blocks_analysis(client, user_auth, admin_auth, model):
    a = upload(client, user_auth, img_bytes())
    client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"maintenance_mode": True}})
    assert client.post(f"{V}/analyses/{a['id']}/analyze", headers=user_auth).status_code == 503


def test_existing_upload_validation_still_enforced(client, user_auth, model):
    r = client.post(f"{V}/analyses", headers=user_auth, files={"file": ("x.png", io.BytesIO(b"<html>"), "image/png")})
    assert r.status_code == 415
    assert client.post(f"{V}/analyses", files={"file": ("a.png", img_bytes(), "image/png")}).status_code == 401


def test_ml_info_endpoint(client, model, no_model_after=None):
    r = client.get(f"{V}/ml/info")
    assert r.status_code == 200 and r.json()["available"] is True and r.json()["supported_crops"] == ["Tomato"]


def test_ml_info_when_model_missing(client, no_model):
    r = client.get(f"{V}/ml/info")
    assert r.status_code == 200 and r.json()["available"] is False


def test_health_still_works_without_model(client, no_model):
    assert client.get(f"{V}/health").json()["status"] == "ok"
