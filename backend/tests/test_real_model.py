"""The REAL trained model (installed from the Colab results) and the full workflow built on it.

Artifact/threshold/metrics tests always run. Real-image tests use the git-ignored local datasets in ml/data and are skipped
(with a reason) on a machine that does not have them. The AI provider is always a deterministic mock.
"""
import csv
import io
import json
import random
from datetime import timedelta
from pathlib import Path

import pytest
from PIL import Image, ImageEnhance

from app.ai import registry
from app.core.errors import AppError
from app.services import analysis_service, analysis_workflow, ml_service
from tests.ai_fakes import NO_PLANT, Scenario, diseased, install, payload

BACK = Path(__file__).resolve().parents[1]
APP_ML = BACK / "app" / "ml"
DATA = BACK / "ml" / "data"
V = "/api/v1"


@pytest.fixture(autouse=True)
def real_model(settings, monkeypatch):
    monkeypatch.setattr(settings, "ml_model_dir", "")          # the default location: backend/app/ml (the real model)
    ml_service.reset_ml_service(); analysis_workflow.reset_gate()
    monkeypatch.setattr(analysis_service, "AI_RETRY_COOLDOWN", timedelta(0))
    monkeypatch.setattr(analysis_service, "AI_FORCE_COOLDOWN", timedelta(0))
    yield
    ml_service.reset_ml_service(); registry.unregister("fake")


# ------------------------------------------------------------------ artifacts
def test_only_the_runtime_artifacts_are_installed():
    names = sorted(p.name for p in APP_ML.iterdir())
    assert names == ["agro_model.onnx", "classes.json", "model_meta.json", "thresholds.json"]
    assert (APP_ML / "agro_model.onnx").stat().st_size == 6_228_649 < 7_000_000          # 6.23 MB, as reported by Colab
    for banned in ("*.jpg", "*.jpeg", "*.png", "*.csv", "*.pt", "*.pth", "*.ipynb", "*.zip"):
        assert not list(BACK.joinpath("app").rglob(banned)), banned                      # no training data/checkpoints/notebooks in the app


def test_metadata_identifies_the_colab_run():
    meta = json.loads((APP_ML / "model_meta.json").read_text())
    assert meta["run"] == "colab" and meta["trained_epoch"] == 14 and "MobileNetV3-Small" in meta["model"]
    assert meta["input"]["size"] == 224 and meta["input"]["resize"] == 256
    assert meta["supported_crops"] == ["Apple", "Bell Pepper", "Cherry", "Corn", "Grape", "Peach", "Potato", "Soybean", "Strawberry", "Tomato"]


def test_classes_json_matches_the_metadata_and_the_model_head():
    meta = json.loads((APP_ML / "model_meta.json").read_text()); classes = json.loads((APP_ML / "classes.json").read_text())
    assert classes == [c["id"] for c in meta["classes"]] and len(classes) == 36
    assert classes[-2:] == ["__unknown_plant__", "__non_plant__"] and sum("___" in c for c in classes) == 34


def test_thresholds_are_exactly_the_colab_calibrated_values():
    t = json.loads((APP_ML / "thresholds.json").read_text()); meta = json.loads((APP_ML / "model_meta.json").read_text())
    for src in (t, meta["decision"]):
        assert src["temperature"] == 0.5581487417221069 and src["t_confidence"] == 0.92 and abs(src["t_non_plant"] - 0.445) < 1e-12
    s = ml_service.get_ml_service()
    assert (s.T, s.t_conf) == (0.5581487417221069, 0.92) and abs(s.t_np - 0.445) < 1e-12        # what inference actually uses


def test_onnx_runtime_loads_and_produces_36_scores():
    import numpy as np
    s = ml_service.get_ml_service()
    out = s.session.run(None, {"image": np.zeros((1, 3, 224, 224), dtype=np.float32)})[0]
    assert out.shape == (1, 36) and np.isfinite(out).all() and s.load_ms < 5000
    assert s.session.get_providers()[0] in ("CPUExecutionProvider", "CoreMLExecutionProvider")


def test_ml_info_endpoint_describes_the_real_model(client):
    r = client.get(f"{V}/ml/info")
    b = r.json()
    assert r.status_code == 200 and b["available"] is True and b["version"] == "colab" and "MobileNetV3-Small" in b["model"]
    assert b["supported_crops"] == ["Apple", "Bell Pepper", "Cherry", "Corn", "Grape", "Peach", "Potato", "Soybean", "Strawberry", "Tomato"]
    assert len(b["supported_conditions"]) == 34 and "Tomato: Early Blight" in b["supported_conditions"] and "Tomato: Healthy" in b["supported_conditions"]
    assert set(b["outcomes"]) == {"DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"}
    assert client.get(f"{V}/health").json()["ml_model"] == "installed"


def test_recorded_colab_metrics_are_untouched():
    """The Colab evaluation is the source of truth. This pins the numbers so they cannot be altered by accident. Datasets are reported separately."""
    m = json.loads((BACK / "ml" / "reports" / "colab" / "metrics.json").read_text())
    A, B = m["A_plantvillage_test_known_classes"], m["B_plantdoc_test_real_world_known_classes"]
    assert (A["accuracy_raw_argmax"], A["accuracy_final_pipeline"], A["accuracy_when_answered"], A["f1_macro"], A["pipeline_coverage"]) == (0.9835, 0.919, 0.9973, 0.9784, 0.9215)
    assert (B["accuracy_raw_argmax"], B["accuracy_final_pipeline"], B["accuracy_when_answered"], B["pipeline_coverage"]) == (0.6274, 0.3255, 0.7931, 0.4104)
    assert m["C_coco_test_non_plant"]["no_plant_recall"] == 0.9905
    assert m["C_false_no_plant_rate_on_real_plant_test_images"] == {"plantvillage": 0.0, "plantdoc": 0.0047}
    D = m["D_unknown_unsupported_plants"]
    assert D["plantvillage_Blueberry_novel_crop"]["correctly_refused(UNKNOWN or NO_PLANT)"] == 0.96 and D["plantvillage_Orange_Squash_seen_unsupported_crops"]["UNKNOWN"] == 1.0
    assert D["plantdoc_unsupported_crops(Blueberry/Raspberry novel+Squash)"]["UNKNOWN"] == 0.7917
    assert m["thresholds"] == {"temperature": 0.5581487417221069, "t_non_plant": 0.44499999999999995, "t_confidence": 0.92}
    perf = json.loads((BACK / "ml" / "reports" / "colab" / "performance_export.json").read_text())
    assert perf["model_size_mb"] == 6.23 and "blended_accuracy" not in m and "overall_accuracy" not in m      # never one blended number


# ------------------------------------------------------------------ real images
needs_data = pytest.mark.skipif(not (DATA / "manifest.csv").exists(), reason="local datasets (backend/ml/data) not present")


def rows(pred, n):
    allrows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] == "test" and pred(r)]
    return sorted(allrows, key=lambda r: r["path"])[:n]


def blob(r) -> bytes:
    return (DATA / r["path"]).read_bytes()


def outcome(data: bytes):
    return ml_service.get_ml_service().predict(data)


@needs_data
def test_real_plantvillage_disease_images_are_diagnosed():
    res = [outcome(blob(r)) for r in rows(lambda r: r["label_out"] == "Apple___Apple_scab", 10)]
    ok = [x for x in res if x["classification_type"] == "DISEASE" and x["crop"] == "Apple" and x["disease"] == "Apple Scab"]
    assert len(ok) >= 8 and all(x["classification_type"] in {"DISEASE", "UNKNOWN"} for x in res)       # may abstain, never calls it healthy


@needs_data
def test_real_plantvillage_healthy_images_are_recognised_as_healthy():
    res = [outcome(blob(r)) for r in rows(lambda r: r["label_out"] == "Grape___healthy", 10)]
    assert sum(x["classification_type"] == "HEALTHY" and x["crop"] == "Grape" and x["disease"] is None for x in res) >= 8


@needs_data
def test_real_world_plantdoc_images_return_valid_states_and_often_abstain():
    res = [outcome(blob(r)) for r in rows(lambda r: r["source"] == "plantdoc" and r["role"] == "known", 40)]
    assert all(x["classification_type"] in {"DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"} and 0 <= x["confidence"] <= 1 for x in res)
    assert any(x["classification_type"] == "UNKNOWN" for x in res)         # documented limitation: real-world photos are often refused rather than guessed


@needs_data
def test_unsupported_plants_are_not_forced_into_a_known_diagnosis():
    orange = [outcome(blob(r))["classification_type"] for r in rows(lambda r: r["role"] == "unknown_seen" and r["source"] == "plantvillage", 15)]
    blueberry = [outcome(blob(r))["classification_type"] for r in rows(lambda r: r["role"] == "unknown_novel" and r["source"] == "plantvillage", 25)]
    assert orange.count("UNKNOWN") == 15 and blueberry.count("UNKNOWN") >= 21


@needs_data
def test_everyday_non_plant_photos_are_identified_as_no_plant():
    res = [outcome(blob(r))["classification_type"] for r in rows(lambda r: r["source"] == "coco", 30)]
    assert res.count("NO_PLANT") >= 28 and not {"DISEASE", "HEALTHY"} & set(res)


@needs_data
def test_poor_quality_images_do_not_become_confident_diagnoses():
    random.seed(3)
    kinds = []
    for r in rows(lambda r: r["source"] == "plantvillage" and r["label_out"].endswith("Black_rot"), 10):
        im = Image.open(DATA / r["path"]).convert("RGB")
        noisy = Image.blend(im, Image.effect_noise(im.size, 90).convert("RGB"), 0.6)
        dark = ImageEnhance.Brightness(im).enhance(0.12)
        for x in (noisy, dark):
            b = io.BytesIO(); x.save(b, "PNG"); kinds.append(outcome(b.getvalue())["classification_type"])
    assert sum(k in ("UNKNOWN", "NO_PLANT") for k in kinds) >= 16 and kinds.count("HEALTHY") == 0           # refuses instead of guessing


def test_corrupt_inputs_are_clean_errors_not_ml_states():
    s = ml_service.get_ml_service()
    for bad, code in [(b"", "invalid_image"), (random.Random(1).randbytes(4000), "invalid_image"), (b"\x89PNG\r\n\x1a\n" + b"x" * 200, "invalid_image")]:
        with pytest.raises(AppError) as e:
            s.predict(bad)
        assert e.value.code == code


# ------------------------------------------------------------------ full Phase 3 workflow on the REAL model (AI is mocked)
@pytest.fixture
def sc(client, admin_auth):
    s = Scenario(); install(s)
    assert client.put(f"{V}/admin/ai", headers=admin_auth, json={"provider": "fake", "enabled": True}).status_code == 200
    return s


def first_with_state(pred, state):
    for r in rows(pred, 40):
        if outcome(blob(r))["classification_type"] == state:
            return r
    pytest.skip(f"no sample classified {state}")


def analyze(client, headers, r):
    up = client.post(f"{V}/analyses", headers=headers, files={"file": ("photo.jpg", blob(r), "image/jpeg")})
    assert up.status_code == 201, up.text
    return client.post(f"{V}/analyses/{up.json()['id']}/analyze", headers=headers).json()


@needs_data
def test_real_model_disease_goes_to_the_ai_as_advice_only(client, admin_auth, sc):
    r = first_with_state(lambda r: r["label_out"] == "Tomato___Early_blight", "DISEASE")
    sc.respond(diseased("Something Else", ml_consistency="consistent"))
    b = analyze(client, admin_auth, r)
    ml = b["result"]["ml"]
    assert ml["model_version"] == "colab" and ml["classification_type"] == "DISEASE" and (ml["crop"], ml["disease"]) == ("Tomato", "Early Blight")     # the real model
    f = b["result"]["final"]
    assert f["status"] == "DISEASE" and (f["crop"], f["disease"], f["disease_source"]) == ("Tomato", "Early Blight", "ml") and f["treatment"]            # AI never overrides
    req = sc.requests[0]
    assert req.image and "ALREADY identified" in req.prompt and "Tomato" in req.prompt and "Early Blight" in req.prompt                                  # image + the real ML result


@needs_data
def test_real_model_healthy_is_inspected_independently_by_the_ai(client, user_auth, sc):
    r = first_with_state(lambda r: r["label_out"] == "Grape___healthy", "HEALTHY")
    sc.respond(payload(crop="Grape", plant="Grape"))
    b = analyze(client, user_auth, r)
    assert b["result"]["ml"]["classification_type"] == "HEALTHY" and b["result"]["ml"]["model_version"] == "colab"
    assert b["result"]["final"]["status"] == "HEALTHY"
    req = sc.requests[0]
    assert req.image and "ML_HEALTHY" in req.prompt and "ALREADY identified" not in req.prompt and "Grape" not in req.prompt          # independent: ML result withheld


@needs_data
def test_real_model_healthy_but_ai_sees_disease_keeps_both_internally(client, admin_auth, sc):
    r = first_with_state(lambda r: r["label_out"] == "Grape___healthy", "HEALTHY")
    sc.respond(diseased("Black Rot", crop="Grape", plant="Grape"))
    b = analyze(client, admin_auth, r)
    f = b["result"]["final"]
    assert f["status"] == "DISEASE" and f["disease_source"] == "ai" and f["disagreement"] and b["result"]["ml"]["classification_type"] == "HEALTHY"


@needs_data
def test_real_model_unknown_is_analysed_independently_by_the_ai(client, user_auth, sc):
    r = first_with_state(lambda r: r["role"] == "unknown_seen" and r["source"] == "plantvillage", "UNKNOWN")
    sc.respond(payload(health_status="uncertain", crop=None, plant=None))
    b = analyze(client, user_auth, r)
    assert b["result"]["ml"]["classification_type"] == "UNKNOWN" and b["result"]["final"]["status"] == "UNCERTAIN"
    assert sc.requests[0].image and "ML_UNKNOWN" in sc.requests[0].prompt


@needs_data
def test_real_model_no_plant_still_reaches_the_ai(client, user_auth, sc):
    r = first_with_state(lambda r: r["source"] == "coco", "NO_PLANT")
    sc.respond(NO_PLANT)
    b = analyze(client, user_auth, r)
    assert b["result"]["ml"]["classification_type"] == "NO_PLANT" and len(sc.requests) == 1 and sc.requests[0].image           # not auto-rejected
    assert b["result"]["final"]["status"] == "REJECTED"
    sc.respond(diseased("Leaf Spot", crop="Rose", plant="Rose"))                                                              # the AI finds a plant after all
    again = client.post(f"{V}/analyses/{b['id']}/analyze?force=true", headers=user_auth).json()
    assert again["result"]["ml"]["classification_type"] == "NO_PLANT" and again["result"]["final"]["status"] == "DISEASE" and len(sc.requests) == 2


@needs_data
def test_real_model_ai_failure_keeps_the_real_ml_result_and_retries_without_rerunning_it(client, user_auth, sc):
    from app.ai.base import ProviderTimeout
    r = first_with_state(lambda r: r["label_out"] == "Apple___Apple_scab", "DISEASE")
    sc.respond(ProviderTimeout("slow"), diseased("x", ml_consistency="consistent"))
    b = analyze(client, user_auth, r)
    assert b["status"] == "partial" and b["result"]["ml"]["disease"] == "Apple Scab"
    retry = client.post(f"{V}/analyses/{b['id']}/analyze", headers=user_auth).json()
    assert retry["status"] == "completed" and retry["result"]["ml"] == b["result"]["ml"]


def test_corrupt_upload_is_rejected_before_the_real_model_runs(client, user_auth):
    r = client.post(f"{V}/analyses", headers=user_auth, files={"file": ("a.jpg", b"\xff\xd8\xff" + b"junk" * 100, "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"


def test_real_model_without_any_ai_configured_returns_the_basic_result(client, user_auth, tmp_path):
    im = Image.new("RGB", (200, 200), (30, 30, 30)); b = io.BytesIO(); im.save(b, "JPEG")
    up = client.post(f"{V}/analyses", headers=user_auth, files={"file": ("a.jpg", b.getvalue(), "image/jpeg")}).json()
    r = client.post(f"{V}/analyses/{up['id']}/analyze", headers=user_auth).json()
    assert r["status"] == "completed" and r["ai_status"] == "not_configured" and r["result"]["final"] is None
    assert r["result"]["ml"]["classification_type"] in {"DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"} and r["result"]["ml"]["model_version"] == "colab"
