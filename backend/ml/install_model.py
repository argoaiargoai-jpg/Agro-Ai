"""Install the model you downloaded from Colab into the backend (run this on your Mac).

    python install_model.py ~/Downloads/agroai_phase2_results.zip

It checks the files before copying, so a broken or half-trained download cannot silently become the "real" model.
"""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from common import APP_ML, REPORTS


def main(zip_path: str):
    tmp = Path(tempfile.mkdtemp())
    zipfile.ZipFile(zip_path).extractall(tmp)
    root = next(tmp.rglob("agro_model.onnx")).parent
    meta = json.loads((root / "model_meta.json").read_text())
    metrics = json.loads(next(tmp.rglob("metrics.json")).read_text())
    assert meta["classes"] and meta["decision"]["t_confidence"] > 0, "metadata incomplete"
    classes = json.loads((root / "classes.json").read_text()); thr = json.loads((root / "thresholds.json").read_text())
    assert classes == [c["id"] for c in meta["classes"]], "classes.json does not match model_meta.json"
    assert all(thr[k] == meta["decision"][k] for k in ("temperature", "t_non_plant", "t_confidence")), "thresholds.json does not match model_meta.json"
    import onnxruntime as ort
    s = ort.InferenceSession(str(root / "agro_model.onnx"), providers=["CPUExecutionProvider"])
    out = s.run(None, {"image": np.random.rand(1, 3, meta["input"]["size"], meta["input"]["size"]).astype("float32")})[0]
    assert out.shape[1] == len(meta["classes"]), "model outputs do not match the class list"
    a = metrics["A_plantvillage_test_known_classes"]; b = metrics["B_plantdoc_test_real_world_known_classes"]
    print(f"Model run '{meta['run']}' OK. PlantVillage test acc {a['accuracy_raw_argmax']} | PlantDoc (real-world) test acc {b['accuracy_raw_argmax']}")
    APP_ML.mkdir(parents=True, exist_ok=True)
    # Runtime needs only the model + its metadata (classes and thresholds are embedded in model_meta.json).
    # classes.json / thresholds.json are installed too, as tiny audit copies that tests check for consistency.
    for f in ("agro_model.onnx", "model_meta.json", "classes.json", "thresholds.json"): shutil.copy(root / f, APP_ML / f)
    dest = REPORTS / meta["run"]; dest.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".png", ".md") and f.name != "model_meta.json": shutil.copy(f, dest / f.name)
    print("Installed into", APP_ML, "| reports in", dest)


if __name__ == "__main__":
    main(sys.argv[1])
