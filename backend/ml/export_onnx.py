"""Export the chosen checkpoint to a small ONNX file + metadata for the backend, and measure it.

    python export_onnx.py --run v1
Writes backend/app/ml/{agro_model.onnx, model_meta.json}  (the ONLY ML files that get deployed).
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

import decision
from common import *  # noqa: F401,F403
from mldata import eval_transform
from train import build_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="v1"); ap.add_argument("--deploy", action="store_true", help="write into backend/app/ml (otherwise reports/<run>/export)"); a = ap.parse_args()
    ck = torch.load(RUNS / a.run / "best.pt", map_location="cpu"); classes = ck["classes"]
    cfg = json.load(open(RUNS / a.run / "thresholds.json"))
    m = build_model(len(classes), pretrained=False); m.load_state_dict(ck["model"]); m.eval()
    if ck.get("smoke") and a.deploy: raise SystemExit("REFUSING to deploy a smoke-test model (it was not really trained).")
    out = APP_ML if a.deploy else REPORTS / a.run / "export"
    out.mkdir(parents=True, exist_ok=True); path = out / "agro_model.onnx"
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(m, dummy, str(path), input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    # parity: ONNX must reproduce PyTorch
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    diff = float(np.abs(m(x).detach().numpy() - sess.run(None, {"image": x.numpy()})[0]).max())
    print(f"ONNX vs PyTorch max |diff| = {diff:.2e}")
    assert diff < 1e-3

    crops = sorted({c.split("___")[0] for c in classes if "___" in c})
    from common import pretty_crop, pretty_condition
    meta = dict(
        model="MobileNetV3-Small (ImageNet transfer learning), 224x224 input", run=a.run, trained_epoch=ck["epoch"],
        input=dict(size=IMG_SIZE, resize=256, mean=list(MEAN), std=list(STD), layout="NCHW float32"),
        classes=[dict(id=c, special=("unknown" if c == decision.UNKNOWN else "non_plant" if c == decision.NON_PLANT else None),
                      crop=None if "___" not in c else pretty_crop(c.split("___")[0]), condition=None if "___" not in c else pretty_condition(c),
                      healthy=c.endswith("___healthy")) for c in classes],
        supported_crops=sorted({pretty_crop(c) for c in crops}),
        decision={k: cfg[k] for k in ("temperature", "t_non_plant", "t_confidence")}, decision_method=cfg["method"])
    json.dump(meta, open(out / "model_meta.json", "w"), indent=1)

    # ---- performance on this machine (CPU, 1 thread, like a small free-tier container) ----
    size_mb = os.path.getsize(path) / 1e6
    t0 = time.perf_counter(); so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s1 = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]); load_ms = (time.perf_counter() - t0) * 1000
    im = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
    xin = eval_transform()(im).unsqueeze(0).numpy()
    for _ in range(5): s1.run(None, {"image": xin})
    ts = []
    for _ in range(60):
        t = time.perf_counter(); s1.run(None, {"image": xin}); ts.append((time.perf_counter() - t) * 1000)
    perf = dict(model_size_mb=round(size_mb, 2), load_time_ms=round(load_ms, 1), inference_ms_median_1thread=round(float(np.median(ts)), 1),
                inference_ms_p95_1thread=round(float(np.percentile(ts, 95)), 1), onnx_vs_torch_max_abs_diff=diff)
    (REPORTS / a.run).mkdir(parents=True, exist_ok=True)
    json.dump(perf, open(REPORTS / a.run / "performance_export.json", "w"), indent=1); print(perf); print("exported to", out)


if __name__ == "__main__":
    main()
