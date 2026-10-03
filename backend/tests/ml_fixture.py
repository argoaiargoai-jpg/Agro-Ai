"""A tiny ONNX 'model' used ONLY to test the ML plumbing (loading, preprocessing, thresholds, API).

It is NOT a trained model and says nothing about crops: it maps average image colour to a class by construction
(red -> Early Blight, green -> healthy, blue -> non_plant, flat grey -> unknown). Real accuracy is measured elsewhere.
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CLASSES = [
    dict(id="Tomato___Early_blight", special=None, crop="Tomato", condition="Early Blight", healthy=False),
    dict(id="Tomato___healthy", special=None, crop="Tomato", condition=None, healthy=True),
    dict(id="__unknown_plant__", special="unknown", crop=None, condition=None, healthy=False),
    dict(id="__non_plant__", special="non_plant", crop=None, condition=None, healthy=False),
]
W = np.array([[12, -6, -6], [-6, 12, -6], [0, 0, 0], [-6, -6, 12]], dtype=np.float32)   # rows: classes, cols: R,G,B
B = np.array([0, 0, 3, 0], dtype=np.float32)                                              # unknown wins on flat grey


def build(dest: Path, t_conf=0.5, t_np=0.6, temperature=1.0) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4])
    nodes = [helper.make_node("GlobalAveragePool", ["image"], ["g"]), helper.make_node("Flatten", ["g"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "W", "B"], ["logits"], transB=1)]
    g = helper.make_graph(nodes, "plumbing", [x], [y], [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.checker.check_model(m); onnx.save(m, str(dest / "agro_model.onnx"))
    meta = dict(model="PLUMBING TEST FIXTURE (not a trained model)", run="test-fixture", trained_epoch=0,
                input=dict(size=224, resize=256, mean=[0, 0, 0], std=[1, 1, 1], layout="NCHW float32"),
                classes=CLASSES, supported_crops=["Tomato"],
                decision=dict(temperature=temperature, t_non_plant=t_np, t_confidence=t_conf), decision_method={})
    (dest / "model_meta.json").write_text(json.dumps(meta))
    return dest
