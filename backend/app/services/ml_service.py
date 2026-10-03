"""Our own crop-health model: image -> structured ML result.

Lightweight on purpose: ONNX Runtime + NumPy + Pillow only (no PyTorch/TensorFlow in the deployed app).
One ONNX session is loaded lazily and shared by all requests.

Outcomes:  DISEASE | HEALTHY | UNKNOWN | NO_PLANT   (see ml/README.md for how the thresholds were chosen)
"""
import io
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import get_settings
from app.core.errors import AppError

log = logging.getLogger("agroai.ml")

UNKNOWN_ID = "__unknown_plant__"
NON_PLANT_ID = "__non_plant__"
MIN_SIDE = 48

MESSAGES = {
    "DISEASE": "A disease was detected on a supported crop.",
    "HEALTHY": "The leaf looks healthy for a supported crop.",
    "UNKNOWN": "This looks like a plant, but we can't identify it or its condition confidently. It may be an unsupported crop or disease, or the photo may be unclear.",
    "NO_PLANT": "This image doesn't appear to show a plant or leaf.",
}


def softmax(z: np.ndarray, temperature: float) -> np.ndarray:
    z = z.astype(np.float64) / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


class MLService:
    def __init__(self, model_dir: Path, threads: int, max_concurrency: int):
        import onnxruntime as ort  # imported lazily so the rest of the API works without the ML extras

        t0 = time.perf_counter()
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "agro_model.onnx"), so, providers=["CPUExecutionProvider"])
        self.classes = self.meta["classes"]
        self.ids = [c["id"] for c in self.classes]
        self.i_unknown, self.i_non_plant = self.ids.index(UNKNOWN_ID), self.ids.index(NON_PLANT_ID)
        d = self.meta["decision"]
        self.T, self.t_np, self.t_conf = d["temperature"], d["t_non_plant"], d["t_confidence"]
        inp = self.meta["input"]
        self.size, self.resize = inp["size"], inp["resize"]
        self.mean = np.array(inp["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(inp["std"], dtype=np.float32).reshape(3, 1, 1)
        self.gate = threading.BoundedSemaphore(max_concurrency)
        self.load_ms = (time.perf_counter() - t0) * 1000
        log.info("ML model loaded in %.0f ms (%d outputs)", self.load_ms, len(self.classes))

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, data: bytes) -> np.ndarray:
        """bytes -> normalised float32 tensor (1,3,224,224). Mirrors training: resize short side 256, centre-crop 224."""
        Image.MAX_IMAGE_PIXELS = get_settings().ml_max_pixels
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                im = ImageOps.exif_transpose(im)
                if im.width * im.height > get_settings().ml_max_pixels:
                    raise AppError(413, "image_too_large", "That image has too many pixels. Please use a smaller photo.")
                im = im.convert("RGB")
        except AppError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, SyntaxError) as exc:
            raise AppError(422, "invalid_image", "We couldn't read that image. It may be corrupt or incomplete.") from exc
        if min(im.size) < MIN_SIDE:
            raise AppError(422, "image_too_small", f"The image is too small to analyze (minimum {MIN_SIDE}px on the short side).")
        w, h = im.size
        scale = self.resize / min(w, h)
        im = im.resize((max(round(w * scale), self.size), max(round(h * scale), self.size)), Image.BILINEAR)
        w, h = im.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        im = im.crop((left, top, left + self.size, top + self.size))
        x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None]

    # ---------------------------------------------------------------- inference
    def predict(self, data: bytes) -> dict:
        x = self.preprocess(data)
        t0 = time.perf_counter()
        with self.gate:
            logits = self.session.run(None, {"image": x})[0][0]
        ms = (time.perf_counter() - t0) * 1000
        return self.decide(logits, ms)

    def decide(self, logits: np.ndarray, inference_ms: float = 0.0) -> dict:
        p = softmax(logits, self.T)
        top = int(p.argmax())
        base = dict(model_version=self.meta["run"], inference_ms=round(inference_ms, 1),
                    scores=dict(unknown_plant=round(float(p[self.i_unknown]), 4), non_plant=round(float(p[self.i_non_plant]), 4)))
        if p[self.i_non_plant] >= self.t_np:
            kind, conf, cls = "NO_PLANT", float(p[self.i_non_plant]), None
        elif top in (self.i_unknown, self.i_non_plant) or p[top] < self.t_conf:
            kind, conf, cls = "UNKNOWN", float(p[top]), None
        else:
            cls = self.classes[top]
            kind, conf = ("HEALTHY" if cls["healthy"] else "DISEASE"), float(p[top])
        return dict(classification_type=kind, crop=cls["crop"] if cls else None,
                    disease=cls["condition"] if cls and kind == "DISEASE" else None,
                    confidence=round(conf, 4), message=MESSAGES[kind], supported_crops=self.meta["supported_crops"], **base)

    def info(self) -> dict:
        return dict(available=True, model=self.meta["model"], version=self.meta["run"], supported_crops=self.meta["supported_crops"],
                    supported_conditions=sorted({f"{c['crop']}: {c['condition'] or 'Healthy'}" for c in self.classes if c["crop"]}),
                    outcomes=["DISEASE", "HEALTHY", "UNKNOWN", "NO_PLANT"])


_service: MLService | None = None
_lock = threading.Lock()


def get_ml_service() -> MLService:
    """Load once, on first use. Raises 503 (never fakes a result) if the model files are not present."""
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                s = get_settings()
                d = s.ml_path
                if not (d / "agro_model.onnx").exists() or not (d / "model_meta.json").exists():
                    raise AppError(503, "ml_unavailable", "The analysis model is not installed on this server.")
                try:
                    _service = MLService(d, s.ml_threads, s.ml_max_concurrency)
                except Exception as exc:  # noqa: BLE001
                    log.exception("ML model failed to load")
                    raise AppError(503, "ml_unavailable", "The analysis model could not be loaded.") from exc
    return _service


def reset_ml_service() -> None:  # used by tests
    global _service
    _service = None
