"""Turn raw model scores into one of DISEASE / HEALTHY / UNKNOWN / NO_PLANT.

Pure numpy so the exact same rules run in evaluation, calibration and the FastAPI backend.

Rules (thresholds come from VALIDATION data only, see calibrate.py):
  1. p = softmax(logits / T)                       T = temperature fitted on validation data (calibration)
  2. if p[non_plant] >= t_non_plant       -> NO_PLANT
  3. else if the top class is 'unknown_plant', or its probability < t_confidence  -> UNKNOWN
  4. else the top class is a supported crop class -> HEALTHY or DISEASE
"""
import numpy as np

UNKNOWN = "__unknown_plant__"
NON_PLANT = "__non_plant__"


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def decide(logits: np.ndarray, classes: list[str], cfg: dict) -> list[dict]:
    """logits: (N, C). Returns one dict per row: outcome, class (or None), confidence, probs of the 2 special classes."""
    p = softmax(logits, cfg["temperature"])
    i_unk, i_np = classes.index(UNKNOWN), classes.index(NON_PLANT)
    out = []
    for row in p:
        top = int(row.argmax())
        if row[i_np] >= cfg["t_non_plant"]:
            out.append(dict(outcome="NO_PLANT", cls=None, confidence=float(row[i_np]), p_unknown=float(row[i_unk]), p_non_plant=float(row[i_np])))
        elif top == i_unk or row[top] < cfg["t_confidence"] or top == i_np:
            out.append(dict(outcome="UNKNOWN", cls=None, confidence=float(row[top]), p_unknown=float(row[i_unk]), p_non_plant=float(row[i_np])))
        else:
            name = classes[top]
            out.append(dict(outcome="HEALTHY" if name.endswith("___healthy") else "DISEASE", cls=name,
                            confidence=float(row[top]), p_unknown=float(row[i_unk]), p_non_plant=float(row[i_np])))
    return out
