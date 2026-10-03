"""Run the INSTALLED model on real held-out test images and record what actually happens.

    cd backend && python ml/verify_installed_model.py          (needs the git-ignored local datasets in ml/data)
This is a smoke/behaviour check on small seeded samples, NOT a replacement for the Colab evaluation in reports/colab/metrics.json.
"""
import csv, io, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

BACK = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(BACK))
from app.core.errors import AppError  # noqa: E402
from app.services import ml_service  # noqa: E402

DATA = BACK / "ml" / "data"
random.seed(7)
svc = ml_service.get_ml_service()
by_key = {(c["crop"], c["condition"]): c["id"] for c in svc.classes if c["crop"]}
rows = list(csv.DictReader(open(DATA / "manifest.csv")))
test = [r for r in rows if r["split"] == "test"]


def sample(pred, n):
    pool = [r for r in test if pred(r)]
    return random.sample(pool, min(n, len(pool)))


def run(path_or_bytes):
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else (DATA / path_or_bytes).read_bytes()
    r = svc.predict(data)
    pid = by_key.get((r["crop"], r["disease"])) if r["crop"] else None
    return r["classification_type"], pid, r["confidence"]


def evaluate(name, items, expect=None):
    out = Counter(); correct = 0; confs = []
    for r in items:
        kind, pid, conf = run(r["path"]); out[kind] += 1; confs.append(conf)
        if expect == "same_class" and pid == r["label_out"]: correct += 1
    res = dict(n=len(items), outcomes=dict(out), median_confidence=round(sorted(confs)[len(confs) // 2], 3) if confs else None)
    if expect == "same_class": res["exact_crop_and_condition_correct"] = correct
    print(f"{name:52s} n={res['n']:3d} {dict(out)}" + (f"  exact-correct={correct}" if expect else "")); return res


R = {}
print("== 1-2. PlantVillage (lab) held-out test images")
dis = [("Tomato___Early_blight",), ("Tomato___Late_blight",), ("Apple___Apple_scab",), ("Grape___Black_rot",), ("Corn_(maize)___Common_rust_",), ("Potato___Early_blight",), ("Peach___Bacterial_spot",), ("Pepper,_bell___Bacterial_spot",)]
R["pv_disease"] = {c[0]: evaluate(f"  disease  {c[0]}", sample(lambda r, c=c: r["source"] == "plantvillage" and r["label_out"] == c[0], 12), "same_class") for c in dis}
hlt = ["Tomato___healthy", "Apple___healthy", "Grape___healthy", "Corn_(maize)___healthy", "Potato___healthy", "Strawberry___healthy", "Cherry_(including_sour)___healthy"]
R["pv_healthy"] = {c: evaluate(f"  healthy  {c}", sample(lambda r, c=c: r["source"] == "plantvillage" and r["label_out"] == c, 12), "same_class") for c in hlt}
print("== 3-4. PlantDoc (real-world) held-out test images")
pd_known = [r for r in test if r["source"] == "plantdoc" and r["role"] == "known"]
R["plantdoc_disease"] = evaluate("  PlantDoc disease (all test)", [r for r in pd_known if not r["label_out"].endswith("___healthy")], "same_class")
R["plantdoc_healthy"] = evaluate("  PlantDoc healthy (all test)", [r for r in pd_known if r["label_out"].endswith("___healthy")], "same_class")
print("== 5. unsupported plants (must NOT become a confident disease/healthy)")
R["unsupported_blueberry_pv"] = evaluate("  Blueberry (never trained)", sample(lambda r: r["source"] == "plantvillage" and r["role"] == "unknown_novel", 60))
R["unsupported_orange_squash_pv"] = evaluate("  Orange / Squash (unsupported crops)", sample(lambda r: r["source"] == "plantvillage" and r["role"] == "unknown_seen", 60))
R["unsupported_plantdoc"] = evaluate("  PlantDoc unsupported crops (all test)", [r for r in test if r["source"] == "plantdoc" and r["role"] in ("unknown_novel", "unknown_seen")])
print("== 6. non-plant (COCO test): phones, laptops, people, cars, food")
ann = json.load(open(DATA / "coco" / "annotations" / "instances_val2017.json")); cat = {c["id"]: c["name"] for c in ann["categories"]}
img_cats = defaultdict(set)
for a in ann["annotations"]: img_cats[a["image_id"]].add(cat[a["category_id"]])
fname2id = {i["file_name"]: i["id"] for i in ann["images"]}
coco_test = [r for r in test if r["source"] == "coco"]
R["non_plant"] = {}
for label, names in [("cell phone", {"cell phone"}), ("laptop", {"laptop"}), ("person", {"person"}), ("car", {"car"}), ("food (pizza/cake/sandwich)", {"pizza", "cake", "sandwich", "donut"}), ("building/street (traffic light, bench)", {"traffic light", "bench", "fire hydrant"})]:
    pool = [r for r in coco_test if img_cats[fname2id[Path(r["path"]).name]] & names]
    R["non_plant"][label] = evaluate(f"  {label}", random.sample(pool, min(15, len(pool))))
R["non_plant"]["random sample"] = evaluate("  random COCO test photos", sample(lambda r: r["source"] == "coco", 60))
print("== 7. poor-quality images (degraded copies of correctly classified PlantVillage disease photos)")
base = [r for r in sample(lambda r: r["source"] == "plantvillage" and r["role"] == "known" and not r["label_out"].endswith("healthy"), 120)
        if run(r["path"])[1] == r["label_out"]][:30]
def degrade(im, kind):
    if kind == "blur": return im.filter(ImageFilter.GaussianBlur(7))
    if kind == "tiny 48px": return im.resize((48, 48)).resize(im.size)
    if kind == "very dark": return ImageEnhance.Brightness(im).enhance(0.12)
    if kind == "over-exposed": return ImageEnhance.Brightness(im).enhance(4.0)
    if kind == "heavy jpeg q=3":
        b = io.BytesIO(); im.save(b, "JPEG", quality=3); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    if kind == "noise": return Image.blend(im, Image.effect_noise(im.size, 90).convert("RGB"), 0.6)
R["poor_quality"] = {}
for kind in ("blur", "tiny 48px", "very dark", "over-exposed", "heavy jpeg q=3", "noise"):
    out = Counter(); same = 0
    for r in base:
        im = degrade(Image.open(DATA / r["path"]).convert("RGB"), kind); b = io.BytesIO(); im.save(b, "PNG"); k, pid, _ = run(b.getvalue()); out[k] += 1; same += pid == r["label_out"]
    R["poor_quality"][kind] = dict(n=len(base), outcomes=dict(out), still_correct=same); print(f"  {kind:16s} n={len(base)} {dict(out)}  still-exactly-correct={same}")
print("== 8. corrupted / invalid input (must be a clean error, not an ML state)")
good = (DATA / base[0]["path"]).read_bytes()
R["corrupted"] = {}
for label, data in [("truncated JPEG", good[:len(good) // 3]), ("random bytes", random.randbytes(5000)), ("empty", b""), ("PNG header + garbage", b"\x89PNG\r\n\x1a\n" + b"x" * 300), ("tiny 20x20", (lambda b: (Image.new("RGB", (20, 20)).save(b, "PNG"), b.getvalue())[1])(io.BytesIO()))]:
    try: res = f"UNEXPECTED ML STATE {svc.predict(data)['classification_type']}"
    except AppError as e: res = f"{e.status_code} {e.code}"
    R["corrupted"][label] = res; print(f"  {label:24s} -> {res}")
json.dump(R, open(BACK / "ml" / "reports" / "colab" / "real_image_checks.json", "w"), indent=1)
print("\nsaved ml/reports/colab/real_image_checks.json")
