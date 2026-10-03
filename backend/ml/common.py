"""Shared constants for the AGRO AI ML pipeline (used by training, evaluation and export)."""
import os
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
DATA = ML_DIR / "data"
PV_DIR = DATA / "plantvillage"
PD_DIR = DATA / "plantdoc" / "repo"
COCO_DIR = DATA / "coco"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.csv"
RUNS = Path(os.environ.get("AGRO_RUNS_DIR", ML_DIR / "runs"))        # on Colab: a Google Drive folder, so a disconnect loses nothing
REPORTS = Path(os.environ.get("AGRO_REPORTS_DIR", ML_DIR / "reports"))
APP_ML = ML_DIR.parent / "app" / "ml"          # the ONLY thing deployed: model + metadata

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# PlantVillage crops we deliberately treat as NOT supported (they teach/measure the UNKNOWN outcome).
UNKNOWN_SEEN_CROPS = ("Orange", "Squash")      # used to TRAIN the unknown_plant class
UNKNOWN_NOVEL_VAL_CROP = "Raspberry"           # never trained on; used to pick the rejection threshold
UNKNOWN_NOVEL_TEST_CROP = "Blueberry"          # never trained on, never used for tuning; final open-set test

UNKNOWN = "__unknown_plant__"
NON_PLANT = "__non_plant__"

# PlantDoc folder -> PlantVillage class (None = a crop we do not support -> UNKNOWN)
PLANTDOC_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": None,
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": None,
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": None,
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def pretty_crop(raw: str) -> str:
    return {"Corn_(maize)": "Corn", "Cherry_(including_sour)": "Cherry", "Pepper,_bell": "Bell Pepper"}.get(raw, raw)


def pretty_condition(cls: str) -> str | None:
    """'Tomato___Early_blight' -> 'Early Blight'; healthy -> None."""
    cond = cls.split("___", 1)[1]
    if cond == "healthy":
        return None
    cond = {
        "Cercospora_leaf_spot Gray_leaf_spot": "Gray Leaf Spot",
        "Common_rust_": "Common Rust",
        "Esca_(Black_Measles)": "Esca (Black Measles)",
        "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf Blight (Isariopsis Leaf Spot)",
        "Spider_mites Two-spotted_spider_mite": "Spider Mites (Two-spotted)",
        "Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
        "Tomato_mosaic_virus": "Mosaic Virus",
    }.get(cond, cond)
    return cond.replace("_", " ").strip().title() if cond == cond.lower() or "_" in cond else cond
