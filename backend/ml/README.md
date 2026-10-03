# AGRO AI — our own crop-health model (Phase 2)

> **Status: code complete and tested locally; the real training run happens on Google Colab Free.**
> No model has been trained yet, so **no accuracy number exists yet** and `backend/app/ml/` is intentionally empty.
> Until a model is installed the app says "analysis model is not installed" — it never makes up a diagnosis.

## The big picture (for beginners)
A photo goes through: **upload checks → our model → one of four answers**

| Answer | Meaning |
|---|---|
| `DISEASE` | a supported crop with a recognised disease |
| `HEALTHY` | a supported crop that looks healthy |
| `UNKNOWN` | looks like a plant, but we can't say what/why confidently (unsupported crop, unseen disease, unclear photo) |
| `NO_PLANT` | not a plant (phone, person, car, food…) |

The model is **MobileNetV3-Small** (about 2.5 M parameters, ~6 MB as a file). We use **transfer learning**: it starts from a network that already understands everyday images and we only teach it crops. It has **36 outputs**: 34 crop-condition classes (e.g. `Tomato___Early_blight`) + `unknown_plant` + `non_plant`.

## Data (three separate collections — never blended into one score)
| Dataset | What it is | Used for |
|---|---|---|
| **PlantVillage** (CC-BY-SA-3.0, 54,305 photos, 38 classes, 14 crops) | clean lab photos on plain backgrounds | main training data + the "lab" test |
| **PlantDoc** (CC-BY-4.0, 2,572 photos) | real-world photos from the internet | trains real-world robustness + the **real-world test** |
| **COCO val2017** (4,742 photos after removing ones with plants) | phones, people, cars, food, buildings… | teaches/tests `NO_PLANT` |

What we found when inspecting them (measured, not assumed):
* **Class imbalance is large**: the biggest class has 36× more images than the smallest → balanced sampling + per-class metrics.
* **PlantVillage is "too easy"**: plain backgrounds, one leaf per photo. Models often score >95% on it and then fail on phone photos in a field. That is why PlantDoc (real photos) is evaluated **separately** and also used in training.
* **Leakage is real**: PlantVillage photographs each leaf ~4 times, so a random split would put near-identical photos in train and test. We group photos of the same leaf (official `leaf-map.json` + number blocks + perceptual hashes) and split **by group**; images near a split border are purged. In PlantDoc, **168 training photos were near-copies of test photos** and were removed.
* **Honest scope (supported crops)**: Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Soybean, Strawberry, Tomato (34 conditions). Blueberry, Raspberry, Squash and Orange are deliberately **unsupported** and used to teach/measure `UNKNOWN`. Everything else is unsupported too.
* **Known limitation of the "lean" data choice**: UNKNOWN is learned from only two seen crops (Orange, Squash) and measured on two never-seen crops (Raspberry for tuning, Blueberry for the final test). There are no generic garden/flower photos, so UNKNOWN on arbitrary plants is weaker than on those crops. Adding Oxford Flowers-102 later would help.

## How `UNKNOWN` and `NO_PLANT` are decided (no made-up thresholds)
`decision.py` (same rules in training evaluation and in the server):
1. softmax with a **temperature T** (fitted on validation data so confidences are trustworthy)
2. `NO_PLANT` if P(non_plant) ≥ **t_non_plant** — the smallest threshold that wrongly calls ≤ 1% of *real plant validation photos* (lab **and** real-world) "not a plant". (A real leaf called "not a plant" is the worse mistake.)
3. `UNKNOWN` if the top class is `unknown_plant` or its probability < **t_confidence** — chosen on validation data by maximising **Youden's J** = (domain-balanced rate of correct+accepted known photos) − (rate of a never-seen crop, Raspberry, wrongly accepted).
4. otherwise `HEALTHY`/`DISEASE` for the winning crop class.
`evaluate.py calibrate` prints the full sweep so you can see the trade-off. All thresholds come from **validation** photos only; **test** photos are only used in the final report.

Why one model with extra classes instead of a separate plant/non-plant model? One small network keeps RAM/startup low on Render Free (one file, one load), and COCO gives plenty of negatives. If the report shows `NO_PLANT` is weak, a tiny second binary model is the next step.

## Where things run
| Where | What |
|---|---|
| **Your Mac** | development, unit/API tests, tiny smoke tests (`train.py --smoke`) |
| **Google Colab Free** | download data, full training, evaluation, confusion matrices, export |
| **Render (later)** | only `agro_model.onnx` + `model_meta.json` (~6 MB). **Never the datasets.** |

## Train on Colab — step by step
1. Mac: `cd backend/ml && ./make_colab_bundle.sh` → creates `agroai_ml_bundle.zip`.
2. Go to https://colab.research.google.com → *File ▸ Upload notebook* → `backend/ml/notebooks/agroai_phase2_colab.ipynb`.
3. *Runtime ▸ Change runtime type ▸ **T4 GPU*** → Save.
4. *Runtime ▸ Run all* (it will ask you to allow Google Drive and to upload the zip). It downloads data, prepares the split, trains, evaluates and exports. Progress is saved in Drive (`MyDrive/agroai_phase2`); if Colab disconnects re-run and it **resumes**.
5. The last cell downloads `agroai_phase2_results.zip`.
6. Mac: `cd backend/ml && source .venv-train/bin/activate && python install_model.py ~/Downloads/agroai_phase2_results.zip` — it validates the files and copies the model into `backend/app/ml/`.
7. Run the backend tests again (`cd backend && python -m pytest -q`) and try the Analyze page.

The notebook shows accuracy / precision / recall / F1 / confusion matrix **separately** for PlantVillage, PlantDoc, COCO and unsupported plants, plus the weakest classes and common confusions. Training time on Colab has not been measured yet.

## Files
| File | Purpose |
|---|---|
| `common.py` | paths, constants, PlantDoc→PlantVillage label map, pretty names |
| `prepare_data.py` | builds `data/manifest.csv` (labels + leak-free train/val/test) |
| `mldata.py` | dataset class + augmentations |
| `train.py` | transfer-learning training (resume, early stopping, LR schedule, GPU mixed precision, `--smoke`) |
| `decision.py` | the DISEASE/HEALTHY/UNKNOWN/NO_PLANT rules |
| `evaluate.py` | `logits` → `calibrate` (validation) → `report` (test, per dataset) |
| `export_onnx.py` | ONNX export, parity check vs PyTorch, size/speed measurements (`--deploy` writes into the app) |
| `install_model.py` | validates & installs the Colab download on your Mac |
| `make_colab_bundle.sh`, `notebooks/…ipynb` | Colab workflow |
| `backend/app/services/ml_service.py` | server-side inference (ONNX Runtime + NumPy + Pillow only) |

`data/` and `runs/` are git-ignored. A smoke-test model can never be deployed (`export_onnx.py --deploy` refuses it).
