"""REAL end-to-end check: real ML model + real Gemini API + real leaf photos, through the full backend (temporary SQLite DB).

    cd backend && source .venv/bin/activate && python scripts/gemini_real_flows.py
Needs GEMINI_API_KEY (environment or backend/.env) and the git-ignored datasets in ml/data (PlantVillage, COCO).
The key is never printed. The database is a throw-away temp file: your Neon/production data is never touched.
Costs about 4 analysis requests plus a model-list call.
"""
import base64
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

BACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACK))
_tmp = tempfile.mkdtemp(prefix="agroai-realflows-")
os.environ.update(ENVIRONMENT="development", DATABASE_URL=f"sqlite:///{_tmp}/t.db", UPLOAD_DIR=f"{_tmp}/uploads", EMAIL_BACKEND="console",
                  EXPOSE_DEV_OTP="true", ADMIN_EMAIL="admin@example.com", ADMIN_PASSWORD="Throwaway-Admin-Pass-7", BCRYPT_ROUNDS="4",
                  OTP_RESEND_COOLDOWN_SECONDS="0", AI_USER_HOURLY_LIMIT="100")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.ai import registry  # noqa: E402
from app.ai.gemini import GeminiProvider  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services import ml_service  # noqa: E402

DATA = BACK / "ml" / "data"
s = get_settings()
KEY = s.gemini_api_key.get_secret_value()
V = "/api/v1"
FORBIDDEN = ("UNKNOWN", "NO_PLANT", "ml_state", "disagree", "classifier", "our model", "the model predicted")


def first(pattern, n=1):
    return sorted(DATA.glob(pattern))[:n]


def find_by_state(candidates, state):
    svc = ml_service.get_ml_service()
    for name, data in candidates:
        if svc.predict(data)["classification_type"] == state:
            return name, data
    return None


def coco_images(n=40):
    z = zipfile.ZipFile(DATA / "coco" / "val2017.zip")
    names = [x for x in z.namelist() if x.endswith(".jpg")][:n]
    return [(x, z.read(x)) for x in names]


def main() -> int:
    print(f"model={s.gemini_model}  key_configured={bool(KEY.strip())}")
    if not KEY.strip():
        print("STOP: GEMINI_API_KEY is not set (environment or backend/.env). Nothing was sent anywhere."); return 2

    # 1) key + model name validity, using the same REST API the adapter uses
    r = httpx.get(f"{s.gemini_base_url.rstrip('/')}/v1beta/models", params={"pageSize": 200}, headers={"x-goog-api-key": KEY}, timeout=30)
    print(f"1) models.list -> HTTP {r.status_code}")
    if r.status_code != 200:
        print("   key rejected or API unreachable:", (r.json().get("error", {}).get("status") if r.headers.get("content-type", "").startswith("application/json") else r.text[:80])); return 1
    models = {m["name"].split("/", 1)[1]: m.get("supportedGenerationMethods", []) for m in r.json().get("models", [])}
    ok = s.gemini_model in models and "generateContent" in models[s.gemini_model]
    print(f"   key accepted. configured model '{s.gemini_model}' available for generateContent: {ok}")
    if not ok:
        flash = sorted(m for m, meth in models.items() if "generateContent" in meth and "flash" in m and "image" not in m and "tts" not in m)
        print("   models that support generateContent and contain 'flash':", ", ".join(flash[:15]) or "(none)")
        print("   -> set GEMINI_MODEL to one of these and re-run."); return 1

    # 2) pick real images that the real ML model puts in each state
    pv = lambda pat: [(p.name, p.read_bytes()) for p in first(pat, 25)]  # noqa: E731
    picks = {
        "DISEASE": find_by_state(pv("plantvillage/color/Tomato___Early_blight/*"), "DISEASE"),
        "HEALTHY": find_by_state(pv("plantvillage/color/Grape___healthy/*"), "HEALTHY"),
        "UNKNOWN": find_by_state(pv("plantvillage/color/Blueberry___healthy/*") + pv("plantvillage/color/Orange___Haunglongbing*/*"), "UNKNOWN"),
        "NO_PLANT": find_by_state(coco_images(), "NO_PLANT"),
    }
    missing = [k for k, v in picks.items() if v is None]
    if missing:
        print("STOP: could not find real images for ML states:", missing); return 3

    # 3) record what the adapter really sends to Gemini (image bytes, mime, whether ML context is in the prompt)
    sent = []

    class Recording(GeminiProvider):
        def analyze(self, request):
            sent.append(dict(mime=request.image_mime, n=len(request.image or b""), jpeg=(request.image or b"")[:2] == b"\xff\xd8",
                             ml_context="ALREADY identified" in request.prompt, prompt_case=request.prompt.split(".")[0]))
            return super().analyze(request)
    registry.register("gemini", lambda st: Recording(st))

    failures = 0
    with TestClient(app) as c:
        reg = c.post(f"{V}/auth/register", json={"email": "farmer@example.com", "password": "Farmer-Pass-123", "full_name": "Real Flow"}).json()
        h = {"Authorization": "Bearer " + c.post(f"{V}/auth/verify-otp", json={"email": "farmer@example.com", "code": reg["dev_otp"]}).json()["access_token"]}
        print(f"2) AI provider state: {c.get(f'{V}/analyses/summary', headers=h).status_code} (user created in a temp database)")
        for state, (name, data) in picks.items():
            before = len(sent)
            aid = c.post(f"{V}/analyses", headers=h, files={"file": (name, data, "image/jpeg")}).json()["id"]
            resp = c.post(f"{V}/analyses/{aid}/analyze", headers=h)
            b = resp.json()
            res = b.get("result") or {}
            ml, fin = res.get("ml") or {}, res.get("final")
            call = sent[before:]
            shown = [fin.get(k) for k in ("headline", "rejection_reason", "disclaimer", "ai_notes", "plant", "crop", "disease")] if fin else []
            shown += [x for k in ("symptoms", "immediate_actions", "treatment", "prevention", "warnings", "monitoring") for x in (fin or {}).get(k, [])]
            customer_text = " ".join(str(x) for x in shown if x).lower()          # every free-text field the UI renders
            leaks = [w for w in FORBIDDEN if w.lower() in customer_text] + [k for k in ("ai_provider", "ai_model", "ai_error_code") if b.get(k)]
            ok_flow = resp.status_code == 200 and b.get("status") == "completed" and fin is not None and len(call) == 1 and call[0]["n"] > 1000 and call[0]["jpeg"] \
                and call[0]["mime"] == "image/jpeg" and (call[0]["ml_context"] == (state == "DISEASE")) and not leaks and "ai" not in res
            failures += not ok_flow
            print(f"\n[{state}] image={name[:28]}  ML->{ml.get('classification_type')}  HTTP {resp.status_code}  status={b.get('status')}  ai_status={b.get('ai_status') or '(hidden from customers)'}")
            print(f"   gemini calls={len(call)}  image_bytes_sent={call[0]['n'] if call else 0}  mime={call[0]['mime'] if call else None}  jpeg_magic={call[0]['jpeg'] if call else None}  ml_context_in_prompt={call[0]['ml_context'] if call else None}")
            if fin:
                print(f"   customer result: {fin['status']} | {fin['headline']} | plant={fin.get('plant')} disease={fin.get('disease')} severity={fin.get('severity')}")
                print(f"   symptoms={len(fin['symptoms'])} treatment={len(fin['treatment'])} prevention={len(fin['prevention'])} warnings={len(fin['warnings'])}")
            else:
                print("   no final result:", res.get("ai_error"))
            print(f"   internal terms leaked to customer: {leaks or 'none'}   flow_ok={ok_flow}")
    print("\nRESULT:", "ALL FLOWS OK" if not failures else f"{failures} flow(s) did not behave as required")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
