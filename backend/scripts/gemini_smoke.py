"""One-off check of the REAL Gemini API with YOUR key. Costs a couple of tiny requests.

    cd backend && source .venv/bin/activate && GEMINI_API_KEY=... python scripts/gemini_smoke.py
(or put the key in backend/.env). The key is never printed. Tests never call Gemini; this script is the only thing that does.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.ai.base import ProviderError  # noqa: E402
from app.ai.gemini import GeminiProvider  # noqa: E402
from app.ai.prompts import build_request  # noqa: E402
from app.ai.schemas import AIAnalysis  # noqa: E402
from app.core.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    p = GeminiProvider(s)
    print(f"model={s.gemini_model}  base={s.gemini_base_url}  key_configured={p.is_configured()}")
    if not p.is_configured():
        print("FAIL: GEMINI_API_KEY is not set"); return 2
    try:
        print("1) ping:", p.ping())
        buf = io.BytesIO(); Image.new("RGB", (320, 240), (40, 120, 40)).save(buf, "JPEG")
        req, case = build_request({"classification_type": "UNKNOWN"}, buf.getvalue(), "image/jpeg")
        r = p.analyze(req)
        a = AIAnalysis.model_validate(r.data)
        print(f"2) structured analysis OK ({case}): plant_present={a.plant_present} health_status={a.health_status} latency={r.latency_ms:.0f}ms usage={r.usage}")
    except ProviderError as e:
        print(f"FAIL [{e.ai_status}] {e.user_message} | detail: {e.detail}"); return 1
    print("SUCCESS: the Gemini adapter works with your key and model."); return 0


if __name__ == "__main__":
    raise SystemExit(main())
