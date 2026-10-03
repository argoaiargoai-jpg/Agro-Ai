# AGRO AI

Crop-health web app: a farmer uploads a leaf photo and gets a diagnosis plus practical guidance.

> **ML status: model installed.** Our own MobileNetV3-Small model (trained on Google Colab) is installed in `backend/app/ml/`.
> Results are reported **per dataset, never as one blended accuracy** (section 7). Real-world photos are the weak spot: most
> PlantDoc images end as `UNKNOWN`. Real Gemini is still unverified (no key was available).

---

## 1. Architecture

```
React (Vite)  ──HTTPS──▶  FastAPI  ──▶  SQL database (SQLite dev / PostgreSQL prod)
 Netlify                   Render        users · sessions · analyses · settings · audit log
                              │
                              ├─ upload checks (type, decode, size, pixels)
                              ├─ OUR ML model  (ONNX Runtime, CPU)   → DISEASE | HEALTHY | UNKNOWN | NO_PLANT
                              └─ AI guidance provider (Gemini, behind an interface) → structured advice
```

| Folder | What it is |
|---|---|
| `frontend/` | React 18 + Vite app (green/white design system, no UI framework) |
| `backend/app/` | FastAPI: `api/` routes, `services/` logic, `models/` tables, `schemas/`, `core/` config+security, `ai/` provider layer |
| `backend/app/ml/` | **Only** the deployed model: `agro_model.onnx` + `model_meta.json` (installed Colab model, 6.23 MB) |
| `backend/ml/` | Training & evaluation code, Colab notebook. Datasets/checkpoints live in git-ignored `data/` and `runs/` and are **never deployed** |
| `backend/migrations/` | Alembic migrations (`0001` initial schema, `0002` AI guidance columns) |
| `netlify.toml`, `render.yaml` | Deployment configuration (nothing is deployed yet) |

**How an analysis works:** upload → validated and stored → our model classifies it (`DISEASE/HEALTHY/UNKNOWN/NO_PLANT`) →
the selected AI provider adds explanation/advice (for `DISEASE` the identity stays ours; for the other states the AI looks
independently) → one structured report is saved and shown. The customer sees a single unified AGRO AI result; the internal
ML/AI details are stored for debugging, and are returned by the API and shown in the UI **only to administrators**
(`schemas/analysis.py: present()` removes provider output, disagreement notes and provider/model names from customer responses).

## 2. Local development

Requires Python 3.12+ (tested on 3.14) and Node 20+.

```bash
# backend  (http://localhost:8000, API docs at /docs in development)
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # then edit: see "First administrator" below
uvicorn app.main:app --reload --port 8000

# frontend (http://localhost:5173, proxies /api to the backend)
cd frontend
npm install
npm run dev
```

**First administrator:** set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in `backend/.env` (there is no default password); the account is
created on first start. For local email verification set `EXPOSE_DEV_OTP=true` (the code is shown on screen and logged).
Development creates tables automatically; production uses migrations (section 5).

## 3. Environment variables

Every variable is documented, with safe placeholders only, in **`backend/.env.example`** (and `frontend/.env.example`).
A test (`test_every_setting_is_documented_in_env_example`) fails if a setting is added without documentation.

| Group | Variables |
|---|---|
| Core / security | `ENVIRONMENT`, `SECRET_KEY` (required in production, ≥32 chars), `BCRYPT_ROUNDS` |
| Database | `DATABASE_URL` (SQLite dev; hosted PostgreSQL in production, `postgres://` is converted automatically) |
| URLs / CORS | `FRONTEND_URL`, `BACKEND_URL`, `CORS_ORIGINS` (exact origins, never `*`), `CORS_ORIGIN_REGEX` (deploy previews) |
| Sessions | `ACCESS_TOKEN_MINUTES`, `REFRESH_TOKEN_DAYS`, `COOKIE_SECURE`, `COOKIE_SAMESITE`, `MAX_FAILED_LOGINS`, `LOCKOUT_MINUTES` |
| Email / OTP | `EMAIL_BACKEND`, `SMTP_*`, `EMAIL_FROM`, `OTP_*`, `EXPOSE_DEV_OTP` |
| Google sign-in | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` |
| Admin bootstrap | `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_NAME` |
| AI guidance | `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_BASE_URL`, `AI_TIMEOUT_SECONDS`, `AI_MAX_RETRIES`, `AI_MAX_CONCURRENCY`, `AI_MAX_IMAGE_SIDE`, `AI_USER_HOURLY_LIMIT` |
| ML | `ML_MODEL_DIR`, `ML_THREADS`, `ML_MAX_CONCURRENCY`, `ML_MAX_PIXELS` |
| Uploads | `UPLOAD_DIR`, `MAX_UPLOAD_MB` |
| Frontend (build time) | `VITE_API_URL` — **the only variable the browser ever sees; never put a secret in a `VITE_` variable** (the build refuses to run if you do) |

In `ENVIRONMENT=production` the server **refuses to start** on an unsafe configuration: weak/default `SECRET_KEY`, non-secure cookies,
a known default admin password, non-HTTPS URLs/origins, or a SQLite database.

## 4. Testing

```bash
cd backend && source .venv/bin/activate && python -m pytest -q       # backend
cd frontend && npm test && npm run build                              # frontend
```

Latest verified run (groups: auth/OTP, Google OAuth, ML plumbing, AI workflow/Gemini adapter/admin, upload security, production config/CORS/cookies, migrations):

| Suite | Result |
|---|---|
| Backend (pytest) | **376 passed** |
| Frontend unit tests (vitest) | **10 passed** |
| Frontend production build | **passes** |

Tests never call Gemini (the AI is mocked; `tests/test_gemini_image_payload.py` checks that the real adapter sends the actual image bytes, MIME type and prompt in every ML case). Most tests use a tiny labelled fixture model (`tests/ml_fixture.py`); `tests/test_real_model.py` (22 tests) checks the installed real model, its exact thresholds, `/ml/info` and the four-case workflow.

## 5. Database migrations

```bash
cd backend
alembic upgrade head          # apply all
alembic downgrade 0001        # step back (Phase 1 data is preserved)
alembic check                 # confirms migrations match the models
```
`0002_ai_guidance_columns` only adds **nullable** columns to `analyses`, so pre-existing analyses keep working. Upgrade → downgrade → upgrade
with data present is covered by `tests/test_migrations.py`. Production (`render.yaml`) runs `alembic upgrade head` on every start.

## 6. AI guidance (Gemini)

* The provider layer is `backend/app/ai/`: the workflow depends on the `AIProvider` interface; `gemini.py` is the only Gemini-specific file.
* **The API key is backend-only:** set `GEMINI_API_KEY` in the server environment. It is never stored in the database, returned by an API, logged,
  or shown in the admin console, and the frontend build contains no key.
* **Admin ▸ AI provider** selects the provider, switches guidance on/off, shows whether a key is configured and has a *Test connection* button.
* **Verify your real key once** (the automated tests never call Gemini): `cd backend && GEMINI_API_KEY=... python scripts/gemini_smoke.py`.
  **Real Gemini has not been verified in this repository** (no key was available); the model name `GEMINI_MODEL` follows Google's current docs and is configurable.
* Failure handling: timeouts, rate limits, 5xx, malformed/blocked answers, bad key, disabled and not-configured are separate states; the ML result is
  always kept and guidance can be retried without re-running the model. Per-user hourly limit, retry/force cooldowns and atomic duplicate-request
  protection bound provider calls (there are no automatic retry loops in the frontend).

## 7. ML model

Trained on **Google Colab Free** (`backend/ml/notebooks/agroai_phase2_colab.ipynb`, see `backend/ml/README.md`); installed with
`python backend/ml/install_model.py <agroai_phase2_results.zip>`. MobileNetV3-Small, 224x224, ONNX 6.23 MB, 36 outputs
(34 crop-condition classes + unknown plant + non-plant). Calibrated thresholds: temperature 0.5581, `t_non_plant` 0.445, `t_confidence` 0.92.

Held-out results from the Colab run, **kept separate on purpose** (do not average them):

| Dataset | Raw top-1 | Final pipeline | Accuracy when answered | Coverage |
|---|---|---|---|---|
| PlantVillage (lab photos, n=6346) | 98.35% | 91.90% | 99.73% | 92.15% (macro F1 97.84%) |
| PlantDoc (real-world, n=212) | 62.74% | 32.55% | 79.31% | 41.04% |

| Rejection test | Result |
|---|---|
| COCO non-plant images → `NO_PLANT` (n=948) | 99.05% recall |
| False `NO_PLANT` on real plants | PlantVillage 0%, PlantDoc 0.47% |
| Unsupported crops → not diagnosed | Blueberry 96.01%, Orange/Squash 100%, PlantDoc unsupported 79.17% |

**Limitations:** only PlantVillage meets an 80% bar; on real-world photos most images are answered `UNKNOWN` (by design, instead of guessing),
and some unsupported crops (about 4% of Blueberry, 21% of PlantDoc-unsupported) are still accepted as `HEALTHY`.
Inference (M1 Mac, runtime-only install): ~103 MB RSS idle, ~165 MB peak after 15 analyses, ~23 ms model time, ~190 ms median request.
Render Free's CPU is slower. Reports live in `backend/ml/reports/colab/`.

## 8. Security overview

Uploads: JPEG/PNG/WebP decided by file *content*, fully decoded (corrupt files rejected), pixel/dimension limits read from the header before decoding
(decompression-bomb safe), 48 px minimum, size limit enforced *before* the body is buffered, random on-disk names, sanitised display names, files served only to their owner
with `nosniff`. Auth: 15-minute access token held in memory, rotating hashed refresh token in an `HttpOnly` cookie (`Secure` + configurable `SameSite`), reuse detection,
lockout, OTP expiry/attempt limits/resend cooldown, server-side role checks. API: explicit-origin CORS only, security headers, no stack traces/DB errors/provider details in responses.

## 9. Netlify (frontend) — prepared, not deployed

`netlify.toml` (repo root) sets `base=frontend`, `publish=dist`, the single-page-app redirect (`/* → /index.html`, so refreshing `/admin` or `/analysis/…` works) and security headers.

1. New site from the Git repo (Netlify reads `netlify.toml`).
2. Environment variable: `VITE_API_URL=https://<your-render-service>.onrender.com` (HTTPS only, no trailing slash, no `/api/v1`).
3. Deploy, then add the site's URL to the backend's `CORS_ORIGINS` and `FRONTEND_URL`.

## 10. Render (backend) — prepared, not deployed

`render.yaml` is a Blueprint (free plan, `autoDeployTrigger: off`): build `pip install -r requirements.txt`, start
`alembic upgrade head && uvicorn app.main:app --workers 1 …`, health check `/api/v1/health`. Fill the secret variables (`sync: false`) in the dashboard.
Only runtime packages are installed; datasets, checkpoints, training libraries and notebooks are not part of the running service.

## 11. Production workflow (when you decide to deploy)

1. Create a hosted PostgreSQL database; copy its URL. 2. Create the Render service from `render.yaml`; set `DATABASE_URL`, `FRONTEND_URL`, `BACKEND_URL`,
`CORS_ORIGINS`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, SMTP settings, `GEMINI_API_KEY`. 3. Check `/api/v1/health`. 4. Deploy the frontend to Netlify with `VITE_API_URL`.
5. Sign in as the admin, open **Admin ▸ AI provider ▸ Test connection**. 6. Commit `backend/app/ml/*` (the installed model) and run real-image checks after deploy.

## 12. Known limitations (read before deploying)

* **Render Free storage is ephemeral:** uploaded images (and any SQLite file) are lost on restart/spin-down/redeploy. The database must be hosted PostgreSQL; images need external object
  storage before real use (not built yet). Free Render Postgres also **expires after 30 days**.
* **Cold start:** a free service sleeps after 15 min idle; the first request can take about a minute.
* **Cross-site cookies:** Netlify and Render are different sites, so the refresh cookie needs `COOKIE_SAMESITE=none` + `COOKIE_SECURE=true`. Safari/iOS may still block third-party
  cookies (users would be signed out on reload). Fixes: use one parent domain (e.g. `app.example.com` + `api.example.com` with `COOKIE_SAMESITE=lax`) or the commented Netlify proxy in `netlify.toml`.
* No per-IP rate limit on registration/login/OTP email sending (accounts lock and OTPs cool down per account only).
* Real Gemini is unverified until a key is supplied. PostgreSQL could not be exercised locally (no server available); the code path is standard SQLAlchemy/Alembic.
