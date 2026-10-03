"""Migrations 0001 -> 0002: upgrade, downgrade, upgrade again, with Phase 1 data present throughout."""
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models import Analysis
from app.schemas.analysis import AnalysisOut

BACKEND = Path(__file__).resolve().parents[1]
AI_COLS = ["ai_provider", "ai_model", "ai_status", "ai_error_code", "ml_completed_at", "ai_attempted_at", "ai_completed_at"]
UID = uuid.uuid4().hex


def alembic(db: Path, *args: str) -> str:
    r = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=BACKEND, capture_output=True, text=True,
                       env={**os.environ, "DATABASE_URL": f"sqlite:///{db}"})
    assert r.returncode == 0, r.stderr
    return r.stdout + r.stderr


def columns(db: Path):
    with sqlite3.connect(db) as c:
        return {r[1]: r for r in c.execute("pragma table_info(analyses)")}


def legacy_rows(db: Path):
    with sqlite3.connect(db) as c:
        return c.execute("select id,user_id,crop_type,source,image_path,original_filename,content_type,size_bytes,notes,status,result,confidence,error_message,created_at,updated_at from analyses order by id").fetchall()


@pytest.fixture
def phase1_db(tmp_path):
    db = tmp_path / "phase1.db"
    alembic(db, "upgrade", "0001")
    with sqlite3.connect(db) as c:
        c.execute("insert into users(id,email,full_name,role,is_active,is_verified,auth_provider,preferences,failed_login_count,created_at,updated_at) values (?,?,?,?,?,?,?,?,?,?,?)",
                  (UID, "old@example.com", "Old User", "user", 1, 1, "password", "{}", 0, "2026-01-01 00:00:00", "2026-01-01 00:00:00"))
        rows = [("a" * 32, "uploaded", None, None, None), ("b" * 32, "completed", json.dumps({"ml": {"classification_type": "HEALTHY"}, "external_ai": None, "stage": "ml_only"}), 0.91, None),
                ("c" * 32, "failed", None, None, "We couldn't read that image.")]
        for rid, status, result, conf, err in rows:
            c.execute("insert into analyses(id,user_id,crop_type,source,image_path,original_filename,content_type,size_bytes,notes,status,result,confidence,error_message,created_at,updated_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (rid, UID, "Tomato", "upload", f"{UID}/{rid}.png", "leaf.png", "image/png", 1234, "old note", status, result, conf, err, "2026-01-02 00:00:00", "2026-01-02 00:00:00"))
    return db


def test_upgrade_adds_nullable_ai_columns_and_keeps_every_phase1_value(phase1_db):
    before = legacy_rows(phase1_db)
    assert len(before) == 3 and not set(AI_COLS) & set(columns(phase1_db))
    alembic(phase1_db, "upgrade", "head")
    cols = columns(phase1_db)
    for c in AI_COLS:
        assert c in cols and cols[c][3] == 0 and cols[c][4] is None, c            # present, NOT NULL flag off, no default required
    assert legacy_rows(phase1_db) == before                                      # all old columns byte-for-byte identical
    with sqlite3.connect(phase1_db) as c:
        assert c.execute(f"select count(*) from analyses where {' and '.join(x + ' is null' for x in AI_COLS)}").fetchone()[0] == 3
        assert c.execute("select count(*) from users").fetchone()[0] == 1
        idx = {r[1] for r in c.execute("pragma index_list(analyses)")}
        assert {"ix_analyses_ai_status", "ix_analyses_ai_attempted_at"} <= idx


def test_legacy_rows_load_through_the_current_orm_and_api_schema(phase1_db):
    alembic(phase1_db, "upgrade", "head")
    with Session(create_engine(f"sqlite:///{phase1_db}")) as db:
        rows = db.scalars(select(Analysis).order_by(Analysis.id)).all()
        assert [r.status for r in rows] == ["uploaded", "completed", "failed"]
        for r in rows:
            out = AnalysisOut.model_validate(r)
            assert out.ai_status is None and out.ai_provider is None and out.ai_completed_at is None
        assert rows[1].result["stage"] == "ml_only" and rows[1].confidence == 0.91


def test_downgrade_then_upgrade_again_loses_nothing_from_phase1(phase1_db):
    before = legacy_rows(phase1_db)
    alembic(phase1_db, "upgrade", "head")
    alembic(phase1_db, "downgrade", "0001")
    assert not set(AI_COLS) & set(columns(phase1_db)) and legacy_rows(phase1_db) == before
    alembic(phase1_db, "upgrade", "head")
    assert set(AI_COLS) <= set(columns(phase1_db)) and legacy_rows(phase1_db) == before
    assert "head" in alembic(phase1_db, "current")


def test_new_data_written_after_upgrade_survives_a_downgrade_of_old_columns(phase1_db):
    alembic(phase1_db, "upgrade", "head")
    with sqlite3.connect(phase1_db) as c:
        c.execute("update analyses set ai_status='completed', ai_provider='gemini' where id=?", ("b" * 32,))
    alembic(phase1_db, "downgrade", "0001")
    assert legacy_rows(phase1_db)[1][9] == "completed"                           # the Phase 1 columns of that row are untouched


def test_fresh_database_migrates_to_head_and_matches_the_models(tmp_path):
    db = tmp_path / "fresh.db"
    alembic(db, "upgrade", "head")
    assert "No new upgrade operations detected" in alembic(db, "check")
    alembic(db, "downgrade", "base"); alembic(db, "upgrade", "head")


def test_a_legacy_style_row_works_through_the_running_api(client, user_auth):
    """An analysis created before Phase 3 (no AI fields, old result shape) must still be readable, listable and cacheable."""
    from app.db.session import SessionLocal
    me = client.get("/api/v1/users/me", headers=user_auth).json()
    with SessionLocal() as db:
        a = Analysis(user_id=uuid.UUID(me["id"]), source="upload", image_path="x/y.png", content_type="image/png", size_bytes=10, status="completed",
                     result={"ml": {"classification_type": "HEALTHY", "crop": "Tomato", "disease": None, "confidence": 0.9, "message": "ok", "supported_crops": [], "model_version": "x", "inference_ms": 1},
                             "external_ai": None, "stage": "ml_only"}, confidence=0.9)
        db.add(a); db.commit(); aid = str(a.id)
    got = client.get(f"/api/v1/analyses/{aid}", headers=user_auth)
    assert got.status_code == 200 and got.json()["ai_status"] is None and got.json()["status"] == "completed"
    assert client.get("/api/v1/analyses", headers=user_auth).json()["total"] == 1
    assert client.get("/api/v1/analyses/summary", headers=user_auth).status_code == 200
    assert client.post(f"/api/v1/analyses/{aid}/analyze", headers=user_auth).json()["id"] == aid        # completed -> served from storage, no model needed


# ----------------------------------------------------------------------------- 0003: counters (alternating Gemini keys)
def tables(db: Path) -> set[str]:
    with sqlite3.connect(db) as c:
        return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def test_0003_adds_the_counters_table_and_downgrade_removes_only_it(phase1_db):
    alembic(phase1_db, "upgrade", "0002")
    assert "counters" not in tables(phase1_db)
    before = legacy_rows(phase1_db)
    alembic(phase1_db, "upgrade", "head")
    assert "counters" in tables(phase1_db) and legacy_rows(phase1_db) == before
    with sqlite3.connect(phase1_db) as c:
        assert [r[1] for r in c.execute("pragma table_info(counters)")] == ["name", "value"]
    alembic(phase1_db, "downgrade", "0002")
    assert "counters" not in tables(phase1_db) and "analyses" in tables(phase1_db) and legacy_rows(phase1_db) == before
    alembic(phase1_db, "upgrade", "head")
    assert "counters" in tables(phase1_db)


def test_counter_upsert_is_atomic_on_a_migrated_database(tmp_path):
    db = tmp_path / "c.db"
    alembic(db, "upgrade", "head")
    assert "No new upgrade operations detected" in alembic(db, "check")
    code = ("import os; os.environ['DATABASE_URL']='sqlite:///%s'; from app.ai import keyring; "
            "print([keyring.next_number() for _ in range(5)])" % db)
    r = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("[1, 2, 3, 4, 5]")
