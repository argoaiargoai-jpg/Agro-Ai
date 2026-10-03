"""Phase 3: external-AI guidance tracking columns on analyses

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

import app.db.base

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("analyses") as b:
        b.add_column(sa.Column("ai_provider", sa.String(length=32), nullable=True))
        b.add_column(sa.Column("ai_model", sa.String(length=80), nullable=True))
        b.add_column(sa.Column("ai_status", sa.String(length=24), nullable=True))
        b.add_column(sa.Column("ai_error_code", sa.String(length=40), nullable=True))
        b.add_column(sa.Column("ml_completed_at", app.db.base.UTCDateTime(), nullable=True))
        b.add_column(sa.Column("ai_attempted_at", app.db.base.UTCDateTime(), nullable=True))
        b.add_column(sa.Column("ai_completed_at", app.db.base.UTCDateTime(), nullable=True))
        b.create_index("ix_analyses_ai_status", ["ai_status"])
        b.create_index("ix_analyses_ai_attempted_at", ["ai_attempted_at"])


def downgrade() -> None:
    with op.batch_alter_table("analyses") as b:
        b.drop_index("ix_analyses_ai_attempted_at")
        b.drop_index("ix_analyses_ai_status")
        for c in ("ai_completed_at", "ai_attempted_at", "ml_completed_at", "ai_error_code", "ai_status", "ai_model", "ai_provider"):
            b.drop_column(c)
