from alembic import context
from sqlalchemy import pool

import app.models  # noqa: F401
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import make_engine

target_metadata = Base.metadata
url = get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=url.startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=url.startswith("sqlite"))
        with context.begin_transaction():
            context.run_migrations()


(run_migrations_offline if context.is_offline_mode() else run_migrations_online)()
