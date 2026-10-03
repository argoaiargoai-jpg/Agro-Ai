"""Alternating selection between the Gemini keys, backed by an atomic counter row in the database.

One SQL statement (INSERT .. ON CONFLICT DO UPDATE .. RETURNING) both increments and reads the counter, so two simultaneous requests can never
receive the same number, and the sequence survives restarts and multiple workers. Request 1 -> slot 1, 2 -> slot 2, 3 -> slot 1, ...
The counter is touched ONCE per analysis request; a failover to the other key does not increment it.
"""
import logging

from sqlalchemy.dialects import postgresql, sqlite

from app.models.counter import Counter

log = logging.getLogger("agroai.ai.keyring")
GEMINI_COUNTER = "gemini_requests"


def next_number(name: str = GEMINI_COUNTER) -> int:
    """Atomically increment the named counter and return the NEW value (1 for the first call ever)."""
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        dialect = db.get_bind().dialect.name
        insert = (postgresql if dialect == "postgresql" else sqlite).insert
        stmt = insert(Counter).values(name=name, value=1)
        stmt = stmt.on_conflict_do_update(index_elements=[Counter.name], set_={"value": Counter.value + 1}).returning(Counter.value)
        n = db.execute(stmt).scalar_one()
        db.commit()
        return int(n)


def primary_index(key_count: int) -> int:
    """Index (0-based) of the key to try first for this request. Never raises: a counter problem must not block an analysis."""
    if key_count <= 1:
        return 0
    try:
        return (next_number() - 1) % key_count
    except Exception:  # noqa: BLE001
        log.exception("Gemini key counter unavailable; using the first key for this request")
        return 0
