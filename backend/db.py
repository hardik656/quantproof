"""Optional persistence layer (SQLAlchemy 2).

State that used to live only in process memory or /tmp JSON files (paid access
tokens, PRO-token expiries, TradingView webhook sessions) goes through a small
key-value store:

* ``DATABASE_URL`` unset  -> JSON files under ``QP_STATE_DIR`` (default /tmp),
  i.e. exactly the previous behaviour.
* ``DATABASE_URL`` set    -> a ``kv_state`` table in Postgres (or sqlite for tests).
  Schema is managed by Alembic (``alembic upgrade head``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import JSON, DateTime, String, create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

log = logging.getLogger("quantproof.db")


class Base(DeclarativeBase):
    pass


class KVState(Base):
    __tablename__ = "kv_state"

    namespace: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class StateStore(Protocol):
    def load(self, namespace: str) -> dict[str, Any]: ...
    def save(self, namespace: str, data: dict[str, Any]) -> None: ...


class JsonFileStore:
    """One JSON file per namespace (legacy behaviour)."""

    def __init__(self, directory: str | os.PathLike[str] = "/tmp", filenames: dict[str, str] | None = None):
        self.directory = Path(directory)
        self.filenames = filenames or {}

    def _path(self, namespace: str) -> Path:
        return self.directory / self.filenames.get(namespace, f"qp_{namespace}.json")

    def load(self, namespace: str) -> dict[str, Any]:
        path = self._path(namespace)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", path, exc)
            return {}
        # report tokens were historically stored as a JSON list
        return {k: True for k in raw} if isinstance(raw, list) else dict(raw)

    def save(self, namespace: str, data: dict[str, Any]) -> None:
        path = self._path(namespace)
        try:
            path.write_text(json.dumps(data, default=str))
        except OSError as exc:
            log.warning("could not write %s: %s", path, exc)


class SqlStore:
    """Namespace/key/value rows in a relational database."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def load(self, namespace: str) -> dict[str, Any]:
        try:
            with Session(self.engine) as s:
                rows = s.scalars(select(KVState).where(KVState.namespace == namespace))
                return {r.key: r.value for r in rows}
        except SQLAlchemyError as exc:
            log.error("state load failed for %s: %s", namespace, exc)
            return {}

    def save(self, namespace: str, data: dict[str, Any]) -> None:
        try:
            with Session(self.engine) as s, s.begin():
                s.execute(delete(KVState).where(KVState.namespace == namespace))
                s.add_all(
                    KVState(namespace=namespace, key=str(k), value=json.loads(json.dumps(v, default=str)))
                    for k, v in data.items()
                )
        except SQLAlchemyError as exc:
            log.error("state save failed for %s: %s", namespace, exc)


LEGACY_FILENAMES = {
    "report_tokens": "qp_report_tokens.json",
    "pro_tokens": "qp_pro_tokens.json",
    "tv_sessions": "qp_tv_sessions.json",
}


def build_store(database_url: str | None = None) -> StateStore:
    url = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
    if not url:
        return JsonFileStore(os.getenv("QP_STATE_DIR", "/tmp"), LEGACY_FILENAMES)  # noqa: S108
    engine = create_engine(url, pool_pre_ping=True, future=True)
    if os.getenv("QP_DB_CREATE_ALL") == "1":
        Base.metadata.create_all(engine)  # tests / local dev; production uses Alembic
    log.info("using SQL state store (%s)", engine.url.render_as_string(hide_password=True))
    return SqlStore(engine)


store: StateStore = build_store()
