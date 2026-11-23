"""Privacy-safe counts of validation attempts.

A row is a timestamp, an outcome, a short format label, and a score band.
The anonymous id is a salted hash of a random browser id, or — only when
``QP_USAGE_SALT`` is set — of the caller IP. The raw IP, the browser id, and
the uploaded file are never written.

When ``DATABASE_URL`` points at the SQL store, rows go to ``validation_usage``.
Otherwise they stay in memory for this process and a one-line log is emitted
either way. Counting must never change the validation response.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import db
from services.client_ip import client_ip

log = logging.getLogger("quantproof.usage")

ALLOWED_FORMATS = frozenset(
    {
        "csv",
        "xlsx",
        "ibkr_flex",
        "kite",
        "alpaca",
        "ibkr",
        "tradingview",
        "portfolio",
        "mt5",
        "options",
        "generic",
        "prop_firms",
        "sample",
    }
)

_PATH_FORMATS = {
    "/validate": "csv",
    "/validate/kite": "kite",
    "/validate/alpaca": "alpaca",
    "/validate/portfolio": "portfolio",
    "/validate/prop-firms": "prop_firms",
    "/connect/ibkr": "ibkr",
}

_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_MEMORY_CAP = 5000
_memory: list[dict] = []
_sql_down = False


def format_from_upload(filename: str, note: str = "") -> str:
    """Coarse kind of an upload. The note is only scanned, never stored."""
    name = (filename or "").lower()
    text = note or ""
    if name.endswith(".xml") or "IBKR Flex" in text:
        return "ibkr_flex"
    if text.startswith("📊 MT5") or "MT5_FLAGS:" in text:
        return "mt5"
    if text.startswith("OPTIONS:") or "OPTIONS_FLAGS:" in text:
        return "options"
    if "Zerodha" in text or "Kite TradeBook" in text or "Kite Orders" in text:
        return "kite"
    if "Alpaca" in text:
        return "alpaca"
    if "TradingView" in text:
        return "tradingview"
    if name.endswith(".xlsx"):
        return "xlsx"
    return "csv"


def score_bucket(score: object) -> str | None:
    try:
        number = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    number = max(0.0, min(100.0, number))
    if number < 40:
        return "0-39"
    if number < 60:
        return "40-59"
    if number < 80:
        return "60-79"
    return "80-100"


def outcome_for_status(status: int) -> str:
    if 200 <= status < 300:
        return "success"
    if status == 429:
        return "rate_limited"
    if 400 <= status < 500:
        return "rejected"
    return "error"


def anonymous_id(client_header: str, ip: str, salt: str | None = None) -> tuple[str | None, str]:
    """Return ``(hash, source)``. Source is ``client``, ``ip``, or ``none``.

    A browser id is hashed even without a salt: it is a random UUID, not a
    person. An IP is hashed only when a salt is configured, so a database
    leak cannot be walked against the public IPv4 space.
    """
    if salt is None:
        salt = os.getenv("QP_USAGE_SALT", "").strip()
    header = (client_header or "").strip()
    if len(header) == 36 and _UUID.match(header):
        material = f"client\0{salt}\0{header.lower()}"
        return hashlib.sha256(material.encode()).hexdigest(), "client"
    if salt and ip and ip not in {"", "unknown"}:
        material = f"ip\0{salt}\0{ip}"
        return hashlib.sha256(material.encode()).hexdigest(), "ip"
    return None, "none"


def mark_usage(request, *, fmt: str | None = None, score: float | None = None) -> None:
    """Stash hints for the middleware. Never raises into the request."""
    if request is None:
        return
    state = getattr(request, "state", None)
    if state is None:
        return
    if fmt in ALLOWED_FORMATS:
        try:
            state.usage_format = fmt
        except (AttributeError, TypeError):
            return
    if score is None:
        return
    if score_bucket(score) is None:
        return
    try:
        state.usage_score = float(score)
    except (AttributeError, TypeError, ValueError):
        return


def default_format(path: str) -> str | None:
    if path in _PATH_FORMATS:
        return _PATH_FORMATS[path]
    parts = [part for part in (path or "").split("/") if part]
    if len(parts) == 4 and parts[0] == "webhook" and parts[1] == "tradingview" and parts[3] == "validate":
        return "tradingview"
    return None


def record_http(scope: dict, status_code: int) -> None:
    if scope.get("type") != "http" or (scope.get("method") or "").upper() != "POST":
        return
    path = scope.get("path") or ""
    fallback = default_format(path)
    if fallback is None:
        return
    hinted = _state_value(scope.get("state"), "usage_format")
    fmt = hinted if isinstance(hinted, str) and hinted in ALLOWED_FORMATS else fallback
    if _header(scope, "x-qp-sample") == "1" and fmt in {"csv", "xlsx"}:
        fmt = "sample"
    outcome = outcome_for_status(status_code)
    bucket = None
    if outcome == "success":
        bucket = score_bucket(_state_value(scope.get("state"), "usage_score"))
    digest, source = anonymous_id(_header(scope, "x-qp-client"), _caller_ip(scope))
    _persist(
        {
            "created_at": datetime.now(UTC),
            "client_hash": digest,
            "id_source": source,
            "outcome": outcome,
            "format": fmt,
            "score_bucket": bucket,
        }
    )


def summary() -> dict:
    """Aggregates only. Client hashes stay in the table."""
    rows, persistence = _rows()
    completed = 0
    sample_completed = 0
    rejected = 0
    rate_limited = 0
    errors = 0
    successes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    by_browser: Counter[str] = Counter()
    first_at = None
    last_at = None
    for row in rows:
        outcome = _field(row, "outcome")
        fmt = _field(row, "format")
        created = _field(row, "created_at")
        formats[str(fmt)] += 1
        if first_at is None or (created is not None and created < first_at):
            first_at = created
        if last_at is None or (created is not None and created > last_at):
            last_at = created
        if outcome == "rejected":
            rejected += 1
        elif outcome == "rate_limited":
            rate_limited += 1
        elif outcome == "error":
            errors += 1
        elif outcome == "success" and fmt == "sample":
            sample_completed += 1
        elif outcome == "success":
            completed += 1
            successes[str(fmt)] += 1
            bucket = _field(row, "score_bucket")
            if bucket:
                buckets[str(bucket)] += 1
            digest = _field(row, "client_hash")
            if digest:
                by_browser[str(digest)] += 1
    return {
        "persistence": persistence,
        "ip_hash_enabled": bool(os.getenv("QP_USAGE_SALT", "").strip()),
        "total": len(rows),
        "completed": completed,
        "sample_completed": sample_completed,
        "rejected": rejected,
        "rate_limited": rate_limited,
        "errors": errors,
        "repeat_browsers": sum(1 for count in by_browser.values() if count >= 2),
        "by_format": dict(sorted(formats.items())),
        "completed_by_format": dict(sorted(successes.items())),
        "score_buckets": dict(sorted(buckets.items())),
        "first_at": first_at.isoformat() if isinstance(first_at, datetime) else None,
        "last_at": last_at.isoformat() if isinstance(last_at, datetime) else None,
    }


def clear() -> None:
    """Drop counted rows. Used by tests."""
    global _sql_down
    _memory.clear()
    _sql_down = False
    engine = getattr(db.store, "engine", None)
    if engine is None:
        return
    try:
        with Session(engine) as session, session.begin():
            session.execute(delete(db.ValidationUsage))
    except SQLAlchemyError:
        return


def _persist(event: dict) -> None:
    global _sql_down
    log.info(
        "validation_usage outcome=%s format=%s score_bucket=%s id_source=%s",
        event["outcome"],
        event["format"],
        event["score_bucket"] or "-",
        event["id_source"],
    )
    engine = getattr(db.store, "engine", None)
    if engine is None or _sql_down:
        _remember(event)
        return
    try:
        with Session(engine) as session, session.begin():
            session.add(db.ValidationUsage(**event))
    except SQLAlchemyError:
        if not _sql_down:
            log.warning("validation usage table is unavailable; counting in memory for this process")
            _sql_down = True
        _remember(event)


def _remember(event: dict) -> None:
    _memory.append(event)
    if len(_memory) > _MEMORY_CAP:
        del _memory[: len(_memory) - _MEMORY_CAP]


def _rows() -> tuple[list, str]:
    engine = getattr(db.store, "engine", None)
    if engine is not None and not _sql_down:
        try:
            with Session(engine) as session:
                rows = list(session.scalars(select(db.ValidationUsage).order_by(db.ValidationUsage.id)).all())
            return rows, "sql"
        except SQLAlchemyError:
            pass
    return list(_memory), "memory"


def _field(row, name: str):
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _state_value(state, name: str):
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get(name)
    inner = getattr(state, "_state", None)
    if isinstance(inner, dict) and name in inner:
        return inner[name]
    return getattr(state, name, None)


def _header(scope: dict, name: str) -> str:
    target = name.lower().encode()
    for key, value in scope.get("headers") or []:
        if not isinstance(key, bytes) or key.lower() != target:
            continue
        try:
            text = value.decode("latin-1")
        except (UnicodeError, AttributeError):
            return ""
        return text[:80]
    return ""


def _caller_ip(scope: dict) -> str:
    headers: dict[str, str] = {}
    for key, value in scope.get("headers") or []:
        if not isinstance(key, bytes) or not isinstance(value, bytes):
            continue
        try:
            headers[key.decode("latin-1").lower()] = value.decode("latin-1")
        except UnicodeError:
            continue
    client = scope.get("client")
    host = client[0] if isinstance(client, (list, tuple)) and client else ""

    class _Headers:
        def get(self, header_name: str, default: str = "") -> str:
            return headers.get(header_name.lower(), default)

    return client_ip(SimpleNamespace(headers=_Headers(), client=SimpleNamespace(host=host or "")))
