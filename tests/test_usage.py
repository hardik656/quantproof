"""Validation usage counter: aggregates only, no file contents, no raw IPs."""

import io
import logging

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

import db
from services import tokens, usage

CLIENT = "12345678-1234-4123-8123-123456789abc"
OTHER = "abcdefab-1234-4123-8123-123456789abc"
SALT = "usage-salt-not-in-output"


@pytest.fixture(autouse=True)
def _clean_usage():
    usage.clear()
    yield
    usage.clear()


def _csv() -> bytes:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"date": pd.date_range("2023-01-02", periods=40, freq="B"), "pnl": rng.normal(0.003, 0.01, 40)}
    )
    return frame.to_csv(index=False).encode()


def _rows():
    engine = db.store.engine
    with Session(engine) as session:
        return list(session.scalars(select(db.ValidationUsage).order_by(db.ValidationUsage.id)).all())


def test_anonymous_id_hashes_browser_id_without_keeping_it():
    digest, source = usage.anonymous_id(CLIENT, "203.0.113.10", salt="")
    assert source == "client"
    assert digest is not None
    assert CLIENT not in digest
    assert "203.0.113.10" not in digest
    again, _ = usage.anonymous_id(CLIENT.upper(), "203.0.113.99", salt="")
    assert again == digest


def test_anonymous_id_hashes_ip_only_when_salted():
    digest, source = usage.anonymous_id("not-an-id", "203.0.113.10", salt=SALT)
    assert source == "ip"
    assert digest is not None
    assert "203.0.113.10" not in digest
    assert usage.anonymous_id("not-an-id", "203.0.113.10", salt=SALT)[0] == digest
    assert usage.anonymous_id("not-an-id", "203.0.113.11", salt=SALT)[0] != digest
    assert usage.anonymous_id("alice@example.com", "203.0.113.10", salt="") == (None, "none")


def test_format_labels_are_coarse():
    assert usage.format_from_upload("trades.csv") == "csv"
    assert usage.format_from_upload("book.xlsx") == "xlsx"
    assert usage.format_from_upload("flex.xml") == "ibkr_flex"
    assert usage.format_from_upload("a.csv", "Parsed 12 closed trade(s) from IBKR Flex CSV.") == "ibkr_flex"
    assert usage.format_from_upload("a.csv", "📊 MT5: 12 completed trade(s)") == "mt5"
    assert usage.format_from_upload("a.csv", "OPTIONS: 12 single-leg trades parsed.") == "options"
    assert (
        usage.format_from_upload("a.csv", "📊 Parsed 12 completed trade(s) from Zerodha Kite TradeBook")
        == "kite"
    )
    assert usage.format_from_upload("a.csv", "Parsed 12 completed trade(s) from Alpaca orders") == "alpaca"
    assert (
        usage.format_from_upload("a.csv", "Parsed 12 closed trade(s) from TradingView List of Trades.")
        == "tradingview"
    )
    assert usage.score_bucket(79.9) == "60-79"
    assert usage.score_bucket(80) == "80-100"
    assert usage.score_bucket(None) is None


def test_validate_records_success_without_changing_the_response(client, monkeypatch):
    monkeypatch.setenv("QP_USAGE_SALT", SALT)
    response = client.post(
        "/validate",
        files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
        headers={"X-QP-Client": CLIENT, "X-Forwarded-For": "203.0.113.80"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "fundable_score" in body
    assert "client_hash" not in body
    assert "usage" not in body
    assert CLIENT not in response.text
    assert "203.0.113.80" not in response.text
    rows = _rows()
    assert len(rows) == 1
    assert rows[0].outcome == "success"
    assert rows[0].format == "csv"
    assert rows[0].id_source == "client"
    assert rows[0].score_bucket in {"0-39", "40-59", "60-79", "80-100"}
    assert CLIENT not in (rows[0].client_hash or "")
    assert "203.0.113.80" not in (rows[0].client_hash or "")
    stored = " ".join(
        str(getattr(rows[0], column)) for column in ("client_hash", "format", "outcome", "score_bucket")
    )
    assert "pnl" not in stored
    counts = usage.summary()
    assert counts["persistence"] == "sql"
    assert counts["completed"] == 1
    assert counts["repeat_browsers"] == 0
    assert SALT not in str(counts)


def test_repeat_browsers_ignore_the_built_in_sample(client):
    headers = {"X-QP-Client": CLIENT, "X-Forwarded-For": "203.0.113.81"}
    for _ in range(2):
        assert (
            client.post(
                "/validate", files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")}, headers=headers
            ).status_code
            == 200
        )
    sample = client.post(
        "/validate",
        files={"file": ("sample.csv", io.BytesIO(_csv()), "text/csv")},
        headers={**headers, "X-QP-Sample": "1"},
    )
    assert sample.status_code == 200
    other = client.post(
        "/validate",
        files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
        headers={"X-QP-Client": OTHER, "X-Forwarded-For": "203.0.113.82"},
    )
    assert other.status_code == 200
    counts = usage.summary()
    assert counts["completed"] == 3
    assert counts["sample_completed"] == 1
    assert counts["repeat_browsers"] == 1
    assert counts["completed_by_format"] == {"csv": 3}
    assert counts["by_format"]["sample"] == 1
    assert CLIENT not in str(counts)
    assert OTHER not in str(counts)


def test_rejected_upload_and_rate_limit_are_counted_without_the_ip(client, monkeypatch, caplog):
    monkeypatch.setenv("QP_USAGE_SALT", SALT)
    caplog.set_level(logging.INFO, logger="quantproof.usage")
    ip = "203.0.113.83"
    rejected = client.post(
        "/validate",
        files={"file": ("notes.txt", io.BytesIO(b"not,a,trade,file"), "text/plain")},
        headers={"X-QP-Client": "alice@example.com", "X-Forwarded-For": ip},
    )
    assert rejected.status_code == 400
    for _ in range(tokens.FREE_DAILY_LIMIT):
        assert (
            client.post(
                "/validate",
                files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
                headers={"X-Forwarded-For": ip},
            ).status_code
            == 200
        )
    limited = client.post(
        "/validate",
        files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
        headers={"X-Forwarded-For": ip},
    )
    assert limited.status_code == 429
    counts = usage.summary()
    assert counts["rejected"] == 1
    assert counts["completed"] == tokens.FREE_DAILY_LIMIT
    assert counts["rate_limited"] == 1
    blob = (
        str(counts)
        + "\n"
        + "\n".join(record.getMessage() for record in caplog.records if record.name == "quantproof.usage")
    )
    for secret in (ip, "alice@example.com", SALT, "not,a,trade,file"):
        assert secret not in blob
    hashed = [row for row in _rows() if row.id_source == "ip"]
    assert hashed
    assert all(ip not in (row.client_hash or "") for row in hashed)


def test_debug_usage_requires_the_admin_token_and_returns_counts_only(client, monkeypatch):
    monkeypatch.delenv("DEBUG_ADMIN_TOKEN", raising=False)
    assert client.get("/debug/usage").status_code == 404
    monkeypatch.setenv("DEBUG_ADMIN_TOKEN", "s3cret-value")
    assert client.get("/debug/usage", params={"t": "nope"}).status_code == 404
    client.post(
        "/validate",
        files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
        headers={"X-QP-Client": CLIENT},
    )
    allowed = client.get("/debug/usage", params={"t": "s3cret-value"})
    assert allowed.status_code == 200
    text = allowed.text
    assert "s3cret-value" not in text
    assert CLIENT not in text
    assert "client_hash" not in text
    assert allowed.json()["completed"] == 1


def test_counter_failure_does_not_change_validate(client, monkeypatch):
    def explode(_event):
        raise RuntimeError("counter failed")

    monkeypatch.setattr(usage, "_persist", explode)
    response = client.post("/validate", files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")})
    assert response.status_code == 200
    assert "fundable_score" in response.json()


def test_memory_fallback_counts_without_a_database(monkeypatch):
    class Bare:
        pass

    monkeypatch.setattr(db, "store", Bare())
    usage._memory.clear()
    usage._sql_down = False
    usage.record_http(
        {
            "type": "http",
            "method": "POST",
            "path": "/validate",
            "headers": [(b"x-qp-client", CLIENT.encode())],
            "client": ("203.0.113.90", 1234),
            "state": {"usage_format": "csv", "usage_score": 72},
        },
        200,
    )
    usage.record_http({"type": "http", "method": "GET", "path": "/health", "headers": [], "state": {}}, 200)
    usage.record_http(
        {"type": "http", "method": "POST", "path": "/validate/ai-fixes", "headers": [], "state": {}},
        200,
    )
    counts = usage.summary()
    assert counts["persistence"] == "memory"
    assert counts["completed"] == 1
    assert counts["total"] == 1
    assert counts["score_buckets"] == {"60-79": 1}
    assert CLIENT not in str(counts)
