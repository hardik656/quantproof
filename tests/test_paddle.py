"""Paddle checkout + webhook: signature is mandatory, errors don't leak provider detail."""

import hashlib
import hmac
import json
import time

import pytest

SECRET = "pdl_ntfset_test_secret"
PRICE_LITE = "pri_" + "a" * 26


def _sign(body: bytes, ts: int | None = None, secret: str = SECRET) -> str:
    ts = int(time.time()) if ts is None else ts
    h1 = hmac.new(secret.encode(), f"{ts}:".encode() + body, hashlib.sha256).hexdigest()
    return f"ts={ts};h1={h1}"


def _event(txn_id: str = "txn_" + "b" * 26, price_id: str = PRICE_LITE) -> bytes:
    return json.dumps(
        {
            "event_type": "transaction.completed",
            "data": {"id": txn_id, "items": [{"price": {"id": price_id}}]},
        }
    ).encode()


@pytest.fixture
def paddle_env(monkeypatch):
    monkeypatch.setenv("PADDLE_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("PADDLE_PRICE_LITE", PRICE_LITE)
    monkeypatch.setenv("PADDLE_API_KEY", "pdl_live_apikey_test")


def test_webhook_rejects_missing_signature(client, paddle_env):
    r = client.post("/report/webhook-paddle", content=_event())
    assert r.status_code == 400


def test_webhook_rejects_bad_signature(client, paddle_env):
    r = client.post(
        "/report/webhook-paddle", content=_event(), headers={"Paddle-Signature": "ts=1;h1=deadbeef"}
    )
    assert r.status_code == 400


def test_webhook_rejects_stale_timestamp(client, paddle_env):
    body = _event()
    r = client.post(
        "/report/webhook-paddle",
        content=body,
        headers={"Paddle-Signature": _sign(body, ts=int(time.time()) - 7200)},
    )
    assert r.status_code == 400


def test_webhook_unconfigured_secret_fails_closed(client, monkeypatch):
    monkeypatch.delenv("PADDLE_WEBHOOK_SECRET", raising=False)
    body = _event()
    r = client.post("/report/webhook-paddle", content=body, headers={"Paddle-Signature": _sign(body)})
    assert r.status_code == 503


def test_signed_webhook_grants_token_and_poll_returns_it(client, paddle_env):
    txn = "txn_" + "c" * 26
    body = _event(txn)
    r = client.post("/report/webhook-paddle", content=body, headers={"Paddle-Signature": _sign(body)})
    assert r.status_code == 200 and r.json()["tier"] == "lite"
    poll = client.get(f"/report/paddle-token?txn_id={txn}").json()
    assert poll["status"] == "ready" and poll["tier"] == "lite" and poll["token"]


def test_unsigned_forged_txn_gets_no_token(client, paddle_env):
    txn = "txn_" + "d" * 26
    client.post("/report/webhook-paddle", content=_event(txn))
    assert client.get(f"/report/paddle-token?txn_id={txn}").json()["status"] == "pending"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_checkout_success_uses_checkout_url_field(client, paddle_env, monkeypatch):
    import requests

    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(json)
        return _Resp(
            201,
            {"data": {"id": "txn_" + "e" * 26, "checkout": {"url": "https://quantproof.net/?_ptxn=txn_x"}}},
        )

    monkeypatch.setattr(requests, "post", fake_post)
    r = client.post("/report/create-checkout-paddle-lite", json={})
    assert r.status_code == 200
    assert r.json()["checkout_url"].startswith("https://quantproof.net/")
    assert r.json()["transaction_id"].startswith("txn_")
    assert sent["checkout"] == {"url": "https://quantproof.net/"}
    assert "return_url" not in sent["checkout"]


def test_checkout_provider_error_is_502_without_leaking(client, paddle_env, monkeypatch):
    import requests

    def fake_post(*a, **k):
        return _Resp(
            403,
            {"error": {"code": "forbidden", "detail": "You aren't permitted"}, "meta": {"request_id": "r1"}},
        )

    monkeypatch.setattr(requests, "post", fake_post)
    for tier in ("lite", "report", "portfolio", "pro"):
        monkeypatch.setenv(f"PADDLE_PRICE_{tier.upper()}", PRICE_LITE)
        r = client.post(f"/report/create-checkout-paddle-{tier}", json={})
        assert r.status_code == 502
        assert "forbidden" not in r.text and "permitted" not in r.text


def test_checkout_unconfigured_is_503(client, monkeypatch):
    monkeypatch.delenv("PADDLE_API_KEY", raising=False)
    assert client.post("/report/create-checkout-paddle-pro", json={}).status_code == 503
