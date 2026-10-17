import io

import numpy as np
import pandas as pd

import main
from services import portfolio, tokens


def _csv(n=120, mu=0.003, seed=1) -> bytes:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"date": pd.date_range("2023-01-02", periods=n, freq="B"), "pnl": rng.normal(mu, 0.01, n)}
    )
    return df.to_csv(index=False).encode()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": main.API_VERSION}


def test_root_and_ping(client):
    assert "29" in client.get("/").json()["checks"]
    assert client.get("/ping").json()["status"] == "alive"
    assert client.head("/ping").status_code == 200


def test_validate_rejects_non_csv(client):
    r = client.post("/validate", files={"file": ("t.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 400


def test_validate_happy_path(client):
    r = client.post("/validate", files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["fundable_score"] <= 100
    assert body["currency"] == "USD"
    assert len(body["checks"]) >= 25


def test_validate_rate_limit(client):
    for _ in range(tokens.FREE_DAILY_LIMIT):
        client.post("/validate", files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")})
    r = client.post("/validate", files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")})
    assert r.status_code == 429


def test_token_status_unknown(client):
    r = client.get("/report/token-status", headers={"x-token": "nope"})
    assert r.status_code == 200


def test_debug_token_requires_param(client):
    assert "error" in client.get("/debug/token").json()


def test_debug_reset_requires_pro(client):
    assert client.get("/debug/reset-limit?t=bad").status_code == 403


def test_token_tiers(monkeypatch):
    for k in ("TEST_REPORT_TOKEN", "DEV_TOKEN", "ADMIN_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert tokens._token_tier("") == "none"
    tokens.REPORT_ACCESS_TOKENS.add("LITE-abc")
    assert tokens._token_tier("LITE-abc") == "lite"
    tokens.REPORT_ACCESS_TOKENS.discard("LITE-abc")
    monkeypatch.setenv("DEV_TOKEN", "devtok")
    assert tokens._token_tier("devtok") == "pro"
    assert tokens._is_pro_token("devtok")


def test_webhook_rejects_bad_signature(client):
    r = client.post("/report/webhook", content=b"{}", headers={"x-razorpay-signature": "bad"})
    assert r.status_code in (400, 401, 403, 503)


def test_tradingview_session_lifecycle(client):
    sid = client.get("/webhook/tradingview/new-session").json()["session_id"]
    st = client.get(f"/webhook/tradingview/{sid}/status")
    assert st.status_code == 200
    assert client.post(f"/webhook/tradingview/{sid}/reset").json()["status"] == "reset"
    assert client.get("/webhook/tradingview/does-not-exist/status").status_code == 404


def test_portfolio_math():
    rng = np.random.default_rng(0)
    dfs = [pd.DataFrame({"pnl": rng.normal(0.001, 0.01, 100)}) for _ in range(3)]
    corr = portfolio._portfolio_correlation(dfs)
    assert corr.shape == (3, 3)
    w = portfolio._risk_parity_weights(dfs)
    assert abs(sum(w) - 1) < 1e-6
    assert isinstance(portfolio._portfolio_verdict(70.0, 0.2, 3), str)


def _rzp_body(secret, order="order_1", payment="pay_1"):
    import hashlib
    import hmac
    import json

    sig = hmac.new(secret.encode(), f"{order}|{payment}".encode(), hashlib.sha256).hexdigest()
    return json.dumps({"razorpay_order_id": order, "razorpay_payment_id": payment, "razorpay_signature": sig})


def test_razorpay_webhook_accepts_valid_signature(client, monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "s3cret")
    r = client.post("/report/webhook", content=_rzp_body("s3cret"))
    assert r.status_code == 200, r.text


def test_razorpay_webhook_rejects_forged_signature(client, monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "s3cret")
    r = client.post("/report/webhook", content=_rzp_body("attacker-guess"))
    assert r.status_code == 400


def test_razorpay_webhook_refuses_without_secret(client, monkeypatch):
    """Previously fell back to a hard-coded secret, letting anyone mint report tokens."""
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    r = client.post("/report/webhook", content=_rzp_body("quantproof-fallback"))
    assert r.status_code == 503


def test_razorpay_webhook_rejects_missing_fields(client, monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "s3cret")
    assert client.post("/report/webhook", content=b'{"razorpay_order_id": "x"}').status_code == 400
    assert client.post("/report/webhook", content=b"not json").status_code == 400
