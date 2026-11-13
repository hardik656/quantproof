"""Fixes from the 13 Nov 2026 live audit: rate-limit IP, debug routes, SEBI OPS."""

import io
from pathlib import Path

import pandas as pd

from services.client_ip import client_ip
from services.sebi import _compute_sebi_checks
from services.tokens import _validation_counts

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

_HARDCODED = ("dev-reset", "test-123", "quantproof-test-112211")


class _Req:
    def __init__(self, headers, host="203.0.113.8"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def _csv(n=20) -> bytes:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=n, freq="B"),
            "pnl": [0.01] * n,
        }
    )
    return df.to_csv(index=False).encode()


def test_client_ip_uses_rightmost_forwarded_hop():
    req = _Req({"x-forwarded-for": "1.2.3.4, 203.0.113.9"})
    assert client_ip(req) == "203.0.113.9"


def test_client_ip_ignores_cf_connecting_ip():
    req = _Req({"cf-connecting-ip": "1.2.3.4", "x-forwarded-for": "198.51.100.4"})
    assert client_ip(req) == "198.51.100.4"
    only_cf = _Req({"cf-connecting-ip": "1.2.3.4"}, host="203.0.113.8")
    assert client_ip(only_cf) == "203.0.113.8"


def test_client_ip_extra_trusted_hop(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    req = _Req({"x-forwarded-for": "9.9.9.9, 203.0.113.5, 104.16.0.1"})
    assert client_ip(req) == "203.0.113.5"


def test_spoofed_forwarded_for_does_not_change_bucket(client):
    real = {"X-Forwarded-For": "198.51.100.20"}
    files = {"file": ("t.csv", io.BytesIO(_csv()), "text/csv")}
    for _ in range(3):
        assert client.post("/validate", files=files, headers=real).status_code == 200
    assert (
        client.post(
            "/validate", files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")}, headers=real
        ).status_code
        == 429
    )
    spoofed = {"X-Forwarded-For": "203.0.113.99, 198.51.100.20", "CF-Connecting-IP": "8.8.8.8"}
    again = client.post(
        "/validate",
        files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
        headers=spoofed,
    )
    assert again.status_code == 429
    assert "1.2.3.4" not in _validation_counts
    assert "203.0.113.99" not in _validation_counts
    assert "8.8.8.8" not in _validation_counts


def test_rejected_upload_does_not_consume_a_try(client):
    headers = {"X-Forwarded-For": "203.0.113.50"}
    for _ in range(4):
        bad = client.post(
            "/validate",
            files={"file": ("t.txt", io.BytesIO(b"x"), "text/plain")},
            headers=headers,
        )
        assert bad.status_code == 400
        empty = client.post(
            "/validate",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
            headers=headers,
        )
        assert empty.status_code == 400
    assert _validation_counts["203.0.113.50"]["count"] == 0
    ok = client.post(
        "/validate",
        files={"file": ("t.csv", io.BytesIO(_csv()), "text/csv")},
        headers=headers,
    )
    assert ok.status_code == 200
    assert _validation_counts["203.0.113.50"]["count"] == 1


def test_rejected_kite_and_alpaca_do_not_consume_a_try(client):
    headers = {"X-Forwarded-For": "203.0.113.60"}
    for _ in range(4):
        kite = client.post(
            "/validate/kite",
            files={"file": ("t.txt", io.BytesIO(b"x"), "text/plain")},
            headers=headers,
        )
        assert kite.status_code == 400
        alpaca = client.post(
            "/validate/alpaca",
            json={"api_key": "short", "secret_key": "short"},
            headers=headers,
        )
        assert alpaca.status_code == 400
    assert _validation_counts["203.0.113.60"]["count"] == 0


def test_debug_routes_reject_hardcoded_tokens(client, monkeypatch):
    monkeypatch.delenv("DEBUG_ADMIN_TOKEN", raising=False)
    _validation_counts["203.0.113.70"]["count"] = 3
    for token in _HARDCODED:
        assert client.get("/debug/reset-limit", params={"t": token}).status_code == 404
        assert client.get("/debug/token", params={"t": token}).status_code == 404
    assert _validation_counts["203.0.113.70"]["count"] == 3
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_debug_routes_require_env_secret_and_do_not_leak_it(client, monkeypatch):
    monkeypatch.setenv("DEBUG_ADMIN_TOKEN", "s3cret-value")
    monkeypatch.setenv("TEST_REPORT_TOKEN", "should-not-leak")
    _validation_counts["203.0.113.71"]["count"] = 2
    assert client.get("/debug/reset-limit", params={"t": "nope"}).status_code == 404
    assert _validation_counts["203.0.113.71"]["count"] == 2
    reset = client.get("/debug/reset-limit", params={"t": "s3cret-value"})
    assert reset.status_code == 200
    assert "203.0.113.71" not in _validation_counts
    info = client.get("/debug/token", params={"t": "s3cret-value"})
    assert info.status_code == 200
    body = info.text
    assert "should-not-leak" not in body
    assert "env_vars" not in body
    assert "TEST_REPORT_TOKEN" not in body


def test_api_security_headers(client):
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert "max-age=" in headers["strict-transport-security"]
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"
    policy = headers["content-security-policy"]
    assert "frame-ancestors 'none'" in policy
    assert "default-src 'none'" in policy


def test_ops_counts_closed_trades_and_never_passes_on_error(monkeypatch):
    same_second = pd.Timestamp("2024-01-02 10:15:01")
    under = _compute_sebi_checks(pd.DataFrame({"date": [same_second] * 9, "pnl": [0.01] * 9}), None)
    ops = under["ops_threshold"]
    assert ops["passed"] is False
    assert ops["status"] == "not_evaluated"
    assert ops["value"] == 9
    assert ops["counted"] == "closed_trades"

    over = _compute_sebi_checks(pd.DataFrame({"date": [same_second] * 10, "pnl": [0.01] * 10}), None)
    assert over["ops_threshold"]["status"] == "fail"
    assert over["ops_threshold"]["passed"] is False
    assert over["ops_threshold"]["value"] == 10

    spread = pd.date_range("2024-01-02 10:00:00", periods=12, freq="s")
    quiet = _compute_sebi_checks(pd.DataFrame({"date": spread, "pnl": [0.01] * 12}), None)
    assert quiet["ops_threshold"]["value"] == 1
    assert quiet["ops_threshold"]["passed"] is False

    missing = _compute_sebi_checks(pd.DataFrame({"pnl": [0.01] * 5}), None)
    assert missing["ops_threshold"]["passed"] is False
    assert missing["ops_threshold"]["status"] == "not_evaluated"
    assert missing["session_logout"]["passed"] is False

    def boom(*_a, **_k):
        raise RuntimeError("timestamp failure")

    monkeypatch.setattr(pd, "to_datetime", boom)
    broken = _compute_sebi_checks(pd.DataFrame({"date": [same_second], "pnl": [0.01]}), None)
    assert broken["ops_threshold"]["passed"] is False
    assert broken["ops_threshold"]["status"] == "not_evaluated"
    assert broken["session_logout"]["passed"] is False


def test_frontend_assets_prices_and_canonicals():
    for name in (
        "og-image.png",
        "favicon.ico",
        "favicon.svg",
        "step1-upload.png",
        "step2-depth.png",
        "step3-analysis.png",
        "step4-score.png",
        "cookie-notice.js",
    ):
        assert (FRONTEND / name).stat().st_size > 50

    pricing = (FRONTEND / "pricing.html").read_text()
    assert "choosePlan('lite')" in pricing
    assert "create-checkout-paddle-lite" in pricing
    assert 'data-usd="$1.27"' in pricing
    assert 'data-usd="$1.20"' not in pricing
    assert "table-scroll" in pricing
    assert "og:image" in pricing
    assert "https://quantproof.net/og-image.png" in pricing

    index = (FRONTEND / "index.html").read_text()
    assert "get('plan')" in index
    assert "localhost" in index
    assert "0 0 1.448" not in index
    assert 'rel="icon"' in index

    privacy = (FRONTEND / "privacy-policy.html").read_text()
    assert "Google Analytics" in privacy
    assert "Formspree" in privacy
    assert "Paddle" in privacy
    assert "Groq" in privacy
    assert "We do not currently use cookies" not in privacy
    assert "client-side in your browser" not in privacy.lower()

    render = (ROOT / "render.yaml").read_text()
    for header in (
        "Strict-Transport-Security",
        "Content-Security-Policy",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Referrer-Policy",
        "frame-ancestors 'none'",
        "googletagmanager.com",
        "formspree.io",
        "paddle.com",
        "razorpay.com",
    ):
        assert header in render

    sitemap = (FRONTEND / "sitemap.xml").read_text()
    pages = {
        "index.html": "https://quantproof.net/",
        "pricing.html": "https://quantproof.net/pricing",
        "methodology.html": "https://quantproof.net/methodology",
        "faq.html": "https://quantproof.net/faq",
        "privacy-policy.html": "https://quantproof.net/privacy-policy",
        "terms-of-service.html": "https://quantproof.net/terms-of-service",
        "refund-policy.html": "https://quantproof.net/refund-policy",
        "ftmo-preparation-guide.html": "https://quantproof.net/ftmo-preparation-guide",
    }
    for name, url in pages.items():
        html = (FRONTEND / name).read_text()
        assert f'rel="canonical" href="{url}"' in html or f"rel='canonical' href='{url}'" in html
        assert f"<loc>{url}</loc>" in sitemap
        assert "cookie-notice.js" in html
