"""Health, root, keep-alive and debug endpoints.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, HTTPException

from config import API_VERSION
from services.tokens import REPORT_ACCESS_TOKENS, _is_pro_token, _token_tier, _validation_counts

router = APIRouter()


@router.get("/")
def root():
    return {
        "status": "QuantProof API is live",
        "version": API_VERSION,
        "checks": "37 scored checks + 4 plausibility checks + 5 crash simulations",
        "new_in_v1.4": [
            "Information Ratio (vs Benchmark) - detects beta-capture disguised as alpha",
            "Beta Exposure (vs Benchmark) - flags high market correlation",
            "Crash exposure model hardened: higher floor, AR(1) penalty, [UNCALIBRATED] labels",
            "Regime auto-detection via SPY 20-day SMA (yfinance) for CSV + Alpaca paths",
            "Alpaca: position-state tracker replaces broken buy/sell stack pairing",
        ],
        "new_in_v1.3": [
            "CVaR/Expected Shortfall",
            "Fractional Kelly + Ruin Probability",
            "Market Impact (Almgren-Chriss)",
            "Deflated Sharpe",
            "Purged Walk-Forward CV",
            "HMM Regime Detection",
            "Capacity Analysis",
            "Interactive Dashboard API",
        ],
    }


@router.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}


def _debug_authorized(provided: str) -> bool:
    """True only when DEBUG_ADMIN_TOKEN is set and ``provided`` matches it.

    The old hardcoded tokens (dev-reset, test-123, quantproof-test-112211) are
    not accepted. An unset secret disables the route entirely.
    """
    secret = os.getenv("DEBUG_ADMIN_TOKEN", "").strip()
    token = (provided or "").strip()
    if not secret or not token:
        return False
    secret_b, token_b = secret.encode(), token.encode()
    if len(secret_b) != len(token_b):
        return False
    return hmac.compare_digest(secret_b, token_b)


@router.get("/debug/reset-limit")
def debug_reset_limit(t: str = ""):
    """Reset every in-memory rate-limit counter. Disabled unless DEBUG_ADMIN_TOKEN is set."""
    if not _debug_authorized(t):
        raise HTTPException(404, "Not found")
    cleared = len(_validation_counts)
    _validation_counts.clear()
    return {"status": "ok", "cleared_ips": cleared, "message": "All rate limits reset"}


@router.get("/debug/token")
def debug_token(t: str = ""):
    """Check the tier of the admin token. Does not report which secrets are configured."""
    if not _debug_authorized(t):
        raise HTTPException(404, "Not found")
    tier = _token_tier(t)
    return {
        "token_preview": t[:8] + "..." if len(t) > 8 else t,
        "tier": tier,
        "is_pro": _is_pro_token(t),
        "in_token_store": t in REPORT_ACCESS_TOKENS,
    }


@router.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    """
    Keep-alive endpoint.
    Hit this every 10 minutes via an external cron (cron-job.org, UptimeRobot, etc.)
    to prevent Render free tier from spinning down.
    URL to monitor: https://quantproof-backend.onrender.com/ping
    """
    return {"status": "alive", "version": API_VERSION}
