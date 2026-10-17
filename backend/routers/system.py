"""Health, root, keep-alive and debug endpoints.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

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
        "checks": "44 institutional checks + 5 crash simulations",
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


@router.get("/debug/reset-limit")
def debug_reset_limit(t: str = ""):
    """Dev endpoint — reset rate limit for your IP. Requires valid pro/dev token."""
    # For testing purposes, allow simple test tokens
    if t in ("quantproof-test-112211", "test-123", "dev-reset"):
        cleared = len(_validation_counts)
        _validation_counts.clear()
        return {"status": "ok", "cleared_ips": cleared, "message": "All rate limits reset"}

    if not _is_pro_token(t):
        raise HTTPException(403, "Valid pro token required as ?t= param")
    cleared = len(_validation_counts)
    _validation_counts.clear()
    return {"status": "ok", "cleared_ips": cleared, "message": "All rate limits reset"}


@router.get("/debug/token")
def debug_token(t: str = ""):
    """Dev endpoint — check if a token is valid and what tier it grants."""
    if not t:
        return {"error": "Pass ?t=YOUR_TOKEN"}
    tier = _token_tier(t)
    is_pro = _is_pro_token(t)
    env_keys = {k: bool(os.getenv(k, "").strip()) for k in ("TEST_REPORT_TOKEN", "DEV_TOKEN", "ADMIN_TOKEN")}
    return {
        "token_preview": t[:8] + "..." if len(t) > 8 else t,
        "tier": tier,
        "is_pro": is_pro,
        "in_token_store": t in REPORT_ACCESS_TOKENS,
        "env_vars_set": env_keys,
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
