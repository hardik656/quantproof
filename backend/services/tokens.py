"""Rate limiting, access-token tiers and token persistence.

Token state is persisted through ``db.store`` (JSON files by default,
Postgres when DATABASE_URL is set).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

from db import store

# ── Rate limiting & token tiers ─────────────────────────────────────────────
_validation_counts: dict = defaultdict(lambda: {"count": 0, "reset": 0})
FREE_DAILY_LIMIT = 3

# Token prefixes
TOKEN_LITE = "LITE-"  # ₹99  — fixes + AI top-3 in browser
TOKEN_REPORT = "RPT-"  # ₹299 — PDF + forced n=20 + full AI
TOKEN_PORTFOLIO = "PORT-"  # ₹199 — portfolio correlation endpoint
TOKEN_PRO = "PRO-"  # ₹999 — 30-day: unlimited + n=50 + all

# Pro token store: token -> expiry_timestamp
_PRO_TOKENS: dict = {}


def _load_pro_tokens() -> None:
    _PRO_TOKENS.update(store.load("pro_tokens"))


def _save_pro_tokens() -> None:
    store.save("pro_tokens", _PRO_TOKENS)


_load_pro_tokens()


def _is_pro_token(token: str) -> bool:
    """
    Returns True if token grants pro-level access.
    Dev override: set TEST_REPORT_TOKEN env var on Render to any value,
    then paste that exact value into the dev panel on ?dev=1.
    """
    if not token:
        return False
    # Check all dev override env vars
    for _env_key in ("TEST_REPORT_TOKEN", "DEV_TOKEN", "ADMIN_TOKEN"):
        _test = os.getenv(_env_key, "").strip()
        if _test and token.strip() == _test:
            return True
    # Real PRO token check
    if not token.startswith(TOKEN_PRO):
        return False
    expiry = _PRO_TOKENS.get(token, 0)
    return time.time() < expiry


def _token_tier(token: str) -> str:
    """Return the tier of a token: 'pro', 'report', 'portfolio', 'lite', or 'none'."""
    # Check all dev override env vars (same as _is_pro_token)
    for _env_key in ("TEST_REPORT_TOKEN", "DEV_TOKEN", "ADMIN_TOKEN"):
        _test = os.getenv(_env_key, "").strip()
        if _test and token.strip() == _test:
            return "pro"
    if _is_pro_token(token):
        return "pro"
    if token in REPORT_ACCESS_TOKENS:
        if token.startswith(TOKEN_PRO):
            return "pro"
        if token.startswith(TOKEN_REPORT):
            return "report"
        if token.startswith(TOKEN_PORTFOLIO):
            return "portfolio"
        if token.startswith(TOKEN_LITE):
            return "lite"
        return "report"  # legacy tokens without prefix = RPT
    return "none"


# Daily validations per tier, as advertised on /pricing. Pro is unlimited.
TIER_DAILY_LIMITS = {"none": FREE_DAILY_LIMIT, "lite": 3, "report": 5, "portfolio": 10}


def check_rate_limit(ip: str, token: str = "") -> tuple[bool, int]:
    """Per-IP daily limit. Returns (allowed, remaining) or (False, seconds_until_reset).

    Only a *valid* token raises the limit: a string that merely starts with
    ``PRO-`` is not enough.
    """
    tier = _token_tier(token) if token else "none"
    if tier == "pro":
        if ip in _validation_counts:
            _validation_counts[ip]["count"] = 0
        return True, 999
    limit = TIER_DAILY_LIMITS.get(tier, FREE_DAILY_LIMIT)
    now = time.time()
    rec = _validation_counts[ip]
    if now > rec["reset"]:
        rec["count"] = 0
        rec["reset"] = now + 86400
    if rec["count"] >= limit:
        remaining_secs = int(rec["reset"] - now)
        return False, remaining_secs
    rec["count"] += 1
    return True, limit - rec["count"]


REPORT_ACCESS_TOKENS: set[str] = set()


def _save_tokens() -> None:
    store.save("report_tokens", {t: True for t in REPORT_ACCESS_TOKENS})


def _load_tokens() -> None:
    REPORT_ACCESS_TOKENS.update(store.load("report_tokens").keys())


_load_tokens()
