"""Rate limiting, access-token tiers and token persistence.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import io
import os
import time
import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

# ── Rate limiting & token tiers ─────────────────────────────────────────────
_validation_counts: dict = defaultdict(lambda: {"count": 0, "reset": 0})
FREE_DAILY_LIMIT = 3

# Token prefixes
TOKEN_LITE      = "LITE-"   # ₹99  — fixes + AI top-3 in browser
TOKEN_REPORT    = "RPT-"    # ₹299 — PDF + forced n=20 + full AI
TOKEN_PORTFOLIO = "PORT-"   # ₹199 — portfolio correlation endpoint
TOKEN_PRO       = "PRO-"    # ₹999 — 30-day: unlimited + n=50 + all

# Pro token store: token -> expiry_timestamp
_PRO_TOKENS: dict = {}
_PRO_TOKENS_FILE = "/tmp/qp_pro_tokens.json"

def _load_pro_tokens():
    try:
        import json
        if os.path.exists(_PRO_TOKENS_FILE):
            with open(_PRO_TOKENS_FILE) as f:
                _PRO_TOKENS.update(json.load(f))
    except Exception:
        pass

def _save_pro_tokens():
    try:
        import json
        with open(_PRO_TOKENS_FILE, "w") as f:
            json.dump(_PRO_TOKENS, f)
    except Exception:
        pass

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
    if _is_pro_token(token):     return "pro"
    if token in REPORT_ACCESS_TOKENS:
        if token.startswith(TOKEN_PRO):      return "pro"
        if token.startswith(TOKEN_REPORT):   return "report"
        if token.startswith(TOKEN_PORTFOLIO):return "portfolio"
        if token.startswith(TOKEN_LITE):     return "lite"
        return "report"  # legacy tokens without prefix = RPT
    return "none"

def check_rate_limit(ip: str, token: str = "") -> tuple[bool, int]:
    # Pro tokens bypass rate limit entirely AND clear any existing counter for this IP
    if _is_pro_token(token) or token.startswith(TOKEN_PRO):
        # Reset counter so they don't get blocked even if they hit limit before setting token
        if ip in _validation_counts:
            _validation_counts[ip]["count"] = 0
        return True, 999
    now = time.time()
    rec = _validation_counts[ip]
    if now > rec["reset"]:
        rec["count"] = 0
        rec["reset"] = now + 86400
    if rec["count"] >= FREE_DAILY_LIMIT:
        remaining_secs = int(rec["reset"] - now)
        return False, remaining_secs
    rec["count"] += 1
    return True, FREE_DAILY_LIMIT - rec["count"]

REPORT_ACCESS_TOKENS = set()
TOKENS_FILE = "/tmp/qp_report_tokens.json"

def _save_tokens():
    try:
        import json
        with open(TOKENS_FILE, "w") as f:
            json.dump(list(REPORT_ACCESS_TOKENS), f)
    except Exception:
        pass

def _load_tokens():
    try:
        import json
        if os.path.exists(TOKENS_FILE):
            with open(TOKENS_FILE, "r") as f:
                tokens = json.load(f)
                REPORT_ACCESS_TOKENS.update(set(tokens))
    except Exception:
        pass

_load_tokens()
