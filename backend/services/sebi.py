"""SEBI algo-trading compliance checks.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import pandas as pd

# =========================================================
# SEBI COMPLIANCE CONFIGURATION
# =========================================================

SEBI_CONFIG = {
    "ops_threshold": {
        "max_orders_per_second": 10,
        "description": "SEBI limit for orders per second before mandatory registration",
    },
    "session_logout": {
        "market_start": "09:15",
        "market_end": "15:30",
        "timezone": "Asia/Kolkata",
        "description": "NSE market hours for session logout verification",
    },
    "turnover_threshold": {
        "daily_limit": 1000000,  # 1 crore INR
        "description": "Daily turnover threshold for enhanced reporting",
    },
    "last_updated": "2024-01-01",
    "version": "1.0",
}


# =========================================================
# SEBI COMPLIANCE CHECKS — KITE ONLY
# =========================================================


def _compute_sebi_checks(df: pd.DataFrame, request, registered_ip: str = "") -> dict:
    """
    Compute SEBI algo framework compliance checks.
    - OPS threshold: verified from timestamps (fully accurate)
    - Session logout: verified from timestamps (indirect)
    - Static IP: soft check — compare registered_ip to request IP in memory, never stored
    Returns a dict passed to the frontend for display. Does NOT affect Fundable Score.
    """
    result = {}

    # ── CHECK 1: OPS Threshold (<10 orders/second) ─────────────────────────
    # SEBI: strategies placing >=10 orders/second require mandatory registration
    try:
        ops_limit = SEBI_CONFIG["ops_threshold"]["max_orders_per_second"]
        if "date" in df.columns and df["date"].notna().any():
            # Group by second, count orders
            df_copy = df.copy()
            df_copy["second"] = df_copy["date"].dt.floor("S")
            ops_counts = df_copy.groupby("second").size()
            max_ops = ops_counts.max() if len(ops_counts) > 0 else 0

            result["ops_threshold"] = {
                "passed": max_ops < ops_limit,
                "value": max_ops,
                "note": f"Max {max_ops} orders in any single second — {'within' if max_ops < ops_limit else 'exceeds'} SEBI limit of {ops_limit} OPS",
                "method": "FROM_DATA",
            }
        else:
            result["ops_threshold"] = {
                "passed": True,
                "value": "N/A",
                "note": "No timestamp data — OPS check skipped",
                "method": "FROM_DATA",
            }
    except Exception:
        result["ops_threshold"] = {
            "passed": True,
            "value": "N/A",
            "note": "OPS check failed",
            "method": "FROM_DATA",
        }

    # ── CHECK 2: Daily Session Logout ───────────────────────────────────────
    # SEBI: API session must auto-logout before next trading day
    # Indirect check: look for orders outside NSE market hours (9:15am–3:30pm)
    try:
        if "date" in df.columns and df["date"].notna().any():
            dates = pd.to_datetime(df["date"])
            hours = dates.dt.hour + dates.dt.minute / 60
            # Outside NSE hours: before 9:00 or after 16:00 (allowing 30min buffer)
            outside = ((hours < 9.0) | (hours > 16.0)) & (hours > 0)
            outside_count = int(outside.sum())
            result["session_logout"] = {
                "passed": outside_count == 0,
                "value": outside_count,
                "note": (
                    "No orders outside market hours detected."
                    if outside_count == 0
                    else f"{outside_count} order(s) detected outside NSE market hours — verify session auto-logout."
                ),
                "method": "FROM_DATA",
            }
        else:
            result["session_logout"] = {
                "passed": True,
                "value": "N/A",
                "note": "No timestamp data",
                "method": "FROM_DATA",
            }
    except Exception:
        result["session_logout"] = {
            "passed": True,
            "value": "N/A",
            "note": "Session check failed",
            "method": "FROM_DATA",
        }

    # ── CHECK 3: Static IP — soft check ────────────────────────────────────
    # Compare registered_ip (user-entered) to actual request IP
    # Performed in memory only — nothing written to disk or database
    try:
        if registered_ip and registered_ip.strip() and request:
            current_ip = (
                request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or (request.client.host if request.client else "")
                or "unknown"
            )
            reg = registered_ip.strip()
            match = reg == current_ip
            result["ip_check"] = {
                "result": "match" if match else "mismatch",
                "match": match,
                "registered_ip": reg,
                "current_ip": current_ip,  # shown to user — their own IP
                "note": (
                    "IP matches registered address."
                    if match
                    else f"Current IP ({current_ip}) does not match registered IP ({reg}). "
                    "Note: VPN or proxy use would cause a mismatch — this is a soft check only."
                ),
                "method": "SOFT_CHECK",
            }
        else:
            result["ip_check"] = {
                "result": "not_entered",
                "match": None,
                "note": "No registered IP provided — check skipped. Verify directly with Zerodha.",
                "method": "SOFT_CHECK",
            }
    except Exception:
        result["ip_check"] = {
            "result": "not_entered",
            "match": None,
            "note": "IP check failed",
            "method": "SOFT_CHECK",
        }

    return result
