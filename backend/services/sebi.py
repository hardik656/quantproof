"""SEBI algo-trading compliance checks.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import pandas as pd

from services.client_ip import client_ip

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


def _ops_not_evaluated(note: str, value="N/A") -> dict:
    """A check that did not run, or that cannot prove compliance, is never a pass."""
    return {
        "passed": False,
        "status": "not_evaluated",
        "evaluated": False,
        "value": value,
        "note": note,
        "method": "FROM_DATA",
        "counted": "closed_trades",
    }


def _compute_sebi_checks(df: pd.DataFrame, request, registered_ip: str = "") -> dict:
    """
    Compute SEBI algo framework compliance checks for display. Does NOT affect Fundable Score.

    Orders per second vs closed trades
    -----------------------------------
    SEBI's registration line is orders per second (order messages), not completed
    round trips. Both Kite parsers reduce the CSV to FIFO-paired closed trades
    before this function runs: one row per completed trade, timestamped at the
    closing fill. Entry orders, cancels, rejects, and still-open orders are gone.
    This branch does not change those parsers.

    Choice: count closed trades per second and treat that count as a lower bound
    on the order rate. It is not relabelled as an order count.

    - If the lower bound is already at or above 10, the check fails. The order
      rate cannot be lower than the closed-trade rate, so the registration line
      has been met.
    - If the lower bound is under 10, or there is no usable timestamp, or the
      calculation errors, ``passed`` is false and ``status`` is ``not_evaluated``.
      A trade rate under 10 can still hide an order rate over 10 (each trade is
      normally at least an entry and an exit). That result must not be shown as a pass.
    """
    result = {}

    # ── CHECK 1: OPS Threshold (<10 orders/second) ─────────────────────────
    # SEBI: strategies placing >=10 orders/second require mandatory registration.
    # pandas 3 removed the "S" offset alias; "s" is seconds.
    try:
        ops_limit = SEBI_CONFIG["ops_threshold"]["max_orders_per_second"]
        if "date" in df.columns and df["date"].notna().any():
            seconds = pd.to_datetime(df["date"], errors="coerce").dropna().dt.floor("s")
            if len(seconds) == 0:
                result["ops_threshold"] = _ops_not_evaluated(
                    "No usable timestamps. OPS was not evaluated and is not a pass."
                )
            else:
                max_ops = int(seconds.groupby(seconds).size().max())
                if max_ops >= ops_limit:
                    result["ops_threshold"] = {
                        "passed": False,
                        "status": "fail",
                        "evaluated": True,
                        "value": max_ops,
                        "note": (
                            f"Max {max_ops} closed trades in one second. "
                            "The file has already been reduced to completed trades, "
                            "so the order rate is at least this high and meets or exceeds "
                            f"SEBI's {ops_limit} orders/second registration line."
                        ),
                        "method": "FROM_DATA",
                        "counted": "closed_trades",
                    }
                else:
                    result["ops_threshold"] = _ops_not_evaluated(
                        (
                            f"Max {max_ops} closed trades in one second, under {ops_limit}. "
                            "SEBI's limit counts orders, not closed trades. Each trade is usually "
                            "at least an entry and an exit, and cancelled or still-open orders are "
                            "absent, so this is a lower bound only and is not a pass."
                        ),
                        value=max_ops,
                    )
        else:
            result["ops_threshold"] = _ops_not_evaluated(
                "No timestamp data. OPS was not evaluated and is not a pass."
            )
    except Exception:
        result["ops_threshold"] = _ops_not_evaluated("OPS check could not be completed. This is not a pass.")

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
            outside_ok = outside_count == 0
            result["session_logout"] = {
                "passed": outside_ok,
                "status": "pass" if outside_ok else "warning",
                "value": outside_count,
                "note": (
                    "No activity outside market hours detected in this file."
                    if outside_ok
                    else f"{outside_count} row(s) detected outside NSE market hours — verify session auto-logout."
                ),
                "method": "FROM_DATA",
            }
        else:
            result["session_logout"] = {
                "passed": False,
                "status": "not_evaluated",
                "value": "N/A",
                "note": "No timestamp data. Session logout was not evaluated and is not a pass.",
                "method": "FROM_DATA",
            }
    except Exception:
        result["session_logout"] = {
            "passed": False,
            "status": "not_evaluated",
            "value": "N/A",
            "note": "Session check could not be completed. This is not a pass.",
            "method": "FROM_DATA",
        }

    # ── CHECK 3: Static IP — soft check ────────────────────────────────────
    # Compare registered_ip (user-entered) to actual request IP
    # Performed in memory only — nothing written to disk or database
    try:
        if registered_ip and registered_ip.strip() and request:
            current_ip = client_ip(request)
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
