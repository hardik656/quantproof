"""TradingView webhook session endpoints.

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
from validator import QuantProofValidator
import uuid
from config import extract_n_trials
from schemas import CheckResultResponse, CrashSimResponse, ValidationResponse
from services.benchmarks import assign_regimes, get_benchmark_type, get_nifty_benchmark, get_spy_benchmark
from services.cache import _cache_validation

router = APIRouter()

# =========================================================
# TRADINGVIEW WEBHOOK
# =========================================================

TV_SESSIONS: dict = {}
TV_SESSION_TTL_SECONDS      = 7 * 24 * 3600
TV_MAX_ALERTS_PER_SESSION   = 2000
TV_SESSIONS_FILE            = "/tmp/qp_tv_sessions.json"

def _tv_save_sessions():
    try:
        import json
        serialisable = {}
        for sid, sess in TV_SESSIONS.items():
            s = dict(sess)
            alerts_clean = []
            for a in s.get("alerts", []):
                ac = dict(a)
                if hasattr(ac.get("timestamp"), "isoformat"):
                    ac["timestamp"] = ac["timestamp"].isoformat()
                alerts_clean.append(ac)
            s["alerts"] = alerts_clean
            serialisable[sid] = s
        with open(TV_SESSIONS_FILE, "w") as f:
            json.dump(serialisable, f)
    except Exception as e:
        print(f"[tv_save] Warning: could not persist sessions: {e}")

def _tv_load_sessions():
    try:
        import json
        if not os.path.exists(TV_SESSIONS_FILE):
            return
        with open(TV_SESSIONS_FILE) as f:
            saved = json.load(f)
        now = time.time()
        loaded = 0
        for sid, sess in saved.items():
            if now > sess.get("expires_at", 0):
                continue
            TV_SESSIONS[sid] = sess
            loaded += 1
        print(f"[tv_load] Restored {loaded} TV session(s) from disk")
    except Exception as e:
        print(f"[tv_load] Warning: could not load sessions: {e}")

_tv_load_sessions()


def _tv_cleanup():
    now     = time.time()
    expired = [sid for sid, data in TV_SESSIONS.items()
               if now - data.get("created_at", 0) > TV_SESSION_TTL_SECONDS]
    for sid in expired:
        del TV_SESSIONS[sid]
    if expired:
        _tv_save_sessions()


def _tv_pair_alerts(alerts: list) -> tuple[list, list]:
    """
    Pair TradingView BUY/SELL alerts into closed round-trip trades.
    """
    if not alerts:
        return [], []

    def _parse_ts(ts):
        if ts is None:
            return time.time()
        if isinstance(ts, (int, float)):
            return float(ts)
        try:
            return pd.Timestamp(ts).timestamp()
        except Exception:
            return time.time()

    def _ts_to_datetime(ts_float):
        try:
            return pd.Timestamp(ts_float, unit="s")
        except Exception:
            return None

    sorted_alerts = sorted(alerts, key=lambda x: _parse_ts(x.get("timestamp")))

    lots         = defaultdict(list)
    trades       = []
    audit_flags  = []
    orphan_sells = []

    for alert in sorted_alerts:
        symbol = str(alert.get("symbol", "UNKNOWN"))
        action = str(alert.get("action", "")).upper().strip()
        price  = float(alert.get("price", 0))
        qty    = float(alert.get("quantity", 1))
        ts     = _parse_ts(alert.get("timestamp"))
        date   = _ts_to_datetime(ts)

        if price <= 0:
            continue

        if action == "BUY":
            if lots[symbol] and lots[symbol][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[symbol] and lots[symbol][0]["side"] == "short":
                    lot       = lots[symbol][0]
                    close_qty = min(remaining, lot["qty"])
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[symbol].pop(0)
                        continue
                    row = {"pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6), "ticker": symbol, "side": "short"}
                    if date: row["date"] = date
                    trades.append(row)
                    lot["qty"] -= close_qty
                    remaining  -= close_qty
                    if lot["qty"] == 0:
                        lots[symbol].pop(0)
                if remaining > 0:
                    lots[symbol].append({"qty": remaining, "price": price, "date": date, "side": "long"})
            else:
                lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "long"})

        elif action == "SELL":
            if not lots[symbol]:
                lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "short"})
                continue

            if lots[symbol][0]["side"] == "short":
                lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "short"})
                continue

            remaining = qty
            while remaining > 0 and lots[symbol] and lots[symbol][0]["side"] == "long":
                lot       = lots[symbol][0]
                close_qty = min(remaining, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[symbol].pop(0)
                    continue
                row = {"pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6), "ticker": symbol, "side": "long"}
                if date: row["date"] = date
                trades.append(row)
                lot["qty"] -= close_qty
                remaining  -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

        elif action in ("CLOSE", "EXIT", "FLAT"):
            if lots[symbol]:
                lot = lots[symbol][0]
                try:
                    pnl_pct = (
                        (price - lot["price"]) / lot["price"] if lot["side"] == "long"
                        else (lot["price"] - price) / lot["price"]
                    )
                except ZeroDivisionError:
                    lots[symbol].clear()
                    continue
                row = {"pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6), "ticker": symbol, "side": lot["side"]}
                if date: row["date"] = date
                trades.append(row)
                lots[symbol].clear()

    open_positions = {s: (sum(l["qty"] for l in ls), ls[0]["side"]) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)} {side})" for s, (q, side) in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f" {len(open_positions)} position(s) still open "
            f"(no closing alert yet): {sym_list}"
        )

    total_alerts = len(alerts)
    completed    = len(trades)
    audit_flags.append(
        f"📊 {total_alerts} alert(s) → {completed} completed trade(s) | "
        f"{len(open_positions)} position(s) open"
    )

    return trades, audit_flags


class TVSessionResponse(BaseModel):
    session_id:  str
    webhook_url: str
    max_alerts:  int
    ttl_hours:   int

class TVAlert(BaseModel):
    symbol:    str
    action:    str
    price:     float
    quantity:  Optional[float] = 1.0
    timestamp: Optional[float] = None

class TVStatusResponse(BaseModel):
    session_id:      str
    alerts_received: int
    trades_paired:   int
    status:          str
    created_at:      float
    expires_at:      float


@router.get("/webhook/tradingview/new-session", response_model=TVSessionResponse)
def create_tv_session():
    _tv_cleanup()
    session_id  = str(uuid.uuid4())
    webhook_url = (
        f"{os.getenv('BASE_URL', 'https://quantproof-backend.onrender.com')}"
        f"/webhook/tradingview/{session_id}"
    )
    TV_SESSIONS[session_id] = {
        "session_id":  session_id,
        "webhook_url": webhook_url,
        "alerts":      [],
        "created_at":  time.time(),
        "expires_at":  time.time() + TV_SESSION_TTL_SECONDS,
    }
    _tv_save_sessions()
    return TVSessionResponse(
        session_id=session_id,
        webhook_url=webhook_url,
        max_alerts=TV_MAX_ALERTS_PER_SESSION,
        ttl_hours=TV_SESSION_TTL_SECONDS // 3600,
    )


@router.post("/webhook/tradingview/{session_id}")
def receive_tv_alert(session_id: str, alert: TVAlert):
    _tv_cleanup()
    if session_id not in TV_SESSIONS:
        raise HTTPException(404, "Session not found or expired")
    session = TV_SESSIONS[session_id]
    if time.time() > session["expires_at"]:
        del TV_SESSIONS[session_id]
        raise HTTPException(404, "Session expired")
    if len(session["alerts"]) >= TV_MAX_ALERTS_PER_SESSION:
        raise HTTPException(429, "Alert limit exceeded for this session")
    alert_dict = alert.dict()
    if alert_dict.get("timestamp") is None:
        alert_dict["timestamp"] = time.time()
    session["alerts"].append(alert_dict)
    return {"status": "received", "alerts_count": len(session["alerts"])}


@router.get("/webhook/tradingview/{session_id}/status", response_model=TVStatusResponse)
def get_tv_status(session_id: str):
    _tv_cleanup()
    if session_id not in TV_SESSIONS:
        raise HTTPException(404, "Session not found or expired")
    session = TV_SESSIONS[session_id]
    alerts  = session["alerts"]
    trades, _ = _tv_pair_alerts(alerts)
    status  = "active"
    if time.time() > session["expires_at"]:
        status = "expired"
    elif len(alerts) >= TV_MAX_ALERTS_PER_SESSION:
        status = "full"
    return TVStatusResponse(
        session_id=session_id,
        alerts_received=len(alerts),
        trades_paired=len(trades),
        status=status,
        created_at=session["created_at"],
        expires_at=session["expires_at"],
    )


@router.post("/webhook/tradingview/{session_id}/validate", response_model=ValidationResponse)
def validate_tv_alerts(session_id: str, enhanced: bool = Form(False), request: Request = None):
    """
    Validate TradingView session alerts using QuantProof.
    """
    _tv_cleanup()
    if session_id not in TV_SESSIONS:
        raise HTTPException(404, "Session not found or expired")

    session = TV_SESSIONS[session_id]
    alerts  = session["alerts"]

    if len(alerts) < 10:
        raise HTTPException(
            400,
            f"Need at least 10 alerts for validation, only have {len(alerts)}. "
            "Keep sending webhook alerts and try again."
        )

    trades, tv_flags = _tv_pair_alerts(alerts)

    if len(trades) < 10:
        raise HTTPException(
            400,
            f"Only paired {len(trades)} completed trade(s) from {len(alerts)} alert(s). "
            "Need at least 10 completed round-trips (BUY + SELL pairs). "
            "Keep sending alerts and validate again when you have more trades."
        )

    df = pd.DataFrame(trades)

    regime_note = ""
    if "date" in df.columns and df["date"].notna().any():
        df, regime_note = assign_regimes(df)

    try:
        _bm_type = get_benchmark_type(df, None)
        _bm, _bm_note = (get_nifty_benchmark(df) 
            if _bm_type == "nifty" 
            else get_spy_benchmark(df))
        n_trials = extract_n_trials(request)
        validator = QuantProofValidator(
            df,
            benchmark_returns=_bm,
            benchmark_type=_bm_type,
            n_trials=n_trials,
            enable_prop_firm=enhanced
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    import sys
    try:
        report = validator.run(enhanced=enhanced)
    except Exception as e:
        # Log the full traceback so we can see 
        # exactly which line is overflowing
        import traceback
        tb = traceback.format_exc()
        print(f"[TV_VALIDATE_ERROR] {tb}", 
              file=sys.stderr, flush=True)
        raise HTTPException(400, 
            f"Validation failed: {str(e)}")

    scored        = [c for c in report.checks if c.category != "Plausibility"]
    top_issues    = [c.fix  for c in sorted(scored, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]]
    visible_checks = [c for c in report.checks if c.category != "Plausibility"]

    date_range = "No date data"
    if "date" in df.columns and df["date"].notna().any():
        try:
            dates = pd.to_datetime(df["date"]).dropna()
            date_range = f"{dates.min().date()} to {dates.max().date()}"
        except Exception:
            pass

    audit_note = (
        f"📊 {len(alerts)} TradingView alerts → {len(trades)} paired trades"
        + (f" | {regime_note}" if regime_note else "")
    )
    extra_tv_flags = [f for f in tv_flags if f.strip() not in audit_note]

    return ValidationResponse(
        fundable_score=report.score,
        grade=report.grade,
        summary=(
            f"QuantProof analyzed {report.total_trades} TradingView-constructed trades across "
            f"{len(scored)} institutional checks + 5 crash simulations. "
            f"Source: TradingView webhook alerts — paired into trades. "
            f"Fundable Score: {report.score}/100 ({report.grade.split('—')[0].strip()})."
        ),
        checks=[CheckResultResponse(**vars(c)) for c in visible_checks],
        crash_sims=[CrashSimResponse(
            crash_name=s.crash_name, year=s.year,
            description=s.description,
            market_drop=s.market_drop,
            strategy_drop=s.strategy_drop,
            survived=s.survived,
            emotional_verdict=s.emotional_verdict
        ) for s in report.crash_sims],
        total_trades=report.total_trades,
        profitable_trades=report.profitable_trades,
        date_range=date_range,
        sharpe=round(report.sharpe, 2),
        max_drawdown=round(report.max_drawdown, 4),
        win_rate=round(report.win_rate, 1),
        top_issues=top_issues,
        top_strengths=top_strengths,
        validation_hash=report.validation_hash + "-TV",
        validation_date=report.validation_date,
        engine_version=report.engine_version,
        audit_flags=report.audit_flags + [audit_note] + extra_tv_flags,
        plausibility_summary=report.plausibility_summary,
        scale_warning=getattr(report, '_scale_warning', False),
        currency="USD",
        assumptions=report.assumptions + [
            "Trade data sourced from TradingView webhook alerts — self-reported",
            "Trades constructed by pairing BUY/SELL alerts in chronological order",
            "Not broker-verified — cannot be used for prop firm compliance submissions",
        ],
        dashboard=None,
        ai_plan=None,
        equity_curve=getattr(report, 'equity_curve', None),
        prop_firm_compliance=getattr(report, 'prop_firm_compliance', None),
    )

    # CRITICAL: Always cache TV validation results, even if errors occur
    _vr_cache = {
        "fundable_score": report.score,
        "grade": report.grade,
        "summary": (
            f"QuantProof analyzed {report.total_trades} TradingView-constructed trades across "
            f"{len(scored)} institutional checks + 5 crash simulations. "
            f"Source: TradingView webhook alerts — paired into trades. "
            f"Fundable Score: {report.score}/100 ({report.grade.split('—')[0].strip()})."
        ),
        "checks": [CheckResultResponse(**vars(c)) for c in visible_checks],
        "crash_sims": [CrashSimResponse(
            crash_name=s.crash_name, year=s.year,
            description=s.description,
            market_drop=s.market_drop,
            strategy_drop=s.strategy_drop,
            survived=s.survived,
            emotional_verdict=s.emotional_verdict
        ) for s in report.crash_sims],
        "top_issues": top_issues,
        "top_strengths": top_strengths,
        "regime_note": regime_note,
        "audit_flags": report.audit_flags + [audit_note] + extra_tv_flags,
        "plausibility_summary": report.plausibility_summary,
        "scale_warning": getattr(report, '_scale_warning', False),
        "sebi_checks": {},  # TV is US-based, no SEBI checks
        "sebi_declaration": {},  # TV is US-based, no SEBI declaration
        "equity_curve": getattr(report, 'equity_curve', None),
    }
    
    # Add prop firm compliance to VR cache only when enhanced validation is triggered
    if enhanced and getattr(report, 'prop_firm_compliance', None):
        _vr_cache["prop_firm_compliance"] = getattr(report, 'prop_firm_compliance', None)
    
    print(f"[DEBUG] TV Cache - Saving validation hash: '{report.validation_hash + '-TV'}'")
    _cache_validation(report.validation_hash + "-TV", _vr_cache)
    print(f"[DEBUG] TV Cache - Successfully stored in memory")


@router.post("/webhook/tradingview/{session_id}/reset")
def reset_tv_session(session_id: str):
    _tv_cleanup()
    if session_id not in TV_SESSIONS:
        raise HTTPException(404, "Session not found or expired")
    TV_SESSIONS[session_id]["alerts"] = []
    return {"status": "reset", "message": "Alerts cleared, session remains active"}


