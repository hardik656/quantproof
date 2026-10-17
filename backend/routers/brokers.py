"""Broker-connected validation: Alpaca, Interactive Brokers, Zerodha Kite.

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
from fastapi import Response
import sys
from validator import QuantProofValidator, ValidationDashboard
from config import extract_n_trials
from schemas import CheckResultResponse, CrashSimResponse, ValidationResponse
from services.benchmarks import assign_regimes, get_benchmark_type, get_nifty_benchmark, get_spy_benchmark
from services.cache import _cache_validation, _save_validation_cache
from services.ibkr import parse_ibkr_trade_improved
from services.parsers import _dedup_trades, _is_kite_csv, _is_kite_orders_csv, _parse_kite_csv, _parse_kite_orders_csv, pair_alpaca_orders, retry_on_failure
from services.sebi import _compute_sebi_checks
from services.tokens import FREE_DAILY_LIMIT, _validation_counts, check_rate_limit

router = APIRouter()

# =========================================================
# ALPACA BROKER INTEGRATION — VERIFIED VALIDATION
# =========================================================

class AlpacaConnectRequest(BaseModel):
    api_key:    str
    secret_key: str
    mode:       str = "paper"
    enhanced:    bool = False

def _scrub_keys(d: dict) -> dict:
    sensitive = {"api_key", "secret_key", "key", "secret", "token", "password", "auth"}
    return {k: "[REDACTED]" if k.lower() in sensitive else v for k, v in d.items()}

@router.post("/validate/alpaca", response_model=ValidationResponse)
async def validate_alpaca(payload: AlpacaConnectRequest, request: Request = None):
    BROKER_DAILY_LIMIT = 3
    if request:
        # Prefer X-Forwarded-For (Render proxy passes real user IP here)
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "")) or "unknown"
        key_hash = "broker:" + hashlib.sha256(f"broker:{ip}".encode()).hexdigest()[:16]
        rec = _validation_counts[key_hash]
        now = time.time()
        if rec.get("reset", 0) == 0 or now > rec["reset"]:
            rec["count"] = 0
            rec["reset"] = now + 86400
        if rec["count"] >= BROKER_DAILY_LIMIT:
            hours_left = int((rec["reset"] - now) / 3600)
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limit",
                        "message": f"Free limit of {BROKER_DAILY_LIMIT} Alpaca verifications/day reached. Resets in ~{hours_left}h.",
                        "hours_left": hours_left}
            )
        rec["count"] += 1

    api_key    = payload.api_key.strip()
    secret_key = payload.secret_key.strip()
    if not api_key or not secret_key:
        raise HTTPException(400, "API key and secret key are required.")
    if len(api_key) < 10 or len(secret_key) < 10:
        raise HTTPException(400, "Invalid key format - keys are too short.")

    is_paper = payload.mode == "paper"
    base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"

    try:
        import httpx

        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "accept":              "application/json"
        }

        @retry_on_failure(max_retries=3, delay=1.0)
        async def _fetch_alpaca_account(client, base_url, headers):
            return await client.get(f"{base_url}/v2/account", headers=headers)

        @retry_on_failure(max_retries=3, delay=1.0)
        async def _fetch_alpaca_orders(client, base_url, headers):
            return await client.get(
                f"{base_url}/v2/orders",
                headers=headers,
                params={"status": "closed", "limit": 500, "direction": "desc"}
            )

        async with httpx.AsyncClient(timeout=20) as client:
            acct_resp = await _fetch_alpaca_account(client, base_url, headers)
            if acct_resp.status_code == 401:
                raise HTTPException(401, "Invalid Alpaca credentials. Please check your API key and secret.")
            if acct_resp.status_code == 403:
                raise HTTPException(403, "Alpaca API access denied. Make sure your keys have read permissions.")
            if not acct_resp.is_success:
                raise HTTPException(400, f"Alpaca connection failed (status {acct_resp.status_code}).")
            try:
                acct_currency = str(acct_resp.json().get("currency", "USD")).upper()
            except Exception:
                acct_currency = "USD"

            orders_resp = await _fetch_alpaca_orders(client, base_url, headers)
            if not orders_resp.is_success:
                raise HTTPException(400, "Could not fetch order history from Alpaca.")
        orders = orders_resp.json()

        api_key = secret_key = "[DISCARDED]"
        headers = {}
        try:
            payload.api_key = payload.secret_key = "[DISCARDED]"
        except Exception:
            pass

        if not orders or len(orders) < 3:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(orders) if orders else 0,
                "trades_needed": 10,
                "message": (
                    f"Only {len(orders) if orders else 0} orders found in your Alpaca history. "
                    "QuantProof needs at least 10 completed round-trip trades (a BUY and a SELL of the same stock). "
                    "Make sure you have selected the correct account type (paper vs live) and have enough trading history."
                )
            })

        df, parse_flags = pair_alpaca_orders(orders)

        _orphan_flag = next((f for f in parse_flags if "orphan" in f.lower() or "predates" in f.lower()), "")
        _open_flag   = next((f for f in parse_flags if "open position" in f.lower()), "")

        if len(df) < 10:
            _detail_lines = []
            if _orphan_flag:
                _detail_lines.append(
                    "Some sells were excluded because their matching buy happened before "
                    "the visible order history window - these are positions you held before "
                    "the period covered by the API."
                )
            if _open_flag:
                _detail_lines.append(
                    "Some buys are still open (no closing sell yet) and cannot be counted as completed trades."
                )
            _detail = " ".join(_detail_lines) if _detail_lines else (
                "Make sure you have at least 10 closed positions - a BUY and a matching SELL "
                "for the same stock, both within the history window."
            )
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(df),
                "trades_needed": 10,
                "message": (
                    f"Only {len(df)} completed round-trip trade(s) found - need 10. "
                    f"{_detail}"
                )
            })

        total_filled_sides = len([o for o in orders if o.get("status") == "filled"])
        paired_sides = len(df) * 2
        unpaired_pct = (total_filled_sides - paired_sides) / max(total_filled_sides, 1) * 100
        if unpaired_pct > 30:
            print(f"[validate/alpaca] WARNING: {unpaired_pct:.0f}% of orders unpaired - possible options/shorts")

        # —? Run validation — FIX: off event loop with timeout ———————————————?
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
            enable_prop_firm=payload.enhanced
        )
        try:
            loop   = asyncio.get_event_loop()
            report = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: validator.run(enhanced=payload.enhanced)),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(503, {
                "error": "validation_timeout",
                "message": "Validation timed out. Try reducing the date range or number of trades."
            })

        r = report
        scored_checks = [c for c in r.checks if c.category != "Plausibility"]
        top_issues    = [c.fix  for c in sorted(scored_checks, key=lambda x: x.score)[:3]]
        top_strengths = [c.name for c in sorted(scored_checks, key=lambda x: x.score, reverse=True)[:3]]

        date_range = "No date data"
        if "date" in df.columns:
            try:
                date_range = f"{df['date'].dropna().min().date()} to {df['date'].dropna().max().date()}"
            except Exception:
                pass

        check_count = len(scored_checks)
        summary = (
            f"QuantProof VERIFIED analysis of {r.total_trades} trades pulled directly from Alpaca "
            f"({'paper' if is_paper else 'live'} account) across {check_count} institutional checks "
            f"+ 5 crash simulations. "
            f"Fundable Score: {r.score}/100 ({r.grade.split('?')[0].strip()}). "
            f"{'Core risk management needs work.' if r.score < 60 else 'Edge shows real promise ?? focus on execution costs.'}"
        )

        visible_checks = [c for c in r.checks if c.category != "Plausibility"]

        try:
            dashboard_obj  = ValidationDashboard(validator, report)
            dashboard_data = dashboard_obj.generate_interactive_report()
        except Exception:
            dashboard_data = None

        _vr_cache = {
            "fundable_score": min(100.0, round(r.score, 1)), "grade": r.grade,
            "sharpe": round(r.sharpe, 2), "max_drawdown": round(r.max_drawdown, 4),
            "win_rate": round(r.win_rate, 1), "total_trades": r.total_trades,
            "date_range": date_range, "is_verified": True,
            "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                        "value":c.value,"insight":c.insight,"fix":c.fix,
                        "category":c.category} for c in r.checks],
            "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                            "description":s.description,"market_drop":s.market_drop,
                            "strategy_drop":s.strategy_drop,"survived":s.survived,
                            "emotional_verdict":s.emotional_verdict} for s in r.crash_sims],
            "top_issues": top_issues, "top_strengths": top_strengths,
            "audit_flags": r.audit_flags + parse_flags + ([_bm_note] if _bm_note else []),
            "equity_curve": getattr(r, 'equity_curve', None),
            "monte_carlo": getattr(r, 'monte_carlo', None),
            "sebi_checks":  {},  # Empty for non-SEBI brokers (Alpaca is US-based)
        }
        
        # Add prop firm compliance to VR cache only when enhanced validation is triggered
        if payload.enhanced and getattr(r, 'prop_firm_compliance', None):
            _vr_cache["prop_firm_compliance"] = getattr(r, 'prop_firm_compliance', None)
        
        _cache_validation(r.validation_hash, _vr_cache)
        _save_validation_cache()

        return ValidationResponse(
            fundable_score=min(100.0, round(r.score, 1)), grade=r.grade, summary=summary,
            checks=[CheckResultResponse(**vars(c)) for c in visible_checks],
            crash_sims=[CrashSimResponse(
            crash_name=s.crash_name, year=s.year,
            description=s.description,
            market_drop=s.market_drop,
            strategy_drop=s.strategy_drop,
            survived=s.survived,
            emotional_verdict=s.emotional_verdict
        ) for s in r.crash_sims],
            total_trades=r.total_trades, profitable_trades=r.profitable_trades,
            date_range=date_range, sharpe=round(r.sharpe, 2),
            max_drawdown=round(r.max_drawdown, 4), win_rate=round(r.win_rate, 1),
            top_issues=top_issues, top_strengths=top_strengths,
            validation_hash=r.validation_hash + "-VERIFIED",
            validation_date=r.validation_date,
            engine_version=r.engine_version + "-VERIFIED",
            audit_flags=r.audit_flags + parse_flags + ([_bm_note] if _bm_note else []) + [
    f"Source: Alpaca {'Paper' if is_paper else 'Live'} Trading API",
    "Trades independently verified - not user-uploaded",
    "Credential handling: keys discarded after fetch, never logged",
],
            plausibility_summary=r.plausibility_summary,
            assumptions=r.assumptions + [
    "Trade data sourced directly from Alpaca API - not user-submitted",
    f"Account type: {'paper trading' if is_paper else 'live trading'}",
    "API credentials were used read-only and discarded after data fetch",
],
            dashboard=dashboard_data,
            ai_plan=None,
            scale_warning=False,
            currency=acct_currency,
            equity_curve=getattr(r, 'equity_curve', None),
            monte_carlo=getattr(r, 'monte_carlo', None),
            prop_firm_compliance=getattr(r, 'prop_firm_compliance', None),
        )
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e)
        for sensitive in [payload.api_key, payload.secret_key]:
            if sensitive and sensitive != "[DISCARDED]" and len(sensitive) > 4:
                err_str = err_str.replace(sensitive, "[REDACTED]")
        raise HTTPException(500, f"Broker validation failed: {err_str}")



# =========================================================
# IBKR BROKER INTEGRATION
# =========================================================

class IBKRConnectRequest(BaseModel):
    account_id: str
    flex_token: str
    query_id:   str
    days_back:  int = 365
    enhanced:    bool = False

class IBKRConnectResponse(BaseModel):
    source:       str
    account_id:   str
    trades_pulled: int
    date_range:   str
    validation:   ValidationResponse

@router.post("/connect/ibkr", response_model=IBKRConnectResponse)
async def connect_ibkr(payload: IBKRConnectRequest, request: Request = None):
    if request:
        # Prefer X-Forwarded-For (Render proxy passes real user IP here)
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "")) or "unknown"
        allowed, _ = check_rate_limit(ip)
        if not allowed:
            raise HTTPException(429, f"Daily limit of {FREE_DAILY_LIMIT} validations reached.")

    import httpx, xml.etree.ElementTree as ET
    from datetime import datetime, timedelta

    request_url = (
        f"https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
        f"?t={payload.flex_token}&q={payload.query_id}&v=3"
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(request_url)
        resp.raise_for_status()
        root   = ET.fromstring(resp.text)
        status = root.findtext("Status")
        if status != "Success":
            raise HTTPException(400, f"IBKR Flex request failed: {root.findtext('ErrorMessage') or 'Unknown error'}")
        reference_code = root.findtext("ReferenceCode")
        base_url       = root.findtext("Url")
    except ET.ParseError:
        raise HTTPException(400, "Could not parse IBKR response.")
    except httpx.HTTPError as e:
        raise HTTPException(503, f"Could not reach IBKR servers: {str(e)}")

    import asyncio
    await asyncio.sleep(5)

    retrieve_url = f"{base_url}?t={payload.flex_token}&q={reference_code}&v=3"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp2 = await client.get(retrieve_url)
        resp2.raise_for_status()
        report_root = ET.fromstring(resp2.text)
    except Exception as e:
        raise HTTPException(503, f"Could not retrieve IBKR report: {str(e)}")

    trades      = []
    audit_flags_ibkr = []
    open_skipped = 0
    acct_currency = "USD"

    for stmt in report_root.iter("FlexStatement"):
        acct = stmt.get("accountId", "")
        stmt_currency = stmt.get("currency", "") or stmt.get("Currency", "")
        if stmt_currency:
            acct_currency = stmt_currency.upper()
        if payload.account_id and acct != payload.account_id:
            continue
        for trade in stmt.iter("Trade"):
            try:
                # Use improved parser
                trade_data, skip_reason = parse_ibkr_trade_improved(trade)
                if trade_data is None:
                    if skip_reason == "opening_position":
                        open_skipped += 1
                    continue
                trades.append(trade_data)
            except (ValueError, ZeroDivisionError):
                continue

    if open_skipped > 0:
        audit_flags_ibkr.append(
            f"— {open_skipped} opening trade leg(s) skipped — "
            f"only closed trades contribute to P&L"
        )

    if len(trades) < 10:
        raise HTTPException(
            400,
            f"Only {len(trades)} closed trade(s) found in IBKR export. Need at least 10. "
            "Make sure your Flex Query includes: openCloseIndicator, fifoPnlRealized, "
            "netCash, quantity, tradePrice, symbol, tradeDate. "
            "Also check your date range covers enough completed trades."
        )

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["pnl"])

    if "date" in df.columns and df["date"].notna().any():
        df, regime_note = assign_regimes(df)
    else:
        regime_note = ""

    try:
        _bm_type = get_benchmark_type(df, None)
        _bm, _bm_note = (get_nifty_benchmark(df) 
            if _bm_type == "nifty" 
            else get_spy_benchmark(df))
        n_trials = max(1, min(50, n_trials))  # cap: already parsed from Form
        validator = QuantProofValidator(
            df,
            benchmark_returns=_bm,
            benchmark_type=_bm_type,
            n_trials=n_trials,
            enable_prop_firm=payload.enhanced
        )
        report    = validator.run(enhanced=payload.enhanced)
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored        = [c for c in report.checks if c.category != "Plausibility"]
    top_issues    = [c.fix  for c in sorted(scored, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]]

    date_range = "No date data"
    try:
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    except Exception:
        pass

    check_count = len(scored)
    summary = (
        f"QuantProof analyzed {report.total_trades} IBKR-verified trades across "
        f"{check_count} institutional checks + 5 crash simulations. "
        f"Source: Interactive Brokers Flex API — data integrity verified. "
        f"Fundable Score: {report.score}/100 ({report.grade.split(chr(8212))[0].strip()})."
    )

    extra_flags = ([regime_note] if regime_note else []) + \
                  ([_bm_note] if _bm_note else []) + \
                  ["SOURCE: Interactive Brokers Flex API — trade data verified",
                   "VERIFIED: Flex Web Service — read-only, credentials never stored"]

    _vr_cache = {
        "fundable_score": min(100.0, round(report.score, 1)),
        "grade":          report.grade,
        "sharpe":         round(report.sharpe, 2),
        "max_drawdown":   round(report.max_drawdown, 4),
        "win_rate":       round(report.win_rate, 1),
        "total_trades":   report.total_trades,
        "date_range":     date_range,
        "is_verified":    True,
        "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                    "value":c.value,"insight":c.insight,"fix":c.fix,
                    "category":c.category} for c in report.checks],
        "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                        "description":s.description,"market_drop":s.market_drop,
                        "strategy_drop":s.strategy_drop,"survived":s.survived,
                        "emotional_verdict":s.emotional_verdict} for s in report.crash_sims],
        "top_issues":    top_issues,
        "top_strengths": top_strengths,
        "audit_flags":   report.audit_flags + extra_flags + audit_flags_ibkr,
        "equity_curve": getattr(report, 'equity_curve', None),
    }
    
    # Add prop firm compliance to VR cache only when enhanced validation is triggered
    if payload.enhanced and getattr(report, 'prop_firm_compliance', None):
        _vr_cache["prop_firm_compliance"] = getattr(report, 'prop_firm_compliance', None)
    
    _cache_validation(report.validation_hash + "-IBKR", _vr_cache)
    _save_validation_cache()

    validation_resp = ValidationResponse(
        fundable_score=min(100.0, round(report.score, 1)),
        grade=report.grade,
        summary=summary,
        checks=[CheckResultResponse(**vars(c)) for c in report.checks],
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
        validation_hash=report.validation_hash + "-IBKR",
        validation_date=report.validation_date,
        engine_version=report.engine_version,
        audit_flags=report.audit_flags + extra_flags + audit_flags_ibkr,
        plausibility_summary=report.plausibility_summary,
        scale_warning=getattr(report, '_scale_warning', False),
        currency=acct_currency,
        assumptions=report.assumptions + [
            "Trade data sourced from IBKR Flex Web Service",
            "Only closing trades included — opening legs excluded (no realised P&L)",
            "P&L uses fifoPnlRealized when available, netCash/cost-basis as fallback",
        ],
        dashboard=None, ai_plan=None,
        equity_curve=getattr(report, 'equity_curve', None),
        prop_firm_compliance=getattr(report, 'prop_firm_compliance', None),
    )

    return IBKRConnectResponse(
        source="IBKR_FLEX", account_id=payload.account_id,
        trades_pulled=len(trades), date_range=date_range,
        validation=validation_resp,
    )


# =========================================================
# KITE TRADEBOOK VALIDATION ENDPOINT
# =========================================================

@router.post("/validate/kite", response_model=ValidationResponse)
async def validate_kite(
    file: UploadFile = File(...),
    currency: str = Form("INR"),
    n_trials: int = Form(10),
    enhanced: bool = Form(False),
    registered_ip: str = Form(""),
    sebi_algo_id: str = Form("unsure"),
    sebi_box_type: str = Form("white"),
    request: Request = None
):
    """
    Validate Zerodha Kite TradeBook CSV.
    Also accepts Kite Orders CSV — smart_parse_csv() handles routing.
    """
    if request:
        # Prefer X-Forwarded-For (Render proxy passes real user IP here)
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "")) or "unknown"
        allowed, remaining = check_rate_limit(ip)
        if not allowed:
            hours_left = remaining // 3600
            raise HTTPException(429, {
                "error": "rate_limit",
                "message": f"Daily limit of {FREE_DAILY_LIMIT} validations reached. Resets in ~{hours_left}h.",
                "hours_left": hours_left
            })

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")

    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(400, f"File too large ({len(contents)//1024}KB). Max 5MB — a real trade CSV is never larger.")
    raw      = contents.decode("utf-8", errors="ignore")

    try:
        df_probe = pd.read_csv(io.StringIO(raw), nrows=5)
        is_tradebook = _is_kite_csv(df_probe)
        is_orders    = _is_kite_orders_csv(df_probe)
        if not is_tradebook and not is_orders:
            raise HTTPException(
                400,
                "Invalid Kite CSV format. "
                "Please export from Zerodha Kite → Console → TradeBook or Orders → Download."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {str(e)}")

    if _is_kite_csv(df_probe):
        try:
            df_fo_check = pd.read_csv(io.StringIO(raw))
            df_fo_check.columns = [c.lower().strip() for c in df_fo_check.columns]
            if "segment" in df_fo_check.columns:
                segments = df_fo_check["segment"].str.upper().str.strip()
                fo_count  = (segments.isin(["FO", "F&O", "OPTIONS", "FUTURES"])).sum()
                total     = len(df_fo_check)
                fo_pct    = fo_count / max(total, 1)
                if fo_pct > 0.30:
                    raise HTTPException(
                        400,
                        f"⚠️ {fo_count} of {total} trades ({fo_pct*100:.0f}%) are F&O segment. "
                        "QuantProof calculates P&L as (exit_price - entry_price) / entry_price, "
                        "which is not accurate for options (premiums, lot sizes, and expiry affect real P&L). "
                        "For options analysis, export only your equity trades, or manually calculate "
                        "per-trade P&L as a percentage and upload as a generic CSV."
                    )
        except HTTPException:
            raise
        except Exception:
            pass

    try:
        if _is_kite_orders_csv(df_probe):
            df, kite_note, kite_flags = _parse_kite_orders_csv(raw)
        else:
            df, kite_note, kite_flags = _parse_kite_csv(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    df = _dedup_trades(df)
    n_trials = max(1, min(50, n_trials))  # cap: prevent runaway requests
    if len(df) < 10:
        raise HTTPException(400, {
            "error": "insufficient_trades",
            "trades_found": len(df),
            "trades_needed": 10,
            "message": (
                f"Only {len(df)} completed trades found — need 10. "
                "A completed trade requires both a BUY and a matching SELL of the same stock within the export. "
                "Export a longer date range from Zerodha Console to include more closed positions."
            )
        })

    regime_note = ""
    if "date" in df.columns and df["date"].notna().any():
        df, regime_note = assign_regimes(df, "nifty")
        kite_note += " | " + regime_note

    try:
        _bm_type = get_benchmark_type(df, currency)
        _bm, _bm_note = (get_nifty_benchmark(df) 
            if _bm_type == "nifty" 
            else get_spy_benchmark(df))
        n_trials = max(1, min(50, n_trials))  # cap: already parsed from Form
        validator = QuantProofValidator(
            df,
            benchmark_returns=_bm,
            benchmark_type=_bm_type,
            n_trials=n_trials,
            enable_prop_firm=enhanced
        )
        report    = validator.run(enhanced=enhanced)
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored        = [c for c in report.checks if c.category != "Plausibility"]
    top_issues    = [c.fix  for c in sorted(scored, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]]

    date_range = "No date data"
    if "date" in df.columns and df["date"].notna().any():
        try:
            dates = pd.to_datetime(df["date"]).dropna()
            date_range = f"{dates.min().date()} to {dates.max().date()}"
        except Exception:
            pass

    visible_checks = [c for c in report.checks if c.category != "Plausibility"]

    try:
        dashboard_obj  = ValidationDashboard(validator, report)
        dashboard_data = dashboard_obj.generate_interactive_report()
    except Exception:
        dashboard_data = None

    detected_currency = (currency or "INR").upper()

    # Cache validation so /report/generate and /validate/ai-fixes can access it
    _sebi_ch = _compute_sebi_checks(df, request, registered_ip)
    _vr_cache = {
        "fundable_score": min(100.0, round(report.score, 1)),
        "grade": report.grade,
        "sharpe": round(report.sharpe, 2),
        "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1),
        "total_trades": report.total_trades,
        "date_range": date_range,
        "is_verified": False,
        "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                    "value":c.value,"insight":c.insight,"fix":c.fix,
                    "category":c.category} for c in report.checks],
        "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                        "description":s.description,"market_drop":s.market_drop,
                        "strategy_drop":s.strategy_drop,"survived":s.survived,
                        "emotional_verdict":s.emotional_verdict} for s in report.crash_sims],
        "top_issues":    [c.fix  for c in sorted(visible_checks, key=lambda x: x.score)[:3]],
        "top_strengths": [c.name for c in sorted(visible_checks, key=lambda x: x.score, reverse=True)[:3]],
        "audit_flags":   report.audit_flags + [kite_note] + kite_flags + ([_bm_note] if _bm_note else []),
        "equity_curve":  getattr(report, 'equity_curve', None),
        "sebi_checks":   _sebi_ch,
        "sebi_declaration": {"algo_id": sebi_algo_id, "box_type": sebi_box_type},
    }
    
    # Add prop firm compliance to VR cache only when enhanced validation is triggered
    if enhanced and getattr(report, 'prop_firm_compliance', None):
        _vr_cache["prop_firm_compliance"] = getattr(report, 'prop_firm_compliance', None)
    
    _cache_validation(report.validation_hash + "-KITE", _vr_cache)
    _save_validation_cache()

    return ValidationResponse(
        fundable_score=report.score,
        grade=report.grade,
        summary=(
            f"QuantProof analyzed {report.total_trades} Kite trades across "
            f"{len(scored)} institutional checks + 5 crash simulations. "
            f"Source: Zerodha Kite {'Orders' if _is_kite_orders_csv(df_probe) else 'TradeBook'} CSV — self-reported data. "
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
        validation_hash=report.validation_hash + "-KITE",
        validation_date=report.validation_date,
        engine_version=report.engine_version,
        audit_flags=report.audit_flags + [kite_note] + kite_flags + ([_bm_note] if _bm_note else []),
        plausibility_summary=report.plausibility_summary,
        scale_warning=getattr(report, '_scale_warning', False),
        currency=detected_currency,
        assumptions=report.assumptions + [
            "Trade data sourced from Zerodha Kite CSV — self-reported",
            "Trades paired using FIFO lot matching (BUY→SELL per symbol)",
        ],
        dashboard=dashboard_data,
        ai_plan=None,
        equity_curve=getattr(report, 'equity_curve', None),
        sebi_checks=_sebi_ch,
        sebi_declaration={"algo_id": sebi_algo_id, "box_type": sebi_box_type},
        prop_firm_compliance=getattr(report, 'prop_firm_compliance', None),
    )


