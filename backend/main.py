"""
QuantProof — FastAPI Backend v1.4
Synced to validator v6.2

CHANGES FROM v1.3:  
  - Version bumped to 2.6.0
  - Root endpoint: 38 → 44 checks
  - All summary strings: '3 crash simulations' → '5 crash simulations'
  - smart_parse_csv(): auto-assigns regime via yfinance SPY SMA when dates
    present but no regime column (fallback: SPY monthly lookup table)
  - /validate/alpaca + /report/generate-broker: replaced broken buy/sell
    stack pairing with position-state tracker (sort by filled_at, FIFO lots,
    orphan sell detection)
  - All Alpaca + IBKR paths now auto-assign regime after trade pairing
  - benchmark_returns now inferred from SPY when date column present.
    IR + Beta checks are scored (not informational) for dated strategies.
    Falls back to None silently — always backward-compatible.

PARSER FIXES (v1.4.1):
  - Added Kite Orders CSV support (_is_kite_orders_csv, _parse_kite_orders_csv)
    Handles: transaction_type, order_execution_time, filled_quantity, average_price
  - Added Alpaca UI export support (_is_alpaca_ui_export, _parse_alpaca_ui_export)
    Handles: tab-separated UI table with human-readable dates like "Feb 24, 2026, 08:40:53 PM"
  - smart_parse_csv() now detects both new formats before falling through to generic parse
  - pair_alpaca_orders() hardened to handle both REST API JSON and UI export dict shapes
    (normalises filled_qty/avg_fill_price/filled_at column names from both sources)

PATCH FIXES:
  - Removed dead record_validation calls (analytics.py not present in this version)
  - Normalized PnL clipping to ?0.99 across all parsers to match validator guard
  - Fixed misleading orphan sell audit message in pair_alpaca_orders
  - Moved validator.run() off event loop in /validate/alpaca (run_in_executor)
  - Fixed _get_cached_validation to use LRU eviction instead of FIFO
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import os
import re as _re
import sys
import time
import hashlib
import traceback
import uuid
import asyncio
from collections import defaultdict

from validator import QuantProofValidator, ValidationDashboard
from config import API_VERSION, extract_n_trials
from schemas import CheckResultResponse, CrashSimResponse, ValidationResponse
from services.ai_plans import generate_ai_plan, generate_portfolio_fix_plan
from services.benchmarks import assign_regimes, get_benchmark_type, get_nifty_benchmark, get_spy_benchmark
from services.cache import VALIDATION_RESULT_CACHE, _PDF_SEMAPHORE, _cache_validation, _get_cached_validation, _save_validation_cache
from services.ibkr import parse_ibkr_trade_improved
from services.parsers import _dedup_trades, _is_kite_csv, _is_kite_orders_csv, _parse_kite_csv, _parse_kite_orders_csv, pair_alpaca_orders, retry_on_failure, smart_parse_csv
from services.portfolio import PortfolioResponse, _diversification_score, _drawdown_coincidence, _portfolio_correlation, _portfolio_verdict, _risk_parity_weights
from services.sebi import _compute_sebi_checks
from services.tokens import FREE_DAILY_LIMIT, REPORT_ACCESS_TOKENS, TOKEN_LITE, TOKEN_PORTFOLIO, TOKEN_PRO, TOKEN_REPORT, _PRO_TOKENS, _is_pro_token, _save_pro_tokens, _save_tokens, _token_tier, _validation_counts, check_rate_limit

# Import prop firm compliance module
try:
    from prop_firm_compliance import PropFirmComplianceEngine, PropFirmRules, QuantProofValidatorWithPropFirms
    print(f"✅ main.py PropFirmComplianceEngine import successful")
except ImportError as e:
    # Fallback if prop_firm_compliance.py doesn't exist yet
    print(f"❌ main.py PropFirmComplianceEngine ImportError: {e}")
    PropFirmComplianceEngine = None
    PropFirmRules = None
    QuantProofValidatorWithPropFirms = None
except Exception as e:
    # Handle any other import errors
    print(f"❌ main.py PropFirmComplianceEngine Exception: {e}")
    import traceback
    traceback.print_exc()
    PropFirmComplianceEngine = None
    PropFirmRules = None
    QuantProofValidatorWithPropFirms = None

# ================= HELPER FUNCTIONS =================




app = FastAPI(title="QuantProof API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/validate", response_model=ValidationResponse)
async def validate(file: UploadFile = File(...), currency: str = Form("USD"), n_trials: int = Form(10), enhanced: bool = Form(False), request: Request = None):
    if request:
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "")) or "unknown"
        # Accept dev token from either header
        _pro_tok = (request.headers.get("x-pro-token", "")
                    or request.headers.get("x-report-token", "")
                    or request.headers.get("x-lite-token", ""))
        allowed, remaining = check_rate_limit(ip, _pro_tok)
        if not allowed:
            hours_left = remaining // 3600
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limit",
                        "message": f"Free limit of {FREE_DAILY_LIMIT} validations/day reached. Resets in ~{hours_left}h.",
                        "hours_left": hours_left}
            )

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(400, f"File too large ({len(contents)//1024}KB). Max 5MB — a real trade CSV is never larger.")

    try:
        df, regime_note = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = _dedup_trades(df)
    n_trials = max(1, min(50, n_trials))  # cap: prevent runaway requests
    if len(df) < 10:
        raise HTTPException(status_code=400, detail=f"Only found {len(df)} valid trades. Need at least 10.")

    try:
        _bm_type = get_benchmark_type(df, currency)
        _bm, _bm_note = (get_nifty_benchmark(df) 
            if _bm_type == "nifty" 
            else get_spy_benchmark(df))
        # n_trials already parsed from Form(10) above — do NOT call extract_n_trials
        # which reads raw request body (already consumed) and returns 1
        validator = QuantProofValidator(
            df,
            benchmark_returns=_bm,
            benchmark_type=_bm_type,
            n_trials=n_trials,
            enable_prop_firm=enhanced
        )
        report = validator.run(enhanced=enhanced)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

    r = report
    scored_checks = [c for c in r.checks if c.category != "Plausibility"]
    top_issues    = [c.fix  for c in sorted(scored_checks, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored_checks, key=lambda x: x.score, reverse=True)[:3]]

    date_range = "No date data"
    if "date" in validator.df.columns:
        try:
            date_range = f"{validator.df['date'].min().date()} to {validator.df['date'].max().date()}"
        except Exception:
            pass

    check_count = len(scored_checks)
    summary = (
        f"QuantProof analyzed {r.total_trades} trades across {check_count} institutional checks "
        f"+ 5 crash simulations. Fundable Score: {r.score}/100 ({r.grade.split('?')[0].strip()}). "
        f"{'Core risk management needs work ?? see the action plan below.' if r.score < 60 else 'Edge shows statistical promise ?? review execution costs and regime coverage.'}"
    )

    detected_currency = (currency or "USD").upper()
    if detected_currency == "USD" and "currency" in df.columns:
        currencies = df["currency"].dropna().unique()
        if len(currencies) == 1:
            detected_currency = str(currencies[0]).upper()
        elif len(currencies) > 1:
            detected_currency = "MULTI"

    try:
        dashboard_obj  = ValidationDashboard(validator, report)
        dashboard_data = dashboard_obj.generate_interactive_report()
    except Exception:
        dashboard_data = None

    visible_checks = [c for c in r.checks if c.category != "Plausibility"]

    existing_flags_text = " ".join(r.audit_flags)
    _options_flags = []
    if "OPTIONS_FLAGS:" in regime_note:
        _parts = regime_note.split("OPTIONS_FLAGS:", 1)
        regime_note = _parts[0].rstrip(" |")
        _options_flags = _parts[1].split("||") if len(_parts) > 1 else []

    _mt5_flags = []
    _mt5_scale_warn = False
    if "MT5_FLAGS:" in regime_note:
        _parts = regime_note.split("MT5_FLAGS:", 1)
        regime_note = _parts[0].rstrip(" |")
        _mt5_flags = _parts[1].split("||") if len(_parts) > 1 else []
    if "MT5_SCALE_WARNING" in regime_note:
        regime_note = regime_note.replace(" | MT5_SCALE_WARNING", "").replace("MT5_SCALE_WARNING", "").strip(" |")
        _mt5_scale_warn = True

    # Put SPY failure note FIRST so it appears at top of audit trail, not buried
    _spy_flags_first = [_bm_note] if (_bm_note and "unavailable" in _bm_note) else []
    _spy_flags_rest  = [_bm_note] if (_bm_note and "unavailable" not in _bm_note) else []
    extra_flags = _spy_flags_first + ([regime_note]  if (regime_note and regime_note not in existing_flags_text) else []) + _spy_flags_rest + _options_flags + _mt5_flags

    _vr_cache = {
        "fundable_score": min(100.0, round(r.score, 1)),
        "grade":          r.grade,
        "sharpe":         round(r.sharpe, 2),
        "max_drawdown":   round(r.max_drawdown, 4),
        "win_rate":       round(r.win_rate, 1),
        "total_trades":   r.total_trades,
        "date_range":     date_range,
        "is_verified":    False,
        "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                    "value":c.value,"insight":c.insight,"fix":c.fix,
                    "category":c.category} for c in r.checks],
        "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                        "description":s.description,"market_drop":s.market_drop,
                        "strategy_drop":s.strategy_drop,"survived":s.survived,
                        "emotional_verdict":s.emotional_verdict} for s in r.crash_sims],
        "top_issues":    top_issues,
        "top_strengths": top_strengths,
        "audit_flags":   r.audit_flags + extra_flags,
        "equity_curve":  getattr(r, 'equity_curve', None),
        "monte_carlo": getattr(r, 'monte_carlo', None),
    }
    
    # Add prop firm compliance to VR cache only when enhanced validation is triggered
    if enhanced and getattr(r, 'prop_firm_compliance', None):
        _vr_cache["prop_firm_compliance"] = getattr(r, 'prop_firm_compliance', None)
    
    _cache_validation(r.validation_hash, _vr_cache)
    _save_validation_cache()

    return ValidationResponse(
        fundable_score=min(100.0, round(r.score, 1)),
        grade=r.grade,
        summary=summary,
        checks=[CheckResultResponse(**vars(c)) for c in visible_checks],
        crash_sims=[CrashSimResponse(
            crash_name=s.crash_name, year=s.year,
            description=s.description,
            market_drop=s.market_drop,
            strategy_drop=s.strategy_drop,
            survived=s.survived,
            emotional_verdict=s.emotional_verdict
        ) for s in r.crash_sims],
        total_trades=r.total_trades,
        profitable_trades=r.profitable_trades,
        date_range=date_range,
        sharpe=round(r.sharpe, 2),
        max_drawdown=round(r.max_drawdown, 4),
        win_rate=round(r.win_rate, 1),
        top_issues=top_issues,
        top_strengths=top_strengths,
        validation_hash=r.validation_hash,
        validation_date=r.validation_date,
        engine_version=r.engine_version,
        audit_flags=r.audit_flags + extra_flags,
        plausibility_summary=r.plausibility_summary,
        scale_warning=getattr(r, '_scale_warning', False) or _mt5_scale_warn,
        currency=detected_currency,
        assumptions=r.assumptions,
        dashboard=dashboard_data,
        ai_plan=None,
        equity_curve=getattr(r, 'equity_curve', None),
        monte_carlo=getattr(r, 'monte_carlo', None),
        prop_firm_compliance=getattr(r, 'prop_firm_compliance', None),
    )

@app.post("/validate/ai-fixes")
async def get_ai_fixes(
    validation_hash: str = Form(...),
    x_lite_token: str = Header(default=""),
    request: Request = None
):
    """
    ₹99 endpoint — returns top 3 AI fixes as JSON for display in browser.
    Requires LITE or higher token.
    """
    # _is_pro_token now handles TEST_REPORT_TOKEN via FIX 2
    if _is_pro_token(x_lite_token):
        tier = "pro"
    else:
        tier = _token_tier(x_lite_token)
    if tier == "none":
        raise HTTPException(402, {
            "error": "payment_required",
            "message": "Unlock required to access AI fix plan.",
            "price_inr": 99,
        })

    cached = _get_cached_validation(validation_hash)
    if not cached:
        raise HTTPException(400, {
            "error": "cache_expired",
            "message": "Validation result not found. Re-run your validation first."
        })

    # If AI plan already cached, return top 3
    ai_plan = cached.get("ai_plan")
    if not ai_plan:
        # Generate it now
        try:
            class _CR:
                def __init__(self, d): self.__dict__.update(d)
            class _Rep:
                def __init__(self, vr):
                    self.score = vr["fundable_score"]; self.grade = vr["grade"]
                    self.sharpe = vr["sharpe"]; self.max_drawdown = vr["max_drawdown"]
                    self.win_rate = vr["win_rate"]; self.total_trades = vr["total_trades"]
                    self.checks = [_CR(c) for c in vr.get("checks", [])]
                    self.audit_flags = []
                    self.prop_firm_compliance = vr.get("prop_firm_compliance")
            ai_plan = await generate_ai_plan(_Rep(cached), cached.get("date_range", ""))
            if ai_plan:
                cached["ai_plan"] = ai_plan
                _cache_validation(validation_hash, cached)
        except Exception as e:
            raise HTTPException(500, f"AI plan generation failed: {str(e)}")

    if not ai_plan:
        return {"fixes": [], "summary": "No AI plan available — re-run validation."}

    fixes = ai_plan.get("fixes", [])
    all_fixes = sorted(fixes, key=lambda x: x.get("rank", 99))
    # Lite = top 3 in browser + teaser for full PDF
    # Report/Pro = all fixes, no teaser
    if tier in ("report", "pro"):
        shown_fixes = all_fixes
    else:
        shown_fixes = all_fixes[:3]

    # All tiers get ALL failed check details (insight + fix) — that's what they paid for.
    # The difference: ₹99 gets AI top-3 summary, ₹299 gets full AI ranked plan in PDF.
    all_checks  = cached.get("checks", [])
    failed_checks = [
        {
            "name":     c.get("name", "") if isinstance(c, dict) else getattr(c, "name", ""),
            "category": c.get("category", "") if isinstance(c, dict) else getattr(c, "category", ""),
            "score":    c.get("score", 0) if isinstance(c, dict) else getattr(c, "score", 0),
            "insight":  c.get("insight", "") if isinstance(c, dict) else getattr(c, "insight", ""),
            "fix":      c.get("fix", "") if isinstance(c, dict) else getattr(c, "fix", ""),
            "value":    c.get("value", "") if isinstance(c, dict) else getattr(c, "value", ""),
        }
        for c in all_checks
        if not (c.get("passed") if isinstance(c, dict) else getattr(c, "passed", True))
        and (c.get("category") if isinstance(c, dict) else getattr(c, "category", "")) != "Plausibility"
    ]
    # Sort by score ascending (worst first)
    failed_checks.sort(key=lambda x: x.get("score", 0))

    return {
        "summary":          ai_plan.get("summary", ""),
        "fixes":            shown_fixes,        # AI fixes: top-3 for ₹99, all for ₹299+
        "failed_checks":    failed_checks,       # ALL failed checks with full insight+fix
        "encouragement":    ai_plan.get("encouragement", "") if tier in ("report", "pro") else "",
        "tier":             tier,
        "total_failures":   len(all_fixes),
        "full_plan_in_pdf": tier not in ("report", "pro"),
    }


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

@app.post("/validate/alpaca", response_model=ValidationResponse)
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
# PDF REPORT — PAYWALL ENDPOINT
# =========================================================

from fastapi.responses import Response as FastAPIResponse
try:
    from generate_report import generate_pdf_report as _generate_pdf_report
    _PDF_AVAILABLE = True
except ImportError:
    _generate_pdf_report = None
    _PDF_AVAILABLE = False



_TEST_TOKEN = os.getenv("TEST_REPORT_TOKEN", "")
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
REPORT_PRICE_USD    = 4
REPORT_PRICE_INR    = 299


@app.post("/report/generate")
async def generate_report_endpoint(
    file: UploadFile = File(...),
    currency: str = Form("USD"),
    enhanced: bool = Form(False),
    registered_ip: str = Form(""),
    sebi_algo_id: str = Form("unsure"),
    sebi_box_type: str = Form("white"),
    x_report_token: str = Header(default=""),
    x_report_source: str = Header(default=""),
    request: Request = None
):
    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok: return True
        if _is_pro_token(token): return True          # Pro = unlimited PDFs
        if token in REPORT_ACCESS_TOKENS: return True
        if token.startswith("cs_"): return False
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(
            status_code=402,
            detail={"error": "payment_required",
                    "message": "Purchase required to download the full analysis report.",
                    "price_inr": REPORT_PRICE_INR,
                    "upgrade_url": "https://quantproof-frontend.onrender.com/#upgrade"}
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")

    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(400, f"File too large ({len(contents)//1024}KB). Max 5MB — a real trade CSV is never larger.")
    try:
        df, regime_note = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(df) < 10:
        raise HTTPException(400, f"Only {len(df)} trades found. Need at least 10.")

    try:
        _bm_type = get_benchmark_type(df, currency)
        _bm, _bm_note = (get_nifty_benchmark(df) 
            if _bm_type == "nifty" 
            else get_spy_benchmark(df))
        n_trials = extract_n_trials(request) if request else 1
        # PDF tier: force minimum n=20 for RPT and PRO tokens
        _tier = _token_tier(x_report_token)
        if _tier in ("report", "pro"):
            n_trials = max(20, min(50, n_trials))
        n_trials = max(1, min(50, n_trials))
        validator = QuantProofValidator(
            df,
            benchmark_returns=_bm,
            benchmark_type=_bm_type,
            n_trials=n_trials,
            enable_prop_firm=enhanced
        )
        report = validator.run(enhanced=enhanced)
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored = [c for c in report.checks if c.category != "Plausibility"]
    date_range = "No date data"
    if "date" in validator.df.columns:
        try:
            date_range = f"{validator.df['date'].min().date()} to {validator.df['date'].max().date()}"
        except Exception:
            pass

    PROP_FIRM_CHECK_NAMES = {
        "Prop Firm Compliance (FTMO/Topstep/The5ers)",
        "Institutional Compliance (CTA/Fund Standard)",
        "Prop Firm Compliance",
    }
    def _csv_check_filter(c):
        if c.name in PROP_FIRM_CHECK_NAMES:
            return {
                "name": c.name, "passed": False, "score": 0,
                "value": "⚠️ NOT APPLICABLE — CSV Upload",
                "insight": (
                    "Prop firm and institutional compliance certificates require "
                    "broker-verified trade data. CSV uploads cannot be used for "
                    "compliance submissions as the data is self-reported and unverified."
                ),
                "fix": "Connect your Alpaca account (or IBKR when available) to generate a verified compliance report.",
                "category": c.category,
            }
        return {"name": c.name, "passed": c.passed, "score": c.score,
                "value": c.value, "insight": c.insight, "fix": c.fix,
                "category": c.category}

    vr = {
        "fundable_score": min(100.0, round(report.score, 1)), "grade": report.grade,
        "sharpe": round(report.sharpe, 2), "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1), "total_trades": report.total_trades,
        "date_range": date_range,
        "n_trials": n_trials,
        "is_verified": x_report_source in ("alpaca", "ibkr"),
        "checks": [
            (_csv_check_filter(c) if x_report_source == "csv" else
             {"name": c.name, "passed": c.passed, "score": c.score,
              "value": c.value, "insight": c.insight, "fix": c.fix,
              "category": c.category})
            for c in report.checks
        ],
        "crash_sims": [
            {"crash_name": s.crash_name, "year": s.year, "description": s.description,
             "market_drop":s.market_drop, "strategy_drop": s.strategy_drop,
             "survived": s.survived, "emotional_verdict": s.emotional_verdict}
            for s in report.crash_sims
        ],
        "top_issues":    [c.fix  for c in sorted(scored, key=lambda x: x.score)[:4]],
        "top_strengths": [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]],
        "regime_note":   regime_note,
        "audit_flags":   report.audit_flags + ([_bm_note] if _bm_note else []) + (
                         [regime_note] if regime_note else []),
        "plausibility_summary": getattr(report, "plausibility_summary", None),
        "equity_curve": getattr(report, 'equity_curve', None),
        "monte_carlo": getattr(report, 'monte_carlo', None),
        "prop_firm_compliance": getattr(report, 'prop_firm_compliance', None),
        # Compute SEBI checks for all data (function handles non-Indian data gracefully)
        "sebi_checks": _compute_sebi_checks(df, request, registered_ip),
        "sebi_declaration": {"algo_id": sebi_algo_id, "box_type": sebi_box_type},
    }

    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan
    _cache_validation(report.validation_hash, vr)
    _save_validation_cache()

    try:
        if not _PDF_AVAILABLE:
            raise HTTPException(503, "PDF module not available.")
        async with _PDF_SEMAPHORE:
            pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        # Log full traceback for debugging
        import traceback
        tb = traceback.format_exc()
        print(f"[PDF_GENERATE_ERROR] {tb}", file=sys.stderr, flush=True)
        raise HTTPException(400, f"PDF generation failed: {str(e)}")

    score     = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    filename  = f"QuantProof_Report_{grade_str}_{score}.pdf"
    return FastAPIResponse(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/report/create-checkout")
async def create_checkout(payload: dict):
    import hmac, hashlib
    try:
        import razorpay
    except ImportError:
        raise HTTPException(503, {"error": "razorpay_not_installed",
            "message": "razorpay package not installed. Run: pip install razorpay"})
    key_id     = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(503, {"error": "razorpay_not_configured",
            "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment."})
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order  = client.order.create({
            "amount": 29900, "currency": "INR", "payment_capture": 1,
            "notes": {"product": "QuantProof Full Report"}
        })
        return {"order_id": order["id"], "amount": order["amount"],
                "currency": order["currency"], "key_id": key_id, "checkout_url": None}
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})



@app.post("/report/webhook")
async def payment_webhook(request: Request):
    import hashlib, hmac, time, json
    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id   = data.get("razorpay_order_id", "")
    signature  = data.get("razorpay_signature", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    token_now  = hashlib.sha256(f"report:{current_hour}:{key_secret}:{payment_id}".encode()).hexdigest()[:32]
    token_next = hashlib.sha256(f"report:{current_hour+1}:{key_secret}:{payment_id}".encode()).hexdigest()[:32]
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    _save_tokens()
    return {"status": "ok", "token": token_now}


@app.post("/report/create-checkout-lite")
async def create_checkout_lite(payload: dict):
    import hmac, hashlib
    try:
        import razorpay
    except ImportError:
        raise HTTPException(503, {"error": "razorpay_not_installed",
            "message": "razorpay package not installed. Run: pip install razorpay"})
    key_id     = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(503, {"error": "razorpay_not_configured",
            "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment."})
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order  = client.order.create({
            "amount": 9900, "currency": "INR", "payment_capture": 1,
            "notes": {"product": "QuantProof Lite Unlock"}
        })
        return {"order_id": order["id"], "amount": order["amount"],
                "currency": order["currency"], "key_id": key_id, "checkout_url": None}
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@app.post("/report/webhook-lite")
async def payment_webhook_lite(request: Request):
    import hashlib, hmac, time, json, uuid
    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id   = data.get("razorpay_order_id", "")
    signature  = data.get("razorpay_signature", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    token_now  = f"LITE-{hashlib.sha256(f'lite:{current_hour}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    token_next = f"LITE-{hashlib.sha256(f'lite:{current_hour+1}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    _save_tokens()
    return {"status": "ok", "token": token_now}


# =========================================================
# PORTFOLIO PDF REPORT
# =========================================================

@app.post("/report/generate-portfolio")
async def generate_portfolio_report_endpoint(
    files: List[UploadFile] = File(...),
    currency: str = Form("USD"),
    strategy_names: str = Form(""),
    x_portfolio_token: str = Header(default=""),
    request: Request = None,
):
    """Portfolio PDF — requires PORT or PRO token."""
    if not (_is_pro_token(x_portfolio_token) or
            _token_tier(x_portfolio_token) in ("portfolio", "pro")):
        raise HTTPException(402, {"error": "payment_required",
            "message": "Portfolio PDF requires ₹199 Portfolio unlock or Pro plan.",
            "price_inr": 199})

    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available.")
    if len(files) < 2:
        raise HTTPException(400, "Need at least 2 strategy CSV files.")
    if len(files) > 3:
        raise HTTPException(400, "Maximum 3 strategies.")

    names_list = [n.strip() for n in strategy_names.split(",") if n.strip()]
    while len(names_list) < len(files):
        names_list.append(f"Strategy {len(names_list)+1}")

    audit_flags = []
    strategy_summaries = []
    dfs = []
    sharpes = []
    _tier = _token_tier(x_portfolio_token)
    _n    = 20 if _tier == "pro" else 10

    for i, file in enumerate(files):
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(400, f"File {i+1} must be a CSV.")
        contents = await file.read()
        if len(contents) > 5_000_000:
            raise HTTPException(400, f"File too large ({len(contents)//1024}KB). Max 5MB — a real trade CSV is never larger.")
        try:
            df, regime_note = smart_parse_csv(contents)
        except (ValueError, HTTPException) as e:
            raise HTTPException(400, f"Strategy {i+1}: {str(e)}")
        df = _dedup_trades(df)
        if len(df) < 10:
            raise HTTPException(400, f"Strategy {i+1} has only {len(df)} trades.")
        try:
            _bm_type = get_benchmark_type(df, currency)
            _bm, _bm_note = (get_nifty_benchmark(df) if _bm_type=="nifty"
                             else get_spy_benchmark(df))
            validator = QuantProofValidator(df, benchmark_returns=_bm,
                                            benchmark_type=_bm_type, n_trials=_n)
            report = validator.run()
        except Exception as e:
            raise HTTPException(500, f"Validation failed for strategy {i+1}: {str(e)}")

        date_range = "No date data"
        if "date" in df.columns and df["date"].notna().any():
            try:
                date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
            except Exception: pass
        else:
            audit_flags.append(f"⚠ {names_list[i]}: no date column — correlation positional.")

        _cl = [{"name":c.name,"passed":c.passed,"score":round(float(c.score),1),
                "category":c.category,"insight":c.insight or "","fix":c.fix or "",
                "value":str(c.value or "")} for c in report.checks if c.category!="Plausibility"]
        _fail = [c for c in _cl if not c["passed"]]
        _pass = [c for c in _cl if c["passed"]]
        _cs   = getattr(report,"crash_sims",[])
        _surv = int(sum(1 for c in _cs if (c.get("survived") if isinstance(c,dict) else getattr(c,"survived",False))))
        _eq   = None
        try:
            _eq = [round(float(v),4) for v in getattr(report,"equity_curve",None) or []]
            if len(_eq)<3: _eq=None
        except Exception: pass

        strategy_summaries.append({
            "name":str(names_list[i]),"fundable_score":float(round(report.score,1)),
            "grade":report.grade.split("—")[0].strip(),
            "sharpe":float(round(report.sharpe,2)),"max_drawdown":float(round(report.max_drawdown,4)),
            "win_rate":float(round(report.win_rate,1)),"total_trades":int(report.total_trades),
            "date_range":str(date_range),"checks_passed":int(len(_pass)),
            "checks_failed":int(len(_fail)),"checks_total":int(len(_cl)),
            "crash_survived":_surv,"crash_total":int(len(_cs)),
            "top_issues":[str(f"{c['name']}: {c['insight']}") for c in _fail[:3] if c.get("insight")],
            "top_strengths":[str(c["name"]) for c in _pass if c.get("score",0)>=80][:3],
            "equity_curve":_eq,
            "checks":[{"name":str(c.get("name","")),"passed":bool(c.get("passed",False)),
                       "score":float(c.get("score",0)),"category":str(c.get("category","")),
                       "insight":str(c.get("insight","")),"fix":str(c.get("fix","")),
                       "value":str(c.get("value",""))} for c in _cl[:44]],
        })
        dfs.append(df); sharpes.append(report.sharpe)
        if regime_note: audit_flags.append(f"{names_list[i]}: {regime_note}")

    corr_mat        = _portfolio_correlation(dfs)
    rp_weights      = _risk_parity_weights(dfs)
    n               = len(dfs)
    off_signed      = [float(corr_mat[i][j]) for i in range(n) for j in range(i+1,n)]
    avg_corr_signed = float(np.mean(off_signed)) if off_signed else 0.0
    avg_corr_abs    = float(np.mean([abs(v) for v in off_signed])) if off_signed else 0.0
    max_len = max(len(df) for df in dfs)
    weighted = np.zeros(max_len)
    for df,w in zip(dfs,rp_weights):
        p = df["pnl"].values.astype(float)
        weighted += np.pad(p,(0,max_len-len(p)),constant_values=0.0)*w
    cs_std = float(weighted.std())
    cs_mean = float(weighted.mean())
    _pdf_total_trades = sum(s["total_trades"] for s in strategy_summaries)
    _pdf_date_ranges  = [s["date_range"] for s in strategy_summaries if s.get("date_range","No date data") != "No date data"]
    try:
        if _pdf_date_ranges:
            _ps = min(pd.to_datetime(r.split(" to ")[0], errors="coerce") for r in _pdf_date_ranges)
            _pe = max(pd.to_datetime(r.split(" to ")[1], errors="coerce") for r in _pdf_date_ranges)
            _py = max((_pe - _ps).days / 365.25, 0.1)
            _paf = float(np.sqrt(_pdf_total_trades / _py))
        else:
            _paf = float(np.sqrt(max_len))
    except Exception:
        _paf = float(np.sqrt(252))
    cs = float(cs_mean / cs_std * _paf) if cs_std > 1e-10 else 0.0
    eq_c = np.cumprod(1+np.clip(weighted,-0.99,0.99))
    rm_c = np.maximum.accumulate(eq_c)
    cd   = float(((eq_c-rm_c)/np.where(rm_c>0,rm_c,1.0)).min())
    cw   = float((weighted>0).mean()*100)
    ddc  = _drawdown_coincidence(dfs)
    div  = float(_diversification_score(corr_mat,sharpes))
    ceq  = [float(round(v*100/eq_c[0],2)) for v in eq_c] if eq_c[0]>0 else []
    try:
        from scipy import stats as _scs; import math as _m
        _sr_std = float(np.sqrt((1+0.5*cs**2)/max(max_len-1,1)))
        _e_max  = ((1-_m.e**(-_m.pi/2))*_scs.norm.ppf(1-1.0/max(n,1))+
                   (_m.e**(-_m.pi/2))*_scs.norm.ppf(1-1.0/(max(n,1)*_m.e)))
        _dsr_z  = (cs-max(_e_max,0))/_sr_std if _sr_std>0 else 0
        pdsr    = float(round(min(float(_scs.norm.cdf(_dsr_z)*100),99.8),1))
    except Exception: pdsr = None

    verdict = _portfolio_verdict(div, avg_corr_signed, n)
    plan    = await generate_portfolio_fix_plan(
        strategies=strategy_summaries,
        corr_matrix=[[float(v) for v in row] for row in corr_mat.tolist()],
        rp_weights=rp_weights, div_score=div, dd_coincidence=ddc,
        combined_sharpe=cs, combined_dd=cd, avg_corr=avg_corr_signed,
        portfolio_dsr=pdsr, strategy_names=names_list,
    )

    best = max(strategy_summaries, key=lambda s: s["fundable_score"])
    vr = {
        "fundable_score": best["fundable_score"], "grade": best["grade"],
        "sharpe": float(round(cs,2)), "max_drawdown": float(round(cd,4)),
        "win_rate": float(round(cw,1)),
        "total_trades": int(sum(s["total_trades"] for s in strategy_summaries)),
        "date_range": "Portfolio — " + " / ".join(s["date_range"] for s in strategy_summaries),
        "n_trials": _n, "is_verified": False,
        "checks": best.get("checks",[]), "crash_sims": [],
        "top_issues": best.get("top_issues",[]),
        "top_strengths": best.get("top_strengths",[]),
        "audit_flags": audit_flags, "equity_curve": ceq[:200] or None, "ai_plan": None,
        "prop_firm_compliance": best.get("prop_firm_compliance", None),
        "portfolio": {
            "strategies": strategy_summaries,
            "correlation_matrix": [[float(v) for v in row] for row in corr_mat.tolist()],
            "strategy_names": names_list,
            "combined_sharpe": float(round(cs,2)), "combined_drawdown": float(round(cd,4)),
            "combined_win_rate": float(round(cw,1)), "diversification_score": float(div),
            "avg_correlation": float(round(avg_corr_signed,3)),
            "portfolio_verdict": verdict,
            "rp_weights": [float(round(w,3)) for w in rp_weights],
            "dd_coincidence": float(ddc), "portfolio_dsr": pdsr,
            "portfolio_plan": plan,
        },
    }

    try:
        async with _PDF_SEMAPHORE:
            pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        import traceback
        print(f"[PORTFOLIO_PDF_ERROR] {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(400, f"Portfolio PDF failed: {str(e)}")

    filename = f"QuantProof_Portfolio_{n}strategies.pdf"
    return FastAPIResponse(content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# =========================================================
# PORTFOLIO CHECKOUT — ₹199
# =========================================================

@app.post("/report/create-checkout-portfolio")
async def create_checkout_portfolio(payload: dict):
    import hmac, hashlib
    try:
        import razorpay
    except ImportError:
        raise HTTPException(503, {"error": "razorpay_not_installed",
            "message": "razorpay package not installed."})
    key_id     = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(503, {"error": "razorpay_not_configured",
            "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment."})
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order  = client.order.create({
            "amount": 19900, "currency": "INR", "payment_capture": 1,
            "notes": {"product": "QuantProof Portfolio Analysis"}
        })
        return {"order_id": order["id"], "amount": order["amount"],
                "currency": order["currency"], "key_id": key_id}
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@app.post("/report/webhook-portfolio")
async def payment_webhook_portfolio(request: Request):
    import hashlib, hmac, time, json
    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id   = data.get("razorpay_order_id", "")
    signature  = data.get("razorpay_signature", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    token_now  = f"PORT-{hashlib.sha256(f'port:{current_hour}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    token_next = f"PORT-{hashlib.sha256(f'port:{current_hour+1}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    _save_tokens()
    return {"status": "ok", "token": token_now}


# =========================================================
# PRO PLAN CHECKOUT — ₹999 / 30 days
# =========================================================

@app.post("/report/create-checkout-pro")
async def create_checkout_pro(payload: dict):
    import hmac, hashlib
    try:
        import razorpay
    except ImportError:
        raise HTTPException(503, {"error": "razorpay_not_installed",
            "message": "razorpay package not installed."})
    key_id     = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(503, {"error": "razorpay_not_configured",
            "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment."})
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order  = client.order.create({
            "amount": 99900, "currency": "INR", "payment_capture": 1,
            "notes": {"product": "QuantProof Pro — 30 days"}
        })
        return {"order_id": order["id"], "amount": order["amount"],
                "currency": order["currency"], "key_id": key_id}
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@app.post("/report/webhook-pro")
async def payment_webhook_pro(request: Request):
    import hashlib, hmac, time, json
    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id   = data.get("razorpay_order_id", "")
    signature  = data.get("razorpay_signature", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    # Pro token — 30-day expiry stored in _PRO_TOKENS
    expiry = time.time() + 30 * 24 * 3600
    token  = f"PRO-{hashlib.sha256(f'pro:{current_hour}:{key_secret}:{payment_id}'.encode()).hexdigest()[:36]}"
    _PRO_TOKENS[token] = expiry
    REPORT_ACCESS_TOKENS.add(token)
    _save_pro_tokens()
    _save_tokens()
    return {"status": "ok", "token": token, "expires_in_days": 30}


# =========================================================
# PADDLE INTERNATIONAL PAYMENTS
# =========================================================

@app.post("/report/create-checkout-paddle-lite")
async def create_checkout_paddle_lite(payload: dict):
    import requests as _requests
    api_key  = os.environ.get("PADDLE_API_KEY", "")
    price_id = os.environ.get("PADDLE_PRICE_LITE", "")
    if not api_key or not price_id:
        raise HTTPException(503, "Paddle not configured")
    res = _requests.post(
        "https://api.paddle.com/transactions",
        json={"items": [{"price_id": price_id, "quantity": 1}],
              "checkout": {"return_url": "https://quantproof.net/success"},
              "custom_data": {"user_id": "anonymous"}},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30
    )
    data = res.json()
    url = data.get("data", {}).get("checkout", {}).get("url", "")
    if not url:
        raise HTTPException(500, f"Paddle error: {data}")
    return {"checkout_url": url}

@app.post("/report/create-checkout-paddle-report")
async def create_checkout_paddle_report(request: dict):
    import requests
    import os

    print(f"[DEBUG] Paddle checkout called with request: {request}")
    
    PADDLE_API_KEY = os.getenv("PADDLE_API_KEY")
    price_id = os.getenv("PADDLE_PRICE_REPORT")
    user_id = request.get("user_id", "anonymous")

    print(f"[DEBUG] Paddle env vars - API Key: {'*' * 10 if PADDLE_API_KEY else 'NOT SET'}, Price ID: {price_id}")
    print(f"[DEBUG] User ID from request: {user_id}")

    if not PADDLE_API_KEY or not price_id:
        print(f"[DEBUG] Missing config - API Key: {bool(PADDLE_API_KEY)}, Price ID: {bool(price_id)}")
        raise HTTPException(503, "Paddle not configured")

    payload = {
        "items": [
            {
                "price_id": price_id,
                "quantity": 1
            }
        ],
        "checkout": {
            "return_url": "https://quantproof.net/success"
        },
        "custom_data": {
            "user_id": user_id
        }
    }

    print(f"[DEBUG] Paddle payload: {payload}")

    headers = {
        "Authorization": f"Bearer {PADDLE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        print(f"[DEBUG] Making Paddle API call...")
        res = requests.post(
            "https://api.paddle.com/transactions",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"[DEBUG] Paddle API response - Status: {res.status_code}")
        print(f"[DEBUG] Paddle API response - Headers: {dict(res.headers)}")
        print(f"[DEBUG] Paddle API response - Body: {res.text[:500]}...")

        if res.status_code != 201:
            print(f"[DEBUG] Paddle API error - Status: {res.status_code}, Response: {res.text}")
            raise HTTPException(500, f"Paddle API error: {res.status_code} - {res.text}")

        data = res.json()
        print(f"[DEBUG] Parsed JSON response: {data}")

        if "data" not in data or "checkout" not in data["data"]:
            print(f"[DEBUG] Invalid response structure - data key: {'data' in data}, checkout key: {'checkout' in data.get('data', {})}")
            raise HTTPException(500, "Invalid Paddle response")

        checkout_url = data["data"]["checkout"]["url"]
        print(f"[DEBUG] Success! Checkout URL: {checkout_url}")

        return {
            "checkout_url": checkout_url
        }
        
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Request exception: {e}")
        raise HTTPException(500, f"Paddle API request failed: {str(e)}")
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Paddle checkout error: {str(e)}")

@app.post("/report/create-checkout-paddle-portfolio")
async def create_checkout_paddle_portfolio(payload: dict):
    import requests as _requests
    api_key  = os.environ.get("PADDLE_API_KEY", "")
    price_id = os.environ.get("PADDLE_PRICE_PORTFOLIO", "")
    if not api_key or not price_id:
        raise HTTPException(503, "Paddle not configured")
    res = _requests.post(
        "https://api.paddle.com/transactions",
        json={"items": [{"price_id": price_id, "quantity": 1}],
              "checkout": {"return_url": "https://quantproof.net/success"},
              "custom_data": {"user_id": "anonymous"}},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30
    )
    data = res.json()
    url = data.get("data", {}).get("checkout", {}).get("url", "")
    if not url:
        raise HTTPException(500, f"Paddle error: {data}")
    return {"checkout_url": url}

@app.post("/report/create-checkout-paddle-pro")
async def create_checkout_paddle_pro(payload: dict):
    import requests as _requests
    api_key  = os.environ.get("PADDLE_API_KEY", "")
    price_id = os.environ.get("PADDLE_PRICE_PRO", "")
    if not api_key or not price_id:
        raise HTTPException(503, "Paddle not configured")
    res = _requests.post(
        "https://api.paddle.com/transactions",
        json={"items": [{"price_id": price_id, "quantity": 1}],
              "checkout": {"return_url": "https://quantproof.net/success"},
              "custom_data": {"user_id": "anonymous"}},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30
    )
    data = res.json()
    url = data.get("data", {}).get("checkout", {}).get("url", "")
    if not url:
        raise HTTPException(500, f"Paddle error: {data}")
    return {"checkout_url": url}

@app.get("/report/paddle-token")
async def get_paddle_token(txn_id: str = ""):
    """Frontend polls this after Paddle payment to get the token."""
    import hashlib, time
    if not txn_id:
        return {"status": "pending"}

    key_secret   = os.environ.get("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    current_hour = int(time.time() // 3600)

    for tier, prefix in [
        ("report",    TOKEN_REPORT),
        ("lite",      TOKEN_LITE),
        ("portfolio", TOKEN_PORTFOLIO),
        ("pro",       TOKEN_PRO),
    ]:
        for hour_offset in (0, -1, 1):
            candidate = f"{prefix}{hashlib.sha256(f'{tier}:{current_hour + hour_offset}:{key_secret}:{txn_id}'.encode()).hexdigest()[:32]}"
            if candidate in REPORT_ACCESS_TOKENS:
                return {"status": "ready", "token": candidate, "tier": tier}

    return {"status": "pending"}

@app.post("/report/webhook-paddle")
async def paddle_webhook(request: Request):
    import hashlib, hmac, time, json

    body = await request.body()

    webhook_secret   = os.environ.get("PADDLE_WEBHOOK_SECRET", "")
    paddle_signature = request.headers.get("Paddle-Signature", "")
    if webhook_secret and paddle_signature:
        try:
            ts = ""
            h1 = ""
            for part in paddle_signature.split(";"):
                if part.startswith("ts="):
                    ts = part[3:]
                elif part.startswith("h1="):
                    h1 = part[3:]
            signed_payload = f"{ts}:{body.decode('utf-8')}"
            expected = hmac.new(
                webhook_secret.encode(),
                signed_payload.encode(),
                hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, h1):
                raise HTTPException(status_code=400, detail="Invalid Paddle signature")
        except HTTPException:
            raise
        except Exception:
            pass

    try:
        event = json.loads(body)
    except Exception:
        return {"status": "ignored"}

    event_type = event.get("event_type", "")
    if event_type not in ("transaction.completed", "transaction.paid"):
        return {"status": "ignored"}

    txn      = event.get("data", {})
    txn_id   = txn.get("id", "")
    items    = txn.get("items", [])
    price_id = items[0].get("price", {}).get("id", "") if items else ""

    PRICE_LITE      = os.environ.get("PADDLE_PRICE_LITE", "")
    PRICE_REPORT    = os.environ.get("PADDLE_PRICE_REPORT", "")
    PRICE_PORTFOLIO = os.environ.get("PADDLE_PRICE_PORTFOLIO", "")
    PRICE_PRO       = os.environ.get("PADDLE_PRICE_PRO", "")

    if price_id == PRICE_LITE:
        tier         = "lite"
        token_prefix = TOKEN_LITE
    elif price_id == PRICE_PORTFOLIO:
        tier         = "portfolio"
        token_prefix = TOKEN_PORTFOLIO
    elif price_id == PRICE_PRO:
        tier         = "pro"
        token_prefix = TOKEN_PRO
    else:
        tier         = "report"
        token_prefix = TOKEN_REPORT

    key_secret   = os.environ.get("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    current_hour = int(time.time() // 3600)
    token_now    = f"{token_prefix}{hashlib.sha256(f'{tier}:{current_hour}:{key_secret}:{txn_id}'.encode()).hexdigest()[:32]}"
    token_next   = f"{token_prefix}{hashlib.sha256(f'{tier}:{current_hour+1}:{key_secret}:{txn_id}'.encode()).hexdigest()[:32]}"

    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)

    if tier == "pro":
        expiry = time.time() + 30 * 24 * 3600
        _PRO_TOKENS[token_now]  = expiry
        _PRO_TOKENS[token_next] = expiry
        _save_pro_tokens()

    _save_tokens()
    print(f"[PADDLE] tier={tier} txn={txn_id} token={token_now[:12]}...")
    return {"status": "ok", "tier": tier, "txn_id": txn_id}

@app.get("/report/token-status")
async def token_status(x_token: str = Header(default="")):
    """Check what tier a token grants and when it expires."""
    tier = _token_tier(x_token)
    if tier == "none":
        return {"tier": "none", "valid": False}
    if tier == "pro":
        expiry = _PRO_TOKENS.get(x_token, 0)
        days_left = max(0, int((expiry - time.time()) / 86400))
        return {"tier": "pro", "valid": True, "days_remaining": days_left}
    return {"tier": tier, "valid": True}


# =========================================================
# BROKER PDF REPORT
# =========================================================

class BrokerReportRequest(BaseModel):
    api_key:         str
    secret_key:      str
    mode:            str = "paper"
    enhanced:        bool = False
    validation_hash: str = ""
    validation_data: dict = {}  # For TV mode - bypass cache

@app.post("/report/generate-broker")
async def generate_broker_report(
    payload: BrokerReportRequest,
    x_report_token: str = Header(default=""),
):
    print(f"[DEBUG] Broker PDF payload: {payload}")
    print(f"[DEBUG] Broker PDF token: {x_report_token[:10] if x_report_token else 'None'}...")
    
    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok: return True
        if _is_pro_token(token): return True          # Pro = unlimited PDFs
        if token in REPORT_ACCESS_TOKENS: return True
        if token.startswith("cs_"): return False
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(
            status_code=402,
            detail={"error": "payment_required",
                    "message": "Purchase required to download the full analysis report.",
                    "price_inr": REPORT_PRICE_INR,
                    "upgrade_url": "https://quantproof-frontend.onrender.com/#upgrade"}
        )

    if not payload.validation_hash:
        print(f"[DEBUG] Broker PDF - MISSING validation_hash")
        raise HTTPException(400, {"error": "missing_validation_hash", "message": "Validation hash is required"})
    
    # Special handling for TradingView mode - use sent data instead of cache
    if payload.mode == 'tradingview':
        print(f"[DEBUG] Broker PDF - TradingView mode detected")
        if payload.validation_data:
            print(f"[DEBUG] Broker PDF - Using sent TV validation data")
            cached_vr = payload.validation_data
        else:
            print(f"[DEBUG] Broker PDF - No TV data sent, checking cache")
            cached_vr = _get_cached_validation(payload.validation_hash)
            if not cached_vr:
                raise HTTPException(400, {
                    "error": "tv_data_missing", 
                    "message": "TradingView validation data missing - please re-run validation then download PDF immediately."
                })
    elif payload.mode == 'ibkr':
        print(f"[DEBUG] Broker PDF - IBKR mode detected")
        if payload.validation_data:
            print(f"[DEBUG] Broker PDF - Using sent IBKR validation data")
            cached_vr = payload.validation_data
        else:
            print(f"[DEBUG] Broker PDF - No IBKR data sent, checking cache")
            cached_vr = _get_cached_validation(payload.validation_hash)
            if not cached_vr:
                raise HTTPException(400, {
                    "error": "ibkr_data_missing", 
                    "message": "IBKR validation data missing - please re-run validation then download PDF immediately."
                })
    else:
        print(f"[DEBUG] Broker PDF - Checking cache for validation_hash: '{payload.validation_hash}'")
        print(f"[DEBUG] Broker PDF - Cache keys available: {list(VALIDATION_RESULT_CACHE.keys())[:5]}...")  # Show first 5 keys
        cached_vr = _get_cached_validation(payload.validation_hash)
        if not cached_vr:
            print(f"[DEBUG] Broker PDF - Cache miss for validation_hash")
            raise HTTPException(400, {
                "error": "cache_expired",
                "message": "Validation result expired — please re-run your validation then download the PDF immediately."
            })
    print(f"[DEBUG] Broker PDF - Cache hit, proceeding with PDF generation")
    
    if not cached_vr.get("ai_plan"):
        try:
            class _CachedCheck:
                def __init__(self, d): self.__dict__.update(d)
            class _CachedReport:
                def __init__(self, vr):
                    self.score = vr["fundable_score"]; self.grade = vr["grade"]
                    self.sharpe = vr["sharpe"]; self.max_drawdown = vr["max_drawdown"]
                    self.win_rate = vr["win_rate"]; self.total_trades = vr["total_trades"]
                    self.checks = [_CachedCheck(c) for c in vr.get("checks", [])]
                    self.audit_flags = []
                    self.prop_firm_compliance = vr.get("prop_firm_compliance")
            cached_vr["ai_plan"] = await generate_ai_plan(
                _CachedReport(cached_vr), cached_vr.get("date_range", "")
            )
        except Exception as _e:
            issues = cached_vr.get("top_issues", [])
            cached_vr["ai_plan"] = {
                "summary": "Top improvements: " + "; ".join(issues[:3]) if issues else "See failed checks above.",
                "fixes": [],
                "encouragement": ""
            } if issues else None
    
    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available.")
    try:
        async with _PDF_SEMAPHORE:
            pdf_bytes = _generate_pdf_report(cached_vr)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")
    score     = int(cached_vr["fundable_score"])
    grade_str = cached_vr["grade"].split("—")[0].strip().replace(" ", "")
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"'}
    )

    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok: return True
        if _is_pro_token(token): return True          # Pro = unlimited PDFs
        if token in REPORT_ACCESS_TOKENS: return True
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(402, {"error": "payment_required", "message": "Purchase required."})

    api_key    = payload.api_key.strip()
    secret_key = payload.secret_key.strip()
    if not api_key or not secret_key:
        raise HTTPException(400, "API key and secret required.")

    is_paper = payload.mode == "paper"
    base_url  = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"

    try:
        import httpx
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key, "accept": "application/json"}
        
        @retry_on_failure(max_retries=3, delay=1.0)
        async def _fetch_alpaca_account_v2(client, base_url, headers):
            return await client.get(f"{base_url}/v2/account", headers=headers)
        
        @retry_on_failure(max_retries=3, delay=1.0)
        async def _fetch_alpaca_orders_v2(client, base_url, headers):
            return await client.get(
                f"{base_url}/v2/orders", headers=headers,
                params={"status": "closed", "limit": 500, "direction": "desc"}
            )
        
        async with httpx.AsyncClient(timeout=20) as client:
            acct_resp = await _fetch_alpaca_account_v2(client, base_url, headers)
            if not acct_resp.is_success:
                raise HTTPException(401, "Invalid Alpaca credentials.")
            orders_resp = await _fetch_alpaca_orders_v2(client, base_url, headers)
        raw_orders = orders_resp.json() if orders_resp.is_success else []
        orders     = raw_orders if isinstance(raw_orders, list) else []

        api_key = secret_key = "[DISCARDED]"
        headers = {}

        df, parse_flags = pair_alpaca_orders(orders)

        _orphan_flag = next((f for f in parse_flags if "orphan" in f.lower() or "predates" in f.lower()), "")
        _open_flag   = next((f for f in parse_flags if "open position" in f.lower()), "")

        if len(df) < 10:
            _detail_lines = []
            if _orphan_flag:
                _detail_lines.append(
                    "Some sells were excluded because their matching buy happened before the visible history window."
                )
            if _open_flag:
                _detail_lines.append("Some positions are still open (no closing sell yet).")
            _detail = " ".join(_detail_lines) if _detail_lines else ""
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "message": (
                    f"Only {len(df)} completed round-trip trade(s) found - need 10. "
                    f"{_detail}"
                ),
                "trades_found": len(df), "trades_needed": 10
            })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Alpaca fetch failed: {str(e)[:200]}")

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
            n_trials=n_trials
        )
        report    = validator.run()
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored = [c for c in report.checks if c.category != "Plausibility"]
    top_issues    = [c.fix  for c in sorted(scored, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]]

    date_range = "No date data"
    if "date" in df.columns:
        try:
            date_range = f"{df['date'].dropna().min().date()} to {df['date'].dropna().max().date()}"
        except Exception:
            pass

    vr = {
        "fundable_score": min(100.0, round(report.score, 1)), "grade": report.grade,
        "sharpe": round(report.sharpe, 2), "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1), "total_trades": report.total_trades,
        "date_range": date_range,
        "n_trials": n_trials,
        "is_verified": True,
        "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                    "value":c.value,"insight":c.insight,"fix":c.fix,
                    "category":c.category} for c in report.checks],
        "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                        "description":s.description,"market_drop":s.market_drop,
                        "strategy_drop":s.strategy_drop,"survived":s.survived,
                        "emotional_verdict":s.emotional_verdict} for s in report.crash_sims],
        "top_issues":    top_issues,
        "top_strengths": top_strengths,
        "audit_flags":   report.audit_flags + parse_flags + ([_bm_note] if _bm_note else []),
        "equity_curve": getattr(report, 'equity_curve', None),
        "sebi_checks": _compute_sebi_checks(df, request, registered_ip),
        "sebi_declaration": {"algo_id": "unsure", "box_type": "white"},
    }

    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan
    _cache_validation(report.validation_hash, vr)
    _save_validation_cache()

    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available on this server.")
    try:
        async with _PDF_SEMAPHORE:
            pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    score     = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    return FastAPIResponse(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"'}
    )


@app.post("/validate/prop-firms", response_model=ValidationResponse)
async def validate_prop_firms(file: UploadFile = File(...), currency: str = Form("USD"), n_trials: int = Form(10), request: Request = None):
    """Enhanced validation with prop firm compliance analysis"""
    # Reuse the same validation logic but with prop firm compliance enabled
    return await validate(file=file, currency=currency, n_trials=n_trials, enhanced=True, request=request)

@app.post("/validate/portfolio", response_model=PortfolioResponse)
async def validate_portfolio(
    files: List[UploadFile] = File(...),
    currency: str = Form("USD"),
    strategy_names: str = Form(""),   # comma-separated names
    enhanced: bool = Form(False),
    x_portfolio_token: str = Header(default=""),
    request: Request = None,
):
    """
    Portfolio correlation analysis — Free to try, blurred results.
    Accepts 2-3 CSV files. Returns per-strategy scores + correlation matrix
    + combined stats + diversification score (blurred without payment).
    """
    # Check if user has paid access (PRO or PORT tokens)
    has_paid_access = (_is_pro_token(x_portfolio_token) or
                      _token_tier(x_portfolio_token) in ("portfolio", "pro"))
    
    # Allow free analysis but mark results as preview
    is_preview = not has_paid_access

    if len(files) < 2:
        raise HTTPException(400, "Portfolio analysis requires at least 2 strategy CSV files.")
    if len(files) > 3:
        raise HTTPException(400, "Maximum 3 strategies per portfolio analysis.")

    names_list = [n.strip() for n in strategy_names.split(",") if n.strip()]
    while len(names_list) < len(files):
        names_list.append(f"Strategy {len(names_list) + 1}")

    audit_flags = []
    strategy_summaries = []
    dfs = []
    sharpes = []

    for i, file in enumerate(files):
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(400, f"File {i+1} must be a CSV.")
        contents = await file.read()
        if len(contents) > 5_000_000:
            raise HTTPException(400, f"File too large ({len(contents)//1024}KB). Max 5MB — a real trade CSV is never larger.")
        try:
            df, regime_note = smart_parse_csv(contents)
        except (ValueError, HTTPException) as e:
            raise HTTPException(400, f"Strategy {i+1} ({names_list[i]}): {str(e)}")

        df = _dedup_trades(df)
        if len(df) < 10:
            raise HTTPException(400,
                f"Strategy {i+1} ({names_list[i]}) has only {len(df)} trades — need at least 10.")

        try:
            _bm_type = get_benchmark_type(df, currency)
            _bm, _bm_note = (get_nifty_benchmark(df)
                if _bm_type == "nifty" else get_spy_benchmark(df))
            validator = QuantProofValidator(
                df, benchmark_returns=_bm, benchmark_type=_bm_type, n_trials=10, enable_prop_firm=enhanced
            )
            report = validator.run(enhanced=enhanced)
        except Exception as e:
            raise HTTPException(500, f"Validation failed for strategy {i+1}: {str(e)}")

        date_range = "No date data"
        _has_dates = "date" in df.columns and df["date"].notna().any()
        if _has_dates:
            try:
                date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
            except Exception:
                pass
        else:
            audit_flags.append(
                f"⚠ {names_list[i]}: no date column detected — correlation computed "
                f"positionally (trade 1 vs trade 1). If strategies trade at different "
                f"frequencies, correlation accuracy is reduced. Upload dated CSVs for "
                f"date-aligned correlation."
            )

        # Build per-strategy richness — same as single strategy validator
        _checks_list = [
            {"name": c.name, "passed": c.passed, "score": round(c.score, 1),
             "category": c.category, "insight": c.insight or "", "fix": c.fix or ""}
            for c in report.checks if c.category != "Plausibility"
        ]
        _failed     = [c for c in _checks_list if not c["passed"]]
        _passed_chk = [c for c in _checks_list if c["passed"]]
        _crash_sims = getattr(report, "crash_sims", [])
        _survived   = sum(1 for c in _crash_sims if
                          (c.get("survived") if isinstance(c, dict) else getattr(c, "survived", False)))
        # Equity curve
        _eq = None
        try:
            _eq = [round(float(v), 4) for v in getattr(report, "equity_curve", None) or []]
            if len(_eq) < 3: _eq = None
        except Exception: _eq = None
        # Top issues and strengths
        _top_issues = [f"{c['name']}: {c['insight']}" for c in _failed[:3] if c.get("insight")]
        _top_strengths = [f"{c['name']}" for c in _passed_chk
                          if c.get("score", 0) >= 80][:3]
        # Cast all numerics to plain Python types — numpy.int64/float64 break Pydantic
        strategy_summary = {
            "name":             str(names_list[i]),
            "fundable_score":   float(round(report.score, 1)),
            "grade":            report.grade.split("—")[0].strip(),
            "sharpe":           float(round(report.sharpe, 2)),
            "max_drawdown":     float(round(report.max_drawdown, 4)),
            "win_rate":         float(round(report.win_rate, 1)),
            "total_trades":     int(report.total_trades),
            "date_range":       str(date_range),
            "regime_note":      str(regime_note),
            "source_format":    regime_note.split("|")[0].strip()[:40] if regime_note else "CSV",
            "checks_passed":    int(len(_passed_chk)),
            "checks_failed":    int(len(_failed)),
            "checks_total":     int(len(_checks_list)),
            "crash_survived":   int(_survived),
            "crash_total":      int(len(_crash_sims)),
            "top_issues":       [str(x) for x in _top_issues],
            "top_strengths":    [str(x) for x in _top_strengths],
            "equity_curve":     _eq,
            "checks":           [
                {
                    "name":     str(c.get("name","")),
                    "passed":   bool(c.get("passed", False)),
                    "score":    float(c.get("score", 0)),
                    "category": str(c.get("category","")),
                    "insight":  str(c.get("insight","")),
                    "fix":      str(c.get("fix","")),
                    "value":    str(c.get("value","")),
                }
                for c in _checks_list[:44]
            ],
        }
        
        # Add prop firm compliance data if enhanced validation is enabled
        if enhanced and getattr(report, 'prop_firm_compliance', None):
            strategy_summary["prop_firm_compliance"] = getattr(report, 'prop_firm_compliance', None)
        
        strategy_summaries.append(strategy_summary)
        dfs.append(df)
        sharpes.append(report.sharpe)
        if regime_note:
            audit_flags.append(f"{names_list[i]}: {regime_note}")

    # Correlation matrix (normalised PnL — scale independent)
    corr_mat = _portfolio_correlation(dfs)

    # Risk parity weights — institutional standard allocation
    rp_weights = _risk_parity_weights(dfs)

    # Off-diagonal correlation stats — signed and absolute
    n = len(dfs)
    off_diag_signed = [float(corr_mat[i][j]) for i in range(n) for j in range(i+1, n)]
    off_diag_abs    = [abs(v) for v in off_diag_signed]
    avg_corr_signed = float(np.mean(off_diag_signed)) if off_diag_signed else 0.0
    avg_corr        = float(np.mean(off_diag_abs))    if off_diag_abs    else 0.0

    # Combined stats — risk-parity weighted, fillna(0) not dropna
    max_len = max(len(df) for df in dfs)
    weighted_pnl = np.zeros(max_len)
    for df, w in zip(dfs, rp_weights):
        pnl = df["pnl"].values.astype(float)
        padded = np.pad(pnl, (0, max_len - len(pnl)), constant_values=0.0)
        weighted_pnl += padded * w

    combined_std    = float(weighted_pnl.std())
    combined_mean   = float(weighted_pnl.mean())
    # Annualise by actual trade frequency, not a fixed 252.
    # total_trades spread across the longest strategy's date range.
    # If no date range is available, fall back to the number of observations.
    _total_trades = sum(s["total_trades"] for s in strategy_summaries)
    _date_ranges  = [s["date_range"] for s in strategy_summaries if s.get("date_range","No date data") != "No date data"]
    try:
        if _date_ranges:
            _start = min(pd.to_datetime(r.split(" to ")[0], errors="coerce") for r in _date_ranges)
            _end   = max(pd.to_datetime(r.split(" to ")[1], errors="coerce") for r in _date_ranges)
            _years = max((_end - _start).days / 365.25, 0.1)
            _ann_factor = float(np.sqrt(_total_trades / _years))
        else:
            _ann_factor = float(np.sqrt(max_len))
    except Exception:
        _ann_factor = float(np.sqrt(252))
    combined_sharpe = float(combined_mean / combined_std * _ann_factor) if combined_std > 1e-10 else 0.0

    # Combined drawdown from actual weighted equity curve
    eq_combined  = np.cumprod(1 + np.clip(weighted_pnl, -0.99, 0.99))
    roll_max_c   = np.maximum.accumulate(eq_combined)
    combined_dd  = float(((eq_combined - roll_max_c) / np.where(roll_max_c > 0, roll_max_c, 1.0)).min())
    combined_wr  = float((weighted_pnl > 0).mean() * 100)
    combined_trades = int(sum(s["total_trades"] for s in strategy_summaries))

    # Combined equity curve (normalised to 100)
    combined_equity = [float(round(v * 100 / eq_combined[0], 2)) for v in eq_combined] if eq_combined[0] > 0 else []

    # Drawdown coincidence — % of drawdown periods where 2+ strategies fall together
    dd_coincidence = _drawdown_coincidence(dfs)

    div_score = float(_diversification_score(corr_mat, sharpes))
    verdict   = _portfolio_verdict(div_score, avg_corr_signed, n)

    # Portfolio-level DSR estimate (Bailey & Lopez de Prado 2012)
    # Uses expected max Sharpe correction for n combined strategies
    # Returns raw p-value capped at 99.8 to avoid misleading 100/100 at high Sharpe
    total_obs = max_len
    try:
        from scipy import stats as _scipy_stats
        import math as _math
        _sr_std = float(np.sqrt((1 + 0.5 * combined_sharpe**2) / max(total_obs - 1, 1)))
        _e_max  = ((1 - _math.e**(-_math.pi/2)) *
                   _scipy_stats.norm.ppf(1 - 1.0/max(n, 1)) +
                   (_math.e**(-_math.pi/2)) *
                   _scipy_stats.norm.ppf(1 - 1.0/(max(n, 1) * _math.e)))
        _dsr_z  = (combined_sharpe - max(_e_max, 0)) / _sr_std if _sr_std > 0 else 0
        _raw_p  = float(_scipy_stats.norm.cdf(_dsr_z) * 100)
        # Cap at 99.8 — 100/100 is mathematically possible but misleading
        # (signals "no selection bias" which is never strictly true)
        portfolio_dsr = float(round(min(_raw_p, 99.8), 1))
    except Exception:
        portfolio_dsr = None

    audit_flags.append(
        f"Portfolio: {n} strategies, avg abs correlation {avg_corr:.2f}, "
        f"avg signed correlation {avg_corr_signed:.2f}, "
        f"div score {div_score}/100, DD coincidence {dd_coincidence:.0%}, "
        f"risk parity weights {[f'{w:.2f}' for w in rp_weights]}"
    )

    # Generate portfolio-level fix plan
    portfolio_plan = await generate_portfolio_fix_plan(
        strategies=strategy_summaries,
        corr_matrix=[[float(v) for v in row] for row in corr_mat.tolist()],
        rp_weights=rp_weights,
        div_score=div_score,
        dd_coincidence=dd_coincidence,
        combined_sharpe=combined_sharpe,
        combined_dd=combined_dd,
        avg_corr=avg_corr_signed,
        portfolio_dsr=portfolio_dsr,
        strategy_names=names_list,
    )

    # Cache portfolio validation for PDF/AI generation with prop firm compliance data
    portfolio_hash = f"portfolio-{'-'.join(sorted(names_list))}"
    _vr_cache = {
        "fundable_score": float(round(combined_sharpe * 10, 1)),  # Portfolio score from combined Sharpe
        "grade": "Portfolio Analysis",
        "sharpe": float(round(combined_sharpe, 2)),
        "max_drawdown": float(round(combined_dd, 4)),
        "win_rate": float(round(combined_wr, 1)),
        "total_trades": int(combined_trades),
        "date_range": "Portfolio Analysis",
        "is_verified": False,
        "strategies": strategy_summaries,
        "correlation_matrix": [[float(v) for v in row] for row in corr_mat.tolist()],
        "combined_sharpe": float(round(combined_sharpe, 2)),
        "combined_drawdown": float(round(combined_dd, 4)),
        "combined_win_rate": float(round(combined_wr, 1)),
        "diversification_score": float(div_score),
        "avg_correlation": float(round(avg_corr_signed, 3)),
        "portfolio_verdict": str(verdict),
        "audit_flags": [str(f) for f in audit_flags],
        "portfolio_dsr": portfolio_dsr,
        "combined_equity": [float(v) for v in combined_equity[:200]],
        "portfolio_plan": portfolio_plan,
    }
    
    # Add aggregated prop firm compliance data if enhanced validation is enabled
    if enhanced:
        prop_firm_data = {}
        for strategy in strategy_summaries:
            if "prop_firm_compliance" in strategy:
                prop_firm_data[strategy["name"]] = strategy["prop_firm_compliance"]
        
        if prop_firm_data:
            _vr_cache["prop_firm_compliance"] = prop_firm_data
    
    _cache_validation(portfolio_hash, _vr_cache)
    _save_validation_cache()

    # Apply blurring for free users (preview mode)
    if is_preview:
        # Blur key metrics to drive conversion
        blurred_rp_weights = [round(w, 1) if i == 0 else 0.0 for i, w in enumerate(rp_weights)]
        # Only show first strategy's weight clearly, others as 0
        blurred_combined_equity = combined_equity[:50] if combined_equity else []  # Show only first 50 points
        # Blur portfolio plan details
        blurred_plan = {
            "summary": portfolio_plan.get("summary", "")[:100] + "..." if len(portfolio_plan.get("summary", "")) > 100 else portfolio_plan.get("summary", ""),
            "portfolio_fixes": portfolio_plan.get("portfolio_fixes", [])[:1],  # Show only first fix
            "allocation_verdict": "Unlock full portfolio analysis to see complete allocation verdict",
            "strongest_combination": "Unlock full portfolio analysis to see detailed strategy combinations"
        }
    else:
        blurred_rp_weights = [float(round(w, 3)) for w in rp_weights]
        blurred_combined_equity = [float(v) for v in combined_equity[:200]]
        blurred_plan = portfolio_plan

    return PortfolioResponse(
        strategies=strategy_summaries,
        correlation_matrix=[[float(v) for v in row] for row in corr_mat.tolist()],
        strategy_names=[str(nm) for nm in names_list],
        combined_sharpe=float(round(combined_sharpe, 2)),
        combined_drawdown=float(round(combined_dd, 4)),
        combined_win_rate=float(round(combined_wr, 1)),
        diversification_score=float(div_score),
        avg_correlation=float(round(avg_corr_signed, 3)),
        combined_total_trades=int(combined_trades),
        portfolio_verdict=str(verdict),
        audit_flags=[str(f) for f in audit_flags],
        n_trials=10,
        rp_weights=blurred_rp_weights,
        dd_coincidence=float(dd_coincidence),
        portfolio_dsr=portfolio_dsr,
        combined_equity=blurred_combined_equity,
        portfolio_plan=blurred_plan,
        is_preview=is_preview,
    )



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

@app.post("/connect/ibkr", response_model=IBKRConnectResponse)
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

@app.post("/validate/kite", response_model=ValidationResponse)
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


# ================= ROUTERS =================
from routers import system as _system_router, tradingview as _tradingview_router

app.include_router(_system_router.router)
app.include_router(_tradingview_router.router)


# ================= SERVER STARTUP =================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting QuantProof Backend Server...")
    print("📍 Enhanced Validation endpoints ready")
    print("📍 Prop Firm Compliance integration enabled")
    print("📍 Server running on: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)































