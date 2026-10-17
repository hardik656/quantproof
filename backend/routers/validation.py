"""Strategy validation endpoints (CSV upload, AI fixes, prop firms, portfolio).

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
from schemas import CheckResultResponse, CrashSimResponse, ValidationResponse
from services.ai_plans import generate_ai_plan, generate_portfolio_fix_plan
from services.benchmarks import get_benchmark_type, get_nifty_benchmark, get_spy_benchmark
from services.cache import _cache_validation, _get_cached_validation, _save_validation_cache
from services.parsers import _dedup_trades, smart_parse_csv
from services.portfolio import PortfolioResponse, _diversification_score, _drawdown_coincidence, _portfolio_correlation, _portfolio_verdict, _risk_parity_weights
from services.tokens import FREE_DAILY_LIMIT, _is_pro_token, _token_tier, check_rate_limit

router = APIRouter()

@router.post("/validate", response_model=ValidationResponse)
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

@router.post("/validate/ai-fixes")
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



@router.post("/validate/prop-firms", response_model=ValidationResponse)
async def validate_prop_firms(file: UploadFile = File(...), currency: str = Form("USD"), n_trials: int = Form(10), request: Request = None):
    """Enhanced validation with prop firm compliance analysis"""
    # Reuse the same validation logic but with prop firm compliance enabled
    return await validate(file=file, currency=currency, n_trials=n_trials, enhanced=True, request=request)

@router.post("/validate/portfolio", response_model=PortfolioResponse)
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



