"""PDF reports, checkout and signature-verified payment webhooks (Razorpay, Paddle).

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from config import extract_n_trials
from services.ai_plans import generate_ai_plan, generate_portfolio_fix_plan
from services.benchmarks import get_benchmark_type, get_nifty_benchmark, get_spy_benchmark
from services.cache import (
    _PDF_SEMAPHORE,
    VALIDATION_RESULT_CACHE,
    _cache_validation,
    _get_cached_validation,
    _save_validation_cache,
)
from services.parsers import _dedup_trades, pair_alpaca_orders, retry_on_failure, smart_parse_csv
from services.portfolio import (
    _diversification_score,
    _drawdown_coincidence,
    _portfolio_correlation,
    _portfolio_verdict,
    _risk_parity_weights,
)
from services.sebi import _compute_sebi_checks
from services.tokens import (
    _PRO_TOKENS,
    REPORT_ACCESS_TOKENS,
    TOKEN_LITE,
    TOKEN_PORTFOLIO,
    TOKEN_PRO,
    TOKEN_REPORT,
    _is_pro_token,
    _save_pro_tokens,
    _save_tokens,
    _token_tier,
)
from validator import QuantProofValidator

router = APIRouter()


def _razorpay_secret() -> str:
    """Webhook HMAC secret. Refuse to verify with a guessable default."""
    secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not secret:
        raise HTTPException(503, "Payments are not configured")
    return secret


# =========================================================
# PDF REPORT — PAYWALL ENDPOINT
# =========================================================


try:
    from generate_report import generate_pdf_report as _generate_pdf_report

    _PDF_AVAILABLE = True
except ImportError:
    _generate_pdf_report = None
    _PDF_AVAILABLE = False


_TEST_TOKEN = os.getenv("TEST_REPORT_TOKEN", "")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
REPORT_PRICE_USD = 4
REPORT_PRICE_INR = 299


@router.post("/report/generate")
async def generate_report_endpoint(
    file: UploadFile = File(...),
    currency: str = Form("USD"),
    enhanced: bool = Form(False),
    registered_ip: str = Form(""),
    sebi_algo_id: str = Form("unsure"),
    sebi_box_type: str = Form("white"),
    x_report_token: str = Header(default=""),
    x_report_source: str = Header(default=""),
    request: Request = None,
):
    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok:
            return True
        if _is_pro_token(token):
            return True  # Pro = unlimited PDFs
        if token in REPORT_ACCESS_TOKENS:
            return True
        if token.startswith("cs_"):
            return False
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(
            status_code=402,
            detail={
                "error": "payment_required",
                "message": "Purchase required to download the full analysis report.",
                "price_inr": REPORT_PRICE_INR,
                "upgrade_url": "https://quantproof.net/pricing",
            },
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")

    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(
            400, f"File too large ({len(contents) // 1024}KB). Max 5MB — a real trade CSV is never larger."
        )
    try:
        df, regime_note = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(df) < 10:
        raise HTTPException(400, f"Only {len(df)} trades found. Need at least 10.")

    try:
        _bm_type = get_benchmark_type(df, currency)
        _bm, _bm_note = get_nifty_benchmark(df) if _bm_type == "nifty" else get_spy_benchmark(df)
        n_trials = extract_n_trials(request) if request else 1
        # PDF tier: force minimum n=20 for RPT and PRO tokens
        _tier = _token_tier(x_report_token)
        if _tier in ("report", "pro"):
            n_trials = max(20, min(50, n_trials))
        n_trials = max(1, min(50, n_trials))
        validator = QuantProofValidator(
            df, benchmark_returns=_bm, benchmark_type=_bm_type, n_trials=n_trials, enable_prop_firm=enhanced
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
                "name": c.name,
                "passed": False,
                "score": 0,
                "value": "⚠️ NOT APPLICABLE — CSV Upload",
                "insight": (
                    "Prop firm and institutional compliance certificates require "
                    "broker-verified trade data. CSV uploads cannot be used for "
                    "compliance submissions as the data is self-reported and unverified."
                ),
                "fix": "Connect your Alpaca account (or IBKR when available) to generate a verified compliance report.",
                "category": c.category,
            }
        return {
            "name": c.name,
            "passed": c.passed,
            "score": c.score,
            "value": c.value,
            "insight": c.insight,
            "fix": c.fix,
            "category": c.category,
        }

    vr = {
        "fundable_score": min(100.0, round(report.score, 1)),
        "grade": report.grade,
        "sharpe": round(report.sharpe, 2),
        "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1),
        "total_trades": report.total_trades,
        "date_range": date_range,
        "n_trials": n_trials,
        "is_verified": x_report_source in ("alpaca", "ibkr"),
        "checks": [
            (
                _csv_check_filter(c)
                if x_report_source == "csv"
                else {
                    "name": c.name,
                    "passed": c.passed,
                    "score": c.score,
                    "value": c.value,
                    "insight": c.insight,
                    "fix": c.fix,
                    "category": c.category,
                }
            )
            for c in report.checks
        ],
        "crash_sims": [
            {
                "crash_name": s.crash_name,
                "year": s.year,
                "description": s.description,
                "market_drop": s.market_drop,
                "strategy_drop": s.strategy_drop,
                "survived": s.survived,
                "emotional_verdict": s.emotional_verdict,
            }
            for s in report.crash_sims
        ],
        "top_issues": [c.fix for c in sorted(scored, key=lambda x: x.score)[:4]],
        "top_strengths": [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]],
        "regime_note": regime_note,
        "audit_flags": report.audit_flags
        + ([_bm_note] if _bm_note else [])
        + ([regime_note] if regime_note else []),
        "plausibility_summary": getattr(report, "plausibility_summary", None),
        "equity_curve": getattr(report, "equity_curve", None),
        "monte_carlo": getattr(report, "monte_carlo", None),
        "prop_firm_compliance": getattr(report, "prop_firm_compliance", None),
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

    score = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    filename = f"QuantProof_Report_{grade_str}_{score}.pdf"
    return FastAPIResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/report/create-checkout")
async def create_checkout(payload: dict):
    try:
        import razorpay
    except ImportError:
        raise HTTPException(
            503,
            {
                "error": "razorpay_not_installed",
                "message": "razorpay package not installed. Run: pip install razorpay",
            },
        )
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(
            503,
            {
                "error": "razorpay_not_configured",
                "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment.",
            },
        )
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.create(
            {
                "amount": 29900,
                "currency": "INR",
                "payment_capture": 1,
                "notes": {"product": "QuantProof Full Report"},
            }
        )
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": key_id,
            "checkout_url": None,
        }
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@router.post("/report/webhook")
async def payment_webhook(request: Request):
    import hmac
    import time

    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id = data.get("razorpay_order_id", "")
    signature = data.get("razorpay_signature", "")
    key_secret = _razorpay_secret()
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    token_now = hashlib.sha256(f"report:{current_hour}:{key_secret}:{payment_id}".encode()).hexdigest()[:32]
    token_next = hashlib.sha256(f"report:{current_hour + 1}:{key_secret}:{payment_id}".encode()).hexdigest()[
        :32
    ]
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    _save_tokens()
    return {"status": "ok", "token": token_now}


@router.post("/report/create-checkout-lite")
async def create_checkout_lite(payload: dict):
    try:
        import razorpay
    except ImportError:
        raise HTTPException(
            503,
            {
                "error": "razorpay_not_installed",
                "message": "razorpay package not installed. Run: pip install razorpay",
            },
        )
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(
            503,
            {
                "error": "razorpay_not_configured",
                "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment.",
            },
        )
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.create(
            {
                "amount": 9900,
                "currency": "INR",
                "payment_capture": 1,
                "notes": {"product": "QuantProof Lite Unlock"},
            }
        )
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": key_id,
            "checkout_url": None,
        }
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@router.post("/report/webhook-lite")
async def payment_webhook_lite(request: Request):
    import hmac
    import time

    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id = data.get("razorpay_order_id", "")
    signature = data.get("razorpay_signature", "")
    key_secret = _razorpay_secret()
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    token_now = (
        f"LITE-{hashlib.sha256(f'lite:{current_hour}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    )
    token_next = f"LITE-{hashlib.sha256(f'lite:{current_hour + 1}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    _save_tokens()
    return {"status": "ok", "token": token_now}


# =========================================================
# PORTFOLIO PDF REPORT
# =========================================================


@router.post("/report/generate-portfolio")
async def generate_portfolio_report_endpoint(
    files: List[UploadFile] = File(...),
    currency: str = Form("USD"),
    strategy_names: str = Form(""),
    x_portfolio_token: str = Header(default=""),
    request: Request = None,
):
    """Portfolio PDF — requires PORT or PRO token."""
    if not (_is_pro_token(x_portfolio_token) or _token_tier(x_portfolio_token) in ("portfolio", "pro")):
        raise HTTPException(
            402,
            {
                "error": "payment_required",
                "message": "Portfolio PDF requires ₹199 Portfolio unlock or Pro plan.",
                "price_inr": 199,
            },
        )

    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available.")
    if len(files) < 2:
        raise HTTPException(400, "Need at least 2 strategy CSV files.")
    if len(files) > 3:
        raise HTTPException(400, "Maximum 3 strategies.")

    names_list = [n.strip() for n in strategy_names.split(",") if n.strip()]
    while len(names_list) < len(files):
        names_list.append(f"Strategy {len(names_list) + 1}")

    audit_flags = []
    strategy_summaries = []
    dfs = []
    sharpes = []
    _tier = _token_tier(x_portfolio_token)
    _n = 20 if _tier == "pro" else 10

    for i, file in enumerate(files):
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(400, f"File {i + 1} must be a CSV.")
        contents = await file.read()
        if len(contents) > 5_000_000:
            raise HTTPException(
                400,
                f"File too large ({len(contents) // 1024}KB). Max 5MB — a real trade CSV is never larger.",
            )
        try:
            df, regime_note = smart_parse_csv(contents)
        except (ValueError, HTTPException) as e:
            raise HTTPException(400, f"Strategy {i + 1}: {str(e)}")
        df = _dedup_trades(df)
        if len(df) < 10:
            raise HTTPException(400, f"Strategy {i + 1} has only {len(df)} trades.")
        try:
            _bm_type = get_benchmark_type(df, currency)
            _bm, _bm_note = get_nifty_benchmark(df) if _bm_type == "nifty" else get_spy_benchmark(df)
            validator = QuantProofValidator(df, benchmark_returns=_bm, benchmark_type=_bm_type, n_trials=_n)
            report = validator.run()
        except Exception as e:
            raise HTTPException(500, f"Validation failed for strategy {i + 1}: {str(e)}")

        date_range = "No date data"
        if "date" in df.columns and df["date"].notna().any():
            try:
                date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
            except Exception:
                pass
        else:
            audit_flags.append(f"⚠ {names_list[i]}: no date column — correlation positional.")

        _cl = [
            {
                "name": c.name,
                "passed": c.passed,
                "score": round(float(c.score), 1),
                "category": c.category,
                "insight": c.insight or "",
                "fix": c.fix or "",
                "value": str(c.value or ""),
            }
            for c in report.checks
            if c.category != "Plausibility"
        ]
        _fail = [c for c in _cl if not c["passed"]]
        _pass = [c for c in _cl if c["passed"]]
        _cs = getattr(report, "crash_sims", [])
        _surv = int(
            sum(
                1
                for c in _cs
                if (c.get("survived") if isinstance(c, dict) else getattr(c, "survived", False))
            )
        )
        _eq = None
        try:
            _eq = [round(float(v), 4) for v in getattr(report, "equity_curve", None) or []]
            if len(_eq) < 3:
                _eq = None
        except Exception:
            pass

        strategy_summaries.append(
            {
                "name": str(names_list[i]),
                "fundable_score": float(round(report.score, 1)),
                "grade": report.grade.split("—")[0].strip(),
                "sharpe": float(round(report.sharpe, 2)),
                "max_drawdown": float(round(report.max_drawdown, 4)),
                "win_rate": float(round(report.win_rate, 1)),
                "total_trades": int(report.total_trades),
                "date_range": str(date_range),
                "checks_passed": int(len(_pass)),
                "checks_failed": int(len(_fail)),
                "checks_total": int(len(_cl)),
                "crash_survived": _surv,
                "crash_total": int(len(_cs)),
                "top_issues": [str(f"{c['name']}: {c['insight']}") for c in _fail[:3] if c.get("insight")],
                "top_strengths": [str(c["name"]) for c in _pass if c.get("score", 0) >= 80][:3],
                "equity_curve": _eq,
                "checks": [
                    {
                        "name": str(c.get("name", "")),
                        "passed": bool(c.get("passed", False)),
                        "score": float(c.get("score", 0)),
                        "category": str(c.get("category", "")),
                        "insight": str(c.get("insight", "")),
                        "fix": str(c.get("fix", "")),
                        "value": str(c.get("value", "")),
                    }
                    for c in _cl[:44]
                ],
            }
        )
        dfs.append(df)
        sharpes.append(report.sharpe)
        if regime_note:
            audit_flags.append(f"{names_list[i]}: {regime_note}")

    corr_mat = _portfolio_correlation(dfs)
    rp_weights = _risk_parity_weights(dfs)
    n = len(dfs)
    off_signed = [float(corr_mat[i][j]) for i in range(n) for j in range(i + 1, n)]
    avg_corr_signed = float(np.mean(off_signed)) if off_signed else 0.0
    float(np.mean([abs(v) for v in off_signed])) if off_signed else 0.0
    max_len = max(len(df) for df in dfs)
    weighted = np.zeros(max_len)
    for df, w in zip(dfs, rp_weights, strict=False):
        p = df["pnl"].values.astype(float)
        weighted += np.pad(p, (0, max_len - len(p)), constant_values=0.0) * w
    cs_std = float(weighted.std())
    cs_mean = float(weighted.mean())
    _pdf_total_trades = sum(s["total_trades"] for s in strategy_summaries)
    _pdf_date_ranges = [
        s["date_range"] for s in strategy_summaries if s.get("date_range", "No date data") != "No date data"
    ]
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
    eq_c = np.cumprod(1 + np.clip(weighted, -0.99, 0.99))
    rm_c = np.maximum.accumulate(eq_c)
    cd = float(((eq_c - rm_c) / np.where(rm_c > 0, rm_c, 1.0)).min())
    cw = float((weighted > 0).mean() * 100)
    ddc = _drawdown_coincidence(dfs)
    div = float(_diversification_score(corr_mat, sharpes))
    ceq = [float(round(v * 100 / eq_c[0], 2)) for v in eq_c] if eq_c[0] > 0 else []
    try:
        import math as _m

        from scipy import stats as _scs

        _sr_std = float(np.sqrt((1 + 0.5 * cs**2) / max(max_len - 1, 1)))
        _e_max = (1 - _m.e ** (-_m.pi / 2)) * _scs.norm.ppf(1 - 1.0 / max(n, 1)) + (
            _m.e ** (-_m.pi / 2)
        ) * _scs.norm.ppf(1 - 1.0 / (max(n, 1) * _m.e))
        _dsr_z = (cs - max(_e_max, 0)) / _sr_std if _sr_std > 0 else 0
        pdsr = float(round(min(float(_scs.norm.cdf(_dsr_z) * 100), 99.8), 1))
    except Exception:
        pdsr = None

    verdict = _portfolio_verdict(div, avg_corr_signed, n)
    plan = await generate_portfolio_fix_plan(
        strategies=strategy_summaries,
        corr_matrix=[[float(v) for v in row] for row in corr_mat.tolist()],
        rp_weights=rp_weights,
        div_score=div,
        dd_coincidence=ddc,
        combined_sharpe=cs,
        combined_dd=cd,
        avg_corr=avg_corr_signed,
        portfolio_dsr=pdsr,
        strategy_names=names_list,
    )

    best = max(strategy_summaries, key=lambda s: s["fundable_score"])
    vr = {
        "fundable_score": best["fundable_score"],
        "grade": best["grade"],
        "sharpe": float(round(cs, 2)),
        "max_drawdown": float(round(cd, 4)),
        "win_rate": float(round(cw, 1)),
        "total_trades": int(sum(s["total_trades"] for s in strategy_summaries)),
        "date_range": "Portfolio — " + " / ".join(s["date_range"] for s in strategy_summaries),
        "n_trials": _n,
        "is_verified": False,
        "checks": best.get("checks", []),
        "crash_sims": [],
        "top_issues": best.get("top_issues", []),
        "top_strengths": best.get("top_strengths", []),
        "audit_flags": audit_flags,
        "equity_curve": ceq[:200] or None,
        "ai_plan": None,
        "prop_firm_compliance": best.get("prop_firm_compliance", None),
        "portfolio": {
            "strategies": strategy_summaries,
            "correlation_matrix": [[float(v) for v in row] for row in corr_mat.tolist()],
            "strategy_names": names_list,
            "combined_sharpe": float(round(cs, 2)),
            "combined_drawdown": float(round(cd, 4)),
            "combined_win_rate": float(round(cw, 1)),
            "diversification_score": float(div),
            "avg_correlation": float(round(avg_corr_signed, 3)),
            "portfolio_verdict": verdict,
            "rp_weights": [float(round(w, 3)) for w in rp_weights],
            "dd_coincidence": float(ddc),
            "portfolio_dsr": pdsr,
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
    return FastAPIResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =========================================================
# PORTFOLIO CHECKOUT — ₹199
# =========================================================


@router.post("/report/create-checkout-portfolio")
async def create_checkout_portfolio(payload: dict):
    try:
        import razorpay
    except ImportError:
        raise HTTPException(
            503, {"error": "razorpay_not_installed", "message": "razorpay package not installed."}
        )
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(
            503,
            {
                "error": "razorpay_not_configured",
                "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment.",
            },
        )
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.create(
            {
                "amount": 19900,
                "currency": "INR",
                "payment_capture": 1,
                "notes": {"product": "QuantProof Portfolio Analysis"},
            }
        )
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": key_id,
        }
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@router.post("/report/webhook-portfolio")
async def payment_webhook_portfolio(request: Request):
    import hmac
    import time

    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id = data.get("razorpay_order_id", "")
    signature = data.get("razorpay_signature", "")
    key_secret = _razorpay_secret()
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    token_now = (
        f"PORT-{hashlib.sha256(f'port:{current_hour}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    )
    token_next = f"PORT-{hashlib.sha256(f'port:{current_hour + 1}:{key_secret}:{payment_id}'.encode()).hexdigest()[:32]}"
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    _save_tokens()
    return {"status": "ok", "token": token_now}


# =========================================================
# PRO PLAN CHECKOUT — ₹999 / 30 days
# =========================================================


@router.post("/report/create-checkout-pro")
async def create_checkout_pro(payload: dict):
    try:
        import razorpay
    except ImportError:
        raise HTTPException(
            503, {"error": "razorpay_not_installed", "message": "razorpay package not installed."}
        )
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(
            503,
            {
                "error": "razorpay_not_configured",
                "message": "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Render environment.",
            },
        )
    try:
        client = razorpay.Client(auth=(key_id, key_secret))
        order = client.order.create(
            {
                "amount": 99900,
                "currency": "INR",
                "payment_capture": 1,
                "notes": {"product": "QuantProof Pro — 30 days"},
            }
        )
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": key_id,
        }
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@router.post("/report/webhook-pro")
async def payment_webhook_pro(request: Request):
    import hmac
    import time

    body = await request.body()
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    payment_id = data.get("razorpay_payment_id", "")
    order_id = data.get("razorpay_order_id", "")
    signature = data.get("razorpay_signature", "")
    key_secret = _razorpay_secret()
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields")
    expected = hmac.new(key_secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature")
    current_hour = int(time.time() // 3600)
    # Pro token — 30-day expiry stored in _PRO_TOKENS
    expiry = time.time() + 30 * 24 * 3600
    token = f"PRO-{hashlib.sha256(f'pro:{current_hour}:{key_secret}:{payment_id}'.encode()).hexdigest()[:36]}"
    _PRO_TOKENS[token] = expiry
    REPORT_ACCESS_TOKENS.add(token)
    _save_pro_tokens()
    _save_tokens()
    return {"status": "ok", "token": token, "expires_in_days": 30}


# =========================================================
# PADDLE INTERNATIONAL PAYMENTS
# =========================================================


_PADDLE_API = "https://api.paddle.com/transactions"
_PADDLE_PRICE_ENV = {
    "lite": "PADDLE_PRICE_LITE",
    "report": "PADDLE_PRICE_REPORT",
    "portfolio": "PADDLE_PRICE_PORTFOLIO",
    "pro": "PADDLE_PRICE_PRO",
}


def _create_paddle_checkout(tier: str, user_id: str = "anonymous") -> dict:
    """Create a Paddle Billing transaction and return its hosted checkout link.

    ``checkout.url`` must be an approved domain in Paddle (or omitted to use the
    default payment link). Paddle appends ``?_ptxn=<txn_id>``; the frontend opens
    the overlay for that transaction and polls /report/paddle-token afterwards.
    """
    import requests

    api_key = os.environ.get("PADDLE_API_KEY", "")
    price_id = os.environ.get(_PADDLE_PRICE_ENV[tier], "")
    if not api_key or not price_id:
        raise HTTPException(503, "International payments are not configured.")
    checkout_url = os.environ.get("PADDLE_CHECKOUT_URL", "https://quantproof.net/")
    payload = {
        "items": [{"price_id": price_id, "quantity": 1}],
        "checkout": {"url": checkout_url},
        "custom_data": {"user_id": str(user_id or "anonymous")[:64], "tier": tier},
    }
    try:
        res = requests.post(
            _PADDLE_API,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"[PADDLE] tier={tier} request failed: {type(e).__name__}")
        raise HTTPException(502, "Payment provider unreachable. Please try again.")
    try:
        data = res.json()
    except ValueError:
        data = {}
    url = (data.get("data") or {}).get("checkout", {}).get("url") if isinstance(data, dict) else None
    if res.status_code >= 300 or not url:
        err = (data.get("error") or {}) if isinstance(data, dict) else {}
        print(
            f"[PADDLE] tier={tier} status={res.status_code} code={err.get('code')} "
            f"detail={err.get('detail')} request_id={(data.get('meta') or {}).get('request_id') if isinstance(data, dict) else None}"
        )
        raise HTTPException(
            502, "International checkout is temporarily unavailable. Please use Razorpay or try later."
        )
    return {"checkout_url": url, "transaction_id": data["data"].get("id", "")}


@router.post("/report/create-checkout-paddle-lite")
async def create_checkout_paddle_lite(payload: dict):
    return _create_paddle_checkout("lite", (payload or {}).get("user_id", "anonymous"))


@router.post("/report/create-checkout-paddle-report")
async def create_checkout_paddle_report(payload: dict):
    return _create_paddle_checkout("report", (payload or {}).get("user_id", "anonymous"))


@router.post("/report/create-checkout-paddle-portfolio")
async def create_checkout_paddle_portfolio(payload: dict):
    return _create_paddle_checkout("portfolio", (payload or {}).get("user_id", "anonymous"))


@router.post("/report/create-checkout-paddle-pro")
async def create_checkout_paddle_pro(payload: dict):
    return _create_paddle_checkout("pro", (payload or {}).get("user_id", "anonymous"))


_PADDLE_IPS_URL = "https://api.paddle.com/ips"
_PADDLE_IP_CACHE: dict = {"nets": None, "fetched": 0.0}
_PADDLE_IP_TTL_S = 6 * 3600


def _paddle_webhook_networks() -> list | None:
    """Paddle's published webhook source CIDRs (cached). None if unavailable."""
    import ipaddress
    import time

    import requests

    now = time.time()
    if _PADDLE_IP_CACHE["nets"] is not None and now - _PADDLE_IP_CACHE["fetched"] < _PADDLE_IP_TTL_S:
        return _PADDLE_IP_CACHE["nets"]
    try:
        res = requests.get(_PADDLE_IPS_URL, timeout=5)
        cidrs = res.json()["data"]["ipv4_cidrs"]
        nets = [ipaddress.ip_network(c, strict=False) for c in cidrs]
        if not nets:
            raise ValueError("empty list")
        _PADDLE_IP_CACHE.update(nets=nets, fetched=now)
    except Exception as e:  # noqa: BLE001 - never let the allowlist break webhooks
        print(f"[PADDLE] IP list fetch failed ({type(e).__name__}); relying on signature only")
        return _PADDLE_IP_CACHE["nets"]  # stale list if we have one, else None
    return _PADDLE_IP_CACHE["nets"]


def _webhook_client_ip(request: Request) -> str:
    """Client IP behind Render's proxy.

    Render's load balancer appends the address it saw to X-Forwarded-For, so the
    right-most entry is the one a caller cannot forge (left-most entries can be
    set by anyone). Set PADDLE_TRUSTED_PROXY_HOPS if more trusted proxies sit in front.
    """
    xff = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    hops = max(int(os.environ.get("PADDLE_TRUSTED_PROXY_HOPS", "1") or 1), 1)
    if len(xff) >= hops:
        return xff[-hops]
    return request.client.host if request.client else ""


def _paddle_ip_allowed(request: Request) -> bool:
    """IP allowlist on top of signature verification.

    PADDLE_IP_ALLOWLIST: "log" (default) records mismatches only, "enforce"
    rejects them, "off" skips the check. If Paddle's IP list can't be fetched
    the check is skipped; the signature still has to be valid.
    """
    import ipaddress

    mode = os.environ.get("PADDLE_IP_ALLOWLIST", "log").strip().lower()
    if mode == "off":
        return True
    nets = _paddle_webhook_networks()
    if not nets:
        return True
    ip = _webhook_client_ip(request)
    try:
        ok = any(ipaddress.ip_address(ip) in n for n in nets)
    except ValueError:
        ok = False
    if not ok:
        print(f"[PADDLE] webhook from non-Paddle IP {ip!r} (mode={mode})")
        return mode != "enforce"
    return True


def _verify_paddle_signature(body: bytes, header: str, secret: str, tolerance_s: int = 3600) -> bool:
    """Paddle Billing signature: header ``ts=<unix>;h1=<hex hmac-sha256(secret, f"{ts}:{body}")>``."""
    import hmac
    import time

    parts = dict(p.split("=", 1) for p in header.split(";") if "=" in p)
    ts, h1 = parts.get("ts", ""), parts.get("h1", "")
    if not ts.isdigit() or not h1:
        return False
    if abs(time.time() - int(ts)) > tolerance_s:
        return False
    expected = hmac.new(secret.encode(), f"{ts}:".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, h1)


@router.get("/report/paddle-token")
async def get_paddle_token(txn_id: str = ""):
    """Frontend polls this after Paddle payment to get the token."""
    import time

    if not txn_id:
        return {"status": "pending"}

    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    current_hour = int(time.time() // 3600)

    for tier, prefix in [
        ("report", TOKEN_REPORT),
        ("lite", TOKEN_LITE),
        ("portfolio", TOKEN_PORTFOLIO),
        ("pro", TOKEN_PRO),
    ]:
        for hour_offset in (0, -1, 1):
            candidate = f"{prefix}{hashlib.sha256(f'{tier}:{current_hour + hour_offset}:{key_secret}:{txn_id}'.encode()).hexdigest()[:32]}"
            if candidate in REPORT_ACCESS_TOKENS:
                return {"status": "ready", "token": candidate, "tier": tier}

    return {"status": "pending"}


@router.post("/report/webhook-paddle")
async def paddle_webhook(request: Request):
    import time

    body = await request.body()

    webhook_secret = os.environ.get("PADDLE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Paddle webhook not configured")
    if not _paddle_ip_allowed(request):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not _verify_paddle_signature(body, request.headers.get("Paddle-Signature", ""), webhook_secret):
        raise HTTPException(status_code=400, detail="Invalid Paddle signature")

    try:
        event = json.loads(body)
    except Exception:
        return {"status": "ignored"}

    event_type = event.get("event_type", "")
    if event_type not in ("transaction.completed", "transaction.paid"):
        return {"status": "ignored"}

    txn = event.get("data", {})
    txn_id = txn.get("id", "")
    items = txn.get("items", [])
    price_id = items[0].get("price", {}).get("id", "") if items else ""

    PRICE_LITE = os.environ.get("PADDLE_PRICE_LITE", "")
    PRICE_PORTFOLIO = os.environ.get("PADDLE_PRICE_PORTFOLIO", "")
    PRICE_PRO = os.environ.get("PADDLE_PRICE_PRO", "")

    if price_id == PRICE_LITE:
        tier = "lite"
        token_prefix = TOKEN_LITE
    elif price_id == PRICE_PORTFOLIO:
        tier = "portfolio"
        token_prefix = TOKEN_PORTFOLIO
    elif price_id == PRICE_PRO:
        tier = "pro"
        token_prefix = TOKEN_PRO
    else:
        tier = "report"
        token_prefix = TOKEN_REPORT

    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "quantproof-fallback")
    current_hour = int(time.time() // 3600)
    token_now = f"{token_prefix}{hashlib.sha256(f'{tier}:{current_hour}:{key_secret}:{txn_id}'.encode()).hexdigest()[:32]}"
    token_next = f"{token_prefix}{hashlib.sha256(f'{tier}:{current_hour + 1}:{key_secret}:{txn_id}'.encode()).hexdigest()[:32]}"

    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)

    if tier == "pro":
        expiry = time.time() + 30 * 24 * 3600
        _PRO_TOKENS[token_now] = expiry
        _PRO_TOKENS[token_next] = expiry
        _save_pro_tokens()

    _save_tokens()
    print(f"[PADDLE] tier={tier} txn={txn_id} token={token_now[:12]}...")
    return {"status": "ok", "tier": tier, "txn_id": txn_id}


@router.get("/report/token-status")
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
    api_key: str
    secret_key: str
    mode: str = "paper"
    enhanced: bool = False
    validation_hash: str = ""
    validation_data: dict = {}  # For TV mode - bypass cache


@router.post("/report/generate-broker")
async def generate_broker_report(
    payload: BrokerReportRequest,
    x_report_token: str = Header(default=""),
):
    print(f"[DEBUG] Broker PDF payload: {payload}")
    print(f"[DEBUG] Broker PDF token: {x_report_token[:10] if x_report_token else 'None'}...")

    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok:
            return True
        if _is_pro_token(token):
            return True  # Pro = unlimited PDFs
        if token in REPORT_ACCESS_TOKENS:
            return True
        if token.startswith("cs_"):
            return False
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(
            status_code=402,
            detail={
                "error": "payment_required",
                "message": "Purchase required to download the full analysis report.",
                "price_inr": REPORT_PRICE_INR,
                "upgrade_url": "https://quantproof.net/pricing",
            },
        )

    if not payload.validation_hash:
        print("[DEBUG] Broker PDF - MISSING validation_hash")
        raise HTTPException(
            400, {"error": "missing_validation_hash", "message": "Validation hash is required"}
        )

    # Special handling for TradingView mode - use sent data instead of cache
    if payload.mode == "tradingview":
        print("[DEBUG] Broker PDF - TradingView mode detected")
        if payload.validation_data:
            print("[DEBUG] Broker PDF - Using sent TV validation data")
            cached_vr = payload.validation_data
        else:
            print("[DEBUG] Broker PDF - No TV data sent, checking cache")
            cached_vr = _get_cached_validation(payload.validation_hash)
            if not cached_vr:
                raise HTTPException(
                    400,
                    {
                        "error": "tv_data_missing",
                        "message": "TradingView validation data missing - please re-run validation then download PDF immediately.",
                    },
                )
    elif payload.mode == "ibkr":
        print("[DEBUG] Broker PDF - IBKR mode detected")
        if payload.validation_data:
            print("[DEBUG] Broker PDF - Using sent IBKR validation data")
            cached_vr = payload.validation_data
        else:
            print("[DEBUG] Broker PDF - No IBKR data sent, checking cache")
            cached_vr = _get_cached_validation(payload.validation_hash)
            if not cached_vr:
                raise HTTPException(
                    400,
                    {
                        "error": "ibkr_data_missing",
                        "message": "IBKR validation data missing - please re-run validation then download PDF immediately.",
                    },
                )
    else:
        print(f"[DEBUG] Broker PDF - Checking cache for validation_hash: '{payload.validation_hash}'")
        print(
            f"[DEBUG] Broker PDF - Cache keys available: {list(VALIDATION_RESULT_CACHE.keys())[:5]}..."
        )  # Show first 5 keys
        cached_vr = _get_cached_validation(payload.validation_hash)
        if not cached_vr:
            print("[DEBUG] Broker PDF - Cache miss for validation_hash")
            raise HTTPException(
                400,
                {
                    "error": "cache_expired",
                    "message": "Validation result expired — please re-run your validation then download the PDF immediately.",
                },
            )
    print("[DEBUG] Broker PDF - Cache hit, proceeding with PDF generation")

    if not cached_vr.get("ai_plan"):
        try:

            class _CachedCheck:
                def __init__(self, d):
                    self.__dict__.update(d)

            class _CachedReport:
                def __init__(self, vr):
                    self.score = vr["fundable_score"]
                    self.grade = vr["grade"]
                    self.sharpe = vr["sharpe"]
                    self.max_drawdown = vr["max_drawdown"]
                    self.win_rate = vr["win_rate"]
                    self.total_trades = vr["total_trades"]
                    self.checks = [_CachedCheck(c) for c in vr.get("checks", [])]
                    self.audit_flags = []
                    self.prop_firm_compliance = vr.get("prop_firm_compliance")

            cached_vr["ai_plan"] = await generate_ai_plan(
                _CachedReport(cached_vr), cached_vr.get("date_range", "")
            )
        except Exception as _e:
            issues = cached_vr.get("top_issues", [])
            cached_vr["ai_plan"] = (
                {
                    "summary": "Top improvements: " + "; ".join(issues[:3])
                    if issues
                    else "See failed checks above.",
                    "fixes": [],
                    "encouragement": "",
                }
                if issues
                else None
            )

    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available.")
    try:
        async with _PDF_SEMAPHORE:
            pdf_bytes = _generate_pdf_report(cached_vr)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")
    score = int(cached_vr["fundable_score"])
    grade_str = cached_vr["grade"].split("—")[0].strip().replace(" ", "")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"'
        },
    )

    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok:
            return True
        if _is_pro_token(token):
            return True  # Pro = unlimited PDFs
        if token in REPORT_ACCESS_TOKENS:
            return True
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(402, {"error": "payment_required", "message": "Purchase required."})

    api_key = payload.api_key.strip()
    secret_key = payload.secret_key.strip()
    if not api_key or not secret_key:
        raise HTTPException(400, "API key and secret required.")

    is_paper = payload.mode == "paper"
    base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"

    try:
        import httpx

        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "accept": "application/json",
        }

        @retry_on_failure(max_retries=3, delay=1.0)
        async def _fetch_alpaca_account_v2(client, base_url, headers):
            return await client.get(f"{base_url}/v2/account", headers=headers)

        @retry_on_failure(max_retries=3, delay=1.0)
        async def _fetch_alpaca_orders_v2(client, base_url, headers):
            return await client.get(
                f"{base_url}/v2/orders",
                headers=headers,
                params={"status": "closed", "limit": 500, "direction": "desc"},
            )

        async with httpx.AsyncClient(timeout=20) as client:
            acct_resp = await _fetch_alpaca_account_v2(client, base_url, headers)
            if not acct_resp.is_success:
                raise HTTPException(401, "Invalid Alpaca credentials.")
            orders_resp = await _fetch_alpaca_orders_v2(client, base_url, headers)
        raw_orders = orders_resp.json() if orders_resp.is_success else []
        orders = raw_orders if isinstance(raw_orders, list) else []

        api_key = secret_key = "[DISCARDED]"
        headers = {}

        df, parse_flags = pair_alpaca_orders(orders)

        _orphan_flag = next((f for f in parse_flags if "orphan" in f.lower() or "predates" in f.lower()), "")
        _open_flag = next((f for f in parse_flags if "open position" in f.lower()), "")

        if len(df) < 10:
            _detail_lines = []
            if _orphan_flag:
                _detail_lines.append(
                    "Some sells were excluded because their matching buy happened before the visible history window."
                )
            if _open_flag:
                _detail_lines.append("Some positions are still open (no closing sell yet).")
            _detail = " ".join(_detail_lines) if _detail_lines else ""
            raise HTTPException(
                400,
                {
                    "error": "insufficient_trades",
                    "message": (f"Only {len(df)} completed round-trip trade(s) found - need 10. {_detail}"),
                    "trades_found": len(df),
                    "trades_needed": 10,
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Alpaca fetch failed: {str(e)[:200]}")

    try:
        _bm_type = get_benchmark_type(df, None)
        _bm, _bm_note = get_nifty_benchmark(df) if _bm_type == "nifty" else get_spy_benchmark(df)
        n_trials = max(1, min(50, int(getattr(payload, "n_trials", 10))))  # cap: already parsed from Form
        validator = QuantProofValidator(df, benchmark_returns=_bm, benchmark_type=_bm_type, n_trials=n_trials)
        report = validator.run()
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored = [c for c in report.checks if c.category != "Plausibility"]
    top_issues = [c.fix for c in sorted(scored, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]]

    date_range = "No date data"
    if "date" in df.columns:
        try:
            date_range = f"{df['date'].dropna().min().date()} to {df['date'].dropna().max().date()}"
        except Exception:
            pass

    vr = {
        "fundable_score": min(100.0, round(report.score, 1)),
        "grade": report.grade,
        "sharpe": round(report.sharpe, 2),
        "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1),
        "total_trades": report.total_trades,
        "date_range": date_range,
        "n_trials": n_trials,
        "is_verified": True,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "score": c.score,
                "value": c.value,
                "insight": c.insight,
                "fix": c.fix,
                "category": c.category,
            }
            for c in report.checks
        ],
        "crash_sims": [
            {
                "crash_name": s.crash_name,
                "year": s.year,
                "description": s.description,
                "market_drop": s.market_drop,
                "strategy_drop": s.strategy_drop,
                "survived": s.survived,
                "emotional_verdict": s.emotional_verdict,
            }
            for s in report.crash_sims
        ],
        "top_issues": top_issues,
        "top_strengths": top_strengths,
        "audit_flags": report.audit_flags + parse_flags + ([_bm_note] if _bm_note else []),
        "equity_curve": getattr(report, "equity_curve", None),
        "sebi_checks": _compute_sebi_checks(df, None, ""),
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

    score = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    return FastAPIResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"'
        },
    )
