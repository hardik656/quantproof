"""
QuantProof — FastAPI Backend v1.3
Final Boss validator — matches ValidationReport + ValidationDashboard
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io

from validator import QuantProofValidator, ValidationDashboard

app = FastAPI(title="QuantProof API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# RESPONSE MODELS
# =========================================================

class CheckResultResponse(BaseModel):
    name: str
    passed: bool
    score: float
    value: str
    insight: str
    fix: str
    category: str

class CrashSimResponse(BaseModel):
    crash_name: str
    year: str
    description: str
    market_drop: float
    strategy_drop: float
    survived: bool
    emotional_verdict: str

class ValidationResponse(BaseModel):
    # Core score
    fundable_score: float
    grade: str
    summary: str
    # Trade stats
    total_trades: int
    profitable_trades: int
    date_range: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    # Checks
    checks: List[CheckResultResponse]
    crash_sims: List[CrashSimResponse]
    # Insights
    top_issues: List[str]
    top_strengths: List[str]
    # Institutional
    validation_hash: str
    validation_date: str
    engine_version: str
    audit_flags: List[str]
    plausibility_summary: str
    assumptions: List[str]
    # Dashboard data (optional — for future chart rendering)
    dashboard: Optional[Dict[str, Any]] = None

# =========================================================
# CSV PARSER
# =========================================================

def smart_parse_csv(contents: bytes) -> pd.DataFrame:
    raw = contents.decode("utf-8", errors="ignore")

    # Primary parse — look for known pnl column names
    try:
        df = pd.read_csv(io.StringIO(raw))
        pnl_names = ["pnl", "profit", "return", "gain", "pl", "p&l", "net", "alpha"]
        for col in df.columns:
            if any(p in col.lower() for p in pnl_names):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) >= 5:
                    if vals.abs().mean() > 1.0:
                        vals = vals / 10000.0
                    vals = vals.clip(-0.50, 0.50)
                    result = pd.DataFrame({"pnl": vals.values})
                    # Preserve date column if present
                    date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "dt"])]
                    if date_cols:
                        result["date"] = df[date_cols[0]].values[:len(vals)]
                    # Preserve volume column if present (for market impact check)
                    vol_cols = [c for c in df.columns if "volume" in c.lower() or "vol" == c.lower()]
                    if vol_cols:
                        result["volume"] = pd.to_numeric(df[vol_cols[0]], errors="coerce").values[:len(vals)]
                    # Preserve regime column if present
                    reg_cols = [c for c in df.columns if "regime" in c.lower()]
                    if reg_cols:
                        result["regime"] = df[reg_cols[0]].values[:len(vals)]
                    return result
    except Exception:
        pass

    # Fallback — flexible parse ignoring bad lines
    try:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        max_cols = max(len(l.split(",")) for l in lines[1:] if l)
        df = pd.read_csv(
            io.StringIO(raw),
            names=[f"col_{i}" for i in range(max_cols)],
            skiprows=1,
            on_bad_lines="skip",
            engine="python"
        )
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and (vals > 0).any() and (vals < 0).any():
                if vals.abs().mean() > 1.0:
                    vals = vals / 10000.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values})
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                if vals.abs().mean() > 1.0:
                    vals = vals / 10000.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values})
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError("No valid PnL column found. Make sure your CSV has a 'pnl' or 'profit' column.")

# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "status": "QuantProof API is live",
        "version": "1.3.0",
        "checks": "30 institutional checks + 3 crash simulations",
        "new_in_v1.3": ["CVaR/Expected Shortfall", "Fractional Kelly + Ruin Probability",
                        "Market Impact (Almgren-Chriss)", "Deflated Sharpe",
                        "Purged Walk-Forward CV", "HMM Regime Detection",
                        "Capacity Analysis", "Interactive Dashboard API"]
    }

@app.post("/validate", response_model=ValidationResponse)
async def validate(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await file.read()

    try:
        df = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df) < 5:
        raise HTTPException(status_code=400, detail=f"Only found {len(df)} valid trades. Need at least 5.")

    try:
        validator = QuantProofValidator(df)
        report = validator.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

    r = report

    # Build top issues and strengths — exclude Plausibility (informational only)
    scored_checks = [c for c in r.checks if c.category != "Plausibility"]
    top_issues    = [c.fix  for c in sorted(scored_checks, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(scored_checks, key=lambda x: x.score, reverse=True)[:3]]

    # Date range
    date_range = "No date data"
    if "date" in validator.df.columns:
        try:
            date_range = f"{validator.df['date'].min().date()} to {validator.df['date'].max().date()}"
        except Exception:
            pass

    # Dynamic summary — uses actual check count
    check_count = len([c for c in r.checks if c.category != "Plausibility"])
    summary = (
        f"QuantProof analyzed {r.total_trades} trades across {check_count} institutional checks "
        f"+ 3 crash simulations. Fundable Score: {r.score}/100 ({r.grade.split('—')[0].strip()}). "
        f"{'Core risk management needs work.' if r.score < 60 else 'Edge shows real promise — focus on execution costs.'}"
    )

    # Generate dashboard data
    try:
        dashboard_obj = ValidationDashboard(validator, report)
        dashboard_data = dashboard_obj.generate_interactive_report()
    except Exception:
        dashboard_data = None

    # Only show non-Plausibility checks to frontend by default
    # Plausibility checks are surfaced via audit_flags instead
    visible_checks = [c for c in r.checks if c.category != "Plausibility"]

    return ValidationResponse(
        fundable_score=r.score,
        grade=r.grade,
        summary=summary,
        checks=[CheckResultResponse(**vars(c)) for c in visible_checks],
        crash_sims=[CrashSimResponse(**vars(s)) for s in r.crash_sims],
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
        audit_flags=r.audit_flags,
        plausibility_summary=r.plausibility_summary,
        assumptions=r.assumptions,
        dashboard=dashboard_data,
    )

@app.get("/sample-csv")
def sample_csv():
    return {
        "format": "CSV with these columns",
        "required": ["pnl (profit/loss per trade as decimal, e.g. 0.015 = 1.5%)"],
        "optional": ["date", "ticker", "side (long/short)", "regime (BULL/BEAR/CONSOLIDATION/TRANSITION)", "volume"],
        "example_row": {
            "date": "2024-01-15",
            "ticker": "AAPL",
            "pnl": 0.0145,
            "side": "long",
            "regime": "BULL"
        },
        "note": "Add regime column for +15 regime coverage score. Add volume for market impact analysis."
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.3.0"}


# =========================================================
# PDF REPORT — PAYWALL ENDPOINT
# =========================================================

from fastapi import Header, Request
from fastapi.responses import Response as FastAPIResponse
import os

# generate_report imported lazily — server starts even without it
try:
    from generate_report import generate_pdf_report as _generate_pdf_report
    _PDF_AVAILABLE = True
except ImportError:
    _generate_pdf_report = None
    _PDF_AVAILABLE = False

# Token gate — populated by Stripe webhook after successful payment
REPORT_ACCESS_TOKENS = set()

# Stripe config — set these as Render environment variables
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY",     "sk_live_REPLACE")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID",       "price_REPLACE")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_REPLACE")
REPORT_PRICE_USD      = 9


@app.post("/report/generate")
async def generate_report_endpoint(
    file: UploadFile = File(...),
    x_report_token: str = Header(default="")
):
    """
    Paid endpoint — returns a full PDF analysis report.
    Gate: x-report-token header must match a paid session token.
    For dev/testing: token "dev-free" bypasses the gate.
    """

    # Token check
    is_paid = (x_report_token in REPORT_ACCESS_TOKENS or
               x_report_token == "dev-free")
    if not is_paid:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "payment_required",
                "message": "Purchase required to download the full analysis report.",
                "price_usd": REPORT_PRICE_USD,
                "stripe_price_id": STRIPE_PRICE_ID,
                "upgrade_url": "https://quantproof-frontend.onrender.com/#upgrade"
            }
        )

    # Parse + validate
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")

    contents = await file.read()
    try:
        df = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(df) < 5:
        raise HTTPException(400, f"Only {len(df)} trades found. Need at least 5.")

    try:
        validator = QuantProofValidator(df)
        report    = validator.run()
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    # Build response dict (same shape as /validate)
    scored  = [c for c in report.checks if c.category != "Plausibility"]
    date_range = "No date data"
    if "date" in validator.df.columns:
        try:
            date_range = f"{validator.df['date'].min().date()} to {validator.df['date'].max().date()}"
        except Exception:
            pass

    vr = {
        "fundable_score": report.score,
        "grade":          report.grade,
        "sharpe":         round(report.sharpe, 2),
        "max_drawdown":   round(report.max_drawdown, 4),
        "win_rate":       round(report.win_rate, 1),
        "total_trades":   report.total_trades,
        "date_range":     date_range,
        "checks": [
            {"name": c.name, "passed": c.passed, "score": c.score,
             "value": c.value, "insight": c.insight, "fix": c.fix,
             "category": c.category}
            for c in report.checks
        ],
        "crash_sims": [
            {"crash_name": s.crash_name, "year": s.year,
             "description": s.description, "market_drop": s.market_drop,
             "strategy_drop": s.strategy_drop, "survived": s.survived,
             "emotional_verdict": s.emotional_verdict}
            for s in report.crash_sims
        ],
        "top_issues":    [c.fix  for c in sorted(scored, key=lambda x: x.score)[:4]],
        "top_strengths": [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]],
    }

    try:
        if not _PDF_AVAILABLE:
            raise HTTPException(503, "PDF module not available. Push generate_report.py to backend/.")
        pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    score     = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    filename  = f"QuantProof_Report_{grade_str}_{score}.pdf"

    return FastAPIResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/report/create-checkout")
async def create_checkout(payload: dict):
    """
    Creates a Stripe Checkout session for the $9 report.
    Frontend redirects user to session.url.
    On success, Stripe webhook calls /report/webhook to issue a token.
    """
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode="payment",
            success_url="https://quantproof-frontend.onrender.com/#report-success?token={CHECKOUT_SESSION_ID}",
            cancel_url="https://quantproof-frontend.onrender.com/#report-cancel",
            metadata={"product": "report_download"}
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except ImportError:
        # stripe not installed — return mock for testing
        return {
            "checkout_url": "https://buy.stripe.com/REPLACE_WITH_REAL_PAYMENT_LINK",
            "session_id": "cs_mock_dev"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/report/webhook")
async def stripe_webhook(request):
    """
    Stripe sends POST here after successful payment.
    Issues a one-time download token tied to the checkout session.
    """
    import hashlib, json
    body = await request.body()

    # Verify Stripe signature to prevent spoofed webhooks
    try:
        import stripe as _stripe
        _stripe.api_key = STRIPE_SECRET_KEY
        sig_header = request.headers.get("stripe-signature", "")
        if STRIPE_WEBHOOK_SECRET and not STRIPE_WEBHOOK_SECRET.startswith("whsec_REPLACE"):
            event = _stripe.Webhook.construct_event(body, sig_header, STRIPE_WEBHOOK_SECRET)
        else:
            event = json.loads(body)   # dev mode — no signature check
    except Exception as e:
        raise HTTPException(400, f"Webhook signature invalid: {e}")

    try:
        if event.get("type") == "checkout.session.completed":
            session_id = event["data"]["object"]["id"]
            token = hashlib.sha256(f"{session_id}:quantproof:secret".encode()).hexdigest()[:32]
            REPORT_ACCESS_TOKENS.add(token)
            return {"status": "ok", "token": token}
    except Exception as e:
        raise HTTPException(400, str(e))

    return {"status": "ignored"}
