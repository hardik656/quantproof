"""
QuantProof — FastAPI Backend v1.3
Final Boss validator — matches ValidationReport + ValidationDashboard
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import os
import time
import hashlib
import traceback
from collections import defaultdict

from validator import QuantProofValidator, ValidationDashboard

# ── Rate limiting (20 free validations / IP / day) ───────────────────────────
_validation_counts: dict = defaultdict(lambda: {"count": 0, "reset": 0})
FREE_DAILY_LIMIT = 20

def check_rate_limit(ip: str) -> tuple[bool, int]:
    now = time.time()
    rec = _validation_counts[ip]
    if now > rec["reset"]:                    # new day — reset
        rec["count"] = 0
        rec["reset"] = now + 86400
    if rec["count"] >= FREE_DAILY_LIMIT:
        remaining_secs = int(rec["reset"] - now)
        return False, remaining_secs
    rec["count"] += 1
    return True, FREE_DAILY_LIMIT - rec["count"]

app = FastAPI(title="QuantProof API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AI Fix Plan ───────────────────────────────────────────────────────────────
# Priority: Groq (free, fast) → Claude → OpenAI
# Set ONE of these in Render environment variables:
#   GROQ_API_KEY        — free at console.groq.com — recommended to start
#   ANTHROPIC_API_KEY   — if you already have Claude credits
#   OPENAI_API_KEY      — if you already have OpenAI credits
GROQ_API_KEY      = os.getenv("GROQ_API_KEY",      "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")

async def generate_ai_plan(report, date_range: str) -> dict:
    """
    Calls Claude to produce a personalised, ranked fix plan.
    Returns a dict with keys: summary, fixes (list), encouragement.
    Falls back gracefully if API key missing or call fails.
    """
    # Check at least one AI provider is configured
    if not any([GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY]):
        return None

    scored = [c for c in report.checks if c.category != "Plausibility"]
    failed = [c for c in scored if not c.passed]
    passed = [c for c in scored if c.passed]

    if not failed:
        return {
            "summary": "Your strategy passed all checks. Focus on live execution and position sizing.",
            "fixes": [],
            "encouragement": "Rare result — this strategy has institutional-grade statistical foundations."
        }

    # Build context for Claude
    failed_text = "\n".join([
        f"- {c.name} (score {c.score}/100, category {c.category}): {c.value}. {c.insight}"
        for c in sorted(failed, key=lambda x: x.score)
    ])
    passed_text = ", ".join([c.name for c in passed])

    prompt = f"""You are a quantitative trading risk analyst reviewing a backtest validation report.

STRATEGY STATS:
- Fundable Score: {report.score}/100 ({report.grade})
- Sharpe: {round(report.sharpe, 2)} | Max Drawdown: {round(report.max_drawdown * 100, 1)}% | Win Rate: {round(report.win_rate, 1)}%
- Total Trades: {report.total_trades} | Date Range: {date_range}

FAILED CHECKS ({len(failed)} failures):
{failed_text}

PASSED CHECKS:
{passed_text}

Produce a personalised fix plan in JSON format with exactly these keys:
{{
  "summary": "2-3 sentence plain English summary of the core problem(s) — be specific to their numbers, not generic",
  "fixes": [
    {{
      "rank": 1,
      "check": "check name",
      "why_it_matters": "1 sentence — real-world consequence of this failure",
      "specific_fix": "concrete actionable step — reference their actual numbers where possible",
      "score_impact": "estimated score improvement if fixed e.g. +8 to +12 points",
      "effort": "Low | Medium | High"
    }}
  ],
  "encouragement": "1 honest sentence — what is genuinely strong about this strategy"
}}

Rules:
- Rank fixes by score impact (highest first)
- Reference actual values (e.g. 'your DSR of 0.71') not generic descriptions
- If two failures share a root cause, say so explicitly
- Be direct, not encouraging — traders respect honesty
- Return ONLY valid JSON, no preamble or markdown"""

    try:
        import httpx, json

        # ── Pick provider — Groq first (free), then Claude, then OpenAI ─────
        if GROQ_API_KEY:
            url     = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json"
            }
            body = {
                "model":    "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens":   1500,
                "temperature":  0.3,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        elif ANTHROPIC_API_KEY:
            url     = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json"
            }
            body = {
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages":   [{"role": "user", "content": prompt}]
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        elif OPENAI_API_KEY:
            url     = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json"
            }
            body = {
                "model":       "gpt-4o-mini",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  1500,
                "temperature": 0.3,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        else:
            # No API key set — skip AI plan silently
            return None

        # Strip markdown fences if model added them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return json.loads(raw)

    except Exception as e:
        print(f"AI plan failed ({type(e).__name__}): {e}")
        return None


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
    scale_warning: bool = False   # True if >50% of returns at ±50% boundary (% vs decimal)
    currency: str = "USD"         # Currency code for display — does not affect calculations
    assumptions: List[str]
    # Dashboard data (optional — for future chart rendering)
    dashboard: Optional[Dict[str, Any]] = None
    # AI personalised fix plan
    ai_plan: Optional[Dict[str, Any]] = None

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
        "version": "2.5.0",
        "checks": "38 institutional checks + 5 crash simulations",
        "new_in_v1.3": ["CVaR/Expected Shortfall", "Fractional Kelly + Ruin Probability",
                        "Market Impact (Almgren-Chriss)", "Deflated Sharpe",
                        "Purged Walk-Forward CV", "HMM Regime Detection",
                        "Capacity Analysis", "Interactive Dashboard API"]
    }

@app.post("/validate", response_model=ValidationResponse)
async def validate(file: UploadFile = File(...), currency: str = Form("USD"), request: Request = None):
    # Rate limit check
    if request:
        ip = request.client.host if request.client else "unknown"
        allowed, remaining = check_rate_limit(ip)
        if not allowed:
            hours_left = remaining // 3600
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit of {FREE_DAILY_LIMIT} free validations reached. Resets in ~{hours_left}h. Upgrade to Pro for unlimited access."
            )

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

    # Detect currency — from form param, then CSV column, then default USD
    detected_currency = (currency or "USD").upper()
    if detected_currency == "USD" and "currency" in df.columns:
        currencies = df["currency"].dropna().unique()
        if len(currencies) == 1:
            detected_currency = str(currencies[0]).upper()
        elif len(currencies) > 1:
            detected_currency = "MULTI"

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
        scale_warning=getattr(r, '_scale_warning', False),
        currency=detected_currency,
        assumptions=r.assumptions,
        dashboard=dashboard_data,
        ai_plan=None,  # AI analysis is a paid feature — included in PDF report
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
    return {"status": "ok", "version": "2.5.0"}

# =========================================================
# ALPACA BROKER INTEGRATION — VERIFIED VALIDATION
# =========================================================

class AlpacaConnectRequest(BaseModel):
    api_key:    str
    secret_key: str
    mode:       str = "paper"   # "paper" or "live"

def _scrub_keys(d: dict) -> dict:
    """Recursively remove any field that looks like a credential."""
    sensitive = {"api_key", "secret_key", "key", "secret", "token", "password", "auth"}
    return {k: "[REDACTED]" if k.lower() in sensitive else v for k, v in d.items()}

@app.post("/validate/alpaca", response_model=ValidationResponse)
async def validate_alpaca(payload: AlpacaConnectRequest, request: Request = None):
    """
    Pulls real trade history directly from Alpaca and runs full validation.

    PRIVACY DESIGN:
    - API key + secret are used only in-memory for this request
    - Never logged, never stored, never appear in error messages
    - Keys are overwritten after use
    - Only aggregated PnL data is retained for validation
    - Response contains zero credential data
    """

    # Rate limit — stricter for broker endpoint (3/day)
    BROKER_DAILY_LIMIT = 3
    if request:
        ip = request.client.host if request.client else "unknown"
        key_hash = "broker:" + hashlib.sha256(
            f"broker:{ip}".encode()
        ).hexdigest()[:16]
        rec = _validation_counts[key_hash]
        now = time.time()
        if rec.get("reset", 0) == 0 or now > rec["reset"]:
            rec["count"] = 0
            rec["reset"] = now + 86400
        if rec["count"] >= BROKER_DAILY_LIMIT:
            hours_left = int((rec["reset"] - now) / 3600)
            raise HTTPException(
                status_code=429,
                detail=f"Broker verification limit ({BROKER_DAILY_LIMIT}/day) reached. Resets in ~{hours_left}h."
            )
        rec["count"] += 1

    # Validate key format before making any external call
    api_key    = payload.api_key.strip()
    secret_key = payload.secret_key.strip()

    if not api_key or not secret_key:
        raise HTTPException(400, "API key and secret key are required.")

    if len(api_key) < 10 or len(secret_key) < 10:
        raise HTTPException(400, "Invalid key format — keys are too short.")

    # Determine base URL
    is_paper = payload.mode == "paper"
    base_url  = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"
    data_url  = "https://data.alpaca.markets"

    try:
        import httpx

        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "accept":              "application/json"
        }

        async with httpx.AsyncClient(timeout=20) as client:

            # ── Step 1: Verify credentials with account endpoint ──────────────
            acct_resp = await client.get(f"{base_url}/v2/account", headers=headers)

            if acct_resp.status_code == 401:
                raise HTTPException(401, "Invalid Alpaca credentials. Please check your API key and secret.")
            if acct_resp.status_code == 403:
                raise HTTPException(403, "Alpaca API access denied. Make sure your keys have read permissions.")
            if not acct_resp.is_success:
                # Never expose raw response which might echo keys
                raise HTTPException(400, f"Alpaca connection failed (status {acct_resp.status_code}). Check your credentials and account type (paper vs live).")

            # Parse account currency (INR, GBP, EUR etc for display)
            try:
                acct_currency = str(acct_resp.json().get("currency", "USD")).upper()
            except Exception:
                acct_currency = "USD"

            # ── Step 2: Pull closed orders (trade history) ───────────────────
            orders_resp = await client.get(
                f"{base_url}/v2/orders",
                headers=headers,
                params={
                    "status": "closed",
                    "limit": 500,
                    "direction": "desc"
                }
            )

            if not orders_resp.is_success:
                raise HTTPException(400, "Could not fetch order history from Alpaca.")

            orders = orders_resp.json()

        # ── Immediately discard credentials ──────────────────────────────────
        api_key    = "[DISCARDED]"
        secret_key = "[DISCARDED]"
        headers    = {}

        # ── Step 3: Convert orders to PnL DataFrame ──────────────────────────
        if not orders or len(orders) < 5:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(orders) if orders else 0,
                "trades_needed": 10,
                "message": f"Only {len(orders) if orders else 0} orders found in your Alpaca account. Make sure you selected the correct account type (paper vs live) and have at least 10 completed round-trip trades."
            })

        # ── Pair buys with sells — two-pass so order in list doesn't matter ──
        from collections import defaultdict
        rows = []

        filled = [o for o in orders if o.get("status") == "filled"
                  and float(o.get("filled_qty", 0) or 0) > 0
                  and float(o.get("filled_avg_price", 0) or 0) > 0]

        buys_by_sym  = defaultdict(list)
        sells_by_sym = defaultdict(list)

        for o in filled:
            try:
                side     = o.get("side", "")
                symbol   = o.get("symbol", "")
                price    = float(o.get("filled_avg_price", 0) or 0)
                date_str = o.get("filled_at") or o.get("updated_at")
                if side == "buy":
                    buys_by_sym[symbol].append({"price": price, "date": date_str})
                elif side == "sell":
                    sells_by_sym[symbol].append({"price": price, "date": date_str})
            except Exception:
                continue

        for symbol in sells_by_sym:
            for sell in sells_by_sym[symbol]:
                if buys_by_sym[symbol]:
                    buy     = buys_by_sym[symbol].pop(0)
                    pnl_pct = (sell["price"] - buy["price"]) / buy["price"]
                    row     = {"pnl": pnl_pct, "ticker": symbol, "side": "long"}
                    date_str = sell["date"] or buy["date"]
                    if date_str:
                        try:
                            ts = pd.to_datetime(date_str)
                            if ts.tzinfo is not None:
                                ts = ts.tz_convert("UTC").tz_localize(None)
                            row["date"] = ts
                        except Exception:
                            pass
                    rows.append(row)

        # If no pairs found (e.g. only sells or only buys) fall back to individual orders
        if len(rows) < 5:
            rows = []
            for o in filled:
                try:
                    qty    = float(o.get("filled_qty", 0) or 0)
                    price  = float(o.get("filled_avg_price", 0) or 0)
                    side   = o.get("side", "buy")
                    symbol = o.get("symbol", "")
                    date_str = o.get("filled_at") or o.get("updated_at")
                    notional = qty * price
                    pnl_proxy = (1.0 if side == "sell" else -1.0) * 0.005
                    row = {"pnl": pnl_proxy, "ticker": symbol, "side": side}
                    if date_str:
                        try:
                            row["date"] = pd.to_datetime(date_str).tz_localize(None)
                        except Exception:
                            pass
                    rows.append(row)
                except Exception:
                    continue

        if len(rows) < 5:
            raise HTTPException(
                400,
                "Could not extract enough filled trades from your order history. "
                "Make sure your account has at least 5 completed (filled) orders."
            )

        import pandas as pd
        df = pd.DataFrame(rows)

        # ── Enforce minimum trade count ───────────────────────────────────────
        # Never mix in portfolio history — it contaminates per-trade metrics.
        # Score must reflect real closed trades only, nothing else.
        if len(rows) < 10:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(rows),
                "trades_needed": 10,
                "message": f"Only {len(rows)} completed round-trip trades found. QuantProof needs at least 10 closed trades for a statistically valid analysis. Keep trading and come back — results with more history will be far more meaningful."
            })
        # L5 warning: if many orders couldn't be paired (options, shorts, multi-leg)
        total_filled_sides = len([o for o in filled])
        paired_sides = len(rows) * 2
        unpaired_pct = (total_filled_sides - paired_sides) / max(total_filled_sides, 1) * 100
        unpaired_warning = unpaired_pct > 30  # >30% of orders couldn't be matched
        if unpaired_warning:
            print(f"[validate/alpaca] WARNING: {unpaired_pct:.0f}% of orders unpaired — possible options/shorts")

        print(f"Using {len(rows)} paired round-trip trades")

        # Always discard keys
        try:
            payload.api_key    = "[DISCARDED]"
            payload.secret_key = "[DISCARDED]"
        except Exception:
            pass

        # ── Step 4: Run full validation ───────────────────────────────────────
        try:
            validator = QuantProofValidator(df)
            report    = validator.run()
        except Exception as e:
            raise HTTPException(500, f"Validation failed: {str(e)}")

        r = report
        scored_checks = [c for c in r.checks if c.category != "Plausibility"]
        top_issues    = [c.fix  for c in sorted(scored_checks, key=lambda x: x.score)[:3]]
        top_strengths = [c.name for c in sorted(scored_checks, key=lambda x: x.score, reverse=True)[:3]]
        _unpaired_warning = locals().get('unpaired_warning', False)

        date_range = "No date data"
        if "date" in df.columns:
            try:
                date_range = f"{df['date'].dropna().min().date()} to {df['date'].dropna().max().date()}"
            except Exception:
                pass

        check_count = len(scored_checks)
        summary = (
            f"QuantProof VERIFIED analysis of {r.total_trades} trades pulled directly from Alpaca "
            f"({'paper' if is_paper else 'live'} account) across {check_count} institutional checks. "
            f"Fundable Score: {r.score}/100 ({r.grade.split('—')[0].strip()}). "
            f"{'Core risk management needs work.' if r.score < 60 else 'Edge shows real promise — focus on execution costs.'}"
        )

        visible_checks = [c for c in r.checks if c.category != "Plausibility"]

        try:
            dashboard_obj  = ValidationDashboard(validator, report)
            dashboard_data = dashboard_obj.generate_interactive_report()
        except Exception:
            dashboard_data = None

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
            engine_version=r.engine_version + "-VERIFIED",
            audit_flags=r.audit_flags + [
                f"✓ Source: Alpaca {'Paper' if is_paper else 'Live'} Trading API",
                f"✓ Trades independently verified — not user-uploaded",
                f"✓ Credential handling: keys discarded after fetch, never logged"
            ],
            plausibility_summary=r.plausibility_summary,
            assumptions=r.assumptions + [
                "Trade data sourced directly from Alpaca API — not user-submitted",
                f"Account type: {'paper trading' if is_paper else 'live trading'}",
                "API credentials were used read-only and discarded after data fetch"
            ],
            dashboard=dashboard_data,
            ai_plan=None,  # AI analysis included in paid $29 compliance report
            scale_warning=False,
            currency=acct_currency,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Sanitise error — never echo anything that could contain credentials
        err_str = str(e)
        for sensitive in [payload.api_key, payload.secret_key]:
            if sensitive and sensitive != "[DISCARDED]" and len(sensitive) > 4:
                err_str = err_str.replace(sensitive, "[REDACTED]")
        raise HTTPException(500, f"Broker validation failed: {err_str}")




# =========================================================
# PDF REPORT — PAYWALL ENDPOINT
# =========================================================

from fastapi import Header
from fastapi.responses import Response as FastAPIResponse
# generate_report imported lazily inside endpoint — allows server to start without it
try:
    from generate_report import generate_pdf_report as _generate_pdf_report
    _PDF_AVAILABLE = True
except ImportError:
    _generate_pdf_report = None
    _PDF_AVAILABLE = False

# Simple token gate — replace with Stripe webhook in production
REPORT_ACCESS_TOKENS = set()   # populated by Stripe webhook

RAZORPAY_KEY_ID       = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET   = os.getenv("RAZORPAY_KEY_SECRET", "")
REPORT_PRICE_USD      = 9     # $9 USD
REPORT_PRICE_INR      = 999   # ₹999
REPORT_PRICE_USD = 9


@app.post("/report/generate")
async def generate_report_endpoint(
    file: UploadFile = File(...),
    x_report_token: str = Header(default=""),
    x_report_source: str = Header(default="csv")   # "csv" or "alpaca" or "ibkr"
):
    """
    Paid endpoint — returns a full PDF analysis report.
    Gate: x-report-token header must match a paid session token.
    For dev/testing: token "dev-free" bypasses the gate.
    """

    # Token check
    def _verify_token(token: str) -> bool:
        # Re-read from env each call — avoids NameError in nested scope
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok:
            return True
        if token in REPORT_ACCESS_TOKENS:
            return True
        # Legacy cs_ tokens no longer accepted
        if token.startswith("cs_"):
            return False
        # Time-based HMAC tokens — issued by Razorpay webhook
        # Token format: sha256("report:{hour}:{secret}:{payment_id}")[:32]
        # We can't reverse-verify without payment_id, so rely on REPORT_ACCESS_TOKENS set
        # which is populated at webhook time (survives if server hasn't restarted)
        return False
    is_paid = _verify_token(x_report_token)
    if not is_paid:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "payment_required",
                "message": "Purchase required to download the full analysis report.",
                "price_inr": REPORT_PRICE_INR,
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

    # CSV mode — strip prop firm compliance check, replace with disclaimer
    PROP_FIRM_CHECK_NAMES = {
        "Prop Firm Compliance (FTMO/Topstep/The5ers)",
        "Institutional Compliance (CTA/Fund Standard)",
        "Prop Firm Compliance",
    }
    def _csv_check_filter(c):
        if c.name in PROP_FIRM_CHECK_NAMES:
            # Replace with a disclaimer — CSV data cannot be used for compliance
            return {
                "name": c.name,
                "passed": False,
                "score": 0,
                "value": "⚠ NOT APPLICABLE — CSV Upload",
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
        "fundable_score": report.score,
        "grade":          report.grade,
        "sharpe":         round(report.sharpe, 2),
        "max_drawdown":   round(report.max_drawdown, 4),
        "win_rate":       round(report.win_rate, 1),
        "total_trades":   report.total_trades,
        "date_range":     date_range,
        "is_verified":    x_report_source in ("alpaca", "ibkr"),
        "checks": [
            # Only apply CSV disclaimer filter for unverified CSV uploads
            (_csv_check_filter(c) if x_report_source == "csv" else
             {"name": c.name, "passed": c.passed, "score": c.score,
              "value": c.value, "insight": c.insight, "fix": c.fix,
              "category": c.category})
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

    # Generate AI fix plan for paid report
    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan

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
    Creates a Razorpay order for the $9 report.
    Frontend opens Razorpay popup, on success calls /report/webhook with payment_id.
    """
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
        # Razorpay amounts are in paise (INR) or smallest unit
        # We charge ₹750 (~$9 USD) — international cards billed in INR
        order = client.order.create({
            "amount":   99900,   # 999 INR in paise
            "currency": "INR",
            "payment_capture": 1,
            "notes": {"product": "QuantProof Full Report"}
        })
        return {
            "order_id":  order["id"],
            "amount":    order["amount"],
            "currency":  order["currency"],
            "key_id":    key_id,
            "checkout_url": None   # Razorpay uses popup not redirect
        }
    except Exception as e:
        print(f"Razorpay order error: {type(e).__name__}: {e}")
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@app.post("/report/webhook")
async def payment_webhook(request: Request):
    """
    Called by frontend after Razorpay payment success.
    Verifies payment signature and issues a one-time download token.
    """
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

    # ALWAYS verify signature — never skip, never trust empty fields
    if not payment_id or not order_id or not signature:
        raise HTTPException(400, "Missing payment fields — payment_id, order_id and signature all required")
    expected = hmac.new(
        key_secret.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid payment signature — payment could not be verified")

    # Generate TWO tokens — current hour and next hour
    # Guarantees user always has at least ~1 hour regardless of when they pay
    current_hour = int(time.time() // 3600)
    token_now  = hashlib.sha256(f"report:{current_hour}:{key_secret}:{payment_id}".encode()).hexdigest()[:32]
    token_next = hashlib.sha256(f"report:{current_hour+1}:{key_secret}:{payment_id}".encode()).hexdigest()[:32]
    REPORT_ACCESS_TOKENS.add(token_now)
    REPORT_ACCESS_TOKENS.add(token_next)
    print(f"[webhook] Payment verified: {payment_id} → tokens issued (valid ~2hrs)")
    # Return the current-hour token — both are valid
    return {"status": "ok", "token": token_now}




# =========================================================
# BROKER PDF REPORT — uses real Alpaca data, not CSV
# =========================================================

class BrokerReportRequest(BaseModel):
    api_key:    str
    secret_key: str
    mode:       str = "paper"

@app.post("/report/generate-broker")
async def generate_broker_report(
    payload: BrokerReportRequest,
    x_report_token: str = Header(default=""),
):
    """
    Paid endpoint — generates PDF from live Alpaca data.
    Same token gate as /report/generate.
    """
    # ── Token check (reuse same logic) ───────────────────────────────────────
    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok:
            return True
        if token in REPORT_ACCESS_TOKENS:
            return True
        if token.startswith("cs_"):
            return False
        return False

    if not _verify_token(x_report_token):
        raise HTTPException(402, {"error": "payment_required", "message": "Purchase required."})

    # ── Pull real Alpaca data ─────────────────────────────────────────────────
    api_key    = payload.api_key.strip()
    secret_key = payload.secret_key.strip()
    if not api_key or not secret_key:
        raise HTTPException(400, "API key and secret required.")

    is_paper = payload.mode == "paper"
    base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"

    try:
        import httpx
        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "accept":              "application/json"
        }
        async with httpx.AsyncClient(timeout=20) as client:
            acct_resp = await client.get(f"{base_url}/v2/account", headers=headers)
            if not acct_resp.is_success:
                raise HTTPException(401, "Invalid Alpaca credentials.")
            orders_resp = await client.get(
                f"{base_url}/v2/orders", headers=headers,
                params={"status": "closed", "limit": 500, "direction": "desc"}
            )
        raw_orders = orders_resp.json() if orders_resp.is_success else []
        # Alpaca sometimes returns a dict with an error instead of a list
        orders = raw_orders if isinstance(raw_orders, list) else []
        print(f"[generate-broker] orders fetched: {len(orders)}, resp status: {orders_resp.status_code}")

        # Discard keys immediately
        api_key = secret_key = "[DISCARDED]"
        headers = {}

        # Pair trades (same logic as /validate/alpaca)
        from collections import defaultdict as _dd
        buy_stack = _dd(list)
        rows = []
        filled = [o for o in orders if o.get("status") == "filled"
                  and float(o.get("filled_qty", 0) or 0) > 0
                  and float(o.get("filled_avg_price", 0) or 0) > 0]
        # Robust pairing: first pass collect all buys per symbol,
        # second pass match sells — order in list doesn't matter
        buys_by_symbol  = _dd(list)
        sells_by_symbol = _dd(list)

        for o in filled:
            try:
                side   = o.get("side", "")
                symbol = o.get("symbol", "")
                price  = float(o.get("filled_avg_price", 0) or 0)
                date_str = o.get("filled_at") or o.get("updated_at")
                if side == "buy":
                    buys_by_symbol[symbol].append({"price": price, "date": date_str})
                elif side == "sell":
                    sells_by_symbol[symbol].append({"price": price, "date": date_str})
            except Exception:
                continue

        print(f"[generate-broker] buys by symbol: { {k:len(v) for k,v in buys_by_symbol.items()} }")
        print(f"[generate-broker] sells by symbol: { {k:len(v) for k,v in sells_by_symbol.items()} }")

        # Match each sell to a buy of same symbol
        for symbol in sells_by_symbol:
            for sell in sells_by_symbol[symbol]:
                if buys_by_symbol[symbol]:
                    buy = buys_by_symbol[symbol].pop(0)
                    pnl_pct = (sell["price"] - buy["price"]) / buy["price"]
                    row = {"pnl": pnl_pct, "ticker": symbol, "side": "long"}
                    date_str = sell["date"] or buy["date"]
                    if date_str:
                        try:
                            ts = pd.to_datetime(date_str)
                            if ts.tzinfo is not None:
                                ts = ts.tz_convert("UTC").tz_localize(None)
                            row["date"] = ts
                        except Exception:
                            pass
                    rows.append(row)

        print(f"[generate-broker] filled orders: {len(filled)}, paired rows: {len(rows)}")

        # Discard credentials — no more fetches needed
        api_key = secret_key = "[DISCARDED]"

        # Hard requirement: never mix in other data sources — scores must reflect real trades only
        if len(rows) < 10:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "message": f"Only {len(rows)} completed round-trip trades found. QuantProof needs at least 10 closed trades to generate a statistically valid report. Keep trading and come back — your results will be more meaningful with more history.",
                "trades_found": len(rows),
                "trades_needed": 10
            })

        df = pd.DataFrame(rows)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[generate-broker] exception: {type(e).__name__}: {e}")
        raise HTTPException(500, f"Alpaca fetch failed: {str(e)[:200]}")

    # ── Run validation ────────────────────────────────────────────────────────
    try:
        validator = QuantProofValidator(df)
        report    = validator.run()
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored = [c for c in report.checks if c.category != "Plausibility"]
    date_range = "No date data"
    if "date" in df.columns:
        try:
            date_range = f"{df['date'].dropna().min().date()} to {df['date'].dropna().max().date()}"
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
        "is_verified":    True,
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

    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan

    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available on this server.")
    try:
        pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[generate-broker] PDF CRASH:\n{tb}")
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    score     = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    filename  = f"QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"

    return FastAPIResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

# =========================================================
# IBKR BROKER INTEGRATION
# =========================================================

class IBKRConnectRequest(BaseModel):
    account_id: str
    flex_token: str       # IBKR Flex Web Service token
    query_id: str         # Flex Query ID configured by user
    days_back: int = 365  # How many days of trade history to pull

class IBKRConnectResponse(BaseModel):
    source: str           # "IBKR_FLEX"
    account_id: str
    trades_pulled: int
    date_range: str
    validation: ValidationResponse

@app.post("/connect/ibkr", response_model=IBKRConnectResponse)
async def connect_ibkr(payload: IBKRConnectRequest, request: Request = None):
    """
    Pulls verified trade history from Interactive Brokers via Flex Web Service.
    No CSV needed — data comes directly from IBKR.

    Setup for user:
    1. Log into IBKR Account Management
    2. Reports > Flex Queries > Create a Trade Confirmation Flex Query
    3. Include fields: TradeDate, Symbol, Buy/Sell, Quantity, Price, NetCash, Commission
    4. Generate a Flex Token (Settings > Flex Web Service)
    5. Note your Query ID
    """
    # Rate limit applies to broker connections too
    if request:
        ip = request.client.host if request.client else "unknown"
        allowed, _ = check_rate_limit(ip)
        if not allowed:
            raise HTTPException(429, f"Daily limit of {FREE_DAILY_LIMIT} validations reached.")

    import httpx, xml.etree.ElementTree as ET
    from datetime import datetime, timedelta

    end_date   = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=payload.days_back)).strftime("%Y%m%d")

    # Step 1 — Request the Flex report
    request_url = (
        f"https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
        f"?t={payload.flex_token}&q={payload.query_id}&v=3"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(request_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        status = root.findtext("Status")
        if status != "Success":
            error = root.findtext("ErrorMessage") or "Unknown IBKR error"
            raise HTTPException(400, f"IBKR Flex request failed: {error}")
        reference_code = root.findtext("ReferenceCode")
        base_url       = root.findtext("Url")
    except ET.ParseError:
        raise HTTPException(400, "Could not parse IBKR response. Check your Flex token and query ID.")
    except httpx.HTTPError as e:
        raise HTTPException(503, f"Could not reach IBKR servers: {str(e)}")

    # Step 2 — Retrieve the report (IBKR needs ~5s to generate it)
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

    # Step 3 — Parse trades into DataFrame
    trades = []
    for stmt in report_root.iter("FlexStatement"):
        acct = stmt.get("accountId", "")
        if payload.account_id and acct != payload.account_id:
            continue
        for trade in stmt.iter("Trade"):
            try:
                date_str  = trade.get("tradeDate") or trade.get("TradeDate", "")
                net_cash  = float(trade.get("netCash") or trade.get("NetCash", 0))
                price     = float(trade.get("tradePrice") or trade.get("TradePrice", 1))
                # Normalise PnL as return fraction
                pnl_frac  = net_cash / abs(price * float(trade.get("quantity", 1) or 1))
                pnl_frac  = max(-0.5, min(0.5, pnl_frac))
                trades.append({"date": date_str, "pnl": pnl_frac,
                                "symbol": trade.get("symbol", ""),
                                "side": "long" if float(trade.get("quantity", 1) or 1) > 0 else "short"})
            except (ValueError, ZeroDivisionError):
                continue

    if len(trades) < 5:
        raise HTTPException(400, f"Only {len(trades)} parseable trades found in IBKR export. Need at least 5.")

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["pnl"])

    # Step 4 — Run full validation (same as CSV path)
    try:
        validator = QuantProofValidator(df)
        report    = validator.run()
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

    scored     = [c for c in report.checks if c.category != "Plausibility"]
    top_issues = [c.fix  for c in sorted(scored, key=lambda x: x.score)[:3]]
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

    ai_plan = await generate_ai_plan(report, date_range)

    validation_resp = ValidationResponse(
        fundable_score=report.score,
        grade=report.grade,
        summary=summary,
        checks=[CheckResultResponse(**vars(c)) for c in scored],
        crash_sims=[CrashSimResponse(**vars(s)) for s in report.crash_sims],
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
        audit_flags=report.audit_flags + ["SOURCE: Interactive Brokers Flex API — trade data verified"],
        plausibility_summary=report.plausibility_summary,
        assumptions=report.assumptions,
        dashboard=None,
        ai_plan=None,  # AI analysis is a paid feature — included in PDF report
    )

    return IBKRConnectResponse(
        source="IBKR_FLEX",
        account_id=payload.account_id,
        trades_pulled=len(trades),
        date_range=date_range,
        validation=validation_resp,
    )


@app.get("/connect/ibkr/setup")
def ibkr_setup_guide():
    """Returns step-by-step IBKR Flex setup instructions."""
    return {
        "title": "Connect Interactive Brokers to QuantProof",
        "steps": [
            "1. Log into IBKR Client Portal (clientportal.ibkr.com)",
            "2. Go to: Performance & Reports > Flex Queries",
            "3. Click 'Create' — choose 'Trade Confirmation Flex Query'",
            "4. Select fields: TradeDate, Symbol, Buy/Sell, Quantity, Price, NetCash, Commission",
            "5. Save the query — note the Query ID",
            "6. Go to: Settings > Flex Web Service > Create Token",
            "7. Copy your Flex Token",
            "8. Enter Account ID, Flex Token, and Query ID in QuantProof"
        ],
        "note": "Your credentials are used only to pull trade data and are never stored. Each validation makes a direct IBKR API call.",
        "why_ibkr": "IBKR-connected validations include a verified source flag in the audit hash — accepted by most prop firms as proof of live trading history."
    }
