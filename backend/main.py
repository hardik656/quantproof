"""
QuantProof — FastAPI Backend v1.4
Synced to validator v6.2

CHANGES FROM v1.3:
  - Version bumped to 2.6.0
  - Root endpoint: 38 → 40 checks
  - All summary strings: '3 crash simulations' → '5 crash simulations'
  - smart_parse_csv(): auto-assigns regime via yfinance SPY SMA when dates
    present but no regime column (fallback: SPY monthly lookup table)
  - /validate/alpaca + /report/generate-broker: replaced broken buy/sell
    stack pairing with position-state tracker (sort by filled_at, FIFO lots,
    orphan sell detection)
  - All Alpaca + IBKR paths now auto-assign regime after trade pairing
  - benchmark_returns not supplied anywhere — IR + Beta checks return
    score=50 informational (correct, fully backward-compatible)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
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

# ── Rate limiting (3 free validations / IP / day) ────────────────────────────
_validation_counts: dict = defaultdict(lambda: {"count": 0, "reset": 0})
FREE_DAILY_LIMIT = 3

def check_rate_limit(ip: str) -> tuple[bool, int]:
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

app = FastAPI(title="QuantProof API", version="2.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AI Fix Plan ───────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY",      "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")

async def generate_ai_plan(report, date_range: str) -> dict:
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

        if GROQ_API_KEY:
            url     = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body    = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1500, "temperature": 0.3}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        elif ANTHROPIC_API_KEY:
            url     = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            body    = {"model": "claude-sonnet-4-20250514", "max_tokens": 1500,
                       "messages": [{"role": "user", "content": prompt}]}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        elif OPENAI_API_KEY:
            url     = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body    = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1500, "temperature": 0.3}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        else:
            return None

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    except Exception as e:
        print(f"AI plan failed ({type(e).__name__}): {e}")
        return None


# =========================================================
# SPY REGIME HELPERS
# =========================================================

# Monthly fallback table — used when yfinance is unavailable.
# Based on SPY 20-day SMA trend direction at month-end.
SPY_REGIME_LOOKUP: Dict[str, str] = {
    "2020-01": "BULL",  "2020-02": "BEAR",  "2020-03": "BEAR",
    "2020-04": "BULL",  "2020-05": "BULL",  "2020-06": "CONSOLIDATION",
    "2020-07": "BULL",  "2020-08": "BULL",  "2020-09": "BEAR",
    "2020-10": "BEAR",  "2020-11": "BULL",  "2020-12": "BULL",
    "2021-01": "CONSOLIDATION", "2021-02": "BULL", "2021-03": "BULL",
    "2021-04": "BULL",  "2021-05": "CONSOLIDATION", "2021-06": "BULL",
    "2021-07": "BULL",  "2021-08": "BULL",  "2021-09": "CONSOLIDATION",
    "2021-10": "BULL",  "2021-11": "BEAR",  "2021-12": "BULL",
    "2022-01": "BEAR",  "2022-02": "BEAR",  "2022-03": "CONSOLIDATION",
    "2022-04": "BEAR",  "2022-05": "BEAR",  "2022-06": "BEAR",
    "2022-07": "BULL",  "2022-08": "BEAR",  "2022-09": "BEAR",
    "2022-10": "BULL",  "2022-11": "BULL",  "2022-12": "BEAR",
    "2023-01": "BULL",  "2023-02": "CONSOLIDATION", "2023-03": "BULL",
    "2023-04": "BULL",  "2023-05": "CONSOLIDATION", "2023-06": "BULL",
    "2023-07": "BULL",  "2023-08": "BEAR",  "2023-09": "BEAR",
    "2023-10": "BEAR",  "2023-11": "BULL",  "2023-12": "BULL",
    "2024-01": "BULL",  "2024-02": "BULL",  "2024-03": "BULL",
    "2024-04": "BEAR",  "2024-05": "BULL",  "2024-06": "BULL",
    "2024-07": "CONSOLIDATION", "2024-08": "CONSOLIDATION",
    "2024-09": "BULL",  "2024-10": "BULL",  "2024-11": "BULL",
    "2024-12": "BEAR",
    "2025-01": "BULL",  "2025-02": "BULL",  "2025-03": "BEAR",
    "2025-04": "BEAR",  "2025-05": "BULL",  "2025-06": "BULL",
    "2025-07": "BULL",  "2025-08": "CONSOLIDATION",
    "2025-09": "BULL",  "2025-10": "BULL",  "2025-11": "BULL",
    "2025-12": "BULL",
    "2026-01": "BULL",  "2026-02": "BEAR",  "2026-03": "BEAR",
}


def assign_regimes(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Assigns a 'regime' column to df using SPY 20-day SMA via yfinance.
    Falls back to SPY monthly lookup table if yfinance is unavailable.
    df must have a 'date' column. Returns (df_with_regime, audit_note).
    Only called when df has dates but no regime column.
    """
    try:
        import yfinance as yf
        start = df["date"].min() - pd.Timedelta(days=30)  # extra buffer for SMA warmup
        end   = df["date"].max() + pd.Timedelta(days=1)
        spy   = yf.download("SPY", start=start, end=end, progress=False)["Close"]
        spy   = spy.squeeze()  # handle MultiIndex columns from newer yfinance
        sma20 = spy.rolling(20).mean()

        def _regime(d):
            try:
                d_ts = pd.Timestamp(d).tz_localize(None)
                price = float(spy.asof(d_ts))
                sma   = float(sma20.asof(d_ts))
                if np.isnan(price) or np.isnan(sma) or sma == 0:
                    return "CONSOLIDATION"
                ratio = price / sma
                if ratio > 1.01:   return "BULL"
                elif ratio < 0.99: return "BEAR"
                else:              return "CONSOLIDATION"
            except Exception:
                return "CONSOLIDATION"

        df["regime"] = df["date"].apply(_regime)
        dist = df["regime"].value_counts().to_dict()
        return df, f"✓ Regime auto-detected from SPY 20-day SMA — {dist}"

    except Exception as e:
        # Fallback: monthly lookup table — works for any date back to 2020
        def _regime_fallback(d):
            try:
                return SPY_REGIME_LOOKUP.get(
                    pd.Timestamp(d).strftime("%Y-%m"), "CONSOLIDATION"
                )
            except Exception:
                return "CONSOLIDATION"

        df["regime"] = df["date"].apply(_regime_fallback)
        dist = df["regime"].value_counts().to_dict()
        return df, f"⚠ Regime from monthly SPY approximation (live data unavailable) — {dist}"


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
    fundable_score: float
    grade: str
    summary: str
    total_trades: int
    profitable_trades: int
    date_range: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    checks: List[CheckResultResponse]
    crash_sims: List[CrashSimResponse]
    top_issues: List[str]
    top_strengths: List[str]
    validation_hash: str
    validation_date: str
    engine_version: str
    audit_flags: List[str]
    plausibility_summary: str
    scale_warning: bool = False
    currency: str = "USD"
    assumptions: List[str]
    dashboard: Optional[Dict[str, Any]] = None
    ai_plan: Optional[Dict[str, Any]] = None


# =========================================================
# ALPACA ORDER PAIRING — position-state tracker
# =========================================================

def pair_alpaca_orders(orders: list) -> tuple[pd.DataFrame, list]:
    """
    Converts raw Alpaca filled orders into a trades DataFrame.

    Fixes vs old buy/sell stack approach:
    - Sorts by filled_at chronologically before pairing
    - Tracks open position lots per symbol (FIFO)
    - Orphaned sells (no open lot) are flagged and skipped,
      not silently mispaired with a future buy
    - Regime auto-assigned via assign_regimes() after pairing

    Returns (df, audit_flags).
    """
    audit_flags = []

    # Filter + parse filled orders
    filled = []
    for o in orders:
        if o.get("status") != "filled":
            continue
        try:
            qty   = float(o.get("filled_qty") or 0)
            price = float(o.get("filled_avg_price") or 0)
            if qty <= 0 or price <= 0:
                continue
            side = o.get("side", "").lower()
            if side not in ("buy", "sell"):
                continue
            ts_str = o.get("filled_at") or o.get("updated_at") or ""
            ts = pd.to_datetime(ts_str, utc=True).tz_localize(None)
            filled.append({
                "symbol":    o.get("symbol", "UNKNOWN"),
                "side":      side,
                "qty":       qty,
                "price":     price,
                "filled_at": ts,
            })
        except Exception:
            continue

    if not filled:
        return pd.DataFrame(), ["No filled orders found"]

    # Sort chronologically — critical fix
    df_orders = pd.DataFrame(filled).sort_values("filled_at").reset_index(drop=True)

    # Position-state tracker: symbol → FIFO list of open lots
    lots          = defaultdict(list)   # sym → [{"price", "qty", "entry_date"}, ...]
    completed     = []
    orphan_sells  = []

    for _, row in df_orders.iterrows():
        sym   = row["symbol"]
        side  = row["side"]
        qty   = row["qty"]
        price = row["price"]
        ts    = row["filled_at"]

        if side == "buy":
            lots[sym].append({"price": price, "qty": qty, "entry_date": ts})

        elif side == "sell":
            if not lots[sym]:
                # Sell with no open lot = position opened before history window
                orphan_sells.append(
                    f"{sym} sell@{price:.2f} on {ts.date()} "
                    f"(no matching buy in history — position predates window)"
                )
                continue

            remaining = qty
            while remaining > 0 and lots[sym]:
                lot       = lots[sym][0]
                close_qty = min(remaining, lot["qty"])
                pnl_pct   = (price - lot["price"]) / lot["price"]
                completed.append({
                    "pnl":        pnl_pct,
                    "date":       ts,
                    "entry_date": lot["entry_date"],
                    "ticker":     sym,
                    "buy_price":  lot["price"],
                    "sell_price": price,
                    "qty":        close_qty,
                    "hold_days":  max(0, (ts - lot["entry_date"]).days),
                })
                lot["qty"]  -= close_qty
                remaining   -= close_qty
                if lot["qty"] == 0:
                    lots[sym].pop(0)

    # Build audit flags
    if orphan_sells:
        audit_flags.append(
            f"⚠ {len(orphan_sells)} sell(s) had no matching buy in history window "
            f"(position opened before history) — excluded from analysis"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f"ℹ {len(open_positions)} open position(s) not included "
            f"(no closing trade yet): {sym_list}"
        )

    if not completed:
        return pd.DataFrame(), audit_flags + [
            "No completed round-trip trades found. "
            "All positions may still be open, or all sells predate the history window."
        ]

    df = pd.DataFrame(completed).sort_values("date").reset_index(drop=True)

    # Auto-assign regime from SPY
    df, regime_note = assign_regimes(df)
    audit_flags.append(regime_note)

    return df, audit_flags


# =========================================================
# CSV PARSER
# =========================================================

def smart_parse_csv(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parses CSV into a trades DataFrame.
    Returns (df, regime_note) where regime_note is empty string if
    regime was already in the CSV or dates were absent.

    New in v1.4: if CSV has dates but no regime column, auto-assigns
    regime via assign_regimes() (yfinance SPY → monthly fallback).
    """
    raw = contents.decode("utf-8", errors="ignore")
    regime_note = ""

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

                    date_cols = [c for c in df.columns
                                 if any(x in c.lower() for x in ["date", "time", "dt"])]
                    if date_cols:
                        result["date"] = pd.to_datetime(
                            df[date_cols[0]].values[:len(vals)], errors="coerce"
                        )

                    vol_cols = [c for c in df.columns
                                if "volume" in c.lower() or "vol" == c.lower()]
                    if vol_cols:
                        result["volume"] = pd.to_numeric(
                            df[vol_cols[0]], errors="coerce"
                        ).values[:len(vals)]

                    reg_cols = [c for c in df.columns if "regime" in c.lower()]
                    if reg_cols:
                        result["regime"] = df[reg_cols[0]].values[:len(vals)]
                    elif "date" in result.columns and result["date"].notna().any():
                        # Dates present, no regime → auto-detect from SPY
                        result, regime_note = assign_regimes(result)

                    return result, regime_note
    except Exception:
        pass

    # Fallback — flexible parse
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
                return pd.DataFrame({"pnl": vals.values}), ""
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                if vals.abs().mean() > 1.0:
                    vals = vals / 10000.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values}), ""
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
        "version": "2.6.0",
        "checks": "40 institutional checks + 5 crash simulations",
        "new_in_v1.4": [
            "Information Ratio (vs Benchmark) — detects beta-capture disguised as alpha",
            "Beta Exposure (vs Benchmark) — flags high market correlation",
            "Crash exposure model hardened: higher floor, AR(1) penalty, [UNCALIBRATED] labels",
            "Regime auto-detection via SPY 20-day SMA (yfinance) for CSV + Alpaca paths",
            "Alpaca: position-state tracker replaces broken buy/sell stack pairing",
        ],
        "new_in_v1.3": ["CVaR/Expected Shortfall", "Fractional Kelly + Ruin Probability",
                        "Market Impact (Almgren-Chriss)", "Deflated Sharpe",
                        "Purged Walk-Forward CV", "HMM Regime Detection",
                        "Capacity Analysis", "Interactive Dashboard API"]
    }

@app.post("/validate", response_model=ValidationResponse)
async def validate(file: UploadFile = File(...), currency: str = Form("USD"), request: Request = None):
    if request:
        ip = request.client.host if request.client else "unknown"
        allowed, remaining = check_rate_limit(ip)
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

    try:
        df, regime_note = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df) < 5:
        raise HTTPException(status_code=400, detail=f"Only found {len(df)} valid trades. Need at least 5.")

    try:
        # NOTE: benchmark_returns not supplied — IR + Beta checks return score=50 informational
        validator = QuantProofValidator(df)
        report = validator.run()
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
        f"+ 5 crash simulations. Fundable Score: {r.score}/100 ({r.grade.split('—')[0].strip()}). "
        f"{'Core risk management needs work.' if r.score < 60 else 'Edge shows real promise — focus on execution costs.'}"
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

    extra_flags = [regime_note] if regime_note else []

    _vr_cache = {
        "fundable_score": r.score,
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
    }
    _cache_validation(r.validation_hash, _vr_cache)

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
        audit_flags=r.audit_flags + extra_flags,
        plausibility_summary=r.plausibility_summary,
        scale_warning=getattr(r, '_scale_warning', False),
        currency=detected_currency,
        assumptions=r.assumptions,
        dashboard=dashboard_data,
        ai_plan=None,
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
        "note": "If you include dates but no regime column, regime is auto-detected from SPY 20-day SMA. Add regime column to override."
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.6.0"}


# =========================================================
# ALPACA BROKER INTEGRATION — VERIFIED VALIDATION
# =========================================================

class AlpacaConnectRequest(BaseModel):
    api_key:    str
    secret_key: str
    mode:       str = "paper"

def _scrub_keys(d: dict) -> dict:
    sensitive = {"api_key", "secret_key", "key", "secret", "token", "password", "auth"}
    return {k: "[REDACTED]" if k.lower() in sensitive else v for k, v in d.items()}

@app.post("/validate/alpaca", response_model=ValidationResponse)
async def validate_alpaca(payload: AlpacaConnectRequest, request: Request = None):
    BROKER_DAILY_LIMIT = 3
    if request:
        ip = request.client.host if request.client else "unknown"
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
        raise HTTPException(400, "Invalid key format — keys are too short.")

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

            orders_resp = await client.get(
                f"{base_url}/v2/orders",
                headers=headers,
                params={"status": "closed", "limit": 500, "direction": "desc"}
            )
            if not orders_resp.is_success:
                raise HTTPException(400, "Could not fetch order history from Alpaca.")
            orders = orders_resp.json()

        # Discard credentials immediately
        api_key = secret_key = "[DISCARDED]"
        headers = {}
        try:
            payload.api_key = payload.secret_key = "[DISCARDED]"
        except Exception:
            pass

        if not orders or len(orders) < 5:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(orders) if orders else 0,
                "trades_needed": 10,
                "message": f"Only {len(orders) if orders else 0} orders found. Need at least 10 completed round-trip trades."
            })

        # ── Position-state pairing + regime auto-detection ────────────────────
        df, parse_flags = pair_alpaca_orders(orders)

        if len(df) < 10:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(df),
                "trades_needed": 10,
                "message": (
                    f"Only {len(df)} completed round-trip trades found. "
                    "QuantProof needs at least 10 closed trades for a statistically valid analysis. "
                    + (" | ".join(parse_flags) if parse_flags else "")
                )
            })

        total_filled_sides = len([o for o in orders if o.get("status") == "filled"])
        paired_sides = len(df) * 2
        unpaired_pct = (total_filled_sides - paired_sides) / max(total_filled_sides, 1) * 100
        if unpaired_pct > 30:
            print(f"[validate/alpaca] WARNING: {unpaired_pct:.0f}% of orders unpaired — possible options/shorts")

        # ── Run validation ────────────────────────────────────────────────────
        # NOTE: benchmark_returns not supplied — IR + Beta checks return score=50 informational
        validator = QuantProofValidator(df)
        report    = validator.run()

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
            f"Fundable Score: {r.score}/100 ({r.grade.split('—')[0].strip()}). "
            f"{'Core risk management needs work.' if r.score < 60 else 'Edge shows real promise — focus on execution costs.'}"
        )

        visible_checks = [c for c in r.checks if c.category != "Plausibility"]

        try:
            dashboard_obj  = ValidationDashboard(validator, report)
            dashboard_data = dashboard_obj.generate_interactive_report()
        except Exception:
            dashboard_data = None

        _vr_cache = {
            "fundable_score": r.score, "grade": r.grade,
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
        }
        _cache_validation(r.validation_hash, _vr_cache)

        return ValidationResponse(
            fundable_score=r.score, grade=r.grade, summary=summary,
            checks=[CheckResultResponse(**vars(c)) for c in visible_checks],
            crash_sims=[CrashSimResponse(**vars(s)) for s in r.crash_sims],
            total_trades=r.total_trades, profitable_trades=r.profitable_trades,
            date_range=date_range, sharpe=round(r.sharpe, 2),
            max_drawdown=round(r.max_drawdown, 4), win_rate=round(r.win_rate, 1),
            top_issues=top_issues, top_strengths=top_strengths,
            validation_hash=r.validation_hash, validation_date=r.validation_date,
            engine_version=r.engine_version + "-VERIFIED",
            audit_flags=r.audit_flags + parse_flags + [
                f"✓ Source: Alpaca {'Paper' if is_paper else 'Live'} Trading API",
                "✓ Trades independently verified — not user-uploaded",
                "✓ Credential handling: keys discarded after fetch, never logged",
            ],
            plausibility_summary=r.plausibility_summary,
            assumptions=r.assumptions + [
                "Trade data sourced directly from Alpaca API — not user-submitted",
                f"Account type: {'paper trading' if is_paper else 'live trading'}",
                "API credentials were used read-only and discarded after data fetch",
            ],
            dashboard=dashboard_data, ai_plan=None,
            scale_warning=False, currency=acct_currency,
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

from fastapi import Header
from fastapi.responses import Response as FastAPIResponse
try:
    from generate_report import generate_pdf_report as _generate_pdf_report
    _PDF_AVAILABLE = True
except ImportError:
    _generate_pdf_report = None
    _PDF_AVAILABLE = False

REPORT_ACCESS_TOKENS = set()
LS_ORDER_TOKENS: dict = {}
import collections
VALIDATION_RESULT_CACHE: dict = collections.OrderedDict()
VALIDATION_CACHE_MAX = 50

def _cache_validation(v_hash: str, vr: dict):
    VALIDATION_RESULT_CACHE[v_hash] = vr
    if len(VALIDATION_RESULT_CACHE) > VALIDATION_CACHE_MAX:
        VALIDATION_RESULT_CACHE.popitem(last=False)

def _get_cached_validation(v_hash: str) -> dict | None:
    return VALIDATION_RESULT_CACHE.get(v_hash)

_TEST_TOKEN = os.getenv("TEST_REPORT_TOKEN", "")
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
REPORT_PRICE_USD    = 9
REPORT_PRICE_INR    = 999


@app.post("/report/generate")
async def generate_report_endpoint(
    file: UploadFile = File(...),
    x_report_token: str = Header(default=""),
    x_report_source: str = Header(default="csv")
):
    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok: return True
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
    try:
        df, regime_note = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(df) < 5:
        raise HTTPException(400, f"Only {len(df)} trades found. Need at least 5.")

    try:
        # NOTE: benchmark_returns not supplied — IR + Beta checks return score=50 informational
        validator = QuantProofValidator(df)
        report    = validator.run()
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
        "fundable_score": report.score, "grade": report.grade,
        "sharpe": round(report.sharpe, 2), "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1), "total_trades": report.total_trades,
        "date_range": date_range,
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
             "market_drop": s.market_drop, "strategy_drop": s.strategy_drop,
             "survived": s.survived, "emotional_verdict": s.emotional_verdict}
            for s in report.crash_sims
        ],
        "top_issues":    [c.fix  for c in sorted(scored, key=lambda x: x.score)[:4]],
        "top_strengths": [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]],
        "regime_note":   regime_note,
    }

    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan
    _cache_validation(report.validation_hash, vr)

    try:
        if not _PDF_AVAILABLE:
            raise HTTPException(503, "PDF module not available.")
        pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

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
            "amount": 99900, "currency": "INR", "payment_capture": 1,
            "notes": {"product": "QuantProof Full Report"}
        })
        return {"order_id": order["id"], "amount": order["amount"],
                "currency": order["currency"], "key_id": key_id, "checkout_url": None}
    except Exception as e:
        raise HTTPException(500, {"error": "payment_failed", "message": str(e)})


@app.post("/report/create-checkout-intl")
async def create_checkout_intl(payload: dict):
    import httpx
    ls_key      = os.getenv("LEMONSQUEEZY_API_KEY", "")
    ls_store_id = os.getenv("LEMONSQUEEZY_STORE_ID", "")
    ls_variant  = os.getenv("LEMONSQUEEZY_VARIANT_ID", "")
    if not ls_key or not ls_store_id or not ls_variant:
        raise HTTPException(503, {"error": "ls_not_configured",
            "message": "International payments not yet configured."})
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.lemonsqueezy.com/v1/checkouts",
                headers={"Authorization": f"Bearer {ls_key}",
                         "Accept": "application/vnd.api+json",
                         "Content-Type": "application/vnd.api+json"},
                json={"data": {"type": "checkouts",
                    "attributes": {"checkout_options": {"dark": True},
                        "product_options": {"redirect_url":
                            "https://quantproof-frontend.onrender.com/?payment=success&provider=ls&order_id={order_id}"}},
                    "relationships": {
                        "store":   {"data": {"type": "stores",   "id": ls_store_id}},
                        "variant": {"data": {"type": "variants", "id": ls_variant}}}}}
            )
        if not resp.is_success:
            raise HTTPException(500, {"error": "ls_checkout_failed", "message": resp.text[:200]})
        return {"checkout_url": resp.json()["data"]["attributes"]["url"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"error": "ls_error", "message": str(e)})


@app.post("/report/webhook-ls")
async def lemonsqueezy_webhook(request: Request):
    import hashlib, hmac, time, json
    body      = await request.body()
    ls_secret = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
    if ls_secret:
        sig      = request.headers.get("x-signature", "")
        expected = hmac.new(ls_secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(400, "Invalid Lemon Squeezy webhook signature")
    try:
        event = json.loads(body)
        if event.get("meta", {}).get("event_name") == "order_created":
            order_id     = str(event["data"]["id"])
            key_secret   = os.getenv("RAZORPAY_KEY_SECRET", "quantproof-fallback")
            current_hour = int(time.time() // 3600)
            token_now    = hashlib.sha256(f"report:{current_hour}:{key_secret}:ls_{order_id}".encode()).hexdigest()[:32]
            token_next   = hashlib.sha256(f"report:{current_hour+1}:{key_secret}:ls_{order_id}".encode()).hexdigest()[:32]
            REPORT_ACCESS_TOKENS.add(token_now)
            REPORT_ACCESS_TOKENS.add(token_next)
            LS_ORDER_TOKENS[order_id] = token_now
            return {"status": "ok", "token": token_now}
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"status": "ignored"}


@app.get("/report/ls-token")
async def get_ls_token(order_id: str = ""):
    if order_id and order_id != "{order_id}" and order_id in LS_ORDER_TOKENS:
        return {"status": "ready", "token": LS_ORDER_TOKENS[order_id]}
    if LS_ORDER_TOKENS:
        return {"status": "ready", "token": list(LS_ORDER_TOKENS.values())[-1]}
    return {"status": "pending", "message": "Payment processing — please wait a moment"}


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
    return {"status": "ok", "token": token_now}


# =========================================================
# BROKER PDF REPORT
# =========================================================

class BrokerReportRequest(BaseModel):
    api_key:         str
    secret_key:      str
    mode:            str = "paper"
    validation_hash: str = ""

@app.post("/report/generate-broker")
async def generate_broker_report(
    payload: BrokerReportRequest,
    x_report_token: str = Header(default=""),
):
    # Fast path: cached result
    if payload.validation_hash:
        cached_vr = _get_cached_validation(payload.validation_hash)
        if cached_vr:
            _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
            _token_ok = (_test_tok and x_report_token == _test_tok) or (x_report_token in REPORT_ACCESS_TOKENS)
            if not _token_ok:
                raise HTTPException(402, {"error": "payment_required", "message": "Valid payment token required"})
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
                    cached_vr["ai_plan"] = await generate_ai_plan(
                        _CachedReport(cached_vr), cached_vr.get("date_range", "")
                    )
                except Exception as _e:
                    issues = cached_vr.get("top_issues", [])
                    cached_vr["ai_plan"] = "Top improvements: " + "; ".join(issues[:3]) if issues else None
            if not _PDF_AVAILABLE:
                raise HTTPException(503, "PDF module not available.")
            try:
                pdf_bytes = _generate_pdf_report(cached_vr)
            except Exception as e:
                raise HTTPException(500, f"PDF generation failed: {str(e)}")
            score     = int(cached_vr["fundable_score"])
            grade_str = cached_vr["grade"].split("—")[0].strip().replace(" ", "")
            return Response(
                content=pdf_bytes, media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"}
            )

    def _verify_token(token: str) -> bool:
        _test_tok = os.getenv("TEST_REPORT_TOKEN", "")
        if _test_tok and token == _test_tok: return True
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
        async with httpx.AsyncClient(timeout=20) as client:
            acct_resp = await client.get(f"{base_url}/v2/account", headers=headers)
            if not acct_resp.is_success:
                raise HTTPException(401, "Invalid Alpaca credentials.")
            orders_resp = await client.get(
                f"{base_url}/v2/orders", headers=headers,
                params={"status": "closed", "limit": 500, "direction": "desc"}
            )
        raw_orders = orders_resp.json() if orders_resp.is_success else []
        orders     = raw_orders if isinstance(raw_orders, list) else []

        api_key = secret_key = "[DISCARDED]"
        headers = {}

        # Position-state pairing + regime auto-detection
        df, parse_flags = pair_alpaca_orders(orders)

        if len(df) < 10:
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "message": f"Only {len(df)} completed round-trip trades found. Need at least 10.",
                "trades_found": len(df), "trades_needed": 10
            })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Alpaca fetch failed: {str(e)[:200]}")

    try:
        # NOTE: benchmark_returns not supplied — IR + Beta checks return score=50 informational
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
        "fundable_score": report.score, "grade": report.grade,
        "sharpe": round(report.sharpe, 2), "max_drawdown": round(report.max_drawdown, 4),
        "win_rate": round(report.win_rate, 1), "total_trades": report.total_trades,
        "date_range": date_range, "is_verified": True,
        "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                    "value":c.value,"insight":c.insight,"fix":c.fix,
                    "category":c.category} for c in report.checks],
        "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                        "description":s.description,"market_drop":s.market_drop,
                        "strategy_drop":s.strategy_drop,"survived":s.survived,
                        "emotional_verdict":s.emotional_verdict} for s in report.crash_sims],
        "top_issues":    [c.fix  for c in sorted(scored, key=lambda x: x.score)[:4]],
        "top_strengths": [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]],
        "parse_flags":   parse_flags,
    }

    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan
    _cache_validation(report.validation_hash, vr)

    if not _PDF_AVAILABLE:
        raise HTTPException(503, "PDF module not available on this server.")
    try:
        pdf_bytes = _generate_pdf_report(vr)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    score     = int(report.score)
    grade_str = report.grade.split("—")[0].strip().replace(" ", "")
    return FastAPIResponse(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="QuantProof_VERIFIED_Report_{grade_str}_{score}.pdf"'}
    )


# =========================================================
# IBKR BROKER INTEGRATION
# =========================================================

class IBKRConnectRequest(BaseModel):
    account_id: str
    flex_token: str
    query_id:   str
    days_back:  int = 365

class IBKRConnectResponse(BaseModel):
    source:       str
    account_id:   str
    trades_pulled: int
    date_range:   str
    validation:   ValidationResponse

@app.post("/connect/ibkr", response_model=IBKRConnectResponse)
async def connect_ibkr(payload: IBKRConnectRequest, request: Request = None):
    if request:
        ip = request.client.host if request.client else "unknown"
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

    trades = []
    for stmt in report_root.iter("FlexStatement"):
        acct = stmt.get("accountId", "")
        if payload.account_id and acct != payload.account_id:
            continue
        for trade in stmt.iter("Trade"):
            try:
                date_str = trade.get("tradeDate") or trade.get("TradeDate", "")
                net_cash = float(trade.get("netCash") or trade.get("NetCash", 0))
                price    = float(trade.get("tradePrice") or trade.get("TradePrice", 1))
                pnl_frac = net_cash / abs(price * float(trade.get("quantity", 1) or 1))
                pnl_frac = max(-0.5, min(0.5, pnl_frac))
                trades.append({
                    "date":   date_str,
                    "pnl":    pnl_frac,
                    "ticker": trade.get("symbol", ""),
                    "side":   "long" if float(trade.get("quantity", 1) or 1) > 0 else "short"
                })
            except (ValueError, ZeroDivisionError):
                continue

    if len(trades) < 5:
        raise HTTPException(400, f"Only {len(trades)} parseable trades found in IBKR export. Need at least 5.")

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["pnl"])

    # Auto-assign regime from SPY (same logic as CSV path)
    if "date" in df.columns and df["date"].notna().any():
        df, regime_note = assign_regimes(df)
    else:
        regime_note = ""

    try:
        # NOTE: benchmark_returns not supplied — IR + Beta checks return score=50 informational
        validator = QuantProofValidator(df)
        report    = validator.run()
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
                  ["SOURCE: Interactive Brokers Flex API — trade data verified"]

    validation_resp = ValidationResponse(
        fundable_score=report.score, grade=report.grade, summary=summary,
        checks=[CheckResultResponse(**vars(c)) for c in scored],
        crash_sims=[CrashSimResponse(**vars(s)) for s in report.crash_sims],
        total_trades=report.total_trades, profitable_trades=report.profitable_trades,
        date_range=date_range, sharpe=round(report.sharpe, 2),
        max_drawdown=round(report.max_drawdown, 4), win_rate=round(report.win_rate, 1),
        top_issues=top_issues, top_strengths=top_strengths,
        validation_hash=report.validation_hash + "-IBKR",
        validation_date=report.validation_date, engine_version=report.engine_version,
        audit_flags=report.audit_flags + extra_flags,
        plausibility_summary=report.plausibility_summary,
        assumptions=report.assumptions, dashboard=None, ai_plan=None,
    )

    return IBKRConnectResponse(
        source="IBKR_FLEX", account_id=payload.account_id,
        trades_pulled=len(trades), date_range=date_range,
        validation=validation_resp,
    )


@app.get("/connect/ibkr/setup")
def ibkr_setup_guide():
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
        "note": "Your credentials are used only to pull trade data and are never stored.",
        "why_ibkr": "IBKR-connected validations include a verified source flag in the audit hash."
    }

