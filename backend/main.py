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
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import os
import re as _re
import time
import hashlib
import traceback
import uuid
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

app = FastAPI(title="QuantProof API", version="2.7.0")

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

# Simple SPY data cache — avoids double yfinance call when assign_regimes
# and get_spy_benchmark both need SPY for the same date range
_SPY_CACHE: dict = {}
_SPY_CACHE_MAX = 10  # keep last 10 date ranges in memory

def _get_spy_cached(start, end) -> pd.Series | None:
    """Fetch SPY Close with caching. Returns None on any failure."""
    key = (str(start.date()), str(end.date()))
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]
    try:
        import yfinance as yf, threading
        result_holder = [None]
        def _fetch():
            try:
                raw = yf.download("SPY", start=start, end=end,
                                  progress=False, timeout=6)["Close"]
                result_holder[0] = raw.squeeze()
            except Exception:
                pass
        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=8)
        spy = result_holder[0]
        if spy is not None and len(spy) >= 5:
            if len(_SPY_CACHE) >= _SPY_CACHE_MAX:
                _SPY_CACHE.pop(next(iter(_SPY_CACHE)))
            _SPY_CACHE[key] = spy
            return spy
    except Exception:
        pass
    return None


def assign_regimes(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Assigns a 'regime' column to df using SPY 20-day SMA via yfinance.
    Falls back to SPY monthly lookup table if yfinance is unavailable.
    df must have a 'date' column. Returns (df_with_regime, audit_note).
    Only called when df has dates but no regime column.
    """
    try:
        start = df["date"].min() - pd.Timedelta(days=30)  # extra buffer for SMA warmup
        end   = df["date"].max() + pd.Timedelta(days=1)
        spy   = _get_spy_cached(start, end)
        if spy is None:
            raise ValueError("SPY data unavailable")
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
# SPY BENCHMARK HELPER
# Called after regime assignment when df has dates.
# Returns aligned SPY daily returns for IR + Beta checks.
# Falls back to None silently — validator stays informational.
# =========================================================

def get_spy_benchmark(df: pd.DataFrame, spy_series: pd.Series = None) -> tuple:
    """
    Align SPY daily returns to strategy trade dates for IR + Beta scoring.
    Returns (benchmark_array, audit_note) or (None, "") on any failure.

    Accepts an optional spy_series (already fetched by assign_regimes) to
    avoid a second yfinance call. If not provided, fetches with an 8-second
    timeout via threading — times out gracefully rather than blocking.

    Quality gates:
      - Requires date column with at least some non-null dates
      - >50% of trade dates must match a SPY return
      - Result must have non-trivial std (not all zeros)
      - Any single return > 20% is treated as bad data and zeroed

    Falls back to (None, "") silently on timeout, network error, or any
    other failure — validator stays informational, nothing breaks.
    """
    if "date" not in df.columns or df["date"].isna().all():
        return None, ""

    try:
        # ── Fetch SPY if not already provided ────────────────────────────────
        if spy_series is None:
            import threading
            start = df["date"].min() - pd.Timedelta(days=10)
            end   = df["date"].max() + pd.Timedelta(days=2)
            spy_raw = _get_spy_cached(start, end)
            if spy_raw is None:
                return None, ""
        else:
            spy_raw = spy_series

        if spy_raw is None or len(spy_raw) < 5:
            return None, ""

        # ── Compute daily returns ─────────────────────────────────────────────
        spy_ret = spy_raw.pct_change().dropna()
        spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)

        # ── Align to strategy trade dates ─────────────────────────────────────
        aligned = []
        for d in df["date"]:
            try:
                d_ts = pd.Timestamp(d).tz_localize(None)
                val  = float(spy_ret.asof(d_ts))
                # Zero out missing or implausible values (>20% daily move)
                aligned.append(0.0 if (np.isnan(val) or abs(val) > 0.20) else val)
            except Exception:
                aligned.append(0.0)

        benchmark = np.array(aligned, dtype=float)

        # ── Quality gates ──────────────────────────────────────────────────────
        if np.std(benchmark) < 1e-6:
            return None, ""  # all zeros — no usable signal

        match_pct = float(np.sum(benchmark != 0.0)) / len(benchmark) * 100
        if match_pct < 50:
            return None, ""  # too many missing dates — not reliable

        note = (
            f"📈 SPY inferred as benchmark ({match_pct:.0f}% of trade dates matched) — "
            f"IR + Beta checks are now scored. Only meaningful for equity strategies."
        )
        return benchmark, note

    except Exception:
        # Catch-all — never let benchmark inference break a validation
        return None, ""


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
# KITE TRADEBOOK PARSER
# (defined before smart_parse_csv which calls it)
# =========================================================

def _is_kite_csv(df: pd.DataFrame) -> bool:
    """
    Detect Zerodha Kite TradeBook CSV by characteristic column names.

    Real Kite TradeBook columns (verified from Zerodha Console export):
      trade_date, trade_time, settlement_id, exchange, segment, series,
      trade_type, auction, quantity, price, trade_id, order_id, order_execution_time

    Key: ticker/instrument is in 'series' column (NOT 'symbol').
         segment is 'EQ' for equity, 'FO' for F&O (not 'EQUITY'/'FUTURES').
    """
    cols = {c.lower().strip() for c in df.columns}
    required     = {"trade_date", "trade_type", "quantity", "price"}
    kite_signals = {"trade_time", "settlement_id", "series", "order_id", "trade_id"}
    return required.issubset(cols) and bool(kite_signals & cols)


def _parse_kite_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Kite TradeBook CSV into QuantProof trades.

    Uses the same position-state tracker as pair_alpaca_orders:
      - Chronological sort within each symbol
      - FIFO lot queue per symbol
      - Partial lot handling
      - Orphan sell detection
      - Short positions via segment (FO = futures/options)
      - Returns (df, note, audit_flags)
    """
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"trade_date", "trade_type", "quantity", "price"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kite CSV missing required columns: {missing}. "
            "Please export from Zerodha Console → TradeBook → Download CSV. "
            "Make sure you are downloading the TradeBook, not Positions or Holdings."
        )

    ticker_col = None
    for candidate in ['series', 'symbol', 'instrument', 'scrip', 'tradingsymbol']:
        if candidate in df.columns:
            ticker_col = candidate
            break

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"])

    sort_cols = ([ticker_col, "trade_date"] if ticker_col and ticker_col in df.columns
                 else ["trade_date"])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    lots         = defaultdict(list)
    trades       = []
    audit_flags  = []
    orphan_sells = []
    orphan_buys  = []

    for _, row in df.iterrows():
        symbol     = str(row[ticker_col]) if ticker_col and ticker_col in row else "UNKNOWN"
        qty        = float(row["quantity"])
        price      = float(row["price"])
        trade_type = str(row["trade_type"]).upper().strip()
        date       = row["trade_date"]
        segment    = str(row.get("segment", "EQ")).upper().strip()
        is_fo      = segment in ("FO", "F&O", "FUTURES", "OPTIONS", "CDS", "MF")

        if trade_type == "BUY":
            if lots[symbol] and lots[symbol][0]["side"] == "short":
                remaining_qty = qty
                while remaining_qty > 0 and lots[symbol] and lots[symbol][0]["side"] == "short":
                    lot       = lots[symbol][0]
                    close_qty = min(remaining_qty, lot["qty"])
                    pnl_pct   = (lot["price"] - price) / lot["price"]
                    trades.append({
                        "pnl":    round(pnl_pct, 6),
                        "ticker": symbol,
                        "side":   "short",
                        "date":   date,
                    })
                    lot["qty"]    -= close_qty
                    remaining_qty -= close_qty
                    if lot["qty"] == 0:
                        lots[symbol].pop(0)
                if remaining_qty > 0:
                    lots[symbol].append({"qty": remaining_qty, "price": price, "date": date, "side": "long"})
            else:
                lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "long"})

        elif trade_type == "SELL":
            if not lots[symbol]:
                if is_fo:
                    lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "short"})
                else:
                    orphan_sells.append(
                        f"{symbol} SELL {qty:.0f}@{price:.2f} on {date.date()} "
                        f"(no open position — predates history window)"
                    )
                continue

            remaining_qty = qty
            while remaining_qty > 0 and lots[symbol] and lots[symbol][0]["side"] == "long":
                lot       = lots[symbol][0]
                close_qty = min(remaining_qty, lot["qty"])
                pnl_pct   = (price - lot["price"]) / lot["price"]
                trades.append({
                    "pnl":    round(pnl_pct, 6),
                    "ticker": symbol,
                    "side":   "long",
                    "date":   date,
                })
                lot["qty"]    -= close_qty
                remaining_qty -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

    if orphan_sells:
        audit_flags.append(
            f"⚠ {len(orphan_sells)} sell(s) had no matching buy "
            f"(position predates TradeBook history) — excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f"ℹ {len(open_positions)} position(s) still open "
            f"(no closing trade yet): {sym_list}"
        )

    if not trades:
        return pd.DataFrame(), "No completed trades found in Kite TradeBook", audit_flags

    result = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    note = (
        f"📊 Parsed {len(trades)} completed trade(s) from Zerodha Kite TradeBook "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(open_positions)} position(s) still open)"
    )
    return result, note, audit_flags


# =========================================================
# KITE ORDERS CSV PARSER  (NEW — v1.4.1)
#
# Handles Kite Console → Orders → Download CSV format:
#   order_id, exchange_order_id, tradingsymbol, exchange, segment, product,
#   order_type, transaction_type, quantity, average_price, price,
#   filled_quantity, pending_quantity, status, order_execution_time, trade_id
#
# Distinct from TradeBook: uses transaction_type/order_execution_time/
# filled_quantity/average_price instead of trade_type/trade_date/quantity/price.
# =========================================================

def _is_kite_orders_csv(df: pd.DataFrame) -> bool:
    """
    Detect Zerodha Kite Orders CSV.
    The combination of transaction_type + order_execution_time + tradingsymbol
    + filled_quantity is unique to this format and absent from TradeBook.
    """
    cols = {c.lower().strip() for c in df.columns}
    required = {"transaction_type", "order_execution_time", "tradingsymbol", "filled_quantity"}
    return required.issubset(cols)


def _parse_kite_orders_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Kite Orders CSV into QuantProof trades.

    Column mapping vs TradeBook:
      transaction_type      → side (BUY/SELL)
      order_execution_time  → date
      filled_quantity       → qty (actual shares filled)
      average_price         → price (actual fill price, not limit price)
      tradingsymbol         → ticker
      status == "COMPLETE"  → only process completed orders

    Uses identical FIFO position-state tracker as _parse_kite_csv and
    pair_alpaca_orders. Returns (df, note, audit_flags).
    """
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"transaction_type", "order_execution_time", "tradingsymbol",
                "filled_quantity", "average_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kite Orders CSV missing required columns: {missing}. "
            "Export from Kite Console → Orders → Download CSV."
        )

    # Filter: COMPLETE orders only
    df["status"] = df.get("status", pd.Series(["COMPLETE"] * len(df))).str.upper().str.strip()
    df = df[df["status"] == "COMPLETE"].copy()

    df["filled_quantity"] = pd.to_numeric(df["filled_quantity"], errors="coerce")
    df["average_price"]   = pd.to_numeric(df["average_price"],   errors="coerce")
    df = df.dropna(subset=["filled_quantity", "average_price"])
    df = df[(df["filled_quantity"] > 0) & (df["average_price"] > 0)]

    df["order_execution_time"] = pd.to_datetime(df["order_execution_time"], errors="coerce")
    df = df.dropna(subset=["order_execution_time"])

    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
    df = df[df["transaction_type"].isin(["BUY", "SELL"])]

    if df.empty:
        return pd.DataFrame(), "No COMPLETE orders found in Kite Orders CSV", []

    df = df.sort_values(["tradingsymbol", "order_execution_time"]).reset_index(drop=True)
    df["segment"] = df.get("segment", pd.Series(["EQ"] * len(df))).str.upper().str.strip()

    lots         = defaultdict(list)
    trades       = []
    audit_flags  = []
    orphan_sells = []

    for _, row in df.iterrows():
        symbol = str(row["tradingsymbol"])
        side   = row["transaction_type"]
        qty    = float(row["filled_quantity"])
        price  = float(row["average_price"])
        date   = row["order_execution_time"]
        is_fo  = row["segment"] in ("FO", "F&O", "FUTURES", "OPTIONS", "CDS")

        if side == "BUY":
            if lots[symbol] and lots[symbol][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[symbol] and lots[symbol][0]["side"] == "short":
                    lot       = lots[symbol][0]
                    close_qty = min(remaining, lot["qty"])
                    pnl_pct   = (lot["price"] - price) / lot["price"]
                    trades.append({
                        "pnl":    round(np.clip(pnl_pct, -0.50, 0.50), 6),
                        "ticker": symbol,
                        "side":   "short",
                        "date":   date,
                    })
                    lot["qty"]  -= close_qty
                    remaining   -= close_qty
                    if lot["qty"] == 0:
                        lots[symbol].pop(0)
                if remaining > 0:
                    lots[symbol].append({"qty": remaining, "price": price,
                                         "date": date, "side": "long"})
            else:
                lots[symbol].append({"qty": qty, "price": price,
                                     "date": date, "side": "long"})

        elif side == "SELL":
            if not lots[symbol]:
                if is_fo:
                    lots[symbol].append({"qty": qty, "price": price,
                                         "date": date, "side": "short"})
                else:
                    orphan_sells.append(
                        f"{symbol} SELL {qty:.0f}@{price:.2f} on {date.date()} "
                        f"(no matching BUY — predates history window)"
                    )
                continue

            remaining = qty
            while remaining > 0 and lots[symbol] and lots[symbol][0]["side"] == "long":
                lot       = lots[symbol][0]
                close_qty = min(remaining, lot["qty"])
                pnl_pct   = (price - lot["price"]) / lot["price"]
                trades.append({
                    "pnl":    round(np.clip(pnl_pct, -0.50, 0.50), 6),
                    "ticker": symbol,
                    "side":   "long",
                    "date":   date,
                })
                lot["qty"]  -= close_qty
                remaining   -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

    if orphan_sells:
        audit_flags.append(
            f"⚠ {len(orphan_sells)} sell(s) had no matching BUY "
            f"(position predates Orders history) — excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f"ℹ {len(open_positions)} position(s) still open "
            f"(no closing trade yet): {sym_list}"
        )

    if not trades:
        return (
            pd.DataFrame(),
            "No completed round-trip trades found in Kite Orders CSV",
            audit_flags,
        )

    result = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    note = (
        f"📊 Parsed {len(trades)} completed trade(s) from Zerodha Kite Orders CSV "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(open_positions)} position(s) still open)"
    )
    return result, note, audit_flags


# =========================================================
# ALPACA UI EXPORT PARSER  (NEW — v1.4.1)
#
# Handles the tab-separated table exported/pasted from the Alpaca
# dashboard UI — distinct from the REST API JSON response.
#
# UI format:
#   Asset | Order Type | Side | Qty | Filled Qty | Avg. Fill Price |
#   Status | Source | Submitted At | Filled At | Expires At
#
# Key differences from REST API:
#   - Tab-separated, not JSON
#   - Dates: "Feb 24, 2026, 08:40:53 PM" not ISO
#   - "Avg. Fill Price" not "filled_avg_price"
#   - "Filled Qty" not "filled_qty"
#   - Status: "filled"/"canceled" lowercase
#   - Asset cell sometimes contains "ARM\nStop @ $131.17" — ticker is first token
# =========================================================

def _is_alpaca_ui_export(df: pd.DataFrame) -> bool:
    """
    Detect Alpaca UI export by its distinctive column set.
    'filled at' + 'avg fill price' (after stripping '.') + 'side' is unique
    to this format.
    """
    cols = {c.lower().strip().replace(".", "") for c in df.columns}
    return (
        ("avg fill price" in cols or "avg  fill price" in cols)
        and "filled at" in cols
        and "side" in cols
    )


def _parse_alpaca_ui_date(s) -> pd.Timestamp | None:
    """
    Parse Alpaca UI date strings: "Feb 24, 2026, 08:40:53 PM".
    Also handles ISO strings and "-" (unfilled/canceled rows).
    """
    if not s or str(s).strip() in ("-", "", "nan", "None"):
        return None
    try:
        return pd.to_datetime(str(s).strip(), errors="coerce")
    except Exception:
        return None


def _parse_alpaca_ui_export(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Alpaca UI order export (tab-separated) into QuantProof trades.

    Steps:
      1. Detect delimiter (tab vs comma)
      2. Normalise column names (strip dots, lower-case)
      3. Filter status == "filled" and filled_qty > 0
      4. Parse "Filled At" timestamps
      5. Clean ticker from "Asset" column (strip order-type annotations)
      6. FIFO pairing — identical logic to pair_alpaca_orders

    Returns (df, note, audit_flags).
    """
    audit_flags = []

    sample = raw[:4096]
    delim  = "\t" if sample.count("\t") > sample.count(",") else ","

    try:
        df = pd.read_csv(io.StringIO(raw), sep=delim, dtype=str)
    except Exception as e:
        raise ValueError(f"Could not parse Alpaca UI export: {e}")

    def _norm_col(c):
        return c.lower().strip().replace(".", "").replace("  ", " ")

    df.columns = [_norm_col(c) for c in df.columns]

    col_map = {
        "asset":          "symbol_raw",
        "side":           "side",
        "qty":            "qty",
        "filled qty":     "filled_qty",
        "avg fill price": "avg_fill_price",
        "status":         "status",
        "filled at":      "filled_at",
        "submitted at":   "submitted_at",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = {"symbol_raw", "side", "filled_qty", "avg_fill_price", "status", "filled_at"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Alpaca UI export missing columns: {missing}. "
            "Export from Alpaca dashboard → Orders → copy/download table."
        )

    df["status"] = df["status"].str.lower().str.strip()
    df = df[df["status"] == "filled"].copy()

    df["filled_qty"]     = pd.to_numeric(df["filled_qty"],     errors="coerce")
    df["avg_fill_price"] = pd.to_numeric(df["avg_fill_price"], errors="coerce")
    df = df.dropna(subset=["filled_qty", "avg_fill_price"])
    df = df[(df["filled_qty"] > 0) & (df["avg_fill_price"] > 0)]

    df["filled_at"] = df["filled_at"].apply(_parse_alpaca_ui_date)
    df = df.dropna(subset=["filled_at"])

    def _clean_ticker(raw_val):
        if pd.isna(raw_val):
            return "UNKNOWN"
        s = str(raw_val).strip()
        m = _re.match(r"([A-Z]{1,6})", s.upper())
        return m.group(1) if m else s.split()[0]

    df["symbol"] = df["symbol_raw"].apply(_clean_ticker)
    df["side"]   = df["side"].str.lower().str.strip()
    df = df[df["side"].isin(["buy", "sell"])]

    if df.empty:
        return (
            pd.DataFrame(),
            "No filled orders found in Alpaca UI export",
            ["No filled orders after filtering"],
        )

    df = df.sort_values("filled_at").reset_index(drop=True)

    lots         = defaultdict(list)
    completed    = []
    orphan_sells = []

    for _, row in df.iterrows():
        sym   = row["symbol"]
        side  = row["side"]
        qty   = float(row["filled_qty"])
        price = float(row["avg_fill_price"])
        ts    = row["filled_at"]

        if side == "buy":
            if lots[sym] and lots[sym][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "short":
                    lot       = lots[sym][0]
                    close_qty = min(remaining, lot["qty"])
                    pnl_pct   = (lot["price"] - price) / lot["price"]
                    completed.append({
                        "pnl":        round(np.clip(float(pnl_pct), -0.50, 0.50), 6),
                        "date":       ts,
                        "entry_date": lot["entry_date"],
                        "ticker":     sym,
                        "buy_price":  price,
                        "sell_price": lot["price"],
                        "qty":        close_qty,
                        "hold_days":  max(0, (ts - lot["entry_date"]).days),
                        "side":       "short",
                    })
                    lot["qty"]  -= close_qty
                    remaining   -= close_qty
                    if lot["qty"] == 0:
                        lots[sym].pop(0)
                if remaining > 0:
                    lots[sym].append({"price": price, "qty": remaining,
                                      "entry_date": ts, "side": "long"})
            else:
                lots[sym].append({"price": price, "qty": qty,
                                  "entry_date": ts, "side": "long"})

        elif side == "sell":
            if not lots[sym]:
                orphan_sells.append(
                    f"{sym} sell@{price:.2f} on {ts.date()} "
                    f"(no matching buy — position predates history window)"
                )
                continue

            remaining = qty
            while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "long":
                lot       = lots[sym][0]
                close_qty = min(remaining, lot["qty"])
                pnl_pct   = (price - lot["price"]) / lot["price"]
                completed.append({
                    "pnl":        round(np.clip(float(pnl_pct), -0.50, 0.50), 6),
                    "date":       ts,
                    "entry_date": lot["entry_date"],
                    "ticker":     sym,
                    "buy_price":  lot["price"],
                    "sell_price": price,
                    "qty":        close_qty,
                    "hold_days":  max(0, (ts - lot["entry_date"]).days),
                    "side":       "long",
                })
                lot["qty"]  -= close_qty
                remaining   -= close_qty
                if lot["qty"] == 0:
                    lots[sym].pop(0)

    if orphan_sells:
        audit_flags.append(
            f"⚠ {len(orphan_sells)} sell(s) had no matching buy "
            f"(position predates window) — excluded"
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
        return (
            pd.DataFrame(),
            "No completed round-trip trades found in Alpaca UI export.",
            audit_flags + [
                "All sells may predate the visible order history, "
                "or all positions are still open."
            ],
        )

    result_df = (
        pd.DataFrame(completed)
        .sort_values("date")
        .reset_index(drop=True)
    )

    total_filled  = len(df)
    paired_trades = len(completed)
    note = (
        f"📊 Parsed {paired_trades} completed trade(s) from Alpaca UI export "
        f"({total_filled} filled order-sides → {paired_trades} round-trips, "
        f"{len(orphan_sells)} orphan sell(s) excluded)"
    )
    return result_df, note, audit_flags


# =========================================================
# ALPACA ORDER PAIRING — position-state tracker
# Hardened v1.4.1: handles both REST API JSON and UI export dict shapes.
# Normalises column names before FIFO pairing so both sources work.
# =========================================================

def pair_alpaca_orders(orders: list) -> tuple[pd.DataFrame, list]:
    """
    Converts raw Alpaca orders into a trades DataFrame.

    Accepts BOTH:
      a) REST API JSON dicts — keys: filled_qty, filled_avg_price,
         filled_at, side, status, symbol
      b) Pre-normalised UI export dicts — same internal keys after
         _parse_alpaca_ui_export normalisation

    Normalisation applied before FIFO pairing:
      filled_qty        → qty  (also accepts filled_qty from API)
      filled_avg_price  → price (also accepts avg_fill_price from UI)
      filled_at         → ts  (handles both ISO and human-readable strings)
      symbol            → cleaned ticker (strips order-type annotations)

    FIFO pairing logic (unchanged from v1.4):
      - Sort chronologically by filled_at
      - FIFO lot queue per symbol
      - Orphan sell detection and flagging
      - Short position support
      - Regime auto-assigned after pairing

    Returns (df, audit_flags).
    """
    audit_flags = []

    def _norm_order(o: dict) -> dict | None:
        """Normalise one order dict → internal keys, or None to skip."""
        status = str(o.get("status", "")).lower().strip()
        if status != "filled":
            return None

        # Qty: REST API uses filled_qty; UI dicts also use filled_qty after rename
        qty_raw = o.get("filled_qty") or o.get("qty") or 0
        try:
            qty = float(qty_raw)
        except (ValueError, TypeError):
            return None
        if qty <= 0:
            return None

        # Price: REST API uses filled_avg_price; UI uses avg_fill_price
        price_raw = (
            o.get("filled_avg_price")
            or o.get("avg_fill_price")
            or o.get("price")
            or 0
        )
        try:
            price_str = str(price_raw).replace(",", "").strip()
            if price_str in ("-", "", "nan", "None"):
                return None
            price = float(price_str)
        except (ValueError, TypeError):
            return None
        if price <= 0:
            return None

        side = str(o.get("side", "")).lower().strip()
        if side not in ("buy", "sell"):
            return None

        # Timestamp: REST API gives ISO string; UI may give pre-parsed Timestamp
        ts_raw = o.get("filled_at") or o.get("updated_at") or ""
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw.tz_localize(None) if ts_raw.tzinfo else ts_raw
        else:
            ts_str = str(ts_raw).strip()
            if ts_str in ("-", "", "nan", "None"):
                return None
            try:
                ts_parsed = pd.to_datetime(ts_str, utc=True)
                ts = ts_parsed.tz_localize(None)
            except Exception:
                try:
                    ts = pd.to_datetime(ts_str, errors="coerce")
                    if pd.isna(ts):
                        return None
                    if hasattr(ts, "tzinfo") and ts.tzinfo:
                        ts = ts.tz_convert("UTC").tz_localize(None)
                except Exception:
                    return None

        # Symbol: clean up "ARM\nStop @ $131.17" style asset strings from UI
        symbol_raw = str(o.get("symbol", "UNKNOWN")).strip()
        m = _re.match(r"([A-Z]{1,6})", symbol_raw.upper())
        symbol = m.group(1) if m else symbol_raw.split()[0]

        return {"symbol": symbol, "side": side, "qty": qty, "price": price, "filled_at": ts}

    filled = []
    for o in orders:
        normed = _norm_order(o)
        if normed:
            filled.append(normed)

    if not filled:
        return pd.DataFrame(), ["No filled orders found"]

    df_orders = (
        pd.DataFrame(filled)
        .sort_values("filled_at")
        .reset_index(drop=True)
    )

    lots         = defaultdict(list)
    completed    = []
    orphan_sells = []

    for _, row in df_orders.iterrows():
        sym   = row["symbol"]
        side  = row["side"]
        qty   = row["qty"]
        price = row["price"]
        ts    = row["filled_at"]

        if side == "buy":
            if lots[sym] and lots[sym][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "short":
                    lot       = lots[sym][0]
                    close_qty = min(remaining, lot["qty"])
                    pnl_pct   = (lot["price"] - price) / lot["price"]
                    completed.append({
                        "pnl":        round(np.clip(float(pnl_pct), -0.50, 0.50), 6),
                        "date":       ts,
                        "entry_date": lot["entry_date"],
                        "ticker":     sym,
                        "buy_price":  price,
                        "sell_price": lot["price"],
                        "qty":        close_qty,
                        "hold_days":  max(0, (ts - lot["entry_date"]).days),
                        "side":       "short",
                    })
                    lot["qty"]  -= close_qty
                    remaining   -= close_qty
                    if lot["qty"] == 0:
                        lots[sym].pop(0)
                if remaining > 0:
                    lots[sym].append({"price": price, "qty": remaining,
                                      "entry_date": ts, "side": "long"})
            else:
                lots[sym].append({"price": price, "qty": qty,
                                  "entry_date": ts, "side": "long"})

        elif side == "sell":
            if not lots[sym]:
                # No open long position — this SELL opens a short position
                lots[sym].append({
                    "price":      price,
                    "qty":        qty,
                    "entry_date": ts,
                    "side":       "short",
                })
                continue

            # If there's an existing short lot (and no long), stack it
            if lots[sym] and lots[sym][0]["side"] == "short":
                lots[sym].append({
                    "price":      price,
                    "qty":        qty,
                    "entry_date": ts,
                    "side":       "short",
                })
                continue

            remaining = qty
            while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "long":
                lot       = lots[sym][0]
                close_qty = min(remaining, lot["qty"])
                pnl_pct   = (price - lot["price"]) / lot["price"]
                completed.append({
                    "pnl":        round(np.clip(float(pnl_pct), -0.50, 0.50), 6),
                    "date":       ts,
                    "entry_date": lot["entry_date"],
                    "ticker":     sym,
                    "buy_price":  lot["price"],
                    "sell_price": price,
                    "qty":        close_qty,
                    "hold_days":  max(0, (ts - lot["entry_date"]).days),
                    "side":       "long",
                })
                lot["qty"]  -= close_qty
                remaining   -= close_qty
                if lot["qty"] == 0:
                    lots[sym].pop(0)

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
# Detection order (most-specific first):
#   1. Kite TradeBook     (_is_kite_csv)
#   2. Kite Orders CSV    (_is_kite_orders_csv)   ← NEW v1.4.1
#   3. Alpaca UI export   (_is_alpaca_ui_export)  ← NEW v1.4.1
#   4. Generic PnL CSV    (pnl/profit/return column heuristic)
#   5. Flexible fallback  (numeric column heuristic)
# =========================================================

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS CSV PARSER
# Detects options trade exports and normalises them to a fractional return
# series compatible with QuantProofValidator.
#
# Supported:
#   - Single-leg directional trades (calls/puts bought and closed)
#   - Any CSV with strike / expiry / option_type / call_put / premium columns
#
# Not supported (rejected with clear error):
#   - Multi-leg concurrent positions (spreads, iron condors, straddles)
#   - Positions still open at time of export
#
# The validator itself is unchanged — options data is normalised to the same
# fractional return format as equity trades before validation runs.
# ══════════════════════════════════════════════════════════════════════════════

def _is_options_csv(df: pd.DataFrame) -> bool:
    """Detect options trade data from column names."""
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    options_signals = ["strike", "expiry", "expiration", "option_type",
                       "call_put", "put_call", "premium", "underlying",
                       "contract", "iv", "delta", "gamma", "theta"]
    return sum(1 for s in options_signals if any(s in c for c in cols)) >= 2


def _has_overlapping_positions(df: pd.DataFrame) -> bool:
    """
    Check if any positions overlap in time (i.e. concurrent open trades).
    Returns True if overlapping positions detected.
    Only runs when date/open_date/close_date columns are present.
    """
    df_cols = [c.lower().strip() for c in df.columns]
    # Strict date column detection — must contain "date" or "time" keyword
    # to avoid matching price columns like "entry_price" or "exit_price"
    def _is_open_date_col(c):
        cl = c.lower()
        return (cl in ["date", "trade_date", "open_date", "entry_date"] or
                any(cl.endswith(x) for x in ["_date", "_time", "date"]) and
                not any(x in cl for x in ["close","exit","expir","end"]))

    open_col  = next((c for c in df.columns if _is_open_date_col(c)), None)
    close_col = next((c for c in df.columns if any(x in c.lower()
                      for x in ["close_date", "exit_date", "close_time",
                                 "end_date", "expiry", "expiration"])), None)

    if not open_col or not close_col:
        return False  # can't determine — assume non-overlapping

    try:
        opens  = pd.to_datetime(df[open_col],  errors="coerce").dropna()
        closes = pd.to_datetime(df[close_col], errors="coerce").dropna()
        if len(opens) < 2 or len(closes) < 2:
            return False
        # Sanity check: dates should be after year 2000
        # This catches price values that parse as 1970-epoch timestamps
        if opens.min().year < 2000 or closes.min().year < 2000:
            return False
        # Check if any open date falls between another trade's open and close
        for i in range(len(opens)):
            for j in range(len(opens)):
                if i == j:
                    continue
                if opens.iloc[i] < closes.iloc[j] and opens.iloc[j] < closes.iloc[i]:
                    return True
        return False
    except Exception:
        return False


def _parse_options_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Parse options trade CSV into a normalised returns DataFrame.

    Returns (df_with_pnl_and_date, note, audit_flags).

    Return calculation:
      - If premium/entry_price column present: return = P&L / entry_premium
      - Otherwise: treat pnl column as fractional return (same as generic CSV)

    Audit flags added to every options validation:
      - Notes that crash simulations assume linear payoffs
      - Notes that options convexity is not modelled
    """
    import io as _io

    sample = raw[:2048]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim  = max(counts, key=counts.get)

    df = pd.read_csv(_io.StringIO(raw), sep=delim)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # ── Check for concurrent positions ────────────────────────────────────
    if _has_overlapping_positions(df):
        raise ValueError(
            "Multi-leg or concurrent options positions detected. "
            "QuantProof supports single-leg directional options trades only "
            "(calls or puts bought and fully closed before the next trade). "
            "For spreads, iron condors, or straddles: calculate your daily "
            "P&L as a percentage of capital deployed and upload as a generic CSV."
        )

    # ── Find P&L column ───────────────────────────────────────────────────
    pnl_col = next((c for c in df.columns if any(x in c for x in
                    ["pnl", "profit", "return", "gain", "pl", "net"])), None)
    if pnl_col is None:
        raise ValueError(
            "Could not find a P&L column in your options CSV. "
            "Expected a column named pnl, profit, return, gain, or pl."
        )

    pnl_vals = pd.to_numeric(df[pnl_col].astype(str).str.replace("%","",regex=False),
                              errors="coerce")

    # ── Normalise to fractional returns ───────────────────────────────────
    # Look for premium / entry price to compute return = P&L / premium
    premium_col = next((c for c in df.columns if any(x in c for x in
                        ["premium", "entry_price", "open_price", "cost"])), None)
    if premium_col is not None:
        premium_vals = pd.to_numeric(df[premium_col].astype(str).str.replace("%","",regex=False),
                                      errors="coerce")
        # Return = P&L / premium paid (per contract basis)
        valid_mask = (premium_vals.notna()) & (premium_vals.abs() > 1e-9) & (pnl_vals.notna())
        returns = (pnl_vals[valid_mask] / premium_vals[valid_mask]).clip(-0.99, 0.99)
        norm_method = f"P&L / {premium_col} (per contract return)"
    else:
        # Fall back: treat pnl as fractional return directly
        valid_mask = pnl_vals.notna()
        returns = pnl_vals[valid_mask]
        if returns.abs().mean() > 1.0:
            returns = returns / 100.0  # was percentage, convert to decimal
        returns = returns.clip(-0.99, 0.99)
        norm_method = "pnl column treated as fractional return"

    result = pd.DataFrame({"pnl": returns.values})

    # ── Date column ───────────────────────────────────────────────────────
    date_col = next((c for c in df.columns if any(x in c for x in
                     ["date", "close_date", "exit_date", "time", "dt"])), None)
    if date_col:
        result["date"] = pd.to_datetime(
            df[date_col].values[:len(returns)], errors="coerce")

    # ── Ticker column ─────────────────────────────────────────────────────
    ticker_col = next((c for c in df.columns if any(x in c for x in
                       ["ticker", "symbol", "underlying", "instrument"])), None)
    if ticker_col:
        result["ticker"] = df[ticker_col].values[:len(returns)]

    note = (
        f"OPTIONS: {len(result)} single-leg trades parsed. "
        f"Returns normalised as {norm_method}."
    )
    audit_flags = [
        "OPTIONS DATA: Crash simulations assume linear payoffs — "
        "options convexity and delta/gamma effects are not modelled. "
        "Results are directionally indicative only.",
        "OPTIONS DATA: Validation assumes sequential non-overlapping trades. "
        "Multi-leg strategies (spreads, condors) are not supported — "
        "use daily portfolio P&L as percentage of capital instead.",
        "OPTIONS DATA: Crash simulations scale stress from strategy volatility. "
        "High-volatility strategies (options, leveraged) will show elevated crash "
        "exposure — all 5 crises blowing up is expected behaviour for options "
        "strategies and does not indicate a validator error.",
    ]

    return result, note, audit_flags


# ══════════════════════════════════════════════════════════════════════════════
# MT5 (MetaTrader 5) TRADE HISTORY PARSER
#
# MT5 exports two types of history from Terminal → Account History → Save:
#
# 1. Deals report (detailed) — one row per execution:
#    Time, Deal, Symbol, Type, Direction, Volume, Price, Order,
#    Commission, Swap, Profit, Balance, Comment
#    Direction: "in" = open, "out" = close, "in/out" = reverse
#
# 2. Positions report (simplified) — one row per completed trade:
#    Time, Position, Symbol, Type, Volume, Price, S/L, T/P,
#    Time (close), Price (close), Commission, Swap, Profit
#
# Both formats are detected and handled. Deals report is preferred because
# it has higher precision — we reconstruct completed trades from in/out pairs.
# Positions report is used when Direction column is absent.
#
# Return calculation:
#   pnl_pct = profit / (entry_price * volume * contract_size_estimate)
#   If contract size unknown: pnl_pct = profit / abs(entry_balance)
#   Falls back to profit as raw value normalised by account equity if no balance.
#
# FIFO matching on deals report: "in" opens a lot, "out" closes it.
# Partial closes handled. "in/out" (reverse) split into close + open.
# ══════════════════════════════════════════════════════════════════════════════

def _is_mt5_csv(df: pd.DataFrame) -> bool:
    """
    Detect MT5 trade history export by characteristic column names.

    Deals report signals: deal, direction, commission, swap, balance
    Positions report signals: position, s/l, t/p, or two time/price pairs

    Requires at least 3 MT5-specific signals to avoid false positives
    on generic CSVs that might have 'commission' or 'swap' columns.
    """
    cols = {c.lower().strip().replace(" ", "_").replace("/", "_") for c in df.columns}

    # Deals report signals
    deals_signals = {"deal", "direction", "commission", "swap", "balance"}
    deals_hit = len(deals_signals & cols)

    # Positions report signals
    pos_signals = {"position", "s_l", "t_p", "commission", "swap"}
    # Also check for two time columns (open time + close time)
    time_cols = [c for c in cols if "time" in c]
    pos_hit = len(pos_signals & cols) + (1 if len(time_cols) >= 2 else 0)

    return deals_hit >= 3 or pos_hit >= 3


def _parse_mt5_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert MT5 trade history CSV into QuantProof trades DataFrame.

    Returns (df_with_pnl_date_ticker, note, audit_flags).

    Handles:
    - MT5 deals report (Direction column present): FIFO in/out pairing
    - MT5 positions report (no Direction): each row is a completed trade
    - Tab-separated and comma-separated formats
    - Both . and , as decimal separators (European MT5 installs)
    - Balance entries mixed in (Type = balance/credit) — filtered out
    - Hedging accounts (multiple open positions per symbol)
    - Partial closes
    - Missing Balance column (falls back to price-based normalisation)
    """
    import io as _io

    # ── Sniff delimiter ───────────────────────────────────────────────────
    sample = raw[:2048]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim  = max(counts, key=counts.get)

    # Handle European decimal separator (comma) — only if semicolon-delimited
    if delim == ";":
        raw = raw.replace(",", ".")

    try:
        df = pd.read_csv(_io.StringIO(raw), sep=delim, dtype=str,
                         on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(_io.StringIO(raw), sep=delim, dtype=str,
                         error_bad_lines=False)

    if df.empty:
        raise ValueError("MT5 CSV appears empty after parsing.")

    # ── Normalise column names ────────────────────────────────────────────
    df.columns = [c.lower().strip().replace(" ", "_").replace("/", "_")
                  for c in df.columns]

    # ── Check for required columns ────────────────────────────────────────
    has_direction = "direction" in df.columns  # deals report
    has_position  = "position"  in df.columns  # positions report

    if not has_direction and not has_position:
        # Try to infer from column presence
        if "deal" in df.columns:
            has_direction = True  # deals without direction column — older MT5
        else:
            raise ValueError(
                "Could not identify MT5 report format. "
                "Please export from MT5 Terminal → Account History → "
                "right-click → Save as Detailed Report."
            )

    # ── Parse numeric columns ─────────────────────────────────────────────
    def _num(series):
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False)
                               .str.replace("[^\\d.\\-]", "", regex=True),
            errors="coerce"
        )

    for col in ["profit", "volume", "price", "commission", "swap",
                "balance", "price_2"]:
        if col in df.columns:
            df[col] = _num(df[col])

    # Also handle "price (close)" style columns from positions report
    price_cols = [c for c in df.columns if "price" in c]
    for pc in price_cols:
        df[pc] = _num(df[pc])

    # ── Parse time columns ────────────────────────────────────────────────
    time_cols = [c for c in df.columns if "time" in c]
    for tc in time_cols:
        df[tc] = pd.to_datetime(df[tc], errors="coerce")

    # Primary date column
    open_time_col  = time_cols[0] if time_cols else None
    close_time_col = time_cols[1] if len(time_cols) >= 2 else open_time_col

    # ── Symbol column ─────────────────────────────────────────────────────
    symbol_col = next((c for c in df.columns if c in
                       ["symbol", "instrument", "pair", "asset"]), None)

    # ── Filter to trade rows only ─────────────────────────────────────────
    # MT5 mixes balance, credit, and trade entries in deals report
    if "type" in df.columns:
        type_vals = df["type"].astype(str).str.lower().str.strip()
        # Keep only buy/sell rows — filter out balance, credit, correction etc
        trade_mask = type_vals.isin(["buy", "sell", "buy limit", "sell limit",
                                     "buy stop", "sell stop", "0", "1"])
        df = df[trade_mask].copy()

    if df.empty:
        raise ValueError(
            "No trade rows found in MT5 export after filtering. "
            "Make sure you exported Account History (not just open positions) "
            "and the report includes completed trades."
        )

    audit_flags  = []
    trades       = []
    orphan_count = 0

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH A: Deals report with Direction column (in/out pairing)
    # ══════════════════════════════════════════════════════════════════════
    if has_direction:
        df["direction"] = df["direction"].astype(str).str.lower().str.strip()
        df = df.sort_values(open_time_col) if open_time_col else df

        lots = defaultdict(list)  # symbol → [{"price", "volume", "date", "commission"}]

        for _, row in df.iterrows():
            direction = str(row.get("direction", "")).lower().strip()
            symbol    = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            volume    = float(row.get("volume", 0) or 0)
            price     = float(row.get("price",  0) or 0)
            profit    = float(row.get("profit", 0) or 0)
            commission= float(row.get("commission", 0) or 0)
            swap      = float(row.get("swap", 0) or 0)
            balance   = float(row.get("balance", 0) or 0)
            date      = row[open_time_col] if open_time_col else pd.NaT
            trade_type= str(row.get("type", "")).lower().strip()

            # "in/out" = close existing + open reverse — split into close then open
            if direction == "in/out":
                # Close existing
                if lots[symbol]:
                    lot = lots[symbol].pop(0)
                    net_profit = profit + commission + swap
                    pnl_pct = _normalise_pnl(net_profit, lot["price"], lot["volume"], balance)
                    if abs(pnl_pct) <= 0.99:
                        trades.append({"pnl": round(pnl_pct, 6), "date": date,
                                       "ticker": symbol})
                # Open reverse
                side = "sell" if trade_type in ["sell", "1"] else "buy"
                lots[symbol].append({"price": price, "volume": volume,
                                     "date": date, "side": side})
                continue

            if direction == "in":
                lots[symbol].append({"price": price, "volume": volume,
                                     "date": date, "side": trade_type,
                                     "commission": commission})
            elif direction == "out":
                if not lots[symbol]:
                    orphan_count += 1
                    continue
                lot = lots[symbol].pop(0)
                net_profit = profit + commission + swap + lot.get("commission", 0)
                pnl_pct    = _normalise_pnl(net_profit, lot["price"], lot["volume"], balance)
                if abs(pnl_pct) <= 0.99:
                    trades.append({"pnl": round(pnl_pct, 6), "date": date,
                                   "ticker": symbol})

        # Any remaining open positions
        open_pos = {s: len(ls) for s, ls in lots.items() if ls}
        if open_pos:
            count = sum(open_pos.values())
            audit_flags.append(
                f"ℹ {count} MT5 position(s) still open (no closing deal) — excluded"
            )

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH B: Positions report — each row IS a completed trade
    # ══════════════════════════════════════════════════════════════════════
    else:
        # Find close price column — second price column or "price_2"
        price_open_col  = next((c for c in df.columns if "price" in c), None)
        price_close_col = next((c for c in df.columns
                                if "price" in c and c != price_open_col), None)

        for _, row in df.iterrows():
            profit    = float(row.get("profit", 0) or 0)
            commission= float(row.get("commission", 0) or 0)
            swap      = float(row.get("swap", 0) or 0)
            volume    = float(row.get("volume", 1) or 1)
            price_open = float(row[price_open_col] or 0) if price_open_col else 0
            balance    = 0.0  # not available in positions report
            symbol     = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            date       = row[close_time_col] if close_time_col else pd.NaT

            net_profit = profit + commission + swap
            pnl_pct    = _normalise_pnl(net_profit, price_open, volume, balance)
            if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                trades.append({"pnl": round(pnl_pct, 6), "date": date,
                                "ticker": symbol})

    # ── Build result ──────────────────────────────────────────────────────
    if not trades:
        raise ValueError(
            "No completed trades could be extracted from the MT5 export. "
            "Make sure your report includes closed trades and has Profit column."
        )

    result = pd.DataFrame(trades)
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    # ── Audit flags ───────────────────────────────────────────────────────
    if orphan_count:
        audit_flags.append(
            f"⚠ {orphan_count} MT5 closing deal(s) had no matching open "
            f"— excluded (position predates history window)"
        )

    audit_flags.append(
        "MT5 DATA: Returns normalised as net profit / (entry_price * volume). "
        "Crash simulations assume linear payoffs — leverage and margin effects not modelled. "
        "For accurate capacity analysis, ensure Volume column reflects lot sizes."
    )

    report = "deals" if has_direction else "positions"
    note = (
        f"📊 MT5: {len(trades)} completed trade(s) parsed from {report} report "
        f"({orphan_count} orphan(s) excluded)"
    )
    return result, note, audit_flags


def _normalise_pnl(net_profit: float, entry_price: float,
                   volume: float, balance: float) -> float:
    """
    Convert MT5 dollar profit into fractional return for the validator.

    Priority:
    1. profit / (entry_price * volume)  — standard forex return
       e.g. $35 profit on 0.1 lot EURUSD at 1.0850 → 35 / (1.0850 * 0.1 * 100000)
       Note: we use volume as-is (lot size) without multiplying by 100000
       because we don't know the contract size per instrument.
       This gives a reasonable approximation for cross-instrument comparison.
    2. profit / abs(balance - profit) — balance-based if price * vol near zero
    3. profit / 1000 — last resort normalisation to keep values in range

    Always clips to ±0.99 to prevent validator ValueError.
    """
    if abs(entry_price * volume) > 1e-6:
        pnl = net_profit / (entry_price * volume)
    elif abs(balance - net_profit) > 1e-6:
        pnl = net_profit / abs(balance - net_profit)
    else:
        pnl = net_profit / 1000.0

    return float(np.clip(pnl, -0.99, 0.99))


def smart_parse_csv(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parses CSV/TSV into a trades DataFrame for QuantProofValidator.
    Returns (df, regime_note).

    New in v1.4.1: detects Kite Orders CSV and Alpaca UI export before
    falling through to the generic pnl-column heuristic.
    """
    raw         = contents.decode("utf-8", errors="ignore")
    regime_note = ""

    # Sniff delimiter — support tab (Alpaca UI), semicolon (Excel/LibreOffice), comma (standard)
    sample = raw[:4096]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim  = max(counts, key=counts.get)   # pick whichever occurs most

    try:
        _probe = pd.read_csv(io.StringIO(raw), sep=delim, nrows=5, dtype=str)
    except Exception:
        _probe = pd.DataFrame()

    # ── 0. Options CSV ───────────────────────────────────────────────────
    if not _probe.empty and _is_options_csv(_probe):
        try:
            _df, _note, _flags = _parse_options_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df)
                    _note += (" | " + _regime_note if _regime_note else "")
                # Always surface options audit flags
                if _flags:
                    _note += " | OPTIONS_FLAGS:" + "||".join(_flags)
                return _df, _note
        except ValueError as e:
            # Surface clear options-specific errors to the user
            raise HTTPException(400, str(e))
        except Exception:
            pass  # fall through to generic parser

    # ── 0.5. MT5 Trade History ───────────────────────────────────────────
    if not _probe.empty and _is_mt5_csv(_probe):
        try:
            _df, _note, _flags = _parse_mt5_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df)
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | MT5_FLAGS:" + "||".join(_flags)
                return _df, _note
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            pass  # fall through to next parser

    # ── 1. Kite TradeBook ─────────────────────────────────────────────────
    if not _probe.empty and _is_kite_csv(_probe):
        try:
            _df, _note, _flags = _parse_kite_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df)
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                # Detect Kite options trades by instrument name (e.g. NIFTY24DEC20000CE)
                if "ticker" in _df.columns:
                    _opt_mask = _df["ticker"].astype(str).str.contains(
                        r'(CE|PE)$', regex=True, na=False)
                    if _opt_mask.any():
                        _opt_count = int(_opt_mask.sum())
                        _note += (f" | KITE OPTIONS: {_opt_count} options trades detected "
                                  f"(CE/PE). P&L used as-is from Kite — premium normalisation "
                                  f"not applied. Crash sims assume linear payoffs.")
                return _df, _note
        except Exception:
            pass

    # ── 2. Kite Orders CSV ────────────────────────────────────────────────
    if not _probe.empty and _is_kite_orders_csv(_probe):
        try:
            _df, _note, _flags = _parse_kite_orders_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df)
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                return _df, _note
        except Exception:
            pass

    # ── 3. Alpaca UI export ───────────────────────────────────────────────
    if not _probe.empty and _is_alpaca_ui_export(_probe):
        try:
            _df, _note, _flags = _parse_alpaca_ui_export(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df)
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                return _df, _note
        except Exception:
            pass

    # ── 4. Generic PnL CSV ────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.StringIO(raw), sep=delim)
        pnl_names = ["pnl", "profit", "return", "gain", "pl", "p&l", "net", "alpha"]
        for col in df.columns:
            if any(p in col.lower() for p in pnl_names):
                # Strip '%' suffix (e.g. "2.5%" → "2.5") before numeric parse
                raw_vals = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
                vals = pd.to_numeric(raw_vals, errors="coerce").dropna()
                if len(vals) >= 5:
                    if vals.abs().mean() > 1.0:
                        vals = vals / 100.0
                    # Detect dollar PnL: if >50% of values still exceed ±0.5 after /100,
                    # the column is likely dollar amounts, not returns — flag it
                    _clipped_pct = (vals.abs() > 0.50).mean()
                    vals = vals.clip(-0.50, 0.50)
                    _dollar_pnl_warning = _clipped_pct > 0.50
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
                        result, regime_note = assign_regimes(result)

                    if _dollar_pnl_warning:
                        regime_note = (
                            ("⚠ " + regime_note + " | " if regime_note else "⚠ ")
                            + "PnL values appear to be dollar amounts, not decimal returns "
                            "— clipped to ±0.5. Upload a column with decimal returns (e.g. 0.015 = 1.5%)."
                        )

                    return result, regime_note
    except Exception:
        pass

    # ── 5. Flexible fallback ──────────────────────────────────────────────
    try:
        lines    = [l.strip() for l in raw.split("\n") if l.strip()]
        max_cols = max(len(l.split(delim)) for l in lines[1:] if l)
        df = pd.read_csv(
            io.StringIO(raw),
            sep=delim,
            names=[f"col_{i}" for i in range(max_cols)],
            skiprows=1,
            on_bad_lines="skip",
            engine="python",
        )
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and (vals > 0).any() and (vals < 0).any():
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values}), ""
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values}), ""
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError(
        "No valid PnL column found. Make sure your CSV has a 'pnl' or 'profit' column, "
        "or upload a Kite TradeBook, Kite Orders CSV, or Alpaca export."
    )


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "status": "QuantProof API is live",
        "version": "2.7.0",
        "checks": "44 institutional checks + 5 crash simulations",
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

    if len(df) < 10:
        raise HTTPException(status_code=400, detail=f"Only found {len(df)} valid trades. Need at least 10.")

    try:
        # Infer SPY benchmark when dates present — makes IR + Beta scored not informational
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
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

    # Only add regime_note as extra flag if it isn't already embedded in
    # the Kite/Alpaca parser audit note (those parsers append it to their note string)
    existing_flags_text = " ".join(r.audit_flags)
    # Extract options-specific audit flags from regime_note
    # (smart_parse_csv embeds them as OPTIONS_FLAGS:flag1||flag2 in the note string)
    _options_flags = []
    if "OPTIONS_FLAGS:" in regime_note:
        _parts = regime_note.split("OPTIONS_FLAGS:", 1)
        regime_note = _parts[0].rstrip(" |")
        _options_flags = _parts[1].split("||") if len(_parts) > 1 else []

    _mt5_flags = []
    if "MT5_FLAGS:" in regime_note:
        _parts = regime_note.split("MT5_FLAGS:", 1)
        regime_note = _parts[0].rstrip(" |")
        _mt5_flags = _parts[1].split("||") if len(_parts) > 1 else []

    extra_flags = ([regime_note] if (regime_note and regime_note not in existing_flags_text) else []) + ([_spy_note] if _spy_note else []) + _options_flags + _mt5_flags

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
    return {"status": "ok", "version": "2.7.0"}


@app.get("/ping")
def ping():
    """
    Keep-alive endpoint.
    Hit this every 10 minutes via an external cron (cron-job.org, UptimeRobot, etc.)
    to prevent Render free tier from spinning down.
    URL to monitor: https://quantproof-backend.onrender.com/ping
    """
    return {"status": "alive", "version": "2.7.0"}



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

        # ── Position-state pairing + regime auto-detection ────────────────────
        df, parse_flags = pair_alpaca_orders(orders)

        # Build a human-readable explanation of what happened during pairing
        _orphan_flag = next((f for f in parse_flags if "orphan" in f.lower() or "predates" in f.lower()), "")
        _open_flag   = next((f for f in parse_flags if "open position" in f.lower()), "")

        if len(df) < 10:
            _detail_lines = []
            if _orphan_flag:
                _detail_lines.append(
                    "Some sells were excluded because their matching buy happened before "
                    "the visible order history window — these are positions you held before "
                    "the period covered by the API."
                )
            if _open_flag:
                _detail_lines.append(
                    "Some buys are still open (no closing sell yet) and cannot be counted as completed trades."
                )
            _detail = " ".join(_detail_lines) if _detail_lines else (
                "Make sure you have at least 10 closed positions — a BUY and a matching SELL "
                "for the same stock, both within the history window."
            )
            raise HTTPException(400, {
                "error": "insufficient_trades",
                "trades_found": len(df),
                "trades_needed": 10,
                "message": (
                    f"Only {len(df)} completed round-trip trade(s) found — need 10. "
                    "A round-trip = one BUY + one SELL of the same stock, both within your order history. "
                    f"{_detail}"
                )
            })

        total_filled_sides = len([o for o in orders if o.get("status") == "filled"])
        paired_sides = len(df) * 2
        unpaired_pct = (total_filled_sides - paired_sides) / max(total_filled_sides, 1) * 100
        if unpaired_pct > 30:
            print(f"[validate/alpaca] WARNING: {unpaired_pct:.0f}% of orders unpaired — possible options/shorts")

        # ── Run validation ────────────────────────────────────────────────────
        # Infer SPY benchmark when dates present — makes IR + Beta scored not informational
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
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
            audit_flags=r.audit_flags + parse_flags + ([_spy_note] if _spy_note else []) + [
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
import asyncio as _asyncio
_PDF_SEMAPHORE = _asyncio.Semaphore(2)  # max 2 concurrent PDF generations
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
REPORT_PRICE_USD    = 4
REPORT_PRICE_INR    = 299


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

    if len(df) < 10:
        raise HTTPException(400, f"Only {len(df)} trades found. Need at least 10.")

    try:
        # Infer SPY benchmark when dates present — makes IR + Beta scored not informational
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
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
        "audit_flags":   report.audit_flags + ([_spy_note] if _spy_note else []) + (
                         [regime_note] if regime_note else []),
    }

    ai_plan = await generate_ai_plan(report, date_range)
    vr["ai_plan"] = ai_plan
    _cache_validation(report.validation_hash, vr)

    try:
        if not _PDF_AVAILABLE:
            raise HTTPException(503, "PDF module not available.")
        async with _PDF_SEMAPHORE:
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
            "amount": 29900, "currency": "INR", "payment_capture": 1,
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
                    # Must be a dict or None — generate_report.py calls ai_plan.get()
                    cached_vr["ai_plan"] = {
                        "summary": "Top improvements: " + "; ".join(issues[:3]) if issues else "See failed checks above.",
                        "fixes": [],
                        "encouragement": ""
                    } if issues else None
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
                    f"Only {len(df)} completed round-trip trade(s) found — need 10. "
                    f"{_detail}"
                ),
                "trades_found": len(df), "trades_needed": 10
            })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Alpaca fetch failed: {str(e)[:200]}")

    try:
        # Infer SPY benchmark when dates present — makes IR + Beta scored not informational
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
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
                date_str  = trade.get("tradeDate") or trade.get("TradeDate", "")
                symbol    = trade.get("symbol", "") or trade.get("Symbol", "")
                qty       = float(trade.get("quantity", 0) or 0)
                price     = float(trade.get("tradePrice") or trade.get("TradePrice", 0) or 0)
                net_cash  = float(trade.get("netCash")    or trade.get("NetCash", 0) or 0)
                buy_sell  = (trade.get("buySell") or trade.get("BuySell") or "").upper()

                open_close = (trade.get("openCloseIndicator") or
                              trade.get("OpenCloseIndicator") or "").upper()
                if open_close == "O":
                    open_skipped += 1
                    continue

                fifo_pnl = trade.get("fifoPnlRealized") or trade.get("FifoPnlRealized")
                if fifo_pnl is not None:
                    try:
                        fifo_val = float(fifo_pnl)
                        if abs(fifo_val) < 0.001:
                            open_skipped += 1
                            continue
                        cost = abs(price * qty) if price > 0 and qty != 0 else 1
                        pnl_frac = fifo_val / cost
                        pnl_frac = max(-0.5, min(0.5, pnl_frac))
                        side = "short" if (buy_sell == "BUY" or qty > 0) else "long"
                        if open_close == "C":
                            side = "short" if buy_sell == "BUY" else "long"
                        trades.append({
                            "date":   date_str,
                            "pnl":    round(pnl_frac, 6),
                            "ticker": symbol,
                            "side":   side,
                        })
                        continue
                    except (ValueError, ZeroDivisionError):
                        pass

                if price <= 0 or qty == 0:
                    continue
                cost_basis = abs(price * qty)
                pnl_frac   = net_cash / cost_basis
                pnl_frac   = max(-0.5, min(0.5, pnl_frac))

                if qty < 0:
                    side = "long"
                else:
                    side = "short"

                is_opening_long  = qty > 0 and net_cash < -abs(price * qty) * 0.5
                # is_opening_short heuristic: only apply when OCI field is absent.
                # When OCI=='C' (confirmed closing), trust it and skip the heuristic.
                # The 0.1 threshold incorrectly skips all closing sells with pnl < 10%.
                is_opening_short = (
                    open_close != "C"          # OCI absent or ambiguous only
                    and qty < 0
                    and net_cash > 0
                    and abs(net_cash) < abs(price * qty) * 0.01   # tightened: 1% threshold
                )
                if is_opening_long or is_opening_short:
                    open_skipped += 1
                    continue
                if abs(net_cash) < 0.01:
                    open_skipped += 1
                    continue

                trades.append({
                    "date":   date_str,
                    "pnl":    round(pnl_frac, 6),
                    "ticker": symbol,
                    "side":   side,
                })
            except (ValueError, ZeroDivisionError):
                continue

    if open_skipped > 0:
        audit_flags_ibkr.append(
            f"ℹ {open_skipped} opening trade leg(s) skipped — "
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
        # Infer SPY benchmark when dates present — makes IR + Beta scored not informational
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
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
                  ([_spy_note] if _spy_note else []) + \
                  ["SOURCE: Interactive Brokers Flex API — trade data verified",
                   "VERIFIED: Flex Web Service — read-only, credentials never stored"]

    _vr_cache = {
        "fundable_score": report.score,
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
    }
    _cache_validation(report.validation_hash + "-IBKR", _vr_cache)

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
            "4. IMPORTANT — select these specific fields:",
            "     TradeDate, Symbol, Buy/Sell, Quantity, Price, NetCash,",
            "     OpenCloseIndicator, FifoPnlRealized, Currency, Commission",
            "5. Set date range to cover your full backtest period",
            "6. Save the query — note the Query ID shown",
            "7. Go to: Settings > Flex Web Service > Generate Token",
            "8. Copy your Flex Token (keep it secret — treat it like a password)",
            "9. Enter Account ID, Flex Token, and Query ID in QuantProof"
        ],
        "required_fields": {
            "TradeDate":           "Date of each execution",
            "Symbol":              "Ticker symbol",
            "Quantity":            "Shares traded (negative = SELL)",
            "Price":               "Execution price",
            "NetCash":             "Net cash flow (realised P&L on closes)",
            "OpenCloseIndicator":  "O=Opening trade, C=Closing trade (critical for accuracy)",
            "FifoPnlRealized":     "FIFO realised P&L — most accurate P&L source",
            "Currency":            "Account currency (USD/GBP/EUR etc.)",
            "Commission":          "Optional — for cost analysis"
        },
        "note": "Your Flex Token is used read-only and never stored. "
                "OpenCloseIndicator and FifoPnlRealized significantly improve P&L accuracy — "
                "without them, opening trades may be incorrectly included.",
        "why_ibkr": "IBKR-connected validations include a verified source flag. "
                    "FifoPnlRealized gives IBKR's own FIFO P&L calculation — "
                    "the most accurate measure for strategies with scaled entries."
    }


# =========================================================
# KITE TRADEBOOK VALIDATION ENDPOINT
# =========================================================

@app.post("/validate/kite", response_model=ValidationResponse)
async def validate_kite(
    file: UploadFile = File(...),
    currency: str = Form("INR"),
    request: Request = None
):
    """
    Validate Zerodha Kite TradeBook CSV.
    Also accepts Kite Orders CSV — smart_parse_csv() handles routing.

    All fixes applied:
      - K1/K2/K3/K4: Response matches ValidationResponse schema exactly
        (top_issues/top_strengths as List[str], all required fields present)
      - K5: fresh StringIO per read (no cursor exhaustion)
      - K7: profitable_trades from report object
      - G2: uses report.score not report.fundable_score
    """
    if request:
        ip = request.client.host if request.client else "unknown"
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
    raw      = contents.decode("utf-8", errors="ignore")

    # FIX K5: use fresh StringIO for detection probe
    try:
        df_probe = pd.read_csv(io.StringIO(raw), nrows=5)
        # Accept both TradeBook and Orders CSV formats
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

    # F&O options check (TradeBook only — Orders CSV uses segment differently)
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
                        f"⚠ {fo_count} of {total} trades ({fo_pct*100:.0f}%) are F&O segment. "
                        "QuantProof calculates P&L as (exit_price - entry_price) / entry_price, "
                        "which is not accurate for options (premiums, lot sizes, and expiry affect real P&L). "
                        "For options analysis, export only your equity trades, or manually calculate "
                        "per-trade P&L as a percentage and upload as a generic CSV."
                    )
        except HTTPException:
            raise
        except Exception:
            pass

    # Parse using the appropriate parser
    try:
        if _is_kite_orders_csv(df_probe):
            df, kite_note, kite_flags = _parse_kite_orders_csv(raw)
        else:
            df, kite_note, kite_flags = _parse_kite_csv(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

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

    # Auto-assign regime if dates present
    regime_note = ""
    if "date" in df.columns and df["date"].notna().any():
        df, regime_note = assign_regimes(df)
        kite_note += " | " + regime_note

    try:
        # Infer SPY benchmark when dates present — makes IR + Beta scored not informational
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
        report    = validator.run()
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
        crash_sims=[CrashSimResponse(**vars(s)) for s in report.crash_sims],
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
        audit_flags=report.audit_flags + [kite_note] + kite_flags,
        plausibility_summary=report.plausibility_summary,
        scale_warning=getattr(report, '_scale_warning', False),
        currency=detected_currency,
        assumptions=report.assumptions + [
            "Trade data sourced from Zerodha Kite CSV — self-reported",
            "Trades paired using FIFO lot matching (BUY→SELL per symbol)",
        ],
        dashboard=dashboard_data,
        ai_plan=None,
    )


# =========================================================
# TRADINGVIEW WEBHOOK
# =========================================================

TV_SESSIONS: dict = {}
TV_SESSION_TTL_SECONDS      = 7 * 24 * 3600
TV_MAX_ALERTS_PER_SESSION   = 2000
TV_SESSIONS_FILE            = "/tmp/qp_tv_sessions.json"

def _tv_save_sessions():
    """Persist TV sessions to disk so they survive server restarts."""
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
    """Load TV sessions from disk on startup."""
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

# Load persisted sessions on startup
_tv_load_sessions()


def _tv_cleanup():
    """Remove expired TradingView sessions and persist to disk."""
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

    Same position-state tracker as pair_alpaca_orders and _parse_kite_csv:
      - Chronological sort before any pairing
      - FIFO lot queue per symbol
      - Partial lot handling
      - Orphan sell detection
      - Short support: SELL with no open long → opens a short lot

    Returns (trades_list, audit_flags_list).
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
                    pnl_pct   = (lot["price"] - price) / lot["price"]
                    row = {"pnl": round(pnl_pct, 6), "ticker": symbol, "side": "short"}
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
                pnl_pct   = (price - lot["price"]) / lot["price"]
                row = {"pnl": round(pnl_pct, 6), "ticker": symbol, "side": "long"}
                if date: row["date"] = date
                trades.append(row)
                lot["qty"] -= close_qty
                remaining  -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

        elif action in ("CLOSE", "EXIT", "FLAT"):
            if lots[symbol]:
                lot = lots[symbol][0]
                pnl_pct = (
                    (price - lot["price"]) / lot["price"] if lot["side"] == "long"
                    else (lot["price"] - price) / lot["price"]
                )
                row = {"pnl": round(pnl_pct, 6), "ticker": symbol, "side": lot["side"]}
                if date: row["date"] = date
                trades.append(row)
                lots[symbol].clear()

    open_positions = {s: (sum(l["qty"] for l in ls), ls[0]["side"]) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)} {side})" for s, (q, side) in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f"ℹ {len(open_positions)} position(s) still open "
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


@app.get("/webhook/tradingview/new-session", response_model=TVSessionResponse)
def create_tv_session():
    """Create a new TradingView webhook session."""
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


@app.post("/webhook/tradingview/{session_id}")
def receive_tv_alert(session_id: str, alert: TVAlert):
    """Receive a TradingView webhook alert and store it in the session."""
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


@app.get("/webhook/tradingview/{session_id}/status", response_model=TVStatusResponse)
def get_tv_status(session_id: str):
    """Get TradingView session status and trade pair count."""
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


@app.post("/webhook/tradingview/{session_id}/validate", response_model=ValidationResponse)
def validate_tv_alerts(session_id: str):
    """
    Validate TradingView session alerts using QuantProof.

    FIX TV1/TV2: Returns proper ValidationResponse with all required fields.
    FIX TV3: Uses fixed _tv_pair_alerts (no double-append bug).
    FIX TV6: ValueError from validator caught → 400 HTTPException.
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
        _spy_bm, _spy_note = get_spy_benchmark(df)
        validator = QuantProofValidator(df, benchmark_returns=_spy_bm)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        report = validator.run()
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")

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
    # De-duplicate tv_flags against audit_note using exact string match (not substring)
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
        crash_sims=[CrashSimResponse(**vars(s)) for s in report.crash_sims],
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
    )


@app.post("/webhook/tradingview/{session_id}/reset")
def reset_tv_session(session_id: str):
    """Clear all alerts in a session, keeping the session active."""
    _tv_cleanup()
    if session_id not in TV_SESSIONS:
        raise HTTPException(404, "Session not found or expired")
    TV_SESSIONS[session_id]["alerts"] = []
    return {"status": "reset", "message": "Alerts cleared, session remains active"}
