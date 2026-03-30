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

def extract_n_trials(request):
    """Extract n_trials from request (JSON or form-data) with safe validation."""
    try:
        if hasattr(request, 'is_json') and request.is_json:
            val = request.json.get("n_trials")
        else:
            val = request.form.get("n_trials") if hasattr(request, 'form') else None
        
        return max(1, min(50, int(val))) if val is not None else 1
    except (ValueError, TypeError, AttributeError):
        return 1

# ================= RESPONSE MODELS =================

class CheckResultResponse(BaseModel):
    name: str
    passed: bool
    score: float
    value: Any
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

    checks: List[CheckResultResponse]
    crash_sims: List[CrashSimResponse]

    total_trades: int
    profitable_trades: int
    date_range: str

    sharpe: float
    max_drawdown: float
    win_rate: float

    top_issues: List[str]
    top_strengths: List[str]

    validation_hash: str
    validation_date: Any
    engine_version: str

    audit_flags: List[str]
    plausibility_summary: Optional[str]

    scale_warning: bool
    currency: str

    assumptions: List[str]
    dashboard: Optional[Dict[str, Any]]
    ai_plan: Optional[Dict[str, Any]]
    equity_curve: Optional[List[float]] = None
    sebi_checks: Optional[Dict[str, Any]] = None
    sebi_declaration: Optional[Dict[str, Any]] = None
    prop_firm_compliance: Optional[Dict[str, Any]] = None


# ── Rate limiting & token tiers ─────────────────────────────────────────────
_validation_counts: dict = defaultdict(lambda: {"count": 0, "reset": 0})
FREE_DAILY_LIMIT = 3

# Token prefixes
TOKEN_LITE      = "LITE-"   # ₹99  — fixes + AI top-3 in browser
TOKEN_REPORT    = "RPT-"    # ₹299 — PDF + forced n=20 + full AI
TOKEN_PORTFOLIO = "PORT-"   # ₹199 — portfolio correlation endpoint
TOKEN_PRO       = "PRO-"    # ₹999 — 30-day: unlimited + n=50 + all

# Pro token store: token -> expiry_timestamp
_PRO_TOKENS: dict = {}
_PRO_TOKENS_FILE = "/tmp/qp_pro_tokens.json"

def _load_pro_tokens():
    try:
        import json
        if os.path.exists(_PRO_TOKENS_FILE):
            with open(_PRO_TOKENS_FILE) as f:
                _PRO_TOKENS.update(json.load(f))
    except Exception:
        pass

def _save_pro_tokens():
    try:
        import json
        with open(_PRO_TOKENS_FILE, "w") as f:
            json.dump(_PRO_TOKENS, f)
    except Exception:
        pass

_load_pro_tokens()

def _is_pro_token(token: str) -> bool:
    """
    Returns True if token grants pro-level access.
    Dev override: set TEST_REPORT_TOKEN env var on Render to any value,
    then paste that exact value into the dev panel on ?dev=1.
    """
    if not token:
        return False
    # Check all dev override env vars
    for _env_key in ("TEST_REPORT_TOKEN", "DEV_TOKEN", "ADMIN_TOKEN"):
        _test = os.getenv(_env_key, "").strip()
        if _test and token.strip() == _test:
            return True
    # Real PRO token check
    if not token.startswith(TOKEN_PRO):
        return False
    expiry = _PRO_TOKENS.get(token, 0)
    return time.time() < expiry

def _token_tier(token: str) -> str:
    """Return the tier of a token: 'pro', 'report', 'portfolio', 'lite', or 'none'."""
    # Check all dev override env vars (same as _is_pro_token)
    for _env_key in ("TEST_REPORT_TOKEN", "DEV_TOKEN", "ADMIN_TOKEN"):
        _test = os.getenv(_env_key, "").strip()
        if _test and token.strip() == _test:
            return "pro"
    if _is_pro_token(token):     return "pro"
    if token in REPORT_ACCESS_TOKENS:
        if token.startswith(TOKEN_PRO):      return "pro"
        if token.startswith(TOKEN_REPORT):   return "report"
        if token.startswith(TOKEN_PORTFOLIO):return "portfolio"
        if token.startswith(TOKEN_LITE):     return "lite"
        return "report"  # legacy tokens without prefix = RPT
    return "none"

def check_rate_limit(ip: str, token: str = "") -> tuple[bool, int]:
    # Pro tokens bypass rate limit entirely AND clear any existing counter for this IP
    if _is_pro_token(token) or token.startswith(TOKEN_PRO):
        # Reset counter so they don't get blocked even if they hit limit before setting token
        if ip in _validation_counts:
            _validation_counts[ip]["count"] = 0
        return True, 999
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

API_VERSION = "2.7.0"

app = FastAPI(title="QuantProof API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# —? AI Fix Plan ——————————————————————————————————————————————————————————————?
GROQ_API_KEY      = os.getenv("GROQ_API_KEY",      "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")

async def generate_ai_plan(report, date_range: str) -> dict:
    if not any([GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY]):
        return None

    scored = [c for c in report.checks if c.category != "Plausibility"]
    failed = [c for c in scored if not c.passed]
    passed = [c for c in scored if c.passed]

    # Get prop firm compliance data if available
    prop_firm_data = getattr(report, 'prop_firm_compliance', None)
    prop_firm_summary = prop_firm_data.get('summary', {}) if prop_firm_data else {}
    prop_firm_firms = prop_firm_data.get('firms', {}) if prop_firm_data else {}

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
    
    # Add prop firm compliance failures to failed_text if engine is enabled
    if prop_firm_data:
        ineligible = [f for f, r in prop_firm_data.get('firms', {}).items() if not r.get('eligible')]
        if ineligible:
            failed_text += f"\n- Prop Firm Compliance: ineligible for {', '.join(ineligible)}"
    
    passed_text = ", ".join([c.name for c in passed])

    # Add prop firm compliance info to the prompt if available
    prop_firm_section = ""
    if prop_firm_data:
        prop_firm_section = f"""
PROP FIRM COMPLIANCE:
- Total Firms Analyzed: {prop_firm_summary.get('total_firms', 0)}
- Eligible Firms: {prop_firm_summary.get('eligible_firms', 0)}
- Average Score: {prop_firm_summary.get('average_score', 0):.1f}/100
- Best Firm: {prop_firm_summary.get('best_firm', 'None')}
- Firm Breakdown: {', '.join([f"{name} ({'ELIGIBLE' if data.get('eligible') else 'NOT ELIGIBLE'}, {data.get('score', 0)}/100)" for name, data in prop_firm_firms.items()])}
"""

    prompt = f"""You are a quantitative trading risk analyst reviewing a backtest validation report.

STRATEGY STATS:
- Fundable Score: {report.score}/100 ({report.grade})
- Sharpe: {round(report.sharpe, 2)} | Max Drawdown: {round(report.max_drawdown * 100, 1)}% | Win Rate: {round(report.win_rate, 1)}%
- Total Trades: {report.total_trades} | Date Range: {date_range}

{prop_firm_section}

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


async def generate_portfolio_fix_plan(
    strategies: list,
    corr_matrix: list,
    rp_weights: list,
    div_score: float,
    dd_coincidence: float,
    combined_sharpe: float,
    combined_dd: float,
    avg_corr: float,
    portfolio_dsr,
    strategy_names: list,
) -> dict:
    """
    Portfolio-level AI fix plan. Thinks as a quant portfolio manager:
    - Identifies the weakest strategy dragging the portfolio
    - Flags high correlation pairs and what to do about it
    - Gives specific allocation advice based on actual numbers
    - References the exact metrics (corr values, weights, coincidence %)
    """
    if not any([GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY]):
        return _portfolio_rule_based_fixes(
            strategies, corr_matrix, rp_weights, div_score,
            dd_coincidence, combined_sharpe, combined_dd, avg_corr, strategy_names
        )

    # Build per-strategy summary for prompt
    strat_lines = []
    for i, s in enumerate(strategies):
        w = rp_weights[i] if i < len(rp_weights) else 1/len(strategies)
        strat_lines.append(
            f"Strategy {i+1} '{s['name']}': "
            f"Score={s['fundable_score']}/100 grade={s['grade']} "
            f"Sharpe={s['sharpe']} MaxDD={round(abs(s['max_drawdown'])*100,1)}% "
            f"WinRate={s['win_rate']}% Trades={s['total_trades']} "
            f"RiskParityWeight={round(w*100,1)}% "
            f"CrashSurvived={s.get('crash_survived','?')}/{s.get('crash_total','5')} "
            f"ChecksFailed={s.get('checks_failed',0)}/{s.get('checks_total',44)}"
        )

    # Build correlation pairs
    n = len(strategies)
    corr_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            v = corr_matrix[i][j] if i < len(corr_matrix) and j < len(corr_matrix[i]) else 0
            corr_pairs.append(
                f"'{strategy_names[i]}' vs '{strategy_names[j]}': {v:+.3f}"
            )

    # Top individual failures across all strategies
    all_failures = []
    for s in strategies:
        for c in s.get("checks", []):
            if not c.get("passed", True):
                all_failures.append({
                    "strategy": s["name"],
                    "check": c.get("name",""),
                    "score": c.get("score",0),
                    "category": c.get("category",""),
                    "insight": c.get("insight",""),
                    "fix": c.get("fix",""),
                })
    all_failures.sort(key=lambda x: x["score"])
    top_failures = all_failures[:8]
    failure_lines = chr(10).join([
        f"  [{f['strategy']}] {f['check']} ({f['category']}, score {f['score']}/100): {f['insight']}"
        for f in top_failures
    ])

    prompt = f"""You are a quantitative portfolio manager reviewing a multi-strategy portfolio validation.
Think like someone who has run prop desks and fund-of-funds allocations for 15 years.
Be specific to the actual numbers. No generic advice.

PORTFOLIO SUMMARY:
- {n} strategies, Diversification Score: {div_score}/100
- Average pairwise correlation: {avg_corr:+.3f} (signed)
- Risk-parity combined Sharpe: {combined_sharpe}
- Combined max drawdown: {round(abs(combined_dd)*100,1)}%
- Drawdown coincidence: {round(dd_coincidence*100,1)}% (% of drawdown time where 2+ strategies fall together)
- Portfolio DSR: {portfolio_dsr if portfolio_dsr is not None else 'n/a'}/100

INDIVIDUAL STRATEGIES:
{chr(10).join(strat_lines)}

PAIRWISE CORRELATIONS (normalised, scale-independent):
{chr(10).join(corr_pairs)}

TOP INDIVIDUAL CHECK FAILURES:
{failure_lines}

Produce a portfolio-level fix plan as JSON with exactly these keys:
{{
  "summary": "2-3 sentences on the portfolio's core problem. Reference actual numbers. Be specific about which strategy is the weak link and why the portfolio correlation profile is good or bad.",
  "portfolio_fixes": [
    {{
      "rank": 1,
      "issue": "name of the portfolio-level issue",
      "why_it_matters": "real consequence — what happens in live trading or a fund allocation context",
      "specific_fix": "concrete step with the actual numbers. e.g. 'Reduce Strategy X allocation from 42% to 20% — its Sharpe of 0.3 with 35% MaxDD is dragging combined return without sufficient diversification benefit given correlation of +0.54 with Strategy Y'",
      "score_impact": "expected change to diversification score or combined Sharpe if this is fixed",
      "effort": "Low | Medium | High"
    }}
  ],
  "allocation_verdict": "One sentence: should this portfolio be deployed as-is, rebalanced, or rebuilt? Reference the key number that drives this.",
  "strongest_combination": "Which strategy pairing adds the most genuine diversification and why — reference the specific correlation value"
}}

Rules:
- portfolio_fixes should have 3-5 items
- Each fix must reference specific numbers from the data above
- If two strategies are highly correlated (>0.5), flag it explicitly with their names and the exact correlation
- If one strategy has a much lower score than others, flag whether it's dragging the portfolio or contributing genuine diversification
- Drawdown coincidence > 50% is a serious problem — flag it
- Return ONLY valid JSON, no markdown or preamble"""

    try:
        import httpx, json

        if GROQ_API_KEY:
            url     = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body    = {"model": "llama-3.3-70b-versatile",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 2000, "temperature": 0.2}
            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        elif ANTHROPIC_API_KEY:
            url     = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            body    = {"model": "claude-sonnet-4-20250514", "max_tokens": 2000,
                       "messages": [{"role": "user", "content": prompt}]}
            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        elif OPENAI_API_KEY:
            url     = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body    = {"model": "gpt-4o-mini",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 2000, "temperature": 0.2}
            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return None

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    except Exception as e:
        print(f"Portfolio AI plan failed ({type(e).__name__}): {e}")
        # Fall back to rule-based
        return _portfolio_rule_based_fixes(
            strategies, corr_matrix, rp_weights, div_score,
            dd_coincidence, combined_sharpe, combined_dd, avg_corr, strategy_names
        )


def _portfolio_rule_based_fixes(
    strategies, corr_matrix, rp_weights, div_score,
    dd_coincidence, combined_sharpe, combined_dd, avg_corr, strategy_names
) -> dict:
    """
    Deterministic fix plan when no LLM is available.
    Generates quant-grade advice from the actual numbers.
    """
    n = len(strategies)
    fixes = []
    rank  = 1

    # 1. High drawdown coincidence
    if dd_coincidence > 0.50:
        fixes.append({
            "rank": rank,
            "issue": "High Drawdown Coincidence",
            "why_it_matters": (
                f"Your strategies drawdown simultaneously {round(dd_coincidence*100,0):.0f}% of the time. "
                f"Portfolio drawdown protection only works when strategies fall independently. "
                f"At {round(dd_coincidence*100,0):.0f}% coincidence your combined max DD of "
                f"{round(abs(combined_dd)*100,1)}% is close to your worst individual strategy."
            ),
            "specific_fix": (
                f"Reduce allocation to the highest-correlated pair. "
                f"Target coincidence below 30% — add a strategy that trades in a "
                f"different regime (e.g. if current strategies are trend-following, "
                f"add mean reversion or a short-vol strategy)."
            ),
            "score_impact": "+10 to +20 diversification score",
            "effort": "High"
        })
        rank += 1

    # 2. High correlation pairs
    for i in range(n):
        for j in range(i+1, n):
            v = corr_matrix[i][j] if i < len(corr_matrix) else 0
            if abs(v) > 0.50:
                fixes.append({
                    "rank": rank,
                    "issue": f"High Correlation: '{strategy_names[i]}' vs '{strategy_names[j]}'",
                    "why_it_matters": (
                        f"Correlation of {v:+.3f} means these two strategies share "
                        f"{round(v**2*100,0):.0f}% of their variance (R²). "
                        f"Running both gives you redundant risk, not diversification."
                    ),
                    "specific_fix": (
                        f"Either (a) reduce the lower-scoring strategy's weight to under 10%, "
                        f"or (b) identify whether they trade the same underlying signal and merge "
                        f"them into one strategy. Risk-parity already reduces the weaker one, "
                        f"but correlation this high means both will drawdown together regardless of weight."
                    ),
                    "score_impact": "+8 to +15 diversification score if pair is separated",
                    "effort": "Medium"
                })
                rank += 1

    # 3. Weak strategy dragging portfolio
    scores = [(s["name"], s["fundable_score"], s["sharpe"], rp_weights[i] if i < len(rp_weights) else 1/n)
              for i, s in enumerate(strategies)]
    weakest = min(scores, key=lambda x: x[1])
    strongest = max(scores, key=lambda x: x[1])
    if weakest[1] < 50 and (strongest[1] - weakest[1]) > 20:
        fixes.append({
            "rank": rank,
            "issue": f"Weak Strategy: '{weakest[0]}' (score {weakest[1]}/100)",
            "why_it_matters": (
                f"'{weakest[0]}' scores {weakest[1]}/100 vs '{strongest[0]}' at {strongest[1]}/100. "
                f"It currently holds {round(weakest[3]*100,1)}% of portfolio weight under risk parity. "
                f"A strategy failing institutional checks introduces correlated tail risk even at low weight."
            ),
            "specific_fix": (
                f"Fix '{weakest[0]}' as a standalone strategy first — run single-strategy "
                f"validation and resolve its failed checks before including it in the portfolio. "
                f"A strategy below 50/100 should not be in a live portfolio regardless of its "
                f"diversification contribution."
            ),
            "score_impact": "+5 to +15 combined Sharpe improvement if replaced with a passing strategy",
            "effort": "High"
        })
        rank += 1

    # 4. Low combined Sharpe vs individuals
    individual_sharpes = [s["sharpe"] for s in strategies]
    avg_sh = sum(individual_sharpes) / len(individual_sharpes) if individual_sharpes else 0
    if avg_sh > 0 and combined_sharpe < avg_sh * 0.8:
        fixes.append({
            "rank": rank,
            "issue": "Combined Sharpe Below Average Individual",
            "why_it_matters": (
                f"Portfolio combined Sharpe ({combined_sharpe:.2f}) is below the average individual "
                f"Sharpe ({avg_sh:.2f}). Risk parity weighting should produce a combined Sharpe "
                f"at or above the average when strategies are sufficiently uncorrelated. "
                f"This indicates negative diversification — strategies are hurt by being combined."
            ),
            "specific_fix": (
                f"Check for regime mismatch: if your strategies all performed well in the "
                f"same bull market period, they will show low correlation in backtest but "
                f"high correlation in live drawdowns. Validate each strategy across different "
                f"market regimes before combining."
            ),
            "score_impact": "Regime-specific — varies by strategy",
            "effort": "High"
        })
        rank += 1

    # 5. Diversification score interpretation
    if div_score >= 75:
        alloc_verdict = (
            f"This portfolio is deployable. Diversification score of {div_score}/100 with "
            f"avg correlation {avg_corr:+.3f} indicates genuine independent alpha across strategies."
        )
        strongest_pair = None
        best_corr = 1.0
        for i in range(n):
            for j in range(i+1, n):
                v = abs(corr_matrix[i][j]) if i < len(corr_matrix) else 1
                if v < best_corr:
                    best_corr = v
                    strongest_pair = f"'{strategy_names[i]}' and '{strategy_names[j]}' (correlation {corr_matrix[i][j]:+.3f})"
    else:
        alloc_verdict = (
            f"Rebalance before deploying. Diversification score {div_score}/100 is insufficient "
            f"— the strategies share too much variance to justify running all simultaneously."
        )
        strongest_pair = "No pair qualifies — all correlations are too high for genuine diversification."

    summary_parts = []
    if div_score >= 75:
        summary_parts.append(
            f"Portfolio shows genuine diversification ({div_score}/100) with avg correlation {avg_corr:+.3f}. "
            f"Risk-parity combined Sharpe of {combined_sharpe:.2f} with {round(abs(combined_dd)*100,1)}% max DD is deployable."
        )
    else:
        summary_parts.append(
            f"Portfolio diversification score of {div_score}/100 is below institutional threshold (75+). "
            f"Average correlation of {avg_corr:+.3f} reduces the expected diversification benefit."
        )

    if dd_coincidence > 0.50:
        summary_parts.append(
            f"Critical issue: {round(dd_coincidence*100,0):.0f}% drawdown coincidence — strategies fall together "
            f"more than half the time, making portfolio drawdown protection largely illusory."
        )

    return {
        "summary": " ".join(summary_parts),
        "portfolio_fixes": fixes if fixes else [{
            "rank": 1,
            "issue": "Portfolio is well-constructed",
            "why_it_matters": "All key metrics pass institutional thresholds.",
            "specific_fix": "Focus on individual strategy improvements and live execution.",
            "score_impact": "Maintain current diversification",
            "effort": "Low"
        }],
        "allocation_verdict": alloc_verdict,
        "strongest_combination": strongest_pair or "Insufficient data",
    }


# =========================================================
# SPY REGIME HELPERS
# =========================================================

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
    "2026-04": "CONSOLIDATION", "2026-05": "CONSOLIDATION", "2026-06": "CONSOLIDATION",
    "2026-07": "CONSOLIDATION", "2026-08": "CONSOLIDATION", "2026-09": "CONSOLIDATION",
    "2026-10": "CONSOLIDATION", "2026-11": "CONSOLIDATION", "2026-12": "CONSOLIDATION",
    "2027-01": "CONSOLIDATION", "2027-02": "CONSOLIDATION", "2027-03": "CONSOLIDATION",
    "2027-04": "CONSOLIDATION", "2027-05": "CONSOLIDATION", "2027-06": "CONSOLIDATION",
    "2027-07": "CONSOLIDATION", "2027-08": "CONSOLIDATION", "2027-09": "CONSOLIDATION",
    "2027-10": "CONSOLIDATION", "2027-11": "CONSOLIDATION", "2027-12": "CONSOLIDATION",
}

# =========================================================
# NIFTY REGIME LOOKUP
# =========================================================
NIFTY_REGIME_LOOKUP: Dict[str, str] = {
    # 2020 — COVID crash Mar, recovery Apr onwards
    "2020-01": "BULL",  "2020-02": "BEAR",  "2020-03": "BEAR",
    "2020-04": "BULL",  "2020-05": "BULL",  "2020-06": "BULL",
    "2020-07": "BULL",  "2020-08": "BULL",  "2020-09": "CONSOLIDATION",
    "2020-10": "BULL",  "2020-11": "BULL",  "2020-12": "BULL",

    # 2021 — strong bull run all year
    "2021-01": "BULL",  "2021-02": "BULL",  "2021-03": "BULL",
    "2021-04": "BEAR",  "2021-05": "BULL",  "2021-06": "BULL",
    "2021-07": "BULL",  "2021-08": "BULL",  "2021-09": "BULL",
    "2021-10": "BULL",  "2021-11": "CONSOLIDATION",  "2021-12": "BULL",

    # 2022 — bear year, rate hikes, FII selloff
    "2022-01": "BEAR",  "2022-02": "BEAR",  "2022-03": "CONSOLIDATION",
    "2022-04": "BEAR",  "2022-05": "BEAR",  "2022-06": "BEAR",
    "2022-07": "BULL",  "2022-08": "BULL",  "2022-09": "CONSOLIDATION",
    "2022-10": "BULL",  "2022-11": "BULL",  "2022-12": "CONSOLIDATION",

    # 2023 — recovery and bull run
    "2023-01": "BULL",  "2023-02": "BEAR",  "2023-03": "CONSOLIDATION",
    "2023-04": "BULL",  "2023-05": "BULL",  "2023-06": "BULL",
    "2023-07": "BULL",  "2023-08": "CONSOLIDATION",  "2023-09": "CONSOLIDATION",
    "2023-10": "BULL",  "2023-11": "BULL",  "2023-12": "BULL",

    # 2024 — bull run to Sep, correction Oct onwards
    "2024-01": "BULL",  "2024-02": "BULL",  "2024-03": "BULL",
    "2024-04": "CONSOLIDATION",  "2024-05": "BULL",  "2024-06": "BULL",
    "2024-07": "BULL",  "2024-08": "BULL",  "2024-09": "BULL",
    "2024-10": "BEAR",  "2024-11": "BEAR",  "2024-12": "BEAR",

    # 2025 — correction continues, slow recovery mid-year
    "2025-01": "BEAR",  "2025-02": "BEAR",  "2025-03": "BEAR",
    "2025-04": "CONSOLIDATION",  "2025-05": "BULL",  "2025-06": "BULL",
    "2025-07": "BULL",  "2025-08": "BULL",  "2025-09": "BULL",
    "2025-10": "BULL",  "2025-11": "CONSOLIDATION",  "2025-12": "BULL",

    # 2026 onwards — unknown future, use CONSOLIDATION
    "2026-01": "CONSOLIDATION", "2026-02": "CONSOLIDATION",
    "2026-03": "CONSOLIDATION", "2026-04": "CONSOLIDATION",
    "2026-05": "CONSOLIDATION", "2026-06": "CONSOLIDATION",
    "2026-07": "CONSOLIDATION", "2026-08": "CONSOLIDATION",
    "2026-09": "CONSOLIDATION", "2026-10": "CONSOLIDATION",
    "2026-11": "CONSOLIDATION", "2026-12": "CONSOLIDATION",
    "2027-01": "CONSOLIDATION", "2027-02": "CONSOLIDATION",
    "2027-03": "CONSOLIDATION", "2027-04": "CONSOLIDATION",
    "2027-05": "CONSOLIDATION", "2027-06": "CONSOLIDATION",
    "2027-07": "CONSOLIDATION", "2027-08": "CONSOLIDATION",
    "2027-09": "CONSOLIDATION", "2027-10": "CONSOLIDATION",
    "2027-11": "CONSOLIDATION", "2027-12": "CONSOLIDATION",
}

_SPY_CACHE: dict = {}
_SPY_CACHE_MAX = 10

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


# =========================================================
# NIFTY BENCHMARK HELPER
# =========================================================
_NIFTY_CACHE: dict = {}
_NIFTY_CACHE_MAX = 10

def _get_nifty_cached(start, end) -> pd.Series | None:
    """Fetch Nifty 50 Close with caching. Returns None on any failure."""
    key = (str(start.date()), str(end.date()))
    if key in _NIFTY_CACHE:
        return _NIFTY_CACHE[key]
    try:
        import yfinance as yf, threading
        result_holder = [None]
        def _fetch():
            try:
                raw = yf.download("^NSEI", start=start, end=end,
                                  progress=False, timeout=6)["Close"]
                result_holder[0] = raw.squeeze()
            except Exception:
                pass
        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=8)
        nifty = result_holder[0]
        if nifty is not None and len(nifty) >= 5:
            if len(_NIFTY_CACHE) >= _NIFTY_CACHE_MAX:
                _NIFTY_CACHE.pop(next(iter(_NIFTY_CACHE)))
            _NIFTY_CACHE[key] = nifty
            return nifty
    except Exception:
        pass
    return None


def _is_non_us_strategy(df: pd.DataFrame) -> bool:
    """
    Detect if the strategy trades non-US assets (forex, metals, crypto, Indian equities).
    Returns True when SPY is not a meaningful regime proxy.
    """
    if "ticker" not in df.columns:
        return False
    tickers = df["ticker"].dropna().astype(str).str.upper().unique()
    # Forex pairs: 6-char alphabetic (EURUSD, GBPJPY, USDINR etc)
    # Metals/commodities: XAUUSD, XAGUSD, XTIUSD
    # Indian equities: typically have suffix -EQ or are NSE listed (no $ sign)
    # Crypto: BTC, ETH, BTCUSD, ETHUSD
    non_us_patterns = [
        lambda t: len(t) == 6 and t.isalpha() and not t.endswith(("SPY","QQQ","IWM","DIA")),
        lambda t: t.startswith(("XAU","XAG","XTI","XBR","BTC","ETH","LTC","XRP")),
        lambda t: t.endswith(("-EQ", ".NS", ".BO", "USDT")),
        lambda t: "USD" in t and len(t) >= 6 and t[:3] not in ("SPY","QQQ","IWM"),
    ]
    non_us_count = sum(
        1 for t in tickers
        if any(p(t) for p in non_us_patterns)
    )
    return non_us_count > len(tickers) * 0.5  # majority non-US


# =========================================================
# INDIAN STRATEGY DETECTION
# =========================================================
def _is_indian_strategy(df: pd.DataFrame, currency: str = None) -> bool:
    """
    Detect if the strategy trades Indian equities.
    Returns True when Nifty 50 is the appropriate benchmark.
    """
    # Check explicit currency parameter
    if currency and currency.upper() == "INR":
        # Check if tickers are forex pairs (exclude forex even with INR currency)
        if "ticker" in df.columns:
            tickers = df["ticker"].dropna().astype(str)
            forex_pairs = [t for t in tickers if len(t) == 6 and t[:3].isalpha() and t[3:].isalpha()]
            if len(forex_pairs) > len(tickers) * 0.5:  # majority forex pairs
                return False
        return True
    
    # Check ticker suffixes for Indian exchanges
    if "ticker" in df.columns:
        tickers = df["ticker"].dropna().astype(str)
        nse_suffixes = [t for t in tickers if t.endswith((".NS", ".BO"))]
        eq_suffixes = [t for t in tickers if t.endswith("-EQ")]
        
        # Exclude forex pairs (3-char currency pairs like USDINR, EURINR)
        forex_pairs = [t for t in tickers if len(t) == 6 and t[:3].isalpha() and t[3:].isalpha()]
        
        indian_indicators = nse_suffixes + eq_suffixes
        if len(indian_indicators) > 0 and len(forex_pairs) <= len(tickers) * 0.5:
            return True
    
    return False


# =========================================================
# BENCHMARK ROUTER
# =========================================================
def get_benchmark_type(df: pd.DataFrame, currency: str = None) -> str:
    """
    Route to appropriate benchmark based on strategy characteristics.
    Returns "nifty" for Indian equity strategies, "spy" otherwise.
    """
    if _is_indian_strategy(df, currency):
        return "nifty"
    return "spy"


def assign_regimes(df: pd.DataFrame, benchmark_type: str = "spy") -> tuple[pd.DataFrame, str]:
    """
    Assigns a 'regime' column to df using SPY 20-day SMA via yfinance.
    Falls back to SPY monthly lookup table if yfinance is unavailable.
    df must have a 'date' column. Returns (df_with_regime, audit_note).
    Only called when df has dates but no regime column.

    IMPROVEMENTS:
    - Dates normalised to date-only before asof() → prevents intraday UTC offset
      from mapping to wrong calendar date for TV/Alpaca UTC timestamps
    - 3-consecutive-day confirmation before regime changes → prevents SMA20 noise
      from causing a regime flip every 3-4 days in sideways markets
    - Non-US strategy warning when SPY is not a meaningful regime proxy
    """
    # Normalise all trade dates to date-only (midnight) before lookup
    # This prevents intraday UTC timestamps crossing calendar boundaries
    def _normalise_date(d):
        try:
            ts = pd.Timestamp(d)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return pd.Timestamp(ts.date())  # strip time component
        except Exception:
            return pd.NaT

    try:
        start = df["date"].min() - pd.Timedelta(days=35)  # 35 days: 30 warmup + 5 buffer
        end   = df["date"].max() + pd.Timedelta(days=1)
        if benchmark_type == "nifty":
            spy   = _get_nifty_cached(start, end)
            if spy is None:
                raise ValueError("Nifty data unavailable")
        else:
            spy   = _get_spy_cached(start, end)
            if spy is None:
                raise ValueError("SPY data unavailable")
        sma20 = spy.rolling(20, min_periods=20).mean()

        # Build a raw regime series for the full SPY date range
        # Then apply 3-day confirmation to prevent noise-driven flips
        spy_idx = spy.index
        raw_regimes = {}
        for dt in spy_idx:
            try:
                dt_naive = pd.Timestamp(dt).tz_localize(None) if getattr(dt, 'tzinfo', None) is None else pd.Timestamp(dt).tz_convert(None)
                price = float(spy.asof(dt_naive))
                sma   = float(sma20.asof(dt_naive))
                if np.isnan(price) or np.isnan(sma) or sma == 0:
                    raw_regimes[dt_naive.date()] = "CONSOLIDATION"
                    continue
                ratio = price / sma
                if ratio > 1.01:   raw_regimes[dt_naive.date()] = "BULL"
                elif ratio < 0.99: raw_regimes[dt_naive.date()] = "BEAR"
                else:              raw_regimes[dt_naive.date()] = "CONSOLIDATION"
            except Exception:
                pass

        # Apply 3-consecutive-day confirmation
        # Regime only changes if the new label persists for 3+ trading days
        confirmed_regimes = {}
        sorted_dates = sorted(raw_regimes.keys())
        current_regime = "CONSOLIDATION"
        pending_regime = None
        pending_count  = 0
        for dt in sorted_dates:
            raw = raw_regimes[dt]
            if raw == current_regime:
                pending_regime = None
                pending_count  = 0
            elif raw == pending_regime:
                pending_count += 1
                if pending_count >= 3:
                    current_regime = pending_regime
                    pending_regime = None
                    pending_count  = 0
            else:
                pending_regime = raw
                pending_count  = 1
            confirmed_regimes[dt] = current_regime

        def _regime(d):
            try:
                dt_norm = _normalise_date(d)
                if pd.isna(dt_norm):
                    return "CONSOLIDATION"
                dt_date = dt_norm.date()
                if dt_date in confirmed_regimes:
                    return confirmed_regimes[dt_date]
                # asof fallback: find closest prior date
                prior = [k for k in sorted_dates if k <= dt_date]
                return confirmed_regimes[prior[-1]] if prior else "CONSOLIDATION"
            except Exception:
                return "CONSOLIDATION"

        df["regime"] = df["date"].apply(_regime)
        dist = df["regime"].value_counts().to_dict()

        # Warn if SPY is not a meaningful proxy for this strategy
        spy_warning = ""
        if benchmark_type != "nifty" and _is_non_us_strategy(df):
            spy_warning = (
                "SPY regime may not be meaningful for non-US/forex/crypto strategies. "
                "Regime labels reflect S&P 500 conditions, not the traded asset class."
            )

        benchmark_label = "Nifty 50" if benchmark_type == "nifty" else "SPY"
        return df, f"✓ Regime auto-detected from {benchmark_label} 20-day SMA (3-day confirmation) — {dist}{spy_warning}"

    except Exception as e:
        def _regime_fallback(d):
            try:
                dt_norm = _normalise_date(d)
                if pd.isna(dt_norm):
                    return "CONSOLIDATION"
                key = dt_norm.strftime("%Y-%m")
                # Extend beyond lookup table: use CONSOLIDATION with explicit note
                lookup = NIFTY_REGIME_LOOKUP if benchmark_type == "nifty" else SPY_REGIME_LOOKUP
                return lookup.get(key, "CONSOLIDATION")
            except Exception:
                return "CONSOLIDATION"

        df["regime"] = df["date"].apply(_regime_fallback)
        dist = df["regime"].value_counts().to_dict()
        # Check for dates beyond the lookup table
        max_date = df["date"].max()
        lookup_end = pd.Timestamp("2027-12-31")
        beyond_note = ""
        if pd.notna(max_date) and pd.Timestamp(max_date) > lookup_end:
            beyond_note = "Some dates beyond lookup table marked CONSOLIDATION."
        fallback_label = "Nifty 50" if benchmark_type == "nifty" else "SPY"
        return df, f"Regime from monthly {fallback_label} approximation (live data unavailable) {dist}{beyond_note}"

# =========================================================
# SPY BENCHMARK HELPER
# =========================================================

def get_spy_benchmark(df: pd.DataFrame, spy_series: pd.Series = None) -> tuple:
    """
    Align SPY daily returns to strategy trade dates for IR + Beta scoring.
    Returns (benchmark_array, audit_note) or (None, audit_note) on any failure.

    IMPROVEMENTS:
    - Date normalisation: timestamps stripped to date-only before lookup,
      preventing intraday UTC offsets (TV, Alpaca) from mapping to wrong calendar day
    - match_pct now counts only TRUE trading day matches (dates in spy_ret.index),
      not asof() forward-fills — fixes the broken quality gate
    - HFT/single-day detection: if all trades share 1 unique date, benchmark
      degenerates to constant array — returns None with clear audit note
    - Non-US strategy detection: forex/crypto/Indian equities get None + warning
      rather than misleading SPY-based IR scores
    - Fetch failure now produces a visible audit note instead of empty string
    """
    if "date" not in df.columns or df["date"].isna().all():
        return None, ""

    # —? Non-US strategy guard ————————————————————————————————————————————————?
    if _is_non_us_strategy(df):
        return None, (
            "SPY benchmark skipped - strategy appears to trade non-US assets "
            "(forex/crypto/Indian equities). IR + Beta vs SPY would be misleading. "
            "IR and Beta checks remain informational."
        )

    try:
        # —? Fetch SPY if not already provided ———————————————————————————————?
        if spy_series is None:
            try:
                start = pd.Timestamp(pd.Timestamp(df["date"].min()).date()) - pd.Timedelta(days=10)
                end   = pd.Timestamp(pd.Timestamp(df["date"].max()).date()) + pd.Timedelta(days=2)
            except Exception:
                start = df["date"].min() - pd.Timedelta(days=10)
                end   = df["date"].max() + pd.Timedelta(days=2)
            spy_raw = _get_spy_cached(start, end)
            if spy_raw is None:
                return None, "⚠️ SPY benchmark unavailable (network/yfinance). IR + Beta checks are informational."
        else:
            spy_raw = spy_series

        if spy_raw is None or len(spy_raw) < 5:
            return None, "⚠️ SPY benchmark unavailable (insufficient data). IR + Beta checks are informational."

        # —? Compute daily returns ————————————————————————————————————————————?
        spy_ret = spy_raw.pct_change().dropna()
        # Normalise index: strip tz info and time component — only date matters
        if hasattr(spy_ret.index, 'tz') and spy_ret.index.tz is not None:
            spy_ret.index = spy_ret.index.tz_convert("UTC").tz_localize(None)
        spy_ret.index = pd.DatetimeIndex(
            [pd.Timestamp(d.date()) for d in pd.to_datetime(spy_ret.index)]
        )

        # Build set of valid trading dates for TRUE match_pct
        trading_dates = set(spy_ret.index.date)

        # —? HFT / single-day degeneracy check ————————————————————————————————?
        unique_trade_dates = set()
        for d in df["date"]:
            try:
                unique_trade_dates.add(pd.Timestamp(d).date())
            except Exception:
                pass
        if len(unique_trade_dates) <= 1:
            return None, (
                "SPY benchmark skipped - all trades occur on 1 unique date. "
                "Benchmark array would be constant (identical SPY return for every trade). "
                "IR + Beta checks remain informational."
            )

        # —? Align to strategy trade dates ————————————————————————————————————?
        aligned = []
        true_matches = 0
        for d in df["date"]:
            try:
                # Normalise to date-only midnight — strips intraday UTC offsets
                d_date = pd.Timestamp(d).date()
                d_ts   = pd.Timestamp(d_date)
                val    = float(spy_ret.asof(d_ts))
                if np.isnan(val) or abs(val) > 0.20:
                    aligned.append(0.0)
                else:
                    aligned.append(val)
                    if d_date in trading_dates:
                        true_matches += 1  # only count actual trading day hits
            except Exception:
                aligned.append(0.0)

        benchmark = np.array(aligned, dtype=float)

        # —? Quality gates —————————————————————————————————————————————————————?
        # Use TRUE match_pct — asof() forward-fill is NOT a real match
        true_match_pct = true_matches / max(len(benchmark), 1) * 100
        if true_match_pct < 30:
            return None, (
                f"SPY benchmark skipped - only {true_match_pct:.0f}% of trade dates "
                f"fall on US trading days (threshold: 30%). "
                f"Strategy likely trades non-US markets or weekends. "
                f"IR + Beta checks remain informational."
            )

        if np.std(benchmark) < 1e-6:
            return None, (
                "SPY benchmark skipped - benchmark has zero variance "
                "(all trades same day or identical SPY returns). "
                "IR + Beta checks remain informational."
            )

        note = (
            f"SPY benchmark aligned ({true_match_pct:.0f}% of trades on US trading days) "
            f"IR + Beta checks are now scored. Only meaningful for US equity strategies."
        )
        return benchmark, note

    except Exception:
        return None, "⚠️ SPY benchmark unavailable (unexpected error). IR + Beta checks are informational."


# =========================================================
# NIFTY BENCHMARK HELPER
# =========================================================

def get_nifty_benchmark(df: pd.DataFrame, nifty_series: pd.Series = None) -> tuple:
    """
    Align Nifty 50 daily returns to strategy trade dates for IR + Beta scoring.
    Returns (benchmark_array, audit_note) or (None, audit_note) on any failure.

    IMPROVEMENTS:
    - Date normalisation: timestamps stripped to date-only before lookup,
      preventing intraday UTC offsets (TV, Alpaca) from mapping to wrong calendar day
    - match_pct now counts only TRUE trading day matches (dates in nifty_ret.index),
      not any calendar date
    - NSE trading day calendar (actual NSE trading days from yfinance)
    - Minimum 30% NSE trading day coverage required for meaningful IR/Beta
    """
    if 'date' not in df.columns:
        return None, "Nifty benchmark skipped - no date column in trades."

    # Normalise all trade dates to date-only (midnight) before lookup
    def _normalise_date(d):
        try:
            ts = pd.Timestamp(d)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return pd.Timestamp(ts.date())  # strip time component
        except Exception:
            return pd.NaT

    try:
        start = df["date"].min() - pd.Timedelta(days=35)  # 35 days: 30 warmup + 5 buffer
        end   = df["date"].max() + pd.Timedelta(days=1)
        nifty = _get_nifty_cached(start, end) if nifty_series is None else nifty_series
        if nifty is None:
            raise ValueError("Nifty data unavailable")
        nifty_ret = nifty.pct_change().dropna()
        
        # Normalise timezone to date-only for reliable asof() lookup
        if hasattr(nifty_ret.index, 'tz') and nifty_ret.index.tz is not None:
            nifty_ret.index = nifty_ret.index.tz_convert("UTC").tz_localize(None)
        nifty_ret.index = pd.DatetimeIndex(
            [pd.Timestamp(d.date()) for d in pd.to_datetime(nifty_ret.index)]
        )

        # Track true NSE trading day matches
        trading_dates = set(nifty_ret.index.date)
        aligned = []
        true_matches = 0

        for trade_date in df["date"]:
            norm_date = _normalise_date(trade_date)
            if pd.isna(norm_date):
                continue
            
            val = float(nifty_ret.asof(norm_date))
            if np.isnan(val) or abs(val) > 0.20:
                aligned.append(0.0)
            else:
                aligned.append(val)
                if norm_date.date() in trading_dates:
                    true_matches += 1

        benchmark = np.array(aligned, dtype=float)
        true_match_pct = true_matches / max(len(benchmark), 1) * 100
        if true_match_pct < 30:  # Require 30% NSE trading day coverage
            return None, (
                f"⚠️ Nifty 50 benchmark coverage too low ({true_match_pct:.1f}% of NSE trading days). "
                "IR + Beta vs Nifty would be misleading."
            )

        valid_mask = ~np.isnan(benchmark)
        if valid_mask.sum() < 5:
            return None, "Insufficient Nifty 50 overlap for meaningful IR + Beta."

        return benchmark, "Nifty 50 benchmark aligned ({}% of NSE trading days)".format(true_match_pct)

    except Exception as e:
        return None, "Nifty 50 benchmark alignment failed: {}".format(str(e)[:100])
# =========================================================
# KITE TRADEBOOK PARSER
# =========================================================

def _is_kite_csv(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip() for c in df.columns}
    required     = {"trade_date", "trade_type", "quantity", "price"}
    kite_signals = {"trade_time", "settlement_id", "series", "order_id", "trade_id"}
    return required.issubset(cols) and bool(kite_signals & cols)


def _parse_kite_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Kite TradeBook CSV into QuantProof trades.
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
    # NOTE: 'series' is excluded — it's a segment identifier (EQ/FO/CDS),
    # not an instrument name. Using it as a ticker key buckets ALL equity
    # trades under 'EQ' and ALL F&O trades under 'FO', breaking FIFO matching.
    for candidate in ['tradingsymbol', 'symbol', 'instrument', 'scrip']:
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

    for _, row in df.iterrows():
        symbol     = str(row[ticker_col]) if ticker_col and ticker_col in row else "UNKNOWN"
        qty        = float(row["quantity"])
        price      = float(row["price"])
        trade_type = str(row["trade_type"]).upper().strip()
        date       = row["trade_date"]
        segment    = str(row.get("segment", "EQ")).upper().strip()
        is_fo      = (segment in ("FO", "F&O", "FUTURES", "OPTIONS", "CDS", "MF",
                          "NFO", "BCD", "MCX", "NCO")
                         or str(row.get("series", "")).upper().strip() in ("FO", "CD"))

        # FIX: skip zero/negative qty rows — prevents phantom trades and div-by-zero
        if qty <= 0 or price <= 0:
            audit_flags.append(
                f"⚠️ Skipped row: {symbol} {trade_type} qty={qty} price={price} "
                f"on {date.date() if hasattr(date, 'date') else date} invalid qty/price"
            )
            continue

        if trade_type == "BUY":
            if lots[symbol] and lots[symbol][0]["side"] == "short":
                remaining_qty = qty
                while remaining_qty > 0 and lots[symbol] and lots[symbol][0]["side"] == "short":
                    lot       = lots[symbol][0]
                    close_qty = min(remaining_qty, lot["qty"])
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[symbol].pop(0)
                        continue
                    trades.append({
                        "pnl":    round(np.clip(pnl_pct, -0.99, 0.99), 6),
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
                        f"(no open position - predates history window)"
                    )
                continue

            remaining_qty = qty
            while remaining_qty > 0 and lots[symbol] and lots[symbol][0]["side"] == "long":
                lot       = lots[symbol][0]
                close_qty = min(remaining_qty, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[symbol].pop(0)
                    continue
                trades.append({
                    "pnl":    round(np.clip(pnl_pct, -0.99, 0.99), 6),
                    "ticker": symbol,
                    "side":   "long",
                    "date":   date,
                })
                lot["qty"]    -= close_qty
                remaining_qty -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

        elif trade_type not in ("BUY", "SELL"):
            # FIX: flag unknown trade types instead of silently dropping
            audit_flags.append(
                f"⚠️ Skipped unknown trade_type '{trade_type}' for {symbol} "
                f"on {date.date() if hasattr(date, 'date') else date}"
            )

    if orphan_sells:
        audit_flags.append(
            f"⚠️ {len(orphan_sells)} sell(s) had no matching buy "
            f"(position predates TradeBook history) excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f" {len(open_positions)} position(s) still open "
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
# KITE ORDERS CSV PARSER
# =========================================================

def _is_kite_orders_csv(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip() for c in df.columns}
    required = {"transaction_type", "order_execution_time", "tradingsymbol", "filled_quantity"}
    return required.issubset(cols)


def _parse_kite_orders_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Kite Orders CSV into QuantProof trades.
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
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[symbol].pop(0)
                        continue
                    trades.append({
                        "pnl":    round(np.clip(pnl_pct, -0.99, 0.99), 6),
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
                        f"(no matching BUY - predates history window)"
                    )
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
                trades.append({
                    "pnl":    round(np.clip(pnl_pct, -0.99, 0.99), 6),
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
            f"⚠️ {len(orphan_sells)} sell(s) had no matching BUY "
            f"(position predates Orders history) excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f" {len(open_positions)} position(s) still open "
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
        f"Parsed {len(trades)} completed trade(s) from Zerodha Kite Orders CSV "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(open_positions)} position(s) still open)"
    )
    return result, note, audit_flags


# =========================================================
# RETRY DECORATOR FOR API CALLS
# =========================================================

import asyncio
from functools import wraps

def retry_on_failure(max_retries=3, delay=1.0):
    """
    Retry decorator for API calls with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = delay * (2 ** attempt)
                    print(f"[RETRY] API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator


# =========================================================
# ALPACA UI EXPORT PARSER
# =========================================================

def _is_alpaca_ui_export(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip().replace(".", "") for c in df.columns}
    return (
        ("avg fill price" in cols or "avg  fill price" in cols)
        and "filled at" in cols
        and "side" in cols
    )


def _parse_alpaca_ui_date(s) -> pd.Timestamp | None:
    if not s or str(s).strip() in ("-", "", "nan", "None"):
        return None
    try:
        return pd.to_datetime(str(s).strip(), errors="coerce")
    except Exception:
        return None


def _parse_alpaca_ui_export(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Alpaca UI order export (tab-separated) into QuantProof trades.
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
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[sym].pop(0)
                        continue
                    completed.append({
                        "pnl":        round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
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
                    f"(no matching buy - position predates history window)"
                )
                continue

            remaining = qty
            while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "long":
                lot       = lots[sym][0]
                close_qty = min(remaining, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[sym].pop(0)
                    continue
                completed.append({
                    "pnl":        round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
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
            f"⚠️ {len(orphan_sells)} sell(s) had no matching buy "
            f"(position predates window) excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f" {len(open_positions)} open position(s) not included "
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
        f"({total_filled} filled order-sides ?+ {paired_trades} round-trips, "
        f"{len(orphan_sells)} orphan sell(s) excluded)"
    )
    return result_df, note, audit_flags


# =========================================================
# ALPACA ORDER PAIRING — position-state tracker
# =========================================================

def pair_alpaca_orders(orders: list) -> tuple[pd.DataFrame, list]:
    """
    Converts raw Alpaca orders into a trades DataFrame.
    Accepts both REST API JSON and UI export dict shapes.
    """
    audit_flags = []

    def _norm_order(o: dict) -> dict | None:
        status = str(o.get("status", "")).lower().strip()
        if status != "filled":
            return None

        qty_raw = o.get("filled_qty") or o.get("qty") or 0
        try:
            qty = float(qty_raw)
        except (ValueError, TypeError):
            return None
        if qty <= 0:
            return None

        price_raw = (
            o.get("filled_avg_price")
            or o.get("avg_fill_price")
            or o.get("price")
            or 0
        )
        try:
            price_str = str(price_raw).replace(",", ".").strip()
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
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[sym].pop(0)
                        continue
                    completed.append({
                        "pnl":        round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
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
                # FIX: message now accurately describes behaviour (short entry, not excluded)
                orphan_sells.append(
                    f"{sym} sell@{price:.2f} on {ts.date()} "
                    f"(short entry - buy predates history, using sell price as cost basis)"
                )
                continue

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
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[sym].pop(0)
                    continue
                completed.append({
                    "pnl":        round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
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
        # FIX: message now accurately describes behaviour (short entries, not excluded)
        audit_flags.append(
            f"{len(orphan_sells)} sell(s) treated as short entries "
            f"(buy predates history window - sell price used as cost basis)"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   \"{s}\"")

    open_positions = {s: sum(l["qty"] for l in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(
            f"{s}({int(q)})" for s, q in list(open_positions.items())[:6]
        )
        audit_flags.append(
            f" {len(open_positions)} open position(s) not included "
            f"(no closing trade yet): {sym_list}"
        )

    if not completed:
        return pd.DataFrame(), audit_flags + [
            "No completed round-trip trades found. "
            "All positions may still be open, or all sells predate the history window."
        ]

    df = pd.DataFrame(completed).sort_values("date").reset_index(drop=True)

    df, regime_note = assign_regimes(df)
    audit_flags.append(regime_note)

    return df, audit_flags


# =========================================================
# OPTIONS CSV PARSER
# =========================================================

def _is_options_csv(df: pd.DataFrame) -> bool:
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    options_signals = ["strike", "expiry", "expiration", "option_type",
                       "call_put", "put_call", "premium", "underlying",
                       "contract", "iv", "delta", "gamma", "theta"]
    return sum(1 for s in options_signals if any(s in c for c in cols)) >= 2


def _has_overlapping_positions(df: pd.DataFrame) -> bool:
    df_cols = [c.lower().strip() for c in df.columns]
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
        return False

    try:
        opens  = pd.to_datetime(df[open_col],  errors="coerce").dropna()
        closes = pd.to_datetime(df[close_col],  errors="coerce").dropna()
        if len(opens) < 2 or len(closes) < 2:
            return False
        if opens.min().year < 2000 or closes.min().year < 2000:
            return False

        # O(n log n) sweep-line: sort all events, track active open count
        # Two positions overlap iff a new open starts before any prior close
        if len(opens) > 500:
            return False  # guard: too many rows to check, assume non-overlapping
        opens_sorted  = opens.sort_values().reset_index(drop=True)
        closes_sorted = closes.sort_values().reset_index(drop=True)

        # Merge events: +1 for open, -1 for close, sorted by time
        events = (
            [(t, 1) for t in opens_sorted] +
            [(t, -1) for t in closes_sorted]
        )
        events.sort(key=lambda x: (x[0], x[1]))  # close before open on same timestamp
        active = 0
        for _, delta in events:
            active += delta
            if active > 1:
                return True

        return False
    except Exception:
        return False


def _parse_options_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    import io as _io

    sample = raw[:2048]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim  = max(counts, key=counts.get)

    df = pd.read_csv(_io.StringIO(raw), sep=delim)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    if _has_overlapping_positions(df):
        raise ValueError(
            "Multi-leg or concurrent options positions detected. "
            "QuantProof supports single-leg directional options trades only "
            "(calls or puts bought and fully closed before the next trade). "
            "For spreads, iron condors, or straddles: calculate your daily "
            "P&L as a percentage of capital deployed and upload as a generic CSV."
        )

    pnl_col = next((c for c in df.columns if any(x in c for x in
                    ["pnl", "profit", "return", "gain", "pl", "net"])), None)
    if pnl_col is None:
        raise ValueError(
            "Could not find a P&L column in your options CSV. "
            "Expected a column named pnl, profit, return, gain, or pl."
        )

    pnl_vals = pd.to_numeric(df[pnl_col].astype(str).str.replace("%","",regex=False),
                              errors="coerce")

    premium_col = next((c for c in df.columns if any(x in c for x in
                        ["premium", "entry_price", "open_price", "cost"])), None)
    if premium_col is not None:
        premium_vals = pd.to_numeric(df[premium_col].astype(str).str.replace("%","",regex=False),
                                      errors="coerce")
        valid_mask = (premium_vals.notna()) & (premium_vals.abs() > 1e-9) & (pnl_vals.notna())
        returns = (pnl_vals[valid_mask] / premium_vals[valid_mask]).clip(-0.99, 0.99)
        norm_method = f"P&L / {premium_col} (per contract return)"
    else:
        valid_mask = pnl_vals.notna()
        returns = pnl_vals[valid_mask]
        if returns.abs().mean() > 1.0:
            returns = returns / 100.0
        returns = returns.clip(-0.99, 0.99)
        norm_method = "pnl column treated as fractional return"

    result = pd.DataFrame({"pnl": returns.values})

    date_col = next((c for c in df.columns if any(x in c for x in
                     ["date", "close_date", "exit_date", "time", "dt"])), None)
    if date_col:
        result["date"] = pd.to_datetime(
            df[date_col].values[:len(returns)], errors="coerce")

    ticker_col = next((c for c in df.columns if any(x in c for x in
                       ["ticker", "symbol", "underlying", "instrument"])), None)
    if ticker_col:
        result["ticker"] = df[ticker_col].values[:len(returns)]

    note = (
        f"OPTIONS: {len(result)} single-leg trades parsed. "
        f"Returns normalised as {norm_method}."
    )
    audit_flags = [
        "OPTIONS DATA: Crash simulations assume linear payoffs ? "
        "options convexity and delta/gamma effects are not modelled. "
        "Results are directionally indicative only.",
        "OPTIONS DATA: Validation assumes sequential non-overlapping trades. "
        "Multi-leg strategies (spreads, condors) are not supported ? "
        "use daily portfolio P&L as percentage of capital instead.",
        "OPTIONS DATA: Crash simulations scale stress from strategy volatility. "
        "High-volatility strategies (options, leveraged) will show elevated crash "
        "exposure - all 5 crises blowing up is expected behaviour for options "
        "strategies and does not indicate a validator error.",
    ]

    return result, note, audit_flags


# =========================================================
# MT5 TRADE HISTORY PARSER
# =========================================================

def _is_mt5_csv(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip().replace(" ", "_").replace("/", "_") for c in df.columns}
    deals_signals = {"deal", "direction", "commission", "swap", "balance"}
    deals_hit = len(deals_signals & cols)
    pos_signals = {"position", "s_l", "t_p", "commission", "swap"}
    time_cols = [c for c in cols if "time" in c]
    pos_hit = len(pos_signals & cols) + (1 if len(time_cols) >= 2 else 0)
    return deals_hit >= 3 or pos_hit >= 3


def _parse_mt5_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    import io as _io

    sample = raw[:2048]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim  = max(counts, key=counts.get)

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

    df.columns = [c.lower().strip().replace(" ", "_").replace("/", "_")
                  for c in df.columns]

    has_direction = "direction" in df.columns
    has_position  = "position"  in df.columns

    if not has_direction and not has_position:
        if "deal" in df.columns:
            has_direction = True
        else:
            raise ValueError(
                "Could not identify MT5 report format. "
                "Please export from MT5 Terminal → Account History → "
                "right-click → Save as Detailed Report."
            )

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

    price_cols = [c for c in df.columns if "price" in c]
    for pc in price_cols:
        df[pc] = _num(df[pc])

    time_cols = [c for c in df.columns if "time" in c]
    for tc in time_cols:
        df[tc] = pd.to_datetime(df[tc], errors="coerce")

    open_time_col  = time_cols[0] if time_cols else None
    close_time_col = time_cols[1] if len(time_cols) >= 2 else open_time_col

    symbol_col = next((c for c in df.columns if c in
                       ["symbol", "instrument", "pair", "asset"]), None)

    if "type" in df.columns:
        type_vals = df["type"].astype(str).str.lower().str.strip()
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

    if has_direction:
        df["direction"] = df["direction"].astype(str).str.lower().str.strip()
        df = df.sort_values(open_time_col) if open_time_col else df

        lots = defaultdict(list)

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

            if direction == "in/out":
                if lots[symbol]:
                    lot = lots[symbol].pop(0)
                    net_profit = profit + commission + swap
                    pnl_pct = _normalise_pnl(net_profit, lot["price"], lot["volume"], balance)
                    if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                        trades.append({"pnl": round(pnl_pct, 6), "date": date,
                                       "ticker": symbol})
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
                if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                    trades.append({"pnl": round(pnl_pct, 6), "date": date,
                                   "ticker": symbol})

        open_pos = {s: len(ls) for s, ls in lots.items() if ls}
        if open_pos:
            count = sum(open_pos.values())
            audit_flags.append(
                f" {count} MT5 position(s) still open (no closing deal) excluded"
            )

    else:
        price_open_col  = next((c for c in df.columns if "price" in c), None)
        price_close_col = next((c for c in df.columns
                                if "price" in c and c != price_open_col), None)

        for _, row in df.iterrows():
            profit    = float(row.get("profit", 0) or 0)
            commission= float(row.get("commission", 0) or 0)
            swap      = float(row.get("swap", 0) or 0)
            volume    = float(row.get("volume", 1) or 1)
            price_open = float(row[price_open_col] or 0) if price_open_col else 0
            balance    = 0.0
            symbol     = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            date       = row[close_time_col] if close_time_col else pd.NaT

            net_profit = profit + commission + swap
            pnl_pct    = _normalise_pnl(net_profit, price_open, volume, balance)
            if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                trades.append({"pnl": round(pnl_pct, 6), "date": date,
                                "ticker": symbol})

    if not trades:
        raise ValueError(
            "No completed trades could be extracted from the MT5 export. "
            "Make sure your report includes closed trades and has Profit column."
        )

    result = pd.DataFrame(trades)
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    if orphan_count:
        audit_flags.append(
            f"⚠️ {orphan_count} MT5 closing deal(s) had no matching open "
            f" excluded (position predates history window)"
        )

    audit_flags.append(
        "MT5 DATA: Returns normalised as net profit / (entry_price * volume). "
        "Crash simulations assume linear payoffs - leverage and margin effects not modelled. "
        "For accurate capacity analysis, ensure Volume column reflects lot sizes."
    )
    # FIX: warn forex/CFD users that contract size is not applied
    _forex_scale_warn = False
    if symbol_col and len(trades) > 0:
        sample_symbols = [t.get("ticker", "") for t in trades[:10]]
        # Forex pairs are typically 6-8 chars like EURUSD, GBPJPY, XAUUSD
        looks_forex = any(
            len(str(s)) in (6, 7, 8) and str(s).isalpha()
            for s in sample_symbols if s and s != "UNKNOWN"
        )
        if looks_forex:
            audit_flags.append(
                "⚠️ MT5 FOREX WARNING: Contract size (100,000 units/lot) is NOT applied. "
                "Returns are normalised as profit/(price*lots) not profit/(price*lots*100000). "
                "All forex PnL values are approximately 1000x overstated and will clip to ?0.99. "
                "To fix: pre-calculate per-trade % return and upload as a generic CSV instead."
            )
            # Embed sentinel so smart_parse_csv can set scale_warning for frontend banner
            _forex_scale_warn = True

    report = "deals" if has_direction else "positions"
    note = (
        f"📊 MT5: {len(trades)} completed trade(s) parsed from {report} report "
        f"({orphan_count} orphan(s) excluded)"
    )
    if _forex_scale_warn:
        note = "MT5_FOREX_SCALE_WARN||" + note
    return result, note, audit_flags


def _normalise_pnl(net_profit: float, entry_price: float,
                   volume: float, balance: float) -> float | None:
    """
    Normalise MT5 net profit to a decimal return.
    Returns None when entry_price*volume is zero (corrupt/missing data) so the
    trade is silently skipped rather than inventing a meaningless number via
    balance-ratio or /1000 fallbacks — those two paths produced values with no
    consistent financial meaning and have been removed.
    """
    if abs(entry_price * volume) > 1e-6:
        pnl = net_profit / (entry_price * volume)
        return float(np.clip(pnl, -0.99, 0.99))
    # entry_price or volume is zero — data is corrupt, skip this trade
    return None


def smart_parse_excel(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parses Excel files (.xlsx, .xls) into a trades DataFrame for QuantProofValidator.
    Returns (df, regime_note).
    """
    try:
        import openpyxl
        df = pd.read_excel(io.BytesIO(contents))
        regime_note = ""
        
        if "date" in df.columns and df["date"].notna().any():
            df, regime_note = assign_regimes(df)
        
        return df, regime_note
        
    except ImportError:
        raise HTTPException(400, "Excel support requires openpyxl package. Please install with: pip install openpyxl")
    except Exception as e:
        raise HTTPException(400, f"Could not parse Excel file: {str(e)}")


def smart_parse_csv(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parses CSV/TSV/Excel into a trades DataFrame for QuantProofValidator.
    Returns (df, regime_note).
    """
    # NEW: Detect Excel files by magic bytes
    if contents.startswith(b'PK\x03\x04') or contents.startswith(b'PK\x05\x06') or contents.startswith(b'PK\x07\x08'):
        return smart_parse_excel(contents)
    
    raw         = contents.decode("utf-8", errors="ignore")
    regime_note = ""

    sample = raw[:4096]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim  = max(counts, key=counts.get)

    try:
        _probe = pd.read_csv(io.StringIO(raw), sep=delim, nrows=5, dtype=str)
    except Exception:
        _probe = pd.DataFrame()

    # —? 0. Options CSV ——————————————————————————————————————————————————?
    if not _probe.empty and _is_options_csv(_probe):
        try:
            _df, _note, _flags = _parse_options_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _bm_type_for_regime = get_benchmark_type(_df, None)
                    _df, _regime_note = assign_regimes(_df, _bm_type_for_regime)
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | OPTIONS_FLAGS:" + "||".join(_flags)
                return _df, _note
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as _e:
            # Parser was positively identified — crash means bad data, not wrong parser
            raise HTTPException(400, f"Options CSV detected but could not be parsed: {_e}")

    # —? 0.5. MT5 Trade History ——————————————————————————————————————————?
    if not _probe.empty and _is_mt5_csv(_probe):
        try:
            _df, _note, _flags = _parse_mt5_csv(raw)
            if len(_df) > 0:
                # Detect MT5 forex scale warning sentinel — promote to scale_warning
                _mt5_forex_scale = False
                if _note.startswith("MT5_FOREX_SCALE_WARN||"):
                    _note = _note[len("MT5_FOREX_SCALE_WARN||"):]
                    _mt5_forex_scale = True
                if "date" in _df.columns and _df["date"].notna().any():
                    _bm_type_for_regime = get_benchmark_type(_df, None)
                    _df, _regime_note = assign_regimes(_df, _bm_type_for_regime)
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | MT5_FLAGS:" + "||".join(_flags)
                if _mt5_forex_scale:
                    _note += " | MT5_SCALE_WARNING"
                return _df, _note
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as _e:
            raise HTTPException(400, f"MT5 CSV detected but could not be parsed: {_e}")

    # —? 1. Kite TradeBook ————————————————————————————————————————————————?
    if not _probe.empty and _is_kite_csv(_probe):
        try:
            _df, _note, _flags = _parse_kite_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df, "nifty")
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                if "ticker" in _df.columns:
                    _opt_mask = _df["ticker"].astype(str).str.contains(
                        r'(CE|PE)$', regex=True, na=False)
                    if _opt_mask.any():
                        _opt_count = int(_opt_mask.sum())
                        _note += (f" | KITE OPTIONS: {_opt_count} options trades detected "
                                  f"(CE/PE). P&L used as-is from Kite — premium normalisation "
                                  f"not applied. Crash sims assume linear payoffs.")
                return _df, _note
        except Exception as _e:
            raise HTTPException(400, f"Kite TradeBook CSV detected but could not be parsed: {_e}")

    # —? 2. Kite Orders CSV ———————————————————————————————————————————————?
    if not _probe.empty and _is_kite_orders_csv(_probe):
        try:
            _df, _note, _flags = _parse_kite_orders_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df, "nifty")
                    _note += (" | " + _regime_note if _regime_note else "")
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                return _df, _note
        except Exception as _e:
            raise HTTPException(400, f"Kite Orders CSV detected but could not be parsed: {_e}")

    # —? 3. Alpaca UI export ——————————————————————————————————————————————?
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
        except Exception as _e:
            raise HTTPException(400, f"Alpaca UI CSV detected but could not be parsed: {_e}")

    # —? 4. Generic PnL CSV ———————————————————————————————————————————————?
    try:
        df = pd.read_csv(io.StringIO(raw), sep=delim)
        pnl_names = ["pnl", "profit", "return", "gain", "pl", "p&l"]
        for col in df.columns:
            if any(p in col.lower() for p in pnl_names):
                raw_vals = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
                vals = pd.to_numeric(raw_vals, errors="coerce").dropna()
                if len(vals) >= 5:
                    _was_divided = vals.abs().mean() > 1.0
                    if _was_divided:
                        vals = vals / 100.0
                    # Detection threshold stays at 0.50 — dollar PnL detector
                    _clipped_pct = (vals.abs() > 0.50).mean()
                    # FIX: clip to ?0.99 to match validator guard (was ?0.50)
                    vals = vals.clip(-0.99, 0.99)
                    _dollar_pnl_warning = _clipped_pct > 0.50
                    # HARD REJECT: only fires when values required /100 normalization
                    # AND still have >80% clipping — meaning they were dollar amounts
                    # (e.g. $450) not valid high-return decimals (e.g. 0.95 = 95% gain).
                    # Without _was_divided gate: a valid 70-95% return strategy
                    # (mean=0.85) would be incorrectly rejected as dollar amounts.
                    if _was_divided and _clipped_pct > 0.80:
                        raise ValueError(
                            f"PnL values appear to be dollar amounts or raw prices, not decimal returns. "
                            f"{int(_clipped_pct*100)}% of values exceed ?50% per trade after normalisation. "
                            "Upload decimal returns (e.g. 0.015 = 1.5%, not 1.5 or $15). "
                            "Check your PnL column and re-upload."
                        )
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
                        _bm_type_for_regime = get_benchmark_type(result, None)
                        result, regime_note = assign_regimes(result, _bm_type_for_regime)

                    if _dollar_pnl_warning:
                        regime_note = (
                            ("⚠️ " + regime_note + " | " if regime_note else "⚠️ ")
                            + "PnL values appear to be dollar amounts, not decimal returns "
                            "? clipped to 0.99. Upload a column with decimal returns (e.g. 0.015 = 1.5%)."
                        )

                    return result, regime_note
    except ValueError:
        raise
    except Exception:
        pass

    # —? 5. Flexible fallback —————————————————————————————————————————————?
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
                # FIX: clip to ?0.99 to match validator guard (was ?0.50)
                vals = vals.clip(-0.99, 0.99)
                return pd.DataFrame({"pnl": vals.values}), ""
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                # FIX: clip to ?0.99 to match validator guard (was ?0.50)
                vals = vals.clip(-0.99, 0.99)
                return pd.DataFrame({"pnl": vals.values}), ""
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError(
        "No valid PnL column found. Make sure your CSV has a 'pnl' or 'profit' column, "
        "or upload a Kite TradeBook, Kite Orders CSV, or Alpaca export."
    )


def _dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows (same date + ticker + pnl) that arise when
    a user uploads the same file twice or a broker export contains duplicate entries.
    Only deduplicates on columns that are present — pnl is always checked.
    """
    if df.empty:
        return df
    key_cols = [c for c in ["date", "ticker", "pnl", "side"] if c in df.columns]
    if not key_cols:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[dedup] Removed {removed} duplicate trade row(s)")
    return df


# =========================================================
# ROUTES
# =========================================================


# =========================================================
# SEBI COMPLIANCE CONFIGURATION
# =========================================================

SEBI_CONFIG = {
    'ops_threshold': {
        'max_orders_per_second': 10,
        'description': 'SEBI limit for orders per second before mandatory registration'
    },
    'session_logout': {
        'market_start': '09:15',
        'market_end': '15:30',
        'timezone': 'Asia/Kolkata',
        'description': 'NSE market hours for session logout verification'
    },
    'turnover_threshold': {
        'daily_limit': 1000000,  # 1 crore INR
        'description': 'Daily turnover threshold for enhanced reporting'
    },
    'last_updated': '2024-01-01',
    'version': '1.0'
}


# =========================================================
# SEBI COMPLIANCE CHECKS — KITE ONLY
# =========================================================

def _compute_sebi_checks(df: pd.DataFrame, request, registered_ip: str = "") -> dict:
    """
    Compute SEBI algo framework compliance checks.
    - OPS threshold: verified from timestamps (fully accurate)
    - Session logout: verified from timestamps (indirect)
    - Static IP: soft check — compare registered_ip to request IP in memory, never stored
    Returns a dict passed to the frontend for display. Does NOT affect Fundable Score.
    """
    result = {}

    # ── CHECK 1: OPS Threshold (<10 orders/second) ─────────────────────────
    # SEBI: strategies placing >=10 orders/second require mandatory registration
    try:
        ops_limit = SEBI_CONFIG['ops_threshold']['max_orders_per_second']
        if "date" in df.columns and df["date"].notna().any():
            # Group by second, count orders
            df_copy = df.copy()
            df_copy["second"] = df_copy["date"].dt.floor("S")
            ops_counts = df_copy.groupby("second").size()
            max_ops = ops_counts.max() if len(ops_counts) > 0 else 0
            
            result["ops_threshold"] = {
                "passed": max_ops < ops_limit,
                "value":  max_ops,
                "note":   f"Max {max_ops} orders in any single second — {'within' if max_ops < ops_limit else 'exceeds'} SEBI limit of {ops_limit} OPS",
                "method": "FROM_DATA",
            }
        else:
            result["ops_threshold"] = {
                "passed": True,
                "value":  "N/A",
                "note":   "No timestamp data — OPS check skipped",
                "method": "FROM_DATA",
            }
    except Exception:
        result["ops_threshold"] = {"passed": True, "value": "N/A", "note": "OPS check failed", "method": "FROM_DATA"}

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
            result["session_logout"] = {
                "passed": outside_count == 0,
                "value":  outside_count,
                "note":   (f"No orders outside market hours detected."
                           if outside_count == 0
                           else f"{outside_count} order(s) detected outside NSE market hours — verify session auto-logout."),
                "method": "FROM_DATA",
            }
        else:
            result["session_logout"] = {
                "passed": True, "value": "N/A",
                "note": "No timestamp data", "method": "FROM_DATA"
            }
    except Exception:
        result["session_logout"] = {"passed": True, "value": "N/A", "note": "Session check failed", "method": "FROM_DATA"}

    # ── CHECK 3: Static IP — soft check ────────────────────────────────────
    # Compare registered_ip (user-entered) to actual request IP
    # Performed in memory only — nothing written to disk or database
    try:
        if registered_ip and registered_ip.strip() and request:
            current_ip = (
                request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or (request.client.host if request.client else "")
                or "unknown"
            )
            reg = registered_ip.strip()
            match = (reg == current_ip)
            result["ip_check"] = {
                "result":        "match" if match else "mismatch",
                "match":         match,
                "registered_ip": reg,
                "current_ip":    current_ip,  # shown to user — their own IP
                "note":          ("IP matches registered address."
                                  if match else
                                  f"Current IP ({current_ip}) does not match registered IP ({reg}). "
                                  "Note: VPN or proxy use would cause a mismatch — this is a soft check only."),
                "method":        "SOFT_CHECK",
            }
        else:
            result["ip_check"] = {
                "result":  "not_entered",
                "match":   None,
                "note":    "No registered IP provided — check skipped. Verify directly with Zerodha.",
                "method":  "SOFT_CHECK",
            }
    except Exception:
        result["ip_check"] = {"result": "not_entered", "match": None, "note": "IP check failed", "method": "SOFT_CHECK"}

    return result


@app.get("/")
def root():
    return {
        "status": "QuantProof API is live",
        "version": API_VERSION,
        "checks": "44 institutional checks + 5 crash simulations",
        "new_in_v1.4": [
            "Information Ratio (vs Benchmark) - detects beta-capture disguised as alpha",
            "Beta Exposure (vs Benchmark) - flags high market correlation",
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


@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}


@app.get("/debug/reset-limit")
def debug_reset_limit(t: str = ""):
    """Dev endpoint — reset rate limit for your IP. Requires valid pro/dev token."""
    # For testing purposes, allow simple test tokens
    if t in ("quantproof-test-112211", "test-123", "dev-reset"):
        cleared = len(_validation_counts)
        _validation_counts.clear()
        return {"status": "ok", "cleared_ips": cleared, "message": "All rate limits reset"}
    
    if not _is_pro_token(t):
        raise HTTPException(403, "Valid pro token required as ?t= param")
    cleared = len(_validation_counts)
    _validation_counts.clear()
    return {"status": "ok", "cleared_ips": cleared, "message": "All rate limits reset"}


@app.get("/debug/token")
def debug_token(t: str = ""):
    """Dev endpoint — check if a token is valid and what tier it grants."""
    if not t:
        return {"error": "Pass ?t=YOUR_TOKEN"}
    tier   = _token_tier(t)
    is_pro = _is_pro_token(t)
    env_keys = {k: bool(os.getenv(k,"").strip()) for k in ("TEST_REPORT_TOKEN","DEV_TOKEN","ADMIN_TOKEN")}
    return {
        "token_preview": t[:8] + "..." if len(t) > 8 else t,
        "tier":          tier,
        "is_pro":        is_pro,
        "in_token_store":t in REPORT_ACCESS_TOKENS,
        "env_vars_set":  env_keys,
    }


@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    """
    Keep-alive endpoint.
    Hit this every 10 minutes via an external cron (cron-job.org, UptimeRobot, etc.)
    to prevent Render free tier from spinning down.
    URL to monitor: https://quantproof-backend.onrender.com/ping
    """
    return {"status": "alive", "version": API_VERSION}


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

REPORT_ACCESS_TOKENS = set()
TOKENS_FILE = "/tmp/qp_report_tokens.json"
LS_ORDER_TOKENS: dict = {}

def _save_tokens():
    try:
        import json
        with open(TOKENS_FILE, "w") as f:
            json.dump(list(REPORT_ACCESS_TOKENS), f)
    except Exception:
        pass

def _load_tokens():
    try:
        import json
        if os.path.exists(TOKENS_FILE):
            with open(TOKENS_FILE, "r") as f:
                tokens = json.load(f)
                REPORT_ACCESS_TOKENS.update(set(tokens))
    except Exception:
        pass

_load_tokens()

import collections
import asyncio as _asyncio
_PDF_SEMAPHORE = _asyncio.Semaphore(2)
VALIDATION_RESULT_CACHE: dict = collections.OrderedDict()
VALIDATION_CACHE_MAX = 50

def _save_validation_cache():
    # No persistence needed - use validation hash as direct cache
    pass

def _load_validation_cache():
    # No persistence needed - use validation hash as direct cache
    pass

def _cache_validation(v_hash: str, vr: dict):
    print(f"[DEBUG] Cache - Storing: '{v_hash}'")
    VALIDATION_RESULT_CACHE[v_hash] = vr
    if len(VALIDATION_RESULT_CACHE) > VALIDATION_CACHE_MAX:
        VALIDATION_RESULT_CACHE.popitem(last=False)
    print(f"[DEBUG] Cache - Size: {len(VALIDATION_RESULT_CACHE)} items")

def _get_cached_validation(v_hash: str) -> dict | None:
    print(f"[DEBUG] Cache Get - Looking for: '{v_hash}'")
    print(f"[DEBUG] Cache Get - Keys: {list(VALIDATION_RESULT_CACHE.keys())}")
    
    if v_hash in VALIDATION_RESULT_CACHE:
        print(f"[DEBUG] Cache Get - FOUND: '{v_hash}'")
        VALIDATION_RESULT_CACHE.move_to_end(v_hash)
        return VALIDATION_RESULT_CACHE[v_hash]
    
    print(f"[DEBUG] Cache Get - MISS: '{v_hash}'")
    return None

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


@app.post("/report/create-checkout-intl")
async def create_checkout_intl(payload: dict):
    """
    LemonSqueezy international checkout — USD/GBP/EUR payments.
    Stubbed pending LemonSqueezy approval. Set env vars to activate:
      LEMONSQUEEZY_API_KEY, LEMONSQUEEZY_STORE_ID, LEMONSQUEEZY_VARIANT_ID
    """
    import httpx
    ls_key      = os.getenv("LEMONSQUEEZY_API_KEY", "")
    ls_store_id = os.getenv("LEMONSQUEEZY_STORE_ID", "")
    ls_variant  = os.getenv("LEMONSQUEEZY_VARIANT_ID", "")
    if not ls_key or not ls_store_id or not ls_variant:
        raise HTTPException(503, {"error": "ls_not_configured",
            "message": "International payments not yet configured. "
                       "Set LEMONSQUEEZY_API_KEY, LEMONSQUEEZY_STORE_ID, and "
                       "LEMONSQUEEZY_VARIANT_ID in Render environment to activate."})
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.lemonsqueezy.com/v1/checkouts",
                headers={
                    "Authorization": f"Bearer {ls_key}",
                    "Accept":        "application/vnd.api+json",
                    "Content-Type":  "application/vnd.api+json",
                },
                json={
                    "data": {
                        "type": "checkouts",
                        "attributes": {
                            "checkout_data": {"custom": {"source": "quantproof"}},
                            "product_options": {
                                "redirect_url": (
                                    f"{os.getenv('BASE_URL', 'https://quantproof-frontend.onrender.com')}"
                                    "/?payment=success&provider=ls&order_id={checkout_id}"
                                )
                            }
                        },
                        "relationships": {
                            "store":   {"data": {"type": "stores",   "id": ls_store_id}},
                            "variant": {"data": {"type": "variants", "id": ls_variant}},
                        }
                    }
                }
            )
        resp.raise_for_status()
        checkout_url = resp.json()["data"]["attributes"]["url"]
        return {"checkout_url": checkout_url}
    except Exception as e:
        raise HTTPException(500, {"error": "ls_error", "message": str(e)})


@app.post("/report/webhook-ls")
async def lemonsqueezy_webhook(request: Request):
    """
    LemonSqueezy payment webhook — receives order_completed events.
    Generates a report access token and stores in LS_ORDER_TOKENS for
    the /report/ls-token polling endpoint to pick up.
    Stubbed pending LemonSqueezy approval.
    """
    import hashlib, hmac, time, json
    body      = await request.body()
    ls_secret = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
    if ls_secret:
        sig      = request.headers.get("x-signature", "")
        expected = hmac.new(ls_secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(400, "Invalid Lemon Squeezy webhook signature")
    try:
        event    = json.loads(body)
        ev_name  = event.get("meta", {}).get("event_name", "")
        if ev_name != "order_created":
            return {"status": "ignored"}
        order_id = str(event.get("data", {}).get("id", ""))
        key_secret = os.getenv("RAZORPAY_KEY_SECRET", os.getenv("LEMONSQUEEZY_API_KEY", "ls"))
        current_hour = int(time.time() // 3600)
        token = (
            f"RPT-{hashlib.sha256(f'ls:{current_hour}:{key_secret}:{order_id}'.encode()).hexdigest()[:40]}"
        )
        REPORT_ACCESS_TOKENS.add(token)
        LS_ORDER_TOKENS[order_id] = token
        _save_tokens()
        return {"status": "ok"}
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
    LS_ORDER_TOKENS[order_id] = token_now
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
    LS_ORDER_TOKENS[order_id] = token_now
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
    LS_ORDER_TOKENS[order_id] = token
    _save_pro_tokens()
    _save_tokens()
    return {"status": "ok", "token": token, "expires_in_days": 30}


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


# =========================================================
# PORTFOLIO VALIDATION — ₹199
# =========================================================

class PortfolioResponse(BaseModel):
    strategies:            List[Dict[str, Any]]
    correlation_matrix:    List[List[float]]
    strategy_names:        List[str]
    combined_sharpe:       float
    combined_drawdown:     float
    combined_win_rate:     float
    diversification_score: float
    avg_correlation:       float = 0.0
    combined_total_trades: int
    portfolio_verdict:     str
    audit_flags:           List[str]
    n_trials:              int = 10
    rp_weights:            List[float] = []        # risk parity allocation weights
    dd_coincidence:        float = 0.0             # fraction of DD periods overlapping
    portfolio_dsr:         Optional[float] = None  # combined strategy DSR score
    combined_equity:       List[float] = []        # normalised combined equity curve
    portfolio_plan:        Optional[Dict[str, Any]] = None  # AI fix plan


def _normalise_pnl_series(pnl_array: np.ndarray) -> np.ndarray:
    """Normalise PnL to unit variance — removes scale-dependence from correlation."""
    std = float(pnl_array.std())
    return pnl_array / std if std > 1e-10 else pnl_array


def _portfolio_correlation(dfs: list) -> np.ndarray:
    """
    Pairwise Pearson correlation on NORMALISED PnL.
    Normalising to unit variance makes correlation scale-independent —
    a ₹100-avg-trade strategy and a ₹10,000-avg-trade strategy
    are compared on signal shape, not magnitude.
    Aligns on date where available (inner join), falls back to positional.
    """
    n = len(dfs)
    mat = np.ones((n, n))
    series = []
    for df in dfs:
        raw = df["pnl"].values.astype(float)
        norm = _normalise_pnl_series(raw)
        if "date" in df.columns and df["date"].notna().any():
            idx = pd.to_datetime(df["date"], errors="coerce")
            s = pd.Series(norm, index=idx).sort_index()
        else:
            s = pd.Series(norm)
        series.append(s)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                si, sj = series[i], series[j]
                if hasattr(si.index, "date") and hasattr(sj.index, "date"):
                    aligned = pd.concat([si, sj], axis=1, join="inner")
                    if len(aligned) >= 5:
                        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    else:
                        min_len = min(len(si), len(sj))
                        corr = float(np.corrcoef(si.values[:min_len], sj.values[:min_len])[0, 1])
                else:
                    min_len = min(len(si), len(sj))
                    corr = float(np.corrcoef(si.values[:min_len], sj.values[:min_len])[0, 1])
                mat[i][j] = mat[j][i] = round(0.0 if np.isnan(corr) else corr, 4)
            except Exception:
                mat[i][j] = mat[j][i] = 0.0
    return mat


def _risk_parity_weights(dfs: list) -> list:
    """
    Inverse-volatility (risk parity) weights.
    Each strategy weighted by 1/vol so all contribute equally to portfolio variance.
    Institutional standard for multi-strategy allocation.
    """
    vols = []
    for df in dfs:
        v = float(df["pnl"].values.astype(float).std())
        vols.append(max(v, 1e-10))
    inv_vol = [1.0 / v for v in vols]
    total = sum(inv_vol)
    return [float(w / total) for w in inv_vol]


def _drawdown_coincidence(dfs: list, threshold: float = -0.02) -> float:
    """
    Fraction of drawdown-period steps where 2+ strategies are simultaneously
    in drawdown below threshold. Institutional target: < 0.30.
    High coincidence = strategies fail together = poor real diversification.
    """
    if len(dfs) < 2:
        return 0.0
    min_len = min(len(df) for df in dfs)
    dd_flags = []
    for df in dfs:
        pnl = df["pnl"].values.astype(float)[:min_len]
        eq = np.cumprod(1 + np.clip(pnl, -0.99, 0.99))
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / np.where(roll_max > 0, roll_max, 1.0)
        dd_flags.append(dd < threshold)
    stacked   = np.stack(dd_flags, axis=0)
    n_down    = stacked.sum(axis=0)
    any_down  = int((n_down >= 1).sum())
    multi_down = int((n_down >= 2).sum())
    return round(float(multi_down / any_down), 3) if any_down > 0 else 0.0


def _diversification_score(corr_matrix: np.ndarray, sharpes: list) -> float:
    """
    0-100 score. Higher = better diversified portfolio.
    Per-pair scoring:
      near-zero corr → 82-90 (ideal diversification)
      negative corr  → 96-100 (true hedge)
      0.0-0.3 corr   → 40-82 (ok)
      0.3-0.6 corr   → 10-40 (moderate concern)
      0.6+ corr      → 0-10  (high concern, minimal diversification)
    """
    n = len(sharpes)
    if n < 2:
        return 50.0

    def _pair_score(c: float) -> float:
        if c <= -0.3:  return 96.0
        elif c <= 0.0: return 82.0 + abs(c) * 47.0
        elif c < 0.3:  return 82.0 - c * 140.0
        elif c < 0.6:  return 40.0 - (c - 0.3) * 100.0
        else:          return max(0.0, 10.0 - (c - 0.6) * 25.0)

    off_diag = [float(corr_matrix[i][j]) for i in range(n) for j in range(i + 1, n)]
    avg_pair     = float(np.mean([_pair_score(c) for c in off_diag])) if off_diag else 50.0
    sharpe_bonus = (sum(1 for s in sharpes if s > 0) / max(n, 1)) * 8.0
    return round(min(100.0, max(0.0, avg_pair + sharpe_bonus)), 1)


def _portfolio_verdict(div_score: float, avg_corr: float, n: int) -> str:
    if div_score >= 75:
        return (f"Strong portfolio — {n} strategies with low average correlation "
                f"({avg_corr:.2f}). Each strategy adds independent alpha. "
                "This combination is institutionally viable for diversified deployment.")
    elif div_score >= 55:
        return (f"Moderate diversification — {n} strategies with some overlap "
                f"(avg correlation {avg_corr:.2f}). Consider strategies from different "
                "asset classes or time horizons to reduce correlation further.")
    elif div_score >= 35:
        return (f"High correlation detected (avg {avg_corr:.2f}) — these strategies "
                f"likely move together. During drawdowns they will all drawdown simultaneously. "
                "Portfolio diversification benefit is limited.")
    else:
        return (f"Poor diversification — strategies are highly correlated "
                f"(avg {avg_corr:.2f}). Running all {n} together provides minimal "
                "risk reduction over running just the best one.")


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
    Portfolio correlation analysis — ₹199 or PRO.
    Accepts 2-3 CSV files. Returns per-strategy scores + correlation matrix
    + combined stats + diversification score.
    """
    # Token check: TEST_REPORT_TOKEN, PRO, or PORT all grant access
    if not (_is_pro_token(x_portfolio_token) or
            _token_tier(x_portfolio_token) in ("portfolio", "pro")):
        raise HTTPException(402, {
            "error": "payment_required",
            "message": "Portfolio analysis requires the ₹199 Portfolio unlock or Pro plan.",
            "price_inr": 199,
        })

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
        rp_weights=[float(round(w, 3)) for w in rp_weights],
        dd_coincidence=float(dd_coincidence),
        portfolio_dsr=portfolio_dsr,
        combined_equity=[float(v) for v in combined_equity[:200]],
        portfolio_plan=portfolio_plan,
    )


# =========================================================
# IBKR CONFIGURATION
# =========================================================

IBKR_CONFIG = {
    'opening_detection': {
        'long_cash_outflow_ratio': 0.8,
        'short_cash_inflow_ratio': 0.2,
        'description': 'Thresholds for detecting opening positions'
    },
    'pnl_calculation': {
        'clip_range': 0.99,
        'min_pnl_threshold': 0.001,
        'description': 'P&L calculation parameters'
    },
    'validation': {
        'min_trades': 10,
        'description': 'Minimum trades required for validation'
    },
    'last_updated': '2024-01-01',
    'version': '1.0'
}


# =========================================================
# IBKR PARSER HELPER FUNCTIONS - Improved Logic
# =========================================================

def calculate_trade_side(buy_sell, qty, open_close, net_cash, price):
    """
    Calculate trade side with unified, clear logic.
    """
    if open_close == 'C':
        # Closed trade - determine by position direction
        return 'long' if qty < 0 else 'short'
    else:
        # Opening trade - determine by action
        return 'long' if buy_sell == 'SELL' else 'short'


def is_opening_position(open_close, qty, net_cash, price):
    """
    Detect if this is an opening position with robust logic.
    """
    # Explicit opening indicator
    if open_close == 'O':
        return True
    
    # Explicit closing indicator
    if open_close == 'C':
        return False
    
    # Heuristic detection when indicator is missing
    trade_value = abs(price * qty) if price > 0 and qty != 0 else 0
    
    if trade_value == 0:
        return False  # Can't determine, assume closing
    
    # Use configuration thresholds
    config = IBKR_CONFIG['opening_detection']
    
    # For long positions (qty > 0): opening requires significant cash outflow
    if qty > 0:
        cash_outflow_ratio = abs(net_cash) / trade_value if trade_value > 0 else 0
        return cash_outflow_ratio > config['long_cash_outflow_ratio']
    
    # For short positions (qty < 0): opening requires minimal cash inflow
    else:
        cash_inflow_ratio = abs(net_cash) / trade_value if trade_value > 0 else 0
        return cash_inflow_ratio < config['short_cash_inflow_ratio']


def calculate_pnl_fraction(net_cash, price, qty, fifo_pnl=None):
    """
    Calculate P&L as fraction with robust error handling.
    """
    import numpy as np
    
    # Use configuration
    config = IBKR_CONFIG['pnl_calculation']
    clip_range = config['clip_range']
    min_threshold = config['min_pnl_threshold']
    
    # Use FIFO P&L when available (most accurate)
    if fifo_pnl is not None:
        try:
            fifo_val = float(fifo_pnl)
            if abs(fifo_val) < min_threshold:  # Tiny P&L, likely rounding error
                return None
            
            cost = abs(price * qty) if price > 0 and qty != 0 else 1
            pnl_frac = fifo_val / cost
            return float(np.clip(pnl_frac, -clip_range, clip_range))
        except (ValueError, ZeroDivisionError):
            pass  # Fall back to net cash method
    
    # Net cash method
    if price <= 0 or qty == 0:
        return None  # Can't calculate without valid price/quantity
    
    cost_basis = abs(price * qty)
    pnl_frac = net_cash / cost_basis
    return float(np.clip(pnl_frac, -clip_range, clip_range))


def parse_ibkr_trade_improved(trade_element):
    """
    Simplified, maintainable IBKR trade parser using helper functions.
    """
    # Extract fields with case-insensitive fallbacks
    date_str = trade_element.get("tradeDate") or trade_element.get("TradeDate", "")
    symbol = trade_element.get("symbol", "") or trade_element.get("Symbol", "")
    qty = float(trade_element.get("quantity", 0) or trade_element.get("Quantity", 0) or 0)
    price = float(trade_element.get("tradePrice") or trade_element.get("TradePrice", 0) or 0)
    net_cash = float(trade_element.get("netCash") or trade_element.get("NetCash", 0) or 0)
    buy_sell = (trade_element.get("buySell") or trade_element.get("BuySell") or "").upper()
    open_close = (trade_element.get("openCloseIndicator") or trade_element.get("OpenCloseIndicator") or "").upper()
    fifo_pnl = trade_element.get("fifoPnlRealized") or trade_element.get("FifoPnlRealized")
    
    # Skip opening positions
    if is_opening_position(open_close, qty, net_cash, price):
        return None, "opening_position"
    
    # Calculate P&L
    pnl_frac = calculate_pnl_fraction(net_cash, price, qty, fifo_pnl)
    if pnl_frac is None:
        return None, "pnl_calculation_failed"
    
    # Calculate side using unified logic
    side = calculate_trade_side(buy_sell, qty, open_close, net_cash, price)
    
    return {
        "date": date_str,
        "pnl": round(pnl_frac, 6),
        "ticker": symbol,
        "side": side,
    }, None


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


@app.get("/webhook/tradingview/new-session", response_model=TVSessionResponse)
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


@app.post("/webhook/tradingview/{session_id}")
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


@app.get("/webhook/tradingview/{session_id}/status", response_model=TVStatusResponse)
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


@app.post("/webhook/tradingview/{session_id}/validate", response_model=ValidationResponse)
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


@app.post("/webhook/tradingview/{session_id}/reset")
def reset_tv_session(session_id: str):
    _tv_cleanup()
    if session_id not in TV_SESSIONS:
        raise HTTPException(404, "Session not found or expired")
    TV_SESSIONS[session_id]["alerts"] = []
    return {"status": "reset", "message": "Alerts cleared, session remains active"}


# ================= SERVER STARTUP =================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting QuantProof Backend Server...")
    print("📍 Enhanced Validation endpoints ready")
    print("📍 Prop Firm Compliance integration enabled")
    print("📍 Server running on: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)




































