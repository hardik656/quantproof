"""LLM fix-plan generation with Groq -> Anthropic -> OpenAI fallback, plus rule-based fallback.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import os

# —? AI Fix Plan ——————————————————————————————————————————————————————————————?
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


async def generate_ai_plan(report, date_range: str) -> dict:
    if not any([GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY]):
        return None

    scored = [c for c in report.checks if c.category != "Plausibility"]
    failed = [c for c in scored if not c.passed]
    passed = [c for c in scored if c.passed]

    # Get prop firm compliance data if available
    prop_firm_data = getattr(report, "prop_firm_compliance", None)
    prop_firm_summary = prop_firm_data.get("summary", {}) if prop_firm_data else {}
    prop_firm_firms = prop_firm_data.get("firms", {}) if prop_firm_data else {}

    if not failed:
        return {
            "summary": "Your strategy passed all checks. Focus on live execution and position sizing.",
            "fixes": [],
            "encouragement": "Rare result — this strategy has institutional-grade statistical foundations.",
        }

    failed_text = "\n".join(
        [
            f"- {c.name} (score {c.score}/100, category {c.category}): {c.value}. {c.insight}"
            for c in sorted(failed, key=lambda x: x.score)
        ]
    )

    # Add prop firm compliance failures to failed_text if engine is enabled
    if prop_firm_data:
        ineligible = [f for f, r in prop_firm_data.get("firms", {}).items() if not r.get("eligible")]
        if ineligible:
            failed_text += f"\n- Prop Firm Compliance: ineligible for {', '.join(ineligible)}"

    passed_text = ", ".join([c.name for c in passed])

    # Add prop firm compliance info to the prompt if available
    prop_firm_section = ""
    if prop_firm_data:
        # Format individual firm details properly
        firm_details = []
        for firm_name, firm_data in prop_firm_firms.items():
            status = "ELIGIBLE" if firm_data.get("eligible") else "NOT ELIGIBLE"
            score = firm_data.get("score", 0)

            # Format violations and passes
            violations = firm_data.get("violations", [])
            passes = firm_data.get("passes", [])

            violation_lines = []
            for v in violations:
                violation_lines.append(
                    f"• {v.get('name', 'Unknown')}: {v.get('actual', 'N/A')} (Required: {v.get('required', 'N/A')})"
                )

            pass_lines = []
            for p in passes:
                pass_lines.append(
                    f"• {p.get('name', 'Unknown')}: {p.get('actual', 'N/A')} (Required: {p.get('required', 'N/A')})"
                )

            firm_section = f"""
{firm_name.upper()} - {status} (Score: {score}/100)
Violations:
{chr(10).join(violation_lines) if violations else "• None"}
Passes:
{chr(10).join(pass_lines) if passes else "• None"}"""

            firm_details.append(firm_section)

        prop_firm_section = f"""
PROP FIRM COMPLIANCE:
- Total Firms Analyzed: {prop_firm_summary.get("total_firms", 0)}
- Eligible Firms: {prop_firm_summary.get("eligible_firms", 0)}
- Average Score: {prop_firm_summary.get("average_score", 0):.1f}/100
- Best Firm: {prop_firm_summary.get("best_firm", "None")}

{chr(10).join(firm_details)}
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
        import json

        import httpx

        if GROQ_API_KEY:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.3,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        elif ANTHROPIC_API_KEY:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        elif OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.3,
            }
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
            strategies,
            corr_matrix,
            rp_weights,
            div_score,
            dd_coincidence,
            combined_sharpe,
            combined_dd,
            avg_corr,
            strategy_names,
        )

    # Build per-strategy summary for prompt
    strat_lines = []
    for i, s in enumerate(strategies):
        w = rp_weights[i] if i < len(rp_weights) else 1 / len(strategies)
        strat_lines.append(
            f"Strategy {i + 1} '{s['name']}': "
            f"Score={s['fundable_score']}/100 grade={s['grade']} "
            f"Sharpe={s['sharpe']} MaxDD={round(abs(s['max_drawdown']) * 100, 1)}% "
            f"WinRate={s['win_rate']}% Trades={s['total_trades']} "
            f"RiskParityWeight={round(w * 100, 1)}% "
            f"CrashSurvived={s.get('crash_survived', '?')}/{s.get('crash_total', '5')} "
            f"ChecksFailed={s.get('checks_failed', 0)}/{s.get('checks_total', 44)}"
        )

    # Build correlation pairs
    n = len(strategies)
    corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            v = corr_matrix[i][j] if i < len(corr_matrix) and j < len(corr_matrix[i]) else 0
            corr_pairs.append(f"'{strategy_names[i]}' vs '{strategy_names[j]}': {v:+.3f}")

    # Top individual failures across all strategies
    all_failures = []
    for s in strategies:
        for c in s.get("checks", []):
            if not c.get("passed", True):
                all_failures.append(
                    {
                        "strategy": s["name"],
                        "check": c.get("name", ""),
                        "score": c.get("score", 0),
                        "category": c.get("category", ""),
                        "insight": c.get("insight", ""),
                        "fix": c.get("fix", ""),
                    }
                )
    all_failures.sort(key=lambda x: x["score"])
    top_failures = all_failures[:8]
    failure_lines = chr(10).join(
        [
            f"  [{f['strategy']}] {f['check']} ({f['category']}, score {f['score']}/100): {f['insight']}"
            for f in top_failures
        ]
    )

    prompt = f"""You are a quantitative portfolio manager reviewing a multi-strategy portfolio validation.
Think like someone who has run prop desks and fund-of-funds allocations for 15 years.
Be specific to the actual numbers. No generic advice.

PORTFOLIO SUMMARY:
- {n} strategies, Diversification Score: {div_score}/100
- Average pairwise correlation: {avg_corr:+.3f} (signed)
- Risk-parity combined Sharpe: {combined_sharpe}
- Combined max drawdown: {round(abs(combined_dd) * 100, 1)}%
- Drawdown coincidence: {round(dd_coincidence * 100, 1)}% (% of drawdown time where 2+ strategies fall together)
- Portfolio DSR: {portfolio_dsr if portfolio_dsr is not None else "n/a"}/100

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
        import json

        import httpx

        if GROQ_API_KEY:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.2,
            }
            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        elif ANTHROPIC_API_KEY:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            }
            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()

        elif OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.2,
            }
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
            strategies,
            corr_matrix,
            rp_weights,
            div_score,
            dd_coincidence,
            combined_sharpe,
            combined_dd,
            avg_corr,
            strategy_names,
        )


def _portfolio_rule_based_fixes(
    strategies,
    corr_matrix,
    rp_weights,
    div_score,
    dd_coincidence,
    combined_sharpe,
    combined_dd,
    avg_corr,
    strategy_names,
) -> dict:
    """
    Deterministic fix plan when no LLM is available.
    Generates quant-grade advice from the actual numbers.
    """
    n = len(strategies)
    fixes = []
    rank = 1

    # 1. High drawdown coincidence
    if dd_coincidence > 0.50:
        fixes.append(
            {
                "rank": rank,
                "issue": "High Drawdown Coincidence",
                "why_it_matters": (
                    f"Your strategies drawdown simultaneously {round(dd_coincidence * 100, 0):.0f}% of the time. "
                    f"Portfolio drawdown protection only works when strategies fall independently. "
                    f"At {round(dd_coincidence * 100, 0):.0f}% coincidence your combined max DD of "
                    f"{round(abs(combined_dd) * 100, 1)}% is close to your worst individual strategy."
                ),
                "specific_fix": (
                    "Reduce allocation to the highest-correlated pair. "
                    "Target coincidence below 30% — add a strategy that trades in a "
                    "different regime (e.g. if current strategies are trend-following, "
                    "add mean reversion or a short-vol strategy)."
                ),
                "score_impact": "+10 to +20 diversification score",
                "effort": "High",
            }
        )
        rank += 1

    # 2. High correlation pairs
    for i in range(n):
        for j in range(i + 1, n):
            v = corr_matrix[i][j] if i < len(corr_matrix) else 0
            if abs(v) > 0.50:
                fixes.append(
                    {
                        "rank": rank,
                        "issue": f"High Correlation: '{strategy_names[i]}' vs '{strategy_names[j]}'",
                        "why_it_matters": (
                            f"Correlation of {v:+.3f} means these two strategies share "
                            f"{round(v**2 * 100, 0):.0f}% of their variance (R²). "
                            f"Running both gives you redundant risk, not diversification."
                        ),
                        "specific_fix": (
                            "Either (a) reduce the lower-scoring strategy's weight to under 10%, "
                            "or (b) identify whether they trade the same underlying signal and merge "
                            "them into one strategy. Risk-parity already reduces the weaker one, "
                            "but correlation this high means both will drawdown together regardless of weight."
                        ),
                        "score_impact": "+8 to +15 diversification score if pair is separated",
                        "effort": "Medium",
                    }
                )
                rank += 1

    # 3. Weak strategy dragging portfolio
    scores = [
        (s["name"], s["fundable_score"], s["sharpe"], rp_weights[i] if i < len(rp_weights) else 1 / n)
        for i, s in enumerate(strategies)
    ]
    weakest = min(scores, key=lambda x: x[1])
    strongest = max(scores, key=lambda x: x[1])
    if weakest[1] < 50 and (strongest[1] - weakest[1]) > 20:
        fixes.append(
            {
                "rank": rank,
                "issue": f"Weak Strategy: '{weakest[0]}' (score {weakest[1]}/100)",
                "why_it_matters": (
                    f"'{weakest[0]}' scores {weakest[1]}/100 vs '{strongest[0]}' at {strongest[1]}/100. "
                    f"It currently holds {round(weakest[3] * 100, 1)}% of portfolio weight under risk parity. "
                    f"A strategy failing institutional checks introduces correlated tail risk even at low weight."
                ),
                "specific_fix": (
                    f"Fix '{weakest[0]}' as a standalone strategy first — run single-strategy "
                    f"validation and resolve its failed checks before including it in the portfolio. "
                    f"A strategy below 50/100 should not be in a live portfolio regardless of its "
                    f"diversification contribution."
                ),
                "score_impact": "+5 to +15 combined Sharpe improvement if replaced with a passing strategy",
                "effort": "High",
            }
        )
        rank += 1

    # 4. Low combined Sharpe vs individuals
    individual_sharpes = [s["sharpe"] for s in strategies]
    avg_sh = sum(individual_sharpes) / len(individual_sharpes) if individual_sharpes else 0
    if avg_sh > 0 and combined_sharpe < avg_sh * 0.8:
        fixes.append(
            {
                "rank": rank,
                "issue": "Combined Sharpe Below Average Individual",
                "why_it_matters": (
                    f"Portfolio combined Sharpe ({combined_sharpe:.2f}) is below the average individual "
                    f"Sharpe ({avg_sh:.2f}). Risk parity weighting should produce a combined Sharpe "
                    f"at or above the average when strategies are sufficiently uncorrelated. "
                    f"This indicates negative diversification — strategies are hurt by being combined."
                ),
                "specific_fix": (
                    "Check for regime mismatch: if your strategies all performed well in the "
                    "same bull market period, they will show low correlation in backtest but "
                    "high correlation in live drawdowns. Validate each strategy across different "
                    "market regimes before combining."
                ),
                "score_impact": "Regime-specific — varies by strategy",
                "effort": "High",
            }
        )
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
            for j in range(i + 1, n):
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
            f"Risk-parity combined Sharpe of {combined_sharpe:.2f} with {round(abs(combined_dd) * 100, 1)}% max DD is deployable."
        )
    else:
        summary_parts.append(
            f"Portfolio diversification score of {div_score}/100 is below institutional threshold (75+). "
            f"Average correlation of {avg_corr:+.3f} reduces the expected diversification benefit."
        )

    if dd_coincidence > 0.50:
        summary_parts.append(
            f"Critical issue: {round(dd_coincidence * 100, 0):.0f}% drawdown coincidence — strategies fall together "
            f"more than half the time, making portfolio drawdown protection largely illusory."
        )

    return {
        "summary": " ".join(summary_parts),
        "portfolio_fixes": fixes
        if fixes
        else [
            {
                "rank": 1,
                "issue": "Portfolio is well-constructed",
                "why_it_matters": "All key metrics pass institutional thresholds.",
                "specific_fix": "Focus on individual strategy improvements and live execution.",
                "score_impact": "Maintain current diversification",
                "effort": "Low",
            }
        ],
        "allocation_verdict": alloc_verdict,
        "strongest_combination": strongest_pair or "Insufficient data",
    }
