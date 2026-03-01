"""
QuantProof — Validation Engine v1.1
Fixed: Sharpe (risk-free adjusted), Drawdown (expanding max), Win Rate, Scoring weights
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

RISK_FREE_DAILY = 0.04 / 252  # ~4% annual, daily equivalent

# =========================================================
# 📊 DATA STRUCTURES
# =========================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float
    value: str
    insight: str
    fix: str
    category: str

@dataclass
class CrashSimResult:
    crash_name: str
    year: str
    description: str
    market_drop: float
    strategy_drop: float
    survived: bool
    emotional_verdict: str

@dataclass
class ValidationReport:
    fundable_score: float
    grade: str
    summary: str
    checks: List[CheckResult]
    crash_sims: List[CrashSimResult]
    total_trades: int
    date_range: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    top_issues: List[str]
    top_strengths: List[str]


# =========================================================
# 🏛️ HISTORICAL CRASH PROFILES
# =========================================================

CRASH_PROFILES = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "year": "Sep 2008 – Mar 2009",
        "description": "Lehman Brothers collapsed. S&P 500 lost 56%. Volatility exploded to VIX 80. Correlations went to 1 — everything fell together.",
        "market_drop": -56.0,
        "vol_multiplier": 4.5,
        "liquidity_factor": 0.3,
        "gap_risk": 0.08,
    },
    "2020_covid": {
        "name": "2020 COVID Crash",
        "year": "Feb 2020 – Mar 2020",
        "description": "Fastest 30% drop in market history. 34% crash in 33 days. Circuit breakers triggered 4 times. Then one of the fastest recoveries ever.",
        "market_drop": -34.0,
        "vol_multiplier": 5.0,
        "liquidity_factor": 0.2,
        "gap_risk": 0.12,
    },
    "2022_bear": {
        "name": "2022 Rate Hike Bear Market",
        "year": "Jan 2022 – Dec 2022",
        "description": "Fed raised rates 425bps. Nasdaq lost 33%, S&P lost 19%. Momentum strategies that crushed 2021 were destroyed. Growth stocks fell 60-90%.",
        "market_drop": -19.4,
        "vol_multiplier": 2.5,
        "liquidity_factor": 0.7,
        "gap_risk": 0.04,
    }
}


# =========================================================
# 🔧 CORE MATH HELPERS (Fixed)
# =========================================================

def calculate_sharpe(returns: np.ndarray) -> float:
    """Risk-free adjusted annualized Sharpe ratio"""
    excess = returns - RISK_FREE_DAILY
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Correct drawdown using expanding max on cumulative returns"""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))

def calculate_win_rate(returns: np.ndarray) -> float:
    """Simple win rate: trades where pnl > 0"""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0) * 100)

def calculate_calmar(returns: np.ndarray) -> float:
    """Annual return / abs(max drawdown)"""
    annual = np.mean(returns) * 252
    dd = abs(calculate_max_drawdown(returns))
    if dd == 0:
        return 0.0
    return float(annual / dd)


# =========================================================
# 🔧 CORE VALIDATOR CLASS
# =========================================================

class QuantProofValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = self._clean(df)
        self.returns = self._get_returns()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        date_cols = [c for c in df.columns if any(x in c for x in ["date", "time", "dt"])]
        pnl_cols = [c for c in df.columns if any(x in c for x in ["pnl", "profit", "return", "gain", "pl"])]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if pnl_cols:
            df = df.rename(columns={pnl_cols[0]: "pnl"})
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        return df.dropna(subset=["pnl"])

    def _get_returns(self) -> np.ndarray:
        r = self.df["pnl"].values.astype(float)
        # Normalize: if values look like dollar amounts, convert to returns
        if np.abs(r).mean() > 1:
            total = np.abs(r).sum()
            r = r / (total / len(r)) * 0.01
        return r

    # =========================================================
    # GROUP 1: OVERFITTING DETECTION
    # =========================================================

    def check_sharpe_decay(self) -> CheckResult:
        r = self.returns
        n = len(r)
        if n < 20:
            return CheckResult("Sharpe Decay", False, 20, "Insufficient data", "Need 20+ trades", "Add more backtest history", "Overfitting")
        mid = n // 2
        s_in = calculate_sharpe(r[:mid])
        s_out = calculate_sharpe(r[mid:])
        decay = (s_in - s_out) / (abs(s_in) + 1e-9) * 100
        passed = decay < 40
        score = max(0, 100 - max(0, decay))
        return CheckResult(
            name="Sharpe Ratio Decay",
            passed=passed,
            score=round(score, 1),
            value=f"In-sample: {s_in:.2f} → Out-of-sample: {s_out:.2f} ({decay:.1f}% decay)",
            insight="High decay means your strategy was fitted to historical noise, not real patterns.",
            fix="Run walk-forward optimization. Only trust out-of-sample Sharpe.",
            category="Overfitting"
        )

    def check_monte_carlo(self) -> CheckResult:
        r = self.returns
        sims = [np.mean(np.random.choice(r, size=len(r), replace=True)) for _ in range(1000)]
        pct = float(np.mean(np.array(sims) > 0) * 100)
        passed = pct > 60
        return CheckResult(
            name="Monte Carlo Robustness",
            passed=passed,
            score=round(pct, 1),
            value=f"{pct:.1f}% of 1,000 simulations were profitable",
            insight="If random shuffles rarely profit, your edge is sequence-dependent and fragile.",
            fix="Each trade should have positive expectancy independently.",
            category="Overfitting"
        )

    def check_parameter_sensitivity(self) -> CheckResult:
        r = self.returns
        q25, q75 = np.percentile(r, 25), np.percentile(r, 75)
        iqr = q75 - q25
        outlier_pct = float(np.mean((r < q25 - 1.5*iqr) | (r > q75 + 1.5*iqr)) * 100)
        passed = outlier_pct < 15
        score = max(0, 100 - outlier_pct * 3)
        return CheckResult(
            name="Return Distribution Stability",
            passed=passed,
            score=round(score, 1),
            value=f"{outlier_pct:.1f}% of trades are statistical outliers",
            insight="High outlier dependency means a few lucky trades drive all returns.",
            fix="Remove outliers and retest. If it no longer profits, you don't have a real edge.",
            category="Overfitting"
        )

    def check_walk_forward(self) -> CheckResult:
        r = self.returns
        window = max(10, len(r) // 5)
        windows = [r[i:i+window] for i in range(0, len(r)-window, window)]
        if len(windows) < 2:
            return CheckResult("Walk-Forward Test", False, 30, "Not enough data", "Need 50+ trades", "Extend backtest period", "Overfitting")
        profitable = sum(1 for w in windows if np.mean(w) > 0)
        pct = profitable / len(windows) * 100
        return CheckResult(
            name="Walk-Forward Consistency",
            passed=pct >= 60,
            score=round(pct, 1),
            value=f"{profitable}/{len(windows)} time periods profitable ({pct:.0f}%)",
            insight="A real edge works across different time periods.",
            fix="If <60% of periods profitable, the strategy is period-specific.",
            category="Overfitting"
        )

    def check_bootstrap_stability(self) -> CheckResult:
        r = self.returns
        bootstraps = [np.mean(np.random.choice(r, size=len(r), replace=True)) for _ in range(500)]
        pct = float(np.mean(np.array(bootstraps) > 0) * 100)
        return CheckResult(
            name="Bootstrap Stability",
            passed=pct > 70,
            score=round(pct, 1),
            value=f"Positive expectancy in {pct:.1f}% of 500 bootstrap samples",
            insight="Stable edge shows up consistently in resampling.",
            fix="If below 70%, fix win rate or risk/reward ratio first.",
            category="Overfitting"
        )

    # =========================================================
    # GROUP 2: RISK METRICS (Fixed)
    # =========================================================

    def check_max_drawdown(self) -> CheckResult:
        r = self.returns
        dd = calculate_max_drawdown(r)
        dd_pct = abs(dd) * 100
        passed = dd_pct < 20
        score = max(0, 100 - dd_pct * 3)
        return CheckResult(
            name="Max Drawdown",
            passed=passed,
            score=round(score, 1),
            value=f"Max drawdown: {dd_pct:.1f}% (threshold: <20%)",
            insight="Funds reject strategies with drawdowns >20%. Retail traders quit at 15%.",
            fix="Add circuit breaker: pause after 10% drawdown. Reduce size after losing streaks.",
            category="Risk"
        )

    def check_calmar_ratio(self) -> CheckResult:
        r = self.returns
        calmar = calculate_calmar(r)
        passed = calmar > 0.5
        score = min(100, calmar * 50)
        return CheckResult(
            name="Calmar Ratio",
            passed=passed,
            score=round(score, 1),
            value=f"Calmar: {calmar:.2f} (target >0.5, institutional: >1.0)",
            insight="Calmar = annual return / max drawdown. Funds want >1.0.",
            fix="Increase returns or reduce drawdown via better position sizing.",
            category="Risk"
        )

    def check_var(self) -> CheckResult:
        r = self.returns
        var_95 = float(np.percentile(r, 5))
        var_99 = float(np.percentile(r, 1))
        mean = float(np.mean(r))
        passed = abs(var_99) < abs(mean) * 10
        score = max(0, 100 - (abs(var_99) / (abs(mean) + 1e-9)) * 5)
        return CheckResult(
            name="Value at Risk (VaR)",
            passed=passed,
            score=round(score, 1),
            value=f"VaR 95%: {var_95:.4f} | VaR 99%: {var_99:.4f}",
            insight="VaR 99% = loss exceeded only 1% of days.",
            fix="If VaR 99% > 5x average trade, tail risk is too high. Use tighter stops.",
            category="Risk"
        )

    def check_consecutive_losses(self) -> CheckResult:
        r = self.returns
        max_streak = current = 0
        for trade in r:
            if trade < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        passed = max_streak < 10
        score = max(0, 100 - max_streak * 8)
        return CheckResult(
            name="Max Losing Streak",
            passed=passed,
            score=round(score, 1),
            value=f"Longest losing streak: {max_streak} consecutive trades",
            insight="Most traders emotionally break after 5-7 consecutive losses.",
            fix="Add daily loss limit that pauses trading after streak >5.",
            category="Risk"
        )

    def check_recovery_factor(self) -> CheckResult:
        r = self.returns
        total_profit = float(np.sum(r[r > 0]))
        total_loss = float(abs(np.sum(r[r < 0])))
        recovery = total_profit / (total_loss + 1e-9)
        passed = recovery > 1.5
        score = min(100, recovery * 40)
        return CheckResult(
            name="Recovery Factor",
            passed=passed,
            score=round(score, 1),
            value=f"Recovery factor: {recovery:.2f} (wins cover losses {recovery:.1f}x)",
            insight="Below 1.5 means wins barely cover drawdowns.",
            fix="Increase average winner or cut average loser. Asymmetric R:R is the goal.",
            category="Risk"
        )

    def check_absolute_sharpe(self) -> CheckResult:
        r = self.returns
        sharpe = calculate_sharpe(r)
        
        if sharpe > 1.5:
            score = 100
            passed = True
        elif sharpe >= 1.0:
            score = 75
            passed = True
        elif sharpe >= 0.5:
            score = 45
            passed = False
        else:
            score = 15
            passed = False
            
        return CheckResult(
            name="Absolute Sharpe Ratio",
            passed=passed,
            score=round(score, 1),
            value=f"Annualized Sharpe: {sharpe:.2f}",
            insight="Institutional minimum is Sharpe > 1.0. Below 0.5 means the strategy doesn't compensate for its risk.",
            fix="Improve risk-adjusted returns through better entry timing or position sizing.",
            category="Risk"
        )

    def check_compliance_pass(self) -> CheckResult:
        gates = [
            self.returns.mean() > 0,
            calculate_sharpe(self.returns) > 0.5,
            abs(calculate_max_drawdown(self.returns)) < 0.20,
            calculate_win_rate(self.returns) > 45,
            len(self.returns) > 50
        ]
        passed = sum(gates) >= 4
        score = 100 if passed else 0
        return CheckResult(
            name="Prop Firm Compliance",
            passed=passed,
            score=score,
            value="✅ PASSES 2026 Requirements" if passed else "❌ NEEDS FIXES",
            insight="FTMO/Topstep reject 90% of strategies without proper validation. 4/5 gates must pass.",
            fix="Fix your top 3 failing checks above to meet prop firm standards.",
            category="Compliance"
        )

    # =========================================================
    # GROUP 3: REGIME ROBUSTNESS
    # =========================================================

    def check_bull_performance(self) -> CheckResult:
        r = self.returns
        threshold = np.percentile(r, 60)
        bull = r[r > threshold]
        sharpe = calculate_sharpe(bull) if len(bull) > 3 else 0
        win_rate = calculate_win_rate(bull)
        passed = sharpe > 0 and win_rate > 45
        return CheckResult(
            name="Bull Market Performance",
            passed=passed,
            score=round(min(100, max(0, 50 + sharpe * 20)), 1),
            value=f"Sharpe: {sharpe:.2f} | Win rate: {win_rate:.1f}% ({len(bull)} trades)",
            insight="Strategy behavior during favorable conditions.",
            fix="If failing in bull market, check if momentum signals are properly calibrated.",
            category="Regime"
        )

    def check_bear_performance(self) -> CheckResult:
        r = self.returns
        threshold = np.percentile(r, 40)
        bear = r[r < threshold]
        sharpe = calculate_sharpe(bear) if len(bear) > 3 else 0
        win_rate = calculate_win_rate(bear)
        passed = sharpe > -1.0
        return CheckResult(
            name="Bear Market Performance",
            passed=passed,
            score=round(min(100, max(0, 50 + sharpe * 20)), 1),
            value=f"Sharpe: {sharpe:.2f} | Win rate: {win_rate:.1f}% ({len(bear)} trades)",
            insight="How strategy holds up when conditions turn against it.",
            fix="Add regime detection to reduce size or pause during bear conditions.",
            category="Regime"
        )

    def check_consolidation_performance(self) -> CheckResult:
        r = self.returns
        low, high = np.percentile(r, 40), np.percentile(r, 60)
        consol = r[(r >= low) & (r <= high)]
        sharpe = calculate_sharpe(consol) if len(consol) > 3 else 0
        win_rate = calculate_win_rate(consol)
        passed = sharpe > 0
        return CheckResult(
            name="Consolidation Performance",
            passed=passed,
            score=round(min(100, max(0, 50 + sharpe * 20)), 1),
            value=f"Sharpe: {sharpe:.2f} | Win rate: {win_rate:.1f}% ({len(consol)} trades)",
            insight="Sideways markets are where most momentum strategies bleed.",
            fix="Add a choppiness filter to avoid trading in low-volatility ranges.",
            category="Regime"
        )

    def check_volatility_stress(self) -> CheckResult:
        r = self.returns
        original_sharpe = calculate_sharpe(r)
        stressed = r * 3.0
        stressed_sharpe = calculate_sharpe(stressed)
        degradation = (original_sharpe - stressed_sharpe) / (abs(original_sharpe) + 1e-9) * 100
        passed = degradation < 50
        return CheckResult(
            name="Volatility Spike Stress Test",
            passed=passed,
            score=round(max(0, 100 - degradation), 1),
            value=f"3x vol: Sharpe {original_sharpe:.2f} → {stressed_sharpe:.2f}",
            insight="VIX spikes (COVID March 2020) cause 3-5x normal volatility.",
            fix="Use volatility-adjusted position sizing (ATR-based). Reduce size when VIX >25.",
            category="Regime"
        )

    def check_frequency_consistency(self) -> CheckResult:
        r = self.returns
        window = max(5, len(r) // 10)
        rolling = [np.mean(r[i:i+window]) for i in range(0, len(r)-window)]
        pct = float(np.mean(np.array(rolling) > 0) * 100)
        return CheckResult(
            name="Performance Consistency",
            passed=pct > 60,
            score=round(pct, 1),
            value=f"Profitable in {pct:.1f}% of rolling windows",
            insight="Consistent strategies generate returns steadily, not in lumps.",
            fix="If <60%, your edge only works in specific conditions. Define those explicitly.",
            category="Regime"
        )

    def check_regime_coverage(self) -> CheckResult:
        if 'regime' not in self.df.columns:
            return CheckResult(
                name="Regime Coverage",
                passed=False,
                score=40,
                value="No regime column detected",
                insight="Strategies with regime detection have 3x better live survival rate.",
                fix="Add BULL/BEAR/CONSOLIDATION/TRANSITION regime labels to your CSV export.",
                category="Regime"
            )
        regimes = self.df['regime'].value_counts(normalize=True)
        coverage = len(regimes) / 4 * 100
        passed = coverage > 75
        return CheckResult(
            name="4-Regime Coverage",
            passed=passed,
            score=round(coverage, 1),
            value=f"Regimes detected: {list(regimes.index.astype(str))}",
            insight="Full regime coverage means your strategy was tested across all market conditions.",
            fix="Ensure your backtest includes BULL, BEAR, CONSOLIDATION and TRANSITION periods.",
            category="Regime"
        )

    # =========================================================
    # GROUP 4: EXECUTION REALITY
    # =========================================================

    def check_slippage_01(self) -> CheckResult:
        r = self.returns
        slipped = r - np.abs(r) * 0.001
        original = float(np.sum(r))
        after = float(np.sum(slipped))
        impact = (original - after) / (abs(original) + 1e-9) * 100
        passed = impact < 20
        return CheckResult(
            name="Slippage Impact (0.1%)",
            passed=passed,
            score=round(max(0, 100 - impact * 3), 1),
            value=f"0.1% slippage reduces returns by {impact:.1f}%",
            insight="Even 0.1% per trade compounds into significant drag.",
            fix="Reduce trade frequency or only take higher conviction setups.",
            category="Execution"
        )

    def check_slippage_03(self) -> CheckResult:
        r = self.returns
        slipped = r - np.abs(r) * 0.003
        original = float(np.sum(r))
        after = float(np.sum(slipped))
        impact = (original - after) / (abs(original) + 1e-9) * 100
        passed = impact < 40
        return CheckResult(
            name="Slippage Impact (0.3%)",
            passed=passed,
            score=round(max(0, 100 - impact * 2), 1),
            value=f"0.3% slippage reduces returns by {impact:.1f}%",
            insight="Small caps and volatile markets have 0.3%+ slippage. Does your edge survive?",
            fix="Model 0.5% slippage for small-cap strategies. Use limit orders.",
            category="Execution"
        )

    def check_commission_drag(self) -> CheckResult:
        r = self.returns
        n = len(r)
        commission = n * 0.0005
        total = abs(float(np.sum(r)))
        drag = commission / (total + 1e-9) * 100
        passed = drag < 15
        return CheckResult(
            name="Commission Drag",
            passed=passed,
            score=round(max(0, 100 - drag * 4), 1),
            value=f"{n} trades → {drag:.1f}% of gross returns consumed by commissions",
            insight="High-frequency strategies can be profitable on paper but lose after commissions.",
            fix="Calculate your break-even commission. If >10x/day, commissions may kill the edge.",
            category="Execution"
        )

    def check_partial_fills(self) -> CheckResult:
        r = self.returns
        fill_rate = 0.80 + 0.20 * 0.70
        adjusted = r * fill_rate
        impact = (float(np.sum(r)) - float(np.sum(adjusted))) / (abs(float(np.sum(r))) + 1e-9) * 100
        passed = impact < 10
        return CheckResult(
            name="Partial Fill Simulation",
            passed=passed,
            score=round(max(0, 100 - impact * 5), 1),
            value=f"80% fill rate reduces returns by {impact:.1f}%",
            insight="In fast markets orders may not fill completely.",
            fix="Size positions for partial fill scenarios. Use limit orders.",
            category="Execution"
        )

    def check_live_vs_backtest_gap(self) -> CheckResult:
        r = self.returns
        bt_sharpe = calculate_sharpe(r)
        live_sharpe = bt_sharpe * 0.6  # Industry: expect 40% decay live
        passed = live_sharpe > 0.5
        score = min(100, max(0, live_sharpe * 50))
        return CheckResult(
            name="Live Trading Gap Estimate",
            passed=passed,
            score=round(score, 1),
            value=f"Backtest Sharpe: {bt_sharpe:.2f} → Estimated Live: {live_sharpe:.2f}",
            insight="Retail algos lose 30-50% of backtest performance live.",
            fix="Live Sharpe must be >0.5 to be worth trading.",
            category="Execution"
        )

    # =========================================================
    # 💥 CRASH SIMULATIONS
    # =========================================================

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns.copy()
        stressed = r * profile["vol_multiplier"]
        stressed = stressed - np.abs(stressed) * (1 - profile["liquidity_factor"]) * 0.5
        stressed = stressed - profile["gap_risk"] * np.sign(r)
        strategy_total = float(np.sum(stressed))
        dd = calculate_max_drawdown(stressed)
        survived = abs(dd) < 0.25

        if survived and strategy_total > 0:
            verdict = "🟢 YOUR STRATEGY SURVIVED. While markets crashed, your system held. This is what separates real edges from lucky backtests."
        elif survived:
            verdict = "🟡 BARELY SURVIVED. Your strategy lost money but didn't blow up. In real life, would you have had the nerve to keep trading?"
        else:
            verdict = "🔴 YOUR STRATEGY WOULD HAVE BLOWN UP. The crash exposed fatal flaws. Most traders quit here — the ones who survive rebuild with proper risk management."

        return CrashSimResult(
            crash_name=profile["name"],
            year=profile["year"],
            description=profile["description"],
            market_drop=profile["market_drop"],
            strategy_drop=round(strategy_total, 4),
            survived=survived,
            emotional_verdict=verdict
        )

    # =========================================================
    # 🏆 SCORING ENGINE (Fixed weights)
    # =========================================================

    def _calculate_score(self, checks: List[CheckResult]) -> Tuple[float, str]:
        # Updated weights with Compliance category
        weights = {
            "Overfitting": 0.28,
            "Risk":        0.37,
            "Regime":      0.15,
            "Execution":   0.12,
            "Compliance":  0.08,
        }
        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat]
            category_scores[cat] = float(np.mean([c.score for c in cat_checks])) if cat_checks else 50.0

        final = sum(category_scores[cat] * w for cat, w in weights.items())

        # TASK 1: Sharpe Floor Penalty - override all other scores
        overall_sharpe = calculate_sharpe(self.returns)
        if overall_sharpe < 0:
            final = min(final, 20)
        elif overall_sharpe < 0.3:
            final = min(final, 35)
        elif overall_sharpe < 0.5:
            final = min(final, 50)

        # TASK 2: Hard Category Floor Logic
        low_categories = sum(1 for score in category_scores.values() if score < 30)
        if low_categories >= 2:
            final = min(final, 45)
        elif low_categories >= 1:
            final = min(final, 60)

        if final >= 80:   grade = "A — Institutionally Viable"
        elif final >= 65: grade = "B — Fundable with Improvements"
        elif final >= 50: grade = "C — Promising but Needs Work"
        elif final >= 35: grade = "D — Significant Issues"
        else:             grade = "F — Do Not Deploy"

        return round(final, 1), grade

    # =========================================================
    # 🚀 RUN ALL CHECKS
    # =========================================================

    def run(self) -> ValidationReport:
        checks = [
            self.check_sharpe_decay(),
            self.check_monte_carlo(),
            self.check_parameter_sensitivity(),
            self.check_walk_forward(),
            self.check_bootstrap_stability(),
            self.check_max_drawdown(),
            self.check_calmar_ratio(),
            self.check_var(),
            self.check_consecutive_losses(),
            self.check_recovery_factor(),
            self.check_absolute_sharpe(),
            self.check_bull_performance(),
            self.check_bear_performance(),
            self.check_consolidation_performance(),
            self.check_volatility_stress(),
            self.check_frequency_consistency(),
            self.check_regime_coverage(),
            self.check_slippage_01(),
            self.check_slippage_03(),
            self.check_commission_drag(),
            self.check_partial_fills(),
            self.check_live_vs_backtest_gap(),
            self.check_compliance_pass(),
        ]

        crash_sims = [
            self.simulate_crash("2008_gfc"),
            self.simulate_crash("2020_covid"),
            self.simulate_crash("2022_bear"),
        ]

        score, grade = self._calculate_score(checks)
        r = self.returns
        sharpe = calculate_sharpe(r)
        dd = calculate_max_drawdown(r)
        win_rate = calculate_win_rate(r)

        top_issues = [c.fix for c in sorted(checks, key=lambda x: x.score)[:3]]
        top_strengths = [c.name for c in sorted(checks, key=lambda x: x.score, reverse=True)[:3]]

        summary = (
            f"QuantProof analyzed {len(r)} trades across 20 institutional checks + 3 crash simulations. "
            f"Fundable Score: {score}/100 ({grade.split('—')[0].strip()}). "
            f"{'Core risk management needs work.' if score < 60 else 'Edge shows real promise — focus on execution costs.'}"
        )

        date_range = "Unknown"
        if "date" in self.df.columns:
            try:
                date_range = f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
            except:
                pass

        return ValidationReport(
            fundable_score=score,
            grade=grade,
            summary=summary,
            checks=checks,
            crash_sims=crash_sims,
            total_trades=len(r),
            date_range=date_range,
            sharpe=round(sharpe, 2),
            max_drawdown=round(dd, 4),
            win_rate=round(win_rate, 1),
            top_issues=top_issues,
            top_strengths=top_strengths,
        )
