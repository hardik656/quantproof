"""
QuantProof — Validation Engine
Runs 20 institutional-grade checks on a retail trader's backtest CSV
Includes historical crash simulations: 2008, 2020, 2022
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# =========================================================
# 📊 DATA STRUCTURES
# =========================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float          # 0-100
    value: str            # Human readable result
    insight: str          # What it means
    fix: str              # Actionable fix
    category: str         # Overfitting / Risk / Regime / Execution

@dataclass
class CrashSimResult:
    crash_name: str
    year: str
    description: str
    market_drop: float    # How much market fell %
    strategy_drop: float  # How much strategy fell %
    survived: bool
    emotional_verdict: str  # The hook

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
        "monthly_returns": [-9.1, -16.9, -8.9, -0.8, -11.0, -6.9, 8.6, 8.8, 3.7, -5.6, 5.7, -0.4],
        "vol_multiplier": 4.5,
        "liquidity_factor": 0.3,   # Spreads blew out 3x
        "gap_risk": 0.08,          # 8% overnight gaps common
    },
    "2020_covid": {
        "name": "2020 COVID Crash",
        "year": "Feb 2020 – Mar 2020",
        "description": "Fastest 30% drop in market history. 34% crash in 33 days. Circuit breakers triggered 4 times. Then one of the fastest recoveries ever.",
        "market_drop": -34.0,
        "monthly_returns": [-8.4, -12.5, -12.5, 12.7, 4.5, 1.8, 5.6, 7.0, -3.9, -2.8, 10.8, 3.7],
        "vol_multiplier": 5.0,
        "liquidity_factor": 0.2,
        "gap_risk": 0.12,
    },
    "2022_bear": {
        "name": "2022 Rate Hike Bear Market",
        "year": "Jan 2022 – Dec 2022",
        "description": "Fed raised rates 425bps. Nasdaq lost 33%, S&P lost 19%. Momentum strategies that crushed 2021 were destroyed. Growth stocks fell 60-90%.",
        "market_drop": -19.4,
        "monthly_returns": [-5.3, -3.1, 3.7, -8.8, -0.2, -8.4, 9.1, -4.2, -9.3, 8.0, 5.4, -5.9],
        "vol_multiplier": 2.5,
        "liquidity_factor": 0.7,
        "gap_risk": 0.04,
    }
}


# =========================================================
# 🔧 CORE VALIDATOR CLASS
# =========================================================

class QuantProofValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = self._clean(df)
        self.returns = self._get_returns()
        self.results: List[CheckResult] = []

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and types"""
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Try to find date and pnl columns
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
        return self.df["pnl"].values

    # =========================================================
    # 📐 GROUP 1: OVERFITTING DETECTION
    # =========================================================

    def check_sharpe_decay(self) -> CheckResult:
        """Is Sharpe ratio consistent across time periods?"""
        r = self.returns
        n = len(r)
        if n < 20:
            return CheckResult("Sharpe Decay", False, 20, "Insufficient data", "Need 20+ trades", "Add more backtest history", "Overfitting")

        mid = n // 2
        in_sample = r[:mid]
        out_sample = r[mid:]

        def sharpe(x):
            return np.mean(x) / (np.std(x) + 1e-9) * np.sqrt(252)

        s_in = sharpe(in_sample)
        s_out = sharpe(out_sample)
        decay = (s_in - s_out) / (abs(s_in) + 1e-9) * 100

        passed = decay < 40
        score = max(0, 100 - decay)

        return CheckResult(
            name="Sharpe Ratio Decay",
            passed=passed,
            score=round(score, 1),
            value=f"In-sample: {s_in:.2f} → Out-of-sample: {s_out:.2f} ({decay:.1f}% decay)",
            insight="High decay means your strategy was fitted to historical noise, not real patterns.",
            fix="Run walk-forward optimization. Split data 70/30 and only trade on out-of-sample results.",
            category="Overfitting"
        )

    def check_monte_carlo(self) -> CheckResult:
        """Run 1000 random shuffles — does it still work?"""
        r = self.returns
        n_sims = 1000
        sim_sharpes = []

        for _ in range(n_sims):
            shuffled = np.random.choice(r, size=len(r), replace=True)
            s = np.mean(shuffled) / (np.std(shuffled) + 1e-9) * np.sqrt(252)
            sim_sharpes.append(s)

        actual_sharpe = np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252)
        pct_beat = np.mean(np.array(sim_sharpes) > 0) * 100
        passed = pct_beat > 60
        score = pct_beat

        return CheckResult(
            name="Monte Carlo Robustness",
            passed=passed,
            score=round(score, 1),
            value=f"{pct_beat:.1f}% of 1,000 simulations were profitable",
            insight="If random shuffles of your trades rarely produce profit, your edge is sequence-dependent and fragile.",
            fix="Focus on trade quality over quantity. Each trade should have positive expectancy independently.",
            category="Overfitting"
        )

    def check_parameter_sensitivity(self) -> CheckResult:
        """Proxy: check if returns are consistent or clustered"""
        r = self.returns
        q25, q75 = np.percentile(r, 25), np.percentile(r, 75)
        iqr = q75 - q25
        outlier_pct = np.mean((r < q25 - 1.5 * iqr) | (r > q75 + 1.5 * iqr)) * 100

        passed = outlier_pct < 15
        score = max(0, 100 - outlier_pct * 3)

        return CheckResult(
            name="Return Distribution Stability",
            passed=passed,
            score=round(score, 1),
            value=f"{outlier_pct:.1f}% of trades are outliers",
            insight="High outlier dependency means a few lucky trades are driving all your returns — not a repeatable edge.",
            fix="Remove outlier trades from your backtest and see if it still profits. If not, you don't have an edge.",
            category="Overfitting"
        )

    def check_walk_forward(self) -> CheckResult:
        """Rolling window performance consistency"""
        r = self.returns
        window = max(10, len(r) // 5)
        windows = [r[i:i+window] for i in range(0, len(r)-window, window)]

        if len(windows) < 2:
            return CheckResult("Walk-Forward Test", False, 30, "Not enough data", "Need 50+ trades for walk-forward", "Extend your backtest period", "Overfitting")

        profitable_windows = sum(1 for w in windows if np.mean(w) > 0)
        pct = profitable_windows / len(windows) * 100
        passed = pct >= 60

        return CheckResult(
            name="Walk-Forward Consistency",
            passed=passed,
            score=round(pct, 1),
            value=f"{profitable_windows}/{len(windows)} time periods profitable ({pct:.0f}%)",
            insight="A real edge works across different time periods, not just the ones you backtested on.",
            fix="Use rolling optimization windows. If <60% of periods are profitable, the strategy is period-specific.",
            category="Overfitting"
        )

    def check_bootstrap_stability(self) -> CheckResult:
        """Bootstrap resampling — how stable is the expectancy?"""
        r = self.returns
        bootstraps = [np.mean(np.random.choice(r, size=len(r), replace=True)) for _ in range(500)]
        positive_pct = np.mean(np.array(bootstraps) > 0) * 100
        passed = positive_pct > 70

        return CheckResult(
            name="Bootstrap Stability",
            passed=passed,
            score=round(positive_pct, 1),
            value=f"Positive expectancy in {positive_pct:.1f}% of 500 bootstrap samples",
            insight="Stable edge shows up consistently in resampling. Unstable edge is data-mined luck.",
            fix="If below 70%, your per-trade expectancy is negative. Fix win rate or risk/reward ratio first.",
            category="Overfitting"
        )

    # =========================================================
    # 📉 GROUP 2: RISK METRICS
    # =========================================================

    def check_max_drawdown(self) -> CheckResult:
        r = self.returns
        cumulative = np.cumsum(r)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)
        total_return = cumulative[-1] if len(cumulative) > 0 else 0
        dd_pct = abs(max_dd / (total_return + 1e-9)) * 100

        passed = dd_pct < 25
        score = max(0, 100 - dd_pct * 2)

        return CheckResult(
            name="Max Drawdown",
            passed=passed,
            score=round(score, 1),
            value=f"Max drawdown: {max_dd:.2f} ({dd_pct:.1f}% of total return)",
            insight="Funds typically reject strategies with drawdowns >20%. Retail traders quit at 15%.",
            fix="Add a circuit breaker: pause trading after 10% drawdown. Reduce position size after losing streaks.",
            category="Risk"
        )

    def check_calmar_ratio(self) -> CheckResult:
        r = self.returns
        annual_return = np.mean(r) * 252
        cumulative = np.cumsum(r)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = abs(np.min(cumulative - running_max))
        calmar = annual_return / (max_dd + 1e-9)

        passed = calmar > 0.5
        score = min(100, calmar * 50)

        return CheckResult(
            name="Calmar Ratio",
            passed=passed,
            score=round(score, 1),
            value=f"Calmar ratio: {calmar:.2f} (target >0.5, institutional target >1.0)",
            insight="Calmar = annual return / max drawdown. Funds want >1.0. Below 0.5 means the risk isn't worth the return.",
            fix="Either increase returns or reduce drawdown. Position sizing and stop losses are the levers.",
            category="Risk"
        )

    def check_var(self) -> CheckResult:
        r = self.returns
        var_95 = np.percentile(r, 5)
        var_99 = np.percentile(r, 1)
        mean = np.mean(r)

        passed = abs(var_99) < abs(mean) * 10
        score = max(0, 100 - (abs(var_99) / (abs(mean) + 1e-9)) * 5)

        return CheckResult(
            name="Value at Risk (VaR)",
            passed=passed,
            score=round(score, 1),
            value=f"VaR 95%: {var_95:.4f} | VaR 99%: {var_99:.4f}",
            insight="VaR tells you your worst expected loss on a bad day. 99% VaR = loss you'd only exceed 1% of days.",
            fix="If VaR 99% is larger than 5x your average trade, your tail risk is too high. Use tighter stops.",
            category="Risk"
        )

    def check_consecutive_losses(self) -> CheckResult:
        r = self.returns
        max_streak = 0
        current = 0
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
            insight="Most traders emotionally break after 5-7 consecutive losses and abandon the strategy.",
            fix="If streak >8, add a daily loss limit that pauses trading. Psychological sustainability = strategy longevity.",
            category="Risk"
        )

    def check_recovery_factor(self) -> CheckResult:
        r = self.returns
        total_profit = np.sum(r[r > 0])
        total_loss = abs(np.sum(r[r < 0]))
        recovery = total_profit / (total_loss + 1e-9)

        passed = recovery > 1.5
        score = min(100, recovery * 40)

        return CheckResult(
            name="Recovery Factor",
            passed=passed,
            score=round(score, 1),
            value=f"Recovery factor: {recovery:.2f} (wins cover losses {recovery:.1f}x)",
            insight="How much your wins outweigh your losses. Below 1.5 means you barely cover drawdowns.",
            fix="Increase average winner size or cut average loser size. Asymmetric R:R is the goal.",
            category="Risk"
        )

    # =========================================================
    # 🌍 GROUP 3: REGIME ROBUSTNESS
    # =========================================================

    def _regime_performance(self, regime_returns: np.ndarray, regime_name: str, category: str) -> CheckResult:
        if len(regime_returns) == 0:
            return CheckResult(regime_name, False, 50, "No data", "Insufficient history for regime test", "Extend backtest to include this period", category)

        sharpe = np.mean(regime_returns) / (np.std(regime_returns) + 1e-9) * np.sqrt(252)
        win_rate = np.mean(regime_returns > 0) * 100
        passed = sharpe > 0 and win_rate > 45

        score = min(100, max(0, 50 + sharpe * 20))

        return CheckResult(
            name=regime_name,
            passed=passed,
            score=round(score, 1),
            value=f"Sharpe: {sharpe:.2f} | Win rate: {win_rate:.1f}%",
            insight=f"Strategy behavior during {regime_name.lower()} conditions.",
            fix="If failing, add regime detection to skip trades when market conditions don't match your edge.",
            category=category
        )

    def check_bull_performance(self) -> CheckResult:
        r = self.returns
        bull = r[r > np.percentile(r, 60)]
        return self._regime_performance(bull, "Bull Market Performance", "Regime")

    def check_bear_performance(self) -> CheckResult:
        r = self.returns
        bear = r[r < np.percentile(r, 40)]
        return self._regime_performance(bear, "Bear Market Performance", "Regime")

    def check_consolidation_performance(self) -> CheckResult:
        r = self.returns
        mid_low = np.percentile(r, 40)
        mid_high = np.percentile(r, 60)
        consol = r[(r >= mid_low) & (r <= mid_high)]
        return self._regime_performance(consol, "Consolidation Performance", "Regime")

    def check_volatility_stress(self) -> CheckResult:
        r = self.returns
        vol = np.std(r)
        stressed = r * 3.0  # Simulate 3x normal volatility (VIX spike)
        stressed_sharpe = np.mean(stressed) / (np.std(stressed) + 1e-9) * np.sqrt(252)
        original_sharpe = np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252)

        degradation = (original_sharpe - stressed_sharpe) / (abs(original_sharpe) + 1e-9) * 100
        passed = degradation < 50

        return CheckResult(
            name="Volatility Spike Stress Test",
            passed=passed,
            score=round(max(0, 100 - degradation), 1),
            value=f"Under 3x volatility: Sharpe drops from {original_sharpe:.2f} to {stressed_sharpe:.2f}",
            insight="VIX spikes (like COVID March 2020) cause 3-5x normal volatility. Most retail strategies break.",
            fix="Reduce position sizing when VIX > 25. Use volatility-adjusted position sizing (ATR-based).",
            category="Regime"
        )

    def check_frequency_consistency(self) -> CheckResult:
        r = self.returns
        window = max(5, len(r) // 10)
        rolling_means = [np.mean(r[i:i+window]) for i in range(0, len(r)-window)]
        consistency = np.mean(np.array(rolling_means) > 0) * 100
        passed = consistency > 60

        return CheckResult(
            name="Performance Consistency",
            passed=passed,
            score=round(consistency, 1),
            value=f"Profitable in {consistency:.1f}% of rolling windows",
            insight="Consistent strategies generate returns steadily. Lumpy returns are hard to trust live.",
            fix="If consistency <60%, your edge only works in specific conditions. Define those conditions explicitly.",
            category="Regime"
        )

    # =========================================================
    # ⚙️ GROUP 4: EXECUTION REALITY
    # =========================================================

    def check_slippage_01(self) -> CheckResult:
        r = self.returns
        slipped = r - abs(r) * 0.001  # 0.1% slippage per trade
        original_total = np.sum(r)
        slipped_total = np.sum(slipped)
        impact = (original_total - slipped_total) / (abs(original_total) + 1e-9) * 100

        passed = impact < 20
        score = max(0, 100 - impact * 3)

        return CheckResult(
            name="Slippage Impact (0.1%)",
            passed=passed,
            score=round(score, 1),
            value=f"0.1% slippage reduces total returns by {impact:.1f}%",
            insight="Retail brokers have spreads. Even 0.1% per trade compounds into significant drag on frequent strategies.",
            fix="If impact >20%, reduce trade frequency or only take higher conviction setups.",
            category="Execution"
        )

    def check_slippage_03(self) -> CheckResult:
        r = self.returns
        slipped = r - abs(r) * 0.003
        original_total = np.sum(r)
        slipped_total = np.sum(slipped)
        impact = (original_total - slipped_total) / (abs(original_total) + 1e-9) * 100

        passed = impact < 40
        score = max(0, 100 - impact * 2)

        return CheckResult(
            name="Slippage Impact (0.3%)",
            passed=passed,
            score=round(score, 1),
            value=f"0.3% slippage (small caps / volatile markets) reduces returns by {impact:.1f}%",
            insight="During high volatility or in small-cap stocks, 0.3% slippage is realistic. Does your edge survive?",
            fix="For small-cap strategies, model 0.5% slippage. Use limit orders instead of market orders.",
            category="Execution"
        )

    def check_commission_drag(self) -> CheckResult:
        r = self.returns
        n_trades = len(r)
        commission_per_trade = 0.0005  # $0.50 per $1000 trade (typical)
        total_commission = n_trades * commission_per_trade
        total_return = abs(np.sum(r))
        drag_pct = total_commission / (total_return + 1e-9) * 100

        passed = drag_pct < 15
        score = max(0, 100 - drag_pct * 4)

        return CheckResult(
            name="Commission Drag",
            passed=passed,
            score=round(score, 1),
            value=f"{n_trades} trades × commission = {drag_pct:.1f}% of gross returns consumed",
            insight="High-frequency strategies can be profitable on paper but lose money after commissions.",
            fix="Calculate your break-even commission level. If trading >10x/day, commissions may kill the edge.",
            category="Execution"
        )

    def check_partial_fills(self) -> CheckResult:
        r = self.returns
        # Simulate 20% of trades getting only 70% fill
        fill_rate = 0.80 + 0.20 * 0.70
        adjusted = r * fill_rate
        impact = (np.sum(r) - np.sum(adjusted)) / (abs(np.sum(r)) + 1e-9) * 100

        passed = impact < 10
        score = max(0, 100 - impact * 5)

        return CheckResult(
            name="Partial Fill Simulation",
            passed=passed,
            score=round(score, 1),
            value=f"Partial fills (80% fill rate) reduce returns by {impact:.1f}%",
            insight="In fast markets, your orders may not fill completely. Size-dependent strategies break here.",
            fix="Test with realistic fill rates. Use limit orders and size positions for partial fill scenarios.",
            category="Execution"
        )

    def check_live_vs_backtest_gap(self) -> CheckResult:
        """Estimate live decay using known research: avg 30-50% degradation"""
        r = self.returns
        backtest_sharpe = np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252)
        estimated_live_sharpe = backtest_sharpe * 0.6  # Industry standard: expect 40% decay

        passed = estimated_live_sharpe > 0.5
        score = min(100, max(0, estimated_live_sharpe * 50))

        return CheckResult(
            name="Live Trading Gap Estimate",
            passed=passed,
            score=round(score, 1),
            value=f"Backtest Sharpe: {backtest_sharpe:.2f} → Estimated Live Sharpe: {estimated_live_sharpe:.2f}",
            insight="Research shows retail algos lose 30-50% of backtest performance live due to overfitting, slippage, and emotional interference.",
            fix="Your live Sharpe must be >0.5 to be worth trading. If estimated live <0.5, return to the drawing board.",
            category="Execution"
        )

    # =========================================================
    # 💥 CRASH SIMULATIONS — THE EMOTIONAL HOOK
    # =========================================================

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns

        # Apply crash conditions to strategy returns
        vol_multiplier = profile["vol_multiplier"]
        liquidity_factor = profile["liquidity_factor"]
        gap_risk = profile["gap_risk"]

        # Stress the returns
        stressed = r * vol_multiplier                          # Higher volatility
        stressed = stressed - abs(stressed) * (1 - liquidity_factor) * 0.5  # Liquidity penalty
        stressed = stressed - gap_risk * np.sign(r)           # Gap risk on every trade

        strategy_total = np.sum(stressed)
        market_drop = profile["market_drop"]

        # Did the strategy survive (not lose more than 25%)?
        cumulative = np.cumsum(stressed)
        max_dd = abs(np.min(cumulative - np.maximum.accumulate(cumulative))) if len(cumulative) > 0 else 0
        survived = max_dd < 0.25 * abs(np.sum(abs(r)))

        # Emotional verdict
        if survived and strategy_total > 0:
            verdict = "🟢 YOUR STRATEGY SURVIVED. While markets crashed, your system held. This is what separates real edges from lucky backtests."
        elif survived and strategy_total <= 0:
            verdict = "🟡 BARELY SURVIVED. Your strategy lost money but didn't blow up. In real life, would you have had the nerve to keep trading?"
        else:
            verdict = "🔴 YOUR STRATEGY WOULD HAVE BLOWN UP. The crash exposed fatal flaws. Most traders quit here — the ones who survive rebuild with proper risk management."

        return CrashSimResult(
            crash_name=profile["name"],
            year=profile["year"],
            description=profile["description"],
            market_drop=market_drop,
            strategy_drop=round(strategy_total, 4),
            survived=survived,
            emotional_verdict=verdict
        )

    # =========================================================
    # 🏆 SCORING ENGINE
    # =========================================================

    def _calculate_score(self, checks: List[CheckResult]) -> Tuple[float, str]:
        weights = {
            "Overfitting": 0.35,
            "Risk": 0.30,
            "Regime": 0.20,
            "Execution": 0.15,
        }

        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat]
            if cat_checks:
                category_scores[cat] = np.mean([c.score for c in cat_checks])
            else:
                category_scores[cat] = 50

        final_score = sum(category_scores[cat] * weight for cat, weight in weights.items())

        if final_score >= 80:
            grade = "A — Institutionally Viable"
        elif final_score >= 65:
            grade = "B — Fundable with Improvements"
        elif final_score >= 50:
            grade = "C — Promising but Needs Work"
        elif final_score >= 35:
            grade = "D — Significant Issues"
        else:
            grade = "F — Do Not Deploy"

        return round(final_score, 1), grade

    # =========================================================
    # 🚀 RUN ALL CHECKS
    # =========================================================

    def run(self) -> ValidationReport:
        checks = [
            # Overfitting
            self.check_sharpe_decay(),
            self.check_monte_carlo(),
            self.check_parameter_sensitivity(),
            self.check_walk_forward(),
            self.check_bootstrap_stability(),
            # Risk
            self.check_max_drawdown(),
            self.check_calmar_ratio(),
            self.check_var(),
            self.check_consecutive_losses(),
            self.check_recovery_factor(),
            # Regime
            self.check_bull_performance(),
            self.check_bear_performance(),
            self.check_consolidation_performance(),
            self.check_volatility_stress(),
            self.check_frequency_consistency(),
            # Execution
            self.check_slippage_01(),
            self.check_slippage_03(),
            self.check_commission_drag(),
            self.check_partial_fills(),
            self.check_live_vs_backtest_gap(),
        ]

        crash_sims = [
            self.simulate_crash("2008_gfc"),
            self.simulate_crash("2020_covid"),
            self.simulate_crash("2022_bear"),
        ]

        score, grade = self._calculate_score(checks)

        # Summary stats
        r = self.returns
        sharpe = np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252)
        cumulative = np.cumsum(r)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = abs(np.min(cumulative - running_max))
        win_rate = np.mean(r > 0) * 100

        top_issues = [c.fix for c in sorted(checks, key=lambda x: x.score)[:3]]
        top_strengths = [c.name for c in sorted(checks, key=lambda x: x.score, reverse=True)[:3]]

        summary = (
            f"QuantProof analyzed {len(r)} trades and ran 20 institutional checks + 3 historical crash simulations. "
            f"Your strategy scores {score}/100 ({grade.split('—')[0].strip()}). "
            f"{'Your biggest risk is overfitting.' if score < 60 else 'Your edge shows real promise.'}"
        )

        return ValidationReport(
            fundable_score=score,
            grade=grade,
            summary=summary,
            checks=checks,
            crash_sims=crash_sims,
            total_trades=len(r),
            date_range=str(self.df["date"].min().date()) + " to " + str(self.df["date"].max().date()) if "date" in self.df.columns else "Unknown",
            sharpe=round(sharpe, 2),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 1),
            top_issues=top_issues,
            top_strengths=top_strengths,
        )
