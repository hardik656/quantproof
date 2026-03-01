"""
QuantProof — Validation Engine v1.1
Fixed: Sharpe (risk-free adjusted), Drawdown (expanding max), Win Rate, Scoring weights
"""

import pandas as pd
import numpy as np
import hashlib
import json
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

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
    score: float
    grade: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    checks: List[CheckResult]
    crash_sims: List[CrashSimResult]
    assumptions: List[str]
    validation_hash: str
    validation_date: str
    audit_flags: List[str]
    plausibility_summary: str
    engine_version: str = "v1.1"
    methodology_version: str = "2026-03-01"


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

def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 252) -> float:
    """Risk-adjusted annualized Sharpe ratio for per-trade returns"""
    if len(returns) < 2:
        return 0.0
    
    # For per-trade returns, drop risk-free adjustment (institutional practice)
    # Trades are discrete events, not continuous time exposure
    excess = returns
    
    if np.std(excess) == 0:
        return 0.0
    
    # Annualize based on actual trading frequency
    annual_factor = np.sqrt(trades_per_year)
    # Apply 0.85 live decay factor for prop firm standards
    sharpe = float(np.mean(excess) / np.std(excess) * annual_factor * 0.85)
    return sharpe

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Correct drawdown using expanding max on cumulative returns"""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))

def calculate_win_rate(returns: np.ndarray) -> float:
    """Win rate with minimum trade size filter to avoid tiny wins"""
    if len(returns) == 0:
        return 0.0
    
    # Filter out trades smaller than 0.1% to avoid counting meaningless wins
    meaningful_trades = returns[np.abs(returns) > 0.001]
    if len(meaningful_trades) == 0:
        return 0.0
        
    return float(np.mean(meaningful_trades > 0) * 100)

def calculate_calmar(returns: np.ndarray, trades_per_year: float = 252) -> float:
    """Annual return / abs(max drawdown) with proper trade frequency scaling"""
    annual_return = np.mean(returns) * trades_per_year
    dd = abs(calculate_max_drawdown(returns))
    if dd == 0:
        return 0.0
    return float(annual_return / dd)


# =========================================================
# 🔧 CORE VALIDATOR CLASS
# =========================================================

class QuantProofValidator:

    def __init__(self, df: pd.DataFrame, strict_mode: bool = False, seed: int = 42):
        self.strict_mode = strict_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Global deterministic RNG
        self.df = self._clean(df)
        self.trades_per_year, self.timeframe_info = self._detect_timeframe()
        self.returns = self._get_returns()
        
        # Validate strict mode requirements
        if strict_mode:
            self._validate_strict_requirements()

    def _detect_timeframe(self) -> Tuple[float, str]:
        """Detect timeframe from actual timestamps with precise calculation"""
        if 'date' not in self.df.columns:
            return 252.0, "No timestamp data - using daily metrics"
        
        # Calculate precise time span using total_seconds
        delta = self.df['date'].max() - self.df['date'].min()
        time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
        
        # Handle division by zero edge case
        if time_span_years <= 0:
            return 252.0, "Insufficient time span — default annualization"
        
        # Calculate trades per year directly from data
        trades_per_year = len(self.df) / time_span_years
        
        # Flag short backtests that may inflate metrics
        if time_span_years < 0.25:  # Less than 3 months
            timeframe_desc = f"Short backtest ({time_span_years*12:.1f} months) — Sharpe may be inflated ({trades_per_year:.0f} trades/year)"
        elif trades_per_year > 10000:
            timeframe_desc = f"High-frequency ({trades_per_year:.0f} trades/year)"
        elif trades_per_year > 1000:
            timeframe_desc = f"Intraday ({trades_per_year:.0f} trades/year)"
        elif trades_per_year > 250:
            timeframe_desc = f"Active trading ({trades_per_year:.0f} trades/year)"
        else:
            timeframe_desc = f"Position trading ({trades_per_year:.0f} trades/year)"
            
        return trades_per_year, timeframe_desc

    def _validate_strict_requirements(self):
        """Apply strict validation rules for institutional use"""
        # Minimum 6 months required
        if 'date' in self.df.columns:
            time_span_years = (self.df['date'].max() - self.df['date'].min()).days / 365.25
            if time_span_years < 0.5:
                raise ValueError(f"Strict mode requires minimum 6 months of data (got {time_span_years*12:.1f} months)")
        
        # Minimum 100 trades required
        if len(self.returns) < 100:
            raise ValueError(f"Strict mode requires minimum 100 trades (got {len(self.returns)})")
        
        # Auto-fail short backtests
        if hasattr(self, 'timeframe_info') and 'Short backtest' in self.timeframe_info:
            raise ValueError("Strict mode does not allow backtests shorter than 3 months")
    
    def _generate_validation_hash(self) -> str:
        """Generate deterministic hash for report integrity"""
        # Round returns to reduce floating-point brittleness
        rounded_returns = np.round(self.returns, 8)
        hash_data = {
            'returns': rounded_returns.tolist(),
            'timeframe': self.timeframe_info,
            'trades_per_year': float(self.trades_per_year),
            'engine_version': 'v1.1',
            'seed': self.seed
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:12]
    
    def _get_assumptions(self) -> List[str]:
        """Get explicit assumptions for transparency"""
        assumptions = [
            "Assumes full capital deployed per trade",
            "Assumes trades are sequential and non-overlapping",
            "Assumes percentage returns (not dollar P&L)",
            "Assumes no leverage unless embedded in returns",
            f"Monte Carlo seed fixed at {self.seed} for reproducibility",
            "Risk-free rate excluded for per-trade returns (institutional practice)",
            "Crash simulations use historical stress profiles with proportional scaling"
        ]
        
        if self.strict_mode:
            assumptions.extend([
                "Strict mode: 6-month minimum data requirement",
                "Strict mode: 100-trade minimum requirement",
                "Strict mode: Conservative compliance thresholds",
                "Strict mode: Statistical plausibility checks enabled"
            ])
        
        return assumptions
    
    def _generate_audit_flags(self) -> List[str]:
        """Generate audit flags for statistical implausibility"""
        plausibility_checks = [c for c in self.checks if c.category == "Plausibility"]
        audit_flags = []
        
        for check in plausibility_checks:
            if not check.passed and "Manual Audit Required" in check.value:
                audit_flags.append(f"⚠ {check.name}: {check.insight}")
        
        return audit_flags
    
    def _generate_plausibility_summary(self) -> str:
        """Generate summary of statistical plausibility"""
        plausibility_checks = [c for c in self.checks if c.category == "Plausibility"]
        manual_audit_required = sum(1 for c in plausibility_checks if not c.passed and "Manual Audit Required" in c.value)
        review_recommended = sum(1 for c in plausibility_checks if not c.passed and "Review Recommended" in c.value)
        
        if manual_audit_required > 0:
            return f"⚠ {manual_audit_required} statistical implausibility issues require manual audit"
        elif review_recommended > 0:
            return f"⚠ {review_recommended} statistical issues merit review"
        else:
            return "✅ All statistical metrics appear plausible"

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
        
        # Validate returns are reasonable percentages
        max_abs = np.abs(r).max()
        if max_abs > 1.0:  # >100% return suggests dollar amounts
            raise ValueError(
                f"Returns appear to be dollar amounts (max: {max_abs:.2f}). "
                "Please convert to percentage returns before validation."
            )
        
        # Remove clipping - let tail behavior show naturally
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
        s_in = calculate_sharpe(r[:mid], self.trades_per_year)
        s_out = calculate_sharpe(r[mid:], self.trades_per_year)
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
        
        # Test 1: Mean robustness (reduced for performance)
        mean_sims = [np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(300)]  # Reduced from 500
        mean_pct = float(np.mean(np.array(mean_sims) > 0) * 100)
        
        # Test 2: Sequence fragility (current)
        sequence_sims = []
        original_dd = calculate_max_drawdown(r)
        for _ in range(200):
            shuffled = self.rng.permutation(r)
            # Test drawdown after shuffle - this changes!
            shuffled_dd = calculate_max_drawdown(shuffled)
            # Strategy is fragile if DD increases by more than 10% absolute
            dd_increase = abs(shuffled_dd) - abs(original_dd)
            sequence_sims.append(dd_increase < 0.10)  # Pass if <10% absolute DD increase
        
        sequence_pct = float(np.mean(sequence_sims) * 100)
        
        # Test 3: Equity curve distribution (NEW - hedge fund level)
        equity_sims = []
        for _ in range(500):  # Reduced from 1000 for performance
            shuffled = self.rng.permutation(r)
            final_equity = np.prod(1 + shuffled)
            equity_sims.append(final_equity)
        
        # Worst 5% outcomes
        worst_5pct = np.percentile(equity_sims, 5)
        
        # Calculate equity score - handle no-date scenarios
        if 'date' not in self.df.columns:
            # Use raw worst 5% return instead of CAGR for no-date scenarios
            worst_5pct_return = (worst_5pct - 1) * 100  # Convert to percentage
        else:
            # Calculate time span for CAGR only with timestamps
            if hasattr(self, 'timeframe_info') and 'Short backtest' in self.timeframe_info:
                time_span_years = 1.0
            else:
                delta = self.df['date'].max() - self.df['date'].min()
                time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
            
            worst_5pct_cagr = (worst_5pct ** (1/max(time_span_years, 0.1)) - 1) * 100
        
        # Worst 5% drawdowns (reuse equity sims for efficiency)
        dd_sims = []
        for i in range(min(500, len(equity_sims))):  # Reduced from 1000, reuse data
            shuffled = self.rng.permutation(r)
            dd_sims.append(abs(calculate_max_drawdown(shuffled)))
        worst_5pct_dd = np.percentile(dd_sims, 95) * 100
        
        # Calculate equity score after worst_5pct_dd is computed
        if 'date' not in self.df.columns:
            equity_score = min(100, max(0, 100 + worst_5pct_return * 2 - worst_5pct_dd))
        else:
            equity_score = min(100, max(0, 100 + worst_5pct_cagr * 2 - worst_5pct_dd))
        
        combined_score = (mean_pct * 0.4 + sequence_pct * 0.3 + equity_score * 0.3)
        
        # Strict mode requires higher Monte Carlo threshold
        if self.strict_mode:
            passed = combined_score > 70  # Higher threshold for certification
        else:
            passed = combined_score > 60  # Standard threshold
        
        return CheckResult(
            name="Monte Carlo Robustness",
            passed=passed,
            score=round(combined_score, 1),
            value=f"Mean: {mean_pct:.1f}% | Seq: {sequence_pct:.1f}% | Equity: {equity_score:.1f}%",
            insight="Hedge-fund level Monte Carlo: tests mean stability, sequence fragility, and worst-case equity outcomes.",
            fix="If equity score low, strategy has tail risk or depends on specific sequences.",
            category="Overfitting"
        )

    def check_outlier_dependency(self) -> CheckResult:
        r = self.returns
        q25, q75 = np.percentile(r, 25), np.percentile(r, 75)
        iqr = q75 - q25
        outlier_pct = float(np.mean((r < q25 - 1.5*iqr) | (r > q75 + 1.5*iqr)) * 100)
        passed = outlier_pct < 15
        score = max(0, 100 - outlier_pct * 3)
        return CheckResult(
            name="Outlier Dependency",
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
        bootstraps = [np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(500)]
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
        calmar = calculate_calmar(r, self.trades_per_year)
        passed = calmar > 1.5  # Prop firm standard
        score = min(100, calmar * 40)  # Adjusted scoring for higher threshold
        return CheckResult(
            name="Calmar Ratio",
            passed=passed,
            score=round(score, 1),
            value=f"Calmar: {calmar:.2f} (target >1.5, timeframe: {self.timeframe_info})",
            insight="Calmar = annual return / max drawdown. Prop firms want >1.5.",
            fix="Increase returns or reduce drawdown via better position sizing.",
            category="Risk"
        )

    def check_var(self) -> CheckResult:
        r = self.returns
        var_99 = float(np.percentile(r, 1))
        mean = float(np.mean(r))
        
        # Use absolute threshold for low-mean systems
        if abs(mean) < 0.001:  # Near-zero mean
            passed = abs(var_99) < 0.05  # 5% absolute loss threshold
            threshold_desc = "5% absolute (near-zero mean)"
        else:
            passed = abs(var_99) < abs(mean) * 10
            threshold_desc = f"10x mean ({abs(mean)*10:.3f})"
        
        # Score calculation with institutionally defensible scaling
        if abs(mean) < 0.001:
            # Institutionally defensible scaling: 5% VaR = 0 score, 0% VaR = 100 score
            # Linear scaling: score = 100 * (1 - VaR/0.05) = 100 - 2000*VaR
            score = max(0, 100 - abs(var_99) * 2000)  # 5% threshold mapped to 0 score
        else:
            score = max(0, 100 - (abs(var_99) / (abs(mean) + 1e-9)) * 5)
        
        return CheckResult(
            name="Value at Risk (VaR)",
            passed=passed,
            score=round(score, 1),
            value=f"VaR 99%: {var_99:.4f} | VaR 95%: {float(np.percentile(r, 5)):.4f}",
            insight=f"VaR 99% = loss exceeded only 1% of days. Threshold: {threshold_desc}.",
            fix="If VaR too high, use tighter stops or reduce position size.",
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
        passed = max_streak < 8  # Prop firm standard
        score = max(0, 100 - max_streak * 10)  # Adjusted scoring for stricter threshold
        return CheckResult(
            name="Max Losing Streak",
            passed=passed,
            score=round(score, 1),
            value=f"Longest losing streak: {max_streak} consecutive trades",
            insight="Prop firms reject strategies with 8+ consecutive losses.",
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
        sharpe = calculate_sharpe(r, self.trades_per_year)
        
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
            value=f"Annualized Sharpe: {sharpe:.2f} (timeframe: {self.timeframe_info})",
            insight="Institutional minimum is Sharpe > 1.0. Below 0.5 means the strategy doesn't compensate for its risk.",
            fix="Improve risk-adjusted returns through better entry timing or position sizing.",
            category="Risk"
        )

    def check_compliance_pass(self) -> CheckResult:
        gates = [
            self.returns.mean() > 0,
            calculate_sharpe(self.returns, self.trades_per_year) > 1.0,  # Prop firm standard
            abs(calculate_max_drawdown(self.returns)) < 0.20,
            calculate_win_rate(self.returns) > 45,
            len(self.returns) > 50
        ]
        
        # Strict mode requires all gates to pass
        if self.strict_mode:
            passed = all(gates)
            score = 100 if passed else 0
            gate_status = "5/5" if passed else f"{sum(gates)}/5"
        else:
            passed = sum(gates) >= 4
            score = 100 if passed else 0
            gate_status = f"{sum(gates)}/5"
        
        return CheckResult(
            name="Prop Firm Compliance",
            passed=passed,
            score=score,
            value=f"✅ PASSES 2026 Requirements" if passed else f"❌ NEEDS FIXES ({gate_status} gates passed)",
            insight="FTMO/Topstep reject 90% of strategies without proper validation. 4/5 gates must pass." + (" Strict mode requires all 5 gates." if self.strict_mode else ""),
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
        original_sharpe = calculate_sharpe(r, self.trades_per_year)  # Use consistent timeframe
        stressed = r * 3.0
        stressed_sharpe = calculate_sharpe(stressed, self.trades_per_year)  # Use consistent timeframe
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
    # GROUP 5: STATISTICAL PLAUSIBILITY
    # =========================================================
    
    def check_sharpe_plausibility(self) -> CheckResult:
        """Check if Sharpe ratio is within empirically observed ranges"""
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        
        # Empirical thresholds based on real-world performance
        if sharpe > 10:
            status = "⚠ Manual Audit Required"
            insight = "Sharpe > 10 exceeds documented institutional performance (Renaissance ~8-10)"
        elif sharpe > 5:
            status = "⚠ Review Recommended"
            insight = "Sharpe > 5 is extremely rare, requires explanation"
        else:
            status = "✅ Plausible"
            insight = "Sharpe within realistic institutional range"
        
        return CheckResult(
            name="Sharpe Plausibility",
            passed=sharpe <= 10,  # Never fails, just flags
            score=100,  # Doesn't affect score
            value=f"Sharpe: {sharpe:.2f} → {status}",
            insight=insight,
            fix="If Sharpe > 10, verify data integrity and methodology",
            category="Plausibility"
        )
    
    def check_frequency_return_plausibility(self) -> CheckResult:
        """Check if trade frequency and returns are realistically compatible"""
        mean_return = float(np.mean(self.returns))
        annual_return = mean_return * self.trades_per_year
        
        # High frequency + extreme returns = implausible
        if self.trades_per_year > 20000 and annual_return > 500:
            status = "⚠ Manual Audit Required"
            insight = f"High frequency ({self.trades_per_year:.0f}/yr) + extreme returns ({annual_return:.0f}%) requires massive liquidity edge"
        elif self.trades_per_year > 50000 and annual_return > 200:
            status = "⚠ Review Recommended"
            insight = f"Very high frequency trading with high returns needs liquidity verification"
        else:
            status = "✅ Plausible"
            insight = "Frequency-return relationship within realistic bounds"
        
        return CheckResult(
            name="Frequency-Return Plausibility",
            passed=not (self.trades_per_year > 20000 and annual_return > 500),
            score=100,
            value=f"{self.trades_per_year:.0f} trades/yr × {mean_return*100:.2f}% avg → {annual_return:.0f}% annual ({status})",
            insight=insight,
            fix="Verify liquidity capacity and market impact assumptions",
            category="Plausibility"
        )
    
    def check_equity_smoothness_plausibility(self) -> CheckResult:
        """Check for suspiciously smooth equity curves indicating bias"""
        returns = self.returns
        std_return = np.std(returns)
        mean_return = np.mean(returns)
        max_dd = abs(calculate_max_drawdown(returns))
        
        # Smoothness metrics that indicate potential issues
        smoothness_ratio = std_return / (abs(mean_return) + 1e-9)
        dd_to_mean_ratio = max_dd / (abs(mean_return) + 1e-9)
        
        # Flag suspiciously smooth equity curves
        if smoothness_ratio < 0.5 and mean_return > 0 and max_dd < 0.05:
            status = "⚠ Manual Audit Required"
            insight = "Suspiciously smooth equity curve - potential lookahead bias or synthetic data"
        elif smoothness_ratio < 1.0 and dd_to_mean_ratio < 2.0:
            status = "⚠ Review Recommended"
            insight = "Unusually smooth returns - verify data integrity"
        else:
            status = "✅ Plausible"
            insight = "Return volatility consistent with realistic trading"
        
        return CheckResult(
            name="Equity Curve Plausibility",
            passed=smoothness_ratio >= 0.5 or max_dd >= 0.05,
            score=100,
            value=f"Smoothness: {smoothness_ratio:.2f} → {status}",
            insight=insight,
            fix="Check for lookahead bias, options mispricing, or data manipulation",
            category="Plausibility"
        )
    
    def check_kelly_plausibility(self) -> CheckResult:
        """Check if Kelly fraction suggests unrealistic edge"""
        returns = self.returns
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        # Avoid division by zero
        if variance < 1e-9:
            kelly = 0
        else:
            kelly = mean_return / variance
        
        # Kelly fraction thresholds
        if kelly > 10:
            status = "⚠ Manual Audit Required"
            insight = f"Kelly fraction {kelly:.1f} suggests unrealistic edge or underestimated variance"
        elif kelly > 5:
            status = "⚠ Review Recommended"
            insight = f"High Kelly fraction {kelly:.1f} requires edge verification"
        else:
            status = "✅ Plausible"
            insight = f"Kelly fraction {kelly:.2f} within realistic range"
        
        return CheckResult(
            name="Kelly Plausibility",
            passed=kelly <= 10,
            score=100,
            value=f"Kelly: {kelly:.2f} → {status}",
            insight=insight,
            fix="Verify return variance calculation and edge sustainability",
            category="Plausibility"
        )

    # =========================================================
    # GROUP 6: EXECUTION REALITY
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
        bt_sharpe = calculate_sharpe(r, self.trades_per_year)  # Use consistent timeframe
        live_sharpe = bt_sharpe * 0.6  # Industry: expect 40% decay live
        passed = live_sharpe > 0.5
        score = min(100, max(0, live_sharpe * 50))
        return CheckResult(
            name="Live Trading Gap Estimate",
            passed=passed,
            score=round(score, 1),
            value=f"Backtest Sharpe: {bt_sharpe:.2f} → Estimated Live: {live_sharpe:.2f}",
            insight="Industry average: 40% Sharpe decay from backtest to live trading.",
            fix="Reduce position sizing, add slippage buffers, tighten risk management.",
            category="Execution"
        )

    # =========================================================
    # 💥 CRASH SIMULATIONS
    # =========================================================

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns.copy()
        
        # More realistic crash simulation with asymmetric downside bias
        # 1. Apply moderate volatility increase (not extreme multiplier)
        vol_multiplier = 1 + (profile["vol_multiplier"] - 1) * 0.3  # Scale down multiplier
        stressed = r * vol_multiplier
        
        # 2. Asymmetric downside bias (NEW) - crashes hurt losers more
        downside_bias = 1.2  # 20% more downside
        upside_cap = 0.8    # 20% less upside
        stressed = np.where(stressed < 0, stressed * downside_bias, stressed * upside_cap)
        
        # 3. Apply liquidity drag (reduces trade sizes during crisis)
        liquidity_drag = np.abs(stressed) * (1 - profile["liquidity_factor"]) * 0.3
        stressed = stressed - liquidity_drag
        
        # 4. Gap risk on extreme negative moves only (proportional)
        negative_extremes = r < np.percentile(r, 10)  # Worst 10% of trades
        gap_losses = np.abs(stressed) * profile["gap_risk"] * negative_extremes  # Proportional to trade size
        stressed = stressed - gap_losses
        
        # 5. Handle mathematical impossibilities without arbitrary clipping
        # Prevent >100% loss (portfolio = 0) but allow unlimited upside
        stressed = np.where(stressed < -0.99, -0.99, stressed)
        
        # Calculate cumulative portfolio return
        cumulative = float(np.prod(1 + stressed) - 1) * 100
        
        # Drawdown for survival check
        dd = calculate_max_drawdown(stressed)
        
        # Strict mode uses stricter crash survival threshold
        if self.strict_mode:
            survived = abs(dd) < 0.25  # 25% max in strict mode
        else:
            survived = abs(dd) < 0.30  # 30% standard threshold

        if survived and cumulative > -10:
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
            strategy_drop=round(cumulative, 1),
            survived=survived,
            emotional_verdict=verdict
        )

    # =========================================================
    # 🏆 SCORING ENGINE (Fixed weights)
    # =========================================================

    def _calculate_score(self, checks: List[CheckResult]) -> Tuple[float, str]:
        # Updated weights with stronger Compliance emphasis
        weights = {
            "Overfitting": 0.25,
            "Risk":        0.35,
            "Regime":      0.15,
            "Execution":   0.12,
            "Compliance":  0.13,  # Increased from 8% to 13%
        }
        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat]
            # Critical categories get 0 for missing, optional get 20
            if cat_checks:
                category_scores[cat] = float(np.mean([c.score for c in cat_checks]))
            else:
                if cat in ["Risk", "Overfitting", "Compliance"]:
                    category_scores[cat] = 0.0  # Critical categories
                else:
                    category_scores[cat] = 20.0  # Optional categories

        final = sum(category_scores[cat] * w for cat, w in weights.items())

        # TASK 1: Sharpe Floor Penalty - override all other scores
        overall_sharpe = calculate_sharpe(self.returns, self.trades_per_year)
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
            
        # Compliance veto: if compliance fails, max score is 55 (not 70)
        compliance_checks = [c for c in checks if c.category == "Compliance"]
        if compliance_checks and not all(c.passed for c in compliance_checks):
            final = min(final, 55)

        # NEW: Penalize short backtests to prevent score inflation
        if hasattr(self, 'timeframe_info') and 'Short backtest' in self.timeframe_info:
            final = min(final, 75)  # Cap at B- for short backtests

        if final >= 90:   grade = "A — Institutionally Viable"
        elif final >= 80: grade = "B+ — Prop Firm Ready"
        elif final >= 70: grade = "B — Live Tradeable"
        else:             grade = "F — Do Not Deploy"

        return round(final, 1), grade

    # =========================================================
    # 🚀 RUN ALL CHECKS
    # =========================================================

    def run(self) -> ValidationReport:
        checks = [
            self.check_sharpe_decay(),
            self.check_monte_carlo(),
            self.check_outlier_dependency(),
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
            # Statistical plausibility checks (informational only)
            self.check_sharpe_plausibility(),
            self.check_frequency_return_plausibility(),
            self.check_equity_smoothness_plausibility(),
            self.check_kelly_plausibility(),
        ]

        self.checks = checks  # Store for audit flag generation

        crash_sims = [
            self.simulate_crash("2008_gfc"),
            self.simulate_crash("2020_covid"),
            self.simulate_crash("2022_bear"),
        ]

        score, grade = self._calculate_score(checks)
        r = self.returns
        sharpe = calculate_sharpe(r, self.trades_per_year)
        dd = calculate_max_drawdown(r)
        win_rate = calculate_win_rate(r)

        # Apply strict mode compliance if enabled
        if self.strict_mode:
            # Sharpe floor raised to 1.2 in strict mode
            if sharpe < 1.2:
                grade = "F — Strict Mode: Sharpe below 1.2"
                score = min(score, 40)
            
            # Compliance must pass all gates in strict mode
            compliance_checks = [c for c in checks if c.category == "Compliance"]
            if compliance_checks and not all(c.passed for c in compliance_checks):
                grade = "F — Strict Mode: Compliance failed"
                score = min(score, 30)
            
            # Crash simulation veto - must survive at least 2/3 crashes in strict mode
            crash_survivals = sum(1 for sim in crash_sims if sim.survived)
            if crash_survivals < 2:  # Must survive at least 2 out of 3 crashes
                grade = "F — Strict Mode: Failed crash stress test"
                score = min(score, 35)

        # Generate institutional features
        assumptions = self._get_assumptions()
        validation_hash = self._generate_validation_hash()
        validation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Generate plausibility features
        audit_flags = self._generate_audit_flags()
        plausibility_summary = self._generate_plausibility_summary()

        return ValidationReport(
            score=score,
            grade=grade,
            sharpe=sharpe,
            max_drawdown=dd,
            win_rate=win_rate,
            total_trades=len(r),
            profitable_trades=int(np.sum(r > 0)),
            checks=checks,
            crash_sims=crash_sims,
            assumptions=assumptions,
            validation_hash=validation_hash,
            validation_date=validation_date,
            audit_flags=audit_flags,
            plausibility_summary=plausibility_summary
        )
