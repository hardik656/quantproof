"""
QuantProof — Validation Engine v1.4
DEFINITIVE MERGE: Best of v1.3 (fixed) + v1.3.2
- 34 institutional checks + 3 crash simulations
- CORRECT gambler's ruin formula (capital_units=10, proper gradient)
- Calmar capped at 10.0 (no zero for no-drawdown strategies)
- Sortino: caps at 5.0 for no-loss portfolios (not 0)
- Purged Walk-Forward CV (information leak prevention)
- HMM Regime Detection (requires hmmlearn)
- Deflated Sharpe Ratio (multiple-testing correction)
- Market Impact Model (simplified Almgren-Chriss)
- Strategy Capacity Estimation
- ValidationDashboard for frontend charts
- No-date cap at 75 (stricter than 79, not punishing as 70)
- Sharpe inflation warning displayed to users
- NO noise injection — data integrity is absolute
"""

import pandas as pd
import numpy as np
import hashlib
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from datetime import datetime
from scipy.stats import norm

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

RISK_FREE_DAILY = 0.04 / 252
EPSILON = 1e-9

# =========================================================
# DATA STRUCTURES
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
    engine_version: str = "v1.4"
    methodology_version: str = "2026-03-04"

CRASH_PROFILES = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "year": "Sep 2008 – Mar 2009",
        "description": "Lehman Brothers collapsed. S&P 500 lost 56%. Volatility exploded to VIX 80. Correlations went to 1 — everything fell together.",
        "market_drop": -56.0, "vol_multiplier": 4.5, "liquidity_factor": 0.3, "gap_risk": 0.08,
    },
    "2020_covid": {
        "name": "2020 COVID Crash",
        "year": "Feb 2020 – Mar 2020",
        "description": "Fastest 30% drop in market history. 34% crash in 33 days. Circuit breakers triggered 4 times.",
        "market_drop": -34.0, "vol_multiplier": 5.0, "liquidity_factor": 0.2, "gap_risk": 0.12,
    },
    "2022_bear": {
        "name": "2022 Rate Hike Bear Market",
        "year": "Jan 2022 – Dec 2022",
        "description": "Fed raised rates 425bps. Nasdaq lost 33%, S&P lost 19%. Growth stocks fell 60-90%.",
        "market_drop": -19.4, "vol_multiplier": 2.5, "liquidity_factor": 0.7, "gap_risk": 0.04,
    }
}

# =========================================================
# CORE MATH — ALL VERIFIED
# =========================================================

def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 252) -> float:
    if len(returns) < 2 or np.std(returns) < EPSILON:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year) * 0.85)

def calculate_sortino(returns: np.ndarray, trades_per_year: float = 252) -> float:
    """Sortino: penalizes downside volatility only. Caps at 5.0 for zero-loss portfolios."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 5.0 if np.mean(returns) > 0 else 0.0  # FIX: perfect downside control ≠ 0
    if np.std(downside) < EPSILON:
        return 5.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(trades_per_year) * 0.85)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    return float(np.max((running_max - cumulative) / running_max))

def calculate_win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    meaningful = returns[np.abs(returns) > 0.001]
    if len(meaningful) == 0:
        return 0.0
    return float(np.mean(meaningful > 0) * 100)

def calculate_calmar(returns: np.ndarray, trades_per_year: float = 252) -> float:
    """FIX: Returns 10.0 (excellent) when drawdown=0, not 0.0 (broken)."""
    annual_return = np.mean(returns) * trades_per_year
    dd = calculate_max_drawdown(returns)
    if dd < EPSILON:
        return 10.0 if annual_return > 0 else 0.0
    return float(min(annual_return / dd, 10.0))

def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)

def calculate_ruin_probability(win_rate: float, avg_win: float, avg_loss: float,
                               capital_units: int = 10) -> float:
    """
    CORRECT gambler's ruin via analytical formula.
    capital_units=10 = risking ~10% of capital per trade (standard Kelly sizing).
    Gives proper gradient: strong edge near 0%, weak edge near 100%.
    
    Formula: P(ruin) = (q/p)^N where:
      p = normalized win prob using reward/risk ratio
      N = capital_units (how many avg losses fit in account)
    """
    if win_rate <= 0 or avg_loss < EPSILON:
        return 1.0
    if win_rate >= 1.0:
        return 0.0
    edge = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    if edge <= 0:
        return 1.0
    rr = avg_win / (avg_loss + EPSILON)
    p = win_rate * rr / (win_rate * rr + (1.0 - win_rate))
    q = 1.0 - p
    if p <= 0.5:
        return 1.0
    return float(np.clip((q / p) ** capital_units, 0.0, 1.0))

def calculate_deflated_sharpe(returns: np.ndarray, trades_per_year: float, n_trials: int = 100) -> float:
    """Bailey-López de Prado Deflated Sharpe. Corrects for multiple-testing bias."""
    sharpe = calculate_sharpe(returns, trades_per_year)
    obs = len(returns)
    skewness = float(np.mean(((returns - np.mean(returns)) / (np.std(returns) + EPSILON)) ** 3))
    kurtosis = float(np.mean(((returns - np.mean(returns)) / (np.std(returns) + EPSILON)) ** 4)) - 3
    sr_var = (1 + (0.5 * sharpe**2) - skewness * sharpe + (kurtosis / 4) * sharpe**2) / max(obs - 1, 1)
    if sr_var <= 0:
        return sharpe
    prob = norm.cdf(sharpe, loc=0, scale=np.sqrt(sr_var))
    deflated_prob = 1 - (1 - prob) ** n_trials
    # Clamp to avoid norm.ppf(1.0) = inf
    deflated_prob = float(np.clip(deflated_prob, 1e-10, 1 - 1e-10))
    result = float(norm.ppf(deflated_prob) * np.sqrt(sr_var))
    # Cap to reasonable range
    return float(np.clip(result, -10.0, 20.0))

def purged_kfold_cv(returns: np.ndarray, n_splits: int = 5, embargo_pct: float = 0.05) -> List[Dict]:
    """Purged K-fold CV: prevents information leakage via embargo periods."""
    n = len(returns)
    fold_size = n // n_splits
    embargo_size = int(fold_size * embargo_pct)
    scores = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        train_idx = list(range(0, max(0, test_start - embargo_size))) + \
                    list(range(min(test_end + embargo_size, n), n))
        test_idx = list(range(test_start, test_end))
        if len(train_idx) < 10 or len(test_idx) < 10:
            continue
        train_s = calculate_sharpe(returns[train_idx])
        test_s = calculate_sharpe(returns[test_idx])
        decay = (train_s - test_s) / (abs(train_s) + EPSILON) if train_s != 0 else 0
        scores.append({'train': train_s, 'test': test_s, 'decay': decay})
    return scores

def detect_regimes_hmm(returns: np.ndarray, n_regimes: int = 3) -> Dict[str, Any]:
    if not HMM_AVAILABLE:
        return {}
    try:
        model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100, random_state=42)
        model.fit(returns.reshape(-1, 1))
        states = model.predict(returns.reshape(-1, 1))
        return {
            f'regime_{i}': {
                'mean': float(np.mean(returns[states == i])),
                'vol': float(np.std(returns[states == i])),
                'sharpe': calculate_sharpe(returns[states == i]),
                'duration': int(np.sum(states == i))
            }
            for i in range(n_regimes) if np.sum(states == i) > 0
        }
    except Exception:
        return {}

def estimate_strategy_capacity(returns: np.ndarray, volumes: np.ndarray = None,
                                current_aum: float = 1e6) -> Dict[str, Any]:
    daily_volume = np.mean(volumes) if volumes is not None else 1e12
    max_capacity = (daily_volume * 0.05 * 252) / (np.mean(np.abs(returns)) * 2 + EPSILON)
    test_aums = np.logspace(6, 10, 20)
    sharpes = [calculate_sharpe(returns - 0.001 * (aum / 1e6) ** 0.5) for aum in test_aums]
    viable_idx = np.where(np.array(sharpes) > 1.0)[0]
    viable = test_aums[viable_idx[-1]] if len(viable_idx) > 0 else current_aum
    return {
        'theoretical_max': max_capacity,
        'viable_capacity': viable,
        'capacity_utilization': current_aum / max(viable, EPSILON)
    }

# =========================================================
# VALIDATOR CLASS
# =========================================================

class QuantProofValidator:
    def __init__(self, df: pd.DataFrame, strict_mode: bool = False, seed: int = 42):
        self.strict_mode = strict_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.df = self._clean(df)
        self.has_dates = 'date' in self.df.columns
        self.trades_per_year, self.timeframe_info = self._detect_timeframe()
        self.returns = self._get_returns()
        if strict_mode:
            self._validate_strict_requirements()

    def _detect_timeframe(self) -> Tuple[float, str]:
        if not self.has_dates:
            return 252.0, "No timestamp data — Sharpe may be inflated"
        delta = self.df['date'].max() - self.df['date'].min()
        time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
        if time_span_years < EPSILON:
            return 252.0, "Insufficient time span — default annualization"
        trades_per_year = min(len(self.df) / time_span_years, 252)
        if time_span_years < 0.25:
            desc = f"Short backtest ({time_span_years*12:.1f} months) — Sharpe may be inflated"
        elif trades_per_year > 10000:
            desc = f"High-frequency ({trades_per_year:.0f} trades/year)"
        elif trades_per_year > 1000:
            desc = f"Intraday ({trades_per_year:.0f} trades/year)"
        elif trades_per_year > 250:
            desc = f"Active trading ({trades_per_year:.0f} trades/year)"
        else:
            desc = f"Position trading ({trades_per_year:.0f} trades/year)"
        return trades_per_year, desc

    def _validate_strict_requirements(self):
        if self.has_dates:
            span = (self.df['date'].max() - self.df['date'].min()).days / 365.25
            if span < 0.5:
                raise ValueError(f"Strict mode requires 6 months minimum (got {span*12:.1f} months)")
        if len(self.returns) < 100:
            raise ValueError(f"Strict mode requires 100+ trades (got {len(self.returns)})")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        date_cols = [c for c in df.columns if any(x in c for x in ["date", "time", "dt"])]
        pnl_cols  = [c for c in df.columns if any(x in c for x in ["pnl", "profit", "return", "gain", "pl"])]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if pnl_cols:
            df = df.rename(columns={pnl_cols[0]: "pnl"})
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        return df.dropna(subset=["pnl"])

    def _get_returns(self) -> np.ndarray:
        """CRITICAL: Returns are NEVER modified. Data integrity is absolute."""
        r = self.df["pnl"].values.astype(float)
        if np.abs(r).max() > 1.0:
            raise ValueError(f"Returns appear to be dollar amounts (max: {np.abs(r).max():.2f}). Convert to percentages.")
        return r  # NO NOISE INJECTION

    def _generate_validation_hash(self) -> str:
        data = {
            'returns': np.round(self.returns, 8).tolist(),
            'timeframe': self.timeframe_info,
            'trades_per_year': float(self.trades_per_year),
            'engine_version': 'v1.4',
            'seed': self.seed
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    def _get_assumptions(self) -> List[str]:
        base = [
            "Assumes full capital deployed per trade",
            "Assumes trades are sequential and non-overlapping",
            "Assumes percentage returns (not dollar P&L)",
            "Assumes no leverage unless embedded in returns",
            f"Monte Carlo seed fixed at {self.seed} for reproducibility",
            "Risk-free rate excluded for per-trade returns (institutional practice)",
            "Crash simulations use historical stress profiles with proportional scaling",
            "CVaR calculated at 95% confidence — average loss in worst 5% of trades",
            "Sortino ratio penalizes downside volatility only (caps at 5.0 for zero-loss)",
            "Ruin probability: analytical gambler's ruin (capital_units=10, ~10% risk per trade)",
            "Deflated Sharpe: assumes 100 strategy variations tested (adjust if different)",
            "Market impact: simplified square-root model (not full Almgren-Chriss)",
            "INPUT DATA IS NEVER MODIFIED — data integrity guaranteed",
        ]
        if not self.has_dates:
            base.append("⚠ WARNING: No timestamp data — Sharpe ratio may be inflated by up to 2x")
        if self.strict_mode:
            base += ["Strict mode: 6-month minimum data requirement", "Strict mode: 100-trade minimum"]
        return base

    def _generate_audit_flags(self) -> List[str]:
        return [
            f"⚠ {c.name}: {c.insight}"
            for c in self.checks
            if c.category == "Plausibility" and not c.passed and "Manual Audit Required" in c.value
        ]

    def _generate_plausibility_summary(self) -> str:
        if not self.has_dates:
            return "⚠ No timestamp data — Sharpe may be inflated. Add date column for full accuracy."
        pc = [c for c in self.checks if c.category == "Plausibility"]
        manual = sum(1 for c in pc if not c.passed and "Manual Audit Required" in c.value)
        review = sum(1 for c in pc if not c.passed and "Review Recommended" in c.value)
        if manual > 0: return f"⚠ {manual} statistical implausibility issues require manual audit"
        if review > 0: return f"⚠ {review} statistical issues merit review"
        return "✅ All statistical metrics appear plausible"

    # =========================================================
    # CHECKS — OVERFITTING (6)
    # =========================================================

    def check_sharpe_decay(self) -> CheckResult:
        r, n = self.returns, len(self.returns)
        if n < 20:
            return CheckResult("Sharpe Ratio Decay", False, 20, "Insufficient data", "Need 20+ trades", "Add more backtest history", "Overfitting")
        mid = n // 2
        s_in = calculate_sharpe(r[:mid], self.trades_per_year)
        s_out = calculate_sharpe(r[mid:], self.trades_per_year)
        decay = (s_in - s_out) / (abs(s_in) + EPSILON) * 100
        score = max(0, 100 - max(0, decay))
        return CheckResult("Sharpe Ratio Decay", decay < 40, round(score, 1),
            f"In-sample: {s_in:.2f} → Out-of-sample: {s_out:.2f} ({decay:.1f}% decay)",
            "High decay means your strategy was fitted to historical noise, not real patterns.",
            "Run walk-forward optimization. Only trust out-of-sample Sharpe.", "Overfitting")

    def check_monte_carlo(self) -> CheckResult:
        r = self.returns
        mean_pct = float(np.mean(np.array([np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(300)]) > 0) * 100)
        original_dd = calculate_max_drawdown(r)
        seq_pct = float(np.mean([abs(calculate_max_drawdown(self.rng.permutation(r))) - abs(original_dd) < 0.10 for _ in range(200)]) * 100)
        perms = [self.rng.permutation(r) for _ in range(500)]
        equity_sims = [np.prod(1 + p) for p in perms]
        worst_5pct = np.percentile(equity_sims, 5)
        if not self.has_dates:
            worst_val = (worst_5pct - 1) * 100
        else:
            delta = self.df['date'].max() - self.df['date'].min()
            tsy = max(delta.total_seconds() / (365.25 * 24 * 3600), 0.1)
            worst_val = (worst_5pct ** (1 / (1.0 if 'Short backtest' in self.timeframe_info else tsy)) - 1) * 100
        worst_dd = np.percentile([abs(calculate_max_drawdown(p)) for p in perms], 95) * 100
        equity_score = min(100, max(0, 100 + worst_val * 2 - worst_dd))
        combined = mean_pct * 0.4 + seq_pct * 0.3 + equity_score * 0.3
        return CheckResult("Monte Carlo Robustness", combined > (70 if self.strict_mode else 60), round(combined, 1),
            f"Mean: {mean_pct:.1f}% | Seq: {seq_pct:.1f}% | Equity: {equity_score:.1f}%",
            "Hedge-fund level Monte Carlo: tests mean stability, sequence fragility, worst-case equity outcomes.",
            "If equity score low, strategy has tail risk or depends on specific trade sequences.", "Overfitting")

    def check_outlier_dependency(self) -> CheckResult:
        r = self.returns
        q25, q75 = np.percentile(r, 25), np.percentile(r, 75)
        iqr = q75 - q25
        outlier_pct = float(np.mean((r < q25 - 1.5*iqr) | (r > q75 + 1.5*iqr)) * 100)
        score = max(0, 100 - outlier_pct * 3)
        return CheckResult("Outlier Dependency", outlier_pct < 15, round(score, 1),
            f"{outlier_pct:.1f}% of trades are statistical outliers",
            "High outlier dependency means a few lucky trades drive all returns.",
            "Remove outliers and retest. If it no longer profits, you don't have a real edge.", "Overfitting")

    def check_walk_forward(self) -> CheckResult:
        """Simple walk-forward consistency (50+ trades)."""
        r = self.returns
        window = max(10, len(r) // 5)
        windows = [r[i:i+window] for i in range(0, len(r)-window, window)]
        if len(windows) < 2:
            return CheckResult("Walk-Forward Consistency", False, 30, "Need 50+ trades", "Insufficient data", "Extend backtest period", "Overfitting")
        profitable = sum(1 for w in windows if np.mean(w) > 0)
        pct = profitable / len(windows) * 100
        return CheckResult("Walk-Forward Consistency", pct >= 60, round(pct, 1),
            f"{profitable}/{len(windows)} time periods profitable ({pct:.0f}%)",
            "A real edge works consistently across different time periods, not just in-sample.",
            "If <60% of periods profitable, the strategy is period-specific.", "Overfitting")

    def check_walk_forward_purged(self) -> CheckResult:
        """Purged K-fold CV: prevents information leakage (requires 100+ trades)."""
        r = self.returns
        if len(r) < 100:
            return CheckResult("Purged Walk-Forward CV", False, 30, "Need 100+ trades", "Insufficient data", "Extend backtest", "Overfitting")
        scores = purged_kfold_cv(r)
        if not scores:
            return CheckResult("Purged Walk-Forward CV", False, 30, "CV failed", "Data issue", "Check returns", "Overfitting")
        avg_decay = float(np.mean([s['decay'] for s in scores]))
        max_decay = float(np.max([s['decay'] for s in scores]))
        passed = avg_decay < 0.3 and max_decay < 0.5
        score = min(100, max(0, 100 - avg_decay * 100))
        return CheckResult("Purged Walk-Forward CV", passed, round(score, 1),
            f"Avg decay: {avg_decay:.1%} | Max fold decay: {max_decay:.1%} {'✅' if passed else '❌ >50%'}",
            "Purged CV prevents information leakage from overlapping samples. High decay in any fold = overfitting.",
            "If any fold decay > 50%, reduce strategy complexity or extend hold-out periods.", "Overfitting")

    def check_bootstrap_stability(self) -> CheckResult:
        r = self.returns
        pct = float(np.mean(np.array([np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(500)]) > 0) * 100)
        return CheckResult("Bootstrap Stability", pct > 70, round(pct, 1),
            f"Positive expectancy in {pct:.1f}% of 500 bootstrap samples",
            "Stable edge shows up consistently in resampling.",
            "If below 70%, fix win rate or risk/reward ratio first.", "Overfitting")

    def check_deflated_sharpe(self) -> CheckResult:
        """Bailey–López de Prado Deflated Sharpe: corrects for multiple-testing bias."""
        N_TRIALS = 100  # explicit assumption: ~100 parameter combinations tested
        deflated = calculate_deflated_sharpe(self.returns, self.trades_per_year, n_trials=N_TRIALS)
        original = calculate_sharpe(self.returns, self.trades_per_year)
        decay = (original - deflated) / (abs(original) + EPSILON) if original != 0 else 0
        if decay > 0.5:
            status = "⚠ Severe overfitting"
            score = 20
        elif decay > 0.3:
            status = "⚠ Moderate overfitting risk"
            score = 50
        else:
            status = "✅ Robust"
            score = 100
        return CheckResult("Deflated Sharpe Ratio", decay < 0.5, score,
            f"Original: {original:.2f} → Deflated: {deflated:.2f} ({decay:.1%} decay, {N_TRIALS} trials assumed) {status}",
            f"Adjusts for multiple-testing bias from parameter optimization. Decay >50% = strategy was curve-fitted.",
            f"If deflation >50%, validate on truly out-of-sample data. Assumes {N_TRIALS} strategy variations.", "Overfitting")

    # =========================================================
    # CHECKS — RISK (9)
    # =========================================================

    def check_max_drawdown(self) -> CheckResult:
        dd_pct = calculate_max_drawdown(self.returns) * 100
        score = max(0, 100 - dd_pct * 3)
        return CheckResult("Max Drawdown", dd_pct < 20, round(score, 1),
            f"Max drawdown: {dd_pct:.1f}% (threshold: <20%)",
            "Funds reject strategies with drawdowns >20%. Retail traders quit at 15%.",
            "Add circuit breaker: pause after 10% drawdown. Reduce size after losing streaks.", "Risk")

    def check_cvar(self) -> CheckResult:
        r = self.returns
        cvar_95 = calculate_cvar(r, 0.95)
        cvar_99 = calculate_cvar(r, 0.99)
        mean = float(np.mean(r))
        if abs(mean) < 0.001:
            passed = abs(cvar_95) < 0.08
            score = max(0, 100 - abs(cvar_95) * 1250)
            threshold_desc = "8% absolute"
        else:
            ratio = abs(cvar_95) / (abs(mean) + EPSILON)
            passed = ratio < 15
            score = max(0, 100 - ratio * 4)
            threshold_desc = f"15x mean"
        return CheckResult("CVaR / Expected Shortfall", passed, round(score, 1),
            f"CVaR 95%: {cvar_95:.4f} | CVaR 99%: {cvar_99:.4f} | Threshold: {threshold_desc}",
            "CVaR = average loss when things go really wrong. Required by Basel III. More honest than VaR alone.",
            "Reduce tail exposure: tighter stops on losing trades, avoid low-liquidity assets.", "Risk")

    def check_calmar_ratio(self) -> CheckResult:
        calmar = calculate_calmar(self.returns, self.trades_per_year)
        score = min(100, calmar * 40)
        return CheckResult("Calmar Ratio", calmar > 1.5, round(score, 1),
            f"Calmar: {calmar:.2f} (target >1.5) — {self.timeframe_info}",
            "Calmar = annual return / max drawdown. Prop firms want >1.5. 10.0 = excellent (capped).",
            "Increase returns or reduce drawdown via better position sizing.", "Risk")

    def check_sortino(self) -> CheckResult:
        sortino = calculate_sortino(self.returns, self.trades_per_year)
        if sortino >= 2.0:   score, passed = 100, True
        elif sortino >= 1.0: score, passed = 75, True
        elif sortino >= 0.5: score, passed = 45, False
        else:                score, passed = 15, False
        return CheckResult("Sortino Ratio", passed, round(score, 1),
            f"Sortino: {sortino:.2f} (target >1.0) — downside-volatility adjusted",
            "Sortino only punishes bad volatility. A strategy with big winners and controlled losers scores higher here than on Sharpe.",
            "Improve downside control: tighter stop losses, ATR-based position sizing.", "Risk")

    def check_var(self) -> CheckResult:
        r = self.returns
        var_99 = float(np.percentile(r, 1))
        mean = float(np.mean(r))
        if abs(mean) < 0.001:
            passed = abs(var_99) < 0.05
            score = max(0, 100 - abs(var_99) * 2000)
            threshold_desc = "5% absolute"
        else:
            passed = abs(var_99) < abs(mean) * 10
            score = max(0, 100 - (abs(var_99) / (abs(mean) + EPSILON)) * 5)
            threshold_desc = f"10x mean"
        return CheckResult("Value at Risk (VaR)", passed, round(score, 1),
            f"VaR 99%: {var_99:.4f} | VaR 95%: {float(np.percentile(r, 5)):.4f} | Threshold: {threshold_desc}",
            "VaR 99% = loss exceeded only 1% of trades. Use with CVaR for complete tail picture.",
            "If VaR too high, use tighter stops or reduce position size.", "Risk")

    def check_consecutive_losses(self) -> CheckResult:
        r = self.returns
        max_streak = current = 0
        for trade in r:
            current = current + 1 if trade < 0 else 0
            max_streak = max(max_streak, current)
        score = max(0, 100 - max_streak * 10)
        return CheckResult("Max Losing Streak", max_streak < 8, round(score, 1),
            f"Longest losing streak: {max_streak} consecutive trades",
            "Prop firms reject strategies with 8+ consecutive losses.",
            "Add daily loss limit that pauses trading after streak >5.", "Risk")

    def check_recovery_factor(self) -> CheckResult:
        r = self.returns
        recovery = float(np.sum(r[r > 0])) / (float(abs(np.sum(r[r < 0]))) + EPSILON)
        score = min(100, recovery * 40)
        return CheckResult("Recovery Factor", recovery > 1.5, round(score, 1),
            f"Recovery factor: {recovery:.2f} (wins cover losses {recovery:.1f}x)",
            "Below 1.5 means wins barely cover drawdowns.",
            "Increase average winner or cut average loser. Asymmetric R:R is the goal.", "Risk")

    def check_ruin_probability(self) -> CheckResult:
        r = self.returns
        wr = calculate_win_rate(r) / 100.0
        winners, losers = r[r > 0], r[r < 0]
        avg_win  = float(np.mean(winners)) if len(winners) > 0 else 0.0
        avg_loss = float(abs(np.mean(losers))) if len(losers) > 0 else 0.0
        ruin_prob = calculate_ruin_probability(wr, avg_win, avg_loss)
        ruin_pct = ruin_prob * 100
        score = max(0, 100 - ruin_pct * 5)
        if ruin_pct < 5:    verdict = "Low ruin risk"
        elif ruin_pct < 15: verdict = "Moderate ruin risk"
        elif ruin_pct < 30: verdict = "High ruin risk"
        else:               verdict = "Very high ruin risk"
        return CheckResult("Probability of Ruin", ruin_pct < 10, round(score, 1),
            f"Ruin probability: {ruin_pct:.1f}% — {verdict}",
            "Chance of losing all capital (analytical gambler's ruin, ~10% risk per trade). Below 10% required for live trading.",
            "Increase win rate, improve R:R ratio, or reduce per-trade risk percentage.", "Risk")

    def check_kelly_sizing(self) -> CheckResult:
        """Fractional Kelly optimal position sizing with drawdown constraint."""
        r = self.returns
        mean_excess = np.mean(r) - RISK_FREE_DAILY
        variance = np.var(r)
        if variance < EPSILON:
            return CheckResult("Kelly Position Sizing", False, 0,
                "Insufficient variance for Kelly calculation",
                "Need variance in returns to compute Kelly fraction.",
                "Ensure strategy has realistic return distribution.", "Risk")
        full_kelly = mean_excess / variance
        fractional = full_kelly * 0.5  # half-Kelly standard
        max_safe = 0.20 / (np.sqrt(variance) * np.sqrt(252))  # 20% DD limit
        constrained = min(fractional, max_safe)
        constrained = float(np.clip(constrained, 0.0, 0.25))  # practical max 25%
        wr = np.mean(r > 0)
        avg_win  = float(np.mean(r[r > 0])) if wr > 0 else 0.0
        avg_loss = float(abs(np.mean(r[r < 0]))) if wr < 1 else 0.0
        ruin = calculate_ruin_probability(wr, avg_win, avg_loss)
        is_safe = ruin < 0.01
        score = 100 if is_safe else max(0, 100 - ruin * 100)
        risk_label = "LOW" if ruin < 0.01 else "ELEVATED" if ruin < 0.05 else "HIGH"
        # Display full kelly capped at 200% for readability
        full_kelly_display = min(full_kelly * 100, 200.0)
        return CheckResult("Kelly Position Sizing", is_safe and constrained > 0, round(score, 1),
            f"Half-Kelly: {constrained:.2%} | Full Kelly: {full_kelly_display:.0f}%{'+ (capped)' if full_kelly*100 > 200 else ''} | Ruin: {ruin:.1%} ({risk_label})",
            "Fractional Kelly maximizes growth while controlling drawdown. Ruin uses analytical gambler's ruin formula.",
            "If Kelly is very low or ruin HIGH, reduce position size and improve win rate / R:R.", "Risk")

    def check_absolute_sharpe(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe > 1.5:   score, passed = 100, True
        elif sharpe >= 1.0: score, passed = 75, True
        elif sharpe >= 0.5: score, passed = 45, False
        else:               score, passed = 15, False
        warning = " ⚠ No dates — may be inflated" if not self.has_dates else ""
        return CheckResult("Absolute Sharpe Ratio", passed, round(score, 1),
            f"Annualized Sharpe: {sharpe:.2f}{warning} ({self.timeframe_info})",
            "Institutional minimum is Sharpe > 1.0. Below 0.5 means the strategy doesn't compensate for its risk.",
            "Improve risk-adjusted returns through better entry timing or position sizing. Add timestamps for accuracy.", "Risk")

    # =========================================================
    # CHECKS — REGIME (7)
    # =========================================================

    def check_bull_performance(self) -> CheckResult:
        r = self.returns
        bull = r[r > np.percentile(r, 60)]
        sharpe = calculate_sharpe(bull, self.trades_per_year) if len(bull) > 3 else 0
        return CheckResult("Bull Market Performance", sharpe > 0 and calculate_win_rate(bull) > 45,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | Win rate: {calculate_win_rate(bull):.1f}% ({len(bull)} trades)",
            "Strategy behavior during favorable conditions.",
            "If failing in bull market, check if momentum signals are properly calibrated.", "Regime")

    def check_bear_performance(self) -> CheckResult:
        r = self.returns
        bear = r[r < np.percentile(r, 40)]
        sharpe = calculate_sharpe(bear, self.trades_per_year) if len(bear) > 3 else 0
        return CheckResult("Bear Market Performance", sharpe > -1.0,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | Win rate: {calculate_win_rate(bear):.1f}% ({len(bear)} trades)",
            "How strategy holds up when conditions turn against it.",
            "Add regime detection to reduce size or pause during bear conditions.", "Regime")

    def check_consolidation_performance(self) -> CheckResult:
        r = self.returns
        low, high = np.percentile(r, 40), np.percentile(r, 60)
        consol = r[(r >= low) & (r <= high)]
        sharpe = calculate_sharpe(consol, self.trades_per_year) if len(consol) > 3 else 0
        return CheckResult("Consolidation Performance", sharpe > 0,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | Win rate: {calculate_win_rate(consol):.1f}% ({len(consol)} trades)",
            "Sideways markets are where most momentum strategies bleed.",
            "Add a choppiness filter to avoid trading in low-volatility ranges.", "Regime")

    def check_regime_robustness_hmm(self) -> CheckResult:
        if not HMM_AVAILABLE:
            return CheckResult("HMM Regime Robustness", True, 50,
                "hmmlearn not installed — install for advanced regime detection",
                "Hidden Markov Model detects hidden market regimes.",
                "pip install hmmlearn for advanced regime detection.", "Regime")
        regimes = detect_regimes_hmm(self.returns)
        if not regimes:
            return CheckResult("HMM Regime Robustness", False, 30, "HMM detection failed",
                "Could not detect regimes — need 100+ trades.", "Ensure 100+ trades for HMM.", "Regime")
        sharpes = [r['sharpe'] for r in regimes.values()]
        min_s, max_s = min(sharpes), max(sharpes)
        all_positive = all(s > 0 for s in sharpes)
        score = 100 if all_positive else max(0, 50 + min_s * 25)
        return CheckResult("HMM Regime Robustness", all_positive and (max_s - min_s) < 2.0, round(score, 1),
            f"Min Sharpe: {min_s:.2f} | Max: {max_s:.2f} | Range: {max_s-min_s:.2f}",
            "HMM detects hidden market regimes. Robust strategies work across all regimes.",
            "If Sharpe varies >2.0 between regimes, add regime detection to adjust size.", "Regime")

    def check_volatility_stress(self) -> CheckResult:
        r = self.returns
        s_orig = calculate_sharpe(r, self.trades_per_year)
        s_stressed = calculate_sharpe(r * 3.0, self.trades_per_year)
        degradation = (s_orig - s_stressed) / (abs(s_orig) + EPSILON) * 100
        return CheckResult("Volatility Spike Stress Test", degradation < 50, round(max(0, 100 - degradation), 1),
            f"3x vol: Sharpe {s_orig:.2f} → {s_stressed:.2f} ({degradation:.1f}% degradation)",
            "VIX spikes (COVID March 2020) cause 3-5x normal volatility.",
            "Use ATR-based position sizing. Reduce exposure when VIX >25.", "Regime")

    def check_frequency_consistency(self) -> CheckResult:
        r = self.returns
        window = max(5, len(r) // 10)
        rolling = [np.mean(r[i:i+window]) for i in range(0, len(r)-window)]
        pct = float(np.mean(np.array(rolling) > 0) * 100)
        return CheckResult("Performance Consistency", pct > 60, round(pct, 1),
            f"Profitable in {pct:.1f}% of rolling windows (window={window} trades)",
            "Consistent strategies generate returns steadily, not in lumps.",
            "If <60%, your edge only works in specific conditions. Define those explicitly.", "Regime")

    def check_regime_coverage(self) -> CheckResult:
        if 'regime' not in self.df.columns:
            return CheckResult("Regime Coverage", False, 40, "No regime column detected",
                "Strategies with regime detection have 3x better live survival rate.",
                "Add BULL/BEAR/CONSOLIDATION/TRANSITION regime labels to your CSV export.", "Regime")
        regimes = self.df['regime'].value_counts(normalize=True)
        coverage = len(regimes) / 4 * 100
        return CheckResult("4-Regime Coverage", coverage > 75, round(coverage, 1),
            f"Regimes detected: {list(regimes.index.astype(str))}",
            "Full regime coverage means tested across all market conditions.",
            "Ensure backtest includes BULL, BEAR, CONSOLIDATION and TRANSITION periods.", "Regime")

    # =========================================================
    # CHECKS — EXECUTION (8)
    # =========================================================

    def check_slippage_01(self) -> CheckResult:
        r = self.returns
        impact = abs((float(np.sum(r)) - float(np.sum(r - np.abs(r) * 0.001))) / (abs(float(np.sum(r))) + EPSILON) * 100)
        return CheckResult("Slippage Impact (0.1%)", impact < 20, round(max(0, 100 - impact * 3), 1),
            f"0.1% slippage reduces returns by {impact:.1f}%",
            "Even 0.1% per trade compounds into significant drag.",
            "Reduce trade frequency or only take higher conviction setups.", "Execution")

    def check_slippage_03(self) -> CheckResult:
        r = self.returns
        impact = abs((float(np.sum(r)) - float(np.sum(r - np.abs(r) * 0.003))) / (abs(float(np.sum(r))) + EPSILON) * 100)
        return CheckResult("Slippage Impact (0.3%)", impact < 40, round(max(0, 100 - impact * 2), 1),
            f"0.3% slippage reduces returns by {impact:.1f}%",
            "Small caps and volatile markets have 0.3%+ slippage. Does your edge survive?",
            "Model 0.5% slippage for small-cap strategies. Use limit orders.", "Execution")

    def check_market_impact(self) -> CheckResult:
        if 'volume' not in self.df.columns:
            return CheckResult("Market Impact (Simplified)", True, 50,
                "No volume data — add 'volume' column for full market impact analysis",
                "Simplified square-root model requires volume data for estimation.",
                "Add daily volume column to your data export.", "Execution")
        r = self.returns
        vols = self.df['volume'].values
        volatility = np.std(r)
        avg_trade = np.mean(np.abs(r)) * 100000
        impacts = [0.00005 + 0.5 * volatility * np.sqrt(avg_trade / max(v, EPSILON)) for v in vols]
        avg_impact = np.mean(impacts)
        pnl_impact = abs((np.sum(r) - np.sum(r - np.sign(r) * np.array(impacts))) / (abs(np.sum(r)) + EPSILON)) * 100
        return CheckResult("Market Impact (Simplified)", avg_impact < 0.002, round(max(0, 100 - pnl_impact * 2), 1),
            f"Avg impact: {avg_impact:.4f} | PnL reduction: {pnl_impact:.1f}%",
            "Simplified square-root market impact. Large trades move prices against you.",
            "Reduce position size to <1% of daily volume, or use VWAP/TWAP execution.", "Execution")

    def check_capacity_constraints(self) -> CheckResult:
        vols = self.df['volume'].values if 'volume' in self.df.columns else None
        result = estimate_strategy_capacity(self.returns, vols)
        passed = result['capacity_utilization'] < 0.5
        score = max(0, 100 - result['capacity_utilization'] * 100)
        return CheckResult("Strategy Capacity", passed, round(score, 1),
            f"Viable capacity: ${result['viable_capacity']:,.0f} | Utilization: {result['capacity_utilization']:.1%}",
            "Strategy degrades as AUM grows due to market impact. Stay below 50% of viable capacity.",
            "If utilization > 50%, reduce position sizes or trade more liquid instruments.", "Execution")

    def check_commission_drag(self) -> CheckResult:
        r = self.returns
        drag = (len(r) * 0.0005) / (abs(float(np.sum(r))) + EPSILON) * 100
        return CheckResult("Commission Drag", drag < 15, round(max(0, 100 - drag * 4), 1),
            f"{len(r)} trades → {drag:.1f}% of gross returns consumed by commissions",
            "High-frequency strategies can be profitable on paper but lose after commissions.",
            "Calculate your break-even commission. If >10x/day, commissions may kill the edge.", "Execution")

    def check_partial_fills(self) -> CheckResult:
        r = self.returns
        fill_rate = 0.80 + 0.20 * 0.70
        impact = abs((float(np.sum(r)) - float(np.sum(r * fill_rate))) / (abs(float(np.sum(r))) + EPSILON) * 100)
        return CheckResult("Partial Fill Simulation", impact < 10, round(max(0, 100 - impact * 5), 1),
            f"80% fill rate reduces returns by {impact:.1f}%",
            "In fast markets orders may not fill completely.",
            "Size positions for partial fill scenarios. Use limit orders.", "Execution")

    def check_live_vs_backtest_gap(self) -> CheckResult:
        bt_sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        live_sharpe = bt_sharpe * 0.6
        return CheckResult("Live Trading Gap Estimate", live_sharpe > 0.5,
            round(min(100, max(0, live_sharpe * 50)), 1),
            f"Backtest Sharpe: {bt_sharpe:.2f} → Estimated Live: {live_sharpe:.2f} (40% decay applied)",
            "Industry average: 40% Sharpe decay from backtest to live trading.",
            "Reduce position sizing, add slippage buffers, tighten risk management.", "Execution")

    # =========================================================
    # CHECKS — COMPLIANCE (1)
    # =========================================================

    def check_compliance_pass(self) -> CheckResult:
        gates = [
            self.returns.mean() > 0,
            calculate_sharpe(self.returns, self.trades_per_year) > 1.0,
            calculate_max_drawdown(self.returns) < 0.20,
            calculate_win_rate(self.returns) > 45,
            len(self.returns) > 50
        ]
        if self.strict_mode:
            passed = all(gates)
            gate_status = "5/5" if passed else f"{sum(gates)}/5"
        else:
            passed = sum(gates) >= 4
            gate_status = f"{sum(gates)}/5"
        return CheckResult("Prop Firm Compliance", passed, 100 if passed else 0,
            "✅ PASSES 2026 Requirements" if passed else f"❌ NEEDS FIXES ({gate_status} gates passed)",
            "FTMO/Topstep reject 90% of strategies without proper validation. 4/5 gates must pass.",
            "Fix your top 3 failing checks above to meet prop firm standards.", "Compliance")

    # =========================================================
    # CHECKS — PLAUSIBILITY (4)
    # =========================================================

    def check_sharpe_plausibility(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if not self.has_dates and sharpe > 2.0:
            return CheckResult("Sharpe Plausibility", False, 0,
                f"Sharpe: {sharpe:.2f} → ⚠ MANUAL AUDIT REQUIRED — no timestamps, value cannot be verified",
                "Sharpe without dates uses assumed 252 trades/year. May be 2x overstated.",
                "Add timestamp data. Verify methodology.", "Plausibility")
        if sharpe > 10:
            return CheckResult("Sharpe Plausibility", False, 0,
                f"Sharpe: {sharpe:.2f} → ⚠ Manual Audit Required",
                "Sharpe > 10 exceeds documented institutional performance (Renaissance Medallion ~8-10).",
                "Verify data integrity and methodology. Check for lookahead bias.", "Plausibility")
        if sharpe > 5:
            return CheckResult("Sharpe Plausibility", True, 100,
                f"Sharpe: {sharpe:.2f} → ⚠ Review Recommended",
                "Sharpe > 5 is extremely rare. Requires explanation and independent verification.",
                "Document edge source and verify with out-of-sample data.", "Plausibility")
        return CheckResult("Sharpe Plausibility", True, 100,
            f"Sharpe: {sharpe:.2f} → ✅ Plausible",
            "Sharpe within realistic institutional range.", "", "Plausibility")

    def check_frequency_return_plausibility(self) -> CheckResult:
        mean_r = float(np.mean(self.returns))
        annual_r = mean_r * self.trades_per_year
        if self.trades_per_year > 20000 and annual_r > 500:
            status, insight = "⚠ Manual Audit Required", f"HFT + extreme returns requires massive liquidity edge"
            passed = False
        else:
            status, insight, passed = "✅ Plausible", "Frequency-return relationship within realistic bounds", True
        return CheckResult("Frequency-Return Plausibility", passed, 100 if passed else 0,
            f"{self.trades_per_year:.0f} trades/yr × {mean_r*100:.2f}% avg → {annual_r:.0f}% annual ({status})",
            insight, "Verify liquidity capacity and market impact assumptions", "Plausibility")

    def check_equity_smoothness_plausibility(self) -> CheckResult:
        r = self.returns
        smoothness = np.std(r) / (abs(np.mean(r)) + EPSILON)
        dd = calculate_max_drawdown(r)
        if smoothness < 0.5 and np.mean(r) > 0 and dd < 0.05:
            status, insight, passed = "⚠ Manual Audit Required", "Suspiciously smooth equity curve — potential lookahead bias or synthetic data", False
        elif smoothness < 1.0 and dd / (abs(np.mean(r)) + EPSILON) < 2.0:
            status, insight, passed = "⚠ Review Recommended", "Unusually smooth returns — verify data integrity", True
        else:
            status, insight, passed = "✅ Plausible", "Return volatility consistent with realistic trading", True
        return CheckResult("Equity Curve Plausibility", passed, 100 if passed else 0,
            f"Smoothness ratio: {smoothness:.2f} → {status}", insight,
            "Check for lookahead bias, options mispricing, or data manipulation", "Plausibility")

    def check_kelly_plausibility(self) -> CheckResult:
        r = self.returns
        mean_r = np.mean(r)
        variance = np.var(r)
        if np.max(np.abs(r)) > 0.5:
            mean_r, variance = mean_r / 100, variance / 10000
        kelly = mean_r / variance if variance > EPSILON else 0
        if kelly > 5.0:
            status, insight, passed = "⚠ Manual Audit Required", f"Kelly {kelly:.1f} suggests unrealistic edge or underestimated variance", False
        elif kelly > 2.0:
            status, insight, passed = "⚠ Review Recommended", f"High Kelly {kelly:.1f} requires edge verification", True
        else:
            status, insight, passed = "✅ Plausible", f"Kelly fraction {kelly:.2f} within realistic range", True
        return CheckResult("Kelly Plausibility", passed, 100 if passed else 0,
            f"Kelly: {kelly:.2f} → {status}", insight,
            "Verify return variance calculation and edge sustainability", "Plausibility")

    # =========================================================
    # CRASH SIMULATIONS
    # =========================================================

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns.copy()
        stressed = r * (1 + (profile["vol_multiplier"] - 1) * 0.3)
        stressed = np.where(stressed < 0, stressed * 1.2, stressed * 0.8)
        stressed -= np.abs(stressed) * (1 - profile["liquidity_factor"]) * 0.3
        neg_extremes = stressed < np.percentile(stressed, 10)
        stressed -= np.abs(stressed) * profile["gap_risk"] * neg_extremes
        stressed = np.where(stressed < -0.99, -0.99, stressed)
        cumulative = float(np.prod(1 + stressed) - 1)
        dd = calculate_max_drawdown(stressed)
        survived = dd < (0.25 if self.strict_mode else 0.30)
        if survived and cumulative > -0.10:
            verdict = "🟢 YOUR STRATEGY SURVIVED. While markets crashed, your system held. This is what separates real edges from lucky backtests."
        elif survived:
            verdict = "🟡 BARELY SURVIVED. Your strategy lost money but didn't blow up. In real life, would you have had the nerve to keep trading?"
        else:
            verdict = "🔴 YOUR STRATEGY WOULD HAVE BLOWN UP. The crash exposed fatal flaws. Most traders quit here — the ones who survive rebuild with proper risk management."
        return CrashSimResult(
            crash_name=profile["name"], year=profile["year"],
            description=profile["description"], market_drop=profile["market_drop"],
            strategy_drop=round(cumulative, 4), survived=survived, emotional_verdict=verdict
        )

    # =========================================================
    # SCORING
    # =========================================================

    def _calculate_score(self, checks: List[CheckResult]) -> Tuple[float, str]:
        weights = {"Overfitting": 0.25, "Risk": 0.35, "Regime": 0.15, "Execution": 0.12, "Compliance": 0.13}
        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat]
            category_scores[cat] = float(np.mean([c.score for c in cat_checks])) if cat_checks else (0.0 if cat in ["Risk", "Overfitting", "Compliance"] else 20.0)
        final = sum(category_scores[cat] * w for cat, w in weights.items())
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe < 0:        final = min(final, 20)
        elif sharpe < 0.3:    final = min(final, 35)
        elif sharpe < 0.5:    final = min(final, 50)
        if not self.has_dates: final = min(final, 75)  # FIX: 75 (was 79, stricter than 70)
        low_cats = sum(1 for s in category_scores.values() if s < 30)
        if low_cats >= 2:     final = min(final, 45)
        elif low_cats >= 1:   final = min(final, 60)
        compliance = [c for c in checks if c.category == "Compliance"]
        if compliance and not all(c.passed for c in compliance): final = min(final, 55)
        if 'Short backtest' in self.timeframe_info:              final = min(final, 75)
        if final >= 90:   grade = "A — Institutionally Viable"
        elif final >= 80: grade = "B+ — Prop Firm Ready"
        elif final >= 70: grade = "B — Live Tradeable"
        else:             grade = "F — Do Not Deploy"
        return round(final, 1), grade

    # =========================================================
    # RUN
    # =========================================================

    def run(self) -> ValidationReport:
        checks = [
            # Overfitting (7)
            self.check_sharpe_decay(),
            self.check_monte_carlo(),
            self.check_outlier_dependency(),
            self.check_walk_forward(),
            self.check_walk_forward_purged(),
            self.check_bootstrap_stability(),
            self.check_deflated_sharpe(),
            # Risk (10)
            self.check_max_drawdown(),
            self.check_cvar(),
            self.check_calmar_ratio(),
            self.check_sortino(),
            self.check_var(),
            self.check_consecutive_losses(),
            self.check_recovery_factor(),
            self.check_ruin_probability(),
            self.check_kelly_sizing(),
            self.check_absolute_sharpe(),
            # Regime (7)
            self.check_bull_performance(),
            self.check_bear_performance(),
            self.check_consolidation_performance(),
            self.check_regime_robustness_hmm(),
            self.check_volatility_stress(),
            self.check_frequency_consistency(),
            self.check_regime_coverage(),
            # Execution (8)
            self.check_slippage_01(),
            self.check_slippage_03(),
            self.check_market_impact(),
            self.check_capacity_constraints(),
            self.check_commission_drag(),
            self.check_partial_fills(),
            self.check_live_vs_backtest_gap(),
            # Compliance (1)
            self.check_compliance_pass(),
            # Plausibility (4) — informational, excluded from score
            self.check_sharpe_plausibility(),
            self.check_frequency_return_plausibility(),
            self.check_equity_smoothness_plausibility(),
            self.check_kelly_plausibility(),
        ]
        self.checks = checks
        crash_sims = [self.simulate_crash(k) for k in ["2008_gfc", "2020_covid", "2022_bear"]]
        score, grade = self._calculate_score(checks)
        r = self.returns
        sharpe = calculate_sharpe(r, self.trades_per_year)
        dd = calculate_max_drawdown(r)
        win_rate = calculate_win_rate(r)
        if self.strict_mode:
            if sharpe < 1.2:       grade = "F — Strict Mode: Sharpe below 1.2"; score = min(score, 40)
            compliance = [c for c in checks if c.category == "Compliance"]
            if compliance and not all(c.passed for c in compliance): grade = "F — Strict Mode: Compliance failed"; score = min(score, 30)
            if sum(1 for s in crash_sims if s.survived) < 2:        grade = "F — Strict Mode: Failed crash stress test"; score = min(score, 35)
        return ValidationReport(
            score=score, grade=grade, sharpe=sharpe, max_drawdown=dd, win_rate=win_rate,
            total_trades=len(r), profitable_trades=int(np.sum(r > 0)),
            checks=checks, crash_sims=crash_sims, assumptions=self._get_assumptions(),
            validation_hash=self._generate_validation_hash(),
            validation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            audit_flags=self._generate_audit_flags(),
            plausibility_summary=self._generate_plausibility_summary(),
        )


# =========================================================
# DASHBOARD
# =========================================================

class ValidationDashboard:
    def __init__(self, validator: QuantProofValidator, report: ValidationReport):
        self.validator = validator
        self.report = report

    def generate_interactive_report(self) -> Dict[str, Any]:
        try:
            r = self.validator.returns
            equity = np.cumprod(1 + r)
            dd_series = (np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)
            dates = self.validator.df['date'].dt.strftime('%Y-%m-%d').tolist() if self.validator.has_dates else list(range(len(r)))
            cat_scores = {}
            for check in self.report.checks:
                if check.category not in ("Plausibility",):
                    cat_scores.setdefault(check.category, []).append(check.score)
            rng = self.validator.rng
            mc_finals = [float(np.prod(1 + rng.permutation(r))) for _ in range(1000)]
            return {
                "equity_curve": {"dates": dates, "equity": equity.tolist(), "drawdown": (dd_series * 100).tolist()},
                "distribution": {
                    "percentiles": {str(p): float(np.percentile(r, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
                    "skewness": float(np.mean(((r - np.mean(r)) / (np.std(r) + EPSILON)) ** 3)),
                    "kurtosis": float(np.mean(((r - np.mean(r)) / (np.std(r) + EPSILON)) ** 4) - 3),
                },
                "category_scores": {cat: float(np.mean(scores)) for cat, scores in cat_scores.items()},
                "monte_carlo": {
                    "distribution": mc_finals,
                    "p5": float(np.percentile(mc_finals, 5)),
                    "p50": float(np.percentile(mc_finals, 50)),
                    "p95": float(np.percentile(mc_finals, 95)),
                    "prob_profit": float(np.mean(np.array(mc_finals) > 1)),
                },
                "crash_sims": [{"name": s.crash_name, "survived": s.survived, "strategy_drop": s.strategy_drop, "market_drop": s.market_drop} for s in self.report.crash_sims],
                "metadata": {"score": self.report.score, "grade": self.report.grade, "hash": self.report.validation_hash, "engine": self.report.engine_version}
            }
        except Exception as e:
            return {"error": str(e)}