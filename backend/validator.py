"""
QuantProof — Validation Engine v1.3.1
Critical Fixes: Removed noise injection, honest ruin proxy labeling, 
transparent deflated Sharpe assumptions, simplified market impact naming
"""

import pandas as pd
import numpy as np
import hashlib
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from datetime import datetime
from scipy import stats
from scipy.stats import norm

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

RISK_FREE_DAILY = 0.04 / 252
EPSILON = 1e-9

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
    engine_version: str = "v1.3.1"
    methodology_version: str = "2026-03-01"

CRASH_PROFILES = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "year": "Sep 2008 – Mar 2009",
        "description": "Lehman Brothers collapsed. S&P 500 lost 56%. Volatility exploded to VIX 80.",
        "market_drop": -56.0,
        "vol_multiplier": 4.5,
        "liquidity_factor": 0.3,
        "gap_risk": 0.08,
    },
    "2020_covid": {
        "name": "2020 COVID Crash",
        "year": "Feb 2020 – Mar 2020",
        "description": "Fastest 30% drop in market history. 34% crash in 33 days.",
        "market_drop": -34.0,
        "vol_multiplier": 5.0,
        "liquidity_factor": 0.2,
        "gap_risk": 0.12,
    },
    "2022_bear": {
        "name": "2022 Rate Hike Bear Market",
        "year": "Jan 2022 – Dec 2022",
        "description": "Fed raised rates 425bps. Nasdaq lost 33%, S&P lost 19%.",
        "market_drop": -19.4,
        "vol_multiplier": 2.5,
        "liquidity_factor": 0.7,
        "gap_risk": 0.04,
    }
}

# =========================================================
# CORE MATH HELPERS
# =========================================================

def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns
    if np.std(excess) < EPSILON:
        return 0.0
    annual_factor = np.sqrt(trades_per_year)
    return float(np.mean(excess) / np.std(excess) * annual_factor * 0.85)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    return float(np.max(drawdown))

def calculate_win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    meaningful_trades = returns[np.abs(returns) > 0.001]
    if len(meaningful_trades) == 0:
        return 0.0
    return float(np.mean(meaningful_trades > 0) * 100)

def calculate_calmar(returns: np.ndarray, trades_per_year: float = 252) -> float:
    annual_return = np.mean(returns) * trades_per_year
    dd = calculate_max_drawdown(returns)
    if dd < EPSILON:
        return 0.0
    return float(annual_return / dd)

def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return var_threshold
    return float(np.mean(tail_losses))

def calculate_fractional_kelly(returns: np.ndarray, fraction: float = 0.5,
                              max_dd_limit: float = 0.20,
                              rf_rate: float = RISK_FREE_DAILY) -> Dict[str, Any]:
    mean_excess = np.mean(returns) - rf_rate
    variance = np.var(returns)
    
    if variance < EPSILON:
        return {'full_kelly': 0, 'fractional_kelly': 0, 'constrained_kelly': 0,
                'recommended_position': 0, 'ruin_risk_proxy': 1.0, 'risk_level': 'EXTREME'}
    
    full_kelly = mean_excess / variance
    fractional = full_kelly * fraction
    max_safe_kelly = max_dd_limit / (np.sqrt(variance) * np.sqrt(252))
    constrained = min(fractional, max_safe_kelly)
    
    win_rate = np.mean(returns > 0)
    avg_win = np.mean(returns[returns > 0]) if win_rate > 0 else 0
    avg_loss = abs(np.mean(returns[returns < 0])) if win_rate < 1 else 0
    
    # Heuristic ruin risk proxy (NOT formal gambler's ruin)
    if win_rate <= 0.5 or avg_loss == 0:
        ruin_proxy = 1.0
        risk_level = 'EXTREME'
    else:
        z = win_rate * avg_win - (1 - win_rate) * avg_loss
        a = np.sqrt(z**2 + avg_win * avg_loss)
        if a < EPSILON:
            ruin_proxy = 1.0
            risk_level = 'EXTREME'
        else:
            p = ((1 + z/a) / 2)
            if constrained * np.std(returns) < EPSILON:
                ruin_proxy = 0.0 if p >= 1 else 1.0
            else:
                p = p ** (1 / (constrained * np.std(returns)))
                ruin_proxy = max(0, 1 - p) if p < 1 else 1.0
            
            risk_level = 'LOW' if ruin_proxy < 0.01 else 'ELEVATED' if ruin_proxy < 0.05 else 'HIGH'
    
    return {
        'full_kelly': full_kelly,
        'fractional_kelly': fractional,
        'constrained_kelly': constrained,
        'recommended_position': constrained * 100,
        'ruin_risk_proxy': ruin_proxy,  # HEURISTIC: Not formal gambler's ruin
        'risk_level': risk_level,
        'is_safe': risk_level == 'LOW'
    }

def calculate_market_impact_simple(trade_size_dollars: float, daily_volume: float,
                                   volatility: float, spread: float = 0.0001) -> float:
    """Simplified square-root impact estimate (NOT full Almgren-Chriss)"""
    temporary = spread / 2
    permanent = 0.5 * volatility * np.sqrt(trade_size_dollars / max(daily_volume, EPSILON))
    return temporary + permanent

def calculate_deflated_sharpe(returns: np.ndarray, trades_per_year: float,
                             n_trials: int) -> float:
    sharpe = calculate_sharpe(returns, trades_per_year)
    obs = len(returns)
    
    skewness = pd.Series(returns).skew()
    kurtosis = pd.Series(returns).kurtosis()
    
    sr_var = (1 + (0.5 * sharpe**2) - skewness * sharpe + 
              (kurtosis / 4) * sharpe**2) / max(obs - 1, 1)
    
    if sr_var <= 0:
        return sharpe
    
    prob = norm.cdf(sharpe, loc=0, scale=np.sqrt(sr_var))
    deflated_prob = 1 - (1 - prob) ** n_trials
    deflated_sharpe = norm.ppf(deflated_prob) * np.sqrt(sr_var)
    
    return deflated_sharpe

def purged_kfold_cv(returns: np.ndarray, n_splits: int = 5,
                   embargo_pct: float = 0.05) -> List[Dict]:
    n = len(returns)
    fold_size = n // n_splits
    embargo_size = int(fold_size * embargo_pct)
    
    scores = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        
        train_indices = list(range(0, max(0, test_start - embargo_size))) + \
                       list(range(min(test_end + embargo_size, n), n))
        test_indices = list(range(test_start, test_end))
        
        if len(train_indices) < 10 or len(test_indices) < 10:
            continue
            
        train_sharpe = calculate_sharpe(returns[train_indices])
        test_sharpe = calculate_sharpe(returns[test_indices])
        
        decay = (train_sharpe - test_sharpe) / (abs(train_sharpe) + EPSILON) if train_sharpe != 0 else 0
        
        scores.append({'train': train_sharpe, 'test': test_sharpe, 'decay': decay})
    
    return scores

def detect_regimes_hmm(returns: np.ndarray, n_regimes: int = 3) -> Dict[str, Any]:
    if not HMM_AVAILABLE:
        return {}
    
    try:
        X = returns.reshape(-1, 1)
        model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                           n_iter=100, random_state=42)
        model.fit(X)
        hidden_states = model.predict(X)
        
        regimes = {}
        for i in range(n_regimes):
            mask = hidden_states == i
            regime_returns = returns[mask]
            if len(regime_returns) > 0:
                regimes[f'regime_{i}'] = {
                    'mean': np.mean(regime_returns),
                    'vol': np.std(regime_returns),
                    'sharpe': calculate_sharpe(regime_returns),
                    'duration': int(np.sum(mask))
                }
        return regimes
    except Exception:
        return {}

def estimate_strategy_capacity(returns: np.ndarray, volumes: np.ndarray = None,
                            current_aum: float = 1e6,
                            max_participation: float = 0.05) -> Dict[str, Any]:
    if volumes is None:
        daily_volume = 1e12
    else:
        daily_volume = np.mean(volumes)
    
    avg_trade_size = np.mean(np.abs(returns)) * current_aum
    max_capacity = (daily_volume * max_participation * 252) / (np.mean(np.abs(returns)) * 2 + EPSILON)
    
    test_aums = np.logspace(6, 10, 20)
    sharpes = []
    
    for aum in test_aums:
        impact = 0.001 * (aum / 1e6) ** 0.5
        adj_returns = returns - impact
        sharpes.append(calculate_sharpe(adj_returns))
    
    viable_idx = np.where(np.array(sharpes) > 1.0)[0]
    viable_capacity = test_aums[viable_idx[-1]] if len(viable_idx) > 0 else current_aum
    
    return {
        'theoretical_max': max_capacity,
        'viable_capacity': viable_capacity,
        'sharpe_at_viable': calculate_sharpe(returns - 0.001 * (viable_capacity / 1e6) ** 0.5),
        'capacity_utilization': current_aum / max(viable_capacity, EPSILON)
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
        self.trades_per_year, self.timeframe_info = self._detect_timeframe()
        self.returns = self._get_returns()
        
        if strict_mode:
            self._validate_strict_requirements()

    def _detect_timeframe(self) -> Tuple[float, str]:
        if 'date' not in self.df.columns:
            return 252.0, "No timestamp data - using daily metrics"
        
        delta = self.df['date'].max() - self.df['date'].min()
        time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
        
        if time_span_years < EPSILON:
            return 252.0, "Insufficient time span — default annualization"
        
        trades_per_year = min(len(self.df) / time_span_years, 252)
        
        if time_span_years < 0.25:
            timeframe_desc = f"Short backtest ({time_span_years*12:.1f} months) — Sharpe may be inflated"
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
        if 'date' in self.df.columns:
            time_span_years = (self.df['date'].max() - self.df['date'].min()).days / 365.25
            if time_span_years < 0.5:
                raise ValueError(f"Strict mode requires minimum 6 months of data")
        
        if len(self.returns) < 100:
            raise ValueError(f"Strict mode requires minimum 100 trades")
        
        if 'Short backtest' in self.timeframe_info:
            raise ValueError("Strict mode does not allow backtests shorter than 3 months")

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
        """CRITICAL: Returns are NEVER modified. Plausibility checks flag issues."""
        r = self.df["pnl"].values.astype(float)
        max_abs = np.abs(r).max()
        
        if max_abs > 1.0:
            raise ValueError(f"Returns appear to be dollar amounts (max: {max_abs:.2f}). Convert to percentages.")
        
        # NO NOISE INJECTION - data integrity is absolute
        return r

    def _generate_validation_hash(self) -> str:
        rounded_returns = np.round(self.returns, 8)
        hash_data = {
            'returns': rounded_returns.tolist(),
            'timeframe': self.timeframe_info,
            'trades_per_year': float(self.trades_per_year),
            'engine_version': 'v1.3.1',
            'seed': self.seed
        }
        return hashlib.sha256(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()[:12]

    def _get_assumptions(self) -> List[str]:
        assumptions = [
            "Assumes full capital deployed per trade",
            "Assumes trades are sequential and non-overlapping",
            "Assumes percentage returns (not dollar P&L)",
            "Assumes no leverage unless embedded in returns",
            f"Monte Carlo seed fixed at {self.seed} for reproducibility",
            "Risk-free rate excluded for per-trade returns (institutional practice)",
            "Crash simulations use historical stress profiles with proportional scaling",
            "Market impact: simplified square-root estimate (not full Almgren-Chriss)",
            "Kelly ruin risk: heuristic proxy (not formal gambler's ruin model)",
            "Deflated Sharpe: assumes 100 strategy variations (adjust if different)",
            "INPUT DATA IS NEVER MODIFIED - integrity guaranteed"
        ]
        
        if self.strict_mode:
            assumptions.extend([
                "Strict mode: 6-month minimum data requirement",
                "Strict mode: 100-trade minimum requirement",
                "Strict mode: Conservative compliance thresholds"
            ])
        
        return assumptions

    def _generate_audit_flags(self) -> List[str]:
        plausibility_checks = [c for c in self.checks if c.category == "Plausibility"]
        audit_flags = []
        for check in plausibility_checks:
            if not check.passed and "Manual Audit Required" in check.value:
                audit_flags.append(f"⚠ {check.name}: {check.insight}")
        return audit_flags

    def _generate_plausibility_summary(self) -> str:
        plausibility_checks = [c for c in self.checks if c.category == "Plausibility"]
        manual_audit = sum(1 for c in plausibility_checks if not c.passed and "Manual Audit Required" in c.value)
        review_rec = sum(1 for c in plausibility_checks if not c.passed and "Review Recommended" in c.value)
        
        if manual_audit > 0:
            return f"⚠ {manual_audit} statistical implausibility issues require manual audit"
        elif review_rec > 0:
            return f"⚠ {review_rec} statistical issues merit review"
        return "✅ All statistical metrics appear plausible"

    # =========================================================
    # CHECKS
    # =========================================================

    def check_sharpe_decay(self) -> CheckResult:
        r = self.returns
        n = len(r)
        if n < 20:
            return CheckResult("Sharpe Decay", False, 20, "Insufficient data", "Need 20+ trades", "Add more backtest history", "Overfitting")
        
        mid = n // 2
        s_in = calculate_sharpe(r[:mid], self.trades_per_year)
        s_out = calculate_sharpe(r[mid:], self.trades_per_year)
        decay = (s_in - s_out) / (abs(s_in) + EPSILON) * 100
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
        mean_sims = [np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(300)]
        mean_pct = float(np.mean(np.array(mean_sims) > 0) * 100)
        
        sequence_sims = []
        original_dd = calculate_max_drawdown(r)
        for _ in range(200):
            shuffled = self.rng.permutation(r)
            shuffled_dd = calculate_max_drawdown(shuffled)
            dd_increase = abs(shuffled_dd) - abs(original_dd)
            sequence_sims.append(dd_increase < 0.10)
        
        sequence_pct = float(np.mean(sequence_sims) * 100)
        
        n_sims = 500
        shared_permutations = [self.rng.permutation(r) for _ in range(n_sims)]
        equity_sims = [np.prod(1 + perm) for perm in shared_permutations]
        worst_5pct = np.percentile(equity_sims, 5)
        
        if 'date' not in self.df.columns:
            worst_5pct_return = (worst_5pct - 1) * 100
        else:
            if 'Short backtest' in self.timeframe_info:
                time_span_years = 1.0
            else:
                delta = self.df['date'].max() - self.df['date'].min()
                time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
            worst_5pct_return = (worst_5pct ** (1/max(time_span_years, 0.1)) - 1) * 100
        
        dd_sims = [abs(calculate_max_drawdown(perm)) for perm in shared_permutations]
        worst_5pct_dd = np.percentile(dd_sims, 95) * 100
        
        equity_score = min(100, max(0, 100 + worst_5pct_return * 2 - worst_5pct_dd))
        combined_score = (mean_pct * 0.4 + sequence_pct * 0.3 + equity_score * 0.3)
        passed = combined_score > (70 if self.strict_mode else 60)
        
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

    def check_walk_forward_purged(self) -> CheckResult:
        r = self.returns
        if len(r) < 100:
            return CheckResult("Walk-Forward (Purged)", False, 30, "Need 100+ trades", "Insufficient data", "Extend backtest", "Overfitting")
        
        scores = purged_kfold_cv(r, n_splits=5, embargo_pct=0.05)
        
        if not scores:
            return CheckResult("Walk-Forward (Purged)", False, 30, "CV failed", "Data issue", "Check returns", "Overfitting")
        
        decays = [s['decay'] for s in scores]
        avg_decay = np.mean(decays)
        max_decay = np.max(decays)
        
        passed = avg_decay < 0.3 and max_decay < 0.5
        score = max(0, 100 - avg_decay * 100)
        
        return CheckResult(
            name="Purged Walk-Forward CV",
            passed=passed,
            score=round(score, 1),
            value=f"Avg decay: {avg_decay:.1%} | Max decay: {max_decay:.1%}",
            insight="Purged CV prevents information leakage from overlapping samples. High decay = overfitting.",
            fix="If decay > 50% in any fold, reduce strategy complexity or increase hold-out periods.",
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

    def check_deflated_sharpe(self) -> CheckResult:
        n_trials_assumed = 100  # EXPLICIT ASSUMPTION
        deflated = calculate_deflated_sharpe(self.returns, self.trades_per_year, n_trials=n_trials_assumed)
        original = calculate_sharpe(self.returns, self.trades_per_year)
        
        decay = (original - deflated) / (original + EPSILON) if original != 0 else 0
        
        if decay > 0.5:
            status = "⚠ Severe overfitting likely"
            insight = f"Sharpe deflated by {decay:.1%} (assuming {n_trials_assumed} trials) - strategy may not survive out-of-sample"
        elif decay > 0.3:
            status = "⚠ Moderate overfitting risk"
            insight = f"Multiple testing bias detected (assumed {n_trials_assumed} trials) - verify with longer backtest"
        else:
            status = "✅ Robust"
            insight = "Deflated Sharpe close to original - edge appears genuine"
        
        return CheckResult(
            name="Deflated Sharpe Ratio",
            passed=decay < 0.5,
            score=100 if decay < 0.3 else 50,
            value=f"Original: {original:.2f} → Deflated: {deflated:.2f} ({decay:.1%} decay, {n_trials_assumed} trials) {status}",
            insight=insight,
            fix=f"If deflation > 50%, test on truly out-of-sample data or adjust n_trials if you tested more/fewer than {n_trials_assumed} variations",
            category="Plausibility"
        )

    def check_max_drawdown(self) -> CheckResult:
        r = self.returns
        dd = calculate_max_drawdown(r)
        dd_pct = dd * 100
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

    def check_cvar(self) -> CheckResult:
        r = self.returns
        cvar_95 = calculate_cvar(r, 0.95)
        cvar_99 = calculate_cvar(r, 0.99)
        var_95 = np.percentile(r, 5)
        
        tail_ratio = abs(cvar_95) / (abs(var_95) + EPSILON)
        
        passed = tail_ratio < 2.5
        score = max(0, 100 - (tail_ratio - 1) * 30)
        
        return CheckResult(
            name="Conditional VaR (Tail Risk)",
            passed=passed,
            score=round(score, 1),
            value=f"CVaR 95%: {cvar_95:.4f} | CVaR 99%: {cvar_99:.4f} | Tail ratio: {tail_ratio:.2f}",
            insight="CVaR measures average loss in worst cases. High tail ratio indicates extreme tail risk.",
            fix="If CVaR >> VaR, add tail hedges or reduce position size in volatile regimes.",
            category="Risk"
        )

    def check_calmar_ratio(self) -> CheckResult:
        r = self.returns
        calmar = calculate_calmar(r, self.trades_per_year)
        passed = calmar > 1.5
        score = min(100, calmar * 40)
        
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
        
        if abs(mean) < 0.001:
            passed = abs(var_99) < 0.05
            threshold_desc = "5% absolute (near-zero mean)"
        else:
            passed = abs(var_99) < abs(mean) * 10
            threshold_desc = f"10x mean ({abs(mean)*10:.3f})"
        
        if abs(mean) < 0.001:
            score = max(0, 100 - abs(var_99) * 2000)
        else:
            score = max(0, 100 - (abs(var_99) / (abs(mean) + EPSILON)) * 5)
        
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
        
        passed = max_streak < 8
        score = max(0, 100 - max_streak * 10)
        
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
        recovery = total_profit / (total_loss + EPSILON)
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

    def check_kelly_sizing(self) -> CheckResult:
        result = calculate_fractional_kelly(self.returns)
        
        passed = result['is_safe'] and result['constrained_kelly'] > 0
        score = 100 if passed else max(0, 100 - result['ruin_risk_proxy'] * 100)
        
        return CheckResult(
            name="Optimal Position Sizing",
            passed=passed,
            score=round(score, 1),
            value=f"Kelly: {result['constrained_kelly']:.2%} | Ruin risk proxy: {result['ruin_risk_proxy']:.2%} ({result['risk_level']})",
            insight="Fractional Kelly with drawdown constraints. Ruin risk is HEURISTIC (not formal gambler's ruin).",
            fix="If risk level not LOW, reduce position size or add stop losses.",
            category="Risk"
        )

    def check_absolute_sharpe(self) -> CheckResult:
        r = self.returns
        sharpe = calculate_sharpe(r, self.trades_per_year)
        
        if sharpe > 1.5:
            score, passed = 100, True
        elif sharpe >= 1.0:
            score, passed = 75, True
        elif sharpe >= 0.5:
            score, passed = 45, False
        else:
            score, passed = 15, False
        
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
            calculate_sharpe(self.returns, self.trades_per_year) > 1.0,
            calculate_max_drawdown(self.returns) < 0.20,
            calculate_win_rate(self.returns) > 45,
            len(self.returns) > 50
        ]
        
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

    def check_bull_performance(self) -> CheckResult:
        r = self.returns
        threshold = np.percentile(r, 60)
        bull = r[r > threshold]
        sharpe = calculate_sharpe(bull, self.trades_per_year) if len(bull) > 3 else 0
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
        sharpe = calculate_sharpe(bear, self.trades_per_year) if len(bear) > 3 else 0
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
        sharpe = calculate_sharpe(consol, self.trades_per_year) if len(consol) > 3 else 0
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

    def check_regime_robustness_hmm(self) -> CheckResult:
        if not HMM_AVAILABLE:
            return CheckResult(
                name="HMM Regime Robustness",
                passed=True,
                score=50,
                value="HMM not available - install hmmlearn",
                insight="Hidden Markov Model regime detection requires hmmlearn package.",
                fix="pip install hmmlearn for advanced regime detection.",
                category="Regime"
            )
        
        regimes = detect_regimes_hmm(self.returns)
        if not regimes:
            return CheckResult(
                name="HMM Regime Robustness",
                passed=False,
                score=30,
                value="HMM detection failed",
                insight="Could not detect regimes - data may be insufficient.",
                fix="Ensure at least 100 trades for regime detection.",
                category="Regime"
            )
        
        sharpe_by_regime = [r['sharpe'] for r in regimes.values()]
        min_sharpe = min(sharpe_by_regime)
        max_sharpe = max(sharpe_by_regime)
        all_positive = all(s > 0 for s in sharpe_by_regime)
        sharpe_range = max_sharpe - min_sharpe
        
        passed = all_positive and sharpe_range < 2.0
        score = 100 if all_positive else max(0, 50 + min_sharpe * 25)
        
        return CheckResult(
            name="HMM Regime Robustness",
            passed=passed,
            score=round(score, 1),
            value=f"Min Sharpe: {min_sharpe:.2f} | Max: {max_sharpe:.2f} | Range: {sharpe_range:.2f}",
            insight="HMM detects hidden market regimes. Robust strategies work across all regimes.",
            fix="If Sharpe varies >2.0 between regimes, add regime detection to adjust size.",
            category="Regime"
        )

    def check_volatility_stress(self) -> CheckResult:
        r = self.returns
        original_sharpe = calculate_sharpe(r, self.trades_per_year)
        stressed = r * 3.0
        stressed_sharpe = calculate_sharpe(stressed, self.trades_per_year)
        degradation = (original_sharpe - stressed_sharpe) / (abs(original_sharpe) + EPSILON) * 100
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

    def check_slippage_01(self) -> CheckResult:
        r = self.returns
        slipped = r - np.abs(r) * 0.001
        original = float(np.sum(r))
        after = float(np.sum(slipped))
        impact = abs((original - after) / (abs(original) + EPSILON) * 100)
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
        impact = abs((original - after) / (abs(original) + EPSILON) * 100)
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

    def check_market_impact(self) -> CheckResult:
        if 'volume' not in self.df.columns:
            return CheckResult(
                name="Market Impact (Simplified)",
                passed=True,
                score=50,
                value="No volume data - using static slippage",
                insight="Simplified square-root impact model requires volume data for full estimation.",
                fix="Add daily volume column to your data export.",
                category="Execution"
            )
        
        r = self.returns
        volumes = self.df['volume'].values
        avg_trade_size = np.mean(np.abs(r)) * 100000
        volatility = np.std(r)
        
        impacts = []
        for i in range(len(r)):
            impact = calculate_market_impact_simple(
                trade_size_dollars=avg_trade_size,
                daily_volume=volumes[i],
                volatility=volatility
            )
            impacts.append(impact)
        
        avg_impact = np.mean(impacts)
        slipped = r - np.sign(r) * np.array(impacts)
        original_pnl = np.sum(r)
        slipped_pnl = np.sum(slipped)
        impact_pct = abs((original_pnl - slipped_pnl) / (abs(original_pnl) + EPSILON)) * 100
        
        passed = avg_impact < 0.002
        score = max(0, 100 - impact_pct * 2)
        
        return CheckResult(
            name="Market Impact (Simplified)",
            passed=passed,
            score=round(score, 1),
            value=f"Avg impact: {avg_impact:.4f} | PnL reduction: {impact_pct:.1f}%",
            insight="Simplified square-root impact estimate. NOT full Almgren-Chriss implementation.",
            fix="Reduce position size to <1% of daily volume, or use VWAP/TWAP execution.",
            category="Execution"
        )

    def check_capacity_constraints(self) -> CheckResult:
        volumes = self.df['volume'].values if 'volume' in self.df.columns else None
        result = estimate_strategy_capacity(self.returns, volumes, current_aum=1e6)
        
        passed = result['capacity_utilization'] < 0.5
        score = max(0, 100 - result['capacity_utilization'] * 100)
        
        return CheckResult(
            name="Strategy Capacity",
            passed=passed,
            score=round(score, 1),
            value=f"Viable capacity: ${result['viable_capacity']:,.0f} | Utilization: {result['capacity_utilization']:.1%}",
            insight="Strategy degrades as AUM grows due to market impact. Stay below 50% of viable capacity.",
            fix="If utilization > 50%, reduce position sizes or trade more liquid instruments.",
            category="Execution"
        )

    def check_commission_drag(self) -> CheckResult:
        r = self.returns
        n = len(r)
        commission = n * 0.0005
        total = abs(float(np.sum(r)))
        drag = commission / (total + EPSILON) * 100
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
        impact = abs((float(np.sum(r)) - float(np.sum(adjusted))) / (abs(float(np.sum(r))) + EPSILON) * 100)
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
        bt_sharpe = calculate_sharpe(r, self.trades_per_year)
        live_sharpe = bt_sharpe * 0.6
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

    def check_sharpe_plausibility(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        
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
            passed=sharpe <= 10,
            score=100,
            value=f"Sharpe: {sharpe:.2f} → {status}",
            insight=insight,
            fix="If Sharpe > 10, verify data integrity and methodology",
            category="Plausibility"
        )

    def check_frequency_return_plausibility(self) -> CheckResult:
        mean_return = float(np.mean(self.returns))
        annual_return = mean_return * self.trades_per_year
        
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
        returns = self.returns
        std_return = np.std(returns)
        mean_return = np.mean(returns)
        max_dd = calculate_max_drawdown(returns)
        
        smoothness_ratio = std_return / (abs(mean_return) + EPSILON)
        dd_to_mean_ratio = max_dd / (abs(mean_return) + EPSILON)
        
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
        returns = self.returns
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if np.max(np.abs(returns)) > 0.5:
            mean_return = mean_return / 100
            variance = variance / 10000
        
        if variance < EPSILON:
            kelly = 0
        else:
            kelly = mean_return / variance
        
        if kelly > 5.0:
            status = "⚠ Manual Audit Required"
            insight = f"Kelly fraction {kelly:.1f} suggests unrealistic edge or underestimated variance"
        elif kelly > 2.0:
            status = "⚠ Review Recommended"
            insight = f"High Kelly fraction {kelly:.1f} requires edge verification"
        else:
            status = "✅ Plausible"
            insight = f"Kelly fraction {kelly:.2f} within realistic range"
        
        return CheckResult(
            name="Kelly Plausibility",
            passed=kelly <= 5.0,
            score=100,
            value=f"Kelly: {kelly:.2f} → {status}",
            insight=insight,
            fix="Verify return variance calculation and edge sustainability",
            category="Plausibility"
        )

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns.copy()
        
        vol_multiplier = 1 + (profile["vol_multiplier"] - 1) * 0.3
        stressed = r * vol_multiplier
        
        downside_bias = 1.2
        upside_cap = 0.8
        stressed = np.where(stressed < 0, stressed * downside_bias, stressed * upside_cap)
        
        liquidity_drag = np.abs(stressed) * (1 - profile["liquidity_factor"]) * 0.3
        stressed = stressed - liquidity_drag
        
        negative_extremes = stressed < np.percentile(stressed, 10)
        gap_losses = np.abs(stressed) * profile["gap_risk"] * negative_extremes
        stressed = stressed - gap_losses
        
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
            crash_name=profile["name"],
            year=profile["year"],
            description=profile["description"],
            market_drop=profile["market_drop"],
            strategy_drop=round(cumulative, 4),
            survived=survived,
            emotional_verdict=verdict
        )

    def _calculate_score(self, checks: List[CheckResult]) -> Tuple[float, str]:
        weights = {
            "Overfitting": 0.25,
            "Risk":        0.35,
            "Regime":      0.15,
            "Execution":   0.12,
            "Compliance":  0.13,
        }
        
        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat]
            if cat_checks:
                category_scores[cat] = float(np.mean([c.score for c in cat_checks]))
            else:
                category_scores[cat] = 0.0 if cat in ["Risk", "Overfitting", "Compliance"] else 20.0
        
        final = sum(category_scores[cat] * w for cat, w in weights.items())
        
        overall_sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if overall_sharpe < 0:
            final = min(final, 20)
        elif overall_sharpe < 0.3:
            final = min(final, 35)
        elif overall_sharpe < 0.5:
            final = min(final, 50)
        
        low_categories = sum(1 for score in category_scores.values() if score < 30)
        if low_categories >= 2:
            final = min(final, 45)
        elif low_categories >= 1:
            final = min(final, 60)
        
        compliance_checks = [c for c in checks if c.category == "Compliance"]
        if compliance_checks and not all(c.passed for c in compliance_checks):
            final = min(final, 55)
        
        if 'Short backtest' in self.timeframe_info:
            final = min(final, 75)
        
        if final >= 90:   grade = "A — Institutionally Viable"
        elif final >= 80: grade = "B+ — Prop Firm Ready"
        elif final >= 70: grade = "B — Live Tradeable"
        else:             grade = "F — Do Not Deploy"
        
        return round(final, 1), grade

    def run(self) -> ValidationReport:
        checks = [
            self.check_sharpe_decay(),
            self.check_monte_carlo(),
            self.check_outlier_dependency(),
            self.check_walk_forward_purged(),
            self.check_bootstrap_stability(),
            self.check_deflated_sharpe(),
            self.check_max_drawdown(),
            self.check_cvar(),
            self.check_calmar_ratio(),
            self.check_var(),
            self.check_consecutive_losses(),
            self.check_recovery_factor(),
            self.check_kelly_sizing(),
            self.check_absolute_sharpe(),
            self.check_bull_performance(),
            self.check_bear_performance(),
            self.check_consolidation_performance(),
            self.check_regime_robustness_hmm(),
            self.check_volatility_stress(),
            self.check_frequency_consistency(),
            self.check_regime_coverage(),
            self.check_slippage_01(),
            self.check_slippage_03(),
            self.check_market_impact(),
            self.check_capacity_constraints(),
            self.check_commission_drag(),
            self.check_partial_fills(),
            self.check_live_vs_backtest_gap(),
            self.check_compliance_pass(),
            self.check_sharpe_plausibility(),
            self.check_frequency_return_plausibility(),
            self.check_equity_smoothness_plausibility(),
            self.check_kelly_plausibility(),
        ]
        
        self.checks = checks
        
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
        
        if self.strict_mode:
            if sharpe < 1.2:
                grade = "F — Strict Mode: Sharpe below 1.2"
                score = min(score, 40)
            
            compliance_checks = [c for c in checks if c.category == "Compliance"]
            if compliance_checks and not all(c.passed for c in compliance_checks):
                grade = "F — Strict Mode: Compliance failed"
                score = min(score, 30)
            

        assumptions = self._get_assumptions()
        validation_hash = self._generate_validation_hash()
        validation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
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
            plausibility_summary=plausibility_summary,
            engine_version='v1.3.1',
            methodology_version='2026-03-01'
        )

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
            plausibility_summary=plausibility_summary,
            engine_version="v1.3.1",
            methodology_version="2026-03-01"
        )
class ValidationDashboard:
    """Interactive dashboard data generator for frontend visualization"""
    
    def __init__(self, validator: QuantProofValidator, report: ValidationReport):
        self.validator = validator
        self.report = report
        self.returns = validator.returns
        
    def generate_interactive_report(self) -> Dict[str, Any]:
        """Generate dashboard data for frontend charts and visualizations"""
        try:
            return {
                "equity_curve": self._generate_equity_data(),
                "regime_analysis": self._generate_regime_data(),
                "risk_metrics": self._generate_risk_data(),
                "performance_attribution": self._generate_attribution_data(),
                "monte_carlo": self._generate_monte_carlo_data()
            }
        except Exception as e:
            # Fallback data if generation fails
            return {
                "equity_curve": {"dates": [], "equity": [], "drawdown": []},
                "regime_analysis": {"regimes": [], "performance": {}},
                "risk_metrics": {"var": 0, "cvar": 0, "max_dd": 0},
                "performance_attribution": {"categories": [], "values": []},
                "monte_carlo": {"distribution": [], "percentiles": {}}
            }
    
    def _generate_equity_data(self) -> Dict[str, Any]:
        """Generate equity curve and drawdown data"""
        equity = np.cumprod(1 + self.returns)
        drawdown = (np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)
        
        dates = None
        if 'date' in self.validator.df.columns:
            dates = self.validator.df['date'].dt.strftime('%Y-%m-%d').tolist()
        
        return {
            "dates": dates or list(range(len(equity))),
            "equity": equity.tolist(),
            "drawdown": (drawdown * 100).tolist(),
            "returns": self.returns.tolist()
        }
    
    def _generate_regime_data(self) -> Dict[str, Any]:
        """Generate regime analysis data"""
        regime_data = {"regimes": [], "performance": {}}
        
        if 'regime' in self.validator.df.columns:
            regimes = self.validator.df['regime'].unique()
            for regime in regimes:
                mask = self.validator.df['regime'] == regime
                regime_returns = self.returns[mask]
                if len(regime_returns) > 0:
                    regime_data["performance"][str(regime)] = {
                        "sharpe": calculate_sharpe(regime_returns, self.validator.trades_per_year),
                        "win_rate": calculate_win_rate(regime_returns),
                        "count": len(regime_returns),
                        "avg_return": np.mean(regime_returns)
                    }
            regime_data["regimes"] = [str(r) for r in regimes]
        
        return regime_data
    
    def _generate_risk_data(self) -> Dict[str, Any]:
        """Generate risk metrics for visualization"""
        return {
            "var_95": float(np.percentile(self.returns, 5)),
            "var_99": float(np.percentile(self.returns, 1)),
            "cvar_95": float(np.mean(self.returns[self.returns <= np.percentile(self.returns, 5)])),
            "cvar_99": float(np.mean(self.returns[self.returns <= np.percentile(self.returns, 1)])),
            "max_drawdown": calculate_max_drawdown(self.returns),
            "volatility": float(np.std(self.returns)),
            "skewness": float(self._calculate_skewness()),
            "kurtosis": float(self._calculate_kurtosis())
        }
    
    def _generate_attribution_data(self) -> Dict[str, Any]:
        """Generate performance attribution by category"""
        category_scores = {}
        for check in self.report.checks:
            if check.category not in category_scores:
                category_scores[check.category] = []
            category_scores[check.category].append(check.score)
        
        categories = []
        values = []
        for cat, scores in category_scores.items():
            categories.append(cat)
            values.append(np.mean(scores))
        
        return {
            "categories": categories,
            "values": values,
            "overall_score": self.report.score
        }
    
    def _generate_monte_carlo_data(self) -> Dict[str, Any]:
        """Generate Monte Carlo simulation data"""
        n_sims = 1000
        final_equities = []
        
        for _ in range(n_sims):
            shuffled = self.validator.rng.permutation(self.returns)
            equity = np.prod(1 + shuffled)
            final_equities.append(equity)
        
        final_equities = np.array(final_equities)
        
        return {
            "distribution": final_equities.tolist(),
            "percentiles": {
                "5th": float(np.percentile(final_equities, 5)),
                "25th": float(np.percentile(final_equities, 25)),
                "50th": float(np.percentile(final_equities, 50)),
                "75th": float(np.percentile(final_equities, 75)),
                "95th": float(np.percentile(final_equities, 95))
            },
            "probability_profit": float(np.mean(final_equities > 1))
        }
    
    def _calculate_skewness(self) -> float:
        """Calculate return skewness"""
        if len(self.returns) < 3:
            return 0.0
        mean = np.mean(self.returns)
        std = np.std(self.returns)
        if std == 0:
            return 0.0
        return np.mean(((self.returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self) -> float:
        """Calculate return kurtosis (excess kurtosis)"""
        if len(self.returns) < 4:
            return 0.0
        mean = np.mean(self.returns)
        std = np.std(self.returns)
        if std == 0:
            return 0.0
        return np.mean(((self.returns - mean) / std) ** 4) - 3
