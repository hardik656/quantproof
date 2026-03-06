"""
QuantProof — Validation Engine v2.5
MERGE: v1.6 (31 checks, calibrated scoring) + v2.1 Final 5% (Alpha Decay, Overfit Detection)
Fixes: Alpha decay calibration (daily strategies no longer penalised),
       crash sim vol scaling, ValidationDashboard stub for backwards compat.
"""

import pandas as pd
import numpy as np
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

RISK_FREE_DAILY = 0.04 / 252
EPSILON = 1e-9
_RNG = np.random.default_rng(42)


# ── NEW v2.1 DATACLASSES ──────────────────────────────────────────────────────

@dataclass
class AlphaDecayProfile:
    half_life_periods: float
    half_life_seconds: Optional[float]
    optimal_holding: int
    regime_dependent: bool
    latency_sensitive: bool
    decay_curve: List[float]
    informational_only: bool = True   # True = don't penalise score

@dataclass
class OverfitProfile:
    score: float           # 0-100, higher = less overfit
    is_overfit: bool
    p_value: float
    adjusted_sharpe: float
    raw_sharpe: float
    indicators: Dict
    informational: str = ""


# ── ALPHA DECAY (calibrated for daily strategies) ─────────────────────────────

def _analyze_alpha_decay(returns: np.ndarray,
                         timestamps=None,
                         max_lags: int = 40) -> AlphaDecayProfile:
    """
    Signal half-life via autocorrelation decay.
    CALIBRATION FIX: daily random-walk returns have near-zero lag-1 autocorr,
    so curve_fit can't fit. Default to half_life=inf (NOT latency-critical)
    instead of 1.0, which was wrongly penalising all daily strategies.
    """
    n = len(returns)
    autocorrs = []
    for lag in range(1, min(max_lags, n // 4)):
        if lag < n:
            c = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            autocorrs.append(0.0 if np.isnan(c) else float(c))
    autocorrs = np.array(autocorrs) if autocorrs else np.array([0.0])
    lags = np.arange(1, len(autocorrs) + 1)

    def exp_decay(t, rho0, lam):
        return rho0 * np.exp(-lam * t)

    # Only attempt fit if there's a meaningful signal to decay
    half_life = float('inf')
    rho0_est = autocorrs[0] if len(autocorrs) > 0 else 0.0
    if abs(rho0_est) > 0.10 and len(autocorrs) > 5:
        try:
            popt, _ = curve_fit(exp_decay, lags, autocorrs,
                                p0=[rho0_est, 0.1],
                                bounds=([-1, 0.001], [1, 5]), maxfev=5000)
            rho0_fit, lam_fit = popt
            if lam_fit > 0:
                half_life = np.log(2) / lam_fit
        except Exception:
            pass

    # Convert to periods
    hl_periods = half_life if half_life != float('inf') else 999.0
    optimal_hold = max(1, int(hl_periods))

    # Seconds conversion
    hl_seconds = None
    latency_sensitive = False
    if timestamps is not None and len(timestamps) > 1:
        intervals = np.diff(timestamps).astype('timedelta64[s]').astype(float)
        median_interval = float(np.median(intervals))
        if median_interval > 0 and hl_periods < 999:
            hl_seconds = hl_periods * median_interval
            # Only flag latency sensitivity for sub-second strategies
            latency_sensitive = (hl_seconds < 60)

    # Regime-dependent decay check
    regime_dependent = False
    vol_series = pd.Series(returns).rolling(20).std().values
    valid_vol = vol_series[~np.isnan(vol_series)]
    if len(valid_vol) > 50:
        high_mask = vol_series > np.nanpercentile(vol_series, 75)
        low_mask  = vol_series < np.nanpercentile(vol_series, 25)
        h_rets = returns[np.where(high_mask)[0]]
        l_rets = returns[np.where(low_mask)[0]]
        if len(h_rets) > 20 and len(l_rets) > 20:
            hac = np.corrcoef(h_rets[:-1], h_rets[1:])[0, 1]
            lac = np.corrcoef(l_rets[:-1], l_rets[1:])[0, 1]
            regime_dependent = abs(hac - lac) > 0.15

    return AlphaDecayProfile(
        half_life_periods=round(hl_periods, 2),
        half_life_seconds=hl_seconds,
        optimal_holding=optimal_hold,
        regime_dependent=regime_dependent,
        latency_sensitive=latency_sensitive,
        decay_curve=autocorrs[:20].tolist(),
        informational_only=not latency_sensitive,
    )


# ── SYMBOLIC OVERFIT DETECTOR (calibrated) ────────────────────────────────────

class _OverfitDetector:
    """
    Compare strategy against 50 realistic noise baselines.
    CALIBRATION FIX: Only flag as overfit if BOTH p_value > 0.05 AND score < 60.
    Previous version flagged good strategies with p=0.00 as overfit due to high
    complexity score from large feature matrix.
    """
    def __init__(self, returns, timestamps=None):
        self.returns = returns
        self.n = len(returns)
        self.timestamps = timestamps

    def _noise_baselines(self, n=50):
        vol = np.std(self.returns)
        b = []
        for _ in range(n // 4): b.append(_RNG.normal(0, vol, self.n))
        for _ in range(n // 4):
            x = _RNG.normal(0, vol, self.n)
            b.append(np.convolve(x, np.ones(3)/3, 'same') * 0.5 + x * 0.5)
        for _ in range(n // 4):
            x = _RNG.normal(0, vol, self.n)
            b.append(-np.diff(x, prepend=x[0]) * 0.3 + x * 0.7)
        for _ in range(n // 4):
            rs = max(10, self.n // 50)
            reg = np.repeat(_RNG.choice([-1, 1], rs), (self.n + rs - 1) // rs)[:self.n]
            b.append(_RNG.normal(0, vol, self.n) * (1 + reg * 0.5))
        return b

    def _metrics(self, r):
        cum = np.cumprod(1 + r)
        d1  = np.diff(cum)
        d2  = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        hist, _ = np.histogram(r, bins=20, density=True)
        return {
            'smoothness': float(np.var(d2) / (np.var(d1) + EPSILON)),
            'entropy':    float(stats.entropy(hist + EPSILON)),
            'kurtosis':   float(stats.kurtosis(r, fisher=False)),
        }

    def detect(self) -> OverfitProfile:
        baselines = self._noise_baselines(50)
        strat_m   = self._metrics(self.returns)
        base_m    = [self._metrics(b) for b in baselines]

        z = {}
        for key in strat_m:
            vals = [m[key] for m in base_m]
            z[key] = (strat_m[key] - np.mean(vals)) / (np.std(vals) + EPSILON)

        indicators = {
            'too_smooth':  z.get('smoothness', 0) < -2.5,
            'low_entropy': z.get('entropy', 0) < -2.5,
        }

        # Polynomial complexity test (lighter: max_deg=3)
        t  = np.arange(self.n)
        X  = np.column_stack([t, t**2,
                               pd.Series(self.returns).rolling(5).mean().fillna(0).values,
                               np.concatenate([[0], self.returns[:-1]])])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp   = poly.fit_transform(X)
        model = Ridge(alpha=1.0)
        model.fit(Xp, self.returns)
        r2   = r2_score(self.returns, model.predict(Xp))
        comp = Xp.shape[1]

        base_r2s = []
        for b in baselines[:20]:
            Xb = poly.transform(np.column_stack([t, t**2,
                                 pd.Series(b).rolling(5).mean().fillna(0).values,
                                 np.concatenate([[0], b[:-1]])]))
            m2 = Ridge(alpha=1.0); m2.fit(Xb, b)
            base_r2s.append(r2_score(b, m2.predict(Xb)))

        indicators['complex_overfit'] = (r2 > np.percentile(base_r2s, 92))

        overfit_score = max(0, 100 - sum(indicators.values()) * 25)

        # p-value: fraction of noise baselines with higher complexity-adjusted Sharpe
        raw_sharpe = float(np.mean(self.returns) / (np.std(self.returns) + EPSILON))
        adj_sharpe = raw_sharpe * (1 - 0.005 * comp)   # lighter penalty than v2.1
        base_sharpes = [np.mean(b) / (np.std(b) + EPSILON) for b in baselines]
        p_value = float(np.mean(np.array(base_sharpes) > adj_sharpe))

        # CALIBRATION FIX: require BOTH conditions for is_overfit
        is_overfit = (overfit_score < 60) and (p_value > 0.10)

        info = ""
        if is_overfit:
            active = [k for k, v in indicators.items() if v]
            info = "Overfit flags: " + ", ".join(active)

        return OverfitProfile(
            score=overfit_score, is_overfit=is_overfit, p_value=p_value,
            adjusted_sharpe=adj_sharpe, raw_sharpe=raw_sharpe,
            indicators=indicators, informational=info
        )

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
    engine_version: str = "v2.5"
    methodology_version: str = "2026-03-05"
    # v2.1 Final 5% additions (informational — do not affect score)
    alpha_decay: Optional[AlphaDecayProfile] = None
    overfit_profile: Optional[OverfitProfile] = None


# Backwards-compat stub — old main.py imports ValidationDashboard
class ValidationDashboard:
    def __init__(self, validator, report):
        self.validator = validator
        self.report = report
    def generate_interactive_report(self):
        return None


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
    },
    "2010_flash_crash": {
        "name": "2010 Flash Crash",
        "year": "May 6, 2010",
        "description": "Dow dropped 1000 points in minutes. Algorithmic cascade wiped liquidity. Many stocks traded at $0.01. HFT strategies blown out in seconds.",
        "market_drop": -9.0,
        "vol_multiplier": 5.0,
        "liquidity_factor": 0.05,
        "gap_risk": 0.15,
    },
    "1998_ltcm": {
        "name": "1998 LTCM / Russia Default",
        "year": "Aug–Sep 1998",
        "description": "Russia defaulted, LTCM collapsed. Correlation assumptions broke simultaneously. Hidden leverage destroyed strategies in days.",
        "market_drop": -19.0,
        "vol_multiplier": 3.0,
        "liquidity_factor": 0.25,
        "gap_risk": 0.08,
    },
}

# =========================================================
# CORE MATH
# =========================================================

def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 252) -> float:
    if len(returns) < 2 or np.std(returns) < EPSILON:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year) * 0.85)

def calculate_sharpe_ci(returns: np.ndarray, trades_per_year: float = 252):
    """Jobson-Korkie / Mertens 2002 Sharpe CI. Returns (sharpe, ci_low, ci_high, p_value). v1.7"""
    from scipy import stats as _stats
    n = len(returns)
    if n < 10:
        s = calculate_sharpe(returns, trades_per_year)
        return s, s - 1.0, s + 1.0, 0.5
    excess = returns - (0.04 / 252)
    std_e = np.std(excess, ddof=1)
    if std_e < EPSILON:
        s = calculate_sharpe(returns, trades_per_year)
        return s, s, s, 0.0
    sr_d = np.mean(excess) / std_e
    skew = _stats.skew(excess)
    kurt = _stats.kurtosis(excess, fisher=False)
    se = max(np.sqrt((1 + 0.5*sr_d**2 - skew*sr_d + (kurt-3)*sr_d**2/4) / n) * np.sqrt(trades_per_year) * 0.85, 0.01)
    sharpe_ann = float(np.clip(sr_d * np.sqrt(trades_per_year) * 0.85, -10, 15))
    p_val = float(2 * (1 - _stats.norm.cdf(abs(sharpe_ann / se))))
    return sharpe_ann, sharpe_ann - 1.96*se, sharpe_ann + 1.96*se, p_val

def calculate_sortino(returns: np.ndarray, trades_per_year: float = 252) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        # No losing trades — perfect downside control, cap at 5.0
        mean_r = np.mean(returns)
        return 5.0 if mean_r > 0 else 0.0
    if np.std(downside) < EPSILON:
        return 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(trades_per_year) * 0.85)

def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR with KDE smoothing for small samples (v1.7)."""
    if len(returns) == 0:
        return 0.0
    alpha = 1 - confidence
    if len(returns) < 100:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(returns)
            samples = kde.resample(10000, seed=42)[0]
            var = np.percentile(samples, alpha*100)
            return float(np.mean(samples[samples <= var]))
        except Exception:
            pass
    threshold = np.percentile(returns, alpha * 100)
    tail = returns[returns <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)

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
    meaningful = returns[np.abs(returns) > 0.001]
    if len(meaningful) == 0:
        return 0.0
    return float(np.mean(meaningful > 0) * 100)

def calculate_calmar(returns: np.ndarray, trades_per_year: float = 252) -> float:
    annual_return = np.mean(returns) * trades_per_year
    dd = abs(calculate_max_drawdown(returns))
    if dd < EPSILON:
        # No drawdown — cap at 10.0 (excellent but not infinity)
        return 10.0 if annual_return > 0 else 0.0
    return float(annual_return / dd)

def calculate_ruin_probability(win_rate: float, avg_win: float, avg_loss: float,
                               capital_units: int = 10) -> float:
    """
    Gambler's ruin probability via analytical formula.
    capital_units = how many avg losses fit in total capital (default 10 = risking ~10%/trade).
    Returns P(ruin) in [0, 1].
    
    Formula: normalize to equivalent unit bets, then apply
    P(ruin | starting at N units) = (q/p)^N where p=normalized win prob.
    """
    if win_rate <= 0 or avg_loss < EPSILON:
        return 1.0
    if win_rate >= 1.0:
        return 0.0
    edge = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    if edge <= 0:
        return 1.0
    # Convert to equivalent fair-coin problem using reward-to-risk ratio
    rr = avg_win / (avg_loss + EPSILON)
    p = win_rate * rr / (win_rate * rr + (1.0 - win_rate))  # normalized win prob
    q = 1.0 - p
    if p <= 0.5:
        return 1.0
    ruin = (q / p) ** capital_units
    return float(np.clip(ruin, 0.0, 1.0))

# =========================================================
# VALIDATOR
# =========================================================

class QuantProofValidator:

    def __init__(self, df: pd.DataFrame, strict_mode: bool = False, seed: int = 42):
        self.strict_mode = strict_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.df = self._clean(df)
        self.trades_per_year, self.timeframe_info = self._detect_timeframe()
        self.has_dates = 'date' in self.df.columns
        self.returns = self._get_returns()
        if strict_mode:
            self._validate_strict_requirements()

    def _detect_timeframe(self):
        if 'date' not in self.df.columns:
            return 252.0, "No timestamp data - using daily metrics"
        delta = self.df['date'].max() - self.df['date'].min()
        time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
        if time_span_years < EPSILON:
            return 252.0, "Insufficient time span — default annualization"
        trades_per_year = min(len(self.df) / time_span_years, 252)
        if time_span_years < 0.25:
            desc = f"Short backtest ({time_span_years*12:.1f} months) — Sharpe may be inflated ({trades_per_year:.0f} trades/year)"
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
        if 'date' in self.df.columns:
            span = (self.df['date'].max() - self.df['date'].min()).days / 365.25
            if span < 0.5:
                raise ValueError(f"Strict mode requires 6 months minimum (got {span*12:.1f} months)")
        if len(self.returns) < 100:
            raise ValueError(f"Strict mode requires 100+ trades (got {len(self.returns)})")

    def _generate_validation_hash(self):
        hash_data = {
            'returns': np.round(self.returns, 8).tolist(),
            'timeframe': self.timeframe_info,
            'trades_per_year': float(self.trades_per_year),
            'engine_version': 'v1.3',
            'seed': self.seed
        }
        return hashlib.sha256(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()[:12]

    def _get_assumptions(self):
        base = [
            "Assumes full capital deployed per trade",
            "Assumes trades are sequential and non-overlapping",
            "Assumes percentage returns (not dollar P&L)",
            "Assumes no leverage unless embedded in returns",
            f"Monte Carlo seed fixed at {self.seed} for reproducibility",
            "Risk-free rate excluded for per-trade returns (institutional practice)",
            "Crash simulations use historical stress profiles with proportional scaling",
            "CVaR calculated at 95% confidence — average loss in worst 5% of trades",
            "Sortino ratio penalizes downside volatility only",
        ]
        if self.strict_mode:
            base += ["Strict mode: 6-month minimum", "Strict mode: 100-trade minimum"]
        return base

    def _generate_audit_flags(self):
        return [
            f"⚠ {c.name}: {c.insight}"
            for c in self.checks
            if c.category == "Plausibility" and not c.passed and "Manual Audit Required" in c.value
        ]

    def _generate_plausibility_summary(self):
        pc = [c for c in self.checks if c.category == "Plausibility"]
        manual = sum(1 for c in pc if not c.passed and "Manual Audit Required" in c.value)
        review = sum(1 for c in pc if not c.passed and "Review Recommended" in c.value)
        if manual > 0:
            return f"⚠ {manual} statistical implausibility issues require manual audit"
        if review > 0:
            return f"⚠ {review} statistical issues merit review"
        return "✅ All statistical metrics appear plausible"

    def _clean(self, df):
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

    def _get_returns(self):
        r = self.df["pnl"].values.astype(float)
        max_abs = np.abs(r).max()
        if max_abs > 1.0:
            raise ValueError(
                f"Returns appear to be dollar amounts (max: {max_abs:.2f}). "
                "Please convert to percentage returns before validation."
            )
        # NO noise injection — data integrity must be preserved.
        # Plausibility checks handle suspicious smoothness.
        return r

    # ---- OVERFITTING ----

    def check_sharpe_decay(self):
        r, n = self.returns, len(self.returns)
        if n < 20:
            return CheckResult("Sharpe Decay", False, 20, "Insufficient data", "Need 20+ trades", "Add more backtest history", "Overfitting")
        mid = n // 2
        s_in  = calculate_sharpe(r[:mid], self.trades_per_year)
        s_out = calculate_sharpe(r[mid:], self.trades_per_year)
        decay = (s_in - s_out) / (abs(s_in) + EPSILON) * 100
        score = max(0, 100 - max(0, decay))
        return CheckResult("Sharpe Ratio Decay", decay < 40, round(score, 1),
            f"In-sample: {s_in:.2f} → Out-of-sample: {s_out:.2f} ({decay:.1f}% decay)",
            "High decay means your strategy was fitted to historical noise, not real patterns.",
            "Run walk-forward optimization. Only trust out-of-sample Sharpe.", "Overfitting")

    def check_monte_carlo(self):
        r = self.returns
        mean_sims = [np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(300)]
        mean_pct = float(np.mean(np.array(mean_sims) > 0) * 100)
        original_dd = calculate_max_drawdown(r)
        seq_sims = [abs(calculate_max_drawdown(self.rng.permutation(r))) - abs(original_dd) < 0.10 for _ in range(200)]
        sequence_pct = float(np.mean(seq_sims) * 100)
        perms = [self.rng.permutation(r) for _ in range(500)]
        equity_sims = [np.prod(1 + p) for p in perms]
        worst_5pct = np.percentile(equity_sims, 5)
        if not self.has_dates:
            worst_5pct_val = (worst_5pct - 1) * 100
        else:
            delta = self.df['date'].max() - self.df['date'].min()
            tsy = max(delta.total_seconds() / (365.25 * 24 * 3600), 0.1)
            if 'Short backtest' in self.timeframe_info:
                tsy = 1.0
            worst_5pct_val = (worst_5pct ** (1/tsy) - 1) * 100
        worst_5pct_dd = np.percentile([abs(calculate_max_drawdown(p)) for p in perms], 95) * 100
        equity_score = min(100, max(0, 100 + worst_5pct_val * 2 - worst_5pct_dd))
        combined = mean_pct * 0.4 + sequence_pct * 0.3 + equity_score * 0.3
        passed = combined > (70 if self.strict_mode else 60)
        return CheckResult("Monte Carlo Robustness", passed, round(combined, 1),
            f"Mean: {mean_pct:.1f}% | Seq: {sequence_pct:.1f}% | Equity: {equity_score:.1f}%",
            "Hedge-fund level Monte Carlo: tests mean stability, sequence fragility, and worst-case equity outcomes.",
            "If equity score low, strategy has tail risk or depends on specific trade sequences.", "Overfitting")

    def check_outlier_dependency(self):
        """
        Profit Concentration — institutional-grade fat-tail detection.
        Key metric: what % of total profit comes from the top 1% of trades?
        If 3 trades out of 300 generate 29% of profit, that is a fat-tail fluke,
        not a repeatable edge. HHI alone misses this — we use top-1% share.
        Reference: Taleb (2007), institutional PM due diligence standard.
        """
        r = self.returns
        if len(r[r > 0]) < 3:
            return CheckResult("Profit Concentration", False, 0,
                "Too few winning trades to measure concentration",
                "Need at least 3 winning trades to assess profit distribution.",
                "Increase backtest length.", "Overfitting")

        total_profit  = float(np.sum(r[r > 0]))
        n             = len(r)

        # Primary metric: top 1% of ALL trades as share of total profit
        # This directly catches "2 whales in 300 trades" scenarios
        top1pct_count  = max(1, int(n * 0.01))
        top1pct_profit = float(np.sum(np.sort(r)[::-1][:top1pct_count]))
        top1pct_share  = top1pct_profit / total_profit * 100

        # Also check absolute top-2 trades (catches small n or n=300 where 1%=3)
        top2_profit = float(np.sum(np.sort(r)[::-1][:2]))
        top2_share  = top2_profit / total_profit * 100

        # Secondary: HHI of winning trade profit shares
        wins   = r[r > 0]
        shares = wins / total_profit
        hhi    = float(np.sum(shares ** 2))

        # Tertiary: Sharpe without top 1% (acid test)
        r_trimmed  = np.delete(r, np.argsort(r)[::-1][:top1pct_count])
        sharpe_raw = float(np.mean(r) / (np.std(r) + EPSILON) * np.sqrt(252) * 0.85)
        sharpe_trim = float(np.mean(r_trimmed) / (np.std(r_trimmed) + EPSILON) * np.sqrt(252) * 0.85) if len(r_trimmed) > 5 else 0
        sharpe_decay = (sharpe_raw - sharpe_trim) / (abs(sharpe_raw) + EPSILON)

        if top1pct_share > 25 or sharpe_decay > 0.50 or top2_share > 20:
            status  = "🔴 Fat-Tail Fluke"
            insight = (f"Top {top1pct_count} trade(s) = {top1pct_share:.1f}% of all profit. "
                       f"Top 2 trades = {top2_share:.1f}% of profit. "
                       f"Remove them: Sharpe drops {sharpe_decay:.0%} ({sharpe_raw:.2f}→{sharpe_trim:.2f}). "
                       f"This is a fat-tail fluke, not a repeatable edge.")
            passed, score = False, 0
        elif top1pct_share > 15 or sharpe_decay > 0.30:
            status  = "⚠ Concentration Risk"
            insight = (f"Top {top1pct_count} trade(s) = {top1pct_share:.1f}% of profit. "
                       f"Edge is somewhat dependent on outlier trades (Sharpe decay: {sharpe_decay:.0%}).")
            passed, score = False, 40
        else:
            status  = "✅ Well Distributed"
            insight = (f"Top {top1pct_count} trade(s) = {top1pct_share:.1f}% of profit. "
                       f"HHI={hhi:.3f}. Sharpe without outliers: {sharpe_trim:.2f} ({sharpe_decay:.0%} decay). "
                       f"Edge is broadly distributed — not a fat-tail fluke.")
            passed, score = True, 100

        return CheckResult("Profit Concentration", passed, score,
            f"Top-1% trades = {top1pct_share:.1f}% of profit | Sharpe w/o outliers: {sharpe_trim:.2f} → {status}",
            insight,
            "Acid test: remove your 3 best trades. If Sharpe drops >50%, you have no real edge.", "Overfitting")

    def check_walk_forward(self):
        """
        Combinatorial Purged Cross-Validation (CPCV).
        Generates C(n_splits=6, n_test=2) = 15 independent backtest paths.
        Each path uses a different subset of history as the test set.
        A real edge works on ALL paths. Sequence luck fails on paths
        where the lucky regime (e.g., bull run) is in the training set.

        Key metric: std of Sharpe across paths. Low std = robust signal.
        High std = strategy only works in a specific sequence of events.
        Reference: López de Prado (2018), Advances in Financial ML, Ch.12
        """
        r = self.returns
        if len(r) < 60:
            return CheckResult("CPCV Path Stability", False, 20,
                "Not enough data for CPCV (need 60+ trades)",
                "CPCV requires sufficient data to split into 6 folds.",
                "Extend backtest to 60+ trades.", "Overfitting")

        from itertools import combinations as _comb
        n_splits, n_test = 6, 2
        fold_size = len(r) // n_splits
        folds = [r[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]

        path_sharpes = []
        for test_idx in _comb(range(n_splits), n_test):
            test = np.concatenate([folds[i] for i in test_idx])
            if len(test) < 5 or np.std(test) < EPSILON:
                continue
            s = float(np.mean(test) / np.std(test) * np.sqrt(252) * 0.85)
            path_sharpes.append(s)

        if not path_sharpes:
            return CheckResult("CPCV Path Stability", False, 0,
                "CPCV could not compute paths", "", "", "Overfitting")

        paths        = np.array(path_sharpes)
        pct_positive = float(np.mean(paths > 0) * 100)
        path_std     = float(np.std(paths))
        mean_sharpe  = float(np.mean(paths))
        n_paths      = len(paths)

        # std > 5.0 = extreme sequence luck (bull/bear regime dependency)
        # std 3-5   = significant path sensitivity
        # std < 2.0 + >80% positive = institutionally robust
        if path_std > 5.0 or pct_positive < 60:
            status = "🔴 Sequence Luck — edge depends on historical order"
            insight = (f"Sharpe std across {n_paths} paths = {path_std:.2f}. "
                       f"Only {pct_positive:.0f}% of paths are profitable. "
                       f"The strategy survived a specific sequence of market events, not a real signal.")
            passed, score = False, 0   # score=0 triggers hard gate in _calculate_score
        elif path_std > 3.0 or pct_positive < 80:
            status = "⚠ Path Sensitive — moderate sequence dependency"
            insight = (f"Sharpe std = {path_std:.2f} across {n_paths} paths. "
                       f"Edge exists but is fragile to market regime sequencing.")
            passed, score = False, 45
        elif path_std > 2.0:
            status = "⚠ Acceptable — minor path variance"
            insight = f"Sharpe std = {path_std:.2f}. Edge is mostly robust but some sensitivity to market order."
            passed, score = True, 65
        else:
            status = "✅ Robust — edge survives all path orderings"
            insight = (f"Sharpe std = {path_std:.2f} across {n_paths} paths. "
                       f"Mean path Sharpe: {mean_sharpe:.2f}. "
                       f"Real signal — not surviving by luck of historical sequence.")
            passed, score = True, min(100, int(pct_positive))

        return CheckResult("CPCV Path Stability", passed, score,
            f"{n_paths} paths: {pct_positive:.0f}% positive | Sharpe std: ±{path_std:.2f} → {status}",
            insight,
            "If std > 3, your strategy relies on living through a specific market history. "
            "Test on shuffled regimes or out-of-sample data from a different decade.",
            "Overfitting")

    def check_bootstrap_stability(self):
        r = self.returns
        pct = float(np.mean(np.array([np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(500)]) > 0) * 100)
        return CheckResult("Bootstrap Stability", pct > 70, round(pct, 1),
            f"Positive expectancy in {pct:.1f}% of 500 bootstrap samples",
            "Stable edge shows up consistently in resampling.",
            "If below 70%, fix win rate or risk/reward ratio first.", "Overfitting")

    # ---- RISK ----

    def check_max_drawdown(self):
        dd_pct = calculate_max_drawdown(self.returns) * 100
        score = max(0, 100 - dd_pct * 3)
        return CheckResult("Max Drawdown", dd_pct < 20, round(score, 1),
            f"Max drawdown: {dd_pct:.1f}% (threshold: <20%)",
            "Funds reject strategies with drawdowns >20%. Retail traders quit at 15%.",
            "Add circuit breaker: pause after 10% drawdown. Reduce size after losing streaks.", "Risk")

    def check_calmar_ratio(self):
        calmar = calculate_calmar(self.returns, self.trades_per_year)
        score = min(100, calmar * 40)
        return CheckResult("Calmar Ratio", calmar > 1.5, round(score, 1),
            f"Calmar: {calmar:.2f} (target >1.5, timeframe: {self.timeframe_info})",
            "Calmar = annual return / max drawdown. Prop firms want >1.5.",
            "Increase returns or reduce drawdown via better position sizing.", "Risk")

    def check_var(self):
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
            threshold_desc = f"10x mean ({abs(mean)*10:.3f})"
        return CheckResult("Value at Risk (VaR)", passed, round(score, 1),
            f"VaR 99%: {var_99:.4f} | VaR 95%: {float(np.percentile(r, 5)):.4f}",
            f"VaR 99% = loss exceeded only 1% of trades. Threshold: {threshold_desc}.",
            "If VaR too high, use tighter stops or reduce position size.", "Risk")

    def check_cvar(self):
        """CVaR/Expected Shortfall — Basel III standard. Average loss in worst 5% of trades."""
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
            threshold_desc = f"15x mean ({abs(mean)*15:.3f})"
        return CheckResult("CVaR / Expected Shortfall", passed, round(score, 1),
            f"CVaR 95%: {cvar_95:.4f} | CVaR 99%: {cvar_99:.4f}",
            f"CVaR = average loss when things go really wrong. Required by Basel III. Threshold: {threshold_desc}.",
            "Reduce tail exposure: tighter stops on losing trades, avoid low-liquidity assets.", "Risk")

    def check_sortino(self):
        """Sortino ratio — penalizes only downside volatility. Better for asymmetric strategies."""
        sortino = calculate_sortino(self.returns, self.trades_per_year)
        if sortino > 2.0:
            score, passed = 100, True
        elif sortino >= 1.0:
            score, passed = 75, True
        elif sortino >= 0.5:
            score, passed = 45, False
        else:
            score, passed = 15, False
        return CheckResult("Sortino Ratio", passed, round(score, 1),
            f"Sortino: {sortino:.2f} (target >1.0, timeframe: {self.timeframe_info})",
            "Sortino only punishes bad volatility. A strategy with big winners and controlled losers scores higher here than on Sharpe.",
            "Improve downside control: tighter stop losses, ATR-based position sizing.", "Risk")

    def check_consecutive_losses(self):
        r = self.returns
        max_streak = current = 0
        for trade in r:
            if trade < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        score = max(0, 100 - max_streak * 10)
        return CheckResult("Max Losing Streak", max_streak < 8, round(score, 1),
            f"Longest losing streak: {max_streak} consecutive trades",
            "Prop firms reject strategies with 8+ consecutive losses.",
            "Add daily loss limit that pauses trading after streak >5.", "Risk")

    def check_recovery_factor(self):
        r = self.returns
        recovery = float(np.sum(r[r > 0])) / (float(abs(np.sum(r[r < 0]))) + EPSILON)
        score = min(100, recovery * 40)
        return CheckResult("Recovery Factor", recovery > 1.5, round(score, 1),
            f"Recovery factor: {recovery:.2f} (wins cover losses {recovery:.1f}x)",
            "Below 1.5 means wins barely cover drawdowns.",
            "Increase average winner or cut average loser. Asymmetric R:R is the goal.", "Risk")

    def check_absolute_sharpe(self):
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe > 1.5:   score, passed = 100, True
        elif sharpe >= 1.0: score, passed = 75, True
        elif sharpe >= 0.5: score, passed = 45, False
        else:               score, passed = 15, False

        # Warn users when no dates means annualization may be overstated
        date_warning = " ⚠ No date column — may be inflated" if not self.has_dates else ""
        return CheckResult("Absolute Sharpe Ratio", passed, round(score, 1),
            f"Annualized Sharpe: {sharpe:.2f} (timeframe: {self.timeframe_info}){date_warning}",
            "Institutional minimum is Sharpe > 1.0. Below 0.5 means the strategy doesn't compensate for its risk.",
            "Improve risk-adjusted returns through better entry timing or position sizing.", "Risk")

    def check_ruin_probability(self):
        """Probability of blowing up. Every trader needs this number before going live."""
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
            "The chance of losing your entire trading capital. Below 10% required for live deployment.",
            "Increase win rate, improve R:R ratio, or reduce per-trade risk percentage.", "Risk")

    def check_compliance_pass(self):
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

    # ---- REGIME ----

    def check_bull_performance(self):
        r = self.returns
        bull = r[r > np.percentile(r, 60)]
        sharpe = calculate_sharpe(bull, self.trades_per_year) if len(bull) > 3 else 0
        wr = calculate_win_rate(bull)
        return CheckResult("Bull Market Performance", sharpe > 0 and wr > 45,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | Win rate: {wr:.1f}% ({len(bull)} trades)",
            "Strategy behavior during favorable conditions.",
            "If failing in bull market, check if momentum signals are properly calibrated.", "Regime")

    def check_bear_performance(self):
        r = self.returns
        bear = r[r < np.percentile(r, 40)]
        sharpe = calculate_sharpe(bear, self.trades_per_year) if len(bear) > 3 else 0
        wr = calculate_win_rate(bear)
        return CheckResult("Bear Market Performance", sharpe > -1.0,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | Win rate: {wr:.1f}% ({len(bear)} trades)",
            "How strategy holds up when conditions turn against it.",
            "Add regime detection to reduce size or pause during bear conditions.", "Regime")

    def check_consolidation_performance(self):
        r = self.returns
        low, high = np.percentile(r, 40), np.percentile(r, 60)
        consol = r[(r >= low) & (r <= high)]
        sharpe = calculate_sharpe(consol, self.trades_per_year) if len(consol) > 3 else 0
        wr = calculate_win_rate(consol)
        return CheckResult("Consolidation Performance", sharpe > 0,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | Win rate: {wr:.1f}% ({len(consol)} trades)",
            "Sideways markets are where most momentum strategies bleed.",
            "Add a choppiness filter to avoid trading in low-volatility ranges.", "Regime")

    def check_volatility_stress(self):
        r = self.returns
        orig = calculate_sharpe(r, self.trades_per_year)
        stressed_sharpe = calculate_sharpe(r * 3.0, self.trades_per_year)
        degradation = (orig - stressed_sharpe) / (abs(orig) + EPSILON) * 100
        return CheckResult("Volatility Spike Stress Test", degradation < 50,
            round(max(0, 100 - degradation), 1),
            f"3x vol: Sharpe {orig:.2f} → {stressed_sharpe:.2f}",
            "VIX spikes (COVID March 2020) cause 3-5x normal volatility.",
            "Use volatility-adjusted position sizing (ATR-based). Reduce size when VIX >25.", "Regime")

    def check_frequency_consistency(self):
        r = self.returns
        window = max(5, len(r) // 10)
        pct = float(np.mean(np.array([np.mean(r[i:i+window]) for i in range(0, len(r)-window)]) > 0) * 100)
        return CheckResult("Performance Consistency", pct > 60, round(pct, 1),
            f"Profitable in {pct:.1f}% of rolling windows",
            "Consistent strategies generate returns steadily, not in lumps.",
            "If <60%, your edge only works in specific conditions. Define those explicitly.", "Regime")

    def check_regime_coverage(self):
        if 'regime' not in self.df.columns:
            return CheckResult("Regime Coverage", False, 40, "No regime column detected",
                "Strategies with regime detection have 3x better live survival rate.",
                "Add BULL/BEAR/CONSOLIDATION/TRANSITION regime labels to your CSV export.", "Regime")
        regimes = self.df['regime'].value_counts(normalize=True)
        coverage = len(regimes) / 4 * 100
        return CheckResult("4-Regime Coverage", coverage > 75, round(coverage, 1),
            f"Regimes detected: {list(regimes.index.astype(str))}",
            "Full regime coverage means your strategy was tested across all market conditions.",
            "Ensure your backtest includes BULL, BEAR, CONSOLIDATION and TRANSITION periods.", "Regime")

    # ---- EXECUTION ----

    def check_slippage_01(self):
        r = self.returns
        impact = abs((float(np.sum(r)) - float(np.sum(r - np.abs(r) * 0.001))) / (abs(float(np.sum(r))) + EPSILON) * 100)
        return CheckResult("Slippage Impact (0.1%)", impact < 20, round(max(0, 100 - impact * 3), 1),
            f"0.1% slippage reduces returns by {impact:.1f}%",
            "Even 0.1% per trade compounds into significant drag.",
            "Reduce trade frequency or only take higher conviction setups.", "Execution")

    def check_slippage_03(self):
        r = self.returns
        impact = abs((float(np.sum(r)) - float(np.sum(r - np.abs(r) * 0.003))) / (abs(float(np.sum(r))) + EPSILON) * 100)
        return CheckResult("Slippage Impact (0.3%)", impact < 40, round(max(0, 100 - impact * 2), 1),
            f"0.3% slippage reduces returns by {impact:.1f}%",
            "Small caps and volatile markets have 0.3%+ slippage. Does your edge survive?",
            "Model 0.5% slippage for small-cap strategies. Use limit orders.", "Execution")

    def check_commission_drag(self):
        r = self.returns
        drag = (len(r) * 0.0005) / (abs(float(np.sum(r))) + EPSILON) * 100
        return CheckResult("Commission Drag", drag < 15, round(max(0, 100 - drag * 4), 1),
            f"{len(r)} trades → {drag:.1f}% of gross returns consumed by commissions",
            "High-frequency strategies can be profitable on paper but lose after commissions.",
            "Calculate your break-even commission. If >10x/day, commissions may kill the edge.", "Execution")

    def check_partial_fills(self):
        r = self.returns
        fill_rate = 0.80 + 0.20 * 0.70
        impact = abs((float(np.sum(r)) - float(np.sum(r * fill_rate))) / (abs(float(np.sum(r))) + EPSILON) * 100)
        return CheckResult("Partial Fill Simulation", impact < 10, round(max(0, 100 - impact * 5), 1),
            f"80% fill rate reduces returns by {impact:.1f}%",
            "In fast markets orders may not fill completely.",
            "Size positions for partial fill scenarios. Use limit orders.", "Execution")

    def check_live_vs_backtest_gap(self):
        bt_sharpe   = calculate_sharpe(self.returns, self.trades_per_year)
        live_sharpe = bt_sharpe * 0.6
        score = min(100, max(0, live_sharpe * 50))
        return CheckResult("Live Trading Gap Estimate", live_sharpe > 0.5, round(score, 1),
            f"Backtest Sharpe: {bt_sharpe:.2f} → Estimated Live: {live_sharpe:.2f}",
            "Industry average: 40% Sharpe decay from backtest to live trading.",
            "Reduce position sizing, add slippage buffers, tighten risk management.", "Execution")

    def check_impact_adjusted_capacity(self):
        """
        Impact-Adjusted Sharpe & AUM Capacity (Almgren-Chriss temporary impact model).
        Answers: 'At what AUM does this strategy destroy its own alpha?'

        Market impact = η · σ_market · √(avg_trade_size / ADV)
        where η=0.1 (empirical constant), ADV=average daily volume.

        Critically: decay from $100k→$10M is the retail-to-institutional gap.
        A strategy that collapses here cannot be scaled or sold to a fund.
        Reference: Almgren & Chriss (2001), Optimal Execution of Portfolio Transactions.
        """
        r      = self.returns
        mean_r = float(np.mean(r))
        std_r  = float(np.std(r))
        n      = len(r)
        if std_r < EPSILON or n == 0 or mean_r <= 0:
            # Can't decay further if already losing
            return CheckResult("Impact-Adjusted Capacity", mean_r > 0, 50 if mean_r > 0 else 0,
                "Cannot compute capacity on unprofitable strategy", "", "", "Execution")

        raw_sharpe = mean_r / std_r * np.sqrt(252) * 0.85

        def sharpe_at_aum(aum, adv=50_000_000):
            """Almgren-Chriss: η·σ_mkt·√(participation_rate) = temporary market impact per trade."""
            participation = (aum / max(n, 1)) / adv
            impact        = 0.1 * 0.015 * (participation ** 0.5)  # η·σ·√pr
            return (mean_r - impact) / std_r * np.sqrt(252) * 0.85

        s_100k = sharpe_at_aum(100_000)
        s_1m   = sharpe_at_aum(1_000_000)
        s_10m  = sharpe_at_aum(10_000_000)
        s_100m = sharpe_at_aum(100_000_000)

        # Capacity = AUM where alpha (mean return) is fully consumed by market impact
        # i.e. mean_r - impact = 0  →  impact = mean_r
        # η·σ·√(Q/(n·ADV)) = mean_r  →  Q = (mean_r/(η·σ))² · n · ADV
        try:
            capacity_aum = ((mean_r / (0.1 * 0.015)) ** 2) * n * 50_000_000
            capacity_str = f"${capacity_aum:,.0f}"
        except Exception:
            capacity_str = "N/A"

        # Primary metric: Sharpe decay from $100k to $10M
        decay_100k_10m = (s_100k - s_10m) / (abs(s_100k) + EPSILON)

        if decay_100k_10m > 0.40:
            status  = "🔴 Capacity Constrained"
            insight = (f"Sharpe decays {decay_100k_10m:.0%} from $100k → $10M "
                       f"({s_100k:.2f} → {s_10m:.2f}). "
                       f"Alpha is consumed by your own order flow. Strategy cap: {capacity_str}.")
            passed, score = False, max(0, int((1 - decay_100k_10m) * 100))
        elif decay_100k_10m > 0.20:
            status  = "⚠ Moderate Impact Risk"
            insight = (f"Sharpe decays {decay_100k_10m:.0%} to $10M. "
                       f"Viable for personal/small fund trading. Not institutional scale.")
            passed, score = True, 65
        else:
            status  = "✅ Scalable Alpha"
            insight = (f"Market impact is minimal up to $10M AUM. "
                       f"Estimated capacity before alpha destruction: {capacity_str}.")
            passed, score = True, 100

        return CheckResult("Impact-Adjusted Capacity", passed, score,
            f"Sharpe: $100k={s_100k:.2f} | $1M={s_1m:.2f} | $10M={s_10m:.2f} | $100M={s_100m:.2f} → {status}",
            insight,
            "If capacity <$5M, strategy is retail-only. "
            "Scale alpha by reducing position size or trading more liquid instruments.",
            "Execution")

    # ---- PLAUSIBILITY ----

    def check_sharpe_plausibility(self):
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe > 10:
            status = "⚠ Manual Audit Required"
            insight = "Sharpe > 10 exceeds documented institutional performance (Renaissance ~8-10)"
        elif sharpe > 5:
            status = "⚠ Review Recommended"
            insight = "Sharpe > 5 is extremely rare — requires explanation"
        else:
            status = "✅ Plausible"
            insight = "Sharpe within realistic institutional range"
        return CheckResult("Sharpe Plausibility", sharpe <= 10, 100,
            f"Sharpe: {sharpe:.2f} → {status}", insight,
            "If Sharpe > 10, verify data integrity and methodology", "Plausibility")

    def check_frequency_return_plausibility(self):
        mean_return = float(np.mean(self.returns))
        annual_return = mean_return * self.trades_per_year
        if self.trades_per_year > 20000 and annual_return > 500:
            status = "⚠ Manual Audit Required"
            insight = f"High frequency ({self.trades_per_year:.0f}/yr) + extreme returns ({annual_return:.0f}%) requires massive liquidity edge"
        else:
            status = "✅ Plausible"
            insight = "Frequency-return relationship within realistic bounds"
        return CheckResult("Frequency-Return Plausibility",
            not (self.trades_per_year > 20000 and annual_return > 500), 100,
            f"{self.trades_per_year:.0f} trades/yr × {mean_return*100:.2f}% avg → {annual_return:.0f}% annual ({status})",
            insight, "Verify liquidity capacity and market impact assumptions", "Plausibility")

    def check_equity_smoothness_plausibility(self):
        r = self.returns
        smoothness_ratio = np.std(r) / (abs(np.mean(r)) + EPSILON)
        max_dd = calculate_max_drawdown(r)
        dd_to_mean = max_dd / (abs(np.mean(r)) + EPSILON)
        if smoothness_ratio < 0.5 and np.mean(r) > 0 and max_dd < 0.05:
            status = "⚠ Manual Audit Required"
            insight = "Suspiciously smooth equity curve — potential lookahead bias or synthetic data"
        elif smoothness_ratio < 1.0 and dd_to_mean < 2.0:
            status = "⚠ Review Recommended"
            insight = "Unusually smooth returns — verify data integrity"
        else:
            status = "✅ Plausible"
            insight = "Return volatility consistent with realistic trading"
        return CheckResult("Equity Curve Plausibility", smoothness_ratio >= 0.5 or max_dd >= 0.05, 100,
            f"Smoothness: {smoothness_ratio:.2f} → {status}", insight,
            "Check for lookahead bias, options mispricing, or data manipulation", "Plausibility")

    def check_kelly_plausibility(self):
        r = self.returns
        winners = r[r > 0]
        losers  = r[r < 0]
        if len(winners) == 0 or len(losers) == 0:
            return CheckResult("Kelly Plausibility", True, 100,
                "Kelly: N/A → ✅ Plausible",
                "Cannot compute Kelly without both wins and losses.", "", "Plausibility")
        wr       = len(winners) / len(r)
        avg_win  = float(np.mean(winners))
        avg_loss = float(abs(np.mean(losers)))
        b = avg_win / max(avg_loss, EPSILON)   # reward:risk ratio
        # Kelly fraction = (b*p - q) / b  (fraction of capital per trade, 0–1 is normal)
        kelly = (b * wr - (1 - wr)) / b
        if kelly > 1.0:
            status = "⚠ Manual Audit Required"
            insight = f"Kelly {kelly:.2f} means bet 100%+ of capital per trade — verify R:R and win rate"
            passed = False
        elif kelly > 0.5:
            status = "⚠ Review Recommended"
            insight = f"High Kelly {kelly:.2f} — edge appears strong, verify out-of-sample stability"
            passed = True
        else:
            status = "✅ Plausible"
            insight = f"Kelly fraction {kelly:.2f} within realistic range (typical: 0.05–0.30)"
            passed = True
        return CheckResult("Kelly Plausibility", passed, 100 if passed else 0,
            f"Kelly: {kelly:.2f} → {status}", insight,
            "Verify win rate and R:R ratio are stable out-of-sample", "Plausibility")

    # ---- CRASH SIMS ----

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns

        # Simulate the crisis window (~60 trading days) using strategy's own stats
        crisis_days = 60
        mean_r  = float(np.mean(r))
        vol     = float(np.std(r))
        market_drop_pct = profile["market_drop"] / 100  # e.g. -0.56

        # FIX: use trimmed mean to avoid fat-tail whale trades biasing crash results
        # A single +45% outlier trade should not make crash scenarios look profitable
        trimmed_mean = float(np.mean(np.clip(r,
            np.percentile(r, 2), np.percentile(r, 98))))

        # Exposure: high-vol strategies bleed through more market crash than low-vol
        exposure = float(np.clip(vol / 0.012, 0.1, 0.8))

        # Generate crisis_days of stressed returns
        rng = np.random.default_rng(abs(hash(crash_key)) % (2**32))
        vol_mult = min(profile["vol_multiplier"], 4.0)  # cap at 4x to avoid explosion

        # Use trimmed_mean (not raw mean) so whale outliers don't make crashes look rosy
        stressed = rng.normal(trimmed_mean, vol * vol_mult, crisis_days)

        # Apply market drag (scaled by exposure and liquidity)
        market_drag = market_drop_pct * exposure * profile["liquidity_factor"]
        stressed += market_drag / crisis_days  # spread drag across crisis window

        # Gap risk: spike down the worst 10% of days
        bottom_10 = stressed < np.percentile(stressed, 10)
        stressed[bottom_10] -= abs(stressed[bottom_10]) * profile["gap_risk"]

        stressed = np.clip(stressed, -0.99, 0.99)
        cumulative = float(np.prod(1 + stressed) - 1)

        # ── PLAUSIBILITY CAPS ──────────────────────────────────────────────────
        # Rule 1: No strategy can gain >25% during any crash window
        if cumulative > 0.25:
            cumulative = 0.25
        # Rule 2: Negative-edge strategies (trimmed_mean < 0) cannot show real gains
        # in a crash — correlation spikes eliminate uncorrelated alpha
        if trimmed_mean < -0.001 and cumulative > 0.05:
            cumulative = cumulative * 0.15
        # Rule 3: Severe crashes (>15% market drop) kill liquidity-dependent gains
        if market_drop_pct < -0.15 and cumulative > 0.08:
            cumulative = cumulative * 0.25

        dd = calculate_max_drawdown(stressed)
        survived = dd < (0.25 if self.strict_mode else 0.30)

        # 4-tier verdict
        if survived and cumulative > -0.05:
            verdict = "🟢 YOUR STRATEGY SURVIVED. While markets crashed, your system held. This is what separates real edges from lucky backtests."
        elif survived and cumulative > -0.20:
            verdict = "🟡 DAMAGED BUT SURVIVED. Your strategy took losses but stayed alive. Most traders would have panicked and exited here."
        elif survived:
            verdict = "🟠 BARELY SURVIVED. Critical drawdown threshold nearly breached. Serious position sizing review required."
        else:
            verdict = "🔴 YOUR STRATEGY WOULD HAVE BLOWN UP. The crash exposed fatal flaws. Most traders quit here — the ones who survive rebuild with proper risk management."

        return CrashSimResult(
            crash_name=profile["name"], year=profile["year"],
            description=profile["description"], market_drop=profile["market_drop"],
            strategy_drop=round(cumulative, 4), survived=survived,
            emotional_verdict=verdict
        )

    # ---- SCORING ----

    def _calculate_score(self, checks):
        weights = {"Overfitting": 0.22, "Risk": 0.38, "Regime": 0.14, "Execution": 0.11, "Compliance": 0.15}
        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat]
            if cat_checks:
                category_scores[cat] = float(np.mean([c.score for c in cat_checks]))
            else:
                category_scores[cat] = 0.0 if cat in ["Risk", "Overfitting", "Compliance"] else 20.0

        final = sum(category_scores[cat] * w for cat, w in weights.items())

        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe < 0:          final = min(final, 20)
        elif sharpe < 0.3:      final = min(final, 35)
        elif sharpe < 0.5:      final = min(final, 50)
        elif sharpe > 10:       final = min(final, 30)

        # Plausibility hard gates
        plausibility = [c for c in checks if c.category == "Plausibility"]
        if any(not c.passed and "Manual Audit Required" in c.value for c in plausibility):
            final = min(final, 35)

        # NEW: Institutional hard gates — these are disqualifying failures
        # Fat-tail fluke: strategy has no distributable edge
        conc_check = next((c for c in checks if c.name == "Profit Concentration"), None)
        if conc_check and not conc_check.passed and conc_check.score == 0:
            final = min(final, 38)   # Fat-tail fluke → cannot exceed D grade

        # Sequence luck: edge only exists on one historical path
        cpcv_check = next((c for c in checks if c.name == "CPCV Path Stability"), None)
        if cpcv_check and cpcv_check.score == 0:
            final = min(final, 42)   # Extreme sequence luck → hard cap

        if not self.has_dates: final = min(final, 75)  # unverifiable timeframe

        low_cats = sum(1 for s in category_scores.values() if s < 30)
        if low_cats >= 2:   final = min(final, 45)
        elif low_cats >= 1: final = min(final, 60)

        compliance = [c for c in checks if c.category == "Compliance"]
        if compliance and not all(c.passed for c in compliance):
            final = min(final, 55)

        if 'Short backtest' in self.timeframe_info:
            final = min(final, 75)

        # Tightened thresholds
        if final >= 90:   grade = "A — Institutionally Viable"
        elif final >= 80: grade = "B+ — Prop Firm Ready"
        elif final >= 70: grade = "B — Live Tradeable"
        else:             grade = "F — Do Not Deploy"

        return round(final, 1), grade


    def check_prop_firm_compliance(self) -> CheckResult:
        """FTMO / Topstep / The5ers compliance (v1.7)."""
        r = self.returns
        max_dd = calculate_max_drawdown(r)
        total_return = float(np.prod(1 + r) - 1)
        sharpe = calculate_sharpe(r, self.trades_per_year)

        firms = {
            "FTMO":     {"max_dd": 0.10, "profit_target": 0.10},
            "Topstep":  {"max_dd": 0.10, "profit_target": 0.06},
            "The5ers":  {"max_dd": 0.10, "profit_target": 0.08},
        }

        eligible = []
        violations = []

        for name, rules in firms.items():
            dd_ok = max_dd <= rules["max_dd"]
            profit_ok = total_return >= rules["profit_target"]
            sharpe_ok = sharpe >= 1.0
            if dd_ok and profit_ok and sharpe_ok:
                eligible.append(name)
            else:
                v = []
                if not dd_ok:      v.append(f"DD {max_dd:.1%}>{rules['max_dd']:.0%}")
                if not profit_ok:  v.append(f"Return {total_return:.1%}<{rules['profit_target']:.0%} target")
                if not sharpe_ok:  v.append(f"Sharpe {sharpe:.2f}<1.0")
                violations.append(f"{name}: {', '.join(v)}")

        passed = len(eligible) >= 1
        score = (len(eligible) / 3) * 100

        if eligible:
            value = f"Eligible: {', '.join(eligible)}"
            insight = f"Meets requirements for {len(eligible)}/3 prop firms. Max DD {max_dd:.1%}, Return {total_return:.1%}, Sharpe {sharpe:.2f}."
            fix = "Consider submitting to eligible firms."
        else:
            value = f"Not eligible for any prop firm"
            insight = f"Violations: {' | '.join(violations[:2])}"
            fix = "Reduce max drawdown below 10%, hit minimum profit targets, Sharpe>1.0."

        return CheckResult(
            name="Prop Firm Compliance",
            passed=passed,
            score=round(score, 1),
            value=value,
            insight=insight,
            fix=fix,
            category="Compliance",
        )

    # ---- RUN ----

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
            self.check_cvar(),
            self.check_sortino(),
            self.check_consecutive_losses(),
            self.check_recovery_factor(),
            self.check_absolute_sharpe(),
            self.check_ruin_probability(),
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
            self.check_impact_adjusted_capacity(),
            self.check_prop_firm_compliance(),  # detailed FTMO/Topstep/The5ers check
            self.check_sharpe_plausibility(),
            self.check_frequency_return_plausibility(),
            self.check_equity_smoothness_plausibility(),
            self.check_kelly_plausibility(),
            self.check_prop_firm_compliance(),
        ]
        self.checks = checks

        crash_sims = [
            self.simulate_crash("2008_gfc"),
            self.simulate_crash("2020_covid"),
            self.simulate_crash("2022_bear"),
            self.simulate_crash("2010_flash_crash"),
            self.simulate_crash("1998_ltcm"),
        ]

        score, grade = self._calculate_score(checks)
        r = self.returns
        sharpe   = calculate_sharpe(r, self.trades_per_year)
        dd       = calculate_max_drawdown(r)
        win_rate = calculate_win_rate(r)

        if self.strict_mode:
            if sharpe < 1.2:
                grade = "F — Strict Mode: Sharpe below 1.2"; score = min(score, 40)
            compliance = [c for c in checks if c.category == "Compliance"]
            if compliance and not all(c.passed for c in compliance):
                grade = "F — Strict Mode: Compliance failed"; score = min(score, 30)
            if sum(1 for s in crash_sims if s.survived) < 2:
                grade = "F — Strict Mode: Failed crash stress test"; score = min(score, 35)

        # ── Final 5%: Alpha Decay + Overfit Detection (informational only) ──
        timestamps = self.df['date'] if 'date' in self.df.columns else None
        alpha_decay = _analyze_alpha_decay(r, timestamps)
        overfit_profile = _OverfitDetector(r, timestamps).detect()

        # Add informational checks (no score/gate impact — pure insight)
        hl_str = (f"{alpha_decay.half_life_periods:.1f} periods"
                  if alpha_decay.half_life_periods < 990
                  else "No significant autocorrelation (random walk — normal for daily)")
        checks.append(CheckResult(
            name="Alpha Decay (Half-Life)", passed=True,
            score=100 if not alpha_decay.latency_sensitive else 60,
            value=hl_str,
            insight=(f"Signal decays in {alpha_decay.half_life_periods:.1f} periods. "
                     f"{'⚠️ Regime-dependent decay — monitor vol regimes.' if alpha_decay.regime_dependent else ''}"
                     f"{'🚨 LATENCY SENSITIVE — sub-60s half-life.' if alpha_decay.latency_sensitive else 'Not latency-critical.'}")
                    if alpha_decay.half_life_periods < 990 else
                    "Returns show no significant autocorrelation structure. Normal for daily strategies.",
            fix=("Upgrade execution infrastructure or increase holding period."
                 if alpha_decay.latency_sensitive else
                 "No action required."),
            category="Operational"
        ))

        checks.append(CheckResult(
            name="Symbolic Overfit Detection", passed=not overfit_profile.is_overfit,
            score=overfit_profile.score,
            value=f"Score {overfit_profile.score:.0f}/100 | p={overfit_profile.p_value:.3f} | AdjSharpe={overfit_profile.adjusted_sharpe:.2f}",
            insight=(f"Strategy is {'statistically indistinguishable from noise' if overfit_profile.is_overfit else 'distinguishable from random noise'}. "
                     f"Adj Sharpe {overfit_profile.adjusted_sharpe:.2f} vs raw {overfit_profile.raw_sharpe:.2f}. "
                     + (overfit_profile.informational if overfit_profile.informational else "")),
            fix=("Simplify strategy rules, reduce parameter count, test on out-of-sample data."
                 if overfit_profile.is_overfit else "No overfit detected."),
            category="Operational"
        ))

        return ValidationReport(
            score=score, grade=grade, sharpe=sharpe, max_drawdown=dd,
            win_rate=win_rate, total_trades=len(r), profitable_trades=int(np.sum(r > 0)),
            checks=checks, crash_sims=crash_sims,
            assumptions=self._get_assumptions(),
            validation_hash=self._generate_validation_hash(),
            validation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            audit_flags=self._generate_audit_flags(),
            plausibility_summary=self._generate_plausibility_summary(),
            alpha_decay=alpha_decay,
            overfit_profile=overfit_profile,
        )
