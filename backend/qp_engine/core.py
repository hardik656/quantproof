"""Shared engine pieces: constants, result dataclasses, metric helpers,
crash profiles and the overfit detector. Split out of validator.py."""

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")


def convert_numpy(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


ENGINE_VERSION = "v6.3"

METHODOLOGY_DATE = "2026-10-17"

RISK_FREE_DAILY = 0.04 / 252

EPSILON = 1e-9

MIN_TRADES_STANDARD = 50

MIN_TRADES_STRICT = 100

PARTIAL_FILL_RATE = 0.80  # methodology: 80% of orders fill, the rest are missed

COMMISSION_INSTITUTIONAL = 0.0002  # 2 bps  — prime broker

COMMISSION_RETAIL_EQUITY = 0.0010  # 10 bps — retail equity

COMMISSION_RETAIL_SPREAD = 0.0020  # 20 bps — HFT / small-cap

CPCV_N_SPLITS = 6

CPCV_N_TEST = 2

CPCV_EMBARGO_FRAC = 0.01  # 1% of dataset as embargo gap

CPCV_ROBUST_THRESHOLD = 1.5  # < this = truly robust

CPCV_ACCEPTABLE_THRESHOLD = 2.5  # < this = acceptable

ADV_DEFAULT = 50_000_000

IMPACT_ETA = 0.10  # Almgren-Chriss η

CRASH_MAX_GAIN = 0.25

CRASH_NEG_EDGE_CAP = 0.15

CRASH_EXPOSURE_MIN = 0.25  # floor for all strategies (was 0.10)

CRASH_EXPOSURE_MIN_NODATE = 0.40  # floor when no date column supplied

CRASH_AUTOCORR_THRESHOLD = 0.20  # |AR(1)| above this → +30% exposure

CRASH_AUTOCORR_MULT = 1.30  # momentum-reversal penalty multiplier

DD_DURATION_MAX_MONTHS = 6

ULCER_INDEX_THRESHOLD = 5.0

AUTOCORR_AR1_WARN = 0.20  # |AR(1)| above this → warn

AUTOCORR_AR1_FAIL = 0.30  # |AR(1)| above this → fail check

LJUNG_BOX_LAGS = [1, 5, 10]

LJUNG_BOX_ALPHA = 0.05


@dataclass
class AlphaDecayProfile:
    half_life_periods: float
    half_life_seconds: Optional[float]
    optimal_holding: int
    regime_dependent: bool
    latency_sensitive: bool
    decay_curve: List[float]
    informational_only: bool = True


@dataclass
class OverfitProfile:
    score: float
    is_overfit: bool
    p_value: float
    adjusted_sharpe: float
    raw_sharpe: float
    indicators: Dict
    informational: str = ""


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float
    value: str
    insight: str
    fix: str
    category: str
    # False for informational results (e.g. benchmark checks with no benchmark).
    # Informational results are reported but excluded from the weighted score.
    scored: bool = True


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
    engine_version: str = ENGINE_VERSION
    methodology_version: str = METHODOLOGY_DATE
    alpha_decay: Optional[AlphaDecayProfile] = None
    overfit_profile: Optional[OverfitProfile] = None
    equity_curve: Optional[list] = None
    prop_firm_compliance: Optional[dict] = None
    monte_carlo: Optional[dict] = None
    enhanced: bool = False


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
        "description": (
            "Lehman Brothers collapsed. S&P 500 lost 56%. VIX hit 80. "
            "Correlations spiked to 1 — everything fell together. "
            "Leverage unwinding forced selling in all asset classes simultaneously."
        ),
        "market_drop": -56.0,
        "vol_multiplier": 4.5,
        "liquidity_factor": 0.3,
        "gap_risk": 0.08,
        "autocorr": 0.40,  # strong loss momentum in crises
        "fat_tail_df": 3,  # t-dist degrees of freedom for tail modelling
    },
    "2020_covid": {
        "name": "2020 COVID Crash",
        "year": "Feb 2020 – Mar 2020",
        "description": (
            "Fastest 30%+ drop in market history — 34% in 33 days. "
            "Circuit breakers triggered 4 times. "
            "Then one of the fastest recoveries ever seen."
        ),
        "market_drop": -34.0,
        "vol_multiplier": 5.0,
        "liquidity_factor": 0.2,
        "gap_risk": 0.12,
        "autocorr": 0.35,
        "fat_tail_df": 2.5,
    },
    "2022_bear": {
        "name": "2022 Rate Hike Bear Market",
        "year": "Jan 2022 – Dec 2022",
        "description": (
            "Fed raised rates 425bps in 12 months. "
            "Nasdaq lost 33%, S&P lost 19%. "
            "Momentum strategies that crushed 2021 were destroyed. "
            "Growth stocks fell 60–90%."
        ),
        "market_drop": -19.4,
        "vol_multiplier": 2.5,
        "liquidity_factor": 0.7,
        "gap_risk": 0.04,
        "autocorr": 0.20,
        "fat_tail_df": 5,
    },
    "2010_flash_crash": {
        "name": "2010 Flash Crash",
        "year": "May 6, 2010",
        "description": (
            "Dow dropped 1000 points in minutes from algorithmic cascade. "
            "Liquidity vanished — many stocks traded at $0.01. "
            "HFT strategies blown out in seconds."
        ),
        "market_drop": -9.0,
        "vol_multiplier": 5.0,
        "liquidity_factor": 0.05,
        "gap_risk": 0.15,
        "autocorr": 0.50,  # extreme autocorr in intraday crash
        "fat_tail_df": 2,
    },
    "1998_ltcm": {
        "name": "1998 LTCM / Russia Default",
        "year": "Aug–Sep 1998",
        "description": (
            "Russia defaulted; LTCM collapsed under hidden leverage. "
            "Correlation assumptions broke simultaneously across all pairs. "
            "Strategies with similar factor exposures were destroyed together."
        ),
        "market_drop": -19.0,
        "vol_multiplier": 3.0,
        "liquidity_factor": 0.25,
        "gap_risk": 0.08,
        "autocorr": 0.30,
        "fat_tail_df": 3,
    },
}


def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 252) -> float:
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    mean = np.mean(returns)
    if std < EPSILON:
        # Zero variance — constant returns
        if mean > EPSILON:
            return 10.0  # effectively infinite positive Sharpe
        if mean < -EPSILON:
            return -10.0  # effectively infinite negative Sharpe
        return 0.0
    return float(mean / std * np.sqrt(trades_per_year))


def calculate_sharpe_ci(
    returns: np.ndarray, trades_per_year: float = 252
) -> Tuple[float, float, float, float]:
    n = len(returns)
    if n < 10:
        s = calculate_sharpe(returns, trades_per_year)
        return s, s - 1.0, s + 1.0, 0.5

    excess = returns - RISK_FREE_DAILY
    std_e = np.std(excess, ddof=1)
    if std_e < EPSILON:
        s = calculate_sharpe(returns, trades_per_year)
        return s, s, s, 0.0

    sr_d = np.mean(excess) / std_e
    skew = float(stats.skew(excess))
    kurt = float(stats.kurtosis(excess, fisher=False))

    var_sr_d = (1 + 0.5 * sr_d**2 - skew * sr_d + (kurt - 3) / 4 * sr_d**2) / n
    se_sr_d = max(np.sqrt(max(var_sr_d, EPSILON)), 0.001)

    sharpe_ann = float(np.clip(sr_d * np.sqrt(trades_per_year), -15, 20))
    se_ann = se_sr_d * np.sqrt(trades_per_year)

    p_val = float(2 * (1 - stats.norm.cdf(abs(sr_d / se_sr_d))))
    return sharpe_ann, sharpe_ann - 1.96 * se_ann, sharpe_ann + 1.96 * se_ann, p_val


def calculate_sortino(returns: np.ndarray, trades_per_year: float = 252, mar: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - mar
    downside_sq = np.minimum(excess, 0.0) ** 2
    downside_dev = np.sqrt(np.mean(downside_sq))
    if downside_dev < EPSILON:
        mean_r = np.mean(returns)
        return 5.0 if mean_r > mar else 0.0
    return float(np.mean(returns) / downside_dev * np.sqrt(trades_per_year))


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95, seed: int = 42) -> float:
    if len(returns) == 0:
        return 0.0
    alpha = 1 - confidence
    if len(returns) < 100:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(returns)
            samples = kde.resample(10000, seed=seed)[0]
            var = np.percentile(samples, alpha * 100)
            tail = samples[samples <= var]
            return float(np.mean(tail)) if len(tail) > 0 else float(var)
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
    drawdown = (running_max - cumulative) / (running_max + EPSILON)
    return float(np.max(drawdown))


def calculate_win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    nonzero = returns[returns != 0.0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.mean(nonzero > 0) * 100)


def calculate_calmar(returns: np.ndarray, trades_per_year: float = 252) -> float:
    if len(returns) == 0:
        return 0.0
    total_return = float(np.prod(1 + returns))
    years = len(returns) / trades_per_year
    try:
        exponent = 1 / max(years, EPSILON)
        if exponent > 100:
            # exponent too large — years near zero
            # (no date data). Cap it.
            exponent = 100
        cagr = total_return**exponent - 1
    except (OverflowError, FloatingPointError):
        cagr = total_return - 1
    dd = abs(calculate_max_drawdown(returns))
    if dd < EPSILON:
        return 10.0 if cagr > 0 else 0.0
    return float(cagr / dd)


def calculate_ruin_probability(
    win_rate: float, avg_win: float, avg_loss: float, capital_units: int = 50
) -> float:
    if win_rate <= 0 or avg_loss < EPSILON:
        return 1.0
    if win_rate >= 1.0:
        return 0.0
    edge = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    if edge <= 0:
        return 1.0
    p = float(win_rate)
    q = 1.0 - p
    rr_disadvantage = (q / p) * (avg_loss / avg_win)
    if rr_disadvantage >= 1.0:
        return 1.0
    ruin = rr_disadvantage**capital_units
    return float(np.clip(ruin, 0.0, 1.0))


def calculate_deflated_sharpe(
    returns: np.ndarray, sharpe_ann: float, n_trials: int = 1, trades_per_year: float = 252
) -> dict:
    n = len(returns)
    if n < 10:
        return {
            "dsr": 0.5,
            "sr_benchmark": 0.0,
            "is_significant": False,
            "n_trials": n_trials,
            "note": "Insufficient data",
        }

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))

    sr_d = np.mean(returns) / (np.std(returns, ddof=1) + EPSILON)

    var_sr_d = (1 + 0.5 * sr_d**2 - skew * sr_d + (kurt - 3) / 4 * sr_d**2) / n
    se_sr_d = max(np.sqrt(max(var_sr_d, EPSILON)), 1e-4)

    se_sr_ann = se_sr_d * np.sqrt(trades_per_year)
    se_sr_ann = max(se_sr_ann, 0.01)

    if n_trials > 1:
        gamma_em = 0.5772156649
        sr_benchmark = (
            (1 - gamma_em) * stats.norm.ppf(1 - 1.0 / n_trials)
            + gamma_em * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        ) * se_sr_ann
    else:
        sr_benchmark = 0.0
    sr_benchmark = max(sr_benchmark, 0.0)

    dsr = float(stats.norm.cdf((sharpe_ann - sr_benchmark) / se_sr_ann))
    sr_min95 = sr_benchmark + 1.645 * se_sr_ann

    # Calculate institutional benchmark (n_trials=20)
    institutional_n_trials = 20
    gamma_em = 0.5772156649
    sr_benchmark_institutional = (
        (1 - gamma_em) * stats.norm.ppf(1 - 1.0 / institutional_n_trials)
        + gamma_em * stats.norm.ppf(1 - 1.0 / (institutional_n_trials * np.e))
    ) * se_sr_ann
    sr_benchmark_institutional = max(sr_benchmark_institutional, 0.0)
    dsr_institutional = float(stats.norm.cdf((sharpe_ann - sr_benchmark_institutional) / se_sr_ann))

    # Add bias interpretation label
    if n_trials <= 2:
        bias_label = "Optimistic (low selection bias adjustment)"
    elif n_trials <= 10:
        bias_label = "Moderate adjustment"
    else:
        bias_label = "Strict (institutional standard)"

    return {
        "dsr": round(dsr, 4),
        "dsr_institutional": round(dsr_institutional, 4),
        "sr_benchmark": round(sr_benchmark, 4),
        "sr_min_95": round(sr_min95, 4),
        "se_sr": round(se_sr_ann, 4),
        "is_significant": dsr > 0.95,
        "n_trials": n_trials,
        "bias_label": bias_label,
        "skew": round(skew, 3),
        "excess_kurt": round(kurt - 3, 3),
    }


def calculate_drawdown_duration(returns: np.ndarray, has_dates: bool = False, dates=None) -> dict:
    if len(returns) == 0:
        return {
            "max_dd_duration": 0,
            "recovery_periods": None,
            "ever_recovered": False,
            "dd_duration_days": None,
            "recovery_days": None,
        }

    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    underwater = equity < running_max

    max_duration = current = 0
    for u in underwater:
        if u:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    dd_series = (running_max - equity) / (running_max + EPSILON)
    peak_dd_idx = int(np.argmax(dd_series))
    peak_equity_before = running_max[peak_dd_idx]

    recovery_periods = None
    ever_recovered = False
    for i in range(peak_dd_idx + 1, len(equity)):
        if equity[i] >= peak_equity_before:
            recovery_periods = i - peak_dd_idx
            ever_recovered = True
            break

    dd_duration_days = recovery_days = None
    if has_dates and dates is not None and len(dates) > 1:
        try:
            total_days = (dates.iloc[-1] - dates.iloc[0]).days
            ppd = len(returns) / max(total_days, 1)
            if max_duration > 0:
                dd_duration_days = round(max_duration / ppd)
            if recovery_periods is not None:
                recovery_days = round(recovery_periods / ppd)
        except Exception:
            pass

    return {
        "max_dd_duration": max_duration,
        "recovery_periods": recovery_periods,
        "ever_recovered": ever_recovered,
        "dd_duration_days": dd_duration_days,
        "recovery_days": recovery_days,
    }


def calculate_ulcer_index(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    dd_pct = (running_max - equity) / (running_max + EPSILON) * 100
    return float(np.sqrt(np.mean(dd_pct**2)))


def calculate_harvey_liu_zhu_min_obs(sharpe_target: float, n_trials: int = 1, alpha: float = 0.05) -> int:
    if sharpe_target <= 0:
        return 9999
    alpha_adj = alpha / max(n_trials, 1)
    z = stats.norm.ppf(1 - alpha_adj / 2)
    variance_factor = 1 + 0.5 * sharpe_target**2
    min_n = int(np.ceil(z**2 * variance_factor / sharpe_target**2))
    return max(min_n, 30)


def _ljung_box_test(returns: np.ndarray, lags: List[int]) -> dict:
    """
    Compute Ljung-Box Q-statistic at specified lags.
    Returns dict of {lag: (Q_stat, p_value)}.
    Reference: Ljung & Box (1978), Biometrika 65(2):297-303.
    """
    n = len(returns)
    results = {}
    for lag in lags:
        if lag >= n:
            continue
        # Autocorrelations at lags 1..lag
        acf_vals = []
        for k in range(1, lag + 1):
            r_lag = np.corrcoef(returns[:-k], returns[k:])[0, 1]
            acf_vals.append(0.0 if np.isnan(r_lag) else r_lag)
        # Q = n*(n+2) * sum( rho_k^2 / (n-k) )
        q_stat = float(n * (n + 2) * sum(rk**2 / max(n - k, 1) for k, rk in enumerate(acf_vals, start=1)))
        p_val = float(1 - stats.chi2.cdf(q_stat, df=lag))
        results[lag] = (round(q_stat, 4), round(p_val, 4))
    return results


def _newey_west_se_sharpe(returns: np.ndarray, trades_per_year: float, max_lag: int = 10) -> float:
    """
    Newey-West HAC standard error for the annualised Sharpe ratio.
    Accounts for serial correlation in returns — critical when AR(1) is
    large (≥0.20) because the IID variance formula underestimates SE.

    Reference: Newey & West (1987), Econometrica 55(3):703-708.
              Lo (2002) — Sharpe ratio under serial correlation.
    """
    n = len(returns)
    if n < 20:
        return float(np.sqrt(1.0 / n) * np.sqrt(trades_per_year))

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < EPSILON:
        return 0.0

    # Demeaned, standardised returns
    z = (returns - mean_r) / std_r

    # HAC variance of z using Bartlett kernel
    gamma0 = float(np.mean(z**2))
    hac_var = gamma0
    for k in range(1, min(max_lag + 1, n)):
        bartlett_w = 1.0 - k / (max_lag + 1)
        gamma_k = float(np.mean(z[k:] * z[:-k]))
        hac_var += 2.0 * bartlett_w * gamma_k

    hac_var = max(hac_var, EPSILON)
    # SE of daily SR = sqrt(HAC_var / n); annualise by sqrt(tpy)
    se_ann = float(np.sqrt(hac_var / n) * np.sqrt(trades_per_year))
    return max(se_ann, 0.01)


def _analyze_alpha_decay(returns: np.ndarray, timestamps=None, max_lags: int = 40) -> "AlphaDecayProfile":
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

    half_life = float("inf")
    rho0_est = autocorrs[0] if len(autocorrs) > 0 else 0.0
    if abs(rho0_est) > 0.10 and len(autocorrs) > 5:
        try:
            popt, _ = curve_fit(
                exp_decay, lags, autocorrs, p0=[rho0_est, 0.1], bounds=([-1, 0.001], [1, 5]), maxfev=5000
            )
            rho0_fit, lam_fit = popt
            if lam_fit > 0:
                half_life = np.log(2) / lam_fit
        except Exception:
            pass

    hl_periods = half_life if half_life != float("inf") else 999.0
    optimal_hold = max(1, int(hl_periods))

    hl_seconds = None
    latency_sensitive = False
    if timestamps is not None and len(timestamps) > 1:
        try:
            intervals = np.diff(timestamps).astype("timedelta64[s]").astype(float)
            median_interval = float(np.median(intervals))
            if median_interval > 0 and hl_periods < 999:
                hl_seconds = hl_periods * median_interval
                latency_sensitive = hl_seconds < 60
        except Exception:
            pass

    regime_dependent = False
    vol_series = pd.Series(returns).rolling(20).std().values
    if np.sum(~np.isnan(vol_series)) > 50:
        high_mask = vol_series > np.nanpercentile(vol_series, 75)
        low_mask = vol_series < np.nanpercentile(vol_series, 25)
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


def _stable_seed(data: bytes) -> int:
    """Process-independent 32-bit seed. Python's hash() of bytes/str is salted per
    process (PYTHONHASHSEED), so it made crash sims and stress tests differ between
    server restarts for the same upload."""
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


class _OverfitDetector:
    def __init__(self, returns: np.ndarray, timestamps=None):
        self.returns = returns
        self.n = len(returns)
        self.timestamps = timestamps
        _seed = _stable_seed(self.returns.tobytes())
        self._rng = np.random.default_rng(_seed)

    def _noise_baselines(self, n: int = 50) -> List[np.ndarray]:
        vol = np.std(self.returns, ddof=1)
        b = []
        for _ in range(n // 4):
            b.append(self._rng.normal(0, vol, self.n))
        for _ in range(n // 4):
            x = self._rng.normal(0, vol, self.n)
            b.append(np.convolve(x, np.ones(3) / 3, "same") * 0.5 + x * 0.5)
        for _ in range(n // 4):
            x = self._rng.normal(0, vol, self.n)
            b.append(-np.diff(x, prepend=x[0]) * 0.3 + x * 0.7)
        for _ in range(n // 4):
            rs = max(10, self.n // 50)
            reg = np.repeat(self._rng.choice([-1, 1], rs), (self.n + rs - 1) // rs)[: self.n]
            b.append(self._rng.normal(0, vol, self.n) * (1 + reg * 0.5))
        return b

    def _metrics(self, r: np.ndarray) -> dict:
        cum = np.cumprod(1 + r)
        d1 = np.diff(cum)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        hist, _ = np.histogram(r, bins=20, density=True)
        return {
            "smoothness": float(np.var(d2) / (np.var(d1) + EPSILON)),
            "entropy": float(stats.entropy(hist + EPSILON)),
            "kurtosis": float(stats.kurtosis(r, fisher=False)),
        }

    def detect(self) -> "OverfitProfile":
        baselines = self._noise_baselines(50)
        strat_m = self._metrics(self.returns)
        base_m = [self._metrics(b) for b in baselines]

        z = {}
        for key in strat_m:
            vals = [m[key] for m in base_m]
            z[key] = (strat_m[key] - np.mean(vals)) / (np.std(vals) + EPSILON)

        indicators = {
            "too_smooth": z.get("smoothness", 0) < -2.5,
            "low_entropy": z.get("entropy", 0) < -2.5,
        }

        t = np.arange(self.n)
        X = np.column_stack(
            [
                t,
                t**2,
                pd.Series(self.returns).rolling(5).mean().fillna(0).values,
                np.concatenate([[0], self.returns[:-1]]),
            ]
        )
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = poly.fit_transform(X)
        model = Ridge(alpha=1.0)
        model.fit(Xp, self.returns)
        r2 = r2_score(self.returns, model.predict(Xp))
        comp = Xp.shape[1]

        base_r2s = []
        for b in baselines[:20]:
            Xb = poly.transform(
                np.column_stack(
                    [t, t**2, pd.Series(b).rolling(5).mean().fillna(0).values, np.concatenate([[0], b[:-1]])]
                )
            )
            m2 = Ridge(alpha=1.0)
            m2.fit(Xb, b)
            base_r2s.append(r2_score(b, m2.predict(Xb)))

        indicators["complex_overfit"] = r2 > np.percentile(base_r2s, 92)

        overfit_score = max(0, 100 - sum(indicators.values()) * 25)

        raw_sharpe = float(np.mean(self.returns) / (np.std(self.returns, ddof=1) + EPSILON))
        adj_sharpe = raw_sharpe * (1 - 0.005 * comp)
        base_sharpes = [np.mean(b) / (np.std(b, ddof=1) + EPSILON) for b in baselines]
        p_value = float(np.mean(np.array(base_sharpes) > adj_sharpe))

        is_overfit = (overfit_score < 60) and (p_value > 0.10)

        info = ""
        if is_overfit:
            active = [k for k, v in indicators.items() if v]
            info = "Overfit flags: " + ", ".join(active)

        return OverfitProfile(
            score=overfit_score,
            is_overfit=is_overfit,
            p_value=p_value,
            adjusted_sharpe=adj_sharpe,
            raw_sharpe=raw_sharpe,
            indicators=indicators,
            informational=info,
        )
