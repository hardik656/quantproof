"""
QuantProof — Validation Engine v6.3

v6.3 (2026-10-17): run() now executes every check the methodology page describes:
37 scored checks (Overfitting 11, Risk 11, Regime 6, Execution 6, Compliance 3) plus
4 plausibility checks (reported, capped, not weighted). Benchmark checks are
informational (scored=False) when no aligned benchmark is available. Partial fills
are simulated on the strategy's own returns instead of a constant.
Institutional Edition — Full Audit Correction

PATCH v6.2 — ADVERSARIAL AUDIT FIXES (delta from v6.1)
═══════════════════════════════════════════════════════

F1. Market-Neutral Benchmark Test added.
    QuantProofValidator now accepts an optional benchmark_returns array.
    When provided, two new checks run in the Overfitting category:
      - check_information_ratio(): computes beta, Jensen's alpha,
        tracking error, and information ratio vs the benchmark.
        Flags beta-capture disguised as alpha (IR < 0.3 = low alpha).
      - check_beta_exposure(): warns when strategy beta > 0.6,
        indicating high correlation to market drift.
    Both checks are informational (score=50, no cap) when no benchmark
    is supplied, so existing usage is fully backward-compatible.
    Fixes audit case C12: random walk with positive drift during bull
    period scored 55 with zero alpha — now flagged explicitly.

F3. Crash simulation exposure model hardened.
    Previous: exposure = clip(vol / 0.012, 0.10, 0.80) — gamed by
    low-vol or undated strategies; all 5 crash sims showed "survived".
    New model:
      - Undated strategies: exposure floor = 0.40 (was 0.10).
        Crash result labels prefixed with [UNCALIBRATED] to prevent
        misleading green signals on dateless submissions.
      - AR(1) autocorrelation adjustment: if |AR(1)| >= 0.20, multiply
        exposure by 1.30 (momentum-reversal penalty in crises).
      - Hard minimum exposure = 0.25 for all strategies (was 0.10).
    Crash simulations remain illustrative only — they do not affect
    prop firm compliance scoring (by design).

PATCH v6.1 — ADVERSARIAL AUDIT FIXES (delta from v6.0)
══════════════════════════════════════════════════════════════════

P1. Score cap at 50 when Sharpe > 5 AND no date column present.
P2. Ljung-Box serial correlation test + Newey-West HAC correction.
P3. Prop Firm Eligible suppressed when Manual Audit Required active.

ORIGINAL CHANGES FROM v3.0 (complete audit log):

══════════════════════════════════════════════════════════════════════════════

CRITICAL BUG FIXES
════════════════════════════════════════════════════════════════════

C1. Sharpe: ddof=0→1 (population→sample std). All derivatives corrected.

C2. Removed arbitrary 0.85 Sharpe multiplier. Was applied inconsistently
    (missing from CPCV fold Sharpes, ruin probability, etc.) and has no
    academic basis. Now using pure textbook SR = mean(r)/std(r,ddof=1)*sqrt(T).

C3. Sortino denominator fixed. Was std(negative_returns_only). Correct
    formula: sqrt(mean(min(r, MAR)^2)) using ALL returns below MAR=0.
    Reference: Sortino & van der Meer (1991).

C4. Ruin probability formula replaced. Was using non-standard Kelly transform
    that produced near-zero values for winning strategies (ruin≈0.0001 for
    win_rate=0.55). Now uses correct gambler's ruin: P=((1-p)/p)^N.

C5. DSR standard error formula corrected. Now uses per-trade SR (mean/std)
    directly, not annualized SR divided back by sqrt(tpy). The 0.85 factor
    was causing SE to be 15% too small, inflating DSR significance.

C6. CPCV purge logic rewritten from scratch. Previous version had both
    purge_start and purge_end set to the same boolean, so right-side folds
    were never purged. Full bidirectional purge+embargo now implemented.

C7. _detect_timeframe: removed cap at 252 for HFT. Was: min(n/years, 252).
    Now: n/years with no cap. HFT strategies with 10k+ trades/year are now
    correctly annualized.

C8. Slippage/commission impact: denominator changed from abs(sum(r)) to
    sum(abs(r)) (gross turnover). Previous formula blew up to infinity for
    zero-sum or losing strategies.

══════════════════════════════════════════════════════════════════
IMPORTANT FIXES
══════════════════════════════════════════════════════════════════

I1. Sharpe CI: 0.85 factor (removed) was applied to center but not SE width.
    Now both center and SE use consistent ddof=1 formula.

I2. calculate_win_rate: removed |r|<0.001 filter that reversed win rate
    for HFT strategies with tiny-but-valid trades. Now counts all non-zero trades.

I3. calculate_calmar: changed to geometric annualization (CAGR) instead of
    arithmetic mean × tpy. Institutional standard.

I4. check_outlier_dependency: now also checks loss concentration (bottom 1%),
    not just profit concentration.

I5. Capacity model: uses strategy's actual volatility (not hardcoded 0.015)
    and turnover-adjusted trade size.

I6. DSR n_trials: added n_trials parameter to QuantProofValidator constructor
    so users can specify how many strategies they tested.

I7. CVaR KDE seed: now uses validator's self.seed, not hardcoded 42.

I8. Harvey-Liu-Zhu docstring: clarified this is Bonferroni-adjusted minimum-n,
    not the full HLZ Bayesian procedure.

I9. Crash sim: now stochastic per-strategy using combined strategy+crash seed
    to maintain reproducibility while adding proper MC variability.

I10. check_commission_drag denominator: sum(abs(r)) for gross turnover.

══════════════════════════════════════════════════════════════════
NEW CHECKS ADDED
══════════════════════════════════════════════════════════════════

N1. Tail Risk Concentration check (loss-side mirror of profit concentration).
N2. White's Reality Check (simplified bootstrap version).
N3. Parameter sensitivity gate: n_trials now flows through to DSR properly.
N4. CPCV threshold calibrated empirically (not arbitrary 2.0).

══════════════════════════════════════════════════════════════════
PATCH v6.1 — ADVERSARIAL AUDIT FIXES
══════════════════════════════════════════════════════════════════

P1. Score cap at 50 when Sharpe > 5 AND no date column present.
    A no-date, high-Sharpe strategy cannot be regime-checked and is
    therefore unverifiable — misleading to score above 50.

P2. Ljung-Box autocorrelation test + Newey-West corrected SE(SR).
    Returns with AR(1) ≥ 0.3 invalidate IID-assuming tests (DSR, CI,
    Bootstrap). New check_autocorrelation() warns and corrects SE(SR).
    Ljung-Box Q-statistic at lags 1, 5, 10 — any p < 0.05 triggers fail.

P3. Prop Firm Eligible gate: "Prop Firm Eligible" label suppressed when
    any Plausibility check carries a "Manual Audit Required" flag.

══════════════════════════════════════════════════════════════════
ARCHITECTURE
══════════════════════════════════════════════════════════════════

- All randomness flows through self.rng (seeded deterministically).
- All named constants in one block at top.
- Type hints throughout.
- No global mutable state.
"""

import hashlib
import json
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from qp_engine.compliance import ComplianceChecks

# Engine names are re-exported here: routers, tests and prop_firm_compliance import them from validator.
from qp_engine.core import (  # noqa: F401
    ADV_DEFAULT,  # noqa: F401
    AUTOCORR_AR1_FAIL,  # noqa: F401
    AUTOCORR_AR1_WARN,  # noqa: F401
    COMMISSION_INSTITUTIONAL,  # noqa: F401
    COMMISSION_RETAIL_EQUITY,  # noqa: F401
    COMMISSION_RETAIL_SPREAD,  # noqa: F401
    CPCV_ACCEPTABLE_THRESHOLD,  # noqa: F401
    CPCV_EMBARGO_FRAC,  # noqa: F401
    CPCV_N_SPLITS,  # noqa: F401
    CPCV_N_TEST,  # noqa: F401
    CPCV_ROBUST_THRESHOLD,  # noqa: F401
    CRASH_AUTOCORR_MULT,  # noqa: F401
    CRASH_AUTOCORR_THRESHOLD,  # noqa: F401
    CRASH_EXPOSURE_MIN,  # noqa: F401
    CRASH_EXPOSURE_MIN_NODATE,  # noqa: F401
    CRASH_MAX_GAIN,  # noqa: F401
    CRASH_NEG_EDGE_CAP,  # noqa: F401
    CRASH_PROFILES,  # noqa: F401
    DD_DURATION_MAX_MONTHS,  # noqa: F401
    ENGINE_VERSION,  # noqa: F401
    EPSILON,  # noqa: F401
    IMPACT_ETA,  # noqa: F401
    LJUNG_BOX_ALPHA,  # noqa: F401
    LJUNG_BOX_LAGS,  # noqa: F401
    METHODOLOGY_DATE,  # noqa: F401
    MIN_TRADES_STANDARD,  # noqa: F401
    MIN_TRADES_STRICT,  # noqa: F401
    PARTIAL_FILL_RATE,  # noqa: F401
    RISK_FREE_DAILY,  # noqa: F401
    ULCER_INDEX_THRESHOLD,  # noqa: F401
    AlphaDecayProfile,  # noqa: F401
    CheckResult,  # noqa: F401
    CrashSimResult,  # noqa: F401
    OverfitProfile,  # noqa: F401
    ValidationDashboard,  # noqa: F401
    ValidationReport,  # noqa: F401
    _analyze_alpha_decay,  # noqa: F401
    _ljung_box_test,  # noqa: F401
    _newey_west_se_sharpe,  # noqa: F401
    _OverfitDetector,  # noqa: F401
    _stable_seed,  # noqa: F401
    calculate_calmar,  # noqa: F401
    calculate_cvar,  # noqa: F401
    calculate_deflated_sharpe,  # noqa: F401
    calculate_drawdown_duration,  # noqa: F401
    calculate_harvey_liu_zhu_min_obs,  # noqa: F401
    calculate_max_drawdown,  # noqa: F401
    calculate_ruin_probability,  # noqa: F401
    calculate_sharpe,  # noqa: F401
    calculate_sharpe_ci,  # noqa: F401
    calculate_sortino,  # noqa: F401
    calculate_ulcer_index,  # noqa: F401
    calculate_win_rate,  # noqa: F401
    convert_numpy,  # noqa: F401
)
from qp_engine.execution import ExecutionChecks
from qp_engine.overfitting import OverfittingChecks
from qp_engine.plausibility import PlausibilityChecks
from qp_engine.regime import RegimeChecks
from qp_engine.risk import RiskChecks

try:
    from prop_firm_compliance import PropFirmComplianceEngine, PropFirmRules, QuantProofValidatorWithPropFirms

    print("✅ PropFirmComplianceEngine import successful")
except ImportError as e:
    # Fallback if prop_firm_compliance.py doesn't exist yet
    print(f"❌ PropFirmComplianceEngine ImportError: {e}")
    PropFirmComplianceEngine = None
    PropFirmRules = None
    QuantProofValidatorWithPropFirms = None
except Exception as e:
    # Handle any other import errors
    print(f"❌ PropFirmComplianceEngine Exception: {e}")
    import traceback

    traceback.print_exc()
    PropFirmComplianceEngine = None
    PropFirmRules = None
    QuantProofValidatorWithPropFirms = None


class QuantProofValidator(
    OverfittingChecks,
    RiskChecks,
    RegimeChecks,
    ExecutionChecks,
    ComplianceChecks,
    PlausibilityChecks,
):
    def __init__(
        self,
        df: pd.DataFrame,
        strict_mode: bool = False,
        seed: int = 42,
        n_trials: int = 20,
        benchmark_type: str = "spy",
        benchmark_returns: Optional[np.ndarray] = None,
        enable_prop_firm: bool = False,
    ):
        self.strict_mode = strict_mode
        self.seed = seed
        # Accept user input directly with minimal validation (>=1 only)
        try:
            self.n_trials = max(1, int(n_trials))
        except (ValueError, TypeError):
            self.n_trials = 1
        self.rng = np.random.default_rng(seed)
        self.benchmark_type = benchmark_type
        self.benchmark_returns = (
            None if benchmark_returns is None else np.asarray(benchmark_returns, dtype=float)
        )
        self.df = self._clean(df)
        self.trades_per_year, self.timeframe_info = self._detect_timeframe()
        self.has_dates = "date" in self.df.columns
        self.returns = self._get_returns()
        self.enable_prop_firm = enable_prop_firm

        # Initialize prop firm compliance engine if enabled
        if self.enable_prop_firm:
            try:
                if PropFirmComplianceEngine is None:
                    print("❌ PropFirmComplianceEngine is None - cannot initialize")
                    self.prop_firm_engine = None
                else:
                    print("🔍 Initializing PropFirmComplianceEngine...")
                    self.prop_firm_engine = PropFirmComplianceEngine(self)
                    print("✅ PropFirmComplianceEngine initialized successfully")
            except Exception as e:
                print(f"Failed to initialize prop firm engine: {e}")
                import traceback

                traceback.print_exc()
                self.prop_firm_engine = None
        else:
            self.prop_firm_engine = None

        # Detect likely scale error (user submitted % not decimal)
        clipped_pct = float(np.mean(np.abs(self.returns) >= 0.499))
        self._scale_warning = clipped_pct > 0.50

        n = len(self.returns)
        min_required = MIN_TRADES_STRICT if strict_mode else MIN_TRADES_STANDARD

        if n < 10:
            raise ValueError(f"Insufficient data: {n} trades. Minimum 10 trades required.")
        self._low_sample_warning = n < min_required
        self._n_trades = n

        if strict_mode:
            self._validate_strict_requirements()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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

    def _detect_timeframe(self) -> Tuple[float, str]:
        if "date" not in self.df.columns:
            return 252.0, "No timestamp data — using daily (252/yr) metrics"
        delta = self.df["date"].max() - self.df["date"].min()
        time_span_years = delta.total_seconds() / (365.25 * 24 * 3600)
        if time_span_years < EPSILON:
            return 252.0, "Insufficient time span — default annualization"
        trades_per_year = len(self.df) / time_span_years
        if time_span_years < 0.25:
            desc = f"Short backtest ({time_span_years * 12:.1f} months) — Sharpe may be inflated ({trades_per_year:.0f} trades/yr)"
        elif trades_per_year > 10_000:
            desc = f"High-frequency ({trades_per_year:.0f} trades/yr)"
        elif trades_per_year > 1_000:
            desc = f"Intraday ({trades_per_year:.0f} trades/yr)"
        elif trades_per_year > 250:
            desc = f"Active trading ({trades_per_year:.0f} trades/yr)"
        else:
            desc = f"Position trading ({trades_per_year:.0f} trades/yr)"
        return trades_per_year, desc

    def _validate_strict_requirements(self):
        if "date" in self.df.columns:
            span = (self.df["date"].max() - self.df["date"].min()).days / 365.25
            if span < 0.5:
                raise ValueError(f"Strict mode requires 6 months minimum (got {span * 12:.1f} months)")
        if len(self.returns) < MIN_TRADES_STRICT:
            raise ValueError(f"Strict mode requires {MIN_TRADES_STRICT}+ trades (got {len(self.returns)})")

    def _get_returns(self) -> np.ndarray:
        r = self.df["pnl"].values.astype(float)
        max_abs = np.abs(r).max()
        if max_abs > 1.0:
            raise ValueError(f"Returns appear to be dollar amounts (max |r|={max_abs:.2f}).")
        return r

    def _generate_validation_hash(self) -> str:
        try:
            # Engine source = this file plus the qp_engine package it was split into.
            import glob
            import os

            files = [__file__] + sorted(
                glob.glob(os.path.join(os.path.dirname(__file__), "qp_engine", "*.py"))
            )
            src = "".join(open(f, encoding="utf-8").read() for f in files)
            src_hash = hashlib.sha256(src.encode()).hexdigest()[:8]
        except Exception:
            src_hash = "unknown"
        hash_data = {
            "returns": np.round(self.returns, 8).tolist(),
            "timeframe": self.timeframe_info,
            "trades_per_year": float(self.trades_per_year),
            "engine_version": ENGINE_VERSION,
            "source_fingerprint": src_hash,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "enhanced": self.enable_prop_firm,  # Include enhanced mode in cache key
            "prop_firm_engine": self.prop_firm_engine is not None,  # Include engine status
        }
        return hashlib.sha256(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()[:12]

    def _get_assumptions(self) -> List[str]:
        base = [
            "Assumes full capital deployed per trade",
            "Assumes trades are sequential and non-overlapping",
            "Assumes fractional returns (not dollar P&L)",
            f"Engine {ENGINE_VERSION} — methodology date {METHODOLOGY_DATE}",
        ]
        if self._low_sample_warning:
            base += [f"⚠ Low sample: {self._n_trades} trades"]
        return base

    def _generate_audit_flags(self) -> List[str]:
        return [
            f"⚠ {c.name}: {c.insight}"
            for c in self.checks
            if c.category == "Plausibility" and not c.passed and "Manual Audit Required" in c.value
        ]

    def _generate_plausibility_summary(self) -> str:
        pc = [c for c in self.checks if c.category == "Plausibility"]
        manual = sum(1 for c in pc if not c.passed and "Manual Audit Required" in c.value)
        review = sum(1 for c in pc if "Review Recommended" in c.value)
        if manual > 0:
            return f"⚠ {manual} statistical implausibility issues require manual audit"
        if review > 0:
            return f"⚠ {review} statistical issues merit review"
        return "✅ All statistical metrics appear plausible"

    def _classify_market_regime_by_date(self) -> Optional[dict]:
        if not self.has_dates or "date" not in self.df.columns:
            return None
        spy_bear_periods = [
            (pd.Timestamp("2000-03-01"), pd.Timestamp("2002-10-31")),
            (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31")),
            (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-31")),
            (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),
        ]
        spy_bull_periods = [
            (pd.Timestamp("2003-03-01"), pd.Timestamp("2007-09-30")),
            (pd.Timestamp("2009-04-01"), pd.Timestamp("2020-01-31")),
            (pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31")),
            (pd.Timestamp("2023-01-01"), pd.Timestamp("2099-12-31")),
        ]
        nifty_bear_periods = [
            (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-31")),
            (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
            (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-03-31")),
        ]
        nifty_bull_periods = [
            (pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31")),
            (pd.Timestamp("2022-07-01"), pd.Timestamp("2024-09-30")),
            (pd.Timestamp("2025-04-01"), pd.Timestamp("2099-12-31")),
        ]
        if self.benchmark_type == "nifty":
            bear_periods = nifty_bear_periods
            bull_periods = nifty_bull_periods
        else:
            bear_periods = spy_bear_periods
            bull_periods = spy_bull_periods
        labels = {}
        for idx, row in self.df.iterrows():
            d = row["date"]
            if pd.isna(d):
                labels[idx] = "UNKNOWN"
                continue
            # Strip timezone info for comparison — regime periods are tz-naive
            try:
                if hasattr(d, "tzinfo") and d.tzinfo is not None:
                    d = d.tz_convert("UTC").tz_localize(None)
                d = pd.Timestamp(d)
            except Exception:
                labels[idx] = "UNKNOWN"
                continue
            if any(s <= d <= e for s, e in bear_periods):
                labels[idx] = "BEAR"
            elif any(s <= d <= e for s, e in bull_periods):
                labels[idx] = "BULL"
            else:
                labels[idx] = "CONSOL"
        return labels

    # ── P2: NEW CHECK — Autocorrelation (Ljung-Box + Newey-West corrected SR) ──

    # ── F1: Market-Neutral Benchmark Checks ──────────────────────────────────

    def _benchmark_usable(self) -> bool:
        """Benchmark must exist, be aligned 1:1 with trades, and not be constant."""
        bm = self.benchmark_returns
        if bm is None:
            return False
        bm = np.asarray(bm, dtype=float)
        if bm.shape != np.shape(self.returns) or not np.all(np.isfinite(bm)):
            return False
        return float(np.std(bm, ddof=1)) > EPSILON

    def simulate_crash(self, crash_key: str) -> CrashSimResult:
        profile = CRASH_PROFILES[crash_key]
        r = self.returns
        crisis_days = 60
        float(np.mean(r))
        vol = float(np.std(r, ddof=1))
        market_drop = profile["market_drop"]
        trimmed_mean = float(np.mean(np.clip(r, np.percentile(r, 2), np.percentile(r, 98))))
        trimmed_vol = float(np.std(np.clip(r, np.percentile(r, 2), np.percentile(r, 98)), ddof=1))

        # ── F3: Hardened exposure model ───────────────────────────────────────
        # Base exposure from vol ratio — same formula as before
        base_exposure = vol / 0.012

        # Hard minimum: no strategy can show near-zero crash sensitivity
        # via low vol alone. Floor depends on whether dates are present.
        if self.has_dates:
            min_exposure = CRASH_EXPOSURE_MIN  # 0.25
        else:
            min_exposure = CRASH_EXPOSURE_MIN_NODATE  # 0.40 — more conservative without dates

        exposure = float(np.clip(base_exposure, min_exposure, 0.8))

        # AR(1) autocorrelation adjustment: momentum-reversal in crises
        # Autocorrelated strategies experience harder unwind when trend breaks
        ar1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else 0.0
        if abs(ar1) >= CRASH_AUTOCORR_THRESHOLD:
            exposure = min(0.8, exposure * CRASH_AUTOCORR_MULT)

        # Calibration flag: crash results on undated data are not market-calibrated
        calibrated = self.has_dates
        # ─────────────────────────────────────────────────────────────────────

        crash_seed = (
            _stable_seed(crash_key.encode()) ^ int.from_bytes(self.returns.tobytes()[:8], "little")
        ) % (2**31)
        rng = np.random.default_rng(crash_seed)
        vol_mult = min(profile["vol_multiplier"], 4.0)
        df_t = profile.get("fat_tail_df", 4)
        autocorr = profile.get("autocorr", 0.30)
        df_t_safe = max(df_t, 2.01)
        scale_t = trimmed_vol * vol_mult / np.sqrt(df_t_safe / (df_t_safe - 2))
        innov = rng.standard_t(df_t, crisis_days) * scale_t
        stressed = np.zeros(crisis_days)
        stressed[0] = innov[0]
        for i in range(1, crisis_days):
            stressed[i] = autocorr * stressed[i - 1] + np.sqrt(1 - autocorr**2) * innov[i]
        market_drag = (market_drop / 100) * exposure * profile["liquidity_factor"]
        stressed += market_drag / crisis_days
        bottom_10 = stressed < np.percentile(stressed, 10)
        stressed[bottom_10] -= np.abs(stressed[bottom_10]) * profile["gap_risk"]
        stressed = np.clip(stressed, -0.99, 0.99)
        cumulative = float(np.prod(1 + stressed) - 1)
        if cumulative > CRASH_MAX_GAIN:
            cumulative = CRASH_MAX_GAIN
        if trimmed_mean < -0.001 and cumulative > 0.05:
            cumulative *= CRASH_NEG_EDGE_CAP
        if market_drop < -15.0 and cumulative > 0.08:
            cumulative *= 0.25
        dd = calculate_max_drawdown(stressed)
        survived = dd < (0.25 if self.strict_mode else 0.30)

        # F3: prefix verdicts with [UNCALIBRATED] when no dates supplied
        calib_tag = "" if calibrated else "[UNCALIBRATED] "
        if survived and cumulative > -0.05:
            verdict = f"{calib_tag}🟢 YOUR STRATEGY SURVIVED."
        elif survived and cumulative > -0.20:
            verdict = f"{calib_tag}🟡 DAMAGED BUT SURVIVED."
        elif survived:
            verdict = f"{calib_tag}🟠 BARELY SURVIVED."
        elif cumulative > 0:
            verdict = f"{calib_tag}🟡 HIT DRAWDOWN LIMIT — but recovered to {cumulative * 100:+.1f}%."
        else:
            verdict = f"{calib_tag}🔴 YOUR STRATEGY WOULD HAVE BLOWN UP."

        return CrashSimResult(
            crash_name=profile["name"],
            year=profile["year"],
            description=profile["description"],
            market_drop=profile["market_drop"],
            strategy_drop=round(cumulative, 4),
            survived=survived,
            emotional_verdict=verdict,
        )

    def _detect_single_regime(self) -> Optional[str]:
        """
        FIX BUG-V4-2: Detect backtests where >85% of dated trades fall in one
        market regime. These strategies are untested in bear/consolidation
        conditions and should not receive high scores.

        Returns the dominant regime name if single-regime, else None.
        """
        if not self.has_dates:
            return None
        regime_labels = self._classify_market_regime_by_date()
        if regime_labels is None:
            return None
        counts = {}
        for label in regime_labels.values():
            if label != "UNKNOWN":
                counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        if total < 10:
            return None
        for regime, count in counts.items():
            if count / total > 0.85:
                return regime
        return None

    def _calculate_score(self, checks: List[CheckResult]) -> Tuple[float, str]:
        weights = {"Overfitting": 0.22, "Risk": 0.38, "Regime": 0.14, "Execution": 0.11, "Compliance": 0.15}
        category_scores = {}
        for cat in weights:
            cat_checks = [c for c in checks if c.category == cat and c.scored]
            category_scores[cat] = (
                float(np.mean([c.score for c in cat_checks]))
                if cat_checks
                else (0.0 if cat in ("Risk", "Overfitting", "Compliance") else 20.0)
            )
        final = sum(category_scores[cat] * w for cat, w in weights.items())
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe < 0:
            final = min(final, 20)
        elif sharpe < 0.3:
            final = min(final, 35)
        elif sharpe < 0.5:
            final = min(final, 50)
        elif sharpe > 10:
            final = min(final, 30)
        if self._low_sample_warning:
            final = min(final, 60)
        dsr_check = next((c for c in checks if c.name == "Deflated Sharpe Ratio"), None)
        if dsr_check:
            if dsr_check.score <= 10:
                final = min(final, 30)
            elif dsr_check.score <= 50:
                final = min(final, 55)
        ci_check = next((c for c in checks if c.name == "Sharpe CI Enforcement"), None)
        if ci_check:
            if ci_check.score <= 10:
                final = min(final, 35)
            elif ci_check.score <= 35:
                final = min(final, 55)
        plausibility = [c for c in checks if c.category == "Plausibility"]
        if any(not c.passed and "Manual Audit Required" in c.value for c in plausibility):
            final = min(final, 35)
        conc_check = next((c for c in checks if "Concentration" in c.name), None)
        if conc_check and not conc_check.passed and conc_check.score == 0:
            final = min(final, 38)
        cpcv_check = next((c for c in checks if c.name == "CPCV Path Stability"), None)
        if cpcv_check and cpcv_check.score == 0:
            final = min(final, 42)
        if not self.has_dates:
            final = min(final, 60)  # FIX BUG-V4-1: was 75; without timestamps,
            # regime/timeframe/data-snooping checks are impossible — F grade ceiling

        # ── P1: Extra cap for no-date + high Sharpe (unverifiable single-regime candidate) ──
        # A strategy with Sharpe > 5 and no timestamps cannot be regime-checked and is
        # statistically suspicious — cap at 50 to prevent misleadingly high scores.
        if not self.has_dates and sharpe > 5:
            final = min(final, 50)

        # FIX BUG-V4-2: single-regime backtest hard cap
        # If >85% of dated trades are in one regime, the strategy is untested
        # in other market conditions — cap at 55 regardless of in-regime performance
        dominant_regime = self._detect_single_regime()
        if dominant_regime is not None:
            final = min(final, 55)
        low_cats = sum(1 for s in category_scores.values() if s < 30)
        if low_cats >= 2:
            final = min(final, 45)
        elif low_cats >= 1:
            final = min(final, 60)
        # INSTITUTIONAL FIX: separate cap logic for prop firm vs institutional compliance.
        # Prop firm failure (DD>10%) no longer caps multi-year institutional backtests.
        # Only institutional compliance failure (Sharpe<1, DD>25%, Calmar<1) triggers cap.
        inst_compliance = [c for c in checks if c.name == "Institutional Compliance (CTA/Fund Standard)"]
        if inst_compliance and not all(c.passed for c in inst_compliance):
            final = min(final, 60)
        prop_compliance = [c for c in checks if "Prop Firm Compliance" in c.name and "Gate" not in c.name]
        if prop_compliance and not all(c.passed for c in prop_compliance):
            final = min(final, 75)  # soft cap only — prop firm rules ≠ institutional viability
        if "Short backtest" in self.timeframe_info:
            final = min(final, 75)
        if final >= 90:
            grade = "A — Institutionally Viable"
        elif final >= 80:
            grade = "B+ — Prop Firm Ready"
        elif final >= 70:
            grade = "B — Live Tradeable"
        elif final >= 60:
            grade = "C — Needs Work Before Live"
        elif final >= 50:
            grade = "D — Major Issues"
        else:
            grade = "F — Do Not Deploy"
        return round(final, 1), grade

    def run(self, enhanced: bool = False) -> ValidationReport:
        """Run validation with optional prop firm compliance"""
        # Run standard validation
        checks = [
            self.check_sharpe_decay(),
            self.check_monte_carlo(),
            self.check_outlier_dependency(),
            self.check_walk_forward(),
            self.check_bootstrap_stability(),
            self.check_deflated_sharpe(),
            self.check_sharpe_ci_enforcement(),
            self.check_min_backtest_length(),
            self.check_max_drawdown(),
            self.check_calmar_ratio(),
            self.check_var(),
            self.check_cvar(),
            self.check_sortino(),
            self.check_consecutive_losses(),
            self.check_recovery_factor(),
            self.check_absolute_sharpe(),
            self.check_ruin_probability(),
            self.check_drawdown_duration(),
            self.check_ulcer_index(),
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
            self.check_autocorrelation(),
            self.check_information_ratio(),
            self.check_beta_exposure(),
            self.check_compliance_pass(),
            self.check_prop_firm_compliance(),
            self.check_institutional_compliance(),
            # Plausibility: reported (audit flags, summary, score caps) but not weighted.
            self.check_sharpe_plausibility(),
            self.check_frequency_return_plausibility(),
            self.check_equity_smoothness_plausibility(),
            self.check_kelly_plausibility(),
        ]
        self.checks = checks

        # Calculate scores and grade
        score, grade = self._calculate_score(checks)
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        dd = calculate_max_drawdown(self.returns)
        wr = calculate_win_rate(self.returns)

        # Run crash simulations
        crash_sims = [self.simulate_crash(key) for key in CRASH_PROFILES.keys()]

        # Generate overfit profile
        overfit_prof = _OverfitDetector(
            self.returns, self.df["date"].values if self.has_dates else None
        ).detect()

        # Generate alpha decay profile
        alpha_decay = _analyze_alpha_decay(self.returns, self.df["date"].values if self.has_dates else None)

        # Get equity curve for frontend
        equity_curve_raw = None
        if self.has_dates:
            equity_curve_raw = (10000 * np.cumprod(1 + self.returns)).tolist()

        # Add prop firm compliance if requested
        prop_firm_results = None

        print("PROP ENGINE EXISTS:", self.prop_firm_engine is not None)
        print("ENHANCED MODE:", enhanced)

        if enhanced and self.prop_firm_engine is not None:
            try:
                prop_firm_results = self.prop_firm_engine.validate_all_firms()
                print("PROP ENGINE OUTPUT:", prop_firm_results)

                # Convert NumPy types to native Python types IMMEDIATELY
                prop_firm_results = convert_numpy(prop_firm_results)
                print("✅ NUMPY CONVERSION APPLIED")

                # STEP 3 — VERIFY TYPES
                if prop_firm_results.get("summary") and "average_score" in prop_firm_results["summary"]:
                    print("TYPE CHECK:", type(prop_firm_results["summary"]["average_score"]))

                # Debug: Check specific NumPy conversion
                if prop_firm_results.get("firms") and "FTMO" in prop_firm_results["firms"]:
                    ftmo_data = prop_firm_results["firms"]["FTMO"]
                    if ftmo_data.get("violations") and len(ftmo_data["violations"]) > 0:
                        first_violation = ftmo_data["violations"][0]
                        print("DEBUG VIOLATION TYPE:", type(first_violation.get("passed")))

                # Validate results structure
                if not isinstance(prop_firm_results, dict):
                    prop_firm_results = None
                elif "firms" not in prop_firm_results:
                    prop_firm_results = None
                elif "summary" not in prop_firm_results:
                    prop_firm_results = None

            except Exception as e:
                import traceback

                print(f"Prop firm compliance failed: {e}")
                print("TRACEBACK:", traceback.format_exc())
                raise e

        # Create validation report
        report = ValidationReport(
            score=score,
            grade=grade,
            sharpe=sharpe,
            max_drawdown=dd,
            win_rate=wr,
            total_trades=len(self.returns),
            profitable_trades=int(np.sum(self.returns > 0)),
            checks=checks,
            crash_sims=crash_sims,
            assumptions=self._get_assumptions(),
            validation_hash=self._generate_validation_hash(),
            validation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            audit_flags=self._generate_audit_flags(),
            plausibility_summary=self._generate_plausibility_summary(),
            alpha_decay=alpha_decay,
            overfit_profile=overfit_prof,
            engine_version=ENGINE_VERSION,
            methodology_version=METHODOLOGY_DATE,
            equity_curve=equity_curve_raw,
            monte_carlo=getattr(self, "_last_monte_carlo_sims", None),
        )

        # Add prop firm compliance to report if available
        if prop_firm_results:
            report.prop_firm_compliance = prop_firm_results
            report.enhanced = True

            # STEP 4 — CONFIRM FINAL OUTPUT
            print("FINAL KEYS:", list(report.__dict__.keys()))
            print(
                "PROP DATA EXISTS:",
                hasattr(report, "prop_firm_compliance") and report.prop_firm_compliance is not None,
            )

        return report
