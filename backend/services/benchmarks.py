"""SPY / Nifty benchmark alignment and market-regime assignment.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# =========================================================
# SPY REGIME HELPERS
# =========================================================

SPY_REGIME_LOOKUP: Dict[str, str] = {
    "2020-01": "BULL",
    "2020-02": "BEAR",
    "2020-03": "BEAR",
    "2020-04": "BULL",
    "2020-05": "BULL",
    "2020-06": "CONSOLIDATION",
    "2020-07": "BULL",
    "2020-08": "BULL",
    "2020-09": "BEAR",
    "2020-10": "BEAR",
    "2020-11": "BULL",
    "2020-12": "BULL",
    "2021-01": "CONSOLIDATION",
    "2021-02": "BULL",
    "2021-03": "BULL",
    "2021-04": "BULL",
    "2021-05": "CONSOLIDATION",
    "2021-06": "BULL",
    "2021-07": "BULL",
    "2021-08": "BULL",
    "2021-09": "CONSOLIDATION",
    "2021-10": "BULL",
    "2021-11": "BEAR",
    "2021-12": "BULL",
    "2022-01": "BEAR",
    "2022-02": "BEAR",
    "2022-03": "CONSOLIDATION",
    "2022-04": "BEAR",
    "2022-05": "BEAR",
    "2022-06": "BEAR",
    "2022-07": "BULL",
    "2022-08": "BEAR",
    "2022-09": "BEAR",
    "2022-10": "BULL",
    "2022-11": "BULL",
    "2022-12": "BEAR",
    "2023-01": "BULL",
    "2023-02": "CONSOLIDATION",
    "2023-03": "BULL",
    "2023-04": "BULL",
    "2023-05": "CONSOLIDATION",
    "2023-06": "BULL",
    "2023-07": "BULL",
    "2023-08": "BEAR",
    "2023-09": "BEAR",
    "2023-10": "BEAR",
    "2023-11": "BULL",
    "2023-12": "BULL",
    "2024-01": "BULL",
    "2024-02": "BULL",
    "2024-03": "BULL",
    "2024-04": "BEAR",
    "2024-05": "BULL",
    "2024-06": "BULL",
    "2024-07": "CONSOLIDATION",
    "2024-08": "CONSOLIDATION",
    "2024-09": "BULL",
    "2024-10": "BULL",
    "2024-11": "BULL",
    "2024-12": "BEAR",
    "2025-01": "BULL",
    "2025-02": "BULL",
    "2025-03": "BEAR",
    "2025-04": "BEAR",
    "2025-05": "BULL",
    "2025-06": "BULL",
    "2025-07": "BULL",
    "2025-08": "CONSOLIDATION",
    "2025-09": "BULL",
    "2025-10": "BULL",
    "2025-11": "BULL",
    "2025-12": "BULL",
    "2026-01": "BULL",
    "2026-02": "BEAR",
    "2026-03": "BEAR",
    "2026-04": "CONSOLIDATION",
    "2026-05": "CONSOLIDATION",
    "2026-06": "CONSOLIDATION",
    "2026-07": "CONSOLIDATION",
    "2026-08": "CONSOLIDATION",
    "2026-09": "CONSOLIDATION",
    "2026-10": "CONSOLIDATION",
    "2026-11": "CONSOLIDATION",
    "2026-12": "CONSOLIDATION",
    "2027-01": "CONSOLIDATION",
    "2027-02": "CONSOLIDATION",
    "2027-03": "CONSOLIDATION",
    "2027-04": "CONSOLIDATION",
    "2027-05": "CONSOLIDATION",
    "2027-06": "CONSOLIDATION",
    "2027-07": "CONSOLIDATION",
    "2027-08": "CONSOLIDATION",
    "2027-09": "CONSOLIDATION",
    "2027-10": "CONSOLIDATION",
    "2027-11": "CONSOLIDATION",
    "2027-12": "CONSOLIDATION",
}

# =========================================================
# NIFTY REGIME LOOKUP
# =========================================================
NIFTY_REGIME_LOOKUP: Dict[str, str] = {
    # 2020 — COVID crash Mar, recovery Apr onwards
    "2020-01": "BULL",
    "2020-02": "BEAR",
    "2020-03": "BEAR",
    "2020-04": "BULL",
    "2020-05": "BULL",
    "2020-06": "BULL",
    "2020-07": "BULL",
    "2020-08": "BULL",
    "2020-09": "CONSOLIDATION",
    "2020-10": "BULL",
    "2020-11": "BULL",
    "2020-12": "BULL",
    # 2021 — strong bull run all year
    "2021-01": "BULL",
    "2021-02": "BULL",
    "2021-03": "BULL",
    "2021-04": "BEAR",
    "2021-05": "BULL",
    "2021-06": "BULL",
    "2021-07": "BULL",
    "2021-08": "BULL",
    "2021-09": "BULL",
    "2021-10": "BULL",
    "2021-11": "CONSOLIDATION",
    "2021-12": "BULL",
    # 2022 — bear year, rate hikes, FII selloff
    "2022-01": "BEAR",
    "2022-02": "BEAR",
    "2022-03": "CONSOLIDATION",
    "2022-04": "BEAR",
    "2022-05": "BEAR",
    "2022-06": "BEAR",
    "2022-07": "BULL",
    "2022-08": "BULL",
    "2022-09": "CONSOLIDATION",
    "2022-10": "BULL",
    "2022-11": "BULL",
    "2022-12": "CONSOLIDATION",
    # 2023 — recovery and bull run
    "2023-01": "BULL",
    "2023-02": "BEAR",
    "2023-03": "CONSOLIDATION",
    "2023-04": "BULL",
    "2023-05": "BULL",
    "2023-06": "BULL",
    "2023-07": "BULL",
    "2023-08": "CONSOLIDATION",
    "2023-09": "CONSOLIDATION",
    "2023-10": "BULL",
    "2023-11": "BULL",
    "2023-12": "BULL",
    # 2024 — bull run to Sep, correction Oct onwards
    "2024-01": "BULL",
    "2024-02": "BULL",
    "2024-03": "BULL",
    "2024-04": "CONSOLIDATION",
    "2024-05": "BULL",
    "2024-06": "BULL",
    "2024-07": "BULL",
    "2024-08": "BULL",
    "2024-09": "BULL",
    "2024-10": "BEAR",
    "2024-11": "BEAR",
    "2024-12": "BEAR",
    # 2025 — correction continues, slow recovery mid-year
    "2025-01": "BEAR",
    "2025-02": "BEAR",
    "2025-03": "BEAR",
    "2025-04": "CONSOLIDATION",
    "2025-05": "BULL",
    "2025-06": "BULL",
    "2025-07": "BULL",
    "2025-08": "BULL",
    "2025-09": "BULL",
    "2025-10": "BULL",
    "2025-11": "CONSOLIDATION",
    "2025-12": "BULL",
    # 2026 onwards — unknown future, use CONSOLIDATION
    "2026-01": "CONSOLIDATION",
    "2026-02": "CONSOLIDATION",
    "2026-03": "CONSOLIDATION",
    "2026-04": "CONSOLIDATION",
    "2026-05": "CONSOLIDATION",
    "2026-06": "CONSOLIDATION",
    "2026-07": "CONSOLIDATION",
    "2026-08": "CONSOLIDATION",
    "2026-09": "CONSOLIDATION",
    "2026-10": "CONSOLIDATION",
    "2026-11": "CONSOLIDATION",
    "2026-12": "CONSOLIDATION",
    "2027-01": "CONSOLIDATION",
    "2027-02": "CONSOLIDATION",
    "2027-03": "CONSOLIDATION",
    "2027-04": "CONSOLIDATION",
    "2027-05": "CONSOLIDATION",
    "2027-06": "CONSOLIDATION",
    "2027-07": "CONSOLIDATION",
    "2027-08": "CONSOLIDATION",
    "2027-09": "CONSOLIDATION",
    "2027-10": "CONSOLIDATION",
    "2027-11": "CONSOLIDATION",
    "2027-12": "CONSOLIDATION",
}

_SPY_CACHE: dict = {}
_SPY_CACHE_MAX = 10


def _get_spy_cached(start, end) -> pd.Series | None:
    """Fetch SPY Close with caching. Returns None on any failure."""
    key = (str(start.date()), str(end.date()))
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]
    try:
        import threading

        import yfinance as yf

        result_holder = [None]

        def _fetch():
            try:
                raw = yf.download("SPY", start=start, end=end, progress=False, timeout=6)["Close"]
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
        import threading

        import yfinance as yf

        result_holder = [None]

        def _fetch():
            try:
                raw = yf.download("^NSEI", start=start, end=end, progress=False, timeout=6)["Close"]
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
        lambda t: len(t) == 6 and t.isalpha() and not t.endswith(("SPY", "QQQ", "IWM", "DIA")),
        lambda t: t.startswith(("XAU", "XAG", "XTI", "XBR", "BTC", "ETH", "LTC", "XRP")),
        lambda t: t.endswith(("-EQ", ".NS", ".BO", "USDT")),
        lambda t: "USD" in t and len(t) >= 6 and t[:3] not in ("SPY", "QQQ", "IWM"),
    ]
    non_us_count = sum(1 for t in tickers if any(p(t) for p in non_us_patterns))
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
        end = df["date"].max() + pd.Timedelta(days=1)
        if benchmark_type == "nifty":
            spy = _get_nifty_cached(start, end)
            if spy is None:
                raise ValueError("Nifty data unavailable")
        else:
            spy = _get_spy_cached(start, end)
            if spy is None:
                raise ValueError("SPY data unavailable")
        sma20 = spy.rolling(20, min_periods=20).mean()

        # Build a raw regime series for the full SPY date range
        # Then apply 3-day confirmation to prevent noise-driven flips
        spy_idx = spy.index
        raw_regimes = {}
        for dt in spy_idx:
            try:
                dt_naive = (
                    pd.Timestamp(dt).tz_localize(None)
                    if getattr(dt, "tzinfo", None) is None
                    else pd.Timestamp(dt).tz_convert(None)
                )
                price = float(spy.asof(dt_naive))
                sma = float(sma20.asof(dt_naive))
                if np.isnan(price) or np.isnan(sma) or sma == 0:
                    raw_regimes[dt_naive.date()] = "CONSOLIDATION"
                    continue
                ratio = price / sma
                if ratio > 1.01:
                    raw_regimes[dt_naive.date()] = "BULL"
                elif ratio < 0.99:
                    raw_regimes[dt_naive.date()] = "BEAR"
                else:
                    raw_regimes[dt_naive.date()] = "CONSOLIDATION"
            except Exception:
                pass

        # Apply 3-consecutive-day confirmation
        # Regime only changes if the new label persists for 3+ trading days
        confirmed_regimes = {}
        sorted_dates = sorted(raw_regimes.keys())
        current_regime = "CONSOLIDATION"
        pending_regime = None
        pending_count = 0
        for dt in sorted_dates:
            raw = raw_regimes[dt]
            if raw == current_regime:
                pending_regime = None
                pending_count = 0
            elif raw == pending_regime:
                pending_count += 1
                if pending_count >= 3:
                    current_regime = pending_regime
                    pending_regime = None
                    pending_count = 0
            else:
                pending_regime = raw
                pending_count = 1
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
        return (
            df,
            f"✓ Regime auto-detected from {benchmark_label} 20-day SMA (3-day confirmation) — {dist}{spy_warning}",
        )

    except Exception:

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
        return (
            df,
            f"Regime from monthly {fallback_label} approximation (live data unavailable) {dist}{beyond_note}",
        )


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
                end = pd.Timestamp(pd.Timestamp(df["date"].max()).date()) + pd.Timedelta(days=2)
            except Exception:
                start = df["date"].min() - pd.Timedelta(days=10)
                end = df["date"].max() + pd.Timedelta(days=2)
            spy_raw = _get_spy_cached(start, end)
            if spy_raw is None:
                return (
                    None,
                    "⚠️ SPY benchmark unavailable (network/yfinance). IR + Beta checks are informational.",
                )
        else:
            spy_raw = spy_series

        if spy_raw is None or len(spy_raw) < 5:
            return (
                None,
                "⚠️ SPY benchmark unavailable (insufficient data). IR + Beta checks are informational.",
            )

        # —? Compute daily returns ————————————————————————————————————————————?
        spy_ret = spy_raw.pct_change().dropna()
        # Normalise index: strip tz info and time component — only date matters
        if hasattr(spy_ret.index, "tz") and spy_ret.index.tz is not None:
            spy_ret.index = spy_ret.index.tz_convert("UTC").tz_localize(None)
        spy_ret.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in pd.to_datetime(spy_ret.index)])

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
                d_ts = pd.Timestamp(d_date)
                val = float(spy_ret.asof(d_ts))
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
    if "date" not in df.columns:
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
        end = df["date"].max() + pd.Timedelta(days=1)
        nifty = _get_nifty_cached(start, end) if nifty_series is None else nifty_series
        if nifty is None:
            raise ValueError("Nifty data unavailable")
        nifty_ret = nifty.pct_change().dropna()

        # Normalise timezone to date-only for reliable asof() lookup
        if hasattr(nifty_ret.index, "tz") and nifty_ret.index.tz is not None:
            nifty_ret.index = nifty_ret.index.tz_convert("UTC").tz_localize(None)
        nifty_ret.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in pd.to_datetime(nifty_ret.index)])

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

        return benchmark, f"Nifty 50 benchmark aligned ({true_match_pct}% of NSE trading days)"

    except Exception as e:
        return None, f"Nifty 50 benchmark alignment failed: {str(e)[:100]}"
