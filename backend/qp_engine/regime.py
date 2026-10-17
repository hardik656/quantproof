"""Regime checks (mixin for QuantProofValidator)."""

import numpy as np

from qp_engine.core import (
    EPSILON,
    CheckResult,
    _stable_seed,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_win_rate,
)


class RegimeChecks:
    def check_bull_performance(self) -> CheckResult:
        regime_labels = self._classify_market_regime_by_date()
        r = self.returns
        if regime_labels is not None:
            indices = [i for i, idx in enumerate(self.df.index) if regime_labels.get(idx) == "BULL"]
            bull = r[indices] if indices else np.array([])
            source = "actual market bull periods"
        else:
            window = max(5, len(r) // 10)
            rm = np.array([np.mean(r[max(0, i - window) : i + 1]) for i in range(len(r))])
            bull = r[rm > np.percentile(rm, 60)]
            source = "rolling return proxy (no dates)"
        if len(bull) < 3:
            return CheckResult(
                "Bull Market Performance",
                False,
                20,
                f"Insufficient trades in bull ({len(bull)}) — {source}",
                "Not tested in bull.",
                "Extend backtest.",
                "Regime",
            )
        sharpe = calculate_sharpe(bull, self.trades_per_year)
        wr = calculate_win_rate(bull)
        return CheckResult(
            "Bull Market Performance",
            sharpe > 0 and wr > 45,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | WR: {wr:.1f}% ({len(bull)} trades) | {source}",
            "Bull market behavior.",
            "Check momentum signal calibration.",
            "Regime",
        )

    def check_bear_performance(self) -> CheckResult:
        regime_labels = self._classify_market_regime_by_date()
        r = self.returns
        if regime_labels is not None:
            indices = [i for i, idx in enumerate(self.df.index) if regime_labels.get(idx) == "BEAR"]
            bear = r[indices] if indices else np.array([])
            source = "actual market bear periods"
        else:
            window = max(5, len(r) // 10)
            rm = np.array([np.mean(r[max(0, i - window) : i + 1]) for i in range(len(r))])
            bear = r[rm < np.percentile(rm, 40)]
            source = "rolling return proxy (no dates)"
        if len(bear) < 3:
            return CheckResult(
                "Bear Market Performance",
                False,
                20,
                f"Insufficient trades in bear ({len(bear)}) — {source}",
                "Not tested in bear.",
                "Extend backtest.",
                "Regime",
            )
        sharpe = calculate_sharpe(bear, self.trades_per_year)
        wr = calculate_win_rate(bear)
        return CheckResult(
            "Bear Market Performance",
            sharpe > -1.0,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | WR: {wr:.1f}% ({len(bear)} trades) | {source}",
            "Bear market behavior.",
            "Add regime detection.",
            "Regime",
        )

    def check_consolidation_performance(self) -> CheckResult:
        regime_labels = self._classify_market_regime_by_date()
        r = self.returns
        if regime_labels is not None:
            indices = [i for i, idx in enumerate(self.df.index) if regime_labels.get(idx) == "CONSOL"]
            consol = r[indices] if indices else np.array([])
            source = "actual consolidation periods"
        else:
            window = max(5, len(r) // 10)
            rm = np.array([np.mean(r[max(0, i - window) : i + 1]) for i in range(len(r))])
            lo, hi = np.percentile(rm, 40), np.percentile(rm, 60)
            consol = r[(rm >= lo) & (rm <= hi)]
            source = "rolling return proxy (no dates)"
        if len(consol) < 3:
            return CheckResult(
                "Consolidation Performance",
                False,
                40,
                f"Insufficient trades in consol ({len(consol)}) — {source}",
                "Not tested in sideways.",
                "Extend backtest.",
                "Regime",
            )
        sharpe = calculate_sharpe(consol, self.trades_per_year)
        wr = calculate_win_rate(consol)
        return CheckResult(
            "Consolidation Performance",
            sharpe > 0,
            round(min(100, max(0, 50 + sharpe * 20)), 1),
            f"Sharpe: {sharpe:.2f} | WR: {wr:.1f}% ({len(consol)} trades) | {source}",
            "Sideways market behavior.",
            "Add choppiness filter.",
            "Regime",
        )

    def check_volatility_stress(self) -> CheckResult:
        r = self.returns
        orig_sharpe = calculate_sharpe(r, self.trades_per_year)
        vol = float(np.std(r, ddof=1))
        mean_r = float(np.mean(r))
        # FIX BUG-V4-4: seed from strategy returns hash, not fixed string
        # Prevents same stressed sequence rewarding/punishing all strategies identically
        rng = np.random.default_rng(_stable_seed(self.returns.tobytes() + b"vol_stress_v4"))
        autocorr = 0.30
        innov = rng.standard_t(3, 60) * vol * 3.0 / np.sqrt(3)
        stressed = np.zeros(60)
        stressed[0] = innov[0]
        for i in range(1, 60):
            stressed[i] = autocorr * stressed[i - 1] + np.sqrt(1 - autocorr**2) * innov[i]
        vol_target_scale = 1.0 / 3.0
        stressed += mean_r * vol_target_scale
        stressed_sharpe = calculate_sharpe(stressed, self.trades_per_year)
        stressed_dd = calculate_max_drawdown(stressed) * 100
        # INSTITUTIONAL FIX: measure relative Sharpe degradation, not absolute Sharpe sign.
        # Previous check failed any strategy where stressed Sharpe < 0, which happens for ALL
        # strategies with vol > ~0.8%/trade when a 3x vol spike overwhelms mean drift.
        # Institutional standard: strategy survives if it retains >30% of its Sharpe under stress.
        # Pass: degradation < 70% (retains 30%+ of edge) AND stress DD < 40%
        degradation = (orig_sharpe - stressed_sharpe) / (abs(orig_sharpe) + EPSILON) * 100
        sharpe_retained = max(0.0, 100.0 - degradation)
        passed = degradation < 70 and stressed_dd < 40
        score = max(0, min(100, sharpe_retained * 0.7 + max(0, 40 - stressed_dd) * 0.75))
        return CheckResult(
            "Volatility Spike Stress Test",
            passed,
            round(score, 1),
            f"Vol spike (3x, t(3)): Sharpe {orig_sharpe:.2f}→{stressed_sharpe:.2f} "
            f"({100 - degradation:.0f}% retained) | Stress DD: {stressed_dd:.1f}%",
            "Fat-tail 3x vol spike with AR(1) autocorrelation. "
            "Pass = retains ≥30% of Sharpe and stress drawdown < 40%. "
            "Institutional standard: strategy must survive a Black Monday-scale event.",
            "Use ATR-based position sizing to reduce vol-of-vol exposure.",
            "Regime",
        )

    def check_frequency_consistency(self) -> CheckResult:
        r = self.returns
        window = max(5, len(r) // 10)
        pct = float(
            np.mean(np.array([np.mean(r[i : i + window]) for i in range(0, len(r) - window)]) > 0) * 100
        )
        return CheckResult(
            "Performance Consistency",
            pct > 60,
            round(pct, 1),
            f"Profitable in {pct:.1f}% of rolling windows",
            "Consistent strategies generate returns steadily.",
            "Define edge conditions explicitly.",
            "Regime",
        )

    def check_regime_coverage(self) -> CheckResult:
        if "regime" not in self.df.columns:
            return CheckResult(
                "Regime Coverage",
                False,
                40,
                "No regime column",
                "Regime detection improves survival.",
                "Add BULL/BEAR/CONSOL labels.",
                "Regime",
            )
        regimes = self.df["regime"].value_counts(normalize=True)
        coverage = len(regimes) / 4 * 100
        return CheckResult(
            "Regime Coverage",
            coverage >= 75,
            round(coverage, 1),
            f"Regimes covered: {list(regimes.index.astype(str))} ({len(regimes)}/3)",
            "Full coverage = all market conditions tested.",
            "Add more history to cover BULL, BEAR and CONSOL.",
            "Regime",
        )
