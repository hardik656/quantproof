"""Plausibility checks (mixin for QuantProofValidator)."""

import numpy as np

from qp_engine.core import (
    EPSILON,
    CheckResult,
    calculate_max_drawdown,
    calculate_sharpe,
)


class PlausibilityChecks:
    def check_sharpe_plausibility(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe > 10:
            status = "⚠ Manual Audit Required"
        elif sharpe > 5:
            status = "⚠ Review Recommended"
        else:
            status = "✅ Plausible"
        return CheckResult(
            "Sharpe Plausibility",
            sharpe <= 10,
            100,
            f"Sharpe: {sharpe:.2f} → {status}",
            "Sharpe > 10 exceeds documented institutional performance.",
            "Verify data integrity.",
            "Plausibility",
        )

    def check_frequency_return_plausibility(self) -> CheckResult:
        mean_r = float(np.mean(self.returns))
        annual_ret_pct = mean_r * self.trades_per_year * 100  # simple annualised return, in %
        implausible = self.trades_per_year > 20_000 and annual_ret_pct > 500
        return CheckResult(
            "Frequency-Return Plausibility",
            not implausible,
            100,
            f"{self.trades_per_year:.0f} trades/yr → {annual_ret_pct:.0f}% annual {'⚠ Manual Audit Required' if implausible else '✅ Plausible'}",
            "HFT + extreme returns requires massive liquidity edge." if implausible else "Within bounds.",
            "Verify liquidity assumptions.",
            "Plausibility",
        )

    def check_equity_smoothness_plausibility(self) -> CheckResult:
        r = self.returns
        smoothness_r = np.std(r, ddof=1) / (abs(np.mean(r)) + EPSILON)
        max_dd = calculate_max_drawdown(r)
        if smoothness_r < 0.5 and np.mean(r) > 0 and max_dd < 0.05:
            status = "⚠ Manual Audit Required"
        elif smoothness_r < 1.0 and max_dd < 0.02:
            status = "⚠ Review Recommended"
        else:
            status = "✅ Plausible"
        return CheckResult(
            "Equity Curve Plausibility",
            smoothness_r >= 0.5 or max_dd >= 0.05,
            100,
            f"Smoothness ratio: {smoothness_r:.2f} → {status}",
            "Suspiciously smooth = lookahead bias.",
            "Check data integrity.",
            "Plausibility",
        )

    def check_kelly_plausibility(self) -> CheckResult:
        r = self.returns
        winners = r[r > 0]
        losers = r[r < 0]
        if len(winners) == 0 or len(losers) == 0:
            return CheckResult(
                "Kelly Plausibility",
                True,
                100,
                "Kelly: N/A → ✅ Plausible",
                "Cannot compute Kelly.",
                "",
                "Plausibility",
            )
        wr = len(winners) / len(r)
        avg_win = float(np.mean(winners))
        avg_loss = float(abs(np.mean(losers)))
        b = avg_win / max(avg_loss, EPSILON)
        kelly = (b * wr - (1 - wr)) / b
        if kelly > 1.0:
            passed = False
            status = "⚠ Manual Audit Required"
        elif kelly > 0.5:
            passed = True
            status = "⚠ Review Recommended"
        else:
            passed = True
            status = "✅ Plausible"
        return CheckResult(
            "Kelly Plausibility",
            passed,
            100 if passed else 0,
            f"Kelly: {kelly:.2f} → {status}",
            f"Kelly {kelly:.2f} fraction.",
            "Verify OOS stability.",
            "Plausibility",
        )
