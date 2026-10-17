"""Risk checks (mixin for QuantProofValidator)."""

import numpy as np

from qp_engine.core import (
    EPSILON,
    ULCER_INDEX_THRESHOLD,
    CheckResult,
    calculate_calmar,
    calculate_cvar,
    calculate_drawdown_duration,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_ulcer_index,
    calculate_win_rate,
)


class RiskChecks:
    def check_max_drawdown(self) -> CheckResult:
        dd_pct = calculate_max_drawdown(self.returns) * 100
        score = max(0, 100 - dd_pct * 3)
        return CheckResult(
            "Max Drawdown",
            dd_pct < 20,
            round(score, 1),
            f"Max drawdown: {dd_pct:.1f}%",
            "Funds reject >20% drawdown.",
            "Add circuit breaker.",
            "Risk",
        )

    def check_calmar_ratio(self) -> CheckResult:
        calmar = calculate_calmar(self.returns, self.trades_per_year)
        score = max(0, min(100, calmar * 40))  # clamp — calmar can be deeply negative
        return CheckResult(
            "Calmar Ratio",
            calmar > 1.5,
            round(score, 1),
            f"Calmar (CAGR/MaxDD): {calmar:.2f}",
            "Prop firms want >1.5.",
            "Improve position sizing.",
            "Risk",
        )

    def check_var(self) -> CheckResult:
        r = self.returns
        var_99 = float(np.percentile(r, 1))
        vol = float(np.std(r, ddof=1))
        # INSTITUTIONAL FIX: VaR should be measured relative to strategy vol, not mean.
        # For a normal distribution VaR_99 ≈ -2.33σ, so ratio ≈ 2.33.
        # Threshold of 4.0σ is generous (fat tails / skew allowed).
        # Previous mean-based denominator incorrectly penalised high-vol strategies.
        ratio = abs(var_99) / (vol + EPSILON)  # ~2.3 for normal; >4 signals extreme fat tails
        passed = ratio < 4.0
        score = max(0, 100 - max(0, ratio - 2.33) * 30)  # full marks at normal-dist VaR
        return CheckResult(
            "Value at Risk (VaR)",
            passed,
            round(score, 1),
            f"VaR 99%: {var_99:.4f} | VaR/σ ratio: {ratio:.2f} (threshold: 4.0σ)",
            "VaR relative to strategy volatility. Normal distribution ≈ 2.33σ. "
            "Ratios above 4.0σ indicate extreme fat tails beyond what vol-targeting can manage.",
            "Reduce leverage or apply volatility-scaled position sizing.",
            "Risk",
        )

    def check_cvar(self) -> CheckResult:
        r = self.returns
        cvar_95 = calculate_cvar(r, 0.95, seed=self.seed)
        calculate_cvar(r, 0.99, seed=self.seed)
        mean = float(np.mean(r))
        if abs(mean) < 0.001:
            passed = abs(cvar_95) < 0.08
            score = max(0, 100 - abs(cvar_95) * 1250)
        else:
            ratio = abs(cvar_95) / (abs(mean) + EPSILON)
            passed = ratio < 15
            score = max(0, 100 - ratio * 4)
        return CheckResult(
            "CVaR / Expected Shortfall",
            passed,
            round(score, 1),
            f"CVaR 95%: {cvar_95:.4f}",
            "Average loss in worst 5%.",
            "Reduce tail exposure.",
            "Risk",
        )

    def check_sortino(self) -> CheckResult:
        sortino = calculate_sortino(self.returns, self.trades_per_year)
        if sortino > 2.0:
            score, passed = 100, True
        elif sortino >= 1.0:
            score, passed = 75, True
        elif sortino >= 0.5:
            score, passed = 45, False
        else:
            score, passed = 15, False
        return CheckResult(
            "Sortino Ratio",
            passed,
            round(score, 1),
            f"Sortino: {sortino:.2f}",
            "Penalises only downside vol.",
            "ATR-based sizing.",
            "Risk",
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

        # FIX BUG-V4-3: zero losing trades over 30+ trades is statistically implausible,
        # not a perfect score — rewards lookahead bias / synthetic data
        min_expected_losses = max(1, int(len(r) * 0.05))  # expect at least 5% losing trades
        if max_streak == 0 and len(r) >= 30 and int(np.sum(self.returns < 0)) == 0:
            return CheckResult(
                "Max Losing Streak",
                False,
                30,
                f"Zero losing trades in {len(r)} trades — statistically implausible "
                f"(expected ~{min_expected_losses} losses at 5% loss rate)",
                "A real strategy should have some losing trades. Zero losses suggests "
                "lookahead bias, synthetic data, or data processing errors.",
                "Check the export for missing losing trades or a look-ahead bug in the backtest.",
                "Risk",
            )

        score = max(0, 100 - max_streak * 10)
        return CheckResult(
            "Max Losing Streak",
            max_streak < 8,
            round(score, 1),
            f"Longest losing streak: {max_streak}",
            "Prop firms reject 8+ consecutive losses.",
            "Pause after streak >5.",
            "Risk",
        )

    def check_recovery_factor(self) -> CheckResult:
        r = self.returns
        recovery = float(np.sum(r[r > 0])) / (float(abs(np.sum(r[r < 0]))) + EPSILON)
        score = min(100, recovery * 40)
        return CheckResult(
            "Recovery Factor",
            recovery > 1.5,
            round(score, 1),
            f"Recovery factor: {recovery:.2f}",
            "Below 1.5 means wins barely cover drawdowns.",
            "Asymmetric R:R.",
            "Risk",
        )

    def check_absolute_sharpe(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        if sharpe > 1.5:
            score, passed = 100, True
        elif sharpe >= 1.0:
            score, passed = 75, True
        elif sharpe >= 0.5:
            score, passed = 45, False
        else:
            score, passed = 15, False
        return CheckResult(
            "Absolute Sharpe Ratio",
            passed,
            round(score, 1),
            f"Annualized Sharpe: {sharpe:.2f}",
            "Institutional minimum Sharpe > 1.0.",
            "Better entry timing.",
            "Risk",
        )

    def check_ruin_probability(self) -> CheckResult:
        r = self.returns
        wr = calculate_win_rate(r) / 100.0
        winners = r[r > 0]
        losers = r[r < 0]
        avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
        avg_loss = float(abs(np.mean(losers))) if len(losers) > 0 else 0.0
        # INSTITUTIONAL FIX: use N=50 capital units (2% risk per trade = 50 units).
        # Previous N=10 (10% risk) massively inflated ruin probability for profitable strategies.
        # At N=50, a strategy with WR=55% has ruin prob <0.01% — correctly near-zero.
        # The check now measures genuine ruin risk at institutional position sizing.
        p, q = max(wr, EPSILON), max(1 - wr, EPSILON)
        rr_ratio = avg_win / (avg_loss + EPSILON)  # reward-to-risk ratio
        if p > q:  # profitable edge exists
            # Generalised gambler's ruin: P(ruin) = ((q/p) / rr_ratio)^N
            adjusted_ratio = (q / p) / (rr_ratio + EPSILON)
            try:
                if adjusted_ratio >= 1.0:
                    ruin_prob = 1.0
                elif adjusted_ratio <= 0:
                    ruin_prob = 0.0
                else:
                    ruin_prob = float(adjusted_ratio**50)
                    ruin_prob = float(np.clip(ruin_prob, 0.0, 1.0))
            except (OverflowError, FloatingPointError, ZeroDivisionError, ValueError):
                ruin_prob = 1.0
        else:
            ruin_prob = 1.0
        ruin_pct = ruin_prob * 100
        score = max(0, 100 - ruin_pct * 10)  # 10% ruin → score 0
        return CheckResult(
            "Probability of Ruin",
            ruin_pct < 5.0,
            round(score, 1),
            f"Ruin probability: {ruin_pct:.2f}% | WR: {wr:.1%} | R:R: {rr_ratio:.2f} | N=50 units",
            "Gambler's ruin at institutional position sizing (2% risk per trade, N=50 units). "
            "Measures probability of losing entire account before doubling it.",
            "Improve win rate or reward-to-risk ratio.",
            "Risk",
        )

    def check_drawdown_duration(self) -> CheckResult:
        r = self.returns
        dd_info = calculate_drawdown_duration(r, self.has_dates, self.df["date"] if self.has_dates else None)
        max_dur = dd_info["max_dd_duration"]
        ever = dd_info["ever_recovered"]
        if dd_info.get("dd_duration_days") is not None:
            dur_days = dd_info["dd_duration_days"]
            dur_display = f"{dur_days} calendar days"
            # INSTITUTIONAL FIX: scale threshold by backtest length.
            # 180-day flat threshold was calibrated for 30-day prop firm challenges.
            # A 7-year institutional backtest can legitimately have a 12-month drawdown.
            # Threshold = 180 days minimum, scaling up with backtest length, capped at 730 days (2 years).
            if self.has_dates:
                bt_days = (self.df["date"].max() - self.df["date"].min()).days
                bt_months = max(1, bt_days / 30.44)
                threshold_days = int(min(730, max(180, 180 + bt_months * 8)))
            else:
                threshold_days = 180
            threshold_met = dur_days < threshold_days
            threshold_display = (
                f"{threshold_days}d threshold (scaled to {bt_months:.0f}mo backtest)"
                if self.has_dates
                else "180d threshold"
            )
        else:
            threshold_met = max_dur < len(r) * 0.25  # raised from 0.20
            dur_display = f"{max_dur} trade periods"
            threshold_display = "25% of backtest"
        if threshold_met and ever:
            passed, score = True, 90
        elif threshold_met and not ever:
            passed, score = False, 50
        else:
            passed, score = False, 20
        return CheckResult(
            "Drawdown Duration & Recovery",
            passed,
            round(score, 1),
            f"Max DD duration: {dur_display} | {threshold_display}",
            "Time underwater in longest drawdown, scaled to backtest length. "
            "Prop firm standard: 180 days. Institutional: up to 24 months.",
            "Add re-entry rules or circuit breakers after extended drawdown periods.",
            "Risk",
        )

    def check_ulcer_index(self) -> CheckResult:
        r = self.returns
        ui = calculate_ulcer_index(r)
        sharpe = calculate_sharpe(r, self.trades_per_year)
        martin = sharpe / (ui + EPSILON)
        if ui < 2.0:
            passed, score = True, 100
        elif ui < ULCER_INDEX_THRESHOLD:
            passed, score = True, 65
        elif ui < 10.0:
            passed, score = False, 30
        else:
            passed, score = False, 5
        return CheckResult(
            "Ulcer Index",
            passed,
            round(score, 1),
            f"Ulcer Index: {ui:.2f} | Martin Ratio: {martin:.2f}",
            "Combines depth+duration of drawdowns.",
            "Reduce position sizing.",
            "Risk",
        )
