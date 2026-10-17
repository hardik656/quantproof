"""Execution checks (mixin for QuantProofValidator)."""

import numpy as np

from qp_engine.core import (
    ADV_DEFAULT,
    COMMISSION_INSTITUTIONAL,
    COMMISSION_RETAIL_EQUITY,
    COMMISSION_RETAIL_SPREAD,
    EPSILON,
    IMPACT_ETA,
    PARTIAL_FILL_RATE,
    CheckResult,
    _stable_seed,
    calculate_sharpe,
)


class ExecutionChecks:
    def check_slippage_01(self) -> CheckResult:
        r = self.returns
        tpy = self.trades_per_year
        base_slip = 0.001
        freq_mult = max(1.0, np.sqrt(tpy / 252))
        eff_slip = base_slip * freq_mult

        # Primary check: does strategy remain profitable after slippage?
        net_after_slip = float(np.mean(r)) - eff_slip
        gross = float(np.sum(np.abs(r))) + EPSILON
        slip_cost = float(np.sum(np.abs(r) * eff_slip))
        impact_pct = slip_cost / gross * 100

        # Fail if: net return goes negative OR slippage > 20% of gross
        passed = net_after_slip > 0 and impact_pct < 20
        # Score: 100 if comfortable positive, scales down as edge erodes
        edge_ratio = net_after_slip / (abs(float(np.mean(r))) + EPSILON)
        score = round(max(0, min(100, edge_ratio * 100 if net_after_slip > 0 else 0)), 1)
        return CheckResult(
            "Slippage Impact (0.1% base)",
            passed,
            score,
            f"Net after 0.1% slip: {net_after_slip * 100:.3f}%/trade | {impact_pct:.1f}% of gross",
            "Strategy must remain profitable after realistic execution costs.",
            "Edge is too thin for 0.1% slippage — increase return per trade or reduce frequency.",
            "Execution",
        )

    def check_slippage_03(self) -> CheckResult:
        r = self.returns
        tpy = self.trades_per_year
        base_slip = 0.003
        freq_mult = max(1.0, np.sqrt(tpy / 252))
        eff_slip = base_slip * freq_mult

        # Primary check: does strategy remain profitable after slippage?
        net_after_slip = float(np.mean(r)) - eff_slip
        gross = float(np.sum(np.abs(r))) + EPSILON
        slip_cost = float(np.sum(np.abs(r) * eff_slip))
        impact_pct = slip_cost / gross * 100

        passed = net_after_slip > 0 and impact_pct < 40
        edge_ratio = net_after_slip / (abs(float(np.mean(r))) + EPSILON)
        score = round(max(0, min(100, edge_ratio * 100 if net_after_slip > 0 else 0)), 1)
        return CheckResult(
            "Slippage Impact (0.3% base)",
            passed,
            score,
            f"Net after 0.3% slip: {net_after_slip * 100:.3f}%/trade | {impact_pct:.1f}% of gross",
            "Small caps and illiquid stocks have 0.3%+ slippage — strategy must survive this.",
            "Edge disappears under realistic small-cap execution — use limit orders or larger targets.",
            "Execution",
        )

    def check_commission_drag(self) -> CheckResult:
        r = self.returns
        tpy = self.trades_per_year
        if tpy <= 252:
            rate, tier = COMMISSION_INSTITUTIONAL, "institutional (2bps)"
        elif tpy <= 2520:
            rate, tier = COMMISSION_RETAIL_EQUITY, "retail equity (10bps)"
        else:
            rate, tier = COMMISSION_RETAIL_SPREAD, "HFT/spread (20bps)"
        total_comm = len(r) * rate
        gross_turnover = float(np.sum(np.abs(r))) + EPSILON
        drag = total_comm / gross_turnover * 100
        passed = drag < 20
        return CheckResult(
            "Commission Drag",
            passed,
            round(max(0, 100 - drag * 4), 1),
            f"{len(r)} trades × {rate * 10000:.0f}bps ({tier}) → {drag:.1f}% of gross",
            "Commission eats into returns.",
            "Reduce frequency if >20% of gross.",
            "Execution",
        )

    def check_partial_fills(self) -> CheckResult:
        """
        Model an 80% fill rate: each trade is filled with probability 0.8 and
        missed (zero return) otherwise. Over 500 seeded simulations, measure how
        often the strategy keeps a positive mean return and how much Sharpe it
        retains. Missed fills are not random in live trading (the best breakouts
        are the ones you miss), so this is a lower bound on the damage.
        """
        r = self.returns
        n_sims = 500
        rng = np.random.default_rng(_stable_seed(r.tobytes() + b"partial_fills_v1"))
        filled = rng.random((n_sims, len(r))) < PARTIAL_FILL_RATE
        sims = np.where(filled, r, 0.0)
        means = sims.mean(axis=1)
        stds = sims.std(axis=1, ddof=1)
        sharpes = np.where(stds > EPSILON, means / np.maximum(stds, EPSILON), 0.0) * np.sqrt(
            self.trades_per_year
        )
        base_sharpe = calculate_sharpe(r, self.trades_per_year)
        pct_positive = float(np.mean(means > 0) * 100)
        p5_sharpe = float(np.percentile(sharpes, 5))
        median_sharpe = float(np.median(sharpes))
        passed = pct_positive >= 90 and p5_sharpe > 0
        return CheckResult(
            "Partial Fill Simulation",
            passed,
            round(pct_positive, 1),
            f"80% fill rate, {n_sims} sims: mean return stays positive in {pct_positive:.0f}% | "
            f"Sharpe {base_sharpe:.2f} → median {median_sharpe:.2f}, 5th pct {p5_sharpe:.2f}",
            "Backtests assume every order fills. With 1 in 5 orders missed, a thin edge can disappear.",
            "Use limit orders with realistic fill assumptions, or widen the edge per trade.",
            "Execution",
        )

    def check_live_vs_backtest_gap(self) -> CheckResult:
        bt_sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        live_sharpe = bt_sharpe * 0.6
        score = min(100, max(0, live_sharpe * 50))
        return CheckResult(
            "Live Trading Gap Estimate",
            live_sharpe > 0.5,
            round(score, 1),
            f"Backtest Sharpe: {bt_sharpe:.2f} → Estimated Live: {live_sharpe:.2f}",
            "40% Sharpe decay from backtest to live.",
            "Reduce sizing, tighten risk management.",
            "Execution",
        )

    def check_impact_adjusted_capacity(self) -> CheckResult:
        r = self.returns
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        n = len(r)
        if std_r < EPSILON or n == 0 or mean_r <= 0:
            return CheckResult(
                "Impact-Adjusted Capacity",
                mean_r > 0,
                50 if mean_r > 0 else 0,
                "Cannot compute capacity on unprofitable strategy",
                "",
                "",
                "Execution",
            )
        sigma_mkt = std_r

        def sharpe_at_aum(aum):
            participation = (aum / max(n, 1)) / ADV_DEFAULT
            impact = IMPACT_ETA * sigma_mkt * np.sqrt(max(participation, 0))
            return (mean_r - impact) / std_r * np.sqrt(self.trades_per_year)

        s_100k = sharpe_at_aum(100_000)
        s_1m = sharpe_at_aum(1_000_000)
        s_10m = sharpe_at_aum(10_000_000)
        s_100m = sharpe_at_aum(100_000_000)
        decay = (s_100k - s_10m) / (abs(s_100k) + EPSILON)
        if decay > 0.40:
            passed, score = False, max(0, int((1 - decay) * 100))
        elif decay > 0.20:
            passed, score = True, 65
        else:
            passed, score = True, 100
        return CheckResult(
            "Impact-Adjusted Capacity",
            passed,
            score,
            f"Sharpe: $100k={s_100k:.2f} | $1M={s_1m:.2f} | $10M={s_10m:.2f} | $100M={s_100m:.2f}",
            "Almgren-Chriss capacity model.",
            "If capacity <$5M, retail-only.",
            "Execution",
        )
