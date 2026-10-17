"""Overfitting checks (mixin for QuantProofValidator)."""

from itertools import combinations as _combinations

import numpy as np

from qp_engine.core import (
    AUTOCORR_AR1_FAIL,
    AUTOCORR_AR1_WARN,
    CPCV_ACCEPTABLE_THRESHOLD,
    CPCV_EMBARGO_FRAC,
    CPCV_N_SPLITS,
    CPCV_N_TEST,
    CPCV_ROBUST_THRESHOLD,
    EPSILON,
    LJUNG_BOX_ALPHA,
    LJUNG_BOX_LAGS,
    CheckResult,
    _ljung_box_test,
    _newey_west_se_sharpe,
    calculate_deflated_sharpe,
    calculate_harvey_liu_zhu_min_obs,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sharpe_ci,
)


class OverfittingChecks:
    def check_sharpe_decay(self) -> CheckResult:
        r, n = self.returns, len(self.returns)
        if n < 20:
            return CheckResult(
                "Sharpe Decay",
                False,
                20,
                "Insufficient data",
                "Need 20+ trades",
                "Add more backtest history",
                "Overfitting",
            )
        mid = n // 2
        s_in = calculate_sharpe(r[:mid], self.trades_per_year)
        s_out = calculate_sharpe(r[mid:], self.trades_per_year)
        decay = (s_in - s_out) / (abs(s_in) + EPSILON) * 100

        # Additional: check return mean distribution shift — catches shuffled high-PnL outliers
        mean_in = float(np.mean(r[:mid]))
        mean_out = float(np.mean(r[mid:]))
        mean_decay = (mean_in - mean_out) / (abs(mean_in) + EPSILON) * 100 if mean_in > 0 else 0

        # Use the worse of Sharpe decay or mean decay
        effective_decay = max(decay, mean_decay * 0.7)
        score = max(0, 100 - max(0, effective_decay))
        passed = effective_decay < 40
        return CheckResult(
            "Sharpe Ratio Decay",
            passed,
            round(score, 1),
            f"In-sample: {s_in:.2f} → Out-of-sample: {s_out:.2f} ({effective_decay:.1f}% effective decay)",
            "High decay means the strategy was fitted to historical noise.",
            "Run walk-forward optimization.",
            "Overfitting",
        )

    def check_monte_carlo(self) -> CheckResult:
        r = self.returns
        mean_sims = [np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(300)]
        mean_pct = float(np.mean(np.array(mean_sims) > 0) * 100)
        original_dd = calculate_max_drawdown(r)
        seq_sims = [
            abs(calculate_max_drawdown(self.rng.permutation(r))) - abs(original_dd) < 0.10 for _ in range(200)
        ]
        sequence_pct = float(np.mean(seq_sims) * 100)
        perms = [self.rng.permutation(r) for _ in range(500)]
        equity_sims = [np.prod(1 + p) for p in perms]

        # Generate full equity curve paths for frontend Monte Carlo display
        base_equity = 10000  # Starting equity
        equity_curve_paths = []
        for perm in perms[:200]:  # Limit for performance
            equity_path = [base_equity]
            cumulative = base_equity
            for ret in perm:
                cumulative *= 1 + ret
                equity_path.append(cumulative)
            # Normalize to start at 100 for frontend compatibility
            normalized_path = [float(100 * (x / base_equity)) for x in equity_path]
            equity_curve_paths.append(normalized_path)

        # Store Monte Carlo data for frontend - ensure all types are JSON serializable
        self._last_monte_carlo_sims = {
            "equity_sims": equity_curve_paths,  # Full paths instead of single values
            "mean_sims": [float(x) for x in mean_sims[:100]],  # Limit for performance
            "sequence_pct": float(sequence_pct),  # Convert to Python float
            "mean_pct": float(mean_pct),  # Convert to Python float
        }

        # DEBUG: Print Monte Carlo data structure
        print("[DEBUG] Monte Carlo data stored:")
        print(f"  - equity_sims length: {len(equity_curve_paths)}")
        print(f"  - First path length: {len(equity_curve_paths[0]) if equity_curve_paths else 0}")
        print(f"  - First path sample: {equity_curve_paths[0][:5] if equity_curve_paths else 'None'}")
        print(f"  - _last_monte_carlo_sims keys: {list(self._last_monte_carlo_sims.keys())}")

        worst_5pct = np.percentile(equity_sims, 5)
        if not self.has_dates:
            worst_5pct_val = (worst_5pct - 1) * 100
        else:
            delta = self.df["date"].max() - self.df["date"].min()
            tsy = max(delta.total_seconds() / (365.25 * 24 * 3600), 0.1)
            if "Short backtest" in self.timeframe_info:
                tsy = 1.0
            worst_5pct_val = (worst_5pct ** (1 / tsy) - 1) * 100
        worst_5pct_dd = np.percentile([abs(calculate_max_drawdown(p)) for p in perms], 95) * 100
        equity_score = min(100, max(0, 100 + worst_5pct_val * 2 - worst_5pct_dd))
        combined = mean_pct * 0.4 + sequence_pct * 0.3 + equity_score * 0.3
        passed = combined > (70 if self.strict_mode else 60)
        return CheckResult(
            "Monte Carlo Robustness",
            passed,
            round(combined, 1),
            f"Mean: {mean_pct:.1f}% | Seq: {sequence_pct:.1f}% | Equity: {equity_score:.1f}%",
            "Three-component MC: mean stability, sequence fragility, worst-case equity.",
            "If equity score low, strategy has tail risk.",
            "Overfitting",
        )

    def check_outlier_dependency(self) -> CheckResult:
        r = self.returns
        wins = r[r > 0]
        losses = r[r < 0]
        if len(wins) < 3:
            return CheckResult(
                "Profit / Loss Concentration",
                False,
                0,
                "Too few winning trades",
                "Need at least 3 winning trades.",
                "Extend backtest.",
                "Overfitting",
            )
        n = len(r)
        total_profit = float(np.sum(wins))
        total_loss = float(abs(np.sum(losses))) if len(losses) > 0 else EPSILON
        top1pct_count = max(1, int(n * 0.01))
        top1pct_profit = float(np.sum(np.sort(r)[::-1][:top1pct_count]))
        top1pct_share = top1pct_profit / total_profit * 100
        top2_share = float(np.sum(np.sort(r)[::-1][:2])) / total_profit * 100
        loss_conc_share = 0.0
        if len(losses) >= 3:
            bot1pct_count = max(1, int(n * 0.01))
            bot1pct_loss = float(abs(np.sum(np.sort(losses)[:bot1pct_count])))
            loss_conc_share = bot1pct_loss / total_loss * 100
        r_trimmed = np.delete(r, np.argsort(r)[::-1][:top1pct_count])
        sharpe_raw = calculate_sharpe(r, self.trades_per_year)
        sharpe_trim = calculate_sharpe(r_trimmed, self.trades_per_year) if len(r_trimmed) > 5 else 0.0
        sharpe_decay = (sharpe_raw - sharpe_trim) / (abs(sharpe_raw) + EPSILON)
        if top1pct_share > 25 or sharpe_decay > 0.50 or top2_share > 20:
            passed, score = False, 0
        elif loss_conc_share > 50 and len(losses) >= 3:
            passed, score = False, 15
        elif top1pct_share > 15 or sharpe_decay > 0.30:
            passed, score = False, 40
        else:
            passed, score = True, 100
        return CheckResult(
            "Profit / Loss Concentration",
            passed,
            score,
            f"Top-1% = {top1pct_share:.1f}% profit | Loss conc = {loss_conc_share:.1f}%",
            "Profit concentration check.",
            "Remove your 3 best trades and recheck Sharpe.",
            "Overfitting",
        )

    def check_walk_forward(self) -> CheckResult:
        r = self.returns
        if len(r) < 60:
            return CheckResult(
                "CPCV Path Stability",
                False,
                20,
                "Not enough data for CPCV",
                "Need 60+ trades.",
                "Extend backtest.",
                "Overfitting",
            )
        n_splits = CPCV_N_SPLITS
        n_test = CPCV_N_TEST
        fold_size = len(r) // n_splits
        folds = [r[i * fold_size : (i + 1) * fold_size] for i in range(n_splits)]
        embargo_sz = max(1, int(len(r) * CPCV_EMBARGO_FRAC))
        path_sharpes = []
        for test_idx_tuple in _combinations(range(n_splits), n_test):
            test_set = set(test_idx_tuple)
            test = np.concatenate([folds[i] for i in test_idx_tuple])
            train_parts = []
            for i in range(n_splits):
                if i in test_set:
                    continue
                fold_data = folds[i].copy()
                is_left_adjacent = (i + 1) in test_set
                is_right_adjacent = (i - 1) in test_set
                if is_left_adjacent:
                    fold_data = fold_data[:-embargo_sz] if len(fold_data) > embargo_sz else np.array([])
                if is_right_adjacent:
                    fold_data = fold_data[embargo_sz:] if len(fold_data) > embargo_sz else np.array([])
                if len(fold_data) > 0:
                    train_parts.append(fold_data)
            if len(test) < 5 or np.std(test, ddof=1) < EPSILON:
                continue
            s = calculate_sharpe(test, self.trades_per_year)
            path_sharpes.append(s)
        if not path_sharpes:
            return CheckResult(
                "CPCV Path Stability", False, 0, "CPCV could not compute paths", "", "", "Overfitting"
            )
        paths = np.array(path_sharpes)
        pct_positive = float(np.mean(paths > 0) * 100)
        path_std = float(np.std(paths))
        float(np.mean(paths))
        n_paths = len(paths)
        if path_std > CPCV_ACCEPTABLE_THRESHOLD or pct_positive < 60:
            passed, score = False, 0
        elif path_std > CPCV_ROBUST_THRESHOLD or pct_positive < 80:
            passed, score = False, 45
        else:
            passed, score = True, min(100, int(pct_positive))
        return CheckResult(
            "CPCV Path Stability",
            passed,
            score,
            f"{n_paths} paths: {pct_positive:.0f}% positive | Sharpe std: ±{path_std:.2f}",
            "CPCV robustness.",
            "If std > 2.5, strategy relies on specific market history.",
            "Overfitting",
        )

    def check_bootstrap_stability(self) -> CheckResult:
        r = self.returns
        pct = float(
            np.mean(
                np.array([np.mean(self.rng.choice(r, size=len(r), replace=True)) for _ in range(500)]) > 0
            )
            * 100
        )
        return CheckResult(
            "Bootstrap Stability",
            pct > 70,
            round(pct, 1),
            f"Positive expectancy in {pct:.1f}% of 500 bootstrap samples",
            "Stable edge shows up consistently in resampling.",
            "Fix win rate or R:R.",
            "Overfitting",
        )

    def check_deflated_sharpe(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        dsr_result = calculate_deflated_sharpe(self.returns, sharpe, self.n_trials, self.trades_per_year)
        dsr = dsr_result["dsr"]
        dsr_inst = dsr_result["dsr_institutional"]
        sr_min = dsr_result["sr_min_95"]
        se = dsr_result["se_sr"]
        bias_label = dsr_result["bias_label"]

        if dsr >= 0.99:
            passed, score = True, 100
        elif dsr >= 0.95:
            passed, score = True, 80
        elif dsr >= 0.85:
            passed, score = False, 50
        else:
            passed, score = False, 10

        # Enhanced output with institutional comparison
        user_dsr_text = (
            f"DSR={dsr:.3f} | SE=±{se:.2f} | Min SR: {sr_min:.2f} | n_trials={self.n_trials} ({bias_label})"
        )
        institutional_text = f"Institutional DSR (n=20): {dsr_inst:.3f}"
        comparison_text = f"User vs Institutional: {dsr:.3f} vs {dsr_inst:.3f}"

        return CheckResult(
            "Deflated Sharpe Ratio",
            passed,
            round(score, 1),
            f"{user_dsr_text} | {institutional_text}",
            f"DSR corrects Sharpe={sharpe:.2f} for non-normality and selection bias. {comparison_text}",
            "Increase backtest length or reduce parameter count.",
            "Overfitting",
        )

    def check_sharpe_ci_enforcement(self) -> CheckResult:
        sharpe, ci_low, ci_high, p_val = calculate_sharpe_ci(self.returns, self.trades_per_year)
        ci_width = ci_high - ci_low
        if ci_low > 0.5:
            passed, score = True, 100
        elif ci_low > 0.0:
            passed, score = True, 70
        elif ci_low > -0.5:
            passed, score = False, 35
        else:
            passed, score = False, 10
        return CheckResult(
            "Sharpe CI Enforcement",
            passed,
            round(score, 1),
            f"Sharpe {sharpe:.2f} | 95% CI: [{ci_low:.2f}, {ci_high:.2f}] | p={p_val:.3f}",
            f"With {len(self.returns)} trades, CI width = ±{ci_width / 2:.2f}.",
            "Collect more backtest data.",
            "Overfitting",
        )

    def check_min_backtest_length(self) -> CheckResult:
        sharpe = calculate_sharpe(self.returns, self.trades_per_year)
        n = len(self.returns)
        min_required = calculate_harvey_liu_zhu_min_obs(max(sharpe, 0.1), n_trials=self.n_trials)
        if n >= min_required * 2:
            passed, score = True, 100
        elif n >= min_required:
            passed, score = True, 70
        elif n >= min_required * 0.5:
            passed, score = False, 35
        else:
            passed, score = False, 5
        return CheckResult(
            "Minimum Backtest Length",
            passed,
            round(score, 1),
            f"{n} trades vs {min_required} required (Sharpe={sharpe:.2f})",
            f"Need ≥{min_required} trades for statistical significance.",
            f"Extend backtest to at least {min_required} trades.",
            "Overfitting",
        )

    def check_autocorrelation(self) -> CheckResult:
        """
        P2: Test for serial correlation in returns using Ljung-Box Q-test.
        If significant autocorrelation is detected, also computes a
        Newey-West corrected Sharpe SE and compares it to the IID SE.

        Significant AR(1) invalidates DSR, CI, and Bootstrap checks because
        they all assume IID returns. Corrected SE(SR) replaces the IID figure.

        Reference:
          Ljung & Box (1978) Biometrika 65(2):297-303
          Newey & West (1987) Econometrica 55(3):703-708
          Lo (2002) J. Portfolio Mgmt — Sharpe under serial correlation
        """
        r = self.returns
        n = len(r)

        if n < 20:
            return CheckResult(
                "Autocorrelation (Ljung-Box)",
                True,
                80,
                "Insufficient data for Ljung-Box test (need 20+ trades)",
                "Cannot assess serial correlation.",
                "Extend backtest.",
                "Overfitting",
            )

        # AR(1) estimate
        ar1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if n > 1 else 0.0
        if np.isnan(ar1):
            ar1 = 0.0

        # Ljung-Box at lags 1, 5, 10
        lb_results = _ljung_box_test(r, LJUNG_BOX_LAGS)

        # Any lag with p < alpha triggers a fail
        lb_fails = {lag: (q, p) for lag, (q, p) in lb_results.items() if p < LJUNG_BOX_ALPHA}
        has_autocorr = len(lb_fails) > 0 or abs(ar1) >= AUTOCORR_AR1_FAIL

        # Newey-West corrected SE vs IID SE
        iid_se = float(np.sqrt(1.0 / n) * np.sqrt(self.trades_per_year))
        nw_se = _newey_west_se_sharpe(r, self.trades_per_year)
        se_inflation = nw_se / max(iid_se, EPSILON)

        # Build display string for LB results
        lb_str = " | ".join(
            f"lag{lag}: Q={q:.2f} p={p:.3f}{'*' if p < LJUNG_BOX_ALPHA else ''}"
            for lag, (q, p) in lb_results.items()
        )

        if has_autocorr:
            passed = False
            if abs(ar1) >= AUTOCORR_AR1_FAIL:
                score = 0
            else:
                score = 30
            value = (
                f"AR(1)={ar1:.3f} | {lb_str} | "
                f"NW-SE={nw_se:.3f} vs IID-SE={iid_se:.3f} "
                f"(SE inflated ×{se_inflation:.2f})"
            )
            insight = (
                f"Significant serial correlation detected (AR(1)={ar1:.3f}). "
                f"All IID-assuming tests (DSR, CI, Bootstrap) are unreliable. "
                f"Newey-West corrects SE(SR) upward by ×{se_inflation:.2f}x — "
                f"true statistical significance is lower than reported."
            )
            fix = (
                "Use Newey-West robust SE for inference. "
                "Check for look-ahead bias or return smoothing artefacts. "
                "AR(1) > 0.3 often signals data snooping or bid/ask bounce."
            )
        elif abs(ar1) >= AUTOCORR_AR1_WARN:
            passed = True
            score = 60
            value = f"AR(1)={ar1:.3f} (marginal) | {lb_str} | NW-SE={nw_se:.3f} vs IID-SE={iid_se:.3f}"
            insight = (
                f"Mild serial correlation (AR(1)={ar1:.3f}). "
                f"Ljung-Box tests pass but NW-SE is ×{se_inflation:.2f}x larger than IID. "
                f"Statistical significance slightly overstated."
            )
            fix = "Monitor AR(1); consider Newey-West robust standard errors."
        else:
            passed = True
            score = 100
            value = f"AR(1)={ar1:.3f} | {lb_str} | NW-SE={nw_se:.3f} ≈ IID-SE={iid_se:.3f}"
            insight = "No significant serial correlation. IID assumption holds."
            fix = ""

        return CheckResult(
            "Autocorrelation (Ljung-Box)", passed, round(score, 1), value, insight, fix, "Overfitting"
        )

    def check_information_ratio(self) -> CheckResult:
        """
        F1 — Information Ratio vs benchmark.

        Computes beta, Jensen's alpha, tracking error, and IR.
        When no benchmark is provided: informational placeholder (score=50).
        When benchmark provided:
          - IR > 0.5  → genuine alpha, PASS
          - IR > 0.3  → marginal alpha, PASS at reduced score
          - IR < 0.3  → likely beta-capture, FAIL
          - alpha < 0 → negative alpha vs benchmark, FAIL
        """
        if not self._benchmark_usable():
            return CheckResult(
                "Information Ratio (vs Benchmark)",
                True,
                50,
                "No benchmark available — informational only, not scored",
                "The index benchmark (SPY or NIFTY) needs dated trades, market data for the period "
                "and a US/Indian equity strategy. Without it, alpha vs a passive index can't be tested.",
                "Upload dated trades so the benchmark can be aligned.",
                "Overfitting",
                scored=False,
            )

        r = self.returns
        bm = self.benchmark_returns
        len(r)

        # OLS beta
        bm_std = float(np.std(bm, ddof=1))
        if bm_std < EPSILON:
            beta = 0.0
        else:
            beta = float(np.cov(r, bm, ddof=1)[0, 1] / (bm_std**2))

        # Jensen's alpha (annualised)
        mean_r = float(np.mean(r))
        mean_bm = float(np.mean(bm))
        alpha_per_trade = mean_r - beta * mean_bm
        alpha_ann = alpha_per_trade * self.trades_per_year

        # Tracking error & IR
        active_returns = r - beta * bm
        te = float(np.std(active_returns, ddof=1)) * np.sqrt(self.trades_per_year)
        ir = alpha_ann / (te + EPSILON)

        # Correlation to benchmark
        corr = float(np.corrcoef(r, bm)[0, 1]) if bm_std > EPSILON else 0.0

        if alpha_ann < 0:
            passed, score = False, 0
            status = "⚠ Negative alpha — strategy underperforms benchmark on risk-adjusted basis"
        elif ir >= 0.5:
            passed, score = True, 100
            status = "✅ Strong alpha — IR ≥ 0.5"
        elif ir >= 0.3:
            passed, score = True, 65
            status = "⚠ Marginal alpha — IR 0.3–0.5"
        else:
            passed, score = False, 20
            status = "⚠ Weak alpha — likely beta-capture, not genuine edge"

        return CheckResult(
            "Information Ratio (vs Benchmark)",
            passed,
            round(score, 1),
            f"IR={ir:.2f} | α={alpha_ann * 100:.2f}%/yr | β={beta:.2f} | "
            f"TE={te * 100:.2f}%/yr | corr={corr:.2f} — {status}",
            "Information Ratio = Jensen's alpha / tracking error. "
            "IR < 0.3 means the strategy's excess return is insufficient "
            "to justify active risk vs a passive benchmark.",
            "If IR < 0.3, the strategy is capturing market beta not alpha. "
            "Test whether simply holding the benchmark delivers similar returns.",
            "Overfitting",
        )

    def check_beta_exposure(self) -> CheckResult:
        """
        F1 — Beta to benchmark.

        High beta (> 0.6) means returns are largely explained by market
        direction. A backtest in a bull market will always look good for
        high-beta strategies. Informational only when no benchmark supplied.
        """
        if not self._benchmark_usable():
            return CheckResult(
                "Beta Exposure (vs Benchmark)",
                True,
                50,
                "No benchmark available — informational only, not scored",
                "The index benchmark needs dated trades and market data; market-neutrality can't be tested without it.",
                "Upload dated trades so the benchmark can be aligned.",
                "Overfitting",
                scored=False,
            )

        r = self.returns
        bm = self.benchmark_returns
        bm_std = float(np.std(bm, ddof=1))
        if bm_std < EPSILON:
            beta = 0.0
        else:
            beta = float(np.cov(r, bm, ddof=1)[0, 1] / (bm_std**2))

        r_sq = float(np.corrcoef(r, bm)[0, 1] ** 2) if bm_std > EPSILON else 0.0

        if beta > 0.8:
            passed, score = False, 0
            status = "⚠ Manual Audit Required — very high market beta, strategy is essentially leveraged index exposure"
        elif beta > 0.6:
            passed, score = False, 30
            status = "⚠ High beta — most returns explained by market direction"
        elif beta > 0.4:
            passed, score = True, 65
            status = "Moderate beta — some market exposure"
        else:
            passed, score = True, 100
            status = "✅ Low beta — strategy is largely market-neutral"

        return CheckResult(
            "Beta Exposure (vs Benchmark)",
            passed,
            round(score, 1),
            f"β={beta:.2f} | R²={r_sq:.2f} — {status}",
            "Beta > 0.6 means over 60% of return variance is explained by "
            "the benchmark. A backtest during a bull market with high beta "
            "does not demonstrate genuine alpha.",
            "Reduce market exposure via hedging or use market-neutral strategy design.",
            "Overfitting",
        )
