# QuantProof — Validation Methodology

**Institutional-grade backtest validation for retail algo traders.**

[→ Try it live](https://quantproof-frontend.onrender.com) &nbsp;|&nbsp; [quantproof@gmail.com](mailto:quantproof@gmail.com)

---

## What is QuantProof?

QuantProof is a free backtest validator. You upload your strategy's trade results — as a CSV, or connected directly to your Alpaca, IBKR, or Zerodha Kite account — and it runs 40 institutional checks plus 5 historical crash simulations.

The output is a **Fundable Score** from 0–100 and a letter grade, along with a breakdown of every check showing exactly what passed, what failed, and why.

It is built for traders who want to know if their strategy is genuinely ready — before risking real money or paying for a prop firm challenge.

---

## Why open methodology?

Every scoring system should be explainable. A number you can't interrogate is a number you can't trust. This document explains every check category, what it measures, and the academic research it is based on.

**What this document does not include:** scoring weights, exact thresholds, formula implementations, or source code. Those live in the engine. This document explains the *what* and *why*, not the *how*.

---

## The Fundable Score

A weighted average across five categories:

| Category | Weight | What it measures |
|---|---|---|
| Risk Management | 38% | Drawdown, tail risk, ruin probability, recovery |
| Overfitting Detection | 22% | Statistical significance, path stability, benchmark alpha. IR + Beta auto-scored via SPY when dates present |
| Compliance | 15% | Prop firm and institutional eligibility |
| Regime Robustness | 14% | Performance across bull, bear, consolidation |
| Execution Reality | 11% | Slippage, commissions, market impact, capacity |

**Score caps apply** to prevent misleading results:
- No date column → cap at 60 (regime checks impossible without timestamps)
- Sharpe >5 with no dates → cap at 50 (statistically unverifiable)
- Single-regime backtest (>85% trades in one regime) → cap at 55
- Institutional compliance failure → cap at 60
- Plausibility Manual Audit Required → cap at 35

---

## Category 1 — Overfitting Detection (22%) — 11 checks (IR and Beta are informational and unscored when no benchmark can be aligned)

The most common failure mode in backtesting. A strategy optimised on historical data will always look good in-sample. These checks test whether the edge is real or memorised.

### Checks in this category

**Sharpe Ratio Decay**
Splits the backtest in half. Compares in-sample vs out-of-sample Sharpe. Also measures mean return decay independently. Uses the worse of the two as the effective decay score. High decay = fitted to noise.

**Monte Carlo Robustness**
Three-component test: bootstrap mean stability (300 resamples), sequence independence (200 permutations), worst-5th-percentile equity curve (500 permutations). A robust edge holds across all three.

**Profit / Loss Concentration**
What percentage of total profit comes from the top 1% of trades? What happens to Sharpe when those trades are removed? Also checks loss concentration. A strategy depending on 2–3 outlier trades is not a strategy — it is luck.

**CPCV Path Stability**
Combinatorial Purged Cross-Validation with 6 splits, 2 test folds, and embargo gaps to prevent data leakage at fold boundaries. Generates multiple walk-forward paths and measures Sharpe standard deviation across paths. Thresholds: path_std < 1.5 = robust, < 2.5 = acceptable, ≥ 2.5 = fails. Calibrated empirically (p95 of random-walk ≈ 2.05 at n=300).

> Reference: López de Prado, *Advances in Financial Machine Learning* (2018), Ch.12

**Bootstrap Stability**
500 bootstrap resamples. Reports percentage with positive mean return. A robust edge shows positive expectancy in the vast majority of resampled scenarios.

**Deflated Sharpe Ratio (DSR)**
Corrects Sharpe for non-normality (skewness and excess kurtosis), finite sample bias, and multiple testing. Uses the Harvey-Liu-Zhu framework to estimate the minimum Sharpe required for statistical significance given how many strategies were tested.

> References:
> - Bailey, Borwein, López de Prado & Zhu — *The Deflated Sharpe Ratio* (2014), Journal of Portfolio Management
> - Harvey, Liu & Zhu — *…and the Cross-Section of Expected Returns* (2016), Review of Financial Studies

**Sharpe Confidence Interval**
95% CI for the annualised Sharpe using the Mertens (2002) standard error formula, corrected for skewness and kurtosis. The lower CI bound is the pessimistic estimate — this is what matters for deployment decisions.

**Minimum Backtest Length**
Harvey-Liu-Zhu minimum observations for statistical significance at the strategy's Sharpe level. A Sharpe of 2.0 on 30 trades is not credible. This check tells you exactly how many trades you need.

**Information Ratio (vs SPY Benchmark)**
When your trade data includes dates, SPY is automatically fetched and aligned to your trade dates. Computes OLS beta, Jensen's alpha (annualised), tracking error, and IR. IR > 0.5 = strong alpha, 0.3-0.5 = marginal, < 0.3 = likely beta-capture. Falls back to score=50 informational if no dates or SPY unavailable.

**Beta Exposure (vs SPY Benchmark)**
SPY is automatically inferred as benchmark when dates are present. Computes market beta and R². Beta > 0.8 triggers Manual Audit Required. Beta > 0.6 = high market exposure. Beta < 0.4 = largely market-neutral. Score=50 informational without dates.

**Autocorrelation (Ljung-Box + Newey-West)**
Ljung-Box Q-statistic at lags 1, 5, and 10 tests for serial correlation in returns. If significant autocorrelation is detected, Newey-West HAC standard error replaces the IID formula and the inflation factor is reported. Serial correlation invalidates DSR, CI, and Bootstrap tests.

> References:
> - Ljung & Box — *On a Measure of Lack of Fit in Time Series Models* (1978), Biometrika 65(2)
> - Newey & West — *A Simple Positive Semi-Definite HAC Covariance Matrix* (1987), Econometrica 55(3)
> - Lo — *The Statistics of Sharpe Ratios* (2002), Financial Analysts Journal

---

## Category 2 — Risk Management (38%)

The largest category by weight. Real alpha can coexist with risk management failures that cause blow-ups in live trading.

### Checks in this category

**Max Drawdown** — Peak-to-trough equity decline. Institutional threshold: <20%. Prop firm threshold: <10%.

**Calmar Ratio** — CAGR / Max Drawdown using geometric annualisation. Threshold: >1.5.

**Value at Risk (VaR 99%)** — 99th percentile loss as a ratio to strategy volatility (VaR/σ). Normal distribution ≈ 2.33σ. Ratios above 4.0σ indicate extreme fat tails.

**CVaR / Expected Shortfall** — Average loss in the worst 5% of outcomes. Kernel density estimation for small samples. Basel III risk measure.

**Sortino Ratio** — Return divided by downside deviation only. Correct Sortino & van der Meer (1991) formula using all returns, not just negative ones.

**Max Losing Streak** — Longest consecutive losing sequence. Zero losing streaks over 20+ trades are flagged as implausible — a known signature of look-ahead bias.

**Recovery Factor** — Total gross profits / total gross losses. Threshold: >1.5.

**Absolute Sharpe Ratio** — Annualised Sharpe with sample standard deviation (ddof=1). Institutional minimum: 1.0.

**Probability of Ruin** — Generalised gambler's ruin at N=50 capital units (2% risk per trade), incorporating the reward-to-risk ratio in the calculation.

> Reference: Thorp — *The Kelly Criterion in Blackjack, Sports Betting and the Stock Market* (2008)

**Drawdown Duration & Recovery** — Time in maximum drawdown, scaled to backtest length. Threshold scales from 180 days minimum (prop firm standard) to 730 days for multi-year institutional backtests.

**Ulcer Index** — Root mean square of all drawdown percentages. Captures depth and duration of all drawdowns, not just the worst one. Also reports Martin Ratio (Sharpe / Ulcer Index).

---

## Category 3 — Regime Robustness (14%)

A strategy backtested in a bull market only is not tested. These checks verify the edge holds across market conditions.

### Checks in this category

**Bull / Bear / Consolidation Performance**
When date data is present, classifies trades using actual market regime periods (S&P 500 bull/bear cycles 2000–2026). Without dates, uses rolling return proxy. Sharpe and win rate reported for each regime separately.

**Volatility Spike Stress Test**
3x volatility spike using Student's t-distribution (fat tails, df=3) with AR(1) autocorrelation to model crisis momentum. Strategy-specific seed ensures each strategy receives a unique stress path. Measures Sharpe degradation and stress drawdown.

**Performance Consistency**
Percentage of rolling windows with positive mean return throughout the backtest.

**Regime Coverage**
Checks whether the backtest includes trades across multiple market conditions.

---

## Category 4 — Execution Reality (11%)

Backtests assume perfect fills at the exact price. Reality does not.

### Checks in this category

**Slippage Impact (0.1% and 0.3%)** — Two thresholds: 0.1% for institutional/large-cap, 0.3% for retail/small-cap/illiquid. Both scale with trade frequency. Net return after slippage must remain positive.

**Commission Drag** — Three tiers by frequency: 2bps institutional, 10bps retail equity, 20bps HFT. Measured as a percentage of gross turnover.

**Partial Fill Simulation** — Each order fills with 80% probability, over 500 seeded runs on your own trades. Passes if the mean return stays positive in at least 90% of runs and the 5th-percentile Sharpe is above zero.

**Live Trading Gap Estimate** — 40% Sharpe degradation from backtest to live, based on empirical research.

**Impact-Adjusted Capacity** — Almgren-Chriss market impact model. Sharpe at $100k, $1M, $10M, $100M AUM.

> Reference: Almgren & Chriss — *Optimal Execution of Portfolio Transactions* (2000), Journal of Risk 3(2)

---

## Category 5 — Compliance (15%)

Two separate standards for different use cases.

**Prop Firm Compliance Gate (5-gate quick check)**
Fast binary gate: positive mean return, Sharpe ≥ 1.0, max drawdown < 20%, win rate > 45%, trades > 50. Requires 4/5 in standard mode, all 5 in strict mode. Quick pre-filter before the detailed per-firm check.

**Prop Firm Compliance (FTMO / Topstep / The5ers; five firms in enhanced mode)** — Challenge rules: max drawdown ≤10%, profit target ≥6-10%, Sharpe ≥1.0. Reports which firms the strategy is eligible for. Note: only broker-verified validations can be used for actual prop firm compliance submissions.

**Institutional Compliance (CTA / Fund Standard)** — Sharpe ≥1.0, max drawdown <25%, Calmar ≥1.0. More appropriate than prop firm rules for multi-year backtests.

---

## Operational Diagnostics (2, not in weighted score)

Computed on every run (`ValidationReport.alpha_decay`, `ValidationReport.overfit_profile`). They don't contribute to the five weighted categories and are not yet shown in the web report or PDF.

**Alpha Decay (Half-Life)**
Compares first-half vs second-half Sharpe. Fits exponential decay to autocorrelation function to estimate alpha half-life. Fails if second-half Sharpe decays >60% or goes very negative.

**Symbolic Overfit Detection**
Compares equity curve characteristics (smoothness, entropy, kurtosis) against 50 noise baselines. Polynomial regression detects pathological predictability. Reports overfit score 0–100 and complexity-penalised adjusted Sharpe.

---

## Single-Regime Cap

When >85% of dated trades fall within one market regime, the score is capped at 55. This is a score cap, not a separate check row. Requires a date column.

---

## Crash Simulations (informational — do not affect Fundable Score)

Five scenarios simulated using fat-tailed Student's t-distribution with AR(1) crisis autocorrelation. Each strategy gets a unique simulation path based on its own volatility and autocorrelation profile.

| Scenario | Period | Market Drop |
|---|---|---|
| Global Financial Crisis | Sep 2008 – Mar 2009 | −56% |
| COVID Crash | Feb 2020 – Mar 2020 | −34% |
| Rate Hike Bear Market | Jan 2022 – Dec 2022 | −19% |
| Flash Crash | May 6, 2010 | −9% |
| LTCM / Russia Default | Aug–Sep 1998 | −19% |

Strategies without date data receive `[UNCALIBRATED]` labels — market regime cannot be verified without timestamps. Strategies with high AR(1) autocorrelation receive a 30% exposure multiplier — autocorrelated strategies experience harder unwind when trends break.

---

## Plausibility Checks (4 checks — informational)

Flag data integrity issues. These run on every validation and are reported in the plausibility summary and audit flags, not in the weighted score. A Manual Audit Required result caps the score at 35.

- **Sharpe Plausibility** — Sharpe >10 triggers Manual Audit Required. No documented institutional strategy has sustained Sharpe >3.0 over 5+ years.
- **Frequency-Return Plausibility** — HFT + extreme annual returns flag impossible liquidity assumptions.
- **Equity Curve Smoothness** — Suspiciously smooth curves with very low max drawdown flag look-ahead bias.
- **Kelly Plausibility** — Kelly fraction >1.0 is mathematically impossible in real strategies.

---

## Supported Data Sources

| Source | Verification | Compliance Eligible |
|---|---|---|
| Generic CSV (pnl column) | Self-reported | No |
| Zerodha Kite TradeBook / Orders CSV | Self-reported | No |
| TradingView Webhook | Self-reported | No |
| Alpaca API (paper or live) | Broker-verified | Yes |
| Interactive Brokers Flex API | Broker-verified | Yes |

Only broker-verified validations can be used for prop firm compliance certificates. CSV and Kite validations are for methodology analysis only.

---

## Engine Version

Current: **v6.3** — Methodology date: 2026-10-17

v6.3: `run()` executes all 37 scored checks and the 4 plausibility checks described here. Before, 8 of the scored checks and all plausibility checks were defined but not run. Partial fills are simulated on the strategy's own trades. Score shifts: [docs/engine-v6.3-score-shift.md](docs/engine-v6.3-score-shift.md).

v6.2:
- Information Ratio and Beta Exposure checks (benchmark-supplied)
- Hardened crash simulation exposure model with AR(1) penalty
- [UNCALIBRATED] labels for undated crash simulations
- Newey-West HAC correction for autocorrelated returns
- Ljung-Box serial correlation test
- Score cap at 50 for high-Sharpe undated strategies

---

*Not financial advice. Validation of historical backtest properties does not guarantee future performance.*

[→ Validate your strategy](https://quantproof.net) &nbsp;|&nbsp; [support@quantproof.net](mailto:support@quantproof.net)
