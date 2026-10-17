"""Tests for the checks wired into QuantProofValidator.run() in engine v6.3:
autocorrelation, information ratio, beta, 0.3% slippage, commission drag,
partial fills, live gap, the 5-gate compliance check and the 4 plausibility checks."""

import numpy as np
import pandas as pd
import pytest

from validator import QuantProofValidator

SCORED_CATEGORIES = {"Overfitting", "Risk", "Regime", "Execution", "Compliance"}


def _df(returns, dated=True, start="2019-01-01"):
    data = {"pnl": np.asarray(returns, dtype=float)}
    if dated:
        data = {"date": pd.date_range(start, periods=len(returns), freq="B"), **data}
    return pd.DataFrame(data)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


@pytest.fixture(scope="module")
def good():
    rng = np.random.default_rng(11)
    return _df(rng.normal(0.0015, 0.01, 500))


@pytest.fixture(scope="module")
def good_report(good):
    return QuantProofValidator(good, n_trials=10).run()


def test_run_contains_37_scored_and_4_plausibility_checks(good_report):
    cats = [c.category for c in good_report.checks]
    assert sum(c in SCORED_CATEGORIES for c in cats) == 37
    assert cats.count("Plausibility") == 4
    assert {
        "Overfitting": 11,
        "Risk": 11,
        "Regime": 6,
        "Execution": 6,
        "Compliance": 3,
    } == {k: cats.count(k) for k in SCORED_CATEGORIES}
    names = [c.name for c in good_report.checks]
    assert len(names) == len(set(names)), "check names must be unique"


# ── Benchmark checks ────────────────────────────────────────────────────────


def test_benchmark_checks_are_informational_without_benchmark(good_report):
    for name in ("Information Ratio (vs Benchmark)", "Beta Exposure (vs Benchmark)"):
        c = _check(good_report, name)
        assert c.scored is False and c.score == 50 and "informational" in c.value


def test_unscored_checks_do_not_change_the_score(good):
    v = QuantProofValidator(good, n_trials=10)
    report = v.run()
    without = [c for c in report.checks if c.scored]
    assert v._calculate_score(report.checks) == v._calculate_score(without)


def test_misaligned_or_constant_benchmark_degrades_gracefully(good):
    for bm in (np.zeros(len(good)), np.ones(10) * 0.01, np.full(len(good), np.nan)):
        r = QuantProofValidator(good, benchmark_returns=bm).run()
        assert _check(r, "Beta Exposure (vs Benchmark)").scored is False


def test_beta_and_ir_with_market_tracking_strategy():
    rng = np.random.default_rng(5)
    bm = rng.normal(0.0005, 0.01, 400)
    strat = 1.0 * bm + rng.normal(-0.0002, 0.002, 400)  # beta ~1, negative alpha
    r = QuantProofValidator(_df(strat), benchmark_returns=bm).run()
    beta = _check(r, "Beta Exposure (vs Benchmark)")
    assert beta.scored and not beta.passed and "Manual Audit Required" in beta.value
    ir = _check(r, "Information Ratio (vs Benchmark)")
    assert ir.scored and not ir.passed


def test_beta_and_ir_with_market_neutral_alpha():
    rng = np.random.default_rng(6)
    bm = rng.normal(0.0005, 0.01, 400)
    strat = rng.normal(0.002, 0.008, 400)
    r = QuantProofValidator(_df(strat), benchmark_returns=bm).run()
    assert _check(r, "Beta Exposure (vs Benchmark)").passed
    assert _check(r, "Information Ratio (vs Benchmark)").passed


# ── Autocorrelation ─────────────────────────────────────────────────────────


def test_autocorrelation_flags_ar1_series():
    rng = np.random.default_rng(8)
    e = rng.normal(0, 0.01, 400)
    r = np.zeros(400)
    for i in range(1, 400):
        r[i] = 0.6 * r[i - 1] + e[i]
    c = _check(QuantProofValidator(_df(r + 0.001)).run(), "Autocorrelation (Ljung-Box)")
    assert not c.passed and "AR(1)" in c.value


def test_autocorrelation_passes_iid(good_report):
    assert _check(good_report, "Autocorrelation (Ljung-Box)").passed


# ── Execution ───────────────────────────────────────────────────────────────


def test_slippage_03_is_stricter_than_01():
    rng = np.random.default_rng(9)
    r = QuantProofValidator(_df(rng.normal(0.002, 0.01, 300))).run()
    s1 = _check(r, "Slippage Impact (0.1% base)")
    s3 = _check(r, "Slippage Impact (0.3% base)")
    assert s3.score < s1.score


def test_commission_tier_follows_frequency(good_report):
    weekly = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-02", periods=200, freq="W-FRI"),
            "pnl": np.random.default_rng(2).normal(0.004, 0.02, 200),
        }
    )
    assert "institutional (2bps)" in _check(QuantProofValidator(weekly).run(), "Commission Drag").value
    assert "retail equity (10bps)" in _check(good_report, "Commission Drag").value
    hf = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02 09:30", periods=3000, freq="15min"),
            "pnl": np.random.default_rng(1).normal(0.0002, 0.002, 3000),
        }
    )
    assert "HFT/spread (20bps)" in _check(QuantProofValidator(hf).run(), "Commission Drag").value


def test_partial_fills_depend_on_the_strategy():
    rng = np.random.default_rng(10)
    strong = _check(QuantProofValidator(_df(rng.normal(0.004, 0.01, 300))).run(), "Partial Fill Simulation")
    thin = _check(QuantProofValidator(_df(rng.normal(0.0002, 0.01, 300))).run(), "Partial Fill Simulation")
    assert strong.passed and strong.score > 95
    assert not thin.passed and thin.score < strong.score
    assert "80% fill rate" in strong.value


def test_partial_fills_deterministic(good):
    a = _check(QuantProofValidator(good).run(), "Partial Fill Simulation")
    b = _check(QuantProofValidator(good).run(), "Partial Fill Simulation")
    assert a == b


def test_live_gap_applies_40pct_sharpe_haircut(good_report):
    c = _check(good_report, "Live Trading Gap Estimate")
    bt, live = (float(x.split(": ")[1]) for x in c.value.split(" → "))
    assert live == pytest.approx(bt * 0.6, abs=0.011)


# ── Compliance gate ─────────────────────────────────────────────────────────


def test_compliance_gate_counts_gates(good_report):
    c = _check(good_report, "Prop Firm Compliance Gate (5-gate)")
    assert "/5 gates" in c.value


def test_compliance_gate_does_not_trigger_prop_firm_soft_cap():
    # A failing quick gate must not apply the per-firm compliance cap (75).
    rng = np.random.default_rng(12)
    v = QuantProofValidator(_df(rng.normal(0.003, 0.01, 40)))
    report = v.run()
    gate = _check(report, "Prop Firm Compliance Gate (5-gate)")
    others = [c for c in report.checks if c is not gate]
    forced_fail = type(gate)(**{**vars(gate), "passed": False, "score": 0})
    s_fail, _ = v._calculate_score(others + [forced_fail])
    forced_pass = type(gate)(**{**vars(gate), "passed": True, "score": 0})
    s_pass, _ = v._calculate_score(others + [forced_pass])
    assert s_fail == s_pass


# ── Plausibility ────────────────────────────────────────────────────────────


def test_plausibility_checks_really_run_and_flag_smooth_curves():
    rng = np.random.default_rng(13)
    smooth = _df(rng.normal(0.004, 0.0015, 300))
    report = QuantProofValidator(smooth).run()
    eq = _check(report, "Equity Curve Plausibility")
    assert not eq.passed and "Manual Audit Required" in eq.value
    assert "manual audit" in report.plausibility_summary
    assert any("Equity Curve Plausibility" in f for f in report.audit_flags)
    assert report.score <= 35  # plausibility cap


def test_plausibility_clean_strategy(good_report):
    assert good_report.plausibility_summary.startswith("✅")
    assert good_report.audit_flags == []


def test_review_recommended_is_reported():
    rng = np.random.default_rng(14)
    r = QuantProofValidator(_df(rng.normal(0.004, 0.006, 300))).run()  # Sharpe ~10
    sp = _check(r, "Sharpe Plausibility")
    assert "Review Recommended" in sp.value or "Manual Audit Required" in sp.value
    assert not r.plausibility_summary.startswith("✅")


def test_plausibility_not_in_api_checks(client):
    import io

    rng = np.random.default_rng(15)
    csv = _df(rng.normal(0.002, 0.01, 60)).to_csv(index=False).encode()
    res = client.post("/validate", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")})
    assert res.status_code == 200, res.text
    body = res.json()
    cats = {c["category"] for c in body["checks"]}
    assert "Plausibility" not in cats
    assert len(body["checks"]) == 37
    assert all("scored" in c for c in body["checks"])


def test_frequency_return_plausibility_uses_percent():
    hf = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02 09:30", periods=3000, freq="1min"),
            "pnl": np.random.default_rng(3).normal(0.0005, 0.001, 3000),
        }
    )
    v = QuantProofValidator(hf)
    c = v.check_frequency_return_plausibility()
    assert v.trades_per_year > 20_000
    assert not c.passed and "Manual Audit Required" in c.value
    calm = QuantProofValidator(_df(np.random.default_rng(4).normal(0.001, 0.01, 300)))
    c2 = calm.check_frequency_return_plausibility()
    assert c2.passed and "% annual" in c2.value
