import numpy as np
import pandas as pd
import pytest

import validator as v
from validator import QuantProofValidator


def test_too_few_trades_rejected():
    with pytest.raises(ValueError, match="Insufficient data"):
        QuantProofValidator(pd.DataFrame({"pnl": [0.01] * 5}))


def test_clean_normalises_column_names():
    df = pd.DataFrame({"Trade Date": pd.date_range("2024-01-01", periods=20), "Profit": [0.01] * 20})
    qv = QuantProofValidator(df)
    assert {"date", "pnl"} <= set(qv.df.columns)


def test_run_is_deterministic(profitable_df):
    a = QuantProofValidator(profitable_df, seed=1).run()
    b = QuantProofValidator(profitable_df, seed=1).run()
    assert a.score == b.score
    assert a.validation_hash == b.validation_hash


def test_score_bounds_and_ordering(profitable_df, losing_df):
    good = QuantProofValidator(profitable_df).run()
    bad = QuantProofValidator(losing_df).run()
    assert 0 <= bad.score <= 100 and 0 <= good.score <= 100
    assert good.score > bad.score


def test_report_structure(profitable_df):
    rep = QuantProofValidator(profitable_df).run()
    # NOTE: engine currently emits 29 CheckResults for a dated return series
    assert len(rep.checks) >= 25
    assert len(rep.crash_sims) == 5
    for c in rep.checks:
        assert isinstance(c.passed, bool | np.bool_)


def test_enhanced_run(profitable_df):
    rep = QuantProofValidator(profitable_df).run(enhanced=True)
    assert 0 <= rep.score <= 100


def test_max_drawdown_known_value():
    r = np.array([0.1, -0.5, 0.2])
    dd = v.calculate_max_drawdown(r)
    assert dd == pytest.approx(-0.5, abs=0.06) or dd == pytest.approx(0.5, abs=0.06)


def test_win_rate():
    assert v.calculate_win_rate(np.array([1.0, -1.0, 2.0, -0.5])) == pytest.approx(
        50.0
    )  # returned as a percentage


def test_sharpe_sign():
    rng = np.random.default_rng(0)
    assert v.calculate_sharpe(rng.normal(0.01, 0.01, 200)) > 0
    assert v.calculate_sharpe(rng.normal(-0.01, 0.01, 200)) < 0


def test_sharpe_zero_variance_is_finite():
    assert np.isfinite(v.calculate_sharpe(np.full(50, 0.01)))


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.002, 0.01, 300)
    sr = v.calculate_sharpe(r)
    few = v.calculate_deflated_sharpe(r, sr, n_trials=1)
    many = v.calculate_deflated_sharpe(r, sr, n_trials=100)
    assert many["dsr"] <= few["dsr"]
    assert many["sr_benchmark"] >= few["sr_benchmark"]


def test_sortino_cvar_calmar_ulcer_finite():
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.02, 250)
    for fn in (v.calculate_sortino, v.calculate_cvar, v.calculate_calmar, v.calculate_ulcer_index):
        out = fn(r)
        assert np.isfinite(out[0] if isinstance(out, tuple) else out)


def test_convert_numpy_roundtrip():
    out = v.convert_numpy({"a": np.float64(1.5), "b": [np.int64(2)], "c": np.array([1, 2])})
    assert out["a"] == 1.5 and out["b"] == [2]
