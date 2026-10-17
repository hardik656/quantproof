import pandas as pd
import pytest

from services import benchmarks, parsers


def test_kite_detection():
    df = pd.DataFrame(columns=["trade_date", "trade_type", "quantity", "price", "trade_id"])
    assert parsers._is_kite_csv(df)
    assert not parsers._is_kite_csv(pd.DataFrame(columns=["date", "pnl"]))


KITE = """symbol,trade_date,trade_type,quantity,price,trade_id,segment
INFY,2024-01-01,buy,10,100,1,EQ
INFY,2024-01-02,buy,10,110,2,EQ
INFY,2024-01-03,sell,15,120,3,EQ
INFY,2024-01-04,sell,5,90,4,EQ
"""


def test_kite_fifo_lot_matching():
    df, _note, _flags = parsers._parse_kite_csv(KITE)
    pnls = sorted(round(p, 4) for p in df["pnl"])
    # FIFO: 10@100 closed at 120 (+20%), 5@110 at 120 (+9.09%), 5@110 at 90 (-18.18%)
    assert pnls == sorted([0.2, round(10 / 110, 4), round(-20 / 110, 4)])


def test_kite_short_cover():
    raw = (
        "symbol,trade_date,trade_type,quantity,price,trade_id,segment\n"
        "NIFTYFUT,2024-01-01,sell,1,200,1,FO\n"
        "NIFTYFUT,2024-01-02,buy,1,180,2,FO\n"
    )
    df, _, _ = parsers._parse_kite_csv(raw)
    assert len(df) == 1
    assert df["pnl"].iloc[0] == pytest.approx(0.1)


def test_kite_missing_columns():
    with pytest.raises(ValueError, match="missing required"):
        parsers._parse_kite_csv("trade_date,price\n2024-01-01,1\n")


def test_kite_skips_invalid_rows():
    raw = KITE + "INFY,2024-01-05,buy,0,100,5,EQ\n"
    _, _, flags = parsers._parse_kite_csv(raw)
    assert any("Skipped row" in f for f in flags)


def _order(side, qty, price, t, sym="AAPL"):
    return {
        "symbol": sym,
        "side": side,
        "status": "filled",
        "filled_qty": str(qty),
        "filled_avg_price": str(price),
        "filled_at": t,
    }


def test_alpaca_pairing_fifo():
    orders = [
        _order("buy", 10, 100, "2024-01-01T10:00:00Z"),
        _order("buy", 10, 200, "2024-01-02T10:00:00Z"),
        _order("sell", 20, 150, "2024-01-03T10:00:00Z"),
    ]
    df, _flags = parsers.pair_alpaca_orders(orders)
    assert len(df) >= 1
    assert "pnl" in df.columns


def test_alpaca_ignores_unfilled():
    orders = [{**_order("buy", 1, 1, "2024-01-01T00:00:00Z"), "status": "canceled"}]
    df, _ = parsers.pair_alpaca_orders(orders)
    assert len(df) == 0


def test_alpaca_ui_detection():
    df = pd.DataFrame(columns=["Symbol", "Side", "Avg. Fill Price", "Filled At"])
    assert parsers._is_alpaca_ui_export(df)


def test_alpaca_ui_date():
    assert parsers._parse_alpaca_ui_date("-") is None
    ts = parsers._parse_alpaca_ui_date("Feb 24, 2026, 08:40:53 PM")
    assert ts.year == 2026 and ts.month == 2


def test_options_and_mt5_detection():
    assert parsers._is_options_csv(pd.DataFrame(columns=["strike", "expiry", "pnl"]))
    assert not parsers._is_options_csv(pd.DataFrame(columns=["date", "pnl"]))
    assert parsers._is_mt5_csv(pd.DataFrame(columns=["Deal", "Direction", "Commission", "Swap"]))


def test_generic_csv_parse():
    rows = "\n".join(f"2024-01-{d:02d},{0.01 if d % 2 else -0.005}" for d in range(1, 29))
    df, _note = parsers.smart_parse_csv(("date,pnl\n" + rows).encode())
    assert len(df) == 28
    assert "pnl" in df.columns


def test_generic_csv_semicolon_delimiter():
    rows = "\n".join(f"2024-01-{d:02d};{0.01}" for d in range(1, 21))
    df, _ = parsers.smart_parse_csv(("date;pnl\n" + rows).encode())
    assert len(df) == 20


def test_smart_parse_routes_kite():
    df, note = parsers.smart_parse_csv(KITE.encode())
    assert len(df) == 3


def test_dedup_trades():
    df = pd.DataFrame({"date": ["2024-01-01"] * 2, "pnl": [0.1, 0.1]})
    assert len(parsers._dedup_trades(df)) == 1
    assert parsers._dedup_trades(pd.DataFrame()).empty


def test_normalise_pnl():
    out = parsers._normalise_pnl(1_000_000.0, 1.0, 1.0, 10_000.0)
    assert -0.99 <= out <= 0.99
    assert parsers._normalise_pnl(5.0, 0.0, 0.0, 0.0) is None


def test_assign_regimes_offline_fallback(profitable_df):
    df, note = benchmarks.assign_regimes(profitable_df.copy(), "spy")
    assert "regime" in df.columns
    assert set(df["regime"].dropna()) <= {"BULL", "BEAR", "CONSOLIDATION"}


def test_benchmark_type_detection():
    df = pd.DataFrame({"ticker": ["RELIANCE.NS", "INFY.NS"], "pnl": [0.1, 0.2]})
    assert benchmarks.get_benchmark_type(df, "INR") == "nifty"
    assert benchmarks.get_benchmark_type(pd.DataFrame({"pnl": [1]}), "USD") == "spy"
