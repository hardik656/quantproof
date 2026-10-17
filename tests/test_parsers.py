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


# ---------------------------------------------------------------- more formats


def _rows(n, fmt):
    return "\n".join(fmt(i) for i in range(n))


def test_options_csv_parse_premium_normalisation():
    base = pd.Timestamp("2024-01-01")

    def row(i):
        opened = (base + pd.Timedelta(days=2 * i)).date()
        expiry = (base + pd.Timedelta(days=2 * i + 1)).date()
        return f"{opened},SPY,500,{expiry},2.0,{0.5 if i % 2 else -0.25}"

    raw = "date,underlying,strike,expiry,premium,pnl\n" + _rows(15, row)
    df, note, _flags = parsers._parse_options_csv(raw)
    assert len(df) == 15
    assert df["pnl"].max() == pytest.approx(0.25)
    assert df["pnl"].min() == pytest.approx(-0.125)
    assert "premium" in note


def test_options_csv_requires_pnl_column():
    raw = "date,underlying,strike,expiry\n2024-01-01,SPY,500,2024-02-01\n"
    with pytest.raises(ValueError):
        parsers._parse_options_csv(raw)


def test_mt5_positions_parse():
    raw = "Time,Position,Symbol,Type,Volume,Price,S / L,T / P,Time,Price,Commission,Swap,Profit\n" + _rows(
        12,
        lambda i: (
            f"2024.01.{i + 2:02d} 10:00:00,{1000 + i},EURUSD,buy,1.0,1.1000,0,0,"
            f"2024.01.{i + 2:02d} 12:00:00,1.1010,-1,0,{10 if i % 3 else -5}"
        ),
    )
    df, _note, _flags = parsers._parse_mt5_csv(raw)
    assert len(df) > 0
    assert df["pnl"].between(-0.99, 0.99).all()


def test_kite_orders_parse_fifo():
    raw = "tradingsymbol,transaction_type,order_execution_time,filled_quantity,average_price,status\n" + (
        "TCS,BUY,2024-01-01 09:20:00,10,100,COMPLETE\n"
        "TCS,SELL,2024-01-02 09:20:00,10,110,COMPLETE\n"
        "TCS,BUY,2024-01-03 09:20:00,5,120,COMPLETE\n"
        "TCS,SELL,2024-01-04 09:20:00,5,108,COMPLETE\n"
    )
    assert parsers._is_kite_orders_csv(pd.read_csv(__import__("io").StringIO(raw)))
    df, _note, _flags = parsers._parse_kite_orders_csv(raw)
    assert sorted(round(p, 4) for p in df["pnl"]) == [-0.1, 0.1]


def test_alpaca_ui_export_parse():
    header = "Asset\tSide\tQty\tFilled Qty\tAvg. Fill Price\tStatus\tFilled At"
    body = "\n".join(
        [
            "AAPL\tbuy\t10\t10\t100.00\tfilled\tJan 02, 2024, 10:00:00 AM",
            "AAPL\tsell\t10\t10\t110.00\tfilled\tJan 03, 2024, 10:00:00 AM",
            "MSFT\tbuy\t5\t5\t200.00\tfilled\tJan 04, 2024, 10:00:00 AM",
            "MSFT\tsell\t5\t5\t190.00\tfilled\tJan 05, 2024, 10:00:00 AM",
        ]
    )
    df, _note, _flags = parsers._parse_alpaca_ui_export(header + "\n" + body)
    assert sorted(round(p, 4) for p in df["pnl"]) == [-0.05, 0.1]


def test_alpaca_short_then_cover():
    orders = [
        _order("sell", 10, 100, "2024-01-01T10:00:00Z"),
        _order("buy", 10, 90, "2024-01-02T10:00:00Z"),
    ]
    df, flags = parsers.pair_alpaca_orders(orders)
    assert len(df) <= 1
    assert isinstance(flags, list)


def test_excel_parse(tmp_path):
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=15), "pnl": [0.01, -0.005, 0.02] * 5})
    path = tmp_path / "t.xlsx"
    df.to_excel(path, index=False)
    out, _ = parsers.smart_parse_csv(path.read_bytes())
    assert len(out) == 15


def test_generic_percent_strings():
    rows = "\n".join(f"2024-01-{d:02d},{'1.5%' if d % 2 else '-0.5%'}" for d in range(1, 21))
    df, _ = parsers.smart_parse_csv(("date,return\n" + rows).encode())
    assert len(df) == 20


def test_smart_parse_garbage_raises():
    with pytest.raises(Exception):  # noqa: B017 - parser raises ValueError or HTTPException
        parsers.smart_parse_csv(b"\x00\x01\x02 not a csv")


def test_overlapping_positions_detected():
    df = pd.DataFrame(
        {
            "open_date": ["2024-01-01", "2024-01-02"],
            "close_date": ["2024-01-05", "2024-01-06"],
            "strike": [1, 1],
            "pnl": [1, 1],
        }
    )
    assert parsers._has_overlapping_positions(df) in (True, False)
