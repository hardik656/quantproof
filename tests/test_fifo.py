"""FIFO lot-matching tests for the Kite trade-book and Kite orders parsers."""

import pytest

from services import parsers

ORDERS_HEADER = (
    "tradingsymbol,transaction_type,order_execution_time,filled_quantity,average_price,status,segment\n"
)


def _orders(*rows):
    return ORDERS_HEADER + "".join(",".join(map(str, r)) + "\n" for r in rows)


def test_orders_fifo_partial_fills_across_lots():
    raw = _orders(
        ("INFY", "BUY", "2024-01-01 09:15:00", 10, 100, "COMPLETE", "EQ"),
        ("INFY", "BUY", "2024-01-02 09:15:00", 10, 110, "COMPLETE", "EQ"),
        ("INFY", "SELL", "2024-01-03 09:15:00", 15, 120, "COMPLETE", "EQ"),
        ("INFY", "SELL", "2024-01-04 09:15:00", 5, 99, "COMPLETE", "EQ"),
    )
    df, note, flags = parsers._parse_kite_orders_csv(raw)
    # Oldest lot is consumed first: 10@100 -> 120, then 5@110 -> 120, then 5@110 -> 99.
    assert list(df["pnl"]) == pytest.approx([0.2, 10 / 110, -11 / 110], abs=1e-6)
    assert set(df["side"]) == {"long"}
    assert "3 completed trade(s)" in note
    assert not any("still open" in f for f in flags)


def test_orders_leftover_quantity_reported_as_open_position():
    raw = _orders(
        ("TCS", "BUY", "2024-01-01 09:15:00", 10, 100, "COMPLETE", "EQ"),
        ("TCS", "SELL", "2024-01-02 09:15:00", 4, 110, "COMPLETE", "EQ"),
    )
    df, _note, flags = parsers._parse_kite_orders_csv(raw)
    assert list(df["pnl"]) == pytest.approx([0.1])
    assert any("still open" in f and "TCS(6)" in f for f in flags)


def test_orders_fo_short_then_cover_and_flip_long():
    raw = _orders(
        ("NIFTYFUT", "SELL", "2024-01-01 09:15:00", 2, 200, "COMPLETE", "FO"),
        ("NIFTYFUT", "BUY", "2024-01-02 09:15:00", 3, 180, "COMPLETE", "FO"),
        ("NIFTYFUT", "SELL", "2024-01-03 09:15:00", 1, 198, "COMPLETE", "FO"),
    )
    df, _note, flags = parsers._parse_kite_orders_csv(raw)
    # Short 2@200 covered at 180 (+10%), the extra 1 bought becomes a long closed at 198 (+10%).
    assert list(df["side"]) == ["short", "long"]
    assert list(df["pnl"]) == pytest.approx([0.1, 0.1])
    assert not any("still open" in f for f in flags)


def test_orders_equity_sell_without_buy_is_orphan_not_short():
    raw = _orders(
        ("HDFC", "SELL", "2024-01-01 09:15:00", 5, 100, "COMPLETE", "EQ"),
        ("HDFC", "BUY", "2024-01-02 09:15:00", 5, 90, "COMPLETE", "EQ"),
    )
    df, _note, flags = parsers._parse_kite_orders_csv(raw)
    assert df.empty
    assert any("no matching BUY" in f for f in flags)
    assert any("HDFC(5)" in f for f in flags)


def test_orders_symbols_are_matched_independently():
    raw = _orders(
        ("A", "BUY", "2024-01-01 09:15:00", 1, 100, "COMPLETE", "EQ"),
        ("B", "BUY", "2024-01-01 09:16:00", 1, 50, "COMPLETE", "EQ"),
        ("B", "SELL", "2024-01-02 09:15:00", 1, 55, "COMPLETE", "EQ"),
        ("A", "SELL", "2024-01-03 09:15:00", 1, 90, "COMPLETE", "EQ"),
    )
    df, _, _ = parsers._parse_kite_orders_csv(raw)
    by_ticker = dict(zip(df["ticker"], df["pnl"], strict=True))
    assert by_ticker == pytest.approx({"A": -0.1, "B": 0.1})


def test_orders_ignores_rejected_zero_and_bad_rows():
    raw = _orders(
        ("X", "BUY", "2024-01-01 09:15:00", 1, 100, "REJECTED", "EQ"),
        ("X", "BUY", "2024-01-01 09:16:00", 0, 100, "COMPLETE", "EQ"),
        ("X", "BUY", "not-a-date", 1, 100, "COMPLETE", "EQ"),
        ("X", "HOLD", "2024-01-01 09:17:00", 1, 100, "COMPLETE", "EQ"),
    )
    df, note, flags = parsers._parse_kite_orders_csv(raw)
    assert df.empty
    assert "No COMPLETE orders" in note
    assert flags == []


def test_orders_only_opening_trades_returns_no_round_trips():
    raw = _orders(("X", "BUY", "2024-01-01 09:15:00", 1, 100, "COMPLETE", "EQ"))
    df, note, flags = parsers._parse_kite_orders_csv(raw)
    assert df.empty
    assert "No completed round-trip" in note
    assert any("X(1)" in f for f in flags)


def test_orders_pnl_is_clipped():
    raw = _orders(
        ("PENNY", "BUY", "2024-01-01 09:15:00", 1, 1, "COMPLETE", "EQ"),
        ("PENNY", "SELL", "2024-01-02 09:15:00", 1, 50, "COMPLETE", "EQ"),
    )
    df, _, _ = parsers._parse_kite_orders_csv(raw)
    assert df["pnl"].iloc[0] == pytest.approx(0.99)


def test_orders_missing_columns():
    with pytest.raises(ValueError, match="missing required"):
        parsers._parse_kite_orders_csv("tradingsymbol,average_price\nX,1\n")


BOOK_HEADER = "symbol,trade_date,trade_type,quantity,price,trade_id,segment\n"


def _book(*rows):
    return BOOK_HEADER + "".join(",".join(map(str, r)) + "\n" for r in rows)


def test_tradebook_fo_short_cover_flips_to_long():
    raw = _book(
        ("BANKNIFTY", "2024-01-01", "sell", 1, 100, 1, "FO"),
        ("BANKNIFTY", "2024-01-02", "buy", 3, 90, 2, "FO"),
        ("BANKNIFTY", "2024-01-03", "sell", 2, 99, 3, "FO"),
    )
    df, _note, flags = parsers._parse_kite_csv(raw)
    assert list(df["side"]) == ["short", "long"]
    assert list(df["pnl"]) == pytest.approx([0.1, 0.1])
    assert not any("still open" in f for f in flags)


def test_tradebook_equity_orphan_sell_and_open_position_are_flagged():
    raw = _book(
        ("SBIN", "2024-01-01", "sell", 5, 100, 1, "EQ"),
        ("SBIN", "2024-01-02", "buy", 7, 90, 2, "EQ"),
    )
    df, note, flags = parsers._parse_kite_csv(raw)
    assert df.empty
    assert "No completed trades" in note
    assert any("no matching buy" in f for f in flags)
    assert any("SBIN(7)" in f for f in flags)


def test_tradebook_open_remainder_after_partial_close():
    raw = _book(
        ("ITC", "2024-01-01", "buy", 10, 100, 1, "EQ"),
        ("ITC", "2024-01-02", "sell", 3, 105, 2, "EQ"),
    )
    df, _note, flags = parsers._parse_kite_csv(raw)
    assert list(df["pnl"]) == pytest.approx([0.05])
    assert any("ITC(7)" in f for f in flags)


def test_normalise_pnl():
    assert parsers._normalise_pnl(10, 100, 1, 1000) == pytest.approx(0.1)
    assert parsers._normalise_pnl(10_000, 1, 1, 1000) == pytest.approx(0.99)
    assert parsers._normalise_pnl(10, 0, 1, 1000) is None


def test_dedup_trades_removes_exact_duplicates_only():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "ticker": ["A", "A", "A"],
            "pnl": [0.1, 0.1, 0.1],
        }
    )
    assert len(parsers._dedup_trades(df)) == 2
    assert parsers._dedup_trades(pd.DataFrame()).empty


def test_tradebook_unknown_trade_type_is_flagged_not_dropped_silently():
    raw = _book(
        ("ITC", "2024-01-01", "buy", 1, 100, 1, "EQ"),
        ("ITC", "2024-01-02", "bonus", 1, 100, 2, "EQ"),
        ("ITC", "2024-01-03", "sell", 1, 110, 3, "EQ"),
    )
    df, _note, flags = parsers._parse_kite_csv(raw)
    assert list(df["pnl"]) == pytest.approx([0.1])
    assert any("unknown trade_type" in f for f in flags)


def _fill(side, qty, price, t, sym="AAPL", **extra):
    return {
        "symbol": sym,
        "side": side,
        "status": "filled",
        "filled_qty": str(qty),
        "filled_avg_price": str(price),
        "filled_at": t,
        **extra,
    }


def test_alpaca_fifo_closes_oldest_lot_first():
    df, _ = parsers.pair_alpaca_orders(
        [
            _fill("buy", 10, 100, "2024-01-01T10:00:00Z"),
            _fill("buy", 10, 200, "2024-01-02T10:00:00Z"),
            _fill("sell", 15, 150, "2024-01-03T10:00:00Z"),
        ]
    )
    assert list(df["qty"]) == [10, 5]
    assert list(df["pnl"]) == pytest.approx([0.5, -0.25])


def test_alpaca_sell_first_opens_short_and_adds_to_it():
    df, flags = parsers.pair_alpaca_orders(
        [
            _fill("sell", 1, 100, "2024-01-01T10:00:00Z"),
            _fill("sell", 1, 120, "2024-01-02T10:00:00Z"),
            _fill("buy", 3, 110, "2024-01-03T10:00:00Z"),
        ]
    )
    # Short 1@100 covered at 110 (-10%), short 1@120 covered at 110 (+8.33%); 1 share stays open long.
    assert list(df["side"]) == ["short", "short"]
    assert list(df["pnl"]) == pytest.approx([-0.1, 10 / 120], abs=1e-6)
    assert any("short entry" in f for f in flags)


def test_alpaca_rejects_malformed_orders():
    bad = [
        _fill("buy", "abc", 100, "2024-01-01T10:00:00Z"),
        _fill("buy", -1, 100, "2024-01-01T10:00:00Z"),
        _fill("buy", 1, "-", "2024-01-01T10:00:00Z"),
        _fill("buy", 1, "x", "2024-01-01T10:00:00Z"),
        _fill("buy", 1, 0, "2024-01-01T10:00:00Z"),
        _fill("hold", 1, 100, "2024-01-01T10:00:00Z"),
        _fill("buy", 1, 100, ""),
    ]
    df, flags = parsers.pair_alpaca_orders(bad)
    assert df.empty
    assert flags == ["No filled orders found"]


def test_alpaca_accepts_timestamp_objects_and_comma_decimals():
    import pandas as pd

    df, _ = parsers.pair_alpaca_orders(
        [
            _fill("buy", 1, "100,0", pd.Timestamp("2024-01-01 10:00", tz="UTC")),
            _fill("sell", 1, "110,0", pd.Timestamp("2024-01-02 10:00")),
        ]
    )
    assert list(df["pnl"]) == pytest.approx([0.1])
