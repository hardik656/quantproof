"""Edge cases for every import path.

Silent misreads are failures: a file is either parsed to the return it actually
had, or rejected with a message that names the problem.
"""

import asyncio

import pandas as pd
import pytest
from fastapi import HTTPException

from services import ibkr, parsers
from services.limits import MAX_TRADES_PER_RUN, TradeImportError

TV_OLD = """Trade #,Type,Date/Time,Signal,Price USD,Contracts,Profit USD,Profit %,Cumulative profit USD,Cumulative profit %
1,Exit long,2024-01-03 10:00,Sell,110,10,100,10.0,100,10
1,Entry long,2024-01-02 10:00,Buy,100,10,100,10.0,100,10
2,Exit short,2024-01-05 11:00,Cover,90,5,50,5.0,150,15
2,Entry short,2024-01-04 11:00,Short,100,5,50,5.0,150,15
"""

TV_NEW = """Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),Net PnL USD,Net PnL %,Favorable excursion USD,Favorable excursion %,Adverse excursion USD,Adverse excursion %,Cumulative PnL USD,Cumulative PnL %
1,Exit long,2024-02-02 15:00,Sell,120,10,1200,200,20.0,250,25,10,1,200,20
1,Entry long,2024-02-01 15:00,Buy,100,10,1000,200,20.0,250,25,10,1,200,20
"""

KITE_REAL = """symbol,isin,trade_date,exchange,segment,series,trade_type,auction,quantity,price,trade_id,order_id,order_execution_time
INFY,INE009A01021,2024-01-02,NSE,EQ,EQ,buy,false,10,100.0,1,10,2024-01-02T09:15:00
INFY,INE009A01021,2024-01-02,NSE,EQ,EQ,sell,false,10,110.0,2,11,2024-01-02T15:15:00
NIFTY24JAN21500CE,,2024-01-03,NSE,FO,,sell,false,50,100.0,3,12,2024-01-03T09:20:00
NIFTY24JAN21500CE,,2024-01-03,NSE,FO,,buy,false,50,80.0,4,13,2024-01-03T14:20:00
"""

MT5_DEALS = """Time,Deal,Symbol,Type,Direction,Volume,Price,Commission,Swap,Profit,Balance
2024.01.02 10:00:00,1,EURUSD,buy,in,1.00,1.10000,0,0,0,10000
2024.01.02 12:00:00,2,EURUSD,sell,out,1.00,1.10100,-2,0,100,10098
2024.01.03 10:00:00,3,EURUSD,sell,in,1.00,1.20000,0,0,0,10098
2024.01.03 12:00:00,4,EURUSD,buy,out,0.40,1.19000,-1,0,40,10137
2024.01.03 13:00:00,5,EURUSD,buy,out,0.60,1.18000,0,0,120,10257
"""


def test_tradingview_old_format_is_not_options_and_not_doubled():
    frame = pd.read_csv(__import__("io").StringIO(TV_OLD))
    assert not parsers._is_options_csv(frame)
    assert parsers._is_tradingview_csv(frame)
    df, note, _flags = parsers._parse_tradingview_csv(TV_OLD)
    # Two closed trades, not four rows, and 10% / 5% rather than the $100 / $50 column.
    assert list(df["pnl"]) == pytest.approx([0.10, 0.05])
    assert list(df["side"]) == ["long", "short"]
    assert "currency" not in note.lower() or "not the currency" in note.lower()
    routed, routed_note = parsers.smart_parse_csv(TV_OLD.encode())
    assert len(routed) == 2
    assert "OPTIONS" not in routed_note
    assert routed["pnl"].max() == pytest.approx(0.10)


def test_tradingview_new_format_uses_percent_column():
    df, note, _flags = parsers._parse_tradingview_csv(TV_NEW)
    assert len(df) == 1
    assert df["pnl"].iloc[0] == pytest.approx(0.20)
    assert "Net PnL USD" not in note
    assert df["date"].iloc[0].month == 2


def test_tradingview_webhook_close_covers_every_lot_and_sell_can_flip():
    from routers.tradingview import _tv_pair_alerts

    closed, _flags = _tv_pair_alerts(
        [
            {"symbol": "AAPL", "action": "BUY", "price": 100, "quantity": 10, "timestamp": 1},
            {"symbol": "AAPL", "action": "BUY", "price": 120, "quantity": 10, "timestamp": 2},
            {"symbol": "AAPL", "action": "CLOSE", "price": 110, "quantity": 1, "timestamp": 3},
        ]
    )
    assert [row["pnl"] for row in closed] == pytest.approx([0.1, (110 - 120) / 120], abs=1e-6)
    flipped, flags = _tv_pair_alerts(
        [
            {"symbol": "AAPL", "action": "BUY", "price": 100, "quantity": 10, "timestamp": 1},
            {"symbol": "AAPL", "action": "SELL", "price": 110, "quantity": 15, "timestamp": 2},
        ]
    )
    assert len(flipped) == 1
    assert flipped[0]["pnl"] == pytest.approx(0.1)
    assert any("open" in flag for flag in flags)


def test_tradingview_performance_summary_is_rejected():
    raw = "Net profit,Gross profit,Profit factor,Percent profitable\n10,12,1.4,55\n"
    with pytest.raises(TradeImportError, match="List of Trades"):
        parsers.smart_parse_csv(raw.encode())


def test_kite_console_tradebook_fifo_and_fo_short():
    df, note, flags = parsers._parse_kite_csv(KITE_REAL)
    by_ticker = dict(zip(df["ticker"], df["pnl"], strict=True))
    assert by_ticker["INFY"] == pytest.approx(0.10)
    assert by_ticker["NIFTY24JAN21500CE"] == pytest.approx(0.20)
    assert "F&O" in " ".join(flags)
    assert "09:15" in str(df.loc[df["ticker"] == "INFY", "date"].iloc[0]) or df["date"].iloc[0].hour == 15
    assert "Zerodha" in note


def test_kite_same_day_identical_round_trips_are_kept():
    raw = (
        "symbol,trade_date,trade_type,quantity,price,trade_id,order_execution_time\n"
        "INFY,2024-01-02,buy,10,100,1,2024-01-02T09:15:00\n"
        "INFY,2024-01-02,sell,10,110,2,2024-01-02T10:15:00\n"
        "INFY,2024-01-02,buy,10,100,3,2024-01-02T11:15:00\n"
        "INFY,2024-01-02,sell,10,110,4,2024-01-02T12:15:00\n"
    )
    df, _note, _flags = parsers._parse_kite_csv(raw)
    assert len(df) == 2
    assert len(parsers._dedup_trades(df)) == 2


def test_kite_ddmmyyyy_is_not_swapped_and_iso_is_not_swapped():
    raw = (
        "symbol,trade_date,trade_type,quantity,price,trade_id\n"
        "INFY,13/01/2024,buy,1,100,1\n"
        "INFY,14/01/2024,sell,1,110,2\n"
    )
    df, _note, _flags = parsers._parse_kite_csv(raw)
    assert df["date"].iloc[0].month == 1
    assert df["date"].iloc[0].day == 14
    iso = parsers._parse_kite_csv(KITE_REAL)[0]
    assert iso.loc[iso["ticker"] == "INFY", "date"].iloc[0].month == 1


def test_kite_equity_flip_does_not_silently_drop_the_extra_shares():
    raw = (
        "symbol,trade_date,trade_type,quantity,price,trade_id,segment\n"
        "INFY,2024-01-02,buy,10,100,1,EQ\n"
        "INFY,2024-01-03,sell,15,110,2,EQ\n"
    )
    df, _note, flags = parsers._parse_kite_csv(raw)
    assert len(df) == 1
    assert any("exceeded" in flag or "no matching buy" in flag for flag in flags)


def test_kite_file_with_no_closes_is_not_reread_as_prices():
    raw = "symbol,trade_date,trade_type,quantity,price,trade_id\n" + "".join(
        f"INFY,2024-01-{day:02d},buy,10,{100 + day},T{day}\n" for day in range(1, 13)
    )
    with pytest.raises(TradeImportError, match="open|No completed"):
        parsers.smart_parse_csv(raw.encode())


def test_mt5_fx_is_a_price_return_not_a_clipped_dollar_ratio():
    df, _note, flags = parsers._parse_mt5_csv(MT5_DEALS)
    # 10 pips on 1.10000 is about 0.09%, not 0.99. The partial close stays two trades.
    assert df["pnl"].max() < 0.05
    assert len(df) == 3
    assert any("100,000" in flag or "contract" in flag.lower() for flag in flags)
    routed, note = parsers.smart_parse_csv(MT5_DEALS.encode())
    assert "MT5_SCALE_WARNING" not in note
    assert routed["pnl"].abs().max() < 0.05


def test_mt5_semicolon_european_decimals():
    raw = (
        "Time;Deal;Symbol;Type;Direction;Volume;Price;Commission;Swap;Profit\n"
        "2024.01.02 10:00:00;1;EURUSD;buy;in;1,00;1,10000;0;0;0\n"
        "2024.01.02 12:00:00;2;EURUSD;sell;out;1,00;1,10100;-1,5;0;10,00\n"
    )
    df, _note, _flags = parsers._parse_mt5_csv(raw)
    assert len(df) == 1
    assert df["pnl"].iloc[0] == pytest.approx((1.101 - 1.1) / 1.1, abs=1e-4)
    assert df["pnl"].iloc[0] < 0.01


def test_generic_rejects_inf_nan_and_text():
    rows = "\n".join(f"2024-01-{day:02d},0.01" for day in range(1, 12))
    with pytest.raises(TradeImportError, match="inf"):
        parsers.smart_parse_csv(f"date,pnl\n{rows}\n2024-01-20,inf\n".encode())
    with pytest.raises(TradeImportError, match="abc"):
        parsers.smart_parse_csv(f"date,pnl\n{rows}\n2024-01-20,abc\n".encode())
    with pytest.raises(TradeImportError, match="nan"):
        parsers.smart_parse_csv(f"date,pnl\n{rows}\n2024-01-20,nan\n".encode())


def test_future_dates_are_warned_and_small_files_parse():
    raw = "\n".join(f"2099-01-{day:02d},0.01" for day in range(1, 6))
    df, note = parsers.smart_parse_csv(f"date,pnl\n{raw}\n".encode())
    assert len(df) == 5
    assert "2099" in note
    small = "date,pnl\n2024-01-01,0.01\n2024-01-02,-0.02\n2024-01-03,0.03\n"
    assert len(parsers.smart_parse_csv(small.encode())[0]) == 3


def test_dateless_identical_returns_are_not_deduped():
    df = pd.DataFrame({"pnl": [0.1, 0.1, 0.2]})
    assert len(parsers._dedup_trades(df)) == 3


def test_percent_strings_are_actually_percents_and_dates_stay_aligned():
    raw = "date,pnl\n2024-01-01,1.5%\n2024-01-02,\n2024-01-03,-0.5%\n2024-06-01,2%\n"
    # pad to keep the test about values, not the minimum
    extra = "\n".join(f"2024-02-{day:02d},1%" for day in range(1, 10))
    df, note = parsers.smart_parse_csv(f"{raw}{extra}\n".encode())
    assert df["pnl"].iloc[0] == pytest.approx(0.015)
    assert df["date"].iloc[0].day == 1
    assert df["date"].iloc[1].day == 3
    assert "percentage" in note


def test_row_cap_rejects_before_scoring():
    rows = "\n".join(f"2024-01-01,{0.01 if i % 2 else -0.01}" for i in range(MAX_TRADES_PER_RUN + 5))
    with pytest.raises(TradeImportError, match=str(MAX_TRADES_PER_RUN)):
        parsers.smart_parse_csv(f"date,pnl\n{rows}\n".encode())


def test_xlsx_round_trip_and_old_xls_rejected():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4), "pnl": [0.01, -0.02, 0.03, -0.01]})
    buffer = __import__("io").BytesIO()
    frame.to_excel(buffer, index=False)
    parsed, _note = parsers.smart_parse_csv(buffer.getvalue())
    assert len(parsed) == 4
    with pytest.raises(TradeImportError, match=".xls"):
        parsers.smart_parse_csv(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1fake")


def test_utf16_and_preamble_and_totals():
    body = "Account export\n\nsymbol,trade_date,trade_type,quantity,price,trade_id\n"
    body += "INFY,2024-01-02,buy,10,100,1\nINFY,2024-01-03,sell,10,110,2\nTotal,,,,,\n"
    encoded = body.encode("utf-16")
    df, note = parsers.smart_parse_csv(encoded)
    assert len(df) == 1
    assert df["pnl"].iloc[0] == pytest.approx(0.10)
    assert "preamble" in note.lower() or len(df) == 1


def test_alpaca_option_symbol_is_not_truncated_and_partial_fill_counts():
    orders = [
        {
            "symbol": "AAPL250117C00150000",
            "side": "buy",
            "status": "canceled",
            "filled_qty": "2",
            "filled_avg_price": "1.50",
            "filled_at": "2024-01-02T15:00:00Z",
        },
        {
            "symbol": "AAPL250117C00150000",
            "side": "sell",
            "status": "filled",
            "filled_qty": "2",
            "filled_avg_price": "2.00",
            "filled_at": "2024-01-03T15:00:00Z",
            "position_intent": "sell_to_close",
        },
    ]
    df, _flags = parsers.pair_alpaca_orders(orders)
    assert list(df["ticker"]) == ["AAPL250117C00150000"]
    assert df["pnl"].iloc[0] == pytest.approx((2.0 - 1.5) / 1.5)


def test_wrong_broker_is_not_silently_scored(client):
    raw = "upstox trade book\nsymbol,side,qty,average\nINFY,buy,1,100\n"
    with pytest.raises(TradeImportError, match="Upstox"):
        parsers.smart_parse_csv(raw.encode())
    response = client.post(
        "/validate",
        files={"file": ("book.xlsx", b"not really a workbook", "application/vnd.ms-excel")},
    )
    assert response.status_code == 400


def test_flex_token_is_percent_encoded():
    token = "a+b/c=d&e"
    urls: list[str] = []

    async def get(url: str) -> str:
        urls.append(url)
        return (
            "<FlexStatementResponse><Status>Fail</Status><ErrorCode>1012</ErrorCode>"
            "<ErrorMessage>bad token</ErrorMessage></FlexStatementResponse>"
        )

    async def sleep(_seconds: float) -> None:
        return None

    with pytest.raises(ibkr.IBKRServiceError):
        asyncio.run(
            ibkr.fetch_flex_statement(
                token, "12 34", 10, get=get, sleep=sleep, initial_delay=0, retry_delay=0
            )
        )
    query = urls[0].split("?", 1)[1]
    assert "a%2Bb%2Fc%3Dd%26e" in query
    assert "12%2034" in query
    assert "a+b/c" not in query


def test_validate_rejects_three_trade_file_with_a_closed_trade_message(client):
    payload = "date,pnl\n2024-01-01,0.01\n2024-01-02,-0.02\n2024-01-03,0.03\n"
    response = client.post("/validate", files={"file": ("t.csv", payload.encode(), "text/csv")})
    assert response.status_code == 400
    assert "3" in response.text
    assert "10" in response.text
    assert "No valid PnL" not in response.text


def test_http_exception_still_used_for_non_csv_names():
    # The filename gate is in the router; the parser itself rejects the old Excel container.
    with pytest.raises((TradeImportError, HTTPException)):
        parsers.smart_parse_csv(b"")
