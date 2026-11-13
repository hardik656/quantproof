"""IBKR Flex import: Activity trades, TradeConfirm, file uploads, and retries."""

import asyncio
import io

import pandas as pd
import pytest
from fastapi import HTTPException

from services import ibkr, parsers


def _trade_xml(**attrs) -> str:
    body = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f"<Trade {body} />"


def _activity(rows: list[str], currency: str = "USD", account: str = "U1234567") -> str:
    joined = "\n".join(rows)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="Activity Flex" type="AF">
  <FlexStatements count="1">
    <FlexStatement accountId="{account}" fromDate="2024-01-01" toDate="2024-12-31" currency="{currency}">
      <Trades>
        {joined}
      </Trades>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
"""


def _confirm(rows: list[str]) -> str:
    joined = "\n".join(rows)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="Trade Confirmation" type="TCF">
  <FlexStatements count="1">
    <FlexStatement accountId="U1234567" fromDate="2024-01-01" toDate="2024-12-31" currency="USD">
      <TradeConfirms>
        {joined}
      </TradeConfirms>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
"""


def _confirm_row(**attrs) -> str:
    body = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f"<TradeConfirm {body} />"


def _closed(xml: str) -> list[dict]:
    trades, _notes, _currency = ibkr.parse_flex_statement(ibkr.ET.fromstring(xml))
    return trades


def test_open_close_and_position_flips_are_classified():
    assert ibkr.is_opening_position("O", 10, -1000, 100)
    assert not ibkr.is_opening_position("C", -10, 1100, 110, fifo_pnl=0)
    assert not ibkr.is_opening_position("C;O", -150, 16000, 110, fifo_pnl=1000)
    assert not ibkr.is_opening_position("O;C", 140, -12000, 90, fifo_pnl=1000)
    assert ibkr.calculate_trade_side("SELL", -150, "C;O", 0, 110) == "long"
    assert ibkr.calculate_trade_side("BUY", 140, "O;C", 0, 90) == "short"


def test_break_even_close_is_kept():
    xml = _activity(
        [
            _trade_xml(
                symbol="AAPL",
                currency="USD",
                tradeDate="20240102",
                tradeTime="100000",
                quantity="10",
                tradePrice="100",
                buySell="BUY",
                openCloseIndicator="O",
                fifoPnlRealized="0",
                multiplier="1",
                fxRateToBase="1",
            ),
            _trade_xml(
                symbol="AAPL",
                currency="USD",
                tradeDate="20240103",
                tradeTime="153045",
                quantity="-10",
                tradePrice="100",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="0",
                multiplier="1",
                fxRateToBase="1",
            ),
        ]
    )
    trades = _closed(xml)
    assert len(trades) == 1
    assert trades[0]["pnl"] == 0.0
    assert trades[0]["side"] == "long"
    assert trades[0]["date"].year == 2024
    assert trades[0]["date"].hour == 15
    assert trades[0]["date"].minute == 30
    assert trades[0]["date"].second == 45


def test_option_multiplier_uses_entry_cost_not_a_clipped_dollar_pnl():
    xml = _activity(
        [
            _trade_xml(
                symbol="AAPL",
                assetCategory="OPT",
                multiplier="100",
                tradeDate="20240301",
                tradeTime="100000",
                quantity="1",
                tradePrice="2.00",
                buySell="BUY",
                openCloseIndicator="O",
                fifoPnlRealized="0",
                currency="USD",
                fxRateToBase="1",
                putCall="C",
                strike="190",
                expiry="20240315",
            ),
            _trade_xml(
                symbol="AAPL",
                assetCategory="OPT",
                multiplier="100",
                tradeDate="20240304",
                tradeTime="110000",
                quantity="-1",
                tradePrice="3.00",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="100",
                currency="USD",
                fxRateToBase="1",
                putCall="C",
                strike="190",
                expiry="20240315",
            ),
        ]
    )
    trades = _closed(xml)
    # Entry cost is 2 x 1 x 100 = 200, not 2. 100/200 = 0.50.
    # Ignoring the multiplier yields 100/2, which the old code clipped to 0.99.
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(0.5)
    assert trades[0]["pnl"] != pytest.approx(0.99)


def test_option_without_an_open_still_applies_multiplier():
    xml = _activity(
        [
            _trade_xml(
                symbol="AAPL",
                assetCategory="OPT",
                multiplier="100",
                tradeDate="20240304",
                tradeTime="110000",
                quantity="-1",
                tradePrice="2",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="50",
                currency="USD",
                fxRateToBase="1",
            )
        ]
    )
    trades = _closed(xml)
    assert trades[0]["pnl"] == pytest.approx(50 / 200)


def test_net_cash_is_not_treated_as_pnl():
    # netCash is about ± notional. Dividing it by price × qty clips to ±0.99.
    no_basis = {
        "symbol": "AAPL",
        "tradeDate": "20240103",
        "tradeTime": "120000",
        "quantity": "-10",
        "tradePrice": "110",
        "buySell": "SELL",
        "openCloseIndicator": "C",
        "netCash": "1100",
        "multiplier": "1",
        "currency": "USD",
        "fxRateToBase": "1",
    }
    missing, reason = ibkr.parse_ibkr_trade_improved(no_basis)
    assert missing is None
    assert reason == "pnl_calculation_failed"

    priced, reason = ibkr.parse_ibkr_trade_improved({**no_basis, "cost": "1000"})
    assert reason is None
    assert priced["pnl"] == pytest.approx(0.1)

    short, _reason = ibkr.parse_ibkr_trade_improved(
        {
            "symbol": "AAPL",
            "tradeDate": "20240104",
            "quantity": "10",
            "tradePrice": "90",
            "buySell": "BUY",
            "openCloseIndicator": "C",
            "netCash": "-900",
            "cost": "-1000",
            "multiplier": "1",
            "currency": "USD",
            "fxRateToBase": "1",
        }
    )
    assert short["pnl"] == pytest.approx(0.1)
    assert short["side"] == "short"


def test_position_flip_close_then_open_uses_the_closed_quantity():
    xml = _activity(
        [
            _trade_xml(
                symbol="AAPL",
                tradeDate="20240201",
                tradeTime="100000",
                quantity="100",
                tradePrice="100",
                buySell="BUY",
                openCloseIndicator="O",
                fifoPnlRealized="0",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            ),
            _trade_xml(
                symbol="AAPL",
                tradeDate="20240202",
                tradeTime="140000",
                quantity="-150",
                tradePrice="110",
                buySell="SELL",
                openCloseIndicator="C;O",
                fifoPnlRealized="1000",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            ),
        ]
    )
    trades = _closed(xml)
    # 100 shares closed at a $10 gain. Using the full 150-share fill would be 1000/16500.
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(0.1)
    assert trades[0]["side"] == "long"


def test_position_flip_open_then_close_covers_the_short():
    xml = _activity(
        [
            _trade_xml(
                symbol="MSFT",
                tradeDate="20240201",
                tradeTime="100000",
                quantity="-100",
                tradePrice="100",
                buySell="SELL",
                openCloseIndicator="O",
                fifoPnlRealized="0",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            ),
            _trade_xml(
                symbol="MSFT",
                tradeDate="20240202",
                tradeTime="140000",
                quantity="140",
                tradePrice="90",
                buySell="BUY",
                openCloseIndicator="O;C",
                fifoPnlRealized="1000",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            ),
        ]
    )
    trades = _closed(xml)
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(0.1)
    assert trades[0]["side"] == "short"


def test_non_base_currency_uses_each_legs_fx_rate():
    xml = _activity(
        [
            _trade_xml(
                symbol="VOD",
                currency="GBP",
                fxRateToBase="1.20",
                tradeDate="20240501",
                tradeTime="090000",
                quantity="10",
                tradePrice="100",
                buySell="BUY",
                openCloseIndicator="O",
                fifoPnlRealized="0",
                multiplier="1",
            ),
            _trade_xml(
                symbol="VOD",
                currency="GBP",
                fxRateToBase="1.30",
                tradeDate="20240508",
                tradeTime="150000",
                quantity="-10",
                tradePrice="110",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="100",
                multiplier="1",
            ),
        ],
        currency="USD",
    )
    trades, notes, base = ibkr.parse_flex_statement(ibkr.ET.fromstring(xml))
    # Base cost 100 * 10 * 1.20 = 1200. Base P&L 100 * 1.30 = 130.
    assert base == "USD"
    assert trades[0]["pnl"] == pytest.approx(round(130 / 1200, 6))
    assert trades[0]["pnl"] != pytest.approx(0.10)
    assert trades[0]["pnl"] != pytest.approx(0.13)
    assert any("fxRateToBase" in note for note in notes)


def test_level_of_detail_keeps_executions_only():
    xml = _activity(
        [
            _trade_xml(
                symbol="AAPL",
                levelOfDetail="EXECUTION",
                tradeDate="20240603",
                tradeTime="100000",
                quantity="-10",
                tradePrice="110",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="100",
                cost="1000",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            ),
            _trade_xml(
                symbol="AAPL",
                levelOfDetail="SYMBOL_SUMMARY",
                tradeDate="20240603",
                tradeTime="100000",
                quantity="-10",
                tradePrice="110",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="999",
                cost="1000",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            ),
        ]
    )
    trades, notes, _currency = ibkr.parse_flex_statement(ibkr.ET.fromstring(xml))
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(0.1)
    assert any("EXECUTION" in note for note in notes)


def test_trade_confirmation_xml_is_fifo_matched():
    xml = _confirm(
        [
            _confirm_row(
                accountId="U1234567",
                symbol="AAPL",
                currency="USD",
                assetCategory="STK",
                multiplier="1",
                tradeDate="20240701",
                dateTime="20240701;100000",
                buySell="BUY",
                quantity="10",
                price="100",
                proceeds="-1000",
                commission="-1",
                code="O",
                fxRateToBase="1",
            ),
            _confirm_row(
                accountId="U1234567",
                symbol="AAPL",
                currency="USD",
                assetCategory="STK",
                multiplier="1",
                tradeDate="20240702",
                dateTime="20240702;153000",
                buySell="SELL",
                quantity="10",
                price="110",
                proceeds="1100",
                commission="-1",
                code="C",
                fxRateToBase="1",
            ),
        ]
    )
    trades = _closed(xml)
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(0.1)
    assert trades[0]["ticker"] == "AAPL"
    assert trades[0]["date"].year == 2024
    assert trades[0]["date"].hour == 15


def test_trade_confirmation_flip_and_scaled_fifo():
    xml = _confirm(
        [
            _confirm_row(
                symbol="IBM",
                currency="USD",
                multiplier="1",
                tradeDate="20240801",
                dateTime="20240801;100000",
                buySell="BUY",
                quantity="10",
                price="100",
                code="O",
                fxRateToBase="1",
            ),
            _confirm_row(
                symbol="IBM",
                currency="USD",
                multiplier="1",
                tradeDate="20240801",
                dateTime="20240801;110000",
                buySell="BUY",
                quantity="10",
                price="200",
                code="O",
                fxRateToBase="1",
            ),
            _confirm_row(
                symbol="IBM",
                currency="USD",
                multiplier="1",
                tradeDate="20240802",
                dateTime="20240802;120000",
                buySell="SELL",
                quantity="15",
                price="150",
                code="C;O",
                fxRateToBase="1",
            ),
        ]
    )
    trades = _closed(xml)
    # Closes 10 @ 100 and 5 @ 200 (basis 2000) against 15 @ 150 (proceeds 2250).
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(250 / 2000)


def test_yyyymmdd_is_not_parsed_as_1970():
    parsed = ibkr.parse_ibkr_datetime("20240115", "153045")
    assert parsed is not None
    assert parsed.year == 2024
    assert parsed.hour == 15
    assert parsed != pd.Timestamp(20240115).to_pydatetime()


def test_flex_csv_upload_keeps_dollar_pnl_and_real_dates():
    header = (
        "ClientAccountID,Symbol,TradeDate,TradeTime,Quantity,TradePrice,Buy/Sell,"
        "Open/CloseIndicator,FifoPnlRealized,CostBasis,CurrencyPrimary,FXRateToBase,"
        "Multiplier,AssetClass,LevelOfDetail,NetCash"
    )
    lines = [header]
    for offset in range(12):
        day = 2 + offset
        lines.append(f"U1234567,AAPL,202401{day:02d},100000,10,100,BUY,O,0,1000,USD,1,1,STK,EXECUTION,-1000")
        lines.append(
            f"U1234567,AAPL,202401{day:02d},150000,-10,125,SELL,C,250,1000,USD,1,1,STK,EXECUTION,1250"
        )
    frame, note = parsers.smart_parse_csv("\n".join(lines).encode())
    assert len(frame) == 12
    assert set(frame["pnl"].round(6)) == {0.25}
    assert int(frame["date"].dt.year.min()) == 2024
    assert int(frame["date"].dt.year.max()) == 2024
    assert "1970" not in note
    assert "Flex CSV" in note


def test_sectioned_flex_csv_is_parsed():
    raw = """Trade,Header,Symbol,TradeDate,TradeTime,Quantity,TradePrice,Buy/Sell,Open/CloseIndicator,FifoPnlRealized,CostBasis,Multiplier,CurrencyPrimary,FXRateToBase
Trade,Data,AAPL,20240301,100000,10,50,BUY,O,0,500,1,USD,1
Trade,Data,AAPL,20240302,153000,-10,60,SELL,C,50,500,1,USD,1
"""
    frame, note = ibkr.parse_ibkr_upload(raw.encode())
    assert len(frame) == 1
    assert frame["pnl"].iloc[0] == pytest.approx(0.1)
    assert frame["date"].iloc[0].year == 2024
    assert "Flex CSV" in note


ACTIVITY_STATEMENT = """Statement,Header,Field Name,Field Value
Statement,Data,BrokerName,Interactive Brokers
Statement,Data,Title,Activity Statement
Statement,Data,Period,"January 1, 2024 - December 31, 2024"
Account Information,Header,Field Name,Field Value
Account Information,Data,Account,U1234567
Account Information,Data,Base Currency,USD
Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,Proceeds,Comm/Fee,Basis,Realized P/L,Code
Trades,Data,Order,Stocks,USD,AAPL,"2024-01-02, 10:00:00",10,100,-1000,-1,1001,0,O
Trades,Data,Order,Stocks,USD,AAPL,"2024-01-03, 11:00:00",-10,150,1500,-1,1001,498,C
Trades,Data,Order,Stocks,USD,MSFT,"2024-01-04, 10:00:00",5,200,-1000,-1,1001,0,O
Trades,Data,Order,Stocks,USD,MSFT,"2024-01-05, 10:00:00",-5,220,1100,-1,1001,98,C
Trades,SubTotal,,Stocks,USD,,,,,,,,596
Trades,Total,,,,,,,,,,596
"""

TWS_LOG = """Symbol,Date/Time,Quantity,T. Price,Proceeds,Comm/Fee,Basis,Realized P/L,Code
AAPL,20240102,10,100,-1000,-1,1001,0,O
AAPL,20240103,-10,110,1100,-1,1001,99,C
MSFT,20240104,10,50,-500,-1,501,0,O
MSFT,20240105,-10,40,400,-1,501,-102,C
IBM,20240108,10,80,-800,-1,801,0,O
IBM,20240109,-10,90,900,-1,801,98,C
"""


def test_activity_statement_and_tws_logs_are_rejected():
    for payload, marker in (
        (ACTIVITY_STATEMENT, "Activity Statement"),
        (TWS_LOG, "TWS"),
    ):
        with pytest.raises(HTTPException) as exc:
            parsers.smart_parse_csv(payload.encode())
        assert exc.value.status_code == 400
        assert marker in str(exc.value.detail)
        assert "Flex" in str(exc.value.detail)


def test_generic_and_kite_csv_are_not_treated_as_ibkr():
    generic = "date,pnl\n" + "\n".join(f"2024-01-{day:02d},0.01" for day in range(1, 12))
    assert ibkr.classify_ibkr_bytes(generic.encode()) == "not_ibkr"
    kite = "symbol,trade_date,trade_type,quantity,price,trade_id,segment\nINFY,2024-01-02,buy,1,1,1,EQ\n"
    assert ibkr.classify_ibkr_bytes(kite.encode()) == "not_ibkr"


def _ready_statement() -> str:
    return _activity(
        [
            _trade_xml(
                symbol="AAPL",
                tradeDate="20240115",
                tradeTime="153045",
                quantity="-10",
                tradePrice="110",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="100",
                cost="1000",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
            )
        ]
    )


def test_statement_generation_retries_then_parses():
    token = "test-flex-token"
    send = (
        "<FlexStatementResponse>"
        "<Status>Success</Status><ReferenceCode>999</ReferenceCode>"
        "<Url>https://example.test/GetStatement</Url>"
        "</FlexStatementResponse>"
    )
    pending = (
        "<FlexStatementResponse><Status>Fail</Status><ErrorCode>1019</ErrorCode>"
        "<ErrorMessage>Statement generation in progress. Please try again shortly.</ErrorMessage>"
        "</FlexStatementResponse>"
    )
    responses = [send, pending, _ready_statement()]
    urls: list[str] = []

    async def get(url: str) -> str:
        urls.append(url)
        return responses.pop(0)

    async def sleep(_seconds: float) -> None:
        return None

    root = asyncio.run(
        ibkr.fetch_flex_statement(
            token,
            "42",
            30,
            get=get,
            sleep=sleep,
            initial_delay=0,
            retry_delay=0,
        )
    )
    assert "p=30" in urls[0]
    assert "q=42" in urls[0]
    assert len(urls) == 3
    trades, _notes, _currency = ibkr.parse_flex_statement(root)
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(0.1)
    assert trades[0]["date"].year == 2024


def test_lookback_is_capped_and_token_is_redacted():
    token = "test-flex-token"
    send = (
        "<FlexStatementResponse><Status>Success</Status><ReferenceCode>1</ReferenceCode>"
        "<Url>https://example.test/GetStatement</Url></FlexStatementResponse>"
    )
    urls: list[str] = []

    async def get(url: str) -> str:
        urls.append(url)
        if len(urls) == 1:
            return send
        return _ready_statement()

    async def sleep(_seconds: float) -> None:
        return None

    asyncio.run(
        ibkr.fetch_flex_statement(token, "7", 900, get=get, sleep=sleep, initial_delay=0, retry_delay=0)
    )
    assert "p=365" in urls[0]
    assert "p=900" not in urls[0]

    async def exploding(url: str) -> str:
        raise RuntimeError(f"connection failed for {url}")

    with pytest.raises(ibkr.IBKRServiceError) as exc:
        asyncio.run(
            ibkr.fetch_flex_statement(
                token, "7", 10, get=exploding, sleep=sleep, initial_delay=0, retry_delay=0
            )
        )
    assert exc.value.status_code == 503
    assert token not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


def test_hard_flex_error_is_not_retried():
    token = "test-flex-token"
    send = (
        "<FlexStatementResponse><Status>Success</Status><ReferenceCode>1</ReferenceCode>"
        "<Url>https://example.test/GetStatement</Url></FlexStatementResponse>"
    )
    invalid = (
        "<FlexStatementResponse><Status>Fail</Status><ErrorCode>1015</ErrorCode>"
        "<ErrorMessage>Token is invalid.</ErrorMessage></FlexStatementResponse>"
    )
    urls: list[str] = []

    async def get(url: str) -> str:
        urls.append(url)
        return send if len(urls) == 1 else invalid

    async def sleep(_seconds: float) -> None:
        return None

    with pytest.raises(ibkr.IBKRServiceError) as exc:
        asyncio.run(
            ibkr.fetch_flex_statement(token, "7", 5, get=get, sleep=sleep, initial_delay=0, retry_delay=0)
        )
    assert exc.value.status_code == 400
    assert len(urls) == 2
    assert "invalid" in str(exc.value).lower()
    assert token not in str(exc.value)


def test_validate_accepts_flex_xml_and_rejects_activity_statement(client):
    rows = []
    for offset in range(10):
        day = 2 + offset
        rows.append(
            _trade_xml(
                symbol="AAPL",
                tradeDate=f"202401{day:02d}",
                tradeTime="100000",
                quantity="10",
                tradePrice="100",
                buySell="BUY",
                openCloseIndicator="O",
                fifoPnlRealized="0",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
                levelOfDetail="EXECUTION",
            )
        )
        rows.append(
            _trade_xml(
                symbol="AAPL",
                tradeDate=f"202401{day:02d}",
                tradeTime="150000",
                quantity="-10",
                tradePrice="110",
                buySell="SELL",
                openCloseIndicator="C",
                fifoPnlRealized="100",
                cost="1000",
                multiplier="1",
                fxRateToBase="1",
                currency="USD",
                levelOfDetail="EXECUTION",
            )
        )
    xml = _activity(rows).encode()
    ok = client.post("/validate", files={"file": ("flex.xml", io.BytesIO(xml), "application/xml")})
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["total_trades"] == 10
    assert "2024" in body["date_range"]
    assert "1970" not in body["date_range"]

    rejected = client.post(
        "/validate",
        files={"file": ("activity.csv", io.BytesIO(ACTIVITY_STATEMENT.encode()), "text/csv")},
    )
    assert rejected.status_code == 400
    assert "Activity Statement" in str(rejected.json()["detail"])
