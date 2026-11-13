"""Broker export parsers (Kite, Alpaca, options, MT5, generic CSV/Excel) with FIFO lot matching.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import asyncio
import io
import re as _re
from functools import wraps

import numpy as np
import pandas as pd
from fastapi import HTTPException

from services.benchmarks import assign_regimes, get_benchmark_type
from services.csv_util import (
    Fill,
    apply_aliases,
    decode_upload,
    describe_open,
    excel_to_csv_text,
    match_fills,
    normalize_header,
    parse_datetimes,
    parse_number,
    parse_numbers,
    pnl_column_rank,
    read_delimited,
    trades_frame,
)
from services.ibkr import classify_ibkr_bytes, parse_ibkr_upload
from services.limits import MAX_TRADES_PER_RUN, TradeImportError

_KITE_FO_SEGMENTS = {"FO", "F&O", "FUTURES", "OPTIONS", "CDS", "MF", "NFO", "BCD", "MCX", "NCO"}
_KITE_FO_EXCHANGES = {"NFO", "BFO", "MCX", "CDS", "BCD", "NCO", "NCDEX"}
_FEE_COLUMNS = (
    "brokerage",
    "charges",
    "total_charges",
    "stt",
    "gst",
    "stamp_duty",
    "sebi_charges",
    "transaction_charges",
    "exchange_charges",
    "fee",
)


def _columns(df: pd.DataFrame) -> set[str]:
    return {normalize_header(column) for column in df.columns}


def _prepared(raw: str):
    table = read_delimited(raw)
    _unused, frame = _prepared_frame(table.frame)
    return table, frame


def _kite_timestamps(df: pd.DataFrame, *, dayfirst: bool) -> tuple[pd.Series, str]:
    """Prefer the fill clock. Fall back to the trade date when the clock is blank."""
    notes = []
    times = pd.Series(pd.NaT, index=df.index)
    dates = pd.Series(pd.NaT, index=df.index)
    if "order_execution_time" in df.columns:
        times, note = parse_datetimes(df["order_execution_time"], dayfirst=dayfirst)
        if note:
            notes.append(note)
    if "trade_date" in df.columns:
        dates, note = parse_datetimes(df["trade_date"], dayfirst=dayfirst)
        if note:
            notes.append(note)
    if "order_execution_time" in df.columns:
        when = times.fillna(dates) if "trade_date" in df.columns else times
    else:
        when = dates
    return when, " ".join(notes)


def _row_fee(row: pd.Series, decimal_comma: bool) -> float:
    total = 0.0
    found = False
    for column in _FEE_COLUMNS:
        if column not in row.index:
            continue
        number, _percent = parse_numbers(pd.Series([row[column]]), decimal_comma=decimal_comma)
        if pd.notna(number.iloc[0]):
            total += abs(float(number.iloc[0]))
            found = True
    return total if found else 0.0


def _is_fo_row(row: pd.Series, symbol: str) -> bool:
    segment = (
        str(row["segment"]).upper().strip() if "segment" in row.index and pd.notna(row.get("segment")) else ""
    )
    exchange = (
        str(row["exchange"]).upper().strip()
        if "exchange" in row.index and pd.notna(row.get("exchange"))
        else ""
    )
    series = (
        str(row["series"]).upper().strip() if "series" in row.index and pd.notna(row.get("series")) else ""
    )
    if segment in _KITE_FO_SEGMENTS or exchange in _KITE_FO_EXCHANGES or series in {"FO", "CD"}:
        return True
    return symbol.upper().endswith(("CE", "PE", "FUT"))


def _stamp(value) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value)


# =========================================================
# KITE TRADEBOOK PARSER
# =========================================================


def _is_kite_csv(df: pd.DataFrame) -> bool:
    _table, frame = _prepared_frame(df)
    cols = set(frame.columns)
    required = {"trade_date", "trade_type", "quantity", "price"}
    kite_signals = {
        "trade_time",
        "settlement_id",
        "series",
        "order_id",
        "trade_id",
        "isin",
        "order_execution_time",
        "auction",
    }
    return required.issubset(cols) and bool(kite_signals & cols)


def _prepared_frame(df: pd.DataFrame) -> tuple[None, pd.DataFrame]:
    frame = df.copy()
    frame.columns = [normalize_header(column) for column in frame.columns]
    frame = apply_aliases(
        frame,
        [
            ("symbol", ("tradingsymbol", "trading_symbol", "instrument", "scrip")),
            (
                "order_execution_time",
                ("order_timestamp", "exchange_timestamp", "exchange_update_timestamp", "order_time"),
            ),
            ("filled_quantity", ("filled_qty",)),
            ("average_price", ("avg_price", "averageprice")),
            ("trade_date", ("date",)),
        ],
    )
    return None, frame


def _parse_kite_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Console Tradebook CSV into QuantProof trades.

    Current Console export (Reports → Tradebook → CSV) is one row per fill:
    symbol, isin, trade_date, exchange, segment, series, trade_type, auction,
    quantity, price, trade_id, order_id, order_execution_time.
    Dates are ISO. Slash dates, when a file has them, are read day-first.
    """
    table, df = _prepared(raw)

    required = {"trade_date", "trade_type", "quantity", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kite CSV missing required columns: {missing}. "
            "Export from Zerodha Console → Reports → Tradebook → CSV. "
            "Download the Tradebook, not the P&L, Positions, or Holdings report."
        )

    # 'series' is a segment code (EQ/FO), not an instrument. Aliases above
    # never map it onto the symbol, or every equity fill would share one book.
    decimal_comma = table.delimiter == ";"
    quantities, _qty_pct = parse_numbers(df["quantity"], decimal_comma=decimal_comma)
    prices, _price_pct = parse_numbers(df["price"], decimal_comma=decimal_comma)
    when, date_note = _kite_timestamps(df, dayfirst=True)
    audit_flags = list(table.notes)
    if date_note:
        audit_flags.append(date_note)

    if "trade_id" in df.columns:
        trade_ids = df["trade_id"].astype(str).str.strip()
        real_id = ~trade_ids.str.lower().isin({"", "nan", "none"})
        duplicate = real_id & trade_ids.duplicated(keep="first")
        if int(duplicate.sum()):
            audit_flags.append(
                f"Skipped {int(duplicate.sum())} duplicate trade_id row(s). "
                "The same fill was in the file more than once and was counted once."
            )
            keep = ~duplicate
            df = df.loc[keep].copy()
            quantities = quantities.loc[keep]
            prices = prices.loc[keep]
            when = when.loc[keep]

    fills: list[Fill] = []
    fo_rows = 0
    seen_rows = 0
    for seq, (idx, row) in enumerate(df.iterrows()):
        symbol = (
            str(row["symbol"]).strip()
            if "symbol" in df.columns and pd.notna(row.get("symbol"))
            else "UNKNOWN"
        )
        if symbol.lower() in {"", "nan", "none"}:
            symbol = "UNKNOWN"
        trade_type = str(row["trade_type"]).upper().strip()
        qty = float(quantities.loc[idx]) if pd.notna(quantities.loc[idx]) else float("nan")
        price = float(prices.loc[idx]) if pd.notna(prices.loc[idx]) else float("nan")
        date = _stamp(when.loc[idx]) if idx in when.index else pd.NaT
        seen_rows += 1
        if trade_type not in {"BUY", "SELL"}:
            audit_flags.append(
                f"⚠️ Skipped unknown trade_type '{trade_type}' for {symbol} "
                f"on {date.date() if pd.notna(date) else date}"
            )
            continue
        if not np.isfinite(qty) or not np.isfinite(price) or qty <= 0 or price <= 0 or pd.isna(date):
            audit_flags.append(
                f"⚠️ Skipped row: {symbol} {trade_type} qty={qty} price={price} "
                f"on {date.date() if pd.notna(date) else date} invalid qty/price"
            )
            continue
        allow_short = _is_fo_row(row, symbol)
        fo_rows += int(allow_short)
        fills.append(
            Fill(
                symbol=symbol,
                side=trade_type.lower(),
                qty=qty,
                price=price,
                date=date,
                allow_short=allow_short,
                fee=_row_fee(row, decimal_comma),
                seq=seq,
            )
        )

    result = match_fills(fills)
    orphan_sells = [item for item in result.orphans if item["kind"] in {"unmatched_sell", "excess_sell"}]
    if orphan_sells:
        audit_flags.append(
            f"⚠️ {len(orphan_sells)} sell(s) had no matching buy "
            f"(position predates TradeBook history) excluded"
        )
        for item in orphan_sells[:3]:
            when_text = (
                item["date"].date()
                if hasattr(item["date"], "date") and pd.notna(item["date"])
                else item["date"]
            )
            kind = (
                "exceeded the open long"
                if item["kind"] == "excess_sell"
                else "no open position - predates history window"
            )
            audit_flags.append(
                f"   └ {item['symbol']} SELL {item['qty']:.0f}@{item['price']:.2f} on {when_text} ({kind})"
            )
    if result.open_qty:
        audit_flags.append(describe_open(result.open_qty))
    if fo_rows:
        audit_flags.append(
            f"F&O: {fo_rows} fill(s) matched per contract symbol (CE/PE/FUT are not merged). "
            "Returns are premium price returns, gross of brokerage, STT, GST, and stamp duty. "
            "The Console tradebook does not include those charges."
        )
    elif seen_rows:
        audit_flags.append(
            "Returns are price returns, gross of brokerage and statutory charges. "
            "Zerodha's tradebook does not include them."
        )

    trades = result.trades
    if not trades:
        return pd.DataFrame(), "No completed trades found in Kite TradeBook", audit_flags

    note = (
        f"📊 Parsed {len(trades)} completed trade(s) from Zerodha Kite TradeBook "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(result.open_qty)} position(s) still open)"
    )
    if any(fill.fee for fill in fills):
        note += " Charges listed on each fill were deducted from that fill's return."
    return trades_frame(trades), note, audit_flags


# =========================================================
# KITE ORDERS CSV PARSER
# =========================================================


def _is_kite_orders_csv(df: pd.DataFrame) -> bool:
    _unused, frame = _prepared_frame(df)
    cols = set(frame.columns)
    return {"transaction_type", "order_execution_time", "symbol", "filled_quantity"}.issubset(cols)


def _parse_kite_orders_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert a Kite Connect order export into QuantProof trades.

    Console does not download an orders CSV; this is the order-book shape
    (transaction_type, tradingsymbol, filled_quantity, average_price, and
    order_execution_time or order_timestamp). Status other than COMPLETE is ignored.
    """
    table, df = _prepared(raw)

    required = {
        "transaction_type",
        "order_execution_time",
        "symbol",
        "filled_quantity",
        "average_price",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kite Orders CSV missing required columns: {missing}. "
            "Export filled orders with tradingsymbol, transaction_type, filled_quantity, "
            "average_price, and order_timestamp (or order_execution_time)."
        )

    decimal_comma = table.delimiter == ";"
    if "status" not in df.columns:
        df["status"] = "COMPLETE"
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df = df[df["status"].isin(["COMPLETE", "FILLED"])].copy()
    if df.empty:
        return pd.DataFrame(), "No COMPLETE orders found in Kite Orders CSV", []

    quantities, _q = parse_numbers(df["filled_quantity"], decimal_comma=decimal_comma)
    prices, _p = parse_numbers(df["average_price"], decimal_comma=decimal_comma)
    when, _date_note = _kite_timestamps(df, dayfirst=True)
    df = df.assign(_qty=quantities, _price=prices, _when=when)
    df["transaction_type"] = df["transaction_type"].astype(str).str.upper().str.strip()
    df = df[df["transaction_type"].isin(["BUY", "SELL"])]
    df = df[df["_qty"].gt(0) & df["_price"].gt(0) & df["_when"].notna()]
    if df.empty:
        return pd.DataFrame(), "No COMPLETE orders found in Kite Orders CSV", []

    audit_flags = list(table.notes)
    fills = []
    for seq, (_idx, row) in enumerate(df.iterrows()):
        symbol = str(row["symbol"]).strip()
        allow_short = _is_fo_row(row, symbol)
        fills.append(
            Fill(
                symbol=symbol,
                side=str(row["transaction_type"]).lower(),
                qty=float(row["_qty"]),
                price=float(row["_price"]),
                date=_stamp(row["_when"]),
                allow_short=allow_short,
                fee=_row_fee(row, decimal_comma),
                seq=seq,
            )
        )
    result = match_fills(fills)
    orphan_sells = [item for item in result.orphans if item["kind"] in {"unmatched_sell", "excess_sell"}]
    if orphan_sells:
        audit_flags.append(
            f"⚠️ {len(orphan_sells)} sell(s) had no matching BUY (position predates Orders history) excluded"
        )
        for item in orphan_sells[:3]:
            when_text = (
                item["date"].date()
                if hasattr(item["date"], "date") and pd.notna(item["date"])
                else item["date"]
            )
            audit_flags.append(
                f"   └ {item['symbol']} SELL {item['qty']:.0f}@{item['price']:.2f} on {when_text} "
                "(no matching BUY - predates history window)"
            )
    if result.open_qty:
        audit_flags.append(describe_open(result.open_qty))
    if not result.trades:
        return pd.DataFrame(), "No completed round-trip trades found in Kite Orders CSV", audit_flags
    note = (
        f"Parsed {len(result.trades)} completed trade(s) from Zerodha Kite Orders CSV "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(result.open_qty)} position(s) still open)"
    )
    return trades_frame(result.trades), note, audit_flags


# =========================================================
# RETRY DECORATOR FOR API CALLS
# =========================================================


def retry_on_failure(max_retries=3, delay=1.0):
    """
    Retry decorator for API calls with exponential backoff
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = delay * (2**attempt)
                    print(
                        f"[RETRY] API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)

        return wrapper

    return decorator


# =========================================================
# ALPACA UI EXPORT PARSER
# =========================================================

_ALPACA_FILLED = {"filled", "partially_filled"}
_ALPACA_PARTIAL_OK = {"canceled", "cancelled", "expired"}


def _is_alpaca_ui_export(df: pd.DataFrame) -> bool:
    cols = _columns(df)
    return "avg_fill_price" in cols and "filled_at" in cols and "side" in cols


def _is_alpaca_orders_csv(df: pd.DataFrame) -> bool:
    """Order rows from GET /v2/orders saved as CSV (symbol, filled_qty, filled_avg_price)."""
    cols = _columns(df)
    return {"symbol", "side", "filled_qty", "filled_avg_price"}.issubset(cols) and (
        "filled_at" in cols or "submitted_at" in cols
    )


def _is_alpaca_activities_csv(df: pd.DataFrame) -> bool:
    """Account activities of type FILL: one row per execution."""
    cols = _columns(df)
    return "activity_type" in cols and {"symbol", "side", "qty", "price", "transaction_time"}.issubset(cols)


def _parse_alpaca_ui_date(s) -> pd.Timestamp | None:
    if s is None or str(s).strip() in ("-", "", "nan", "None"):
        return None
    parsed, _note = parse_datetimes(pd.Series([s]), dayfirst=False)
    value = parsed.iloc[0]
    if pd.isna(value):
        return None
    return pd.Timestamp(value)


def _alpaca_intent(value: object) -> str | None:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in {"buy_to_open", "sell_to_open", "open"}:
        return "open"
    if text in {"buy_to_close", "sell_to_close", "close"}:
        return "close"
    return None


def _clean_alpaca_symbol(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "UNKNOWN"
    token = str(raw).strip().split()
    if not token:
        return "UNKNOWN"
    return token[0].upper()


def _order_to_fill(order: dict, seq: int) -> Fill | None:
    status = str(order.get("status", "filled")).lower().strip()
    qty_raw = order.get("filled_qty")
    if qty_raw in (None, "", "-"):
        qty_raw = order.get("qty") or order.get("quantity") or 0
    qty, _pct = parse_numbers(pd.Series([qty_raw]), decimal_comma=False)
    qty_value = float(qty.iloc[0]) if pd.notna(qty.iloc[0]) else float("nan")
    price_raw = order.get("filled_avg_price")
    if price_raw in (None, "", "-"):
        price_raw = order.get("avg_fill_price") or order.get("price") or 0
    price, _pct = parse_numbers(pd.Series([price_raw]), decimal_comma=False)
    price_value = float(price.iloc[0]) if pd.notna(price.iloc[0]) else float("nan")
    if not np.isfinite(qty_value) or not np.isfinite(price_value) or qty_value <= 0 or price_value <= 0:
        return None
    if status in _ALPACA_FILLED or (status in _ALPACA_PARTIAL_OK and qty_value > 0):
        pass
    else:
        return None
    side = str(order.get("side", "")).lower().strip()
    if side not in {"buy", "sell"}:
        return None
    ts_raw = order.get("filled_at") or order.get("transaction_time") or order.get("submitted_at") or ""
    if isinstance(ts_raw, pd.Timestamp):
        ts = ts_raw.tz_convert("UTC").tz_localize(None) if ts_raw.tzinfo else ts_raw
    else:
        ts = _parse_alpaca_ui_date(ts_raw)
        if ts is None:
            return None
    symbol = _clean_alpaca_symbol(order.get("symbol") or order.get("asset") or order.get("symbol_raw"))
    intent = _alpaca_intent(order.get("position_intent"))
    # A stated close must not invent the opposite position. An unstated side still
    # follows Alpaca's stock account: a sell with no long opens a short.
    allow_short = intent != "close"
    if intent == "open" and side == "sell":
        allow_short = True
    fee_raw = order.get("commission") or order.get("fee") or 0
    fee, _pct = parse_numbers(pd.Series([fee_raw]), decimal_comma=False)
    fee_value = float(fee.iloc[0]) if pd.notna(fee.iloc[0]) else 0.0
    return Fill(
        symbol=symbol,
        side=side,
        qty=qty_value,
        price=price_value,
        date=ts,
        allow_short=allow_short,
        intent=intent,
        fee=abs(fee_value),
        seq=seq,
    )


def _alpaca_flags(result) -> list[str]:
    flags = []
    if result.short_opens:
        flags.append(
            f"{result.short_opens} sell(s) treated as short entries "
            "(buy predates history window - sell price used as cost basis)"
        )
        flags.append("short entry - buy predates history, using sell price as cost basis")
    unmatched = [item for item in result.orphans if item["kind"] != "excess_sell"]
    excess = [item for item in result.orphans if item["kind"] == "excess_sell"]
    if unmatched:
        flags.append(
            f"{len(unmatched)} order(s) were closes with no matching open in this file and were excluded"
        )
    if excess:
        flags.append(
            f"{len(excess)} sell(s) were larger than the open long and the extra shares were not left as a short "
            "because the order was marked to close"
        )
    if result.open_qty:
        shown = ", ".join(f"{symbol}({int(round(qty))})" for symbol, qty in list(result.open_qty.items())[:6])
        flags.append(f" {len(result.open_qty)} open position(s) not included (no closing trade yet): {shown}")
    return flags


def _finish_alpaca(fills: list[Fill], *, assign: bool) -> tuple[pd.DataFrame, list]:
    if not fills:
        return pd.DataFrame(), ["No filled orders found"]
    result = match_fills(fills)
    flags = _alpaca_flags(result)
    if not result.trades:
        return pd.DataFrame(), flags + [
            "No completed round-trip trades found. "
            "All positions may still be open, or all sells predate the history window."
        ]
    frame = trades_frame(result.trades)
    if assign:
        frame, regime_note = assign_regimes(frame)
        if regime_note:
            flags.append(regime_note)
    return frame, flags


def _parse_alpaca_ui_export(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert an Alpaca dashboard orders export into QuantProof trades.

    The orders table downloads (or pastes) as CSV/TSV with Asset or Symbol,
    Side, Filled Qty, Avg. Fill Price, Status, and Filled At. Position Intent,
    when present, is buy_to_open / sell_to_close / sell_to_open / buy_to_close.
    A canceled order that filled some shares is kept for the filled quantity.
    """
    table = read_delimited(raw)
    df = table.frame
    if "symbol" not in df.columns and "asset" in df.columns:
        df = df.rename(columns={"asset": "symbol"})
    required = {"symbol", "side", "avg_fill_price", "filled_at"}
    if "filled_qty" not in df.columns and "qty" not in df.columns:
        required = required | {"filled_qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Alpaca UI export missing columns: {missing}. "
            "Export from the Alpaca dashboard orders table (Asset or Symbol, Side, "
            "Filled Qty, Avg. Fill Price, Status, Filled At)."
        )
    orders = []
    for _, row in df.iterrows():
        orders.append(
            {
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "filled_qty": row["filled_qty"] if "filled_qty" in df.columns else row.get("qty"),
                "avg_fill_price": row.get("avg_fill_price"),
                "status": row.get("status", "filled"),
                "filled_at": row.get("filled_at"),
                "position_intent": row.get("position_intent"),
                "commission": row.get("commission", 0),
            }
        )
    fills = [fill for seq, order in enumerate(orders) if (fill := _order_to_fill(order, seq))]
    frame, flags = _finish_alpaca(fills, assign=False)
    if frame.empty:
        return (
            frame,
            "No filled orders found in Alpaca UI export"
            if not fills
            else ("No completed round-trip trades found in Alpaca UI export."),
            flags,
        )
    note = (
        f"Parsed {len(frame)} completed trade(s) from Alpaca orders "
        f"({len(fills)} filled order(s) → {len(frame)} round-trips). "
        "Returns are price returns. Equity commission is zero at Alpaca; a commission column, when present, is deducted."
    )
    return frame, note, flags + table.notes


def _parse_alpaca_orders_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """CSV saved from the Alpaca orders API or from account FILL activities."""
    table = read_delimited(raw)
    df = table.frame
    orders = []
    for _, row in df.iterrows():
        if "activity_type" in df.columns and str(row.get("activity_type", "")).upper() not in {
            "",
            "FILL",
            "NAN",
        }:
            continue
        orders.append(row.to_dict())
    fills = [fill for seq, order in enumerate(orders) if (fill := _order_to_fill(order, seq))]
    frame, flags = _finish_alpaca(fills, assign=False)
    if frame.empty:
        return frame, "No completed round-trip trades found in the Alpaca export.", flags
    note = f"Parsed {len(frame)} completed trade(s) from Alpaca order history ({len(fills)} filled row(s))."
    return frame, note, flags + table.notes


def pair_alpaca_orders(orders: list) -> tuple[pd.DataFrame, list]:
    """
    Converts raw Alpaca orders into a trades DataFrame.
    Accepts both REST API JSON and UI export dict shapes.
    """
    fills = [fill for seq, order in enumerate(orders) if (fill := _order_to_fill(order, seq))]
    return _finish_alpaca(fills, assign=True)


# =========================================================
# OPTIONS CSV PARSER
# =========================================================


def _is_options_csv(df: pd.DataFrame) -> bool:
    """Token match only. 'iv' must not fire on dividend/delivery, and 'pl' must not fire on multiplier."""
    cols = _columns(df)
    signals = {
        "strike",
        "expiry",
        "expiration",
        "option_type",
        "call_put",
        "put_call",
        "premium",
        "underlying",
        "contract",
        "iv",
        "delta",
        "gamma",
        "theta",
    }
    hits = 0
    for signal in signals:
        if any(
            column == signal or column.endswith("_" + signal) or column.startswith(signal + "_")
            for column in cols
        ):
            hits += 1
    return hits >= 2


def _has_overlapping_positions(df: pd.DataFrame) -> bool:
    [c.lower().strip() for c in df.columns]

    def _is_open_date_col(c):
        cl = c.lower()
        return (
            cl in ["date", "trade_date", "open_date", "entry_date"]
            or any(cl.endswith(x) for x in ["_date", "_time", "date"])
            and not any(x in cl for x in ["close", "exit", "expir", "end"])
        )

    open_col = next((c for c in df.columns if _is_open_date_col(c)), None)
    close_col = next(
        (
            c
            for c in df.columns
            if any(
                x in c.lower()
                for x in ["close_date", "exit_date", "close_time", "end_date", "expiry", "expiration"]
            )
        ),
        None,
    )

    if not open_col or not close_col:
        return False

    try:
        opens = pd.to_datetime(df[open_col], errors="coerce").dropna()
        closes = pd.to_datetime(df[close_col], errors="coerce").dropna()
        if len(opens) < 2 or len(closes) < 2:
            return False
        if opens.min().year < 2000 or closes.min().year < 2000:
            return False

        # Two positions overlap iff a new open starts before any prior close.
        opens_sorted = opens.sort_values().reset_index(drop=True)
        closes_sorted = closes.sort_values().reset_index(drop=True)

        # Merge events: +1 for open, -1 for close, sorted by time
        events = [(t, 1) for t in opens_sorted] + [(t, -1) for t in closes_sorted]
        events.sort(key=lambda x: (x[0], x[1]))  # close before open on same timestamp
        active = 0
        for _, delta in events:
            active += delta
            if active > 1:
                return True

        return False
    except Exception:
        return False


def _named_column(columns, names: tuple[str, ...]) -> str | None:
    for name in names:
        for column in columns:
            if column == name or column.endswith("_" + name):
                return column
    return None


def _parse_options_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    table = read_delimited(raw)
    df = table.frame

    if _has_overlapping_positions(df):
        raise ValueError(
            "Multi-leg or concurrent options positions detected. "
            "QuantProof supports single-leg directional options trades only "
            "(calls or puts bought and fully closed before the next trade). "
            "For spreads, iron condors, or straddles: calculate your daily "
            "P&L as a percentage of capital deployed and upload as a generic CSV."
        )

    ranked = [
        (pnl_column_rank(column), column) for column in df.columns if pnl_column_rank(column) is not None
    ]
    pnl_col = min(ranked)[1] if ranked else None
    if pnl_col is None:
        raise ValueError(
            "Could not find a P&L column in your options CSV. "
            "Expected a column named pnl, profit, return, gain, or pl. "
            f"Columns found: {', '.join(df.columns)}."
        )

    decimal_comma = table.delimiter == ";"
    pnl_vals, pnl_was_percent = parse_numbers(df[pnl_col], decimal_comma=decimal_comma)
    premium_col = _named_column(df.columns, ("premium", "entry_price", "open_price"))
    if premium_col is not None:
        premium_vals, _prem_pct = parse_numbers(df[premium_col], decimal_comma=decimal_comma)
        valid_mask = premium_vals.notna() & premium_vals.abs().gt(1e-9) & pnl_vals.notna()
        returns = (pnl_vals[valid_mask] / premium_vals[valid_mask]).clip(-0.99, 0.99)
        norm_method = f"P&L / {premium_col} (per contract return)"
    else:
        valid_mask = pnl_vals.notna()
        returns = pnl_vals[valid_mask]
        if pnl_was_percent or (len(returns) and returns.abs().mean() > 1.0):
            returns = returns / 100.0
        returns = returns.clip(-0.99, 0.99)
        norm_method = "pnl column treated as fractional return"

    kept = df.loc[valid_mask]
    result = pd.DataFrame({"pnl": returns.to_numpy(), "match_id": range(1, int(valid_mask.sum()) + 1)})

    date_col = _named_column(kept.columns, ("close_date", "exit_date", "date", "trade_date"))
    if date_col:
        parsed_dates, _date_note = parse_datetimes(kept[date_col], dayfirst=False)
        result["date"] = parsed_dates.to_numpy()

    ticker_col = _named_column(kept.columns, ("ticker", "symbol", "underlying", "instrument"))
    if ticker_col:
        result["ticker"] = kept[ticker_col].to_numpy()

    note = f"OPTIONS: {len(result)} single-leg trades parsed. Returns normalised as {norm_method}."
    audit_flags = [
        "OPTIONS DATA: Crash simulations assume linear payoffs ? "
        "options convexity and delta/gamma effects are not modelled. "
        "Results are directionally indicative only.",
        "OPTIONS DATA: Validation assumes sequential non-overlapping trades. "
        "Multi-leg strategies (spreads, condors) are not supported ? "
        "use daily portfolio P&L as percentage of capital instead.",
        "OPTIONS DATA: Crash simulations scale stress from strategy volatility. "
        "High-volatility strategies (options, leveraged) will show elevated crash "
        "exposure - all 5 crises blowing up is expected behaviour for options "
        "strategies and does not indicate a validator error.",
    ]

    return result, note, audit_flags


# =========================================================
# MT5 TRADE HISTORY PARSER
# =========================================================


def _is_mt5_csv(df: pd.DataFrame) -> bool:
    cols = _columns(df)
    # Pandas renames a second Time/Price column Time.1 → time_1.
    if any(column in {"s_l", "s___l"} for column in cols):
        cols.add("s_l")
    deals_signals = {"deal", "direction", "commission", "swap", "balance"}
    deals_hit = len(deals_signals & cols)
    pos_signals = {"position", "s_l", "t_p", "commission", "swap"}
    time_cols = [column for column in cols if "time" in column]
    pos_hit = len(pos_signals & cols) + (1 if len(time_cols) >= 2 else 0)
    return deals_hit >= 3 or pos_hit >= 3


def _mt5_contract(symbol: str) -> float:
    """Standard contract size used only to turn commission and swap into a return.

    The price return itself does not depend on it. FX is 100,000 base units per lot,
    gold 100 oz, silver 5,000 oz. Everything else is treated as 1 unit per lot.
    """
    cleaned = _re.sub(r"[^A-Z]", "", str(symbol).upper())
    if cleaned.startswith(("XAU", "GOLD")):
        return 100.0
    if cleaned.startswith(("XAG", "SILVER")):
        return 5_000.0
    if _re.fullmatch(r"[A-Z]{6}", cleaned):
        return 100_000.0
    return 1.0


def _mt5_side(value: object) -> str | None:
    text = str(value).lower().strip()
    if text in {"buy", "0", "buy limit", "buy stop"}:
        return "long"
    if text in {"sell", "1", "sell limit", "sell stop"}:
        return "short"
    return None


def _mt5_direction(value: object) -> str:
    text = str(value).lower().strip().replace(" ", "")
    return {
        "in": "in",
        "entry": "in",
        "0": "in",
        "out": "out",
        "exit": "out",
        "1": "out",
        "in/out": "in/out",
        "inout": "in/out",
        "2": "in/out",
        "out/in": "out/in",
        "outin": "out/in",
        "3": "out/in",
    }.get(text, "")


def _mt5_price_return(
    side: str, entry: float, exit_price: float, cost: float, volume: float, contract: float
):
    if entry <= 0 or exit_price <= 0 or volume <= 0:
        return None
    gross = (exit_price - entry) / entry if side == "long" else (entry - exit_price) / entry
    denom = entry * volume * contract
    net = gross + (cost / denom if abs(denom) > 1e-12 else 0.0)
    from services.csv_util import clip_return

    return round(clip_return(float(net)), 6)


def _parse_mt5_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """MT5 history.

    Deals view: one row per execution, Direction in/out, matched FIFO.
    A Position column (hedging accounts) keeps concurrent tickets apart.
    Positions view: one row per round trip, two Time columns and two Price columns.
    The return is the price return. Commission, swap, and fee are deducted using a
    standard contract size; they are not guessed from the Profit column, which is
    in account currency and is wrong by ~100,000x if divided by price × lots.
    """
    table = read_delimited(raw)
    df = table.frame
    if df.empty:
        raise ValueError("MT5 CSV appears empty after parsing.")
    if "direction" not in df.columns and "entry" in df.columns:
        df["direction"] = df["entry"]
    has_direction = "direction" in df.columns
    has_position = "position" in df.columns
    if not has_direction and not has_position:
        raise ValueError(
            "Could not identify MT5 report format. "
            "In the terminal open Toolbox → History, choose Deals or Positions, "
            "right-click, and save the report. Activity statements from other platforms are not MT5 history."
        )

    decimal_comma = table.delimiter == ";"
    for column in list(df.columns):
        if any(
            token in column for token in ("price", "volume", "profit", "commission", "swap", "fee", "balance")
        ):
            numbers, _pct = parse_numbers(df[column], decimal_comma=decimal_comma)
            df[column] = numbers

    time_cols = [column for column in df.columns if "time" in column]
    for column in time_cols:
        parsed, _note = parse_datetimes(df[column], dayfirst=False)
        df[column] = parsed
    open_time_col = time_cols[0] if time_cols else None
    close_time_col = time_cols[1] if len(time_cols) >= 2 else open_time_col
    symbol_col = next(
        (column for column in df.columns if column in {"symbol", "instrument", "pair", "asset"}), None
    )
    price_cols = [column for column in df.columns if "price" in column]

    if "type" in df.columns:
        sides = df["type"].map(_mt5_side)
        df = df.loc[sides.notna()].copy()
    if df.empty:
        raise ValueError(
            "No trade rows found in MT5 export after filtering. "
            "Balance, credit, and deposit rows are ignored. "
            "Export closed trades from Account History, not only the current open positions."
        )

    audit_flags = list(table.notes)
    trades = []
    orphan_count = 0
    assumed_contract = False

    def _num(row, column, default=0.0) -> float:
        if column not in row.index:
            return default
        value = row[column]
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if number != number:
            return default
        return number

    if has_direction:
        from collections import defaultdict, deque

        books: dict[str, deque] = defaultdict(deque)
        df = df.sort_values(open_time_col) if open_time_col else df

        def _key(row) -> str:
            symbol = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            if (
                has_position
                and pd.notna(row.get("position"))
                and str(row.get("position")).strip() not in {"", "nan"}
            ):
                return f"{symbol}|{row['position']}"
            return symbol

        def _close(book, qty, exit_price, date, cost, symbol):
            nonlocal orphan_count
            remaining = qty
            closed = 0.0
            while remaining > 1e-9 and book:
                lot = book[0]
                close_qty = min(remaining, lot["qty"])
                entry_cost = lot["cost"] * (close_qty / lot["orig"]) if lot["orig"] else 0.0
                lot["cost"] -= entry_cost
                slice_cost = entry_cost + cost * (close_qty / qty if qty else 0.0)
                pnl = _mt5_price_return(
                    lot["side"], lot["price"], exit_price, slice_cost, close_qty, _mt5_contract(symbol)
                )
                if pnl is not None:
                    trades.append(
                        {
                            "pnl": pnl,
                            "date": date,
                            "ticker": symbol,
                            "side": lot["side"],
                            "match_id": len(trades) + 1,
                        }
                    )
                    closed += close_qty
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] <= 1e-8:
                    book.popleft()
            if remaining > 1e-6 and closed <= 0:
                orphan_count += 1
            return remaining

        for _, row in df.iterrows():
            direction = _mt5_direction(row.get("direction"))
            symbol = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            if _mt5_contract(symbol) != 1.0:
                assumed_contract = True
            side = _mt5_side(row.get("type")) or "long"
            volume = _num(row, "volume")
            price = _num(row, price_cols[0]) if price_cols else 0.0
            cost = _num(row, "commission") + _num(row, "swap") + _num(row, "fee")
            date = row[open_time_col] if open_time_col else pd.NaT
            book = books[_key(row)]
            if direction in {"in/out", "out/in"}:
                leftover = _close(book, volume, price, date, cost, symbol)
                if leftover > 1e-8:
                    book.append(
                        {"qty": leftover, "orig": leftover, "price": price, "side": side, "cost": 0.0}
                    )
                continue
            if direction == "in":
                book.append({"qty": volume, "orig": volume, "price": price, "side": side, "cost": cost})
                continue
            if direction == "out":
                if not book:
                    orphan_count += 1
                    continue
                _close(book, volume, price, date, cost, symbol)
        open_count = sum(len(book) for book in books.values() if book)
        if open_count:
            audit_flags.append(f" {open_count} MT5 position(s) still open (no closing deal) excluded")
    else:
        open_price_col = price_cols[0] if price_cols else None
        close_price_col = price_cols[1] if len(price_cols) > 1 else None
        for _, row in df.iterrows():
            symbol = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            if _mt5_contract(symbol) != 1.0:
                assumed_contract = True
            side = _mt5_side(row.get("type")) or "long"
            volume = _num(row, "volume", 1.0) or 1.0
            entry = _num(row, open_price_col) if open_price_col else 0.0
            exit_price = _num(row, close_price_col) if close_price_col else 0.0
            cost = _num(row, "commission") + _num(row, "swap") + _num(row, "fee")
            date = row[close_time_col] if close_time_col else pd.NaT
            if exit_price > 0 and entry > 0:
                pnl = _mt5_price_return(side, entry, exit_price, cost, volume, _mt5_contract(symbol))
            else:
                pnl_raw = _normalise_pnl(
                    _num(row, "profit") + cost, entry, volume, 0.0, contract=_mt5_contract(symbol)
                )
                pnl = None if pnl_raw is None else round(float(pnl_raw), 6)
            if pnl is not None:
                trades.append(
                    {"pnl": pnl, "date": date, "ticker": symbol, "side": side, "match_id": len(trades) + 1}
                )

    if not trades:
        raise ValueError(
            "No completed trades could be extracted from the MT5 export. "
            "The report needs closed trades, with a Price column and either a Direction column (deals) "
            "or two Price columns (positions)."
        )
    result = trades_frame(trades)
    if orphan_count:
        audit_flags.append(
            f"⚠️ {orphan_count} MT5 closing deal(s) had no matching open "
            " excluded (position predates history window)"
        )
    audit_flags.append(
        "MT5 DATA: Returns are price returns, (exit - entry) / entry for longs and the reverse for shorts. "
        "Commission, swap, and fee are deducted using a standard contract size "
        "(100,000 units for a 6-letter FX pair, 100 oz for gold, 1 unit otherwise). "
        "A broker with a different contract size mis-states the fee, not the price return. "
        "Times are the broker server clock, not converted to your timezone."
    )
    if assumed_contract:
        audit_flags.append(
            "MT5 contract size was assumed for the fee term on FX or metal symbols. "
            "Confirm it against the symbol specification if those charges are large relative to the price move."
        )
    report = "deals" if has_direction else "positions"
    note = f"📊 MT5: {len(trades)} completed trade(s) parsed from {report} report ({orphan_count} orphan(s) excluded)"
    return result, note, audit_flags


def _normalise_pnl(
    net_profit: float, entry_price: float, volume: float, balance: float, contract: float = 1.0
) -> float | None:
    """
    Normalise an MT5 profit figure to a decimal return.

    ``contract`` is the units per lot (100,000 for FX). Returns None when the
    denominator is zero so a missing price is skipped instead of invented.
    ``balance`` is unused; it used to feed a ratio that was not a return.
    """
    del balance
    denom = entry_price * volume * (contract or 1.0)
    if abs(denom) > 1e-6:
        pnl = net_profit / denom
        return float(np.clip(pnl, -0.99, 0.99))
    return None


def _is_tradingview_csv(df: pd.DataFrame) -> bool:
    """List of Trades, old and current. Entry and exit rows share a trade number."""
    frame = df.copy()
    frame.columns = [normalize_header(column) for column in frame.columns]
    if "type" not in frame.columns:
        return False
    if not ({"trade", "trade_number", "trade_no", "signal"} & set(frame.columns)):
        return False
    kinds = frame["type"].astype(str).str.lower()
    return bool(kinds.str.contains(r"\b(?:entry|exit)\b", regex=True).any())


def _tv_percent_column(columns) -> str | None:
    preferred = ("return_pct", "net_pnl_pct", "net_p_and_l_pct", "profit_pct", "net_profit_pct")
    for name in preferred:
        if name in columns:
            return name
    for column in columns:
        if not str(column).endswith("_pct"):
            continue
        if any(
            skip in column for skip in ("cumul", "favorable", "adverse", "run_up", "drawdown", "excursion")
        ):
            continue
        return column
    return None


def _parse_tradingview_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """TradingView Strategy Tester → Trades tab → download.

    Each closed trade is an entry row and an exit row. The currency P&L column
    is dollars; the percentage column is the return. Using both rows, or the
    dollar column, double-counts trades and clips them toward ±99%.
    """
    table = read_delimited(raw)
    df = table.frame
    percent_col = _tv_percent_column(df.columns)
    if percent_col is None:
        raise TradeImportError(
            "This TradingView List of Trades has no percentage column (Profit %, Net PnL %, or Return %). "
            "The currency P&L column is dollars, and treating dollars as returns clips every trade toward ±99%. "
            "On the Trades tab, include the percentage column and download the CSV again."
        )
    id_col = next((name for name in ("trade", "trade_number", "trade_no") if name in df.columns), None)
    kinds = df["type"].astype(str)
    exits = kinds.str.lower().str.contains(r"\bexit\b", regex=True)
    entries = kinds.str.lower().str.contains(r"\bentry\b", regex=True)
    if exits.any():
        chosen = df.loc[exits].copy()
    elif not entries.any():
        chosen = df.copy()
    else:
        raise TradeImportError(
            "This TradingView file has entry rows but no exit rows, so nothing has closed. "
            "Download the List of Trades after the strategy has closed trades."
        )
    decimal_comma = table.delimiter == ";"
    numbers, _was_percent = parse_numbers(chosen[percent_col], decimal_comma=decimal_comma)
    bad = []
    for raw_cell in chosen[percent_col].tolist():
        text = "" if raw_cell is None else str(raw_cell).strip()
        if text == "":
            continue
        number, _percent = parse_number(text, decimal_comma=decimal_comma)
        if text.lower() in {"nan", "inf", "-inf", "+inf", "infinity"} or not np.isfinite(number):
            bad.append(text)
    if bad:
        shown = ", ".join(repr(value) for value in bad[:5])
        raise TradeImportError(
            f"TradingView column '{percent_col}' contains non-numeric values ({shown}). "
            "inf, nan, and text are not returns."
        )
    mask = numbers.notna() & np.isfinite(numbers)
    kept = chosen.loc[mask]
    returns = (numbers[mask] / 100.0).clip(-0.99, 0.99)
    result = pd.DataFrame(
        {
            "pnl": returns.to_numpy(),
            "match_id": np.arange(1, int(mask.sum()) + 1),
            "side": ["short" if "short" in str(value).lower() else "long" for value in kept["type"].tolist()],
        }
    )
    date_col = next(
        (name for name in ("date_and_time", "date_time", "date", "time") if name in kept.columns),
        None,
    )
    flags = list(table.notes)
    if date_col:
        parsed, date_note = parse_datetimes(kept[date_col], dayfirst=False)
        result["date"] = parsed.to_numpy()
        if date_note:
            flags.append(date_note)
        flags.append("TradingView times are the chart timezone and were not converted.")
    if id_col and entries.any() and exits.any():
        open_ids = set(df.loc[entries, id_col].astype(str)) - set(df.loc[exits, id_col].astype(str))
        if open_ids:
            flags.append(f" {len(open_ids)} TradingView trade(s) still open (entry without an exit) excluded")
    if result.empty:
        raise TradeImportError(
            "No closed TradingView trades were found. Exit rows need a numeric percentage return."
        )
    note = (
        f"Parsed {len(result)} closed trade(s) from TradingView List of Trades. "
        f"Entry rows were not counted again. Returns were taken from {percent_col}, not the currency P&L column."
    )
    return result, note, flags


def _date_sanity_note(df: pd.DataFrame) -> str:
    if "date" not in df.columns:
        return ""
    years = pd.to_datetime(df["date"], errors="coerce").dt.year
    notes = []
    future = years[years > pd.Timestamp.utcnow().year + 1]
    if not future.empty:
        shown = ", ".join(str(year) for year in sorted(set(int(year) for year in future.dropna()))[:4])
        notes.append(
            f"Warning: {int(future.shape[0])} trade date(s) are in {shown}, which is after a backtest "
            "should have ended. Check the date column before trusting the score."
        )
    ancient = years[(years < 1990) & years.notna()]
    if not ancient.empty:
        notes.append(
            f"Warning: {int(ancient.shape[0])} trade date(s) are before 1990. "
            "That usually means a date was read as a Unix timestamp or the year and day were swapped."
        )
    return " | ".join(notes)


def _guard_trades(df: pd.DataFrame, note: str) -> tuple[pd.DataFrame, str]:
    if len(df) > MAX_TRADES_PER_RUN:
        raise TradeImportError(
            f"This file contains {len(df)} closed trades. QuantProof scores at most "
            f"{MAX_TRADES_PER_RUN} trades per run — the partial-fill and Monte Carlo checks "
            "copy the whole series hundreds of times and exhaust the server. "
            f"Export a shorter date range (the most recent {MAX_TRADES_PER_RUN} trades is enough) "
            "and upload that file."
        )
    if "pnl" in df.columns:
        values = pd.to_numeric(df["pnl"], errors="coerce")
        if len(values) and not np.isfinite(values.to_numpy(dtype=float)).all():
            raise TradeImportError(
                "The parsed returns contain inf or nan. Delete those cells or replace them with a decimal return."
            )
    warning = _date_sanity_note(df)
    if warning:
        note = f"{note} | {warning}" if note else warning
    return df, note


def _finish_parsed(
    df: pd.DataFrame, note: str, flags: list, benchmark: str | None
) -> tuple[pd.DataFrame, str]:
    if df is None or len(df) == 0:
        detail = note or "No completed trades in this file."
        if flags:
            detail = detail + " " + " | ".join(str(flag) for flag in flags[:6])
        raise TradeImportError(detail)
    df, note = _guard_trades(df, note)
    if flags:
        if benchmark == "options":
            note += " | OPTIONS_FLAGS:" + "||".join(str(flag) for flag in flags)
        elif benchmark == "mt5":
            note += " | MT5_FLAGS:" + "||".join(str(flag) for flag in flags)
        else:
            note += " | " + " | ".join(str(flag) for flag in flags[:4])
    regime_key = None if benchmark in {None, "options", "mt5"} else benchmark
    if "date" in df.columns and df["date"].notna().any():
        bm_type = regime_key or get_benchmark_type(df, None)
        df, regime_note = assign_regimes(df, bm_type)
        if regime_note:
            note += " | " + regime_note
    return df, note


def _call_parser(parser, raw: str, label: str):
    try:
        return parser(raw)
    except TradeImportError:
        raise
    except ValueError as exc:
        raise TradeImportError(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise TradeImportError(f"{label} was detected but could not be parsed: {exc}") from exc


def _screen_return_column(series: pd.Series, *, decimal_comma: bool) -> None:
    bad: list[str] = []
    for raw in series.tolist():
        text = "" if raw is None else str(raw).strip()
        if text == "" or text.lower() in {"none", "null"}:
            continue
        number, _percent = parse_number(text, decimal_comma=decimal_comma)
        if text.lower() in {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"} or not np.isfinite(number):
            bad.append(text)
            if len(bad) >= 5:
                break
    if bad:
        shown = ", ".join(repr(value) for value in bad)
        raise TradeImportError(
            f"The return column contains values that are not numbers ({shown}). "
            "inf, nan, and text such as 'abc' are not returns, and they are not dropped quietly. "
            "Delete those rows or replace them with a decimal (0.015) or a percent (1.5%)."
        )


def _parse_generic_frame(table) -> tuple[pd.DataFrame, str, list]:
    df = table.frame
    decimal_comma = table.delimiter == ";"
    ranked = [(rank, column) for column in df.columns if (rank := pnl_column_rank(column)) is not None]
    if not ranked:
        cols = ", ".join(list(df.columns)[:12]) or "(none)"
        lowered = " ".join(df.columns)
        if "profit_factor" in df.columns or ("gross_profit" in lowered and "net_profit" in lowered):
            raise TradeImportError(
                "This looks like a TradingView performance summary, not the List of Trades. "
                "Open the Strategy Tester, switch to the Trades tab, and download that CSV."
            )
        raise TradeImportError(
            f"No return column found. Columns seen: {cols}. "
            "QuantProof reads a Zerodha Console tradebook, a Kite orders export, an Alpaca orders export, "
            "an MT5 deals or positions report, a TradingView List of Trades (Trades tab, not the performance summary), "
            "an IBKR Flex file, or a CSV with date and pnl columns. "
            "Rename your return column to pnl if this is your own spreadsheet."
        )
    column = min(ranked)[1]
    _screen_return_column(df[column], decimal_comma=decimal_comma)
    numbers, was_percent = parse_numbers(df[column], decimal_comma=decimal_comma)
    mask = numbers.notna() & np.isfinite(numbers)
    skipped = int((~mask).sum())
    values = numbers[mask]
    if values.empty:
        raise TradeImportError(
            f"Column '{column}' has no numeric returns. "
            "A score needs at least 10 closed trades; this file has none."
        )
    notes = list(table.notes)
    if column.endswith("_pct") or was_percent:
        values = values / 100.0
        notes.append(f"Column {column} was read as a percentage, so 1.5 means 1.5% (0.015).")
    else:
        divided = bool(values.abs().mean() > 1.0)
        if divided:
            values = values / 100.0
        clipped_pct = float((values.abs() > 0.50).mean())
        if divided and clipped_pct > 0.80:
            raise TradeImportError(
                "PnL values look like dollar amounts or prices, not decimal returns. "
                f"{int(clipped_pct * 100)}% of them are still larger than 50% after dividing by 100. "
                "Upload decimals (0.015 = 1.5%) or a percent column such as '1.5%'."
            )
        if clipped_pct > 0.50:
            notes.append(
                "Many returns were clipped to ±0.99. The validator cannot score a single trade above 99%. "
                "If these are dollar amounts, upload a percentage column instead."
            )
    values = values.clip(-0.99, 0.99)
    result = pd.DataFrame({"pnl": values.to_numpy(), "match_id": np.arange(1, len(values) + 1)})
    date_col = next(
        (
            name
            for name in (
                "date",
                "close_date",
                "exit_date",
                "trade_date",
                "date_time",
                "date_and_time",
                "time",
            )
            if name in df.columns
        ),
        None,
    )
    if date_col:
        parsed, date_note = parse_datetimes(df.loc[mask, date_col], dayfirst=decimal_comma)
        result["date"] = parsed.to_numpy()
        if date_note:
            notes.append(date_note)
    if skipped:
        notes.append(f"Skipped {skipped} blank return cell(s). The other rows were kept.")
    return result, "", notes


def _headerless_returns(text: str) -> tuple[pd.DataFrame, str]:
    """A file with no header is accepted only when the numbers already look like returns."""
    table_text = text
    try:
        frame = pd.read_csv(io.StringIO(table_text), header=None, dtype=str, on_bad_lines="skip")
    except Exception as exc:
        raise TradeImportError(f"Could not parse this file: {exc}") from exc
    for column in frame.columns:
        numbers, was_percent = parse_numbers(frame[column], decimal_comma=False)
        values = numbers.dropna()
        if len(values) < 1 or not np.isfinite(values).all():
            continue
        if was_percent:
            values = values / 100.0
        elif float(values.abs().mean()) > 1.0 and (values < 0).any():
            values = values / 100.0
        if (values > 0).any() and (values < 0).any() and float(values.abs().median()) <= 1.0:
            clipped = values.clip(-0.99, 0.99)
            result = pd.DataFrame({"pnl": clipped.to_numpy(), "match_id": np.arange(1, len(clipped) + 1)})
            return _guard_trades(
                result,
                "No header was found. The numeric column was read as returns. "
                "Add a header row named date,pnl if that is not what the file contains.",
            )
    raise TradeImportError(
        "No header row and no return column were found. "
        "Add a header with a pnl column (decimals or percents), or export a broker file "
        "QuantProof recognises (Zerodha tradebook, Alpaca, MT5, TradingView List of Trades, IBKR Flex)."
    )


def smart_parse_excel(contents: bytes) -> tuple[pd.DataFrame, str]:
    """Excel workbooks use the same detection as CSV uploads."""
    return smart_parse_csv(contents)


def smart_parse_csv(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parse a broker export or a generic return file into closed trades.

    A format that is recognised and then yields no closed trades is rejected.
    It is not handed to the generic parser, which used to read prices as returns.
    """
    if not contents or not contents.strip(b"\x00 \t\r\n"):
        raise TradeImportError("The file is empty. Export a trade history and upload that file.")
    if contents.startswith(b"\xd0\xcf\x11\xe0"):
        raise TradeImportError("This is an old Excel .xls file. Save it as .xlsx or CSV and upload that.")
    if contents.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        try:
            text = excel_to_csv_text(contents)
        except ValueError as exc:
            raise TradeImportError(str(exc)) from exc
    elif classify_ibkr_bytes(contents) != "not_ibkr":
        try:
            frame, note = parse_ibkr_upload(contents)
        except ValueError as exc:
            raise TradeImportError(str(exc)) from exc
        return _finish_parsed(frame, note, [], None)
    else:
        try:
            text = decode_upload(contents)
        except ValueError as exc:
            raise TradeImportError(str(exc)) from exc

    try:
        table = read_delimited(text)
    except ValueError as exc:
        raise TradeImportError(str(exc)) from exc
    if not table.header_found:
        return _headerless_returns(text)

    probe = table.frame
    routes = (
        (_is_tradingview_csv, _parse_tradingview_csv, "TradingView List of Trades", None),
        (_is_options_csv, _parse_options_csv, "Options CSV", "options"),
        (_is_mt5_csv, _parse_mt5_csv, "MT5 report", "mt5"),
        (_is_kite_csv, _parse_kite_csv, "Zerodha tradebook", "nifty"),
        (_is_kite_orders_csv, _parse_kite_orders_csv, "Kite orders", "nifty"),
        (_is_alpaca_ui_export, _parse_alpaca_ui_export, "Alpaca orders", None),
        (_is_alpaca_orders_csv, _parse_alpaca_orders_csv, "Alpaca orders", None),
        (_is_alpaca_activities_csv, _parse_alpaca_orders_csv, "Alpaca fills", None),
    )
    for detect, parser, label, benchmark in routes:
        if detect(probe):
            frame, note, flags = _call_parser(parser, text, label)
            return _finish_parsed(frame, note, flags, benchmark)

    head = "\n".join(text.splitlines()[:8]).lower()
    for name in ("upstox", "groww", "angel one", "angel broking", "dhan"):
        if name in head and pnl_column_rank(next(iter(probe.columns), "")) is None:
            if not any(pnl_column_rank(column) is not None for column in probe.columns):
                raise TradeImportError(
                    f"This looks like a {name.title()} export. QuantProof does not parse {name.title()} yet. "
                    "Upload a Zerodha Console tradebook, or a CSV with date and pnl columns."
                )
    frame, note, flags = _parse_generic_frame(table)
    return _finish_parsed(frame, note, flags, None)


def _dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a repeated upload of the same dated trades.

    Rows with no date are kept even when the return is identical: two trades
    can both make 1%, and a dateless file has nothing else to tell them apart.
    ``match_id`` keeps same-day round trips that really did happen twice.
    """
    if df.empty or "pnl" not in df.columns:
        return df
    if "match_id" in df.columns:
        keys = [column for column in ("match_id", "date", "ticker", "pnl", "side") if column in df.columns]
    elif "date" in df.columns and df["date"].notna().any():
        keys = [column for column in ("date", "ticker", "pnl", "side") if column in df.columns]
        dated = df["date"].notna()
        if not bool(dated.all()):
            kept = df.loc[dated].drop_duplicates(subset=keys).reset_index(drop=True)
            undated = df.loc[~dated].reset_index(drop=True)
            return pd.concat([kept, undated], ignore_index=True)
    else:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=keys).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[dedup] Removed {removed} duplicate trade row(s)")
    return df
