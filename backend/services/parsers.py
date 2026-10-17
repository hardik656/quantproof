"""Broker export parsers (Kite, Alpaca, options, MT5, generic CSV/Excel) with FIFO lot matching.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import asyncio
import io
import re as _re
from collections import defaultdict
from functools import wraps

import numpy as np
import pandas as pd
from fastapi import HTTPException

from services.benchmarks import assign_regimes, get_benchmark_type

# =========================================================
# KITE TRADEBOOK PARSER
# =========================================================


def _is_kite_csv(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip() for c in df.columns}
    required = {"trade_date", "trade_type", "quantity", "price"}
    kite_signals = {"trade_time", "settlement_id", "series", "order_id", "trade_id"}
    return required.issubset(cols) and bool(kite_signals & cols)


def _parse_kite_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Kite TradeBook CSV into QuantProof trades.
    """
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"trade_date", "trade_type", "quantity", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kite CSV missing required columns: {missing}. "
            "Please export from Zerodha Console → TradeBook → Download CSV. "
            "Make sure you are downloading the TradeBook, not Positions or Holdings."
        )

    ticker_col = None
    # NOTE: 'series' is excluded — it's a segment identifier (EQ/FO/CDS),
    # not an instrument name. Using it as a ticker key buckets ALL equity
    # trades under 'EQ' and ALL F&O trades under 'FO', breaking FIFO matching.
    for candidate in ["tradingsymbol", "symbol", "instrument", "scrip"]:
        if candidate in df.columns:
            ticker_col = candidate
            break

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"])

    sort_cols = [ticker_col, "trade_date"] if ticker_col and ticker_col in df.columns else ["trade_date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    lots = defaultdict(list)
    trades = []
    audit_flags = []
    orphan_sells = []

    for _, row in df.iterrows():
        symbol = str(row[ticker_col]) if ticker_col and ticker_col in row else "UNKNOWN"
        qty = float(row["quantity"])
        price = float(row["price"])
        trade_type = str(row["trade_type"]).upper().strip()
        date = row["trade_date"]
        segment = str(row.get("segment", "EQ")).upper().strip()
        is_fo = segment in (
            "FO",
            "F&O",
            "FUTURES",
            "OPTIONS",
            "CDS",
            "MF",
            "NFO",
            "BCD",
            "MCX",
            "NCO",
        ) or str(row.get("series", "")).upper().strip() in ("FO", "CD")

        # FIX: skip zero/negative qty rows — prevents phantom trades and div-by-zero
        if qty <= 0 or price <= 0:
            audit_flags.append(
                f"⚠️ Skipped row: {symbol} {trade_type} qty={qty} price={price} "
                f"on {date.date() if hasattr(date, 'date') else date} invalid qty/price"
            )
            continue

        if trade_type == "BUY":
            if lots[symbol] and lots[symbol][0]["side"] == "short":
                remaining_qty = qty
                while remaining_qty > 0 and lots[symbol] and lots[symbol][0]["side"] == "short":
                    lot = lots[symbol][0]
                    close_qty = min(remaining_qty, lot["qty"])
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[symbol].pop(0)
                        continue
                    trades.append(
                        {
                            "pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6),
                            "ticker": symbol,
                            "side": "short",
                            "date": date,
                        }
                    )
                    lot["qty"] -= close_qty
                    remaining_qty -= close_qty
                    if lot["qty"] == 0:
                        lots[symbol].pop(0)
                if remaining_qty > 0:
                    lots[symbol].append({"qty": remaining_qty, "price": price, "date": date, "side": "long"})
            else:
                lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "long"})

        elif trade_type == "SELL":
            if not lots[symbol]:
                if is_fo:
                    lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "short"})
                else:
                    orphan_sells.append(
                        f"{symbol} SELL {qty:.0f}@{price:.2f} on {date.date()} "
                        f"(no open position - predates history window)"
                    )
                continue

            remaining_qty = qty
            while remaining_qty > 0 and lots[symbol] and lots[symbol][0]["side"] == "long":
                lot = lots[symbol][0]
                close_qty = min(remaining_qty, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[symbol].pop(0)
                    continue
                trades.append(
                    {
                        "pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6),
                        "ticker": symbol,
                        "side": "long",
                        "date": date,
                    }
                )
                lot["qty"] -= close_qty
                remaining_qty -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

        elif trade_type not in ("BUY", "SELL"):
            # FIX: flag unknown trade types instead of silently dropping
            audit_flags.append(
                f"⚠️ Skipped unknown trade_type '{trade_type}' for {symbol} "
                f"on {date.date() if hasattr(date, 'date') else date}"
            )

    if orphan_sells:
        audit_flags.append(
            f"⚠️ {len(orphan_sells)} sell(s) had no matching buy "
            f"(position predates TradeBook history) excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(lot["qty"] for lot in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(f"{s}({int(q)})" for s, q in list(open_positions.items())[:6])
        audit_flags.append(
            f" {len(open_positions)} position(s) still open (no closing trade yet): {sym_list}"
        )

    if not trades:
        return pd.DataFrame(), "No completed trades found in Kite TradeBook", audit_flags

    result = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    note = (
        f"📊 Parsed {len(trades)} completed trade(s) from Zerodha Kite TradeBook "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(open_positions)} position(s) still open)"
    )
    return result, note, audit_flags


# =========================================================
# KITE ORDERS CSV PARSER
# =========================================================


def _is_kite_orders_csv(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip() for c in df.columns}
    required = {"transaction_type", "order_execution_time", "tradingsymbol", "filled_quantity"}
    return required.issubset(cols)


def _parse_kite_orders_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Zerodha Kite Orders CSV into QuantProof trades.
    """
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.lower().strip() for c in df.columns]

    required = {
        "transaction_type",
        "order_execution_time",
        "tradingsymbol",
        "filled_quantity",
        "average_price",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kite Orders CSV missing required columns: {missing}. "
            "Export from Kite Console → Orders → Download CSV."
        )

    df["status"] = df.get("status", pd.Series(["COMPLETE"] * len(df))).str.upper().str.strip()
    df = df[df["status"] == "COMPLETE"].copy()

    df["filled_quantity"] = pd.to_numeric(df["filled_quantity"], errors="coerce")
    df["average_price"] = pd.to_numeric(df["average_price"], errors="coerce")
    df = df.dropna(subset=["filled_quantity", "average_price"])
    df = df[(df["filled_quantity"] > 0) & (df["average_price"] > 0)]

    df["order_execution_time"] = pd.to_datetime(df["order_execution_time"], errors="coerce")
    df = df.dropna(subset=["order_execution_time"])

    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
    df = df[df["transaction_type"].isin(["BUY", "SELL"])]

    if df.empty:
        return pd.DataFrame(), "No COMPLETE orders found in Kite Orders CSV", []

    df = df.sort_values(["tradingsymbol", "order_execution_time"]).reset_index(drop=True)
    df["segment"] = df.get("segment", pd.Series(["EQ"] * len(df))).str.upper().str.strip()

    lots = defaultdict(list)
    trades = []
    audit_flags = []
    orphan_sells = []

    for _, row in df.iterrows():
        symbol = str(row["tradingsymbol"])
        side = row["transaction_type"]
        qty = float(row["filled_quantity"])
        price = float(row["average_price"])
        date = row["order_execution_time"]
        is_fo = row["segment"] in ("FO", "F&O", "FUTURES", "OPTIONS", "CDS")

        if side == "BUY":
            if lots[symbol] and lots[symbol][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[symbol] and lots[symbol][0]["side"] == "short":
                    lot = lots[symbol][0]
                    close_qty = min(remaining, lot["qty"])
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[symbol].pop(0)
                        continue
                    trades.append(
                        {
                            "pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6),
                            "ticker": symbol,
                            "side": "short",
                            "date": date,
                        }
                    )
                    lot["qty"] -= close_qty
                    remaining -= close_qty
                    if lot["qty"] == 0:
                        lots[symbol].pop(0)
                if remaining > 0:
                    lots[symbol].append({"qty": remaining, "price": price, "date": date, "side": "long"})
            else:
                lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "long"})

        elif side == "SELL":
            if not lots[symbol]:
                if is_fo:
                    lots[symbol].append({"qty": qty, "price": price, "date": date, "side": "short"})
                else:
                    orphan_sells.append(
                        f"{symbol} SELL {qty:.0f}@{price:.2f} on {date.date()} "
                        f"(no matching BUY - predates history window)"
                    )
                continue

            remaining = qty
            while remaining > 0 and lots[symbol] and lots[symbol][0]["side"] == "long":
                lot = lots[symbol][0]
                close_qty = min(remaining, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[symbol].pop(0)
                    continue
                trades.append(
                    {
                        "pnl": round(np.clip(pnl_pct, -0.99, 0.99), 6),
                        "ticker": symbol,
                        "side": "long",
                        "date": date,
                    }
                )
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] == 0:
                    lots[symbol].pop(0)

    if orphan_sells:
        audit_flags.append(
            f"⚠️ {len(orphan_sells)} sell(s) had no matching BUY (position predates Orders history) excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(lot["qty"] for lot in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(f"{s}({int(q)})" for s, q in list(open_positions.items())[:6])
        audit_flags.append(
            f" {len(open_positions)} position(s) still open (no closing trade yet): {sym_list}"
        )

    if not trades:
        return (
            pd.DataFrame(),
            "No completed round-trip trades found in Kite Orders CSV",
            audit_flags,
        )

    result = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    note = (
        f"Parsed {len(trades)} completed trade(s) from Zerodha Kite Orders CSV "
        f"({len(orphan_sells)} orphan sell(s) excluded, "
        f"{len(open_positions)} position(s) still open)"
    )
    return result, note, audit_flags


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


def _is_alpaca_ui_export(df: pd.DataFrame) -> bool:
    cols = {c.lower().strip().replace(".", "") for c in df.columns}
    return ("avg fill price" in cols or "avg  fill price" in cols) and "filled at" in cols and "side" in cols


def _parse_alpaca_ui_date(s) -> pd.Timestamp | None:
    if not s or str(s).strip() in ("-", "", "nan", "None"):
        return None
    try:
        return pd.to_datetime(str(s).strip(), errors="coerce")
    except Exception:
        return None


def _parse_alpaca_ui_export(raw: str) -> tuple[pd.DataFrame, str, list]:
    """
    Convert Alpaca UI order export (tab-separated) into QuantProof trades.
    """
    audit_flags = []

    sample = raw[:4096]
    delim = "\t" if sample.count("\t") > sample.count(",") else ","

    try:
        df = pd.read_csv(io.StringIO(raw), sep=delim, dtype=str)
    except Exception as e:
        raise ValueError(f"Could not parse Alpaca UI export: {e}")

    def _norm_col(c):
        return c.lower().strip().replace(".", "").replace("  ", " ")

    df.columns = [_norm_col(c) for c in df.columns]

    col_map = {
        "asset": "symbol_raw",
        "side": "side",
        "qty": "qty",
        "filled qty": "filled_qty",
        "avg fill price": "avg_fill_price",
        "status": "status",
        "filled at": "filled_at",
        "submitted at": "submitted_at",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = {"symbol_raw", "side", "filled_qty", "avg_fill_price", "status", "filled_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Alpaca UI export missing columns: {missing}. "
            "Export from Alpaca dashboard → Orders → copy/download table."
        )

    df["status"] = df["status"].str.lower().str.strip()
    df = df[df["status"] == "filled"].copy()

    df["filled_qty"] = pd.to_numeric(df["filled_qty"], errors="coerce")
    df["avg_fill_price"] = pd.to_numeric(df["avg_fill_price"], errors="coerce")
    df = df.dropna(subset=["filled_qty", "avg_fill_price"])
    df = df[(df["filled_qty"] > 0) & (df["avg_fill_price"] > 0)]

    df["filled_at"] = df["filled_at"].apply(_parse_alpaca_ui_date)
    df = df.dropna(subset=["filled_at"])

    def _clean_ticker(raw_val):
        if pd.isna(raw_val):
            return "UNKNOWN"
        s = str(raw_val).strip()
        m = _re.match(r"([A-Z]{1,6})", s.upper())
        return m.group(1) if m else s.split()[0]

    df["symbol"] = df["symbol_raw"].apply(_clean_ticker)
    df["side"] = df["side"].str.lower().str.strip()
    df = df[df["side"].isin(["buy", "sell"])]

    if df.empty:
        return (
            pd.DataFrame(),
            "No filled orders found in Alpaca UI export",
            ["No filled orders after filtering"],
        )

    df = df.sort_values("filled_at").reset_index(drop=True)

    lots = defaultdict(list)
    completed = []
    orphan_sells = []

    for _, row in df.iterrows():
        sym = row["symbol"]
        side = row["side"]
        qty = float(row["filled_qty"])
        price = float(row["avg_fill_price"])
        ts = row["filled_at"]

        if side == "buy":
            if lots[sym] and lots[sym][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "short":
                    lot = lots[sym][0]
                    close_qty = min(remaining, lot["qty"])
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[sym].pop(0)
                        continue
                    completed.append(
                        {
                            "pnl": round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
                            "date": ts,
                            "entry_date": lot["entry_date"],
                            "ticker": sym,
                            "buy_price": price,
                            "sell_price": lot["price"],
                            "qty": close_qty,
                            "hold_days": max(0, (ts - lot["entry_date"]).days),
                            "side": "short",
                        }
                    )
                    lot["qty"] -= close_qty
                    remaining -= close_qty
                    if lot["qty"] == 0:
                        lots[sym].pop(0)
                if remaining > 0:
                    lots[sym].append({"price": price, "qty": remaining, "entry_date": ts, "side": "long"})
            else:
                lots[sym].append({"price": price, "qty": qty, "entry_date": ts, "side": "long"})

        elif side == "sell":
            if not lots[sym]:
                orphan_sells.append(
                    f"{sym} sell@{price:.2f} on {ts.date()} "
                    f"(no matching buy - position predates history window)"
                )
                continue

            remaining = qty
            while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "long":
                lot = lots[sym][0]
                close_qty = min(remaining, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[sym].pop(0)
                    continue
                completed.append(
                    {
                        "pnl": round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
                        "date": ts,
                        "entry_date": lot["entry_date"],
                        "ticker": sym,
                        "buy_price": lot["price"],
                        "sell_price": price,
                        "qty": close_qty,
                        "hold_days": max(0, (ts - lot["entry_date"]).days),
                        "side": "long",
                    }
                )
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] == 0:
                    lots[sym].pop(0)

    if orphan_sells:
        audit_flags.append(
            f"⚠️ {len(orphan_sells)} sell(s) had no matching buy (position predates window) excluded"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f"   └ {s}")

    open_positions = {s: sum(lot["qty"] for lot in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(f"{s}({int(q)})" for s, q in list(open_positions.items())[:6])
        audit_flags.append(
            f" {len(open_positions)} open position(s) not included (no closing trade yet): {sym_list}"
        )

    if not completed:
        return (
            pd.DataFrame(),
            "No completed round-trip trades found in Alpaca UI export.",
            audit_flags
            + ["All sells may predate the visible order history, or all positions are still open."],
        )

    result_df = pd.DataFrame(completed).sort_values("date").reset_index(drop=True)

    total_filled = len(df)
    paired_trades = len(completed)
    note = (
        f"📊 Parsed {paired_trades} completed trade(s) from Alpaca UI export "
        f"({total_filled} filled order-sides ?+ {paired_trades} round-trips, "
        f"{len(orphan_sells)} orphan sell(s) excluded)"
    )
    return result_df, note, audit_flags


# =========================================================
# ALPACA ORDER PAIRING — position-state tracker
# =========================================================


def pair_alpaca_orders(orders: list) -> tuple[pd.DataFrame, list]:
    """
    Converts raw Alpaca orders into a trades DataFrame.
    Accepts both REST API JSON and UI export dict shapes.
    """
    audit_flags = []

    def _norm_order(o: dict) -> dict | None:
        status = str(o.get("status", "")).lower().strip()
        if status != "filled":
            return None

        qty_raw = o.get("filled_qty") or o.get("qty") or 0
        try:
            qty = float(qty_raw)
        except (ValueError, TypeError):
            return None
        if qty <= 0:
            return None

        price_raw = o.get("filled_avg_price") or o.get("avg_fill_price") or o.get("price") or 0
        try:
            price_str = str(price_raw).replace(",", ".").strip()
            if price_str in ("-", "", "nan", "None"):
                return None
            price = float(price_str)
        except (ValueError, TypeError):
            return None
        if price <= 0:
            return None

        side = str(o.get("side", "")).lower().strip()
        if side not in ("buy", "sell"):
            return None

        ts_raw = o.get("filled_at") or o.get("updated_at") or ""
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw.tz_localize(None) if ts_raw.tzinfo else ts_raw
        else:
            ts_str = str(ts_raw).strip()
            if ts_str in ("-", "", "nan", "None"):
                return None
            try:
                ts_parsed = pd.to_datetime(ts_str, utc=True)
                ts = ts_parsed.tz_localize(None)
            except Exception:
                try:
                    ts = pd.to_datetime(ts_str, errors="coerce")
                    if pd.isna(ts):
                        return None
                    if hasattr(ts, "tzinfo") and ts.tzinfo:
                        ts = ts.tz_convert("UTC").tz_localize(None)
                except Exception:
                    return None

        symbol_raw = str(o.get("symbol", "UNKNOWN")).strip()
        m = _re.match(r"([A-Z]{1,6})", symbol_raw.upper())
        symbol = m.group(1) if m else symbol_raw.split()[0]

        return {"symbol": symbol, "side": side, "qty": qty, "price": price, "filled_at": ts}

    filled = []
    for o in orders:
        normed = _norm_order(o)
        if normed:
            filled.append(normed)

    if not filled:
        return pd.DataFrame(), ["No filled orders found"]

    df_orders = pd.DataFrame(filled).sort_values("filled_at").reset_index(drop=True)

    lots = defaultdict(list)
    completed = []
    orphan_sells = []

    for _, row in df_orders.iterrows():
        sym = row["symbol"]
        side = row["side"]
        qty = row["qty"]
        price = row["price"]
        ts = row["filled_at"]

        if side == "buy":
            if lots[sym] and lots[sym][0]["side"] == "short":
                remaining = qty
                while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "short":
                    lot = lots[sym][0]
                    close_qty = min(remaining, lot["qty"])
                    try:
                        pnl_pct = (lot["price"] - price) / lot["price"]
                    except ZeroDivisionError:
                        lots[sym].pop(0)
                        continue
                    completed.append(
                        {
                            "pnl": round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
                            "date": ts,
                            "entry_date": lot["entry_date"],
                            "ticker": sym,
                            "buy_price": price,
                            "sell_price": lot["price"],
                            "qty": close_qty,
                            "hold_days": max(0, (ts - lot["entry_date"]).days),
                            "side": "short",
                        }
                    )
                    lot["qty"] -= close_qty
                    remaining -= close_qty
                    if lot["qty"] == 0:
                        lots[sym].pop(0)
                if remaining > 0:
                    lots[sym].append({"price": price, "qty": remaining, "entry_date": ts, "side": "long"})
            else:
                lots[sym].append({"price": price, "qty": qty, "entry_date": ts, "side": "long"})

        elif side == "sell":
            if not lots[sym]:
                # No open long position — this SELL opens a short position
                lots[sym].append(
                    {
                        "price": price,
                        "qty": qty,
                        "entry_date": ts,
                        "side": "short",
                    }
                )
                # FIX: message now accurately describes behaviour (short entry, not excluded)
                orphan_sells.append(
                    f"{sym} sell@{price:.2f} on {ts.date()} "
                    f"(short entry - buy predates history, using sell price as cost basis)"
                )
                continue

            if lots[sym] and lots[sym][0]["side"] == "short":
                lots[sym].append(
                    {
                        "price": price,
                        "qty": qty,
                        "entry_date": ts,
                        "side": "short",
                    }
                )
                continue

            remaining = qty
            while remaining > 0 and lots[sym] and lots[sym][0]["side"] == "long":
                lot = lots[sym][0]
                close_qty = min(remaining, lot["qty"])
                try:
                    pnl_pct = (price - lot["price"]) / lot["price"]
                except ZeroDivisionError:
                    lots[sym].pop(0)
                    continue
                completed.append(
                    {
                        "pnl": round(np.clip(float(pnl_pct), -0.99, 0.99), 6),
                        "date": ts,
                        "entry_date": lot["entry_date"],
                        "ticker": sym,
                        "buy_price": lot["price"],
                        "sell_price": price,
                        "qty": close_qty,
                        "hold_days": max(0, (ts - lot["entry_date"]).days),
                        "side": "long",
                    }
                )
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] == 0:
                    lots[sym].pop(0)

    if orphan_sells:
        # FIX: message now accurately describes behaviour (short entries, not excluded)
        audit_flags.append(
            f"{len(orphan_sells)} sell(s) treated as short entries "
            f"(buy predates history window - sell price used as cost basis)"
        )
        for s in orphan_sells[:3]:
            audit_flags.append(f'   "{s}"')

    open_positions = {s: sum(lot["qty"] for lot in ls) for s, ls in lots.items() if ls}
    if open_positions:
        sym_list = ", ".join(f"{s}({int(q)})" for s, q in list(open_positions.items())[:6])
        audit_flags.append(
            f" {len(open_positions)} open position(s) not included (no closing trade yet): {sym_list}"
        )

    if not completed:
        return pd.DataFrame(), audit_flags + [
            "No completed round-trip trades found. "
            "All positions may still be open, or all sells predate the history window."
        ]

    df = pd.DataFrame(completed).sort_values("date").reset_index(drop=True)

    df, regime_note = assign_regimes(df)
    audit_flags.append(regime_note)

    return df, audit_flags


# =========================================================
# OPTIONS CSV PARSER
# =========================================================


def _is_options_csv(df: pd.DataFrame) -> bool:
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    options_signals = [
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
    ]
    return sum(1 for s in options_signals if any(s in c for c in cols)) >= 2


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

        # O(n log n) sweep-line: sort all events, track active open count
        # Two positions overlap iff a new open starts before any prior close
        if len(opens) > 500:
            return False  # guard: too many rows to check, assume non-overlapping
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


def _parse_options_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    import io as _io

    sample = raw[:2048]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim = max(counts, key=counts.get)

    df = pd.read_csv(_io.StringIO(raw), sep=delim)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    if _has_overlapping_positions(df):
        raise ValueError(
            "Multi-leg or concurrent options positions detected. "
            "QuantProof supports single-leg directional options trades only "
            "(calls or puts bought and fully closed before the next trade). "
            "For spreads, iron condors, or straddles: calculate your daily "
            "P&L as a percentage of capital deployed and upload as a generic CSV."
        )

    pnl_col = next(
        (c for c in df.columns if any(x in c for x in ["pnl", "profit", "return", "gain", "pl", "net"])), None
    )
    if pnl_col is None:
        raise ValueError(
            "Could not find a P&L column in your options CSV. "
            "Expected a column named pnl, profit, return, gain, or pl."
        )

    pnl_vals = pd.to_numeric(df[pnl_col].astype(str).str.replace("%", "", regex=False), errors="coerce")

    premium_col = next(
        (c for c in df.columns if any(x in c for x in ["premium", "entry_price", "open_price", "cost"])), None
    )
    if premium_col is not None:
        premium_vals = pd.to_numeric(
            df[premium_col].astype(str).str.replace("%", "", regex=False), errors="coerce"
        )
        valid_mask = (premium_vals.notna()) & (premium_vals.abs() > 1e-9) & (pnl_vals.notna())
        returns = (pnl_vals[valid_mask] / premium_vals[valid_mask]).clip(-0.99, 0.99)
        norm_method = f"P&L / {premium_col} (per contract return)"
    else:
        valid_mask = pnl_vals.notna()
        returns = pnl_vals[valid_mask]
        if returns.abs().mean() > 1.0:
            returns = returns / 100.0
        returns = returns.clip(-0.99, 0.99)
        norm_method = "pnl column treated as fractional return"

    result = pd.DataFrame({"pnl": returns.values})

    date_col = next(
        (c for c in df.columns if any(x in c for x in ["date", "close_date", "exit_date", "time", "dt"])),
        None,
    )
    if date_col:
        result["date"] = pd.to_datetime(df[date_col].values[: len(returns)], errors="coerce")

    ticker_col = next(
        (c for c in df.columns if any(x in c for x in ["ticker", "symbol", "underlying", "instrument"])), None
    )
    if ticker_col:
        result["ticker"] = df[ticker_col].values[: len(returns)]

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
    cols = {c.lower().strip().replace(" ", "_").replace("/", "_") for c in df.columns}
    deals_signals = {"deal", "direction", "commission", "swap", "balance"}
    deals_hit = len(deals_signals & cols)
    pos_signals = {"position", "s_l", "t_p", "commission", "swap"}
    time_cols = [c for c in cols if "time" in c]
    pos_hit = len(pos_signals & cols) + (1 if len(time_cols) >= 2 else 0)
    return deals_hit >= 3 or pos_hit >= 3


def _parse_mt5_csv(raw: str) -> tuple[pd.DataFrame, str, list]:
    import io as _io

    sample = raw[:2048]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim = max(counts, key=counts.get)

    if delim == ";":
        raw = raw.replace(",", ".")

    try:
        df = pd.read_csv(_io.StringIO(raw), sep=delim, dtype=str, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(_io.StringIO(raw), sep=delim, dtype=str, error_bad_lines=False)

    if df.empty:
        raise ValueError("MT5 CSV appears empty after parsing.")

    df.columns = [c.lower().strip().replace(" ", "_").replace("/", "_") for c in df.columns]

    has_direction = "direction" in df.columns
    has_position = "position" in df.columns

    if not has_direction and not has_position:
        if "deal" in df.columns:
            has_direction = True
        else:
            raise ValueError(
                "Could not identify MT5 report format. "
                "Please export from MT5 Terminal → Account History → "
                "right-click → Save as Detailed Report."
            )

    def _num(series):
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False).str.replace("[^\\d.\\-]", "", regex=True),
            errors="coerce",
        )

    for col in ["profit", "volume", "price", "commission", "swap", "balance", "price_2"]:
        if col in df.columns:
            df[col] = _num(df[col])

    price_cols = [c for c in df.columns if "price" in c]
    for pc in price_cols:
        df[pc] = _num(df[pc])

    time_cols = [c for c in df.columns if "time" in c]
    for tc in time_cols:
        df[tc] = pd.to_datetime(df[tc], errors="coerce")

    open_time_col = time_cols[0] if time_cols else None
    close_time_col = time_cols[1] if len(time_cols) >= 2 else open_time_col

    symbol_col = next((c for c in df.columns if c in ["symbol", "instrument", "pair", "asset"]), None)

    if "type" in df.columns:
        type_vals = df["type"].astype(str).str.lower().str.strip()
        trade_mask = type_vals.isin(
            ["buy", "sell", "buy limit", "sell limit", "buy stop", "sell stop", "0", "1"]
        )
        df = df[trade_mask].copy()

    if df.empty:
        raise ValueError(
            "No trade rows found in MT5 export after filtering. "
            "Make sure you exported Account History (not just open positions) "
            "and the report includes completed trades."
        )

    audit_flags = []
    trades = []
    orphan_count = 0

    if has_direction:
        df["direction"] = df["direction"].astype(str).str.lower().str.strip()
        df = df.sort_values(open_time_col) if open_time_col else df

        lots = defaultdict(list)

        for _, row in df.iterrows():
            direction = str(row.get("direction", "")).lower().strip()
            symbol = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            volume = float(row.get("volume", 0) or 0)
            price = float(row.get("price", 0) or 0)
            profit = float(row.get("profit", 0) or 0)
            commission = float(row.get("commission", 0) or 0)
            swap = float(row.get("swap", 0) or 0)
            balance = float(row.get("balance", 0) or 0)
            date = row[open_time_col] if open_time_col else pd.NaT
            trade_type = str(row.get("type", "")).lower().strip()

            if direction == "in/out":
                if lots[symbol]:
                    lot = lots[symbol].pop(0)
                    net_profit = profit + commission + swap
                    pnl_pct = _normalise_pnl(net_profit, lot["price"], lot["volume"], balance)
                    if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                        trades.append({"pnl": round(pnl_pct, 6), "date": date, "ticker": symbol})
                side = "sell" if trade_type in ["sell", "1"] else "buy"
                lots[symbol].append({"price": price, "volume": volume, "date": date, "side": side})
                continue

            if direction == "in":
                lots[symbol].append(
                    {
                        "price": price,
                        "volume": volume,
                        "date": date,
                        "side": trade_type,
                        "commission": commission,
                    }
                )
            elif direction == "out":
                if not lots[symbol]:
                    orphan_count += 1
                    continue
                lot = lots[symbol].pop(0)
                net_profit = profit + commission + swap + lot.get("commission", 0)
                pnl_pct = _normalise_pnl(net_profit, lot["price"], lot["volume"], balance)
                if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                    trades.append({"pnl": round(pnl_pct, 6), "date": date, "ticker": symbol})

        open_pos = {s: len(ls) for s, ls in lots.items() if ls}
        if open_pos:
            count = sum(open_pos.values())
            audit_flags.append(f" {count} MT5 position(s) still open (no closing deal) excluded")

    else:
        price_open_col = next((c for c in df.columns if "price" in c), None)
        next((c for c in df.columns if "price" in c and c != price_open_col), None)

        for _, row in df.iterrows():
            profit = float(row.get("profit", 0) or 0)
            commission = float(row.get("commission", 0) or 0)
            swap = float(row.get("swap", 0) or 0)
            volume = float(row.get("volume", 1) or 1)
            price_open = float(row[price_open_col] or 0) if price_open_col else 0
            balance = 0.0
            symbol = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            date = row[close_time_col] if close_time_col else pd.NaT

            net_profit = profit + commission + swap
            pnl_pct = _normalise_pnl(net_profit, price_open, volume, balance)
            if pnl_pct is not None and abs(pnl_pct) <= 0.99:
                trades.append({"pnl": round(pnl_pct, 6), "date": date, "ticker": symbol})

    if not trades:
        raise ValueError(
            "No completed trades could be extracted from the MT5 export. "
            "Make sure your report includes closed trades and has Profit column."
        )

    result = pd.DataFrame(trades)
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    if orphan_count:
        audit_flags.append(
            f"⚠️ {orphan_count} MT5 closing deal(s) had no matching open "
            f" excluded (position predates history window)"
        )

    audit_flags.append(
        "MT5 DATA: Returns normalised as net profit / (entry_price * volume). "
        "Crash simulations assume linear payoffs - leverage and margin effects not modelled. "
        "For accurate capacity analysis, ensure Volume column reflects lot sizes."
    )
    # FIX: warn forex/CFD users that contract size is not applied
    _forex_scale_warn = False
    if symbol_col and len(trades) > 0:
        sample_symbols = [t.get("ticker", "") for t in trades[:10]]
        # Forex pairs are typically 6-8 chars like EURUSD, GBPJPY, XAUUSD
        looks_forex = any(
            len(str(s)) in (6, 7, 8) and str(s).isalpha() for s in sample_symbols if s and s != "UNKNOWN"
        )
        if looks_forex:
            audit_flags.append(
                "⚠️ MT5 FOREX WARNING: Contract size (100,000 units/lot) is NOT applied. "
                "Returns are normalised as profit/(price*lots) not profit/(price*lots*100000). "
                "All forex PnL values are approximately 1000x overstated and will clip to ?0.99. "
                "To fix: pre-calculate per-trade % return and upload as a generic CSV instead."
            )
            # Embed sentinel so smart_parse_csv can set scale_warning for frontend banner
            _forex_scale_warn = True

    report = "deals" if has_direction else "positions"
    note = (
        f"📊 MT5: {len(trades)} completed trade(s) parsed from {report} report "
        f"({orphan_count} orphan(s) excluded)"
    )
    if _forex_scale_warn:
        note = "MT5_FOREX_SCALE_WARN||" + note
    return result, note, audit_flags


def _normalise_pnl(net_profit: float, entry_price: float, volume: float, balance: float) -> float | None:
    """
    Normalise MT5 net profit to a decimal return.
    Returns None when entry_price*volume is zero (corrupt/missing data) so the
    trade is silently skipped rather than inventing a meaningless number via
    balance-ratio or /1000 fallbacks — those two paths produced values with no
    consistent financial meaning and have been removed.
    """
    if abs(entry_price * volume) > 1e-6:
        pnl = net_profit / (entry_price * volume)
        return float(np.clip(pnl, -0.99, 0.99))
    # entry_price or volume is zero — data is corrupt, skip this trade
    return None


def smart_parse_excel(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parses Excel files (.xlsx, .xls) into a trades DataFrame for QuantProofValidator.
    Returns (df, regime_note).
    """
    try:
        import openpyxl  # noqa: F401 - fail fast with a clear message if missing

        df = pd.read_excel(io.BytesIO(contents))
        regime_note = ""

        if "date" in df.columns and df["date"].notna().any():
            df, regime_note = assign_regimes(df)

        return df, regime_note

    except ImportError:
        raise HTTPException(
            400, "Excel support requires openpyxl package. Please install with: pip install openpyxl"
        )
    except Exception as e:
        raise HTTPException(400, f"Could not parse Excel file: {str(e)}")


def smart_parse_csv(contents: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parses CSV/TSV/Excel into a trades DataFrame for QuantProofValidator.
    Returns (df, regime_note).
    """
    # NEW: Detect Excel files by magic bytes
    if (
        contents.startswith(b"PK\x03\x04")
        or contents.startswith(b"PK\x05\x06")
        or contents.startswith(b"PK\x07\x08")
    ):
        return smart_parse_excel(contents)

    raw = contents.decode("utf-8", errors="ignore")
    regime_note = ""

    sample = raw[:4096]
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    delim = max(counts, key=counts.get)

    try:
        _probe = pd.read_csv(io.StringIO(raw), sep=delim, nrows=5, dtype=str)
    except Exception:
        _probe = pd.DataFrame()

    # —? 0. Options CSV ——————————————————————————————————————————————————?
    if not _probe.empty and _is_options_csv(_probe):
        try:
            _df, _note, _flags = _parse_options_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _bm_type_for_regime = get_benchmark_type(_df, None)
                    _df, _regime_note = assign_regimes(_df, _bm_type_for_regime)
                    _note += " | " + _regime_note if _regime_note else ""
                if _flags:
                    _note += " | OPTIONS_FLAGS:" + "||".join(_flags)
                return _df, _note
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as _e:
            # Parser was positively identified — crash means bad data, not wrong parser
            raise HTTPException(400, f"Options CSV detected but could not be parsed: {_e}")

    # —? 0.5. MT5 Trade History ——————————————————————————————————————————?
    if not _probe.empty and _is_mt5_csv(_probe):
        try:
            _df, _note, _flags = _parse_mt5_csv(raw)
            if len(_df) > 0:
                # Detect MT5 forex scale warning sentinel — promote to scale_warning
                _mt5_forex_scale = False
                if _note.startswith("MT5_FOREX_SCALE_WARN||"):
                    _note = _note[len("MT5_FOREX_SCALE_WARN||") :]
                    _mt5_forex_scale = True
                if "date" in _df.columns and _df["date"].notna().any():
                    _bm_type_for_regime = get_benchmark_type(_df, None)
                    _df, _regime_note = assign_regimes(_df, _bm_type_for_regime)
                    _note += " | " + _regime_note if _regime_note else ""
                if _flags:
                    _note += " | MT5_FLAGS:" + "||".join(_flags)
                if _mt5_forex_scale:
                    _note += " | MT5_SCALE_WARNING"
                return _df, _note
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as _e:
            raise HTTPException(400, f"MT5 CSV detected but could not be parsed: {_e}")

    # —? 1. Kite TradeBook ————————————————————————————————————————————————?
    if not _probe.empty and _is_kite_csv(_probe):
        try:
            _df, _note, _flags = _parse_kite_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df, "nifty")
                    _note += " | " + _regime_note if _regime_note else ""
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                if "ticker" in _df.columns:
                    _opt_mask = _df["ticker"].astype(str).str.contains(r"(CE|PE)$", regex=True, na=False)
                    if _opt_mask.any():
                        _opt_count = int(_opt_mask.sum())
                        _note += (
                            f" | KITE OPTIONS: {_opt_count} options trades detected "
                            f"(CE/PE). P&L used as-is from Kite — premium normalisation "
                            f"not applied. Crash sims assume linear payoffs."
                        )
                return _df, _note
        except Exception as _e:
            raise HTTPException(400, f"Kite TradeBook CSV detected but could not be parsed: {_e}")

    # —? 2. Kite Orders CSV ———————————————————————————————————————————————?
    if not _probe.empty and _is_kite_orders_csv(_probe):
        try:
            _df, _note, _flags = _parse_kite_orders_csv(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df, "nifty")
                    _note += " | " + _regime_note if _regime_note else ""
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                return _df, _note
        except Exception as _e:
            raise HTTPException(400, f"Kite Orders CSV detected but could not be parsed: {_e}")

    # —? 3. Alpaca UI export ——————————————————————————————————————————————?
    if not _probe.empty and _is_alpaca_ui_export(_probe):
        try:
            _df, _note, _flags = _parse_alpaca_ui_export(raw)
            if len(_df) > 0:
                if "date" in _df.columns and _df["date"].notna().any():
                    _df, _regime_note = assign_regimes(_df)
                    _note += " | " + _regime_note if _regime_note else ""
                if _flags:
                    _note += " | " + " | ".join(_flags[:2])
                return _df, _note
        except Exception as _e:
            raise HTTPException(400, f"Alpaca UI CSV detected but could not be parsed: {_e}")

    # —? 4. Generic PnL CSV ———————————————————————————————————————————————?
    try:
        df = pd.read_csv(io.StringIO(raw), sep=delim)
        pnl_names = ["pnl", "profit", "return", "gain", "pl", "p&l"]
        for col in df.columns:
            if any(p in col.lower() for p in pnl_names):
                raw_vals = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
                vals = pd.to_numeric(raw_vals, errors="coerce").dropna()
                if len(vals) >= 5:
                    _was_divided = vals.abs().mean() > 1.0
                    if _was_divided:
                        vals = vals / 100.0
                    # Detection threshold stays at 0.50 — dollar PnL detector
                    _clipped_pct = (vals.abs() > 0.50).mean()
                    # FIX: clip to ?0.99 to match validator guard (was ?0.50)
                    vals = vals.clip(-0.99, 0.99)
                    _dollar_pnl_warning = _clipped_pct > 0.50
                    # HARD REJECT: only fires when values required /100 normalization
                    # AND still have >80% clipping — meaning they were dollar amounts
                    # (e.g. $450) not valid high-return decimals (e.g. 0.95 = 95% gain).
                    # Without _was_divided gate: a valid 70-95% return strategy
                    # (mean=0.85) would be incorrectly rejected as dollar amounts.
                    if _was_divided and _clipped_pct > 0.80:
                        raise ValueError(
                            f"PnL values appear to be dollar amounts or raw prices, not decimal returns. "
                            f"{int(_clipped_pct * 100)}% of values exceed ?50% per trade after normalisation. "
                            "Upload decimal returns (e.g. 0.015 = 1.5%, not 1.5 or $15). "
                            "Check your PnL column and re-upload."
                        )
                    result = pd.DataFrame({"pnl": vals.values})

                    date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "dt"])]
                    if date_cols:
                        result["date"] = pd.to_datetime(df[date_cols[0]].values[: len(vals)], errors="coerce")

                    vol_cols = [c for c in df.columns if "volume" in c.lower() or "vol" == c.lower()]
                    if vol_cols:
                        result["volume"] = pd.to_numeric(df[vol_cols[0]], errors="coerce").values[: len(vals)]

                    reg_cols = [c for c in df.columns if "regime" in c.lower()]
                    if reg_cols:
                        result["regime"] = df[reg_cols[0]].values[: len(vals)]
                    elif "date" in result.columns and result["date"].notna().any():
                        _bm_type_for_regime = get_benchmark_type(result, None)
                        result, regime_note = assign_regimes(result, _bm_type_for_regime)

                    if _dollar_pnl_warning:
                        regime_note = (
                            ("⚠️ " + regime_note + " | " if regime_note else "⚠️ ")
                            + "PnL values appear to be dollar amounts, not decimal returns "
                            "? clipped to 0.99. Upload a column with decimal returns (e.g. 0.015 = 1.5%)."
                        )

                    return result, regime_note
    except ValueError:
        raise
    except Exception:
        pass

    # —? 5. Flexible fallback —————————————————————————————————————————————?
    try:
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        max_cols = max(len(ln.split(delim)) for ln in lines[1:] if ln)
        df = pd.read_csv(
            io.StringIO(raw),
            sep=delim,
            names=[f"col_{i}" for i in range(max_cols)],
            skiprows=1,
            on_bad_lines="skip",
            engine="python",
        )
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and (vals > 0).any() and (vals < 0).any():
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                # FIX: clip to ?0.99 to match validator guard (was ?0.50)
                vals = vals.clip(-0.99, 0.99)
                return pd.DataFrame({"pnl": vals.values}), ""
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                # FIX: clip to ?0.99 to match validator guard (was ?0.50)
                vals = vals.clip(-0.99, 0.99)
                return pd.DataFrame({"pnl": vals.values}), ""
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError(
        "No valid PnL column found. Make sure your CSV has a 'pnl' or 'profit' column, "
        "or upload a Kite TradeBook, Kite Orders CSV, or Alpaca export."
    )


def _dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows (same date + ticker + pnl) that arise when
    a user uploads the same file twice or a broker export contains duplicate entries.
    Only deduplicates on columns that are present — pnl is always checked.
    """
    if df.empty:
        return df
    key_cols = [c for c in ["date", "ticker", "pnl", "side"] if c in df.columns]
    if not key_cols:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[dedup] Removed {removed} duplicate trade row(s)")
    return df


# =========================================================
# ROUTES
# =========================================================
