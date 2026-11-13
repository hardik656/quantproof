"""Interactive Brokers Flex parsing and Flex Web Service download.

Activity Flex ``<Trade>`` rows are priced from ``fifoPnlRealized`` over the
opening cost (``price × quantity × multiplier``, or IBKR's ``cost`` when the
query includes it). Trade Confirmation ``<TradeConfirm>`` rows have no
realized-P&L field, so opens and closes are matched FIFO. Position flips
(``C;O`` and ``O;C``) keep the closed portion and carry the excess as a new
open leg. Dollar P&L is converted with each leg's ``fxRateToBase``.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

# =========================================================
# IBKR CONFIGURATION
# =========================================================

IBKR_CONFIG = {
    "opening_detection": {
        "long_cash_outflow_ratio": 0.8,
        "short_cash_inflow_ratio": 0.2,
        "description": "Thresholds for detecting opening positions",
    },
    "pnl_calculation": {
        "clip_range": 0.99,
        "min_pnl_threshold": 0.001,
        "description": "P&L calculation parameters",
    },
    "validation": {"min_trades": 10, "description": "Minimum trades required for validation"},
    "last_updated": "2026-11-13",
    "version": "1.1",
}

# Documented Flex Web Service endpoint. ``p`` is the lookback in days (max 365).
FLEX_SEND_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"

# GetStatement answers these while the report is still being built.
PENDING_CODES = {"1001", "1004", "1005", "1006", "1007", "1008", "1009", "1019", "1021"}
# Retrying these makes the next call fail faster.
DO_NOT_RETRY_CODES = {"1018", "1025"}

# Finest level wins so Execution + Symbol Summary is not counted twice.
_LEVEL_RANK = {
    "EXECUTION": 0,
    "ORDER": 1,
    "CLOSED_LOT": 2,
    "SYMBOL_SUMMARY": 3,
    "ASSET_SUMMARY": 4,
    "WASH": 5,
}

_TZ_SUFFIX = re.compile(
    r"\s+(?:UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|HKT|JST|CET|CEST|BST)\b",
    re.IGNORECASE,
)
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")

ACTIVITY_STATEMENT_MESSAGE = (
    "This file is an Interactive Brokers Activity Statement, which QuantProof does not import. "
    "The statement layout is not a trade list: dollar Realized P/L would be read as returns and "
    "YYYYMMDD dates can collapse to 1970. Create an Activity Flex Query (Trades section, "
    "Level of Detail = Executions, including FifoPnlRealized) or a Trade Confirmation Flex Query, "
    "then download it as XML or CSV, or connect with a Flex token."
)
TWS_MESSAGE = (
    "This file looks like an Interactive Brokers TWS trade log, which QuantProof does not import. "
    "TWS logs store dollar P&L and YYYYMMDD timestamps, which would be misread as returns and 1970 dates. "
    "Export an Activity Flex Query or a Trade Confirmation Flex Query as XML or CSV, "
    "or connect with a Flex token."
)
UNSUPPORTED_IBKR_MESSAGE = (
    "This file looks like an Interactive Brokers export QuantProof cannot import. "
    "Upload an Activity Flex Query or a Trade Confirmation Flex Query as XML or CSV "
    "with Level of Detail set to Executions. Activity Statements and TWS trade logs are not supported."
)
NO_CLOSED_TRADES_MESSAGE = (
    "No closed IBKR trades found. Opening fills have no realized P&L and are excluded. "
    "Activity Flex queries need Open/CloseIndicator and FifoPnlRealized (break-even closes are kept). "
    "Trade Confirmation queries need Code O on opens and C on closes; position flips use C;O or O;C. "
    "Set Level of Detail to Executions only."
)


class IBKRServiceError(Exception):
    """Flex Web Service failure. ``status_code`` is the HTTP status to surface."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def redact_secret(text: str, secret: str) -> str:
    if not secret or not text:
        return text
    return text.replace(secret, "[REDACTED]")


def period_days(days_back: int | None) -> int:
    """Flex ``p`` lookback. IBKR accepts 1–365 days; missing or invalid uses 365."""
    try:
        days = int(365 if days_back is None else days_back)
    except (TypeError, ValueError):
        days = 365
    if days < 1:
        days = 365
    return min(days, 365)


def _clip_range() -> float:
    return float(IBKR_CONFIG["pnl_calculation"]["clip_range"])


def _clip_return(frac: float | None) -> float | None:
    if frac is None:
        return None
    return float(np.clip(float(frac), -_clip_range(), _clip_range()))


def _local_tag(tag: str) -> str:
    if not tag:
        return ""
    return tag.split("}")[-1]


def _tokens(value: str) -> list[str]:
    return [part for part in re.split(r"[;,\s]+", (value or "").upper()) if part]


def indicator_kind(open_close: str) -> str:
    """Classify an Open/CloseIndicator or Trade Confirmation code.

    Activity Flex uses O (open), C (close), and C;O or O;C for a position flip.
    Trade Confirmation uses the same O/C letters (P is a partial fill, not a side).
    Exercise, expiry, and assignment realize P&L and count as closes.
    """
    tokens = _tokens(open_close)
    if not tokens:
        return "unknown"
    if "CA" in tokens:
        return "cancel"
    has_open = "O" in tokens
    has_close = "C" in tokens or any(token in {"EX", "EP", "A", "AEX", "MEX"} for token in tokens)
    if has_open and has_close:
        return "flip"
    if has_close:
        return "close"
    if has_open:
        return "open"
    return "unknown"


def calculate_trade_side(buy_sell, qty, open_close, net_cash, price):
    """Side of the *closed* position. A sell closes a long; a buy closes a short.

    Flips realize the side that was closed, which is the opposite of the new leg.
    ``net_cash`` and ``price`` are unused; they stay in the signature so older
    callers keep working.
    """
    del net_cash, price
    kind = indicator_kind(open_close)
    try:
        quantity = float(qty)
    except (TypeError, ValueError):
        quantity = 0.0
    if kind in {"close", "flip"}:
        return "long" if quantity < 0 else "short"
    side = (buy_sell or "").upper()
    if "SELL" in side:
        return "short"
    if "BUY" in side:
        return "long"
    return "long" if quantity > 0 else "short"


def is_opening_position(open_close, qty, net_cash, price, fifo_pnl=None):
    """True when the row should not produce a return.

    Pure opens are skipped. Closes and flips (C;O, O;C) are kept, including
    break-even closes whose fifoPnlRealized is 0. ``net_cash / notional`` is not
    used: net cash is the cash leg, about ± the trade value, for both opens and
    closes, so that ratio cannot tell them apart.
    """
    del qty, net_cash, price
    kind = indicator_kind(open_close)
    if kind == "open":
        return True
    if kind in {"close", "flip"}:
        return False
    if kind == "cancel":
        return True
    if fifo_pnl not in (None, ""):
        try:
            if float(fifo_pnl) != 0.0:
                return False
        except (TypeError, ValueError):
            pass
    return True


def calculate_pnl_fraction(net_cash, price, qty, fifo_pnl=None, multiplier=1.0, cost=None, fx_rate=1.0):
    """Decimal return for one closing fill, or None when it cannot be priced.

    ``fifoPnlRealized`` and the cost basis are in trade currency. Both are
    multiplied by ``fx_rate`` (fxRateToBase). When the same rate applies to the
    P&L and the basis it cancels; to apply a different rate on the opening
    basis, pre-convert both numbers and pass ``fx_rate=1``.

    A fifo value of 0 is a break-even close and returns 0. The previous code
    treated 0 as rounding noise and dropped the trade.

    The previous net-cash fallback divided netCash by price × quantity. netCash
    is proceeds plus commission (about ± notional), not P&L, so that ratio is
    about ±1 and then gets clipped to ±0.99. The corrected fallback is
    ``(netCash - cost) / |cost|`` and is used only when IBKR sent the opening
    basis. Without that basis the function refuses to invent a return.
    """
    fx = fx_rate if fx_rate and fx_rate > 0 else 1.0
    mult = multiplier if multiplier and multiplier > 0 else 1.0

    if fifo_pnl is not None:
        try:
            fifo = float(fifo_pnl)
        except (TypeError, ValueError):
            fifo = None
        if fifo is not None:
            # Exact zero is a real break-even close, not a row to drop.
            if fifo == 0.0:
                return 0.0
            if cost is not None and abs(float(cost)) > 1e-12:
                basis = abs(float(cost))
            elif price and qty and float(price) > 0 and float(qty) != 0:
                basis = abs(float(price) * float(qty) * mult)
            else:
                basis = None
            if not basis:
                return None
            return _clip_return((fifo * fx) / (basis * fx))

    if cost is None or net_cash is None:
        return None
    try:
        cost_v = float(cost)
        cash_v = float(net_cash)
    except (TypeError, ValueError):
        return None
    if abs(cost_v) < 1e-12:
        return None
    frac = (cash_v - cost_v) / abs(cost_v)
    # A ratio this large is notional, not a return. Refuse it instead of clipping.
    if abs(frac) > 5:
        return None
    return _clip_return(frac)


def _to_float(value, default=None):
    if value is None:
        return default
    text = str(value).strip().replace(" ", "")
    if text == "":
        return default
    if _THOUSANDS.fullmatch(text):
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return default


def _parse_date_token(token: str) -> datetime | None:
    token = token.strip()
    for fmt, size in (("%Y%m%d", 8), ("%Y-%m-%d", 10), ("%Y/%m/%d", 10), ("%m/%d/%Y", 10)):
        try:
            parsed = datetime.strptime(token[:size], fmt)
        except ValueError:
            continue
        # An unformatted integer YYYYMMDD must not fall through to the epoch.
        if parsed.year < 1990:
            return None
        return parsed
    return None


def _parse_time_token(token: str) -> tuple[int, int, int]:
    token = token.strip()
    if not token:
        return 0, 0, 0
    for fmt, size in (("%H:%M:%S", 8), ("%H:%M", 5), ("%H%M%S", 6), ("%H%M", 4)):
        try:
            parsed = datetime.strptime(token[:size], fmt)
        except ValueError:
            continue
        return parsed.hour, parsed.minute, parsed.second
    return 0, 0, 0


def parse_ibkr_datetime(date_str: str, time_str: str = "") -> datetime | None:
    """Combine IBKR tradeDate (YYYYMMDD or YYYY-MM-DD) and tradeTime (HHMMSS).

    Also accepts dateTime values such as ``2024-01-15;15:30:45 EDT``. Parsing
    is format-driven so a bare YYYYMMDD number is never read as a Unix timestamp.
    """
    raw_date = _TZ_SUFFIX.sub("", (date_str or "").strip())
    raw_time = _TZ_SUFFIX.sub("", (time_str or "").strip())
    if ";" in raw_date:
        left, right = raw_date.split(";", 1)
        raw_date = left.strip()
        if not raw_time:
            raw_time = right.strip()
    if "," in raw_date:
        left, right = raw_date.split(",", 1)
        raw_date = left.strip()
        if not raw_time:
            raw_time = right.strip()
    parsed_date = _parse_date_token(raw_date)
    if parsed_date is None:
        return None
    hour, minute, second = _parse_time_token(raw_time)
    return datetime(parsed_date.year, parsed_date.month, parsed_date.day, hour, minute, second)


def _attrs(element) -> dict[str, str]:
    if isinstance(element, dict):
        raw = element
    else:
        raw = element.attrib
    return {str(key).strip(): "" if value is None else str(value).strip() for key, value in raw.items()}


def _lookup(attrs: dict[str, str], *names: str) -> str:
    if not attrs:
        return ""
    folded = {key.casefold(): value for key, value in attrs.items()}
    squashed = {re.sub(r"[^a-z0-9]", "", key.casefold()): value for key, value in attrs.items()}
    for name in names:
        direct = folded.get(name.casefold(), "")
        if direct != "":
            return direct
        squeezed = squashed.get(re.sub(r"[^a-z0-9]", "", name.casefold()), "")
        if squeezed != "":
            return squeezed
    return ""


def _norm_level(value: str) -> str:
    token = re.sub(r"[^A-Z]", "", (value or "").upper())
    aliases = {
        "EXECUTION": "EXECUTION",
        "EXECUTIONS": "EXECUTION",
        "ORDER": "ORDER",
        "ORDERS": "ORDER",
        "CLOSEDLOT": "CLOSED_LOT",
        "CLOSEDLOTS": "CLOSED_LOT",
        "SYMBOLSUMMARY": "SYMBOL_SUMMARY",
        "SUMMARY": "SYMBOL_SUMMARY",
        "ASSETSUMMARY": "ASSET_SUMMARY",
        "ASSETCLASS": "ASSET_SUMMARY",
        "WASHSALE": "WASH",
        "WASHSALES": "WASH",
    }
    return aliases.get(token, token)


def _signed_qty(qty: float | None, buy_sell: str) -> float:
    if qty is None or qty == 0:
        return 0.0
    side = (buy_sell or "").upper()
    if "SELL" in side:
        return -abs(qty)
    if "BUY" in side:
        return abs(qty)
    return float(qty)


def _normalize(element, tag: str, seq: int = 0) -> dict | None:
    attrs = _attrs(element)
    buy_sell = _lookup(attrs, "buySell", "Buy/Sell", "BuySell")
    indicator = _lookup(attrs, "openCloseIndicator", "Open/CloseIndicator", "Open/Close Indicator")
    code = _lookup(attrs, "code", "Code", "notes", "Notes/Codes", "Notes")
    if indicator:
        kind = indicator_kind(indicator)
    elif code:
        kind = indicator_kind(code)
    else:
        kind = "unknown"
    if "CA" in _tokens(code) or "(CA" in buy_sell.upper():
        kind = "cancel"
    transaction = _lookup(attrs, "transactionType", "TransactionType").upper()
    if "CANCEL" in transaction:
        kind = "cancel"

    date_time = _lookup(attrs, "dateTime", "DateTime", "Date/Time")
    trade_date = _lookup(attrs, "tradeDate", "TradeDate", "Date")
    trade_time = _lookup(attrs, "tradeTime", "TradeTime", "Time")
    if date_time and (";" in date_time or ":" in date_time):
        timestamp = parse_ibkr_datetime(date_time)
    else:
        timestamp = parse_ibkr_datetime(trade_date or date_time, trade_time)
    if timestamp is None:
        timestamp = parse_ibkr_datetime(trade_date, trade_time)

    multiplier = _to_float(_lookup(attrs, "multiplier", "Multiplier"), default=1.0) or 1.0
    if multiplier <= 0:
        multiplier = 1.0
    fx = _to_float(_lookup(attrs, "fxRateToBase", "FXRateToBase"), default=1.0) or 1.0
    if fx <= 0:
        fx = 1.0
    price = _to_float(
        _lookup(attrs, "tradePrice", "TradePrice", "price", "Price", "T. Price"),
        default=0.0,
    )
    if price is None:
        price = 0.0
    quantity = _to_float(_lookup(attrs, "quantity", "Quantity", "qty"), default=0.0)
    fifo_raw = _lookup(attrs, "fifoPnlRealized", "FifoPnlRealized")
    cost_raw = _lookup(attrs, "cost", "CostBasis", "Cost Basis", "Basis")
    cash_raw = _lookup(attrs, "netCash", "NetCash", "amount", "Amount")
    symbol = _lookup(attrs, "symbol", "Symbol", "underlyingSymbol") or ""
    conid = _lookup(attrs, "conid", "Conid")
    if not symbol and quantity in (None, 0):
        return None
    return {
        "seq": seq,
        "tag": tag,
        "kind": kind,
        "symbol": symbol or "UNKNOWN",
        "conid": conid,
        "asset": _lookup(attrs, "assetCategory", "AssetClass", "Asset Category"),
        "strike": _lookup(attrs, "strike", "Strike"),
        "expiry": _lookup(attrs, "expiry", "Expiry"),
        "putcall": _lookup(attrs, "putCall", "Put/Call"),
        "currency": (_lookup(attrs, "currency", "CurrencyPrimary", "Currency") or "").upper(),
        "account_id": _lookup(attrs, "accountId", "ClientAccountID", "AccountId"),
        "level": _norm_level(_lookup(attrs, "levelOfDetail", "LevelOfDetail", "Level Of Detail")),
        "ts": timestamp,
        "buy_sell": buy_sell,
        "signed_qty": _signed_qty(quantity, buy_sell),
        "price": abs(price),
        "multiplier": multiplier,
        "fx": fx,
        "fifo": _to_float(fifo_raw, default=None),
        "cost": _to_float(cost_raw, default=None),
        "net_cash": _to_float(cash_raw, default=None),
        "indicator": indicator or code,
    }


def _instrument_key(row: dict) -> tuple:
    if row.get("conid"):
        return ("id", row["conid"])
    return (
        "sym",
        row.get("symbol") or "",
        row.get("asset") or "",
        row.get("strike") or "",
        row.get("expiry") or "",
        row.get("putcall") or "",
        row.get("currency") or "",
    )


def _filter_detail(rows: list[dict]) -> tuple[list[dict], str | None]:
    present = {row.get("level") or "" for row in rows}
    present.discard("")
    if not present:
        return rows, None
    best = min(present, key=lambda level: _LEVEL_RANK.get(level, 9))
    kept = [row for row in rows if row.get("level") == best]
    dropped = len(rows) - len(kept)
    if dropped <= 0:
        return kept, None
    return kept, (
        f"Ignored {dropped} coarser Flex row(s) and kept {best} "
        "so executions and summaries are not both counted."
    )


def _split_fill(row: dict, position: float) -> tuple[float, float]:
    """Return (quantity closed, signed quantity opened) for one fill."""
    qty = row["signed_qty"]
    kind = row["kind"]
    if qty == 0 or kind == "cancel":
        return 0.0, 0.0
    opposite = position != 0 and (position > 0) != (qty > 0)
    realized = row["fifo"] not in (None, 0, 0.0)
    if kind == "open" and not opposite:
        return 0.0, qty
    if kind in {"close", "flip"} or realized or opposite:
        if position == 0:
            # History started after the open. A coded close/flip is still a close.
            # The new leg of a flip cannot be sized, so it is not tracked.
            if kind == "open":
                return 0.0, qty
            return abs(qty), 0.0
        close_mag = min(abs(qty), abs(position))
        open_mag = abs(qty) - close_mag
        open_signed = 0.0 if open_mag <= 1e-9 else (open_mag if qty > 0 else -open_mag)
        return close_mag, open_signed
    return 0.0, qty


def _consume_lots(lots: list[dict], close_mag: float, signed_qty: float) -> tuple[float, float, str, float]:
    """FIFO-match ``close_mag`` units. Returns local basis, base basis, side, open FX."""
    side = "long" if signed_qty < 0 else "short"
    if close_mag <= 1e-12 or not lots:
        return 0.0, 0.0, side, 1.0
    remaining = close_mag
    matched_local = 0.0
    matched_base = 0.0
    fx_weight = 0.0
    while remaining > 1e-9 and lots:
        lot = lots[0]
        take = min(remaining, abs(lot["signed_qty"]))
        matched_local += take * lot["unit_local"]
        matched_base += take * lot["unit_base"]
        fx_weight += take * lot["fx"]
        side = "long" if lot["signed_qty"] > 0 else "short"
        leftover = abs(lot["signed_qty"]) - take
        if leftover <= 1e-9:
            lots.pop(0)
        else:
            lot["signed_qty"] = leftover if lot["signed_qty"] > 0 else -leftover
        remaining -= take
    open_fx = fx_weight / close_mag if close_mag else 1.0
    return matched_local, matched_base, side, open_fx


def _close_return(
    row: dict, closed_qty: float, matched_local: float, matched_base: float, open_fx: float, side: str
):
    fx_close = row["fx"] if row["fx"] and row["fx"] > 0 else 1.0
    if not open_fx or open_fx <= 0:
        open_fx = fx_close
    mult = row["multiplier"] if row["multiplier"] and row["multiplier"] > 0 else 1.0
    fifo = row["fifo"]
    whole_fill = abs(abs(row["signed_qty"]) - closed_qty) <= 1e-6

    if fifo is not None:
        if float(fifo) == 0.0:
            return 0.0
        if row["cost"] is not None and abs(row["cost"]) > 1e-12:
            basis_base = abs(row["cost"]) * open_fx
        elif matched_base > 1e-12:
            basis_base = matched_base
        elif row["price"] > 0 and closed_qty > 0:
            basis_base = row["price"] * closed_qty * mult * fx_close
        else:
            return None
        if basis_base <= 1e-12:
            return None
        return _clip_return((float(fifo) * fx_close) / basis_base)

    if matched_local > 1e-12 and row["price"] > 0 and closed_qty > 0:
        entry_base = matched_base if matched_base > 1e-12 else matched_local * open_fx
        exit_base = row["price"] * closed_qty * mult * fx_close
        realized = exit_base - entry_base if side == "long" else entry_base - exit_base
        if entry_base <= 1e-12:
            return None
        return _clip_return(realized / entry_base)

    if whole_fill and row["cost"] is not None and row["net_cash"] is not None:
        return calculate_pnl_fraction(
            row["net_cash"],
            row["price"],
            row["signed_qty"],
            None,
            multiplier=mult,
            cost=row["cost"],
            fx_rate=fx_close,
        )
    return None


def _add_lot(lots: list[dict], signed_qty: float, row: dict) -> None:
    mult = row["multiplier"] if row["multiplier"] and row["multiplier"] > 0 else 1.0
    fx = row["fx"] if row["fx"] and row["fx"] > 0 else 1.0
    if row["price"] > 0:
        unit_local = row["price"] * mult
    elif row["cost"] is not None and abs(signed_qty) > 1e-12:
        unit_local = abs(row["cost"]) / abs(signed_qty)
    else:
        unit_local = 0.0
    lots.append(
        {
            "signed_qty": signed_qty,
            "unit_local": unit_local,
            "unit_base": unit_local * fx,
            "fx": fx,
        }
    )


def _assemble(rows: list[dict]) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    grouped: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for row in sorted(rows, key=lambda item: (item["ts"] is None, item["ts"] or datetime.min, item["seq"])):
        key = _instrument_key(row)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    trades: list[dict] = []
    opens = 0
    orphans = 0
    flips = 0
    exit_notional = False
    saw_fx = False
    unsplit_flips: list[str] = []

    for key in order:
        lots: list[dict] = []
        for row in grouped[key]:
            if row["fx"] not in (None, 1, 1.0):
                saw_fx = True
            position = sum(lot["signed_qty"] for lot in lots)
            close_mag, open_signed = _split_fill(row, position)
            if close_mag > 1e-12:
                matched_local, matched_base, side, open_fx = _consume_lots(lots, close_mag, row["signed_qty"])
                if open_fx not in (None, 1, 1.0):
                    saw_fx = True
                if row["kind"] == "flip" and matched_local <= 1e-12:
                    unsplit_flips.append(row["symbol"])
                if (
                    row["fifo"] not in (None, 0, 0.0)
                    and matched_local <= 1e-12
                    and (row["cost"] is None or abs(row["cost"]) <= 1e-12)
                    and row["price"] > 0
                ):
                    exit_notional = True
                frac = _close_return(row, close_mag, matched_local, matched_base, open_fx, side)
                if frac is None:
                    orphans += 1
                else:
                    trades.append(
                        {
                            "date": row["ts"],
                            "pnl": round(float(frac), 6),
                            "ticker": row["symbol"],
                            "side": side,
                        }
                    )
            if abs(open_signed) > 1e-12:
                _add_lot(lots, open_signed, row)
                if close_mag > 1e-12:
                    flips += 1
                else:
                    opens += 1

    if opens:
        notes.append(f"— {opens} opening trade leg(s) skipped — only closed trades contribute to P&L")
    if flips:
        notes.append(
            f"Split {flips} position flip(s) (C;O or O;C) into the closed return and a new open leg."
        )
    if unsplit_flips:
        symbols = ", ".join(sorted(set(unsplit_flips))[:6])
        notes.append(
            f"Position flip(s) on {symbols} had no earlier open in the file; "
            "realized P&L was kept and the new leg was not tracked."
        )
    if orphans:
        notes.append(f"Skipped {orphans} closing fill(s) with no realized P&L, cost basis, or matching open.")
    if exit_notional:
        notes.append(
            "Some closes had no matching open and no CostBasis, so the return uses "
            "price × quantity × multiplier on that fill."
        )
    if saw_fx:
        notes.append("Converted non-base-currency P&L with fxRateToBase.")
    trades.sort(key=lambda item: (item["date"] is None, item["date"] or datetime.min))
    return trades, notes


def _rows_from_elements(elements: list[tuple[str, object]]) -> tuple[list[dict], list[str]]:
    trade_els = [(tag, el) for tag, el in elements if tag == "Trade"]
    confirm_els = [(tag, el) for tag, el in elements if tag == "TradeConfirm"]
    notes: list[str] = []
    if trade_els and confirm_els:
        notes.append(
            "Trade and TradeConfirm sections were both present; Trade rows were used "
            "so the same fills are not counted twice."
        )
        chosen = trade_els
    else:
        chosen = trade_els or confirm_els
    rows = []
    for seq, (tag, el) in enumerate(chosen):
        row = _normalize(el, tag, seq)
        if row is not None and row["kind"] != "cancel" and row["signed_qty"] != 0:
            rows.append(row)
    kept, level_note = _filter_detail(rows)
    if level_note:
        notes.append(level_note)
    trades, assemble_notes = _assemble(kept)
    notes.extend(assemble_notes)
    return trades, notes


def parse_ibkr_trade_improved(trade_element):
    """Parse one Flex trade element into a closed-trade dict, or (None, reason)."""
    row = _normalize(trade_element, "Trade", 0)
    if row is None or row["kind"] == "cancel" or row["signed_qty"] == 0:
        return None, "pnl_calculation_failed"
    if is_opening_position(row["indicator"], row["signed_qty"], row["net_cash"], row["price"], row["fifo"]):
        # Flips are not openings. Unknown rows with no realized P&L are opens.
        if row["kind"] != "flip":
            return None, "opening_position"
    trades, _notes = _assemble([row])
    if not trades:
        if row["kind"] in {"open", "unknown"} and row["fifo"] in (None, 0, 0.0):
            return None, "opening_position"
        return None, "pnl_calculation_failed"
    return trades[0], None


def parse_flex_statement(root, account_id: str | None = None) -> tuple[list[dict], list[str], str]:
    """Read closed trades from an Activity Flex or Trade Confirmation document.

    Returns ``(trades, notes, base_currency)``. ``account_id`` limits the parse
    to that FlexStatement / trade when the attribute is present.
    """
    currency = "USD"
    wanted = (account_id or "").strip()
    statements = [el for el in root.iter() if _local_tag(el.tag) == "FlexStatement"]
    scopes = statements or [root]
    elements: list[tuple[str, object]] = []
    for stmt in scopes:
        stmt_acct = (stmt.get("accountId") or "").strip() if stmt is not root else ""
        stmt_ccy = ""
        if stmt is not root:
            stmt_ccy = (stmt.get("currency") or stmt.get("Currency") or "").strip()
        if stmt_ccy:
            currency = stmt_ccy.upper()
        if wanted and stmt_acct and stmt_acct != wanted:
            continue
        for el in stmt.iter():
            tag = _local_tag(el.tag)
            if tag == "AccountInformation":
                info_ccy = (el.get("currency") or el.get("baseCurrency") or "").strip()
                if info_ccy:
                    currency = info_ccy.upper()
            if tag not in {"Trade", "TradeConfirm"}:
                continue
            row_acct = (el.get("accountId") or stmt_acct or "").strip()
            if wanted and row_acct and row_acct != wanted:
                continue
            elements.append((tag, el))
    trades, notes = _rows_from_elements(elements)
    return trades, notes, currency


def _decode_upload(contents: bytes) -> str:
    if contents.startswith(b"\xff\xfe") or contents.startswith(b"\xfe\xff"):
        return contents.decode("utf-16")
    return contents.decode("utf-8-sig", errors="ignore")


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _generic_pnl_header(text: str) -> bool:
    """True for a normal date/pnl upload that happens to mention a broker."""
    header = _first_nonempty_line(text).lower()
    if not header or header.startswith("<"):
        return False
    cells = [re.sub(r"[^a-z0-9&]", "", cell.strip().lower()) for cell in re.split(r"[,;\t]", header)]
    pnl_names = {"pnl", "profit", "return", "gain", "pl", "p&l"}
    if not any(cell in pnl_names for cell in cells):
        return False
    blocked = ("fifo", "tprice", "datadiscriminator", "clientaccountid", "opencloseindicator")
    return not any(token in "".join(cells) for token in blocked)


def classify_ibkr_text(text: str) -> str:
    """Return flex_xml, flex_csv, activity_statement, tws, unsupported, or not_ibkr."""
    stripped = text.lstrip("\ufeff \t\r\n")
    if _generic_pnl_header(stripped):
        return "not_ibkr"
    head = stripped[:24000]
    low = head.lower()
    squashed = re.sub(r"[^a-z0-9]", "", low)
    if stripped.startswith("<"):
        if any(
            marker in low
            for marker in ("<flexqueryresponse", "<flexstatement", "<tradeconfirm", "fifopnlrealized")
        ):
            return "flex_xml"
        if "interactive brokers" in low or "interactivebrokers" in low:
            return "unsupported"
        return "not_ibkr"
    flexish = (
        "fifopnlrealized" in squashed
        or "clientaccountid" in squashed
        or "opencloseindicator" in squashed
        or ("levelofdetail" in squashed and "tradeprice" in squashed)
        or ("tradeconfirm" in squashed and "buysell" in squashed)
    )
    if flexish:
        return "flex_csv"
    if "activity statement" in low or "datadiscriminator" in squashed or "fieldnamefieldvalue" in squashed:
        return "activity_statement"
    tws = (
        ("t. price" in low or "t.price" in low)
        and ("realized p/l" in low or "realized p&l" in low or "comm/fee" in low)
    ) or ("execid" in squashed and "permid" in squashed)
    if tws:
        return "tws"
    if "interactive brokers" in low:
        return "unsupported"
    return "not_ibkr"


def classify_ibkr_bytes(contents: bytes) -> str:
    if contents.startswith(b"PK\x03\x04") or contents.startswith(b"PK\x05\x06"):
        return "not_ibkr"
    try:
        text = _decode_upload(contents)
    except UnicodeError:
        return "not_ibkr"
    return classify_ibkr_text(text)


def _delimiter(text: str) -> str:
    sample = text[:4096]
    counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
    return max(counts, key=counts.get)


def _csv_tables(text: str) -> list[tuple[str, list[dict]]]:
    rows = list(csv.reader(io.StringIO(text), delimiter=_delimiter(text)))
    section_hits = 0
    for row in rows[:40]:
        if len(row) >= 2 and row[1].strip().lower() in {"header", "data", "subtotal", "total"}:
            section_hits += 1
    if section_hits >= 2:
        tables: dict[str, list[dict]] = {}
        order: list[str] = []
        header: list[str] | None = None
        section = ""
        for row in rows:
            if len(row) < 2:
                continue
            kind = row[1].strip().lower()
            if kind == "header":
                section = row[0].strip()
                header = [cell.strip() for cell in row[2:]]
                if section not in tables:
                    tables[section] = []
                    order.append(section)
            elif kind == "data" and header is not None:
                values = row[2:]
                record = {
                    header[i]: values[i].strip() if i < len(values) else ""
                    for i in range(len(header))
                    if header[i]
                }
                tables.setdefault(section, []).append(record)
        return [(name, tables[name]) for name in order]
    header_index = None
    for index, row in enumerate(rows[:30]):
        squashed = [re.sub(r"[^a-z0-9]", "", cell.lower()) for cell in row]
        has_symbol = any(cell in {"symbol", "ticker"} for cell in squashed)
        has_qty = any(cell in {"quantity", "qty"} for cell in squashed)
        has_date = any("date" in cell or cell == "datetime" for cell in squashed)
        if has_symbol and has_qty and has_date:
            header_index = index
            break
    if header_index is None:
        return []
    header = [cell.strip() for cell in rows[header_index]]
    records = []
    for row in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in row):
            continue
        records.append(
            {header[i]: row[i].strip() if i < len(row) else "" for i in range(len(header)) if header[i]}
        )
    return [("", records)]


def _tag_for_csv_record(section: str, record: dict) -> str:
    section_key = re.sub(r"[^a-z]", "", section.lower())
    if "confirm" in section_key:
        return "TradeConfirm"
    if section_key in {"trade", "trades"}:
        return "Trade"
    keys = {re.sub(r"[^a-z0-9]", "", key.lower()) for key in record}
    if "fifopnlrealized" in keys or "opencloseindicator" in keys:
        return "Trade"
    if "code" in keys and "fifopnlrealized" not in keys:
        return "TradeConfirm"
    return "Trade"


def parse_flex_csv(text: str) -> tuple[list[dict], list[str]]:
    tables = _csv_tables(text)
    trade_rows: list[dict] = []
    confirm_rows: list[dict] = []
    seq = 0
    for section, records in tables:
        for record in records:
            tag = _tag_for_csv_record(section, record)
            row = _normalize(record, tag, seq)
            seq += 1
            if row is None or row["kind"] == "cancel" or row["signed_qty"] == 0:
                continue
            if tag == "TradeConfirm":
                confirm_rows.append(row)
            else:
                trade_rows.append(row)
    notes: list[str] = []
    if trade_rows and confirm_rows:
        notes.append(
            "Trade and TradeConfirm sections were both present; Trade rows were used "
            "so the same fills are not counted twice."
        )
        chosen = trade_rows
    else:
        chosen = trade_rows or confirm_rows
    kept, level_note = _filter_detail(chosen)
    if level_note:
        notes.append(level_note)
    trades, assemble_notes = _assemble(kept)
    notes.extend(assemble_notes)
    return trades, notes


def parse_ibkr_upload(contents: bytes) -> tuple[pd.DataFrame, str]:
    """Parse a Flex XML/CSV upload, or reject other IBKR files with a clear error."""
    kind = classify_ibkr_bytes(contents)
    if kind == "activity_statement":
        raise ValueError(ACTIVITY_STATEMENT_MESSAGE)
    if kind == "tws":
        raise ValueError(TWS_MESSAGE)
    if kind in {"unsupported", "not_ibkr"}:
        raise ValueError(UNSUPPORTED_IBKR_MESSAGE)
    text = _decode_upload(contents)
    if kind == "flex_xml":
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError("Could not parse this file as IBKR Flex XML.") from exc
        trades, notes, _currency = parse_flex_statement(root)
        source = "IBKR Flex XML"
    else:
        trades, notes = parse_flex_csv(text)
        source = "IBKR Flex CSV"
    if not trades:
        extra = (" " + " ".join(notes)) if notes else ""
        raise ValueError(NO_CLOSED_TRADES_MESSAGE + extra)
    frame = pd.DataFrame(trades)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    note = f"Parsed {len(frame)} closed trade(s) from {source}."
    if notes:
        note += " " + " ".join(notes)
    return frame, note


def _contains_trades(root) -> bool:
    for el in root.iter():
        if _local_tag(el.tag) in {"Trade", "TradeConfirm"}:
            return True
    return False


def _is_pending(root) -> bool:
    if _contains_trades(root):
        return False
    code = (root.findtext("ErrorCode") or "").strip()
    if code in DO_NOT_RETRY_CODES:
        return False
    message = (root.findtext("ErrorMessage") or "").lower()
    if code in PENDING_CODES or "generation in progress" in message:
        return True
    return "try again shortly" in message


def _is_hard_failure(root) -> bool:
    if _contains_trades(root):
        return False
    code = (root.findtext("ErrorCode") or "").strip()
    status = (root.findtext("Status") or "").strip().lower()
    if code:
        return True
    return status in {"fail", "failure", "error"}


async def fetch_flex_statement(
    token: str,
    query_id: str,
    days_back: int | None = 365,
    *,
    get,
    sleep,
    max_attempts: int = 4,
    initial_delay: float = 5.0,
    retry_delay: float = 5.0,
):
    """SendRequest then GetStatement, retrying while IBKR is still generating it.

    ``get`` is an async callable ``(url) -> response text``. ``sleep`` is an
    async callable ``(seconds) -> None``. The token is stripped from errors.
    ``days_back`` is sent as the Flex ``p`` period and capped at 365.
    """
    period = period_days(days_back)
    # quote, not quote_plus: '+' '/' '=' '&' in a Flex token must be percent-encoded.
    # quote_plus would turn a space into '+', which is already a legal token character.
    send_url = (
        FLEX_SEND_URL
        + "?"
        + urlencode({"t": token, "q": query_id, "v": "3", "p": str(period)}, quote_via=quote)
    )
    try:
        send_text = await get(send_url)
    except IBKRServiceError:
        raise
    except Exception as exc:
        raise IBKRServiceError(
            f"Could not reach IBKR servers: {redact_secret(str(exc), token)}",
            503,
        ) from exc
    try:
        root = ET.fromstring(send_text)
    except ET.ParseError as exc:
        raise IBKRServiceError("Could not parse IBKR response.", 400) from exc
    status = (root.findtext("Status") or "").strip()
    if status != "Success":
        message = root.findtext("ErrorMessage") or "Unknown error"
        raise IBKRServiceError(f"IBKR Flex request failed: {redact_secret(message, token)}", 400)
    reference = (root.findtext("ReferenceCode") or "").strip()
    base_url = (root.findtext("Url") or "").strip()
    if not reference or not base_url:
        raise IBKRServiceError("IBKR did not return a statement reference.", 400)
    retrieve_url = base_url + "?" + urlencode({"t": token, "q": reference, "v": "3"}, quote_via=quote)
    last_message = "Statement generation in progress. Please try again shortly."
    for attempt in range(max_attempts):
        await sleep(initial_delay if attempt == 0 else retry_delay)
        try:
            text = await get(retrieve_url)
        except IBKRServiceError:
            raise
        except Exception as exc:
            raise IBKRServiceError(
                f"Could not retrieve IBKR report: {redact_secret(str(exc), token)}",
                503,
            ) from exc
        try:
            report = ET.fromstring(text)
        except ET.ParseError as exc:
            raise IBKRServiceError("Could not parse IBKR statement.", 400) from exc
        if _is_pending(report):
            last_message = report.findtext("ErrorMessage") or last_message
            continue
        if _is_hard_failure(report):
            message = report.findtext("ErrorMessage") or "Unknown error"
            raise IBKRServiceError(f"IBKR Flex statement failed: {redact_secret(message, token)}", 400)
        return report
    raise IBKRServiceError(
        "IBKR is still generating the statement "
        f"({redact_secret(last_message, token)}). Wait a minute and try again.",
        503,
    )


async def download_flex_statement(token: str, query_id: str, days_back: int | None = 365):
    """Download a Flex statement, retrying error 1019 and the other not-ready codes."""
    import asyncio

    import httpx

    async def get(url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as exc:
            raise IBKRServiceError(
                f"Could not reach IBKR servers: {redact_secret(str(exc), token)}",
                503,
            ) from exc

    return await fetch_flex_statement(token, query_id, days_back, get=get, sleep=asyncio.sleep)
