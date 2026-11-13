"""Shared CSV/Excel decoding, number and date parsing, and FIFO matching.

Broker parsers used to guess separately. A wrong guess (day/month swapped on an
ISO date, a price column read as a return, an identified file falling through
to that guess) changes the score with no error. These helpers either parse a
value the same way everywhere or leave a note the caller must surface.
"""

from __future__ import annotations

import csv
import io
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

_HEADER_TOKENS = {
    "date",
    "time",
    "dt",
    "trade_date",
    "trade_type",
    "order_execution_time",
    "order_timestamp",
    "transaction_type",
    "tradingsymbol",
    "symbol",
    "isin",
    "quantity",
    "qty",
    "price",
    "average_price",
    "filled_quantity",
    "filled_qty",
    "filled_at",
    "avg_fill_price",
    "filled_avg_price",
    "side",
    "status",
    "asset",
    "deal",
    "direction",
    "commission",
    "swap",
    "profit",
    "balance",
    "position",
    "volume",
    "pnl",
    "return",
    "return_pct",
    "net_pnl",
    "net_p_and_l",
    "net_p_and_l_pct",
    "trade",
    "trade_number",
    "type",
    "date_and_time",
    "date_time",
    "strike",
    "expiry",
    "expiration",
    "premium",
    "underlying",
    "activity_type",
    "transaction_time",
    "open_date",
    "close_date",
    "signal",
}

_SUMMARY_LABELS = {
    "total",
    "totals",
    "grand total",
    "grand_total",
    "summary",
    "subtotal",
    "sub total",
    "balance",
    "deposit",
    "withdrawal",
}

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$")
_MT5_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}(?:[ T].*)?$")
_YMD_RE = re.compile(r"^\d{8}$")
_SLASH_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:\s+.*)?$")
_EPOCH_RE = re.compile(r"^\d{10,13}$")


def normalize_header(name: object) -> str:
    """Lower-case a column name and turn punctuation into single underscores."""
    text = str(name).replace("\ufeff", "").replace("\u00a0", " ").strip().lower()
    text = text.replace("%", " pct ").replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", text)


def decode_upload(contents: bytes) -> str:
    """Decode UTF-8 (with BOM) or UTF-16. Empty and binary files raise ValueError."""
    if not contents or not contents.strip(b"\x00 \t\r\n"):
        raise ValueError("The file is empty. Export a trade history and upload that file.")
    if contents.startswith(b"\xff\xfe") or contents.startswith(b"\xfe\xff"):
        return contents.decode("utf-16")
    sample = contents[:400]
    if sample.count(b"\x00") > max(8, len(sample) // 4):
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return contents.decode(encoding)
            except UnicodeError:
                continue
        raise ValueError("This file looks like UTF-16 but could not be decoded. Re-export it as UTF-8 CSV.")
    try:
        return contents.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(
            "Could not read this file as UTF-8 or UTF-16 text. "
            "Save the export as CSV (UTF-8) or .xlsx and upload that."
        ) from exc


def excel_to_csv_text(contents: bytes) -> str:
    """Read the first worksheet and return it as CSV text for the same parsers."""
    try:
        frame = pd.read_excel(io.BytesIO(contents), dtype=str)
    except ImportError as exc:
        raise ValueError("Excel support requires the openpyxl package.") from exc
    except Exception as exc:
        raise ValueError(
            "Could not read this Excel file. Save it as .xlsx (not the old .xls format) "
            "or export CSV from the broker."
        ) from exc
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The Excel sheet is empty. Export a sheet that contains the trades.")
    return frame.to_csv(index=False)


def upload_to_text(contents: bytes) -> str:
    """Turn an upload into text. Old Excel .xls is rejected with a specific error."""
    if contents.startswith(b"\xd0\xcf\x11\xe0"):
        raise ValueError(
            "This is an old Excel .xls file. QuantProof reads .xlsx and CSV. "
            "In Excel choose Save As → CSV or Excel Workbook (.xlsx) and upload that."
        )
    if (
        contents.startswith(b"PK\x03\x04")
        or contents.startswith(b"PK\x05\x06")
        or contents.startswith(b"PK\x07\x08")
    ):
        return excel_to_csv_text(contents)
    return decode_upload(contents)


def parse_number(value: object, *, decimal_comma: bool = False) -> tuple[float, bool]:
    """Parse one cell to (number, was_percent). Unparseable values return NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan"), False
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "-", "—", "–", "na", "n/a"}:
        return float("nan"), False
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    text = text.replace("\u00a0", "").replace(" ", "").replace("−", "-").replace("–", "-")
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-"):
        negative = not negative
        text = text[1:]
    was_percent = text.endswith("%")
    if was_percent:
        text = text[:-1]
    text = re.sub(r"(?i)(usd|inr|eur|gbp|jpy|rs\.?)", "", text)
    text = re.sub(r"[₹$€£¥]", "", text)
    text = text.replace("'", "")
    if not text or not re.search(r"\d", text):
        return float("nan"), False
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        grouped = len(parts) > 1 and all(len(part) == 3 and part.isdigit() for part in parts[1:])
        if len(parts) > 2 and grouped:
            text = "".join(parts)
        elif len(parts) == 2 and grouped and not decimal_comma:
            # "1,234" in a comma-separated file is a thousands group.
            # In a semicolon file the same text is a decimal (passed decimal_comma).
            text = "".join(parts)
        else:
            text = text.replace(",", ".")
    elif text.count(".") > 1:
        parts = text.split(".")
        if all(len(part) == 3 and part.isdigit() for part in parts[1:]):
            text = "".join(parts)
    try:
        number = float(text)
    except ValueError:
        return float("nan"), False
    if negative:
        number = -number
    return number, was_percent


def parse_numbers(series: pd.Series, *, decimal_comma: bool = False) -> tuple[pd.Series, bool]:
    """Parse a column. The flag is true when any cell was written with a % sign."""
    parsed = [parse_number(value, decimal_comma=decimal_comma) for value in series]
    values = pd.Series([item[0] for item in parsed], index=series.index, dtype="float64")
    return values, any(item[1] for item in parsed)


def _slash_parts(value: str) -> tuple[int, int] | None:
    match = _SLASH_RE.match(value.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_datetimes(series: pd.Series, *, dayfirst: bool) -> tuple[pd.Series, str]:
    """Parse a date column.

    ISO (`2024-01-05`), MT5 (`2024.01.02 10:00:00`), `YYYYMMDD`, and epoch
    seconds/milliseconds are parsed explicitly. Pandas' ``dayfirst=True`` swaps
    those when both the month and the day are <= 12, which silently moves
    every Zerodha fill to the wrong month. Slash dates use day-first only when
    the file proves it (a day > 12) or when the caller asked for it, and the
    note says which rule was used.
    """
    raw = series.astype(str).str.strip()
    raw = raw.mask(raw.str.lower().isin({"", "nan", "none", "nat", "-", "—"}))
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    note = ""

    present = raw.dropna()
    if present.empty:
        return out, note

    iso = present.str.match(_ISO_RE)
    mt5 = present.str.match(_MT5_RE)
    ymd = present.str.match(_YMD_RE)
    epoch = present.str.match(_EPOCH_RE)
    slash_mask = present.map(lambda value: _slash_parts(value) is not None)
    other = ~(iso | mt5 | ymd | epoch | slash_mask)

    def _assign(mask: pd.Series, parsed: pd.Series) -> None:
        if not bool(mask.any()):
            return
        values = pd.to_datetime(parsed, errors="coerce")
        if getattr(values.dtype, "tz", None) is not None:
            values = values.dt.tz_convert("UTC").dt.tz_localize(None)
        elif values.dtype == object:
            values = values.map(
                lambda item: (
                    item.tz_convert("UTC").tz_localize(None) if getattr(item, "tzinfo", None) else item
                )
            )
        out.loc[mask[mask].index] = pd.to_datetime(values, errors="coerce").to_numpy()

    if iso.any():
        _assign(iso, pd.to_datetime(present[iso], errors="coerce", dayfirst=False, format="mixed"))
    if mt5.any():
        _assign(mt5, pd.to_datetime(present[mt5], errors="coerce", dayfirst=False, format="mixed"))
    if ymd.any():
        _assign(ymd, pd.to_datetime(present[ymd], format="%Y%m%d", errors="coerce"))
    if epoch.any():
        numbers = pd.to_numeric(present[epoch], errors="coerce")
        millis = numbers >= 1e12
        parsed_epoch = pd.Series(pd.NaT, index=numbers.index, dtype="datetime64[ns]")
        if millis.any():
            parsed_epoch.loc[millis] = pd.to_datetime(numbers[millis], unit="ms", errors="coerce")
        if (~millis).any():
            parsed_epoch.loc[~millis] = pd.to_datetime(numbers[~millis], unit="s", errors="coerce")
        _assign(epoch, parsed_epoch)
    if other.any():
        aware = present[other].str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        if aware.any():
            utc = pd.to_datetime(present[other][aware], errors="coerce", utc=True)
            naive = utc.dt.tz_convert("UTC").dt.tz_localize(None)
            _assign(aware, naive)
        if (~aware).any():
            _assign(
                ~aware,
                pd.to_datetime(present[other][~aware], errors="coerce", dayfirst=False, format="mixed"),
            )
    if slash_mask.any():
        parts = [(_slash_parts(value), idx) for idx, value in present[slash_mask].items()]
        first_over = any(pair[0] > 12 for pair, _idx in parts)
        second_over = any(pair[1] > 12 for pair, _idx in parts)
        if first_over and second_over:
            raise ValueError(
                "Dates in this file mix day/month and month/day order "
                "(one value has a day > 12 in the first position and another in the second). "
                "Re-export the dates as YYYY-MM-DD."
            )
        resolved_dayfirst = dayfirst
        ambiguous = False
        if first_over:
            resolved_dayfirst = True
        elif second_over:
            resolved_dayfirst = False
        else:
            ambiguous = True
        parsed_slash = pd.to_datetime(
            present[slash_mask], errors="coerce", dayfirst=resolved_dayfirst, format="mixed"
        )
        _assign(slash_mask, parsed_slash)
        if ambiguous:
            order = "day/month/year" if resolved_dayfirst else "month/day/year"
            note = (
                f"Slash dates are ambiguous (every day and month is 12 or less) and were read as {order}. "
                "Re-export as YYYY-MM-DD if that is not what the file means."
            )
    return out, note


def _detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:80])
    if not sample.strip():
        return ","
    best: tuple[int, int, str] | None = None
    for delimiter in ("\t", ";", ",", "|"):
        try:
            rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
        except csv.Error:
            continue
        widths = [len(row) for row in rows if any(cell.strip() for cell in row)]
        if not widths:
            continue
        mode = max(set(widths), key=widths.count)
        if mode < 2:
            continue
        agree = sum(1 for width in widths if width == mode)
        rank = (agree, mode)
        if best is None or rank > best[:2]:
            best = (agree, mode, delimiter)
    return best[2] if best else ","


def _header_score(cells: list[str]) -> int:
    tokens = [normalize_header(cell) for cell in cells if cell.strip()]
    if len(tokens) < 2:
        return 0
    hits = sum(1 for token in tokens if token in _HEADER_TOKENS)
    # A data row of numbers scores 0. A real header has at least two known names.
    return hits


@dataclass
class LoadedTable:
    frame: pd.DataFrame
    delimiter: str
    header_found: bool
    preamble_rows: int
    notes: list[str] = field(default_factory=list)


def read_delimited(text: str) -> LoadedTable:
    """Read CSV/TSV with the header row wherever it sits in the first 40 lines."""
    if not text or not text.strip():
        raise ValueError("The file is empty. Export a trade history and upload that file.")
    delimiter = _detect_delimiter(text)
    try:
        probed = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        raise ValueError(f"Could not read the delimiter of this file: {exc}") from exc

    best_index = 0
    best_score = 0
    for index, row in enumerate(probed[:40]):
        score = _header_score(row)
        if score > best_score:
            best_index = index
            best_score = score
    header_found = best_score >= 2
    notes: list[str] = []
    if header_found and best_index:
        notes.append(
            f"Skipped {best_index} preamble row(s) before the column header. "
            "Check the preview counts match the export."
        )

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            skiprows=best_index if header_found else 0,
            dtype=str,
            skip_blank_lines=True,
            on_bad_lines="skip",
            keep_default_na=False,
        )
    except TypeError:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            skiprows=best_index if header_found else 0,
            dtype=str,
        )
    except Exception as exc:
        raise ValueError(f"Could not read this file as a table: {exc}") from exc

    frame.columns = [normalize_header(column) for column in frame.columns]
    # Pandas names a blank header "unnamed_n" after normalization of "Unnamed: 0".
    frame = frame.loc[:, [column for column in frame.columns if column and not column.startswith("unnamed")]]
    frame = _drop_summary_rows(frame)
    return LoadedTable(
        frame=frame.reset_index(drop=True),
        delimiter=delimiter,
        header_found=header_found,
        preamble_rows=best_index if header_found else 0,
        notes=notes,
    )


def _drop_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    bad = pd.Series(False, index=frame.index)
    for column in frame.columns:
        if (
            column in {"symbol", "tradingsymbol", "asset", "type", "trade_type", "instrument"}
            or column == frame.columns[0]
        ):
            labels = frame[column].astype(str).str.strip().str.lower()
            bad = (
                bad
                | labels.isin(_SUMMARY_LABELS)
                | labels.str.replace(" ", "", regex=False).isin({"grandtotal", "subtotal"})
            )
            if column in {"symbol", "tradingsymbol", "type", "trade_type"}:
                bad = bad | labels.isin({"symbol", "tradingsymbol", "type", "trade_type"})
    kept = frame.loc[~bad]
    # Drop rows that are entirely empty.
    nonempty = kept.apply(
        lambda row: any(str(value).strip().lower() not in {"", "nan", "none"} for value in row), axis=1
    )
    return kept.loc[nonempty]


def apply_aliases(frame: pd.DataFrame, groups: list[tuple[str, tuple[str, ...]]]) -> pd.DataFrame:
    """Rename the first present alias to the canonical name when that name is absent."""
    renamed = frame
    for canonical, aliases in groups:
        if canonical in renamed.columns:
            continue
        for alias in aliases:
            if alias in renamed.columns:
                renamed = renamed.rename(columns={alias: canonical})
                break
    return renamed


def pnl_column_rank(name: str) -> int | None:
    """Lower rank wins. Percent columns beat dollar P&L columns.

    Cumulative, run-up, and excursion columns are not a per-trade return.
    """
    if any(
        skip in name
        for skip in ("cumul", "favorable", "adverse", "run_up", "drawdown", "excursion", "mfe", "mae")
    ):
        return None
    if name in {"return_pct", "profit_pct", "pnl_pct", "gain_pct", "net_pnl_pct", "net_p_and_l_pct"}:
        return 0
    if name.endswith("_pct") and any(
        token in name for token in ("return", "pnl", "profit", "gain", "p_and_l")
    ):
        return 0
    if name in {"pnl", "return", "profit", "gain", "pl", "p_and_l"}:
        return 1
    if name.endswith(("_pnl", "_return", "_profit")):
        return 2
    if name in {"net_pnl", "net_profit", "net_p_and_l", "realized_pnl", "realized_p_and_l"}:
        return 3
    return None


def clip_return(value: float) -> float:
    if value > 0.99:
        return 0.99
    if value < -0.99:
        return -0.99
    return value


@dataclass
class Fill:
    symbol: str
    side: str  # buy, sell, or close
    qty: float
    price: float
    date: object
    allow_short: bool = False
    intent: str | None = None  # "open", "close", or None
    fee: float = 0.0
    seq: int = 0


@dataclass
class MatchResult:
    trades: list[dict]
    open_qty: dict[str, float]
    orphans: list[dict]
    short_opens: int


def match_fills(fills: list[Fill]) -> MatchResult:
    """FIFO lot match. A sell larger than the long closes it and, when allowed, opens a short with the rest.

    Returns are price returns on the closed quantity, minus fees allocated to that slice,
    clipped to ±0.99 because the validator rejects absolute returns greater than 1.
    Each closed slice gets its own ``match_id`` so two identical same-day round trips
    are not collapsed later.
    """
    lots: dict[str, deque] = defaultdict(deque)
    trades: list[dict] = []
    orphans: list[dict] = []
    short_opens = 0
    next_id = 1

    def _trade(symbol, side, qty, entry_price, exit_price, entry_date, exit_date, fee) -> None:
        nonlocal next_id
        if qty <= 1e-9 or entry_price <= 0 or exit_price <= 0:
            return
        gross = (
            (exit_price - entry_price) / entry_price
            if side == "long"
            else (entry_price - exit_price) / entry_price
        )
        net = gross - (fee / (entry_price * qty))
        hold_days = 0
        try:
            hold_days = max(0, int((pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days))
        except (TypeError, ValueError):
            hold_days = 0
        trades.append(
            {
                "pnl": round(clip_return(float(net)), 6),
                "date": exit_date,
                "ticker": symbol,
                "side": side,
                "qty": qty,
                "match_id": next_id,
                "entry_date": entry_date,
                "buy_price": entry_price if side == "long" else exit_price,
                "sell_price": exit_price if side == "long" else entry_price,
                "hold_days": hold_days,
            }
        )
        next_id += 1

    def _consume(
        symbol: str, cover_side: str, qty: float, price: float, date, fee: float, intent: str | None
    ) -> float:
        book = lots[symbol]
        remaining = qty
        fee_per_unit = fee / qty if qty else 0.0
        while remaining > 1e-9 and book and book[0]["side"] == cover_side:
            lot = book[0]
            close_qty = min(remaining, lot["qty"])
            entry_fee = lot["fee"] * (close_qty / lot["orig_qty"]) if lot["orig_qty"] else 0.0
            lot["fee"] -= entry_fee
            _trade(
                symbol,
                cover_side,
                close_qty,
                lot["price"],
                price,
                lot["date"],
                date,
                entry_fee + fee_per_unit * close_qty,
            )
            lot["qty"] -= close_qty
            remaining -= close_qty
            if lot["qty"] <= 1e-8:
                book.popleft()
        if intent == "close" and remaining > 1e-9:
            orphans.append(
                {
                    "symbol": symbol,
                    "qty": remaining,
                    "price": price,
                    "date": date,
                    "kind": "unmatched_close",
                }
            )
            return 0.0
        return remaining

    ordered = sorted(
        fills, key=lambda fill: (pd.Timestamp(fill.date).value if pd.notna(fill.date) else 0, fill.seq)
    )
    for fill in ordered:
        if fill.qty <= 1e-9 or fill.price <= 0 or fill.side not in {"buy", "sell", "close"}:
            continue
        book = lots[fill.symbol]
        if fill.side == "close":
            while book:
                lot = book.popleft()
                _trade(
                    fill.symbol,
                    lot["side"],
                    lot["qty"],
                    lot["price"],
                    fill.price,
                    lot["date"],
                    fill.date,
                    lot["fee"] + fill.fee,
                )
            continue
        if fill.side == "buy":
            if fill.intent == "close" and not (book and book[0]["side"] == "short"):
                orphans.append(
                    {
                        "symbol": fill.symbol,
                        "qty": fill.qty,
                        "price": fill.price,
                        "date": fill.date,
                        "kind": "unmatched_close",
                    }
                )
                continue
            remaining = fill.qty
            if book and book[0]["side"] == "short":
                remaining = _consume(
                    fill.symbol, "short", fill.qty, fill.price, fill.date, fill.fee, fill.intent
                )
            if remaining > 1e-9:
                lots[fill.symbol].append(
                    {
                        "qty": remaining,
                        "orig_qty": remaining,
                        "price": fill.price,
                        "date": fill.date,
                        "side": "long",
                        "fee": fill.fee * (remaining / fill.qty) if fill.qty else 0.0,
                    }
                )
            continue
        # sell
        if not book:
            if fill.allow_short and fill.intent != "close":
                book.append(
                    {
                        "qty": fill.qty,
                        "orig_qty": fill.qty,
                        "price": fill.price,
                        "date": fill.date,
                        "side": "short",
                        "fee": fill.fee,
                    }
                )
                short_opens += 1
            else:
                orphans.append(
                    {
                        "symbol": fill.symbol,
                        "qty": fill.qty,
                        "price": fill.price,
                        "date": fill.date,
                        "kind": "unmatched_sell",
                    }
                )
            continue
        if book[0]["side"] == "short":
            if fill.intent == "close":
                orphans.append(
                    {
                        "symbol": fill.symbol,
                        "qty": fill.qty,
                        "price": fill.price,
                        "date": fill.date,
                        "kind": "unmatched_close",
                    }
                )
            else:
                book.append(
                    {
                        "qty": fill.qty,
                        "orig_qty": fill.qty,
                        "price": fill.price,
                        "date": fill.date,
                        "side": "short",
                        "fee": fill.fee,
                    }
                )
            continue
        remaining = _consume(fill.symbol, "long", fill.qty, fill.price, fill.date, fill.fee, fill.intent)
        if remaining > 1e-9:
            if fill.allow_short and fill.intent != "close":
                lots[fill.symbol].append(
                    {
                        "qty": remaining,
                        "orig_qty": remaining,
                        "price": fill.price,
                        "date": fill.date,
                        "side": "short",
                        "fee": 0.0,
                    }
                )
                short_opens += 1
            else:
                orphans.append(
                    {
                        "symbol": fill.symbol,
                        "qty": remaining,
                        "price": fill.price,
                        "date": fill.date,
                        "kind": "excess_sell",
                    }
                )

    open_qty = {symbol: sum(lot["qty"] for lot in book) for symbol, book in lots.items() if book}
    return MatchResult(trades=trades, open_qty=open_qty, orphans=orphans, short_opens=short_opens)


def trades_frame(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    frame = pd.DataFrame(trades)
    if "date" in frame.columns:
        frame = frame.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    return frame


def describe_open(open_qty: dict[str, float]) -> str:
    items = list(open_qty.items())
    shown = ", ".join(f"{symbol}({int(round(qty))})" for symbol, qty in items[:6])
    return f" {len(items)} position(s) still open (no closing trade yet): {shown}"
