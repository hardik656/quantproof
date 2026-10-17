"""Interactive Brokers Flex-report helpers.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

import io
import os
import time
import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

# =========================================================
# IBKR CONFIGURATION
# =========================================================

IBKR_CONFIG = {
    'opening_detection': {
        'long_cash_outflow_ratio': 0.8,
        'short_cash_inflow_ratio': 0.2,
        'description': 'Thresholds for detecting opening positions'
    },
    'pnl_calculation': {
        'clip_range': 0.99,
        'min_pnl_threshold': 0.001,
        'description': 'P&L calculation parameters'
    },
    'validation': {
        'min_trades': 10,
        'description': 'Minimum trades required for validation'
    },
    'last_updated': '2024-01-01',
    'version': '1.0'
}


# =========================================================
# IBKR PARSER HELPER FUNCTIONS - Improved Logic
# =========================================================

def calculate_trade_side(buy_sell, qty, open_close, net_cash, price):
    """
    Calculate trade side with unified, clear logic.
    """
    if open_close == 'C':
        # Closed trade - determine by position direction
        return 'long' if qty < 0 else 'short'
    else:
        # Opening trade - determine by action
        return 'long' if buy_sell == 'SELL' else 'short'


def is_opening_position(open_close, qty, net_cash, price):
    """
    Detect if this is an opening position with robust logic.
    """
    # Explicit opening indicator
    if open_close == 'O':
        return True
    
    # Explicit closing indicator
    if open_close == 'C':
        return False
    
    # Heuristic detection when indicator is missing
    trade_value = abs(price * qty) if price > 0 and qty != 0 else 0
    
    if trade_value == 0:
        return False  # Can't determine, assume closing
    
    # Use configuration thresholds
    config = IBKR_CONFIG['opening_detection']
    
    # For long positions (qty > 0): opening requires significant cash outflow
    if qty > 0:
        cash_outflow_ratio = abs(net_cash) / trade_value if trade_value > 0 else 0
        return cash_outflow_ratio > config['long_cash_outflow_ratio']
    
    # For short positions (qty < 0): opening requires minimal cash inflow
    else:
        cash_inflow_ratio = abs(net_cash) / trade_value if trade_value > 0 else 0
        return cash_inflow_ratio < config['short_cash_inflow_ratio']


def calculate_pnl_fraction(net_cash, price, qty, fifo_pnl=None):
    """
    Calculate P&L as fraction with robust error handling.
    """
    import numpy as np
    
    # Use configuration
    config = IBKR_CONFIG['pnl_calculation']
    clip_range = config['clip_range']
    min_threshold = config['min_pnl_threshold']
    
    # Use FIFO P&L when available (most accurate)
    if fifo_pnl is not None:
        try:
            fifo_val = float(fifo_pnl)
            if abs(fifo_val) < min_threshold:  # Tiny P&L, likely rounding error
                return None
            
            cost = abs(price * qty) if price > 0 and qty != 0 else 1
            pnl_frac = fifo_val / cost
            return float(np.clip(pnl_frac, -clip_range, clip_range))
        except (ValueError, ZeroDivisionError):
            pass  # Fall back to net cash method
    
    # Net cash method
    if price <= 0 or qty == 0:
        return None  # Can't calculate without valid price/quantity
    
    cost_basis = abs(price * qty)
    pnl_frac = net_cash / cost_basis
    return float(np.clip(pnl_frac, -clip_range, clip_range))


def parse_ibkr_trade_improved(trade_element):
    """
    Simplified, maintainable IBKR trade parser using helper functions.
    """
    # Extract fields with case-insensitive fallbacks
    date_str = trade_element.get("tradeDate") or trade_element.get("TradeDate", "")
    symbol = trade_element.get("symbol", "") or trade_element.get("Symbol", "")
    qty = float(trade_element.get("quantity", 0) or trade_element.get("Quantity", 0) or 0)
    price = float(trade_element.get("tradePrice") or trade_element.get("TradePrice", 0) or 0)
    net_cash = float(trade_element.get("netCash") or trade_element.get("NetCash", 0) or 0)
    buy_sell = (trade_element.get("buySell") or trade_element.get("BuySell") or "").upper()
    open_close = (trade_element.get("openCloseIndicator") or trade_element.get("OpenCloseIndicator") or "").upper()
    fifo_pnl = trade_element.get("fifoPnlRealized") or trade_element.get("FifoPnlRealized")
    
    # Skip opening positions
    if is_opening_position(open_close, qty, net_cash, price):
        return None, "opening_position"
    
    # Calculate P&L
    pnl_frac = calculate_pnl_fraction(net_cash, price, qty, fifo_pnl)
    if pnl_frac is None:
        return None, "pnl_calculation_failed"
    
    # Calculate side using unified logic
    side = calculate_trade_side(buy_sell, qty, open_close, net_cash, price)
    
    return {
        "date": date_str,
        "pnl": round(pnl_frac, 6),
        "ticker": symbol,
        "side": side,
    }, None

