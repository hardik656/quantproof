"""
QuantProof — FastAPI Backend v1.4
Synced to validator v6.2

CHANGES FROM v1.3:  
  - Version bumped to 2.6.0
  - Root endpoint: 38 → 44 checks
  - All summary strings: '3 crash simulations' → '5 crash simulations'
  - smart_parse_csv(): auto-assigns regime via yfinance SPY SMA when dates
    present but no regime column (fallback: SPY monthly lookup table)
  - /validate/alpaca + /report/generate-broker: replaced broken buy/sell
    stack pairing with position-state tracker (sort by filled_at, FIFO lots,
    orphan sell detection)
  - All Alpaca + IBKR paths now auto-assign regime after trade pairing
  - benchmark_returns now inferred from SPY when date column present.
    IR + Beta checks are scored (not informational) for dated strategies.
    Falls back to None silently — always backward-compatible.

PARSER FIXES (v1.4.1):
  - Added Kite Orders CSV support (_is_kite_orders_csv, _parse_kite_orders_csv)
    Handles: transaction_type, order_execution_time, filled_quantity, average_price
  - Added Alpaca UI export support (_is_alpaca_ui_export, _parse_alpaca_ui_export)
    Handles: tab-separated UI table with human-readable dates like "Feb 24, 2026, 08:40:53 PM"
  - smart_parse_csv() now detects both new formats before falling through to generic parse
  - pair_alpaca_orders() hardened to handle both REST API JSON and UI export dict shapes
    (normalises filled_qty/avg_fill_price/filled_at column names from both sources)

PATCH FIXES:
  - Removed dead record_validation calls (analytics.py not present in this version)
  - Normalized PnL clipping to ?0.99 across all parsers to match validator guard
  - Fixed misleading orphan sell audit message in pair_alpaca_orders
  - Moved validator.run() off event loop in /validate/alpaca (run_in_executor)
  - Fixed _get_cached_validation to use LRU eviction instead of FIFO
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import os
import re as _re
import sys
import time
import hashlib
import traceback
import uuid
import asyncio
from collections import defaultdict

from validator import QuantProofValidator, ValidationDashboard
from config import API_VERSION, extract_n_trials
from schemas import CheckResultResponse, CrashSimResponse, ValidationResponse
from services.ai_plans import generate_ai_plan, generate_portfolio_fix_plan
from services.benchmarks import assign_regimes, get_benchmark_type, get_nifty_benchmark, get_spy_benchmark
from services.cache import VALIDATION_RESULT_CACHE, _PDF_SEMAPHORE, _cache_validation, _get_cached_validation, _save_validation_cache
from services.ibkr import parse_ibkr_trade_improved
from services.parsers import _dedup_trades, _is_kite_csv, _is_kite_orders_csv, _parse_kite_csv, _parse_kite_orders_csv, pair_alpaca_orders, retry_on_failure, smart_parse_csv
from services.portfolio import PortfolioResponse, _diversification_score, _drawdown_coincidence, _portfolio_correlation, _portfolio_verdict, _risk_parity_weights
from services.sebi import _compute_sebi_checks
from services.tokens import FREE_DAILY_LIMIT, REPORT_ACCESS_TOKENS, TOKEN_LITE, TOKEN_PORTFOLIO, TOKEN_PRO, TOKEN_REPORT, _PRO_TOKENS, _is_pro_token, _save_pro_tokens, _save_tokens, _token_tier, _validation_counts, check_rate_limit

# Import prop firm compliance module
try:
    from prop_firm_compliance import PropFirmComplianceEngine, PropFirmRules, QuantProofValidatorWithPropFirms
    print(f"✅ main.py PropFirmComplianceEngine import successful")
except ImportError as e:
    # Fallback if prop_firm_compliance.py doesn't exist yet
    print(f"❌ main.py PropFirmComplianceEngine ImportError: {e}")
    PropFirmComplianceEngine = None
    PropFirmRules = None
    QuantProofValidatorWithPropFirms = None
except Exception as e:
    # Handle any other import errors
    print(f"❌ main.py PropFirmComplianceEngine Exception: {e}")
    import traceback
    traceback.print_exc()
    PropFirmComplianceEngine = None
    PropFirmRules = None
    QuantProofValidatorWithPropFirms = None

# ================= HELPER FUNCTIONS =================




app = FastAPI(title="QuantProof API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= ROUTERS =================
from routers import brokers as _brokers_router
from routers import reports as _reports_router
from routers import system as _system_router
from routers import tradingview as _tradingview_router
from routers import validation as _validation_router

app.include_router(_validation_router.router)
app.include_router(_brokers_router.router)
app.include_router(_reports_router.router)
app.include_router(_system_router.router)
app.include_router(_tradingview_router.router)


# ================= SERVER STARTUP =================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting QuantProof Backend Server...")
    print("📍 Enhanced Validation endpoints ready")
    print("📍 Prop Firm Compliance integration enabled")
    print("📍 Server running on: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)































