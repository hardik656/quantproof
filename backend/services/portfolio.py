"""Portfolio correlation, risk-parity and diversification maths.

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
# PORTFOLIO VALIDATION — ₹199
# =========================================================

class PortfolioResponse(BaseModel):
    strategies:            List[Dict[str, Any]]
    correlation_matrix:    List[List[float]]
    strategy_names:        List[str]
    combined_sharpe:       float
    combined_drawdown:     float
    combined_win_rate:     float
    diversification_score: float
    avg_correlation:       float = 0.0
    combined_total_trades: int
    portfolio_verdict:     str
    audit_flags:           List[str]
    n_trials:              int = 10
    rp_weights:            List[float] = []        # risk parity allocation weights
    dd_coincidence:        float = 0.0             # fraction of DD periods overlapping
    portfolio_dsr:         Optional[float] = None  # combined strategy DSR score
    combined_equity:       List[float] = []        # normalised combined equity curve
    portfolio_plan:        Optional[Dict[str, Any]] = None  # AI fix plan
    is_preview:            bool = False           # True = blurred results for free users


def _normalise_pnl_series(pnl_array: np.ndarray) -> np.ndarray:
    """Normalise PnL to unit variance — removes scale-dependence from correlation."""
    std = float(pnl_array.std())
    return pnl_array / std if std > 1e-10 else pnl_array


def _portfolio_correlation(dfs: list) -> np.ndarray:
    """
    Pairwise Pearson correlation on NORMALISED PnL.
    Normalising to unit variance makes correlation scale-independent —
    a ₹100-avg-trade strategy and a ₹10,000-avg-trade strategy
    are compared on signal shape, not magnitude.
    Aligns on date where available (inner join), falls back to positional.
    """
    n = len(dfs)
    mat = np.ones((n, n))
    series = []
    for df in dfs:
        raw = df["pnl"].values.astype(float)
        norm = _normalise_pnl_series(raw)
        if "date" in df.columns and df["date"].notna().any():
            idx = pd.to_datetime(df["date"], errors="coerce")
            s = pd.Series(norm, index=idx).sort_index()
        else:
            s = pd.Series(norm)
        series.append(s)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                si, sj = series[i], series[j]
                if hasattr(si.index, "date") and hasattr(sj.index, "date"):
                    aligned = pd.concat([si, sj], axis=1, join="inner")
                    if len(aligned) >= 5:
                        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    else:
                        min_len = min(len(si), len(sj))
                        corr = float(np.corrcoef(si.values[:min_len], sj.values[:min_len])[0, 1])
                else:
                    min_len = min(len(si), len(sj))
                    corr = float(np.corrcoef(si.values[:min_len], sj.values[:min_len])[0, 1])
                mat[i][j] = mat[j][i] = round(0.0 if np.isnan(corr) else corr, 4)
            except Exception:
                mat[i][j] = mat[j][i] = 0.0
    return mat


def _risk_parity_weights(dfs: list) -> list:
    """
    Inverse-volatility (risk parity) weights.
    Each strategy weighted by 1/vol so all contribute equally to portfolio variance.
    Institutional standard for multi-strategy allocation.
    """
    vols = []
    for df in dfs:
        v = float(df["pnl"].values.astype(float).std())
        vols.append(max(v, 1e-10))
    inv_vol = [1.0 / v for v in vols]
    total = sum(inv_vol)
    return [float(w / total) for w in inv_vol]


def _drawdown_coincidence(dfs: list, threshold: float = -0.02) -> float:
    """
    Fraction of drawdown-period steps where 2+ strategies are simultaneously
    in drawdown below threshold. Institutional target: < 0.30.
    High coincidence = strategies fail together = poor real diversification.
    """
    if len(dfs) < 2:
        return 0.0
    min_len = min(len(df) for df in dfs)
    dd_flags = []
    for df in dfs:
        pnl = df["pnl"].values.astype(float)[:min_len]
        eq = np.cumprod(1 + np.clip(pnl, -0.99, 0.99))
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / np.where(roll_max > 0, roll_max, 1.0)
        dd_flags.append(dd < threshold)
    stacked   = np.stack(dd_flags, axis=0)
    n_down    = stacked.sum(axis=0)
    any_down  = int((n_down >= 1).sum())
    multi_down = int((n_down >= 2).sum())
    return round(float(multi_down / any_down), 3) if any_down > 0 else 0.0


def _diversification_score(corr_matrix: np.ndarray, sharpes: list) -> float:
    """
    0-100 score. Higher = better diversified portfolio.
    Per-pair scoring:
      near-zero corr → 82-90 (ideal diversification)
      negative corr  → 96-100 (true hedge)
      0.0-0.3 corr   → 40-82 (ok)
      0.3-0.6 corr   → 10-40 (moderate concern)
      0.6+ corr      → 0-10  (high concern, minimal diversification)
    """
    n = len(sharpes)
    if n < 2:
        return 50.0

    def _pair_score(c: float) -> float:
        if c <= -0.3:  return 96.0
        elif c <= 0.0: return 82.0 + abs(c) * 47.0
        elif c < 0.3:  return 82.0 - c * 140.0
        elif c < 0.6:  return 40.0 - (c - 0.3) * 100.0
        else:          return max(0.0, 10.0 - (c - 0.6) * 25.0)

    off_diag = [float(corr_matrix[i][j]) for i in range(n) for j in range(i + 1, n)]
    avg_pair     = float(np.mean([_pair_score(c) for c in off_diag])) if off_diag else 50.0
    sharpe_bonus = (sum(1 for s in sharpes if s > 0) / max(n, 1)) * 8.0
    return round(min(100.0, max(0.0, avg_pair + sharpe_bonus)), 1)


def _portfolio_verdict(div_score: float, avg_corr: float, n: int) -> str:
    if div_score >= 75:
        return (f"Strong portfolio — {n} strategies with low average correlation "
                f"({avg_corr:.2f}). Each strategy adds independent alpha. "
                "This combination is institutionally viable for diversified deployment.")
    elif div_score >= 55:
        return (f"Moderate diversification — {n} strategies with some overlap "
                f"(avg correlation {avg_corr:.2f}). Consider strategies from different "
                "asset classes or time horizons to reduce correlation further.")
    elif div_score >= 35:
        return (f"High correlation detected (avg {avg_corr:.2f}) — these strategies "
                f"likely move together. During drawdowns they will all drawdown simultaneously. "
                "Portfolio diversification benefit is limited.")
    else:
        return (f"Poor diversification — strategies are highly correlated "
                f"(avg {avg_corr:.2f}). Running all {n} together provides minimal "
                "risk reduction over running just the best one.")


