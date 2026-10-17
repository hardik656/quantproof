"""Pydantic response models.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# ================= RESPONSE MODELS =================


class CheckResultResponse(BaseModel):
    name: str
    passed: bool
    score: float
    value: Any
    insight: str
    fix: str
    category: str
    scored: bool = True  # False = informational, excluded from the Fundable Score


class CrashSimResponse(BaseModel):
    crash_name: str
    year: str
    description: str
    market_drop: float
    strategy_drop: float
    survived: bool
    emotional_verdict: str


class ValidationResponse(BaseModel):
    fundable_score: float
    grade: str
    summary: str

    checks: List[CheckResultResponse]
    crash_sims: List[CrashSimResponse]

    total_trades: int
    profitable_trades: int
    date_range: str

    sharpe: float
    max_drawdown: float
    win_rate: float

    top_issues: List[str]
    top_strengths: List[str]

    validation_hash: str
    validation_date: Any
    engine_version: str

    audit_flags: List[str]
    plausibility_summary: Optional[str]

    scale_warning: bool
    currency: str

    assumptions: List[str]
    dashboard: Optional[Dict[str, Any]]
    ai_plan: Optional[Dict[str, Any]]
    equity_curve: Optional[List[float]] = None
    monte_carlo: Optional[Dict[str, Any]] = None
    sebi_checks: Optional[Dict[str, Any]] = None
    sebi_declaration: Optional[Dict[str, Any]] = None
    prop_firm_compliance: Optional[Dict[str, Any]] = None
