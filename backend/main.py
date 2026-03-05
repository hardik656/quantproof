"""
QuantProof — FastAPI Backend v2.1
Production-ready institutional validation engine
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import dataclasses

from validator import QuantProofValidator, ValidationReport

app = FastAPI(title="QuantProof API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# RESPONSE MODELS
# =========================================================

class CheckResultResponse(BaseModel):
    name: str
    category: str
    passed: bool
    score: float
    value: str
    interpretation: str
    fix: str
    severity: str

class CrashSimResponse(BaseModel):
    scenario: str
    market_return: float
    survival_prob: float
    margin_call_prob: float
    strategy_return_mean: float
    strategy_return_ci: tuple
    max_dd_mean: float
    verdict: str

class PropFirmResponse(BaseModel):
    firm: str
    phase1_pass: bool
    phase2_pass: bool
    funding_eligible: bool
    violations: List[str]
    recommended_size: int

class AlphaDecayResponse(BaseModel):
    half_life_periods: float
    half_life_seconds: Optional[float]
    optimal_holding: int
    capacity_limited: float
    regime_dependent: bool
    microstructure_noise: Optional[float]
    latency_sensitivity_bpms: Optional[float]

class ValidationResponse(BaseModel):
    # Core score
    score: float
    grade: str
    deployment_status: str
    verdict: str
    # Core metrics
    sharpe: Optional[Dict[str, Any]]
    max_dd: float
    cvar95: float
    ruin_prob: Optional[Dict[str, Any]]
    win_rate: float
    half_kelly: float
    # Final 5%
    alpha_decay: Optional[AlphaDecayResponse]
    overfit_score: float
    overfit_details: Dict[str, Any]
    # Results
    checks: List[CheckResultResponse]
    crash_results: List[CrashSimResponse]
    prop_firms: List[PropFirmResponse]
    # Meta
    version: str
    timestamp: str
    hash: str
    total_trades: int

# =========================================================
# CSV PARSER
# =========================================================

def smart_parse_csv(contents: bytes) -> pd.DataFrame:
    raw = contents.decode("utf-8", errors="ignore")

    # Primary parse — look for known return column names
    try:
        df = pd.read_csv(io.StringIO(raw))
        return_names = ["return", "pnl", "profit", "gain", "pl", "p&l", "net", "alpha"]
        for col in df.columns:
            if any(p in col.lower() for p in return_names):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) >= 5:
                    # Check if values are percentages and convert to decimals
                    if vals.abs().max() > 1.0:
                        vals = vals / 100.0
                    vals = vals.clip(-0.50, 0.50)
                    result = pd.DataFrame({"return": vals.values})
                    # Preserve date column if present
                    date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "dt"])]
                    if date_cols:
                        result["date"] = df[date_cols[0]].values[:len(vals)]
                    # Preserve volume column if present (for market impact check)
                    vol_cols = [c for c in df.columns if "volume" in c.lower() or "vol" == c.lower()]
                    if vol_cols:
                        result["volume"] = pd.to_numeric(df[vol_cols[0]], errors="coerce").values[:len(vals)]
                    # Preserve regime column if present
                    reg_cols = [c for c in df.columns if "regime" in c.lower()]
                    if reg_cols:
                        result["regime"] = df[reg_cols[0]].values[:len(vals)]
                    return result
    except Exception:
        pass

    # Fallback — flexible parse ignoring bad lines
    try:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        max_cols = max(len(l.split(",")) for l in lines[1:] if l)
        df = pd.read_csv(
            io.StringIO(raw),
            names=[f"col_{i}" for i in range(max_cols)],
            skiprows=1,
            on_bad_lines="skip",
            engine="python"
        )
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and (vals > 0).any() and (vals < 0).any():
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"return": vals.values})
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                if vals.abs().mean() > 1.0:
                    vals = vals / 100.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"return": vals.values})
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError("No valid return column found. Make sure your CSV has a 'return' or 'pnl' column.")

# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "status": "QuantProof API is live",
        "version": "2.1.0",
        "checks": "12 institutional checks + Final 5% analysis",
        "features": ["Alpha Decay Analysis", "Symbolic Overfit Detection", 
                     "Prop Firm Compliance", "Crash Scenarios",
                     "Production-safe RNG", "Data Integrity Checks"]
    }

@app.post("/validate", response_model=ValidationResponse)
async def validate(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await file.read()

    try:
        df = smart_parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(df) < 5:
        raise HTTPException(status_code=400, detail=f"Only found {len(df)} valid trades. Need at least 5.")

    try:
        validator = QuantProofValidator(df)
        report = validator.validate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

    # Convert ValidationReport to dict for JSON serialization
    report_dict = dataclasses.asdict(report)
    
    # Convert nested dataclasses
    if report_dict.get('sharpe'):
        report_dict['sharpe'] = dataclasses.asdict(report.sharpe)
    if report_dict.get('ruin_prob'):
        report_dict['ruin_prob'] = dataclasses.asdict(report.ruin_prob)
    if report_dict.get('alpha_decay'):
        report_dict['alpha_decay'] = dataclasses.asdict(report.alpha_decay)
    
    return report_dict

@app.get("/sample-csv")
def sample_csv():
    return {
        "format": "CSV with these columns",
        "required": ["return (profit/loss per trade as decimal, e.g. 0.015 = 1.5%)"],
        "optional": ["date", "ticker", "side (long/short)", "regime", "volume"],
        "example_row": {
            "date": "2024-01-15",
            "ticker": "AAPL",
            "return": 0.0145,
            "side": "long",
            "regime": "BULL"
        },
        "note": "Returns must be decimals (0.0145), not percentages (1.45)."
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0"}
