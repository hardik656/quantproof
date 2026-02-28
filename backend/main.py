"""
QuantProof — FastAPI Backend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import io

from validator import QuantProofValidator

app = FastAPI(title="QuantProof API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CheckResultResponse(BaseModel):
    name: str
    passed: bool
    score: float
    value: str
    insight: str
    fix: str
    category: str

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
    date_range: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    top_issues: List[str]
    top_strengths: List[str]

def smart_parse_csv(contents: bytes) -> pd.DataFrame:
    raw = contents.decode("utf-8", errors="ignore")

    # Try standard parse with known pnl column names
    try:
        df = pd.read_csv(io.StringIO(raw))
        pnl_names = ["pnl", "profit", "return", "gain", "pl", "p&l", "net", "alpha"]
        for col in df.columns:
            if any(p in col.lower() for p in pnl_names):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) >= 5:
                    return pd.DataFrame({"pnl": vals.values})
    except Exception:
        pass

    # Flexible parse ignoring bad lines
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
        # Find col with both positive and negative values (most likely pnl)
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and (vals > 0).any() and (vals < 0).any():
                return pd.DataFrame({"pnl": vals.values})
        # Fallback: any numeric col with variance
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                return pd.DataFrame({"pnl": vals.values})
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError("No valid PnL column found. Make sure your CSV has a 'pnl' or 'profit' column.")

@app.get("/")
def root():
    return {"status": "QuantProof API is live", "version": "1.0.0"}

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
        report = validator.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

    return ValidationResponse(
        fundable_score=report.fundable_score,
        grade=report.grade,
        summary=report.summary,
        checks=[CheckResultResponse(**vars(c)) for c in report.checks],
        crash_sims=[CrashSimResponse(**vars(s)) for s in report.crash_sims],
        total_trades=report.total_trades,
        date_range=report.date_range,
        sharpe=report.sharpe,
        max_drawdown=report.max_drawdown,
        win_rate=report.win_rate,
        top_issues=report.top_issues,
        top_strengths=report.top_strengths,
    )

@app.get("/sample-csv")
def sample_csv():
    return {
        "format": "CSV with these columns",
        "required": ["pnl (profit/loss per trade as number)"],
        "optional": ["date", "ticker", "side (long/short)", "entry_price", "exit_price"],
        "example_row": {"date": "2024-01-15", "ticker": "AAPL", "pnl": 245.50, "side": "long"}
    }
