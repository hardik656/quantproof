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
    profitable_trades: int
    date_range: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    top_issues: List[str]
    top_strengths: List[str]
    assumptions: List[str]
    validation_hash: str
    validation_date: str
    audit_flags: List[str]
    plausibility_summary: str
    timeframe_info: str
    engine_version: str

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
                    # Normalize dollar amounts to percentage returns
                    if vals.abs().mean() > 1.0:
                        vals = vals / 10000.0
                    vals = vals.clip(-0.50, 0.50)
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
                # Normalize dollar amounts to percentage returns
                if vals.abs().mean() > 1.0:
                    vals = vals / 10000.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values})
        # Fallback: any numeric col with variance
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 5 and vals.std() > 0:
                # Normalize dollar amounts to percentage returns
                if vals.abs().mean() > 1.0:
                    vals = vals / 10000.0
                vals = vals.clip(-0.50, 0.50)
                return pd.DataFrame({"pnl": vals.values})
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {str(e)}")

    raise ValueError("No valid PnL column found. Make sure your CSV has a 'pnl' or 'profit' column.")

@app.get("/")
def root():
  return {"status": "QuantProof API is live", "version": "2.0.0"}

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

    r = report
    checks_filtered = [c for c in r.checks if c.category != "Plausibility"]
    top_issues = [c.fix for c in sorted(checks_filtered, key=lambda x: x.score)[:3]]
    top_strengths = [c.name for c in sorted(checks_filtered, key=lambda x: x.score, reverse=True)[:3]]

    date_range = "Unknown"
    if hasattr(validator, 'df') and 'date' in validator.df.columns:
        try:
            date_range = f"{validator.df['date'].min().date()} to {validator.df['date'].max().date()}"
        except:
            pass

    summary = (
        f"QuantProof analyzed {r.total_trades} trades across 20 institutional checks "
        f"+ 3 crash simulations. Fundable Score: {r.score}/100 ({r.grade.split('—')[0].strip()}). "
        f"{'Core risk management needs work.' if r.score < 60 else 'Edge shows real promise — focus on execution costs.'}"
    )

    return ValidationResponse(
        fundable_score=r.score,
        grade=r.grade,
        summary=summary,
        checks=[CheckResultResponse(**vars(c)) for c in r.checks if c.category != "Plausibility"],
        crash_sims=[CrashSimResponse(**vars(s)) for s in r.crash_sims],
        total_trades=r.total_trades,
        profitable_trades=r.profitable_trades,
        date_range=date_range,
        sharpe=round(r.sharpe, 2),
        max_drawdown=round(r.max_drawdown, 4),
        win_rate=round(r.win_rate, 1),
        top_issues=top_issues,
        top_strengths=top_strengths,
        assumptions=r.assumptions,
        validation_hash=r.validation_hash,
        validation_date=r.validation_date,
        audit_flags=r.audit_flags,
        plausibility_summary=r.plausibility_summary,
        timeframe_info=validator.timeframe_info,
        engine_version=r.engine_version,
    )

@app.get("/sample-csv")
def sample_csv():
    return {
        "format": "CSV with these columns",
        "required": ["pnl (profit/loss per trade as number)"],
        "optional": ["date", "ticker", "side (long/short)", "entry_price", "exit_price"],
        "example_row": {"date": "2024-01-15", "ticker": "AAPL", "pnl": 245.50, "side": "long"}
    }
