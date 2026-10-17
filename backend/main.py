"""QuantProof API — application entry point.

The app is assembled from routers (HTTP layer) and services (domain logic):

    routers/validation.py   /validate, /validate/ai-fixes, /validate/prop-firms, /validate/portfolio
    routers/brokers.py      /validate/alpaca, /connect/ibkr, /validate/kite
    routers/reports.py      /report/* (PDF generation, checkout, payment webhooks)
    routers/tradingview.py  /webhook/tradingview/*
    routers/system.py       /, /health, /ping, /debug/*

Run locally:  uvicorn main:app --reload   (from the backend/ directory)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_VERSION
from routers import brokers, reports, system, tradingview, validation

app = FastAPI(title="QuantProof API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(validation.router)
app.include_router(brokers.router)
app.include_router(reports.router)
app.include_router(system.router)
app.include_router(tradingview.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 - container entrypoint
