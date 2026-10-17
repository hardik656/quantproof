"""QuantProof API — application entry point.

The app is assembled from routers (HTTP layer) and services (domain logic):

    routers/validation.py   /validate, /validate/ai-fixes, /validate/prop-firms, /validate/portfolio
    routers/brokers.py      /validate/alpaca, /connect/ibkr, /validate/kite
    routers/reports.py      /report/* (PDF generation, checkout, payment webhooks)
    routers/tradingview.py  /webhook/tradingview/*
    routers/system.py       /, /health, /ping, /debug/*

Run locally:  uvicorn main:app --reload   (from the backend/ directory)
"""

import os
import sys

# Modules inside backend/ import each other flat (``from validator import ...``).
# Put this directory on sys.path so the app starts both ways:
#   repo root:   uvicorn backend.main:app   (render.yaml)
#   backend/:    uvicorn main:app
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from config import API_VERSION  # noqa: E402
from routers import brokers, reports, system, tradingview, validation  # noqa: E402

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
