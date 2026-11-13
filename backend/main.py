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
from starlette.datastructures import MutableHeaders  # noqa: E402

from config import API_VERSION  # noqa: E402
from routers import brokers, reports, system, tradingview, validation  # noqa: E402
from services.tokens import release_rate_limit  # noqa: E402

# Interactive docs stay off unless explicitly enabled. The public API should not
# advertise every route; set ENABLE_API_DOCS=1 for local development.
_DOCS_ON = os.getenv("ENABLE_API_DOCS", "").strip().lower() in {"1", "true", "yes"}

app = FastAPI(
    title="QuantProof API",
    version=API_VERSION,
    docs_url="/docs" if _DOCS_ON else None,
    redoc_url="/redoc" if _DOCS_ON else None,
    openapi_url="/openapi.json" if _DOCS_ON else None,
)

# API responses are JSON. A locked-down CSP here does not affect the static site
# (that policy lives in render.yaml). Docs, when enabled, need Swagger's own assets.
_API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


def _reserved_rate_key(scope) -> str:
    # Starlette stores request.state in scope["state"] as a plain dict.
    state = scope.get("state")
    if isinstance(state, dict):
        return state.pop("rate_limit_key", None) or ""
    key = getattr(state, "rate_limit_key", None) if state is not None else None
    return key or ""


class _SecurityAndRateLimitMiddleware:
    """Add response headers, and return a free try when the handler rejects the upload."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status_code = 500
        path = scope.get("path") or ""
        docs_path = path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json"

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                for name, value in _SECURITY_HEADERS.items():
                    headers[name] = value
                if not docs_path:
                    headers["Content-Security-Policy"] = _API_CSP
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            key = _reserved_rate_key(scope)
            if key:
                release_rate_limit(key)
            raise
        key = _reserved_rate_key(scope)
        if key and status_code >= 400:
            release_rate_limit(key)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_SecurityAndRateLimitMiddleware)

app.include_router(validation.router)
app.include_router(brokers.router)
app.include_router(reports.router)
app.include_router(system.router)
app.include_router(tradingview.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 - container entrypoint
