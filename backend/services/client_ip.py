"""Client address behind Render's reverse proxy.

The public API (quantproof-backend.onrender.com) is reached directly, not only
through Cloudflare. A caller can therefore set ``X-Forwarded-For`` and
``CF-Connecting-IP`` to anything. Render's load balancer appends the TCP peer
it actually accepted. That right-most address is the one that cannot be forged.
"""

from __future__ import annotations

import os


def trusted_proxy_hops() -> int:
    """How many right-hand X-Forwarded-For entries were added by proxies we trust.

    Render itself counts as 1. ``TRUSTED_PROXY_HOPS`` overrides
    ``PADDLE_TRUSTED_PROXY_HOPS`` so the rate limit and the Paddle webhook
    allowlist stay on the same hop count when only the older variable is set.
    """
    raw = os.environ.get("TRUSTED_PROXY_HOPS") or os.environ.get("PADDLE_TRUSTED_PROXY_HOPS") or "1"
    try:
        return max(int(str(raw).strip() or "1"), 1)
    except ValueError:
        return 1


def client_ip(request, trusted_hops: int | None = None) -> str:
    """Return the caller IP the trusted proxy actually saw.

    ``CF-Connecting-IP`` is ignored on purpose: anyone who can reach the API
    host directly can set it. Raise ``TRUSTED_PROXY_HOPS`` only when an extra
    proxy that the client cannot bypass (for example Cloudflare locked to the
    API) appends its own address in front of Render.
    """
    hops = trusted_proxy_hops() if trusted_hops is None else max(int(trusted_hops), 1)
    header = ""
    if request is not None:
        header = request.headers.get("x-forwarded-for", "") or ""
    parts = [part.strip() for part in header.split(",") if part.strip()]
    if len(parts) >= hops:
        return parts[-hops]
    client = getattr(request, "client", None) if request is not None else None
    host = getattr(client, "host", "") if client else ""
    return host or "unknown"
