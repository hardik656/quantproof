"""Limits shared by the importers and the validator.

A few thousand closed trades is enough for the statistical checks. Past that,
the partial-fill matrix and the Monte Carlo equity paths allocate hundreds of
megabytes per request and exhaust the 512 MB server.
"""

MAX_TRADES_PER_RUN = 8000


class TradeImportError(ValueError):
    """A file was read, but it must not be scored. Routers map this to HTTP 400."""
