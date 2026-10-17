"""App-wide constants and small request helpers.

Extracted verbatim from main.py (behaviour unchanged).
"""

from __future__ import annotations


def extract_n_trials(request):
    """Extract n_trials from request (JSON or form-data) with safe validation."""
    try:
        if hasattr(request, "is_json") and request.is_json:
            val = request.json.get("n_trials")
        else:
            val = request.form.get("n_trials") if hasattr(request, "form") else None

        return max(1, min(50, int(val))) if val is not None else 1
    except (ValueError, TypeError, AttributeError):
        return 1


API_VERSION = "2.7.0"
