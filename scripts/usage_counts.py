#!/usr/bin/env python3
"""Print aggregated validation counts from DATABASE_URL.

Does not print client hashes, IP addresses, file contents, or secrets.
With no DATABASE_URL the command has nothing stored and reports an empty
in-memory count. On the hosted API, use GET /debug/usage?t=$DEBUG_ADMIN_TOKEN
instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.usage import summary  # noqa: E402


def main() -> None:
    print(json.dumps(summary(), indent=2))


if __name__ == "__main__":
    main()
