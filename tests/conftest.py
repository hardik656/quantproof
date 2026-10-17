"""Shared fixtures. Tests run fully offline: yfinance is stubbed so the
benchmark/regime code falls back to its built-in lookup tables."""

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

# Isolated sqlite state DB per test session (exercises the SQLAlchemy store
# and keeps tests away from any real /tmp/qp_*.json files).
_STATE_DIR = tempfile.mkdtemp(prefix="qp-test-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_STATE_DIR}/state.db")
os.environ["QP_STATE_DIR"] = _STATE_DIR
os.environ["QP_DB_CREATE_ALL"] = "1"


def _offline_download(*_args, **_kwargs):
    raise ConnectionError("network disabled in tests")


_fake_yf = types.ModuleType("yfinance")
_fake_yf.download = _offline_download  # type: ignore[attr-defined]
sys.modules["yfinance"] = _fake_yf


@pytest.fixture(scope="session")
def app():
    import main

    return main.app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    from services import tokens

    tokens._validation_counts.clear()
    return TestClient(app)


@pytest.fixture
def profitable_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=300, freq="B"),
            "pnl": rng.normal(0.004, 0.01, 300),
        }
    )


@pytest.fixture
def losing_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=300, freq="B"),
            "pnl": rng.normal(-0.004, 0.01, 300),
        }
    )
