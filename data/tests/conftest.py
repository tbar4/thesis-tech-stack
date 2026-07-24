"""Shared fixtures. Every test runs against a throwaway data home."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dlt_hermetic(monkeypatch, tmp_path):
    """dlt must never phone home or write state outside the test sandbox."""
    monkeypatch.setenv("RUNTIME__DLTHUB_TELEMETRY", "false")
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / ".dlt-data"))


@pytest.fixture
def tmp_data_home(tmp_path, monkeypatch):
    """Point SDA_DATA_HOME at a tmp dir so tiers never touch the real disk."""
    monkeypatch.setenv("SDA_DATA_HOME", str(tmp_path))
    return tmp_path
