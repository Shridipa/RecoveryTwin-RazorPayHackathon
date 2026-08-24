"""
Tests for the temporal splitter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from recoverytwin.data.splitter import temporal_split


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with timestamps across 2024."""
    n = 1000
    timestamps = pd.date_range("2024-01-01", "2024-12-31", periods=n)
    return pd.DataFrame({
        "payment_id": [f"P{i:04d}" for i in range(n)],
        "timestamp": timestamps,
        "amount": np.random.RandomState(42).uniform(100, 10000, n),
        "recovered": np.random.RandomState(42).choice([0, 1], n),
        "intervention": np.random.RandomState(42).choice([0, 1, 2, 3], n),
    })


class TestSplitter:
    def test_split_returns_three_parts(self, sample_df):
        train, val, test = temporal_split(sample_df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_overlap(self, sample_df):
        train, val, test = temporal_split(sample_df)
        train_ids = set(train["payment_id"])
        val_ids = set(val["payment_id"])
        test_ids = set(test["payment_id"])
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_temporal_order(self, sample_df):
        train, val, test = temporal_split(sample_df)
        train_max = pd.to_datetime(train["timestamp"]).max()
        val_min = pd.to_datetime(val["timestamp"]).min()
        val_max = pd.to_datetime(val["timestamp"]).max()
        test_min = pd.to_datetime(test["timestamp"]).min()
        assert train_max < val_min
        assert val_max < test_min

    def test_total_rows_conserved(self, sample_df):
        train, val, test = temporal_split(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)
