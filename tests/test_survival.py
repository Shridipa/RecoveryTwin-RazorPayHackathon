"""
Tests for the survival analysis module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from recoverytwin.survival.kaplan_meier import prepare_survival_data, fit_kaplan_meier


@pytest.fixture
def sample_survival_df():
    """Create sample data with survival information."""
    rng = np.random.RandomState(42)
    n = 200
    recovered = rng.choice([0, 1], n, p=[0.7, 0.3])
    recovery_time = np.where(
        recovered == 1,
        rng.exponential(8, n),
        0.0,
    )

    return pd.DataFrame({
        "payment_id": [f"P{i:04d}" for i in range(n)],
        "intervention": rng.choice([0, 1, 2, 3], n),
        "recovered": recovered,
        "recovery_time_hours": recovery_time,
        "amount": rng.uniform(100, 10000, n),
        "customer_activity": rng.uniform(0, 5, n),
        "failure_reason": rng.choice(["technical_decline", "insufficient_funds"], n),
    })


class TestSurvival:
    def test_prepare_survival_data(self, sample_survival_df):
        data = prepare_survival_data(sample_survival_df)
        assert "duration" in data.columns
        assert "event" in data.columns
        assert (data["event"] == sample_survival_df["recovered"]).all()

    def test_survival_duration_positive(self, sample_survival_df):
        data = prepare_survival_data(sample_survival_df)
        assert (data["duration"] > 0).all()

    def test_survival_duration_capped(self, sample_survival_df):
        data = prepare_survival_data(sample_survival_df, censoring_time=48)
        assert (data["duration"] <= 48).all()

    def test_unrecovered_censored(self, sample_survival_df):
        data = prepare_survival_data(sample_survival_df, censoring_time=168)
        unrecovered = data[data["event"] == 0]
        assert (unrecovered["duration"] == 168).all()

    def test_km_fits(self, sample_survival_df):
        results = fit_kaplan_meier(sample_survival_df)
        assert "overall" in results
        assert "by_intervention" in results
        assert len(results["by_intervention"]) == 4

    def test_km_recovery_rates(self, sample_survival_df):
        results = fit_kaplan_meier(sample_survival_df)
        overall_rate = results["overall"]["event_rate"]
        assert 0 < overall_rate < 1

    def test_km_by_intervention(self, sample_survival_df):
        results = fit_kaplan_meier(sample_survival_df)
        for t in range(4):
            info = results["by_intervention"][t]
            assert info["n"] > 0
            assert 0 <= info["event_rate"] <= 1
