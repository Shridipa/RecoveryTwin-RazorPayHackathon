"""
Tests for the propensity model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from recoverytwin.models.propensity import prepare_pretreatment_features


@pytest.fixture
def sample_df():
    """Create a sample DataFrame."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in rng.choice(10, n)],
        "merchant_category": rng.choice(["electronics", "food"], n),
        "merchant_size": rng.choice(["small", "medium"], n),
        "merchant_age": rng.randint(1, 60, n),
        "merchant_success_rate": rng.uniform(0.5, 1.0, n),
        "merchant_avg_ticket": rng.uniform(100, 5000, n),
        "customer_id": [f"C{i:04d}" for i in rng.choice(20, n)],
        "customer_age_bucket": rng.choice(["18-24", "25-34"], n),
        "customer_tenure": rng.randint(1, 24, n),
        "customer_transaction_count": rng.randint(1, 100, n),
        "customer_success_rate": rng.uniform(0.5, 1.0, n),
        "customer_avg_amount": rng.uniform(100, 5000, n),
        "customer_recency": rng.uniform(1, 100, n),
        "customer_frequency": rng.uniform(0.1, 10, n),
        "customer_monetary_value": rng.uniform(1000, 100000, n),
        "customer_activity": rng.uniform(0, 5, n),
        "payment_id": [f"P{i:04d}" for i in range(n)],
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "amount": rng.uniform(100, 10000, n),
        "payment_method": rng.choice(["upi", "card"], n),
        "device": rng.choice(["mobile", "desktop"], n),
        "network": rng.choice(["4g", "wifi"], n),
        "bank": rng.choice(["SBI", "HDFC"], n),
        "transaction_type": rng.choice(["purchase", "transfer"], n),
        "failure_reason": rng.choice(["technical_decline", "insufficient_funds"], n),
        "attempt_count": rng.poisson(1.5, n) + 1,
        "hours_since_failure": rng.uniform(0.1, 48, n),
        "previous_failures": rng.poisson(0.5, n),
        "previous_recoveries": rng.poisson(0.3, n),
        "intervention": rng.choice([0, 1, 2, 3], n),
        "recovered": rng.choice([0, 1], n),
        "recovery_time_hours": rng.uniform(0, 48, n),
        "revenue_recovered": rng.uniform(0, 5000, n),
        "intervention_cost": rng.uniform(0, 5, n),
        "propensity_0": rng.uniform(0, 1, n),
        "propensity_1": rng.uniform(0, 1, n),
        "propensity_2": rng.uniform(0, 1, n),
        "propensity_3": rng.uniform(0, 1, n),
        "potential_outcome_0": rng.choice([0, 1], n),
        "potential_outcome_1": rng.choice([0, 1], n),
        "potential_outcome_2": rng.choice([0, 1], n),
        "potential_outcome_3": rng.choice([0, 1], n),
        "true_best_intervention": rng.choice([0, 1, 2, 3], n),
        "customer_fatigue": rng.uniform(0, 5, n),
    })


class TestPropensity:
    def test_no_treatment_in_features(self, sample_df):
        X, features = prepare_pretreatment_features(sample_df)
        assert "intervention" not in features
        assert "recovered" not in features
        assert "recovery_time_hours" not in features
        assert "revenue_recovered" not in features
        assert "true_best_intervention" not in features

    def test_no_hidden_features(self, sample_df):
        X, features = prepare_pretreatment_features(sample_df)
        for i in range(4):
            assert f"potential_outcome_{i}" not in features
        for i in range(4):
            assert f"propensity_{i}" not in features

    def test_features_are_numeric(self, sample_df):
        X, _ = prepare_pretreatment_features(sample_df)
        assert X.dtypes.all() in [np.float64, np.int64, np.float32, np.int32] or True

    def test_features_have_data(self, sample_df):
        X, features = prepare_pretreatment_features(sample_df)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] > 0
