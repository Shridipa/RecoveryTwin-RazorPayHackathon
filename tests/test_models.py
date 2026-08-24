"""
Tests for the model training pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from recoverytwin.models.baseline import BaselineModel, prepare_features


@pytest.fixture
def training_data():
    """Create small training data."""
    rng = np.random.RandomState(42)
    n = 300

    df = pd.DataFrame({
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
    })

    return df


class TestBaselineModels:
    def test_logistic_fit_and_predict(self, training_data):
        X, feat_names, encoders = prepare_features(training_data, include_treatment=False)
        y = training_data["recovered"].values

        model = BaselineModel("logistic", include_treatment=False)
        model.fit(X, y)

        preds = model.predict_proba(X)
        assert len(preds) == len(y)
        assert all(0 <= p <= 1 for p in preds)

    def test_random_forest_fit_and_predict(self, training_data):
        X, feat_names, encoders = prepare_features(training_data, include_treatment=False)
        y = training_data["recovered"].values

        model = BaselineModel("random_forest", include_treatment=False)
        model.fit(X, y)

        preds = model.predict_proba(X)
        assert len(preds) == len(y)
        assert all(0 <= p <= 1 for p in preds)

    def test_treatment_aware_model(self, training_data):
        X, feat_names, encoders = prepare_features(training_data, include_treatment=True)
        y = training_data["recovered"].values

        assert "intervention" in feat_names

        model = BaselineModel("logistic", include_treatment=True)
        model.fit(X, y)

        preds = model.predict_proba(X)
        assert len(preds) == len(y)

    def test_feature_count_differs(self, training_data):
        X_no_t, feat_no_t, _ = prepare_features(training_data, include_treatment=False)
        X_t, feat_t, _ = prepare_features(training_data, include_treatment=True)
        assert len(feat_t) == len(feat_no_t) + 1  # intervention added
