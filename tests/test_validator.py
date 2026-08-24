"""
Tests for the data validator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from recoverytwin.data.validator import DataValidator


@pytest.fixture
def valid_df():
    """Create a valid DataFrame for testing."""
    n = 100
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in rng.choice(10, n)],
        "merchant_category": rng.choice(["electronics", "food"], n),
        "merchant_size": rng.choice(["small", "medium", "large"], n),
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
        "recovery_time_hours": np.where(rng.random(n) > 0.5, rng.uniform(0.1, 48, n), 0),
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

    # Fix consistency
    recovered = df["recovered"] == 1
    df.loc[~recovered, "recovery_time_hours"] = 0
    df.loc[recovered, "recovery_time_hours"] = rng.uniform(0.1, 48, recovered.sum())
    df.loc[~recovered, "revenue_recovered"] = 0
    df.loc[recovered, "revenue_recovered"] = df.loc[recovered, "amount"] * 0.9

    return df


class TestValidator:
    def test_all_checks_pass_on_valid_data(self, valid_df):
        validator = DataValidator()
        # No temporal split needed for this test
        summary = validator.run_all(valid_df)
        # Print which checks failed for debugging
        for r in summary["results"]:
            if not r["passed"]:
                print(f"  FAILED: {r['name']}: {r['message']}")
        assert summary["all_pass"]

    def test_schema_check_fails_with_missing_column(self, valid_df):
        df = valid_df.drop(columns=["merchant_id"])
        validator = DataValidator()
        validator.check_schema(df)
        assert not validator.results[-1].passed

    def test_duplicate_payment_check(self, valid_df):
        df = valid_df.copy()
        df = pd.concat([df, df.head(5)], ignore_index=True)
        validator = DataValidator()
        validator.check_duplicate_payments(df)
        assert not validator.results[-1].passed

    def test_invalid_treatment_ids(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "intervention"] = 99
        validator = DataValidator()
        validator.check_valid_treatment_ids(df)
        assert not validator.results[-1].passed

    def test_negative_revenue_detected(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "revenue_recovered"] = -100
        validator = DataValidator()
        validator.check_financial_consistency(df)
        assert not validator.results[-1].passed

    def test_leakage_check(self, valid_df):
        validator = DataValidator()
        validator.check_model_input_features(valid_df)
        # All blocked features should be identified
        assert validator.results[-1].passed  # They are in dataset but blocked

    def test_blocked_features_count(self, valid_df):
        from recoverytwin.data.validator import BLOCKED_FEATURES
        blocked_in_df = BLOCKED_FEATURES & set(valid_df.columns)
        assert len(blocked_in_df) > 0  # Should have some blocked features
