"""
Tests for the causal payment simulator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from recoverytwin.simulator.payment_environment import PaymentEnvironment


@pytest.fixture
def small_env():
    """Small environment for fast tests."""
    return PaymentEnvironment(seed=42)


@pytest.fixture
def small_data(small_env):
    """Generate small dataset."""
    small_env.config["n_transactions"] = 1000
    small_env.config["n_customers"] = 200
    small_env.config["n_merchants"] = 50
    df, meta = small_env.generate()
    return df, meta


class TestSimulator:
    def test_generate_returns_dataframe(self, small_data):
        df, meta = small_data
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_metadata_has_required_fields(self, small_data):
        _, meta = small_data
        assert "n_customers" in meta
        assert "n_merchants" in meta
        assert "n_transactions" in meta
        assert "recovery_rate" in meta
        assert "treatment_distribution" in meta
        assert "revenue_at_risk" in meta

    def test_valid_treatment_ids(self, small_data):
        df, _ = small_data
        assert df["intervention"].isin([0, 1, 2, 3]).all()

    def test_all_treatments_present(self, small_data):
        df, _ = small_data
        treatments = set(df["intervention"].unique())
        assert treatments == {0, 1, 2, 3}

    def test_recovery_binary(self, small_data):
        df, _ = small_data
        assert df["recovered"].isin([0, 1]).all()

    def test_no_negative_revenue(self, small_data):
        df, _ = small_data
        assert (df["revenue_recovered"] >= 0).all()

    def test_revenue_consistency(self, small_data):
        df, _ = small_data
        recovered = df["recovered"] == 1
        unrecovered = df["recovered"] == 0
        assert (df.loc[recovered, "revenue_recovered"] <= df.loc[recovered, "amount"]).all()
        assert (df.loc[unrecovered, "revenue_recovered"] == 0).all()

    def test_recovery_time_positive_when_recovered(self, small_data):
        df, _ = small_data
        recovered = df["recovered"] == 1
        if recovered.any():
            assert (df.loc[recovered, "recovery_time_hours"] > 0).all()

    def test_recovery_time_zero_when_not_recovered(self, small_data):
        df, _ = small_data
        unrecovered = df["recovered"] == 0
        assert (df.loc[unrecovered, "recovery_time_hours"] == 0).all()

    def test_intervention_cost_non_negative(self, small_data):
        df, _ = small_data
        assert (df["intervention_cost"] >= 0).all()

    def test_propensity_columns_exist(self, small_data):
        df, _ = small_data
        for i in range(4):
            assert f"propensity_{i}" in df.columns

    def test_propensity_in_range(self, small_data):
        df, _ = small_data
        for i in range(4):
            col = f"propensity_{i}"
            assert (df[col] >= 0).all() and (df[col] <= 1).all()

    def test_propensity_sum_to_one(self, small_data):
        df, _ = small_data
        prop_sum = sum(df[f"propensity_{i}"] for i in range(4))
        assert np.allclose(prop_sum.values, 1.0, atol=0.01)

    def test_hidden_columns_exist(self, small_data):
        df, _ = small_data
        for i in range(4):
            assert f"potential_outcome_{i}" in df.columns
        assert "true_best_intervention" in df.columns

    def test_heterogeneous_treatment_effects(self, small_data):
        """Recovery rate should vary by treatment."""
        df, _ = small_data
        recovery_by_treatment = df.groupby("intervention")["recovered"].mean()
        # All treatments should have some recovery
        assert all(r > 0 for r in recovery_by_treatment.values)
        # Recovery rates should differ across treatments (heterogeneity)
        rates = recovery_by_treatment.values
        assert rates.max() - rates.min() > 0.01

    def test_reproducibility(self):
        """Same seed should produce same data."""
        env1 = PaymentEnvironment(seed=123)
        env1.config["n_transactions"] = 100
        env1.config["n_customers"] = 20
        env1.config["n_merchants"] = 10
        df1, _ = env1.generate()

        env2 = PaymentEnvironment(seed=123)
        env2.config["n_transactions"] = 100
        env2.config["n_customers"] = 20
        env2.config["n_merchants"] = 10
        df2, _ = env2.generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_financial_validation(self, small_data):
        df, _ = small_data
        env = PaymentEnvironment(seed=42)
        errors = env.validate_financial(df)
        assert len(errors) == 0
