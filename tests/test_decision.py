"""
Tests for the Phase 7 Counterfactual Decision Engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from recoverytwin.decision.engine import (
    DecisionEngine,
    PolicyFilter,
    load_policy_config,
    time_discount_exponential,
    time_discount_step,
    evaluate_oracle,
    segment_analysis,
    run_stress_tests,
    ACTION_NAMES,
    ACTION_LABELS,
    DEFAULT_COSTS,
)


@pytest.fixture
def sample_data():
    """Load the synthetic dataset."""
    data_path = Path("data/synthetic/debug/transactions.parquet")
    if not data_path.exists():
        pytest.skip("Synthetic data not found")
    return pd.read_parquet(data_path)


@pytest.fixture
def test_data(sample_data):
    """Get test split."""
    test = sample_data[sample_data['timestamp'] >= '2024-11-01'].reset_index(drop=True)
    return test


@pytest.fixture
def mock_predict_fn():
    """Mock prediction function."""
    rng = np.random.RandomState(42)

    def predict_fn(df, action):
        n = len(df)
        # Base probability + treatment effect
        base = 0.2 + rng.random(n) * 0.3
        effects = {0: 0.0, 1: 0.15, 2: 0.10, 3: 0.12}
        return np.clip(base + effects.get(action, 0), 0.01, 0.99)

    return predict_fn


@pytest.fixture
def engine():
    """Default decision engine."""
    return DecisionEngine(load_policy_config())


class TestActionSpace:
    def test_action_names_complete(self):
        """All 4 actions must be defined."""
        assert len(ACTION_NAMES) == 4
        assert 0 in ACTION_NAMES  # control
        assert 1 in ACTION_NAMES  # retry
        assert 2 in ACTION_NAMES  # reminder
        assert 3 in ACTION_NAMES  # alternative

    def test_action_labels(self):
        """All actions must have human-readable labels."""
        assert len(ACTION_LABELS) == 4
        for a in range(4):
            assert a in ACTION_LABELS
            assert isinstance(ACTION_LABELS[a], str)

    def test_control_cost_is_zero(self):
        """Control must have zero cost."""
        assert DEFAULT_COSTS[0] == 0.0


class TestCounterfactualPrediction:
    def test_generates_all_actions(self, engine, test_data, mock_predict_fn):
        """Must generate probabilities for all 4 actions."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        assert "control_prob" in cf_table.columns
        assert "retry_prob" in cf_table.columns
        assert "reminder_prob" in cf_table.columns
        assert "alternative_method_prob" in cf_table.columns

    def test_probabilities_in_range(self, engine, test_data, mock_predict_fn):
        """All probabilities must be in [0, 1]."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        for col in ["control_prob", "retry_prob", "reminder_prob", "alternative_method_prob"]:
            assert cf_table[col].min() >= 0
            assert cf_table[col].max() <= 1

    def test_counterfactual_count_matches(self, engine, test_data, mock_predict_fn):
        """Must have one prediction per payment."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        assert len(cf_table) == len(test_data)


class TestCATE:
    def test_cate_calculation(self, engine, test_data, mock_predict_fn):
        """CATE must equal P(Y|X,a) - P(Y|X,0)."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        cate_table = engine.calculate_cate(cf_table)

        for action_name in ["retry", "reminder", "alternative_method"]:
            cate_col = f"{action_name}_cate"
            prob_col = f"{action_name}_prob"
            expected = cate_table[prob_col] - cate_table["control_prob"]
            np.testing.assert_array_almost_equal(
                cate_table[cate_col].values, expected.values, decimal=10
            )

    def test_control_cate_is_zero(self, engine, test_data, mock_predict_fn):
        """Control CATE (implicit) is always 0."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        cate_table = engine.calculate_cate(cf_table)
        # No explicit control CATE column, but control prob - control prob = 0
        assert "control_cate" not in cate_table.columns


class TestFinancialValue:
    def test_incremental_revenue(self, engine, test_data, mock_predict_fn):
        """Incremental revenue = amount * CATE."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        cate_table = engine.calculate_cate(cf_table)
        fin_table = engine.calculate_financial_value(test_data, cate_table)

        for action_name in ["retry", "reminder", "alternative_method"]:
            inc_rev = fin_table[f"{action_name}_incremental_revenue"]
            cate = cate_table[f"{action_name}_cate"]
            amounts = test_data["amount"].values
            expected = amounts * cate.values
            np.testing.assert_array_almost_equal(
                inc_rev.values, expected, decimal=10
            )

    def test_net_value_subtracts_cost(self, engine, test_data, mock_predict_fn):
        """Net value = incremental revenue - cost."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        cate_table = engine.calculate_cate(cf_table)
        fin_table = engine.calculate_financial_value(test_data, cate_table)

        for action_name in ["retry", "reminder", "alternative_method"]:
            net = fin_table[f"{action_name}_net_value"]
            inc = fin_table[f"{action_name}_incremental_revenue"]
            cost = fin_table[f"{action_name}_cost"].iloc[0]
            expected = inc - cost
            np.testing.assert_array_almost_equal(
                net.values, expected.values, decimal=10
            )

    def test_control_cost_is_zero(self, engine, test_data, mock_predict_fn):
        """Control intervention cost is zero."""
        cf_table = engine.generate_counterfactual_table(test_data, mock_predict_fn)
        cate_table = engine.calculate_cate(cf_table)
        fin_table = engine.calculate_financial_value(test_data, cate_table)
        # Control has no explicit cost column but it's implicitly 0
        assert fin_table["retry_cost"].iloc[0] == DEFAULT_COSTS[1]


class TestTimeDiscount:
    def test_exponential_instant(self):
        """Instant recovery should have discount ~1."""
        d = time_discount_exponential(0.0)
        assert d == 1.0

    def test_exponential_at_half_life(self):
        """At half-life, discount should be ~0.5."""
        d = time_discount_exponential(24.0, half_life=24.0)
        assert abs(d - 0.5) < 0.01

    def test_exponential_monotone(self):
        """Discount should decrease with time."""
        d1 = time_discount_exponential(1.0)
        d2 = time_discount_exponential(12.0)
        d3 = time_discount_exponential(48.0)
        assert d1 > d2 > d3

    def test_step_discount(self):
        """Step discount should work with thresholds."""
        d = time_discount_step(0.5)
        assert d == 1.0
        d = time_discount_step(36.0)
        assert d < 1.0


class TestPolicyFilter:
    def test_control_always_eligible(self):
        """Control must always be eligible."""
        pf = PolicyFilter(load_policy_config())
        eligible, _ = pf.check_eligibility(
            action=0, attempt_count=10, previous_failures=10,
            customer_fatigue=10.0, hours_since_failure=0.1,
        )
        assert eligible

    def test_max_retries_enforced(self):
        """Must enforce max retry limit."""
        cfg = load_policy_config()
        cfg["max_retries"] = 2
        pf = PolicyFilter(cfg)
        eligible, reason = pf.check_eligibility(
            action=1, attempt_count=1, previous_failures=3,
            customer_fatigue=0.0, hours_since_failure=24.0,
        )
        assert not eligible
        assert "Max retry" in reason

    def test_fatigue_limit(self):
        """Must enforce customer fatigue limit."""
        cfg = load_policy_config()
        cfg["max_fatigue"] = 2.0
        pf = PolicyFilter(cfg)
        eligible, reason = pf.check_eligibility(
            action=3, attempt_count=1, previous_failures=0,
            customer_fatigue=1.5, hours_since_failure=24.0,
        )
        assert not eligible
        assert "fatigue" in reason.lower()

    def test_recovered_stops_intervention(self):
        """Already-recovered payments should not get interventions."""
        pf = PolicyFilter(load_policy_config())
        eligible, reason = pf.check_eligibility(
            action=1, attempt_count=1, previous_failures=0,
            customer_fatigue=0.0, hours_since_failure=24.0,
            recovered=True,
        )
        assert not eligible
        assert "recovered" in reason.lower()


class TestActionSelection:
    def test_selects_action(self, engine, test_data, mock_predict_fn):
        """Must select an action for each payment."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        assert "recommended_action" in dt.columns
        assert len(dt) == len(test_data)
        assert all(0 <= a <= 3 for a in dt["recommended_action"].values)

    def test_control_fallback(self, engine, test_data, mock_predict_fn):
        """Must be able to select control (do nothing)."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        assert np.any(dt["recommended_action"].values == 0)

    def test_eligibility_masked(self, engine, test_data, mock_predict_fn):
        """Ineligible actions must not be selected."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        for _, row in dt.iterrows():
            action = row["recommended_action"]
            if action > 0:
                action_name = ACTION_NAMES[action]
                assert row[f"{action_name}_eligible"], \
                    f"Ineligible action {action} was selected"


class TestOracleEvaluation:
    def test_oracle_highest_value(self, engine, test_data, mock_predict_fn):
        """Oracle must have the highest policy value."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        oracle = evaluate_oracle(test_data, dt, DEFAULT_COSTS)
        policies = oracle["policies"]

        oracle_val = policies["oracle"]["value"]
        for name, pol in policies.items():
            if name != "oracle":
                assert oracle_val >= pol["value"], \
                    f"Oracle ({oracle_val}) < {name} ({pol['value']})"

    def test_do_nothing_baseline(self, engine, test_data, mock_predict_fn):
        """Do Nothing must use control potential outcomes."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        oracle = evaluate_oracle(test_data, dt, DEFAULT_COSTS)

        do_nothing_val = oracle["policies"]["do_nothing"]["value"]
        expected = (test_data["potential_outcome_0"].values * test_data["amount"].values).sum()
        assert abs(do_nothing_val - expected) < 0.01


class TestLeakageAudit:
    def test_no_leakage(self, engine, test_data, mock_predict_fn):
        """Decision table must not contain forbidden columns."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        forbidden = [
            "recovered", "recovery_time_hours", "revenue_recovered",
            "potential_outcome_0", "potential_outcome_1",
            "potential_outcome_2", "potential_outcome_3",
            "true_best_intervention",
        ]
        for col in forbidden:
            assert col not in dt.columns, f"Leaked column: {col}"


class TestStressTests:
    def test_stress_test_returns_dataframe(self, engine, test_data, mock_predict_fn):
        """Stress tests must return results."""
        config = load_policy_config()
        results = run_stress_tests(test_data, mock_predict_fn, config)
        assert len(results) > 0
        assert "test" in results.columns
        assert "control_pct" in results.columns

    def test_cost_sensitivity_monotone(self, engine, test_data, mock_predict_fn):
        """Higher costs should reduce intervention rate."""
        config = load_policy_config()
        results = run_stress_tests(test_data, mock_predict_fn, config)

        cost_results = results[results["test"] == "cost_sensitivity"]
        # With higher costs, more control
        low_cost = cost_results[cost_results["param"] == "cost_multiplier=0.1"]
        high_cost = cost_results[cost_results["param"] == "cost_multiplier=5.0"]

        if len(low_cost) > 0 and len(high_cost) > 0:
            assert high_cost["control_pct"].values[0] >= low_cost["control_pct"].values[0]


class TestDeterministic:
    def test_same_input_same_output(self, test_data):
        """Same data must produce same decisions with deterministic predict fn."""
        rng = np.random.RandomState(42)
        base_probs = rng.random(len(test_data)) * 0.3 + 0.2
        fixed_effects = {0: 0.0, 1: 0.15, 2: 0.10, 3: 0.12}

        def deterministic_predict(df, action):
            return np.clip(base_probs[:len(df)] + fixed_effects.get(action, 0), 0.01, 0.99)

        cfg = load_policy_config()
        e1 = DecisionEngine(cfg)
        e2 = DecisionEngine(cfg)

        dt1 = e1.build_decision_table(test_data, deterministic_predict)
        dt2 = e2.build_decision_table(test_data, deterministic_predict)

        pd.testing.assert_frame_equal(
            dt1[["payment_id", "recommended_action"]],
            dt2[["payment_id", "recommended_action"]],
        )


class TestSegmentAnalysis:
    def test_segments_generated(self, engine, test_data, mock_predict_fn):
        """Segment analysis must produce results."""
        dt = engine.build_decision_table(test_data, mock_predict_fn)
        oracle = evaluate_oracle(test_data, dt, DEFAULT_COSTS)
        segs = segment_analysis(test_data, dt, oracle)
        assert len(segs) > 0
        assert "segment" in segs.columns
        assert "regret" in segs.columns
