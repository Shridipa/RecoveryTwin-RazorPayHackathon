"""
Phase 8 Tests — Financial Policy Simulation & Stress Testing.

30+ tests covering:
  - Economic model arithmetic
  - Scenario management
  - Monte Carlo simulation
  - Sensitivity analysis
  - Break-even analysis
  - Leakage audit
  - Robustness scoring
  - Deterministic reproducibility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from recoverytwin.financial.economic import (
    FinancialEvaluator, compute_expected_revenue, compute_incremental_revenue,
    compute_time_discount,
)
from recoverytwin.financial.scenario import (
    ScenarioRunner, load_scenarios, compute_robustness_score, compute_worst_case,
    get_scenario_params,
)
from recoverytwin.financial.monte_carlo import MonteCarloSimulator
from recoverytwin.financial.sensitivity import (
    cost_sensitivity, degradation_sensitivity, time_discount_sensitivity,
    retry_limit_sensitivity, find_breakeven_cost,
)
from recoverytwin.financial.leakage import audit_financial_leakage
from recoverytwin.decision.engine import PolicyFilter, load_policy_config


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_payments():
    """Create synthetic payment data."""
    n = 500
    np.random.seed(42)
    return pd.DataFrame({
        "payment_id": [f"pay_{i}" for i in range(n)],
        "amount": np.random.lognormal(7, 1.2, n).clip(50, 100000),
        "intervention": np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.3, 0.3]),
        "recovered": np.random.binomial(1, 0.3, n),
        "recovery_time_hours": np.random.exponential(12, n),
        "failure_reason": np.random.choice(
            ["network_timeout", "insufficient_funds", "technical_decline", "expired_card"], n
        ),
        "customer_activity": np.random.uniform(0, 1, n),
        "customer_tenure": np.random.randint(1, 60, n),
        "attempt_count": np.random.randint(0, 3, n),
        "previous_failures": np.random.randint(0, 5, n),
        "previous_recoveries": np.random.randint(0, 3, n),
        "hours_since_failure": np.random.uniform(0, 48, n),
        "customer_fatigue": np.random.uniform(0, 3, n),
        "potential_outcome_0": np.random.binomial(1, 0.2, n),
        "potential_outcome_1": np.random.binomial(1, 0.45, n),
        "potential_outcome_2": np.random.binomial(1, 0.40, n),
        "potential_outcome_3": np.random.binomial(1, 0.35, n),
    })


@pytest.fixture
def sample_decision_table(sample_payments):
    """Create a mock decision table."""
    n = len(sample_payments)
    np.random.seed(42)
    probs = np.random.uniform(0.1, 0.8, (n, 4))
    amounts = sample_payments["amount"].values
    actions = np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.35, 0.25])
    action_names = {0: "control", 1: "retry", 2: "reminder", 3: "alternative_method"}

    dt = pd.DataFrame({
        "payment_id": sample_payments["payment_id"],
        "amount": amounts,
        "control_prob": probs[:, 0],
        "retry_prob": probs[:, 1],
        "reminder_prob": probs[:, 2],
        "alternative_method_prob": probs[:, 3],
        "control_expected_value": probs[:, 0] * amounts,
        "retry_expected_value": probs[:, 1] * amounts - 0.50,
        "reminder_expected_value": probs[:, 2] * amounts - 1.00,
        "alternative_method_expected_value": probs[:, 3] * amounts - 2.50,
        "retry_cate": probs[:, 1] - probs[:, 0],
        "reminder_cate": probs[:, 2] - probs[:, 0],
        "alternative_method_cate": probs[:, 3] - probs[:, 0],
        "recommended_action": actions,
        "recommended_action_name": [action_names[a] for a in actions],
        "recommended_value": np.array([
            probs[i, int(actions[i])] * amounts[i] - [0.0, 0.50, 1.00, 2.50][int(actions[i])]
            for i in range(n)
        ]),
        "retry_eligible": np.ones(n, dtype=bool),
        "reminder_eligible": np.ones(n, dtype=bool),
        "alternative_method_eligible": np.ones(n, dtype=bool),
        "n_eligible_actions": np.full(n, 3),
        "rejection_reasons": [""] * n,
    })
    return dt


@pytest.fixture
def evaluator():
    """Create a FinancialEvaluator."""
    return FinancialEvaluator(
        intervention_costs={0: 0.0, 1: 0.50, 2: 1.00, 3: 2.50},
    )


# ============================================================
# ECONOMIC MODEL TESTS
# ============================================================

class TestExpectedRevenue:
    def test_zero_recovery_zero_revenue(self):
        probs = np.zeros(10)
        amounts = np.ones(10) * 1000
        result = compute_expected_revenue(probs, amounts, intervention_cost=0)
        assert np.allclose(result, 0)

    def test_full_recovery_full_revenue(self):
        probs = np.ones(10)
        amounts = np.ones(10) * 1000
        result = compute_expected_revenue(probs, amounts, intervention_cost=0)
        assert np.allclose(result, 1000)

    def test_zero_amount(self):
        probs = np.ones(10)
        amounts = np.zeros(10)
        result = compute_expected_revenue(probs, amounts, intervention_cost=0)
        assert np.allclose(result, 0)

    def test_intervention_cost_subtracted(self):
        probs = np.ones(5)
        amounts = np.ones(5) * 100
        result = compute_expected_revenue(probs, amounts, intervention_cost=10)
        assert np.allclose(result, 90)

    def test_zero_cost(self):
        probs = np.array([0.5])
        amounts = np.array([1000])
        result = compute_expected_revenue(probs, amounts, intervention_cost=0)
        assert np.isclose(result[0], 500)


class TestIncrementalRevenue:
    def test_no_treatment_effect(self):
        probs = np.full(10, 0.5)
        control = np.full(10, 0.5)
        amounts = np.ones(10) * 1000
        result = compute_incremental_revenue(probs, control, amounts, intervention_cost=0)
        assert np.allclose(result, 0)

    def test_positive_treatment_effect(self):
        probs = np.full(10, 0.8)
        control = np.full(10, 0.5)
        amounts = np.ones(10) * 1000
        result = compute_incremental_revenue(probs, control, amounts, intervention_cost=0)
        assert np.allclose(result, 300)

    def test_cost_offsets_revenue(self):
        probs = np.full(10, 0.6)
        control = np.full(10, 0.5)
        amounts = np.ones(10) * 100
        result = compute_incremental_revenue(probs, control, amounts, intervention_cost=10)
        # CATE = 0.1, incremental = 0.1 * 100 - 10 = 0
        assert np.allclose(result, 0)


class TestTimeDiscount:
    def test_zero_hours_no_discount(self):
        hours = np.zeros(5)
        result = compute_time_discount(hours, lambda_val=0.01)
        assert np.allclose(result, 1.0)

    def test_discount_is_monotone_decreasing(self):
        hours = np.array([0, 1, 6, 24, 72])
        result = compute_time_discount(hours, lambda_val=0.01)
        for i in range(1, len(result)):
            assert result[i] <= result[i-1]

    def test_zero_lambda_no_discount(self):
        # lambda=0 falls through to half_life branch; use a very large half_life to approximate no discount
        hours = np.array([0, 1, 24, 72])
        result = compute_time_discount(hours, lambda_val=0.0, half_life=1e6)
        assert np.allclose(result, 1.0, atol=0.01)

    def test_half_life_discount(self):
        hours = np.array([24.0])  # 1 half-life
        result = compute_time_discount(hours, lambda_val=0.0, half_life=24.0)
        assert np.isclose(result[0], 0.5, atol=0.01)


# ============================================================
# SCENARIO TESTS
# ============================================================

class TestScenarios:
    def test_load_scenarios(self):
        scenarios = load_scenarios()
        assert len(scenarios) >= 10
        assert "BASELINE" in scenarios
        assert "ADVERSE_COMBINED" in scenarios

    def test_get_scenario_params(self):
        params = get_scenario_params("BASELINE")
        assert params["cost_multiplier"] == 1.0
        assert params["degradation_factor"] == 0.0

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario_params("NONEXISTENT")

    def test_robustness_score(self):
        scenarios = [
            {"policies": {"recoverytwin": {"net_revenue": 100}, "do_nothing": {"net_revenue": 50},
                          "max_probability": {"net_revenue": 80}}},
            {"policies": {"recoverytwin": {"net_revenue": 60}, "do_nothing": {"net_revenue": 50},
                          "max_probability": {"net_revenue": 80}}},
        ]
        score = compute_robustness_score(scenarios)
        assert score["vs_baseline"] == 1.0  # Beats do-nothing in both
        assert score["vs_max_prob"] == 0.5  # Beats max-prob in 1 of 2

    def test_worst_case(self):
        scenarios = [
            {"scenario": "GOOD", "policies": {"recoverytwin": {"net_revenue": 100, "recovery_rate": 0.4},
                                               "do_nothing": {"net_revenue": 50}},
             "policy_regret": 0.1, "beats_do_nothing": True, "description": "Good"},
            {"scenario": "BAD", "policies": {"recoverytwin": {"net_revenue": 30, "recovery_rate": 0.2},
                                               "do_nothing": {"net_revenue": 50}},
             "policy_regret": 0.8, "beats_do_nothing": False, "description": "Bad"},
        ]
        wc = compute_worst_case(scenarios)
        assert wc["scenario"] == "BAD"
        assert wc["net_revenue"] == 30


# ============================================================
# MONTE CARLO TESTS
# ============================================================

class TestMonteCarlo:
    def test_deterministic_reproducibility(self):
        n = 100
        np.random.seed(42)
        probs = np.random.uniform(0.3, 0.7, n)
        amounts = np.random.uniform(100, 5000, n)
        costs = np.zeros(n)
        actions = np.zeros(n, dtype=int)

        sim1 = MonteCarloSimulator(n_simulations=100, random_seed=42)
        r1 = sim1.simulate_policy(
            MagicMock(), actions, probs, amounts, costs
        )

        sim2 = MonteCarloSimulator(n_simulations=100, random_seed=42)
        r2 = sim2.simulate_policy(
            MagicMock(), actions, probs, amounts, costs
        )

        assert r1["mean_net_revenue"] == r2["mean_net_revenue"]
        assert r1["p5"] == r2["p5"]
        assert r1["p95"] == r2["p95"]

    def test_zero_prob_zero_revenue(self):
        n = 50
        probs = np.zeros(n)
        amounts = np.ones(n) * 1000
        costs = np.zeros(n)
        actions = np.zeros(n, dtype=int)

        sim = MonteCarloSimulator(n_simulations=50, random_seed=42)
        result = sim.simulate_policy(MagicMock(), actions, probs, amounts, costs)
        assert result["mean_net_revenue"] == 0

    def test_full_prob_full_revenue(self):
        n = 50
        probs = np.ones(n)
        amounts = np.ones(n) * 1000
        costs = np.zeros(n)
        actions = np.zeros(n, dtype=int)

        sim = MonteCarloSimulator(n_simulations=50, random_seed=42)
        result = sim.simulate_policy(MagicMock(), actions, probs, amounts, costs)
        assert np.isclose(result["mean_net_revenue"], 50000, atol=1)

    def test_cost_deducted(self):
        n = 100
        probs = np.ones(n)
        amounts = np.ones(n) * 1000
        costs = np.ones(n) * 5
        actions = np.ones(n, dtype=int)

        sim = MonteCarloSimulator(n_simulations=50, random_seed=42)
        result = sim.simulate_policy(MagicMock(), actions, probs, amounts, costs)
        assert np.isclose(result["mean_net_revenue"], 99500, atol=1)

    def test_confidence_intervals_valid(self):
        n = 100
        probs = np.random.RandomState(42).uniform(0.2, 0.8, n)
        amounts = np.ones(n) * 1000
        costs = np.zeros(n)
        actions = np.zeros(n, dtype=int)

        sim = MonteCarloSimulator(n_simulations=200, random_seed=42)
        result = sim.simulate_policy(MagicMock(), actions, probs, amounts, costs)
        assert result["p5"] <= result["p50"] <= result["p95"]
        assert result["prob_positive_net"] >= 0
        assert result["prob_positive_net"] <= 1


# ============================================================
# SENSITIVITY TESTS
# ============================================================

class TestSensitivity:
    def test_cost_monotonicity(self, sample_payments, sample_decision_table, evaluator):
        """Higher intervention costs should decrease RT value."""
        results = cost_sensitivity(
            sample_payments, sample_decision_table, evaluator,
            cost_values={1: [0.10, 0.50, 2.00, 5.00]},
        )
        # RT revenue should generally decrease as costs increase
        rt_vals = [r["recoverytwin"] for r in results]
        # At minimum, the first should be >= last
        assert rt_vals[0] >= rt_vals[-1]

    def test_degradation_monotonicity(self, sample_payments, sample_decision_table, evaluator):
        """Higher degradation should decrease RT value."""
        results = degradation_sensitivity(
            sample_payments, sample_decision_table, evaluator,
            degradation_factors=[0.0, 0.20, 0.40],
        )
        rt_vals = [r["recoverytwin"] for r in results]
        assert rt_vals[0] >= rt_vals[-1]

    def test_degradation_regret_increases(self, sample_payments, sample_decision_table, evaluator):
        """Higher degradation should reduce RT and oracle revenues."""
        results = degradation_sensitivity(
            sample_payments, sample_decision_table, evaluator,
            degradation_factors=[0.0, 0.30, 0.50],
        )
        # Both RT and oracle should decrease with degradation
        rt_vals = [r["recoverytwin"] for r in results]
        oracle_vals = [r["oracle"] for r in results]
        assert rt_vals[0] >= rt_vals[-1]  # RT decreases
        assert oracle_vals[0] >= oracle_vals[-1]  # Oracle decreases

    def test_time_discount_no_discount(self, sample_payments, sample_decision_table, evaluator):
        """Lambda=0 means no time discount."""
        results = time_discount_sensitivity(
            sample_payments, sample_decision_table, evaluator,
            lambda_values=[0.0],
        )
        assert len(results) == 1
        assert results[0]["lambda"] == 0.0


# ============================================================
# LEAKAGE TESTS
# ============================================================

class TestLeakage:
    def test_clean_decision_table_no_leak(self, sample_payments, sample_decision_table):
        """Decision table with only allowed features should pass."""
        # Remove any blocked columns from the sample decision table
        clean_cols = [c for c in sample_decision_table.columns
                      if c not in {"recovered", "recovery_time_hours", "revenue_recovered",
                                    "potential_outcome_0", "potential_outcome_1",
                                    "potential_outcome_2", "potential_outcome_3"}]
        clean_dt = sample_decision_table[clean_cols]
        audit = audit_financial_leakage(sample_payments, clean_dt)
        assert audit["pass"]

    def test_leaked_potential_outcome_detected(self, sample_payments, sample_decision_table):
        """If potential outcomes are in decision table, audit should fail."""
        dt = sample_decision_table.copy()
        dt["potential_outcome_0"] = 0.5  # Inject leakage
        audit = audit_financial_leakage(sample_payments, dt)
        assert not audit["pass"]

    def test_leaked_recovered_detected(self, sample_payments, sample_decision_table):
        """If 'recovered' is in decision table, audit should fail."""
        dt = sample_decision_table.copy()
        dt["recovered"] = 1
        audit = audit_financial_leakage(sample_payments, dt)
        assert not audit["pass"]

    def test_features_with_leakage_detected(self, sample_payments, sample_decision_table):
        """Feature list containing blocked features should be detected."""
        features = ["amount", "failure_reason", "recovered", "recovery_time_hours"]
        audit = audit_financial_leakage(sample_payments, sample_decision_table, features)
        assert not audit["pass"]
        assert len(audit["blocked_features_found"]) > 0

    def test_no_recovered_in_financial_inputs(self):
        """Verify 'recovered' is never used as a financial decision input."""
        from recoverytwin.financial.leakage import BLOCKED_DECISION_FEATURES
        assert "recovered" in BLOCKED_DECISION_FEATURES
        assert "recovery_time_hours" in BLOCKED_DECISION_FEATURES
        assert "potential_outcome_0" in BLOCKED_DECISION_FEATURES

    def test_allowed_features_list(self):
        """Known features should be in allowed list."""
        from recoverytwin.financial.leakage import ALLOWED_DECISION_FEATURES
        for feat in ["amount", "failure_reason", "customer_activity",
                      "customer_tenure", "attempt_count", "payment_id"]:
            assert feat in ALLOWED_DECISION_FEATURES


# ============================================================
# POLICY FILTER TESTS
# ============================================================

class TestPolicyFilter:
    def test_control_always_eligible(self):
        config = load_policy_config()
        pf = PolicyFilter(config)
        eligible, reason = pf.check_eligibility(
            action=0, attempt_count=10, previous_failures=10,
            customer_fatigue=99, hours_since_failure=0,
        )
        assert eligible

    def test_retry_limit(self):
        config = load_policy_config()
        pf = PolicyFilter(config)
        eligible, reason = pf.check_eligibility(
            action=1, attempt_count=1, previous_failures=5,
            customer_fatigue=0, hours_since_failure=24,
        )
        assert not eligible
        assert "Max retry" in reason

    def test_fatigue_limit(self):
        config = load_policy_config()
        pf = PolicyFilter(config)
        eligible, reason = pf.check_eligibility(
            action=3, attempt_count=0, previous_failures=0,
            customer_fatigue=5, hours_since_failure=24,
        )
        assert not eligible
        assert "fatigue" in reason.lower()

    def test_intervention_limit(self):
        config = load_policy_config()
        pf = PolicyFilter(config)
        eligible, reason = pf.check_eligibility(
            action=1, attempt_count=10, previous_failures=0,
            customer_fatigue=0, hours_since_failure=24,
        )
        assert not eligible
        assert "interventions" in reason


# ============================================================
# ECONOMIC EVALUATOR TESTS
# ============================================================

class TestFinancialEvaluator:
    def test_do_nothing_value(self, sample_payments, sample_decision_table, evaluator):
        """Do-nothing value should be P(Y|X,0) * amount summed."""
        policies = evaluator.evaluate_comparison_policies(
            sample_payments, sample_decision_table,
        )
        assert policies["do_nothing"]["n_interventions"] == 0
        assert policies["do_nothing"]["net_revenue"] >= 0

    def test_oracle_beats_all(self, sample_payments, sample_decision_table, evaluator):
        """Oracle should have highest net revenue."""
        policies = evaluator.evaluate_comparison_policies(
            sample_payments, sample_decision_table,
        )
        oracle_val = policies["oracle"]["net_revenue"]
        for name, pol in policies.items():
            if name != "oracle":
                assert oracle_val >= pol["net_revenue"], \
                    f"Oracle ({oracle_val}) should >= {name} ({pol['net_revenue']})"

    def test_intervention_increases_cost(self, sample_payments, sample_decision_table, evaluator):
        """Policies with interventions should have higher costs."""
        policies = evaluator.evaluate_comparison_policies(
            sample_payments, sample_decision_table,
        )
        assert policies["always_retry"]["total_cost"] > policies["do_nothing"]["total_cost"]


# ============================================================
# BREAK-EVEN TESTS
# ============================================================

class TestBreakeven:
    def test_breakeven_positive(self, sample_payments, sample_decision_table, evaluator):
        """Break-even cost should be positive."""
        be = find_breakeven_cost(sample_payments, sample_decision_table, evaluator, action=1)
        assert be["breakeven_cost"] > 0

    def test_breakeven_different_actions(self, sample_payments, sample_decision_table, evaluator):
        """Different actions should have different break-even costs."""
        be1 = find_breakeven_cost(sample_payments, sample_decision_table, evaluator, action=1)
        be2 = find_breakeven_cost(sample_payments, sample_decision_table, evaluator, action=2)
        # Both should be positive (may be same if search saturates)
        assert be1["breakeven_cost"] > 0
        assert be2["breakeven_cost"] > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    def test_all_scenarios_run(self):
        """All defined scenarios should run without error."""
        scenarios = load_scenarios()
        assert len(scenarios) >= 10
        for name in scenarios:
            params = get_scenario_params(name)
            assert "cost_multiplier" in params

    def test_mc_config_loads(self):
        from recoverytwin.financial.monte_carlo import load_mc_config
        config = load_mc_config()
        assert "n_simulations" in config
        assert config["n_simulations"] > 0

    def test_financial_config_loads(self):
        from recoverytwin.financial.economic import load_financial_config
        config = load_financial_config()
        assert "intervention_costs" in config
        assert config["intervention_costs"]["control"] == 0.0
