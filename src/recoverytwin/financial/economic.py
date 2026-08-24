"""
RecoveryTwin Financial Policy Simulator — Core Economic Model.

Computes financial metrics for payment recovery policies:
  - Expected revenue per payment
  - Net value after intervention costs
  - Time-adjusted recovery value
  - Merchant margin-adjusted profit
  - ROI calculations
  - Policy comparison against baselines and oracle

All monetary values are synthetic (INR).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import yaml


def load_financial_config(path: str = "configs/financial.yaml") -> Dict:
    """Load financial configuration."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f)


def compute_expected_revenue(
    recovery_probs: np.ndarray,
    amounts: np.ndarray,
    intervention_cost: float = 0.0,
) -> np.ndarray:
    """
    Expected net revenue for each payment under a given action.

    EV(a) = P(recovery | X, a) * amount - cost(a)

    Parameters
    ----------
    recovery_probs : (n,) array of P(recovery | X, a)
    amounts        : (n,) array of payment amounts
    intervention_cost : scalar cost of this intervention

    Returns
    -------
    (n,) array of expected net revenue per payment
    """
    return recovery_probs * amounts - intervention_cost


def compute_incremental_revenue(
    recovery_probs: np.ndarray,
    control_probs: np.ndarray,
    amounts: np.ndarray,
    intervention_cost: float = 0.0,
) -> np.ndarray:
    """
    Incremental revenue of an intervention over control.

    IncrementalRevenue(a) = [P(Y|X,a) - P(Y|X,0)] * amount - cost(a)
    """
    cate = recovery_probs - control_probs
    return amounts * cate - intervention_cost


def compute_time_discount(
    expected_hours: np.ndarray,
    lambda_val: float = 0.0,
    half_life: float = 24.0,
) -> np.ndarray:
    """
    Time discount factor for recovery.

    Two parameterizations:
      1. Exponential: discount = exp(-lambda * t)
      2. Half-life:   discount = exp(-0.693 * t / half_life)

    If lambda_val > 0, uses exponential decay.
    Otherwise uses half_life.
    """
    if lambda_val > 0:
        return np.exp(-lambda_val * expected_hours)
    return np.exp(-0.693 * expected_hours / max(half_life, 0.1))


def compute_merchant_profit(
    revenue: np.ndarray,
    margin: float,
    intervention_cost: float,
) -> np.ndarray:
    """
    Expected merchant profit from recovered payment.

    MerchantProfit = revenue * margin - intervention_cost
    """
    return revenue * margin - intervention_cost


class FinancialEvaluator:
    """
    Evaluates financial performance of a policy on a batch of payments.

    For each payment i and action a:
      - recovery_prob[i, a] = P(Y=1 | X_i, a)
      - expected_revenue[i, a] = recovery_prob[i, a] * amount[i] - cost[a]
      - incremental[i, a] = expected_revenue[i, a] - expected_revenue[i, 0]
      - time_adjusted[i, a] = expected_revenue[i, a] * discount(t_i, a)

    After action selection:
      - total_recovered = sum(expected_revenue[i, selected_a])
      - total_cost = sum(cost[selected_a] for each payment)
      - net_value = total_recovered
      - incremental_value = net_value - do_nothing_value
      - roi = incremental_value / total_cost
    """

    def __init__(
        self,
        intervention_costs: Dict[int, float] = None,
        merchant_margin: float = 0.015,
    ):
        self.costs = intervention_costs or {0: 0.0, 1: 0.50, 2: 1.00, 3: 2.50}
        self.merchant_margin = merchant_margin

    def evaluate_policy(
        self,
        df: pd.DataFrame,
        decision_table: pd.DataFrame,
        action_costs: Dict[int, float] = None,
        time_discount_lambda: float = 0.0,
        time_discount_half_life: float = 24.0,
        degradation_factor: float = 0.0,
        treatment_effect_multiplier: float = 1.0,
        recovery_rate_multiplier: float = 1.0,
        amount_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluate financial performance of the RecoveryTwin policy.

        Returns dict with:
          - net_revenue
          - incremental_revenue
          - recovery_rate
          - intervention_rate
          - roi
          - cost_per_recovered_payment
          - revenue_per_payment
          - action_distribution
          - recovery_by_action
          - avg_recovery_time
        """
        costs = action_costs or self.costs
        n = len(df)
        amounts = df["amount"].values * amount_multiplier

        # Get recommended actions
        if "recommended_action" in decision_table.columns:
            actions = decision_table["recommended_action"].values
        else:
            actions = np.zeros(n, dtype=int)

        # Get true potential outcomes for evaluation
        true_po = np.column_stack([
            df[f"potential_outcome_{a}"].values.astype(float)
            for a in range(4)
        ])

        # Apply scenario modifiers
        modified_po = self._apply_scenario_modifiers(
            true_po, actions, df, costs,
            degradation_factor=degradation_factor,
            treatment_effect_multiplier=treatment_effect_multiplier,
            recovery_rate_multiplier=recovery_rate_multiplier,
        )

        # Financial calculations
        action_costs_arr = np.array([costs.get(int(a), 0.0) for a in actions])

        # Recovery probability under selected action
        recovery_probs = np.array([
            modified_po[i, int(actions[i])] for i in range(n)
        ])

        # Actual outcomes (binary 0/1)
        actual_outcomes = np.array([
            modified_po[i, int(actions[i])] for i in range(n)
        ])

        # Revenue
        gross_revenue = actual_outcomes * amounts
        total_cost = action_costs_arr.sum()
        net_revenue = gross_revenue.sum()
        total_intervention_cost = total_cost

        # Do-nothing baseline
        dn_outcomes = true_po[:, 0]
        dn_revenue = dn_outcomes * amounts

        # Incremental
        incremental_revenue = net_revenue - dn_revenue.sum()

        # Rates
        recovery_rate = actual_outcomes.mean()
        intervention_rate = (actions > 0).mean()

        # ROI (incremental revenue / total intervention cost)
        roi = incremental_revenue / max(total_intervention_cost, 1e-8)

        # Cost per recovered
        n_recovered = actual_outcomes.sum()
        cost_per_recovered = total_intervention_cost / max(n_recovered, 1e-8)

        # Revenue per payment
        revenue_per_payment = net_revenue / max(n, 1)

        # Action distribution
        action_dist = {
            int(a): float(np.mean(actions == a))
            for a in range(4)
        }

        # Recovery by action
        recovery_by_action = {}
        for a in range(4):
            mask = actions == a
            if mask.sum() > 0:
                recovery_by_action[int(a)] = {
                    "count": int(mask.sum()),
                    "recovery_rate": float(modified_po[mask, a].mean()),
                    "avg_amount": float(amounts[mask].mean()),
                    "total_revenue": float((actual_outcomes[mask] * amounts[mask]).sum()),
                }

        return {
            "net_revenue": float(net_revenue),
            "incremental_revenue": float(incremental_revenue),
            "total_intervention_cost": float(total_intervention_cost),
            "recovery_rate": float(recovery_rate),
            "intervention_rate": float(intervention_rate),
            "roi": float(roi),
            "cost_per_recovered_payment": float(cost_per_recovered),
            "recovery_per_payment": float(revenue_per_payment),
            "n_payments": int(n),
            "n_recovered": int(n_recovered),
            "n_intervened": int((actions > 0).sum()),
            "action_distribution": action_dist,
            "recovery_by_action": recovery_by_action,
        }

    def _apply_scenario_modifiers(
        self,
        true_po: np.ndarray,
        actions: np.ndarray,
        df: pd.DataFrame,
        costs: Dict[int, float],
        degradation_factor: float = 0.0,
        treatment_effect_multiplier: float = 1.0,
        recovery_rate_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Apply scenario modifiers to potential outcomes.

        Modifies:
          - Recovery rates (multiplier on baseline P(recovery))
          - Treatment effects (multiplier on incremental P(recovery|treatment) vs control)
          - Degradation (uniform degradation across all treatments)
        """
        n = true_po.shape[0]
        modified = true_po.copy()

        if recovery_rate_multiplier != 1.0:
            # Scale baseline (control) recovery rate
            modified[:, 0] = np.clip(
                modified[:, 0] * recovery_rate_multiplier, 0, 1
            )

        if treatment_effect_multiplier != 1.0 or degradation_factor > 0:
            # Scale treatment effects relative to control
            control_outcome = modified[:, 0]
            for a in range(1, 4):
                treatment_effect = modified[:, a] - control_outcome
                if treatment_effect_multiplier != 1.0:
                    treatment_effect = treatment_effect * treatment_effect_multiplier
                if degradation_factor > 0:
                    treatment_effect = treatment_effect * (1.0 - degradation_factor)
                modified[:, a] = np.clip(
                    control_outcome + treatment_effect, 0, 1
                )

        return modified

    def evaluate_comparison_policies(
        self,
        df: pd.DataFrame,
        decision_table: pd.DataFrame,
        action_costs: Dict[int, float] = None,
        **scenario_kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple policies for comparison:
          - do_nothing
          - always_retry
          - max_probability
          - recoverytwin
          - oracle
        """
        costs = action_costs or self.costs
        n = len(df)
        amounts = df["amount"].values * scenario_kwargs.get("amount_multiplier", 1.0)

        # True potential outcomes
        true_po = np.column_stack([
            df[f"potential_outcome_{a}"].values.astype(float)
            for a in range(4)
        ])

        # Apply scenario modifiers
        modified_po = self._apply_scenario_modifiers(
            true_po, np.zeros(n, dtype=int), df, costs,
            degradation_factor=scenario_kwargs.get("degradation_factor", 0.0),
            treatment_effect_multiplier=scenario_kwargs.get("treatment_effect_multiplier", 1.0),
            recovery_rate_multiplier=scenario_kwargs.get("recovery_rate_multiplier", 1.0),
        )

        policies = {}

        # 1. Do Nothing
        dn_outcomes = modified_po[:, 0]
        dn_revenue = (dn_outcomes * amounts).sum()
        policies["do_nothing"] = {
            "actions": np.zeros(n, dtype=int),
            "net_revenue": float(dn_revenue),
            "recovery_rate": float(dn_outcomes.mean()),
            "n_interventions": 0,
            "total_cost": 0.0,
        }

        # 2. Always Retry
        rt_outcomes = modified_po[:, 1]
        rt_cost = costs.get(1, 0.50) * n
        rt_revenue = (rt_outcomes * amounts).sum()
        policies["always_retry"] = {
            "actions": np.ones(n, dtype=int),
            "net_revenue": float(rt_revenue - rt_cost),
            "recovery_rate": float(rt_outcomes.mean()),
            "n_interventions": n,
            "total_cost": float(rt_cost),
        }

        # 3. Max Probability
        if all(f"control_prob" in decision_table.columns for _ in [1]):
            prob_cols = ["control_prob"] + [f"{['','retry','reminder','alternative_method'][a]}_prob" for a in range(1, 4)]
            pred_probs = np.column_stack([decision_table[c].values for c in prob_cols])
            max_prob_actions = np.argmax(pred_probs, axis=1)
        else:
            max_prob_actions = np.ones(n, dtype=int)

        mp_outcomes = np.array([modified_po[i, int(max_prob_actions[i])] for i in range(n)])
        mp_cost_arr = np.array([costs.get(int(max_prob_actions[i]), 0) for i in range(n)])
        mp_revenue = (mp_outcomes * amounts).sum()
        mp_cost_total = mp_cost_arr.sum()
        policies["max_probability"] = {
            "actions": max_prob_actions,
            "net_revenue": float(mp_revenue - mp_cost_total),
            "recovery_rate": float(mp_outcomes.mean()),
            "n_interventions": int((max_prob_actions > 0).sum()),
            "total_cost": float(mp_cost_total),
        }

        # 4. RecoveryTwin
        if "recommended_action" in decision_table.columns:
            rt_actions = decision_table["recommended_action"].values
        else:
            rt_actions = np.zeros(n, dtype=int)

        # Use the already-evaluated financials from the decision table if available
        if "recommended_value" in decision_table.columns:
            rt_net = decision_table["recommended_value"].sum()
        else:
            rt_out = np.array([modified_po[i, int(rt_actions[i])] for i in range(n)])
            rt_cost_arr = np.array([costs.get(int(rt_actions[i]), 0) for i in range(n)])
            rt_net = (rt_out * amounts).sum() - rt_cost_arr.sum()

        policies["recoverytwin"] = {
            "actions": rt_actions,
            "net_revenue": float(rt_net),
            "recovery_rate": float(np.mean([modified_po[i, int(rt_actions[i])] for i in range(n)])),
            "n_interventions": int((rt_actions > 0).sum()),
            "total_cost": float(sum(costs.get(int(rt_actions[i]), 0) for i in range(n))),
        }

        # 5. Oracle
        oracle_actions = np.argmax(modified_po * amounts[:, None], axis=1)
        oracle_out = np.array([modified_po[i, int(oracle_actions[i])] for i in range(n)])
        oracle_cost_arr = np.array([costs.get(int(oracle_actions[i]), 0) for i in range(n)])
        oracle_revenue = (oracle_out * amounts).sum()
        oracle_cost_total = oracle_cost_arr.sum()
        policies["oracle"] = {
            "actions": oracle_actions,
            "net_revenue": float(oracle_revenue - oracle_cost_total),
            "recovery_rate": float(oracle_out.mean()),
            "n_interventions": int((oracle_actions > 0).sum()),
            "total_cost": float(oracle_cost_total),
        }

        return policies
