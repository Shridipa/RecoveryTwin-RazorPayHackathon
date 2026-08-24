"""
RecoveryTwin Financial Policy Simulator — Sensitivity Analysis.

Analyzes how policy performance changes under different parameter values:
  - Intervention cost sensitivity
  - Treatment effect degradation
  - Time discount sensitivity
  - Retry limit sensitivity
  - Fatigue threshold sensitivity
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml


def cost_sensitivity(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    evaluator,
    cost_values: Dict[int, List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate policy at different intervention cost levels.

    For each cost configuration, evaluate all comparison policies.
    """
    if cost_values is None:
        # Default: vary retry cost from 0 to 5
        cost_values = {
            1: [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00],
            2: [0.10, 0.25, 0.50, 1.00, 2.00],
            3: [0.25, 0.50, 1.00, 2.50, 5.00],
        }

    results = []

    # Get the retry cost range
    retry_costs = cost_values.get(1, [0.50])

    for rc in retry_costs:
        costs = {
            0: 0.0,
            1: rc,
            2: rc * 2.0,  # Reminder = 2x retry
            3: rc * 5.0,  # Alternative = 5x retry
        }

        policies = evaluator.evaluate_comparison_policies(
            df, decision_table, action_costs=costs,
        )

        results.append({
            "retry_cost": rc,
            "reminder_cost": costs[2],
            "alternative_cost": costs[3],
            "do_nothing": policies["do_nothing"]["net_revenue"],
            "always_retry": policies["always_retry"]["net_revenue"],
            "max_probability": policies["max_probability"]["net_revenue"],
            "recoverytwin": policies["recoverytwin"]["net_revenue"],
            "oracle": policies["oracle"]["net_revenue"],
            "rt_incremental": (
                policies["recoverytwin"]["net_revenue"]
                - policies["do_nothing"]["net_revenue"]
            ),
        })

    return results


def degradation_sensitivity(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    evaluator,
    degradation_factors: List[float] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate policy under treatment effect degradation.
    """
    if degradation_factors is None:
        degradation_factors = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    results = []

    for deg in degradation_factors:
        policies = evaluator.evaluate_comparison_policies(
            df, decision_table,
            degradation_factor=deg,
            treatment_effect_multiplier=1.0 - deg,
        )

        results.append({
            "degradation": deg,
            "do_nothing": policies["do_nothing"]["net_revenue"],
            "always_retry": policies["always_retry"]["net_revenue"],
            "max_probability": policies["max_probability"]["net_revenue"],
            "recoverytwin": policies["recoverytwin"]["net_revenue"],
            "oracle": policies["oracle"]["net_revenue"],
            "rt_incremental": (
                policies["recoverytwin"]["net_revenue"]
                - policies["do_nothing"]["net_revenue"]
            ),
            "policy_regret": (
                (policies["oracle"]["net_revenue"] - policies["recoverytwin"]["net_revenue"])
                / max(abs(policies["oracle"]["net_revenue"]), 1e-8)
            ),
        })

    return results


def time_discount_sensitivity(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    evaluator,
    lambda_values: List[float] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate policy under different time discount rates.
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.005, 0.01, 0.02, 0.05]

    results = []

    for lam in lambda_values:
        policies = evaluator.evaluate_comparison_policies(
            df, decision_table,
            time_discount_lambda=lam,
        )

        results.append({
            "lambda": lam,
            "half_life_hours": 0.693 / max(lam, 1e-8) if lam > 0 else float("inf"),
            "do_nothing": policies["do_nothing"]["net_revenue"],
            "always_retry": policies["always_retry"]["net_revenue"],
            "max_probability": policies["max_probability"]["net_revenue"],
            "recoverytwin": policies["recoverytwin"]["net_revenue"],
            "oracle": policies["oracle"]["net_revenue"],
            "rt_incremental": (
                policies["recoverytwin"]["net_revenue"]
                - policies["do_nothing"]["net_revenue"]
            ),
        })

    return results


def retry_limit_sensitivity(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    policy_filter,
    retry_limits: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate policy under different retry limits.
    """
    if retry_limits is None:
        retry_limits = [0, 1, 2, 3, 4, 5]

    results = []

    # Save original
    original_limit = policy_filter.max_retries

    for limit in retry_limits:
        policy_filter.max_retries = limit

        # Re-evaluate action selection
        from recoverytwin.decision.engine import DecisionEngine, load_policy_config
        engine = DecisionEngine(load_policy_config())
        engine.policy_filter = policy_filter

        dt = engine.select_actions(df, decision_table)

        rt_actions = dt["recommended_action"].values
        n = len(rt_actions)

        results.append({
            "retry_limit": limit,
            "n_retries": int((rt_actions == 1).sum()),
            "intervention_rate": float((rt_actions > 0).mean()),
            "n_control": int((rt_actions == 0).sum()),
        })

    # Restore
    policy_filter.max_retries = original_limit

    return results


def find_breakeven_cost(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    evaluator,
    action: int = 1,
    tolerance: float = 0.01,
    max_cost: float = 100.0,
) -> Dict[str, Any]:
    """
    Find the break-even cost for a given intervention.

    Break-even: the cost at which the intervention's incremental value = 0.
    Uses binary search.
    """
    # Find break-even using binary search
    low, high = 0.0, max_cost
    best_val = None

    for _ in range(50):  # Binary search iterations
        mid = (low + high) / 2.0
        costs = {0: 0.0, 1: mid if action == 1 else 0.50,
                 2: mid if action == 2 else 1.00,
                 3: mid if action == 3 else 2.50}

        policies = evaluator.evaluate_comparison_policies(
            df, decision_table, action_costs=costs,
        )

        rt_val = policies["recoverytwin"]["net_revenue"]
        dn_val = policies["do_nothing"]["net_revenue"]
        incremental = rt_val - dn_val

        if incremental > tolerance:
            low = mid
            best_val = incremental
        else:
            high = mid

    return {
        "action": action,
        "breakeven_cost": float((low + high) / 2),
        "incremental_at_breakeven": float(best_val or 0),
    }


def segment_analysis(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    evaluator,
    segment_column: str = "failure_reason",
) -> List[Dict[str, Any]]:
    """
    Evaluate policy performance by segment.
    """
    results = []

    if segment_column not in df.columns:
        return results

    for seg_val in df[segment_column].unique():
        mask = df[segment_column] == seg_val
        if mask.sum() < 10:
            continue

        seg_df = df[mask].reset_index(drop=True)
        seg_dt = decision_table[mask].reset_index(drop=True)

        policies = evaluator.evaluate_comparison_policies(seg_df, seg_dt)

        rt_val = policies["recoverytwin"]["net_revenue"]
        dn_val = policies["do_nothing"]["net_revenue"]
        oracle_val = policies["oracle"]["net_revenue"]

        results.append({
            "segment_column": segment_column,
            "segment_value": str(seg_val),
            "n": int(mask.sum()),
            "do_nothing": dn_val,
            "recoverytwin": rt_val,
            "max_probability": policies["max_probability"]["net_revenue"],
            "oracle": oracle_val,
            "rt_incremental": rt_val - dn_val,
            "regret": (oracle_val - rt_val) / max(abs(oracle_val), 1e-8),
            "recovery_rate_rt": policies["recoverytwin"]["recovery_rate"],
            "intervention_rate_rt": policies["recoverytwin"]["n_interventions"] / max(mask.sum(), 1),
        })

    return results
