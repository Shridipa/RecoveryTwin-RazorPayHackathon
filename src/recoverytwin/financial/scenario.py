"""
RecoveryTwin Financial Policy Simulator — Scenario Definitions.

Named scenarios with parameter modifications for stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml


def load_scenarios(path: str = "configs/financial.yaml") -> Dict[str, Dict]:
    """Load named scenarios from configuration."""
    p = Path(path)
    if not p.exists():
        return _default_scenarios()
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("scenarios", _default_scenarios())


def _default_scenarios() -> Dict[str, Dict]:
    return {
        "BASELINE": {
            "description": "Default parameters",
            "cost_multiplier": 1.0,
            "degradation_factor": 0.0,
            "recovery_rate_multiplier": 1.0,
            "treatment_effect_multiplier": 1.0,
            "amount_multiplier": 1.0,
            "fatigue_threshold": 6,
        },
    }


def get_scenario_params(
    scenario_name: str,
    scenarios: Dict[str, Dict] = None,
) -> Dict[str, Any]:
    """Get parameters for a named scenario."""
    if scenarios is None:
        scenarios = load_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    return scenarios[scenario_name]


class ScenarioRunner:
    """Runs financial evaluation under multiple scenarios."""

    def __init__(
        self,
        df: pd.DataFrame,
        decision_table: pd.DataFrame,
        evaluator,
    ):
        self.df = df
        self.decision_table = decision_table
        self.evaluator = evaluator

    def run_scenario(
        self,
        scenario_name: str,
        scenario_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single scenario and return results."""
        kwargs = {
            "amount_multiplier": scenario_params.get("amount_multiplier", 1.0),
            "degradation_factor": scenario_params.get("degradation_factor", 0.0),
            "treatment_effect_multiplier": scenario_params.get("treatment_effect_multiplier", 1.0),
            "recovery_rate_multiplier": scenario_params.get("recovery_rate_multiplier", 1.0),
        }

        # Cost multiplier
        cost_mult = scenario_params.get("cost_multiplier", 1.0)
        base_costs = self.evaluator.costs
        adjusted_costs = {k: v * cost_mult for k, v in base_costs.items()}

        # Evaluate all comparison policies
        policies = self.evaluator.evaluate_comparison_policies(
            self.df, self.decision_table,
            action_costs=adjusted_costs,
            **kwargs,
        )

        # Calculate regret
        oracle_val = policies["oracle"]["net_revenue"]
        rt_val = policies["recoverytwin"]["net_revenue"]
        dn_val = policies["do_nothing"]["net_revenue"]

        regret = (oracle_val - rt_val) / max(abs(oracle_val), 1e-8)

        return {
            "scenario": scenario_name,
            "description": scenario_params.get("description", ""),
            "policies": {
                name: {
                    "net_revenue": pol["net_revenue"],
                    "recovery_rate": pol["recovery_rate"],
                    "n_interventions": pol["n_interventions"],
                    "total_cost": pol["total_cost"],
                }
                for name, pol in policies.items()
            },
            "recoverytwin_incremental": rt_val - dn_val,
            "policy_regret": regret,
            "beats_do_nothing": rt_val > dn_val,
            "beats_max_probability": rt_val > policies["max_probability"]["net_revenue"],
            "params": scenario_params,
        }

    def run_all_scenarios(
        self,
        scenario_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run all (or selected) scenarios."""
        scenarios = load_scenarios()
        if scenario_names is None:
            scenario_names = list(scenarios.keys())

        results = []
        for name in scenario_names:
            if name in scenarios:
                result = self.run_scenario(name, scenarios[name])
                results.append(result)

        return results


def compute_robustness_score(
    scenario_results: List[Dict[str, Any]],
    baseline_policy: str = "do_nothing",
    target_policy: str = "recoverytwin",
) -> Dict[str, float]:
    """
    Compute robustness score: fraction of scenarios where
    target policy outperforms baseline.
    """
    n = len(scenario_results)
    if n == 0:
        return {"vs_baseline": 0.0, "vs_max_prob": 0.0}

    beats_baseline = sum(
        1 for r in scenario_results
        if r["policies"][target_policy]["net_revenue"]
        > r["policies"][baseline_policy]["net_revenue"]
    )

    beats_max_prob = sum(
        1 for r in scenario_results
        if r["policies"][target_policy]["net_revenue"]
        > r["policies"]["max_probability"]["net_revenue"]
    )

    return {
        "vs_baseline": beats_baseline / n,
        "vs_max_prob": beats_max_prob / n,
        "n_scenarios": n,
        "n_beats_baseline": beats_baseline,
        "n_beats_max_prob": beats_max_prob,
    }


def compute_worst_case(
    scenario_results: List[Dict[str, Any]],
    target_policy: str = "recoverytwin",
) -> Dict[str, Any]:
    """Find worst-case scenario for the target policy."""
    if not scenario_results:
        return {"scenario": None, "net_revenue": 0, "regret": 1.0}

    worst = min(scenario_results, key=lambda r: r["policies"][target_policy]["net_revenue"])
    dn_val = worst["policies"]["do_nothing"]["net_revenue"]
    rt_val = worst["policies"][target_policy]["net_revenue"]

    return {
        "scenario": worst["scenario"],
        "description": worst.get("description", ""),
        "net_revenue": rt_val,
        "incremental_over_do_nothing": rt_val - dn_val,
        "policy_regret": worst["policy_regret"],
        "recovery_rate": worst["policies"][target_policy]["recovery_rate"],
        "beats_do_nothing": rt_val > dn_val,
    }
