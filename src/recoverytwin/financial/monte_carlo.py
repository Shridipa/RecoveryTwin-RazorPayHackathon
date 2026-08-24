"""
RecoveryTwin Financial Policy Simulator — Monte Carlo Simulation.

Runs multiple simulated batches to estimate distributions of
financial outcomes under uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml


def load_mc_config(path: str = "configs/financial.yaml") -> Dict:
    """Load Monte Carlo configuration."""
    p = Path(path)
    if not p.exists():
        return {"n_simulations": 500, "random_seed": 42}
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("monte_carlo", {"n_simulations": 500, "random_seed": 42})


class MonteCarloSimulator:
    """
    Runs Monte Carlo simulations over a batch of payments.

    For each simulation:
      1. Sample recovery outcomes using the model's predicted probabilities
      2. Add noise to payment amounts
      3. Apply intervention costs
      4. Calculate total recovered revenue and net value
    """

    def __init__(
        self,
        n_simulations: int = 500,
        random_seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_seed)

    def simulate_policy(
        self,
        df: pd.DataFrame,
        actions: np.ndarray,
        recovery_probs: np.ndarray,
        amounts: np.ndarray,
        intervention_costs: np.ndarray,
        amount_noise_std: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for a single policy.

        Parameters
        ----------
        df : payment data
        actions : (n,) selected action per payment
        recovery_probs : (n,) P(recovery | X, action) for selected action
        amounts : (n,) payment amounts
        intervention_costs : (n,) cost of selected action per payment
        amount_noise_std : std of multiplicative noise on amounts (0 = no noise)

        Returns
        -------
        Dict with distributions and summary statistics
        """
        n = len(actions)
        results = {
            "total_revenue": np.zeros(self.n_simulations),
            "total_cost": np.zeros(self.n_simulations),
            "net_revenue": np.zeros(self.n_simulations),
            "recovery_count": np.zeros(self.n_simulations),
            "intervention_count": np.zeros(self.n_simulations),
        }

        for sim in range(self.n_simulations):
            # Sample recovery outcomes
            outcomes = self.rng.binomial(1, np.clip(recovery_probs, 0, 1))

            # Apply amount noise
            if amount_noise_std > 0:
                noise = self.rng.normal(1.0, amount_noise_std, n)
                sim_amounts = amounts * np.maximum(noise, 0.1)
            else:
                sim_amounts = amounts

            # Revenue and costs
            revenue = (outcomes * sim_amounts).sum()
            cost = intervention_costs.sum()

            results["total_revenue"][sim] = revenue
            results["total_cost"][sim] = cost
            results["net_revenue"][sim] = revenue - cost
            results["recovery_count"][sim] = outcomes.sum()
            results["intervention_count"][sim] = (actions > 0).sum()

        # Calculate summary statistics
        net_revenues = results["net_revenue"]
        ci_levels = [5, 25, 50, 75, 95]
        percentiles = np.percentile(net_revenues, ci_levels)

        return {
            "n_simulations": self.n_simulations,
            "mean_net_revenue": float(net_revenues.mean()),
            "median_net_revenue": float(np.median(net_revenues)),
            "std_net_revenue": float(net_revenues.std()),
            "ci_lower": float(percentiles[0]),
            "ci_upper": float(percentiles[-1]),
            "p5": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p95": float(percentiles[4]),
            "mean_recovery_rate": float(results["recovery_count"].mean() / n),
            "mean_total_cost": float(results["total_cost"].mean()),
            "prob_positive_net": float((net_revenues > 0).mean()),
            "raw": {k: v.tolist() for k, v in results.items()},
        }

    def compare_policies(
        self,
        df: pd.DataFrame,
        decision_table: pd.DataFrame,
        true_po: np.ndarray,
        action_costs: Dict[int, float],
        amount_noise_std: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo for multiple policies and compare.
        """
        n = len(df)
        amounts = df["amount"].values

        results = {}
        policy_actions = {
            "do_nothing": np.zeros(n, dtype=int),
            "always_retry": np.ones(n, dtype=int),
        }

        # Max probability
        prob_cols = ["control_prob"] + [
            f"{['','retry','reminder','alternative_method'][a]}_prob"
            for a in range(1, 4)
        ]
        if all(c in decision_table.columns for c in prob_cols):
            pred_probs = np.column_stack([decision_table[c].values for c in prob_cols])
            policy_actions["max_probability"] = np.argmax(pred_probs, axis=1)
        else:
            policy_actions["max_probability"] = np.ones(n, dtype=int)

        # RecoveryTwin
        if "recommended_action" in decision_table.columns:
            policy_actions["recoverytwin"] = decision_table["recommended_action"].values
        else:
            policy_actions["recoverytwin"] = np.zeros(n, dtype=int)

        for pol_name, actions in policy_actions.items():
            # Get recovery probs and costs for this policy
            probs = np.array([true_po[i, int(actions[i])] for i in range(n)])
            costs_arr = np.array([action_costs.get(int(actions[i]), 0) for i in range(n)])

            results[pol_name] = self.simulate_policy(
                df, actions, probs, amounts, costs_arr,
                amount_noise_std=amount_noise_std,
            )

        return results
