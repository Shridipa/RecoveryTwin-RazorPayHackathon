"""
Causal Model Evaluation.

Evaluates estimated CATEs against simulator ground truth (potential outcomes).
Provides ATE accuracy, CATE correlation, policy value, and regret metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats


def evaluate_ate(cate_estimates: Dict[int, np.ndarray], 
                 df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate Average Treatment Effect estimation accuracy.
    
    Compares estimated ATE against true ATE from potential outcomes.
    
    Args:
        cate_estimates: Dict mapping treatment_id -> estimated CATE array
        df: DataFrame with potential_outcome_* columns
    
    Returns:
        Dict with ATE evaluation metrics
    """
    results = {}
    
    for t, estimated_cate in cate_estimates.items():
        # True CATE from potential outcomes
        true_cate = df[f'potential_outcome_{t}'].values - df['potential_outcome_0'].values
        
        true_ate = np.mean(true_cate)
        estimated_ate = np.mean(estimated_cate)
        
        # Absolute and relative error
        ate_error = abs(estimated_ate - true_ate)
        ate_relative_error = ate_error / max(abs(true_ate), 1e-8)
        
        # CATE-level metrics
        cate_mse = np.mean((estimated_cate - true_cate) ** 2)
        cate_corr, _ = stats.spearmanr(estimated_cate, true_cate)
        
        # PEHE (Precision in Estimation of Heterogeneous Effects)
        pehe = np.sqrt(np.mean((estimated_cate - true_cate) ** 2))
        
        results[f'treatment_{t}'] = {
            'true_ate': float(true_ate),
            'estimated_ate': float(estimated_ate),
            'ate_error': float(ate_error),
            'ate_relative_error': float(ate_relative_error),
            'cate_mse': float(cate_mse),
            'cate_spearman_corr': float(cate_corr),
            'pehe': float(pehe),
            'n_samples': len(estimated_cate),
        }
    
    return results


def evaluate_policy(cate_estimates: Dict[int, np.ndarray], 
                     df: pd.DataFrame,
                     costs: Dict[int, float] = None) -> Dict[str, Any]:
    """
    Evaluate policy value: how well does the CATE-based policy perform
    compared to the oracle policy and random assignment?
    
    Policy: assign each unit to the treatment with highest estimated CATE.
    Oracle: assign each unit to the treatment with highest true CATE.
    
    Args:
        cate_estimates: Dict mapping treatment_id -> estimated CATE array
        df: DataFrame with potential_outcome_* columns
        costs: Optional intervention costs per treatment
    
    Returns:
        Dict with policy evaluation metrics
    """
    n = len(df)
    treatments = sorted(cate_estimates.keys())
    
    if costs is None:
        costs = {0: 0.0, 1: 2.0, 2: 1.0, 3: 3.0}
    
    # Build CATE matrix: (n_samples, n_treatments)
    cate_matrix = np.zeros((n, len(treatments)))
    for i, t in enumerate(treatments):
        cate_matrix[:, i] = cate_estimates[t]
    
    # True CATE matrix
    true_cate_matrix = np.zeros((n, len(treatments)))
    for i, t in enumerate(treatments):
        true_cate_matrix[:, i] = (
            df[f'potential_outcome_{t}'].values - df['potential_outcome_0'].values
        )
    
    # Oracle policy: maximize true CATE
    oracle_actions = np.argmax(true_cate_matrix, axis=1)
    
    # Learned policy: maximize estimated CATE
    learned_actions = np.argmax(cate_matrix, axis=1)
    
    # Random policy
    rng = np.random.RandomState(42)
    random_actions = rng.randint(0, len(treatments), size=n)
    
    # Evaluate each policy's expected value using true potential outcomes
    # Value = E[Y(action)] - Cost(action) for the units assigned
    def policy_value(actions, label=""):
        values = []
        for i in range(n):
            t = treatments[actions[i]]
            y_true = df[f'potential_outcome_{t}'].iloc[i]
            cost = costs.get(t, 0.0)
            values.append(y_true - cost)
        return np.mean(values)
    
    oracle_value = policy_value(oracle_actions, "oracle")
    learned_value = policy_value(learned_actions, "learned")
    random_value = policy_value(random_actions, "random")
    control_value = policy_value(np.zeros(n, dtype=int), "control")
    
    # Regret: difference from oracle
    regret = oracle_value - learned_value
    regret_pct = regret / max(abs(oracle_value), 1e-8) * 100
    
    # Fraction of units assigned optimally
    optimal_fraction = np.mean(learned_actions == oracle_actions)
    
    return {
        'oracle_value': float(oracle_value),
        'learned_value': float(learned_value),
        'random_value': float(random_value),
        'control_value': float(control_value),
        'regret': float(regret),
        'regret_pct': float(regret_pct),
        'optimal_assignment_fraction': float(optimal_fraction),
        'lift_vs_control': float(learned_value - control_value),
        'lift_vs_random': float(learned_value - random_value),
        'fraction_vs_oracle': float(learned_value / max(oracle_value, 1e-8)),
    }


def evaluate_best_action(cate_estimates: Dict[int, np.ndarray],
                          df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate how well the model identifies the best individual treatment.
    
    True best intervention is known from the simulator.
    """
    n = len(df)
    treatments = sorted(cate_estimates.keys())
    
    # Learned best: argmax of estimated CATE
    cate_matrix = np.column_stack([cate_estimates[t] for t in treatments])
    learned_best = np.argmax(cate_matrix, axis=1) + 1  # +1 because treatments are 1,2,3
    
    # True best from potential outcomes (excluding control)
    true_cate_matrix = np.column_stack([
        df[f'potential_outcome_{t}'].values - df['potential_outcome_0'].values
        for t in treatments
    ])
    true_best = np.argmax(true_cate_matrix, axis=1) + 1
    
    # Accuracy
    accuracy = np.mean(learned_best == true_best)
    
    # Confusion-like breakdown
    from collections import Counter
    learned_dist = Counter(learned_best)
    true_dist = Counter(true_best)
    
    return {
        'accuracy': float(accuracy),
        'n_samples': n,
        'learned_distribution': {str(k): float(v/n) for k, v in learned_dist.items()},
        'true_distribution': {str(k): float(v/n) for k, v in true_dist.items()},
    }


def run_full_causal_evaluation(cate_estimates: Dict[int, np.ndarray],
                                 df: pd.DataFrame,
                                 model_name: str = "model") -> Dict[str, Any]:
    """
    Run complete causal evaluation for a set of CATE estimates.
    
    Returns comprehensive evaluation dictionary.
    """
    ate_eval = evaluate_ate(cate_estimates, df)
    policy_eval = evaluate_policy(cate_estimates, df)
    best_action_eval = evaluate_best_action(cate_estimates, df)
    
    return {
        'model': model_name,
        'ate_evaluation': ate_eval,
        'policy_evaluation': policy_eval,
        'best_action_evaluation': best_action_eval,
    }
