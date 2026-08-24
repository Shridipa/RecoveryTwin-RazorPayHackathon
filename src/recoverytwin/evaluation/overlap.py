"""
Propensity overlap and positivity analysis.

Computes:
- Propensity distributions
- Effective Sample Size (ESS)
- Overlap diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_ess(weights: np.ndarray) -> float:
    """
    Compute Effective Sample Size.

    ESS = (sum(w))^2 / sum(w^2)
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def compute_overlap_diagnostics(
    df: pd.DataFrame,
    treatment_col: str = "intervention",
) -> Dict[str, Any]:
    """
    Compute propensity overlap diagnostics.

    Args:
        df: DataFrame with treatment column and propensity columns
        treatment_col: Name of treatment column

    Returns:
        Dict with overlap diagnostics
    """
    results = {}

    # Treatment distribution
    treatment_counts = df[treatment_col].value_counts().to_dict()
    treatment_pcts = df[treatment_col].value_counts(normalize=True).to_dict()
    results["treatment_distribution"] = treatment_counts
    results["treatment_percentages"] = {k: float(v) for k, v in treatment_pcts.items()}

    # Propensity statistics per treatment
    propensity_stats = {}
    for t in range(4):
        prop_col = f"propensity_{t}"
        if prop_col in df.columns:
            props = df[prop_col].values
            propensity_stats[t] = {
                "mean": float(np.mean(props)),
                "std": float(np.std(props)),
                "min": float(np.min(props)),
                "max": float(np.max(props)),
                "median": float(np.median(props)),
            }
    results["propensity_stats"] = propensity_stats

    # ESS per treatment using inverse propensity weights
    ess_results = {}
    for t in range(4):
        treated = df[df[treatment_col] == t]
        prop_col = f"propensity_{t}"
        if len(treated) > 0 and prop_col in treated.columns:
            props = np.clip(treated[prop_col].values, 1e-6, 1.0)
            weights = 1.0 / props
            ess = compute_ess(weights)
            ess_ratio = ess / len(treated)
            ess_results[t] = {
                "n": len(treated),
                "ess": float(ess),
                "ess_ratio": float(ess_ratio),
                "min_propensity": float(np.min(props)),
                "max_propensity": float(np.max(props)),
                "extreme_count_low": int((props < 0.01).sum()),
                "extreme_count_high": int((props > 0.99).sum()),
            }
    results["ess"] = ess_results

    # Overall overlap check
    min_ess_ratio = min(e["ess_ratio"] for e in ess_results.values()) if ess_results else 0
    results["min_ess_ratio"] = float(min_ess_ratio)
    results["positivity_pass"] = min_ess_ratio > 0.5

    # Support check: no treatment has near-zero support
    min_treatment_pct = min(treatment_pcts.values()) if treatment_pcts else 0
    results["min_treatment_pct"] = float(min_treatment_pct)
    results["support_pass"] = min_treatment_pct > 0.05

    return results


def print_overlap_report(overlap: Dict[str, Any]):
    """Print formatted overlap report."""
    print("\n" + "=" * 60)
    print("POSITIVITY / OVERLAP ANALYSIS")
    print("=" * 60)

    print("\nTreatment Distribution:")
    for t, pct in sorted(overlap["treatment_percentages"].items()):
        count = overlap["treatment_distribution"][t]
        print(f"  Treatment {t}: {count} ({pct:.1%})")

    print("\nESS Analysis:")
    for t, ess_info in sorted(overlap["ess"].items()):
        print(f"  Treatment {t}: ESS={ess_info['ess']:.0f}, "
              f"ratio={ess_info['ess_ratio']:.3f}, "
              f"n={ess_info['n']}")

    print(f"\nMin ESS ratio: {overlap['min_ess_ratio']:.3f}")
    print(f"Positivity: {'[PASS]' if overlap['positivity_pass'] else '[FAIL]'}")
    print(f"Support: {'[PASS]' if overlap['support_pass'] else '[FAIL]'}")
    print("=" * 60)
