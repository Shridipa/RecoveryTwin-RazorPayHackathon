"""
Kaplan-Meier survival analysis by intervention.

Provides non-parametric recovery curves for each treatment group.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from typing import Dict, Any, Optional, List


def prepare_survival_data(
    df: pd.DataFrame,
    censoring_time: float = 168.0,
) -> pd.DataFrame:
    """
    Prepare survival data from the payment dataset.

    For recovered payments: duration = recovery_time_hours, event = 1
    For unrecovered payments: duration = censoring_time, event = 0

    Args:
        df: DataFrame with recovery_time_hours and recovered columns
        censoring_time: Maximum observation time in hours (default 168 = 1 week)

    Returns:
        DataFrame with 'duration' and 'event' columns added
    """
    data = df.copy()

    # Duration: recovery time for recovered, censoring time for unrecovered
    data["duration"] = np.where(
        data["recovered"] == 1,
        data["recovery_time_hours"],
        censoring_time,
    )

    # Event indicator: 1 if recovered, 0 if censored
    data["event"] = data["recovered"].values.astype(int)

    # Ensure duration > 0
    data["duration"] = np.clip(data["duration"], 0.01, censoring_time)

    return data


def fit_kaplan_meier(
    df: pd.DataFrame,
    censoring_time: float = 168.0,
) -> Dict[str, Any]:
    """
    Fit Kaplan-Meier curves for each intervention group.

    Args:
        df: Raw payment data
        censoring_time: Maximum observation time

    Returns:
        Dict with KM results per intervention and overall
    """
    data = prepare_survival_data(df, censoring_time)

    intervention_names = {
        0: "control",
        1: "retry",
        2: "reminder",
        3: "alternative_method",
    }

    results = {}

    # Overall KM
    kmf = KaplanMeierFitter()
    kmf.fit(data["duration"], event_observed=data["event"])
    results["overall"] = {
        "median_survival_time": kmf.median_survival_time_,
        "survival_function": kmf.survival_function_at_times(
            [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]
        ).to_dict(),
        "event_rate": float(data["event"].mean()),
        "n": len(data),
    }

    # Per-intervention KM
    intervention_results = {}
    for t in range(4):
        mask = data["intervention"] == t
        subset = data[mask]

        if len(subset) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(subset["duration"], event_observed=subset["event"])

        # Recovery probability at key timepoints
        timepoints = [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]
        survival_probs = kmf.survival_function_at_times(timepoints)
        # Recovery probability = 1 - survival probability
        recovery_probs = (1 - survival_probs).to_dict()

        intervention_results[t] = {
            "name": intervention_names[t],
            "n": int(mask.sum()),
            "event_rate": float(subset["event"].mean()),
            "median_survival_time": float(kmf.median_survival_time_),
            "recovery_at_timepoints": {int(k): float(v) for k, v in recovery_probs.items()},
            "kmf": kmf,  # Keep for plotting
        }

    results["by_intervention"] = intervention_results

    return results


def print_km_results(results: Dict[str, Any]):
    """Print formatted KM results."""
    print("\n" + "=" * 80)
    print("KAPLAN-MEIER RECOVERY CURVES")
    print("=" * 80)

    timepoints = [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]

    # Header
    header = f"{'Intervention':<20} {'n':>6}"
    for t in timepoints:
        header += f" {t:>5}h"
    print(header)
    print("-" * 80)

    for t, info in sorted(results["by_intervention"].items()):
        row = f"{info['name']:<20} {info['n']:>6}"
        for tp in timepoints:
            prob = info["recovery_at_timepoints"].get(tp, 0)
            row += f" {prob:>5.1%}"
        print(row)

    print("-" * 80)
    overall = results["overall"]
    row = f"{'overall':<20} {overall['n']:>6}"
    print(f"\nOverall event rate: {overall['event_rate']:.1%}")
    print(f"Median survival time (overall): {overall.get('median_survival_time', 'N/A')}")
    print("=" * 80)
