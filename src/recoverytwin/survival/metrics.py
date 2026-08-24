"""
Survival model evaluation metrics.

Includes:
- Concordance Index
- Integrated Brier Score
- Time-dependent AUC
"""

import numpy as np
from typing import Dict, Any


def compute_concordance_index(
    duration: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """
    Compute Harrell's concordance index.

    Higher is better. 0.5 = random, 1.0 = perfect.
    """
    from sksurv.metrics import concordance_index_censored

    result = concordance_index_censored(
        event.astype(bool), duration, risk_scores
    )
    return float(result[0])


def compute_integrated_brier_score(
    train_duration: np.ndarray,
    train_event: np.ndarray,
    test_duration: np.ndarray,
    test_event: np.ndarray,
    test_survival_probs: np.ndarray,
    times: np.ndarray = None,
) -> float:
    """
    Compute Integrated Brier Score.

    Lower is better. 0.0 = perfect, 0.25 = random.
    """
    from sksurv.metrics import integrated_brier_score

    if times is None:
        times = np.linspace(
            test_duration.min() + 0.1,
            np.percentile(test_duration[test_event == 1], 90) if test_event.any() else test_duration.max() * 0.9,
            50,
        )

    # Prepare structured arrays
    train_y = np.array(
        [(bool(e), float(d)) for e, d in zip(train_event, train_duration)],
        dtype=[("event", bool), ("time", float)],
    )
    test_y = np.array(
        [(bool(e), float(d)) for e, d in zip(test_event, test_duration)],
        dtype=[("event", bool), ("time", float)],
    )

    try:
        ibs = integrated_brier_score(train_y, test_y, test_survival_probs, times)
        return float(ibs)
    except Exception:
        return float("nan")


def compute_time_dependent_auc(
    duration: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    times: np.ndarray = None,
) -> Dict[float, float]:
    """
    Compute time-dependent AUC at specified timepoints.

    Returns dict mapping time -> AUC.
    """
    from sksurv.metrics import cumulative_dynamic_auc

    if times is None:
        event_times = duration[event == 1]
        if len(event_times) > 0:
            times = np.percentile(event_times, [25, 50, 75])
        else:
            times = np.array([24, 48, 96])

    # Prepare structured arrays
    y = np.array(
        [(bool(e), float(d)) for e, d in zip(event, duration)],
        dtype=[("event", bool), ("time", float)],
    )

    try:
        auc_values, mean_auc = cumulative_dynamic_auc(y, y, risk_scores, times)
        return {float(t): float(a) for t, a in zip(times, auc_values)}
    except Exception:
        return {}


def survival_metrics_summary(
    duration: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    train_duration: np.ndarray = None,
    train_event: np.ndarray = None,
    survival_probs: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Compute all survival metrics.

    Returns:
        Dict with C-index, IBS, time-dependent AUC
    """
    cindex = compute_concordance_index(duration, event, risk_scores)

    result = {
        "concordance_index": cindex,
    }

    # Time-dependent AUC
    td_auc = compute_time_dependent_auc(duration, event, risk_scores)
    result["time_dependent_auc"] = td_auc

    # IBS (if survival probabilities available)
    if survival_probs is not None and train_duration is not None:
        ibs = compute_integrated_brier_score(
            train_duration, train_event, duration, event, survival_probs
        )
        result["integrated_brier_score"] = ibs

    return result
