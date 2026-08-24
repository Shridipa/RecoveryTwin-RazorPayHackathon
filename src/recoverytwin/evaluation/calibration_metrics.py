"""
Calibration metrics and reliability curve data.
"""

import numpy as np
from typing import Dict, List, Tuple


def reliability_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """
    Compute reliability diagram data.

    Returns list of dicts with bin_center, mean_predicted, fraction_positive, count.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    data = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        if mask.sum() > 0:
            data.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "mean_predicted": float(y_prob[mask].mean()),
                "fraction_positive": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            })

    return data
