"""
SHAP analysis for tree models.

Handles modern SHAP output formats (2D, 3D, class-specific).
"""

import numpy as np
import shap
from typing import Any, Dict, Optional


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list = None,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Compute SHAP values safely handling various output formats.

    Args:
        model: Fitted tree model (XGBoost, LightGBM, RF)
        X: Feature matrix
        feature_names: Feature names
        max_samples: Max samples for SHAP (for speed)

    Returns:
        Dict with shap_values, mean_abs_shap, feature_importance
    """
    # Subsample for speed
    if len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
        idx = np.arange(len(X))

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # Handle different SHAP output formats
    # Could be: 2D (n_samples, n_features), 3D (n_samples, n_features, n_classes),
    # or a list of arrays (one per class)
    if isinstance(shap_values, list):
        # List of arrays (one per class)
        # Use the positive class (class 1) or last class
        if len(shap_values) > 1:
            sv = shap_values[1]  # Positive class
        else:
            sv = shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array (n_samples, n_features, n_classes)
        sv = shap_values[:, :, 1]  # Positive class
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        sv = shap_values
    else:
        sv = np.array(shap_values)

    # Mean absolute SHAP values
    mean_abs_shap = np.abs(sv).mean(axis=0)

    # Feature importance ranking
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(sv.shape[1])]

    importance = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "shap_values": sv,
        "mean_abs_shap": mean_abs_shap,
        "feature_importance": importance,
        "top_10": importance[:10],
        "n_samples": len(sv),
        "n_features": sv.shape[1],
    }


def sanity_check_shap(shap_result: Dict) -> bool:
    """Verify SHAP results are reasonable."""
    sv = shap_result["shap_values"]
    checks = [
        sv is not None,
        sv.ndim >= 2,
        sv.shape[0] > 0,
        sv.shape[1] > 0,
        not np.all(np.isnan(sv)),
    ]
    return all(checks)
