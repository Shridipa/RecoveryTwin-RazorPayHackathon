"""
Model calibration: sigmoid, isotonic.

IMPORTANT: Calibration is fitted on VALIDATION data only.
Test data is only used for final evaluation.

Handles sklearn 1.8.0+ where cv="prefit" may not be supported.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Any, Tuple, Optional
import warnings


class SigmoidCalibrator:
    """Platt scaling (sigmoid) calibration fitted on validation data."""

    def __init__(self):
        self.model = LogisticRegression(C=1.0, random_state=42)
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit on validation data."""
        X = y_prob.reshape(-1, 1)
        self.model.fit(X, y_true)
        self.fitted = True
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        X = y_prob.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibration fitted on validation data."""

    def __init__(self):
        self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit on validation data."""
        self.model.fit(y_prob, y_true)
        self.fitted = True
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        return self.model.predict(y_prob)


def calibrate_model(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    y_test_prob: np.ndarray,
    methods: list = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Calibrate a model using multiple methods.

    Args:
        y_val_true: Validation true labels
        y_val_prob: Validation predicted probabilities
        y_test_prob: Test predicted probabilities
        methods: List of calibration methods ["none", "sigmoid", "isotonic"]

    Returns:
        Dict with calibrated test probabilities for each method
    """
    if methods is None:
        methods = ["none", "sigmoid", "isotonic"]

    results = {}

    # Uncalibrated
    if "none" in methods:
        results["none"] = {
            "test_probs": y_test_prob.copy(),
            "calibrator": None,
        }

    # Sigmoid
    if "sigmoid" in methods:
        calibrator = SigmoidCalibrator()
        calibrator.fit(y_val_true, y_val_prob)
        calibrated_test = calibrator.predict(y_test_prob)
        results["sigmoid"] = {
            "test_probs": calibrated_test,
            "calibrator": calibrator,
        }

    # Isotonic
    if "isotonic" in methods:
        calibrator = IsotonicCalibrator()
        calibrator.fit(y_val_true, y_val_prob)
        calibrated_test = calibrator.predict(y_test_prob)
        results["isotonic"] = {
            "test_probs": calibrated_test,
            "calibrator": calibrator,
        }

    return results


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method_name: str = "unknown",
) -> Dict[str, Any]:
    """Evaluate calibration quality."""
    from recoverytwin.evaluation.metrics import compute_metrics, compute_calibration_metrics

    metrics = compute_metrics(y_true, y_prob)
    cal_metrics = compute_calibration_metrics(y_true, y_prob)

    return {
        "method": method_name,
        **metrics,
        **cal_metrics,
    }
