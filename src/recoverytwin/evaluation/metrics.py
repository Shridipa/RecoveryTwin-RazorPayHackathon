"""
Evaluation metrics for binary classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix,
)
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted labels (if None, derived from y_prob > 0.5)

    Returns:
        Dictionary of metrics
    """
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }

    # ROC AUC (needs both classes)
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5

    # Log loss
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    metrics["log_loss"] = float(log_loss(y_true, y_prob_clipped))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    return metrics


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        ECE value
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / total
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute calibration-specific metrics.

    Returns:
        Brier score, ECE, calibration slope, calibration intercept
    """
    from sklearn.linear_model import LinearRegression

    brier = brier_score_loss(y_true, y_prob)

    ece = compute_ece(y_true, y_prob)

    # Calibration slope and intercept via Platt scaling inverse
    # regressing observed outcome on predicted probability
    X = y_prob.reshape(-1, 1)
    reg = LinearRegression().fit(X, y_true)
    cal_slope = float(reg.coef_[0])
    cal_intercept = float(reg.intercept_)

    return {
        "brier_score": brier,
        "ece": ece,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
    }
