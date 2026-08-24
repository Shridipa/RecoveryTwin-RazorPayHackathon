"""
Baseline models: Logistic Regression and Random Forest.

Supports both P(Y|X) and P(Y|X,T) variants.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path


def prepare_features(
    df: pd.DataFrame,
    include_treatment: bool = False,
    exclude_cols: set = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Prepare feature matrix, encoding categoricals.

    Returns:
        X: Encoded feature DataFrame
        feature_names: List of feature names
    """
    from recoverytwin.data.feature_availability import get_model_features

    if exclude_cols is None:
        exclude_cols = set()

    feature_cols = get_model_features(df, include_treatment=include_treatment)
    feature_cols = [c for c in feature_cols if c not in exclude_cols]

    X = df[feature_cols].copy()

    # Encode categoricals
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Ensure all numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, feature_cols, label_encoders


class BaselineModel:
    """Wrapper for baseline models."""

    def __init__(self, model_type: str = "logistic", include_treatment: bool = False,
                 name: str = None):
        """
        Args:
            model_type: "logistic" or "random_forest"
            include_treatment: Whether to include intervention as feature
            name: Custom name for this model
        """
        self.model_type = model_type
        self.include_treatment = include_treatment
        self.name = name or f"{model_type}_{'p_y_xt' if include_treatment else 'p_y_x'}"
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = None

    def _create_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=20, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        """Fit the model."""
        self.model = self._create_model()

        if self.model_type == "logistic":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def build_baseline_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build and evaluate all baseline models.

    Returns:
        Dict with model results
    """
    from recoverytwin.evaluation.metrics import compute_metrics, compute_calibration_metrics

    results = {}
    target = "recovered"

    model_configs = [
        ("logistic", False),
        ("random_forest", False),
        ("logistic", True),
        ("random_forest", True),
    ]

    for model_type, include_treatment in model_configs:
        variant = "p_y_xt" if include_treatment else "p_y_x"
        name = f"{model_type}_{variant}"
        print(f"\n  Training {name}...")

        # Prepare features
        X_train, feat_names, encoders = prepare_features(train_df, include_treatment=include_treatment)
        X_val, _, _ = prepare_features(val_df, include_treatment=include_treatment,
                                        exclude_cols=set())
        X_test, _, _ = prepare_features(test_df, include_treatment=include_treatment,
                                         exclude_cols=set())

        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        # Build and fit
        model = BaselineModel(model_type=model_type, include_treatment=include_treatment, name=name)
        model.feature_names = feat_names
        model.label_encoders = encoders
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate on all splits
        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
        train_proba = model.predict_proba(X_train)

        val_metrics = compute_metrics(y_val, val_proba)
        test_metrics = compute_metrics(y_test, test_proba)
        train_metrics = compute_metrics(y_train, train_proba)

        # Calibration metrics
        val_cal = compute_calibration_metrics(y_val, val_proba)
        test_cal = compute_calibration_metrics(y_test, test_proba)

        results[name] = {
            "model": model,
            "feature_names": feat_names,
            "n_features": len(feat_names),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_calibration": val_cal,
            "test_calibration": test_cal,
            "include_treatment": include_treatment,
        }

        print(f"    Val  PR-AUC: {val_metrics['pr_auc']:.4f}  ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"    Test PR-AUC: {test_metrics['pr_auc']:.4f}  ROC-AUC: {test_metrics['roc_auc']:.4f}")

    return results


def print_baseline_leaderboard(results: Dict[str, Any]):
    """Print model leaderboard."""
    print("\n" + "=" * 80)
    print("PHASE 3 - BASELINE MODEL LEADERBOARD")
    print("=" * 80)

    header = f"{'Model':<30} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'Brier':>8} {'ECE':>8}"
    print(header)
    print("-" * 80)

    # Sort by test PR-AUC
    sorted_models = sorted(results.items(), key=lambda x: x[1]["test_metrics"]["pr_auc"], reverse=True)

    for name, res in sorted_models:
        m = res["test_metrics"]
        c = res["test_calibration"]
        print(f"{name:<30} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} "
              f"{m['f1']:>8.4f} {m['brier_score']:>8.4f} {c['ece']:>8.4f}")

    print("=" * 80)
