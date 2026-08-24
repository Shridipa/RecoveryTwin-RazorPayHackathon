"""
Random Survival Forest for time-to-recovery.

Non-linear survival model using scikit-survival.
"""

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from typing import Dict, Any, Tuple


def prepare_rsf_data(
    X: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for scikit-survival models.

    sksurv needs structured arrays with (event, time) dtype.

    Returns:
        X_array: Feature matrix (numpy)
        y_structured: Structured array (event, time)
    """
    X_array = np.asarray(X, dtype=np.float32)

    # Create structured array for sksurv
    y_structured = np.array(
        [(bool(e), float(d)) for e, d in zip(event, duration)],
        dtype=[("event", bool), ("time", float)],
    )

    return X_array, y_structured


class RSFModel:
    """Random Survival Forest wrapper."""

    def __init__(self, include_treatment: bool = False, n_estimators: int = 200,
                 max_depth: int = 10, min_samples_leaf: int = 20, random_state: int = 42):
        self.include_treatment = include_treatment
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray,
            X_val: np.ndarray = None, duration_val: np.ndarray = None,
            event_val: np.ndarray = None):
        """Fit the RSF model."""
        X_arr, y_train = prepare_rsf_data(X, duration, event)

        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_arr, y_train)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk score (higher = faster event)."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr)

    def score(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray,
              max_samples: int = 10000) -> float:
        """Compute concordance index."""
        X_arr, y_test = prepare_rsf_data(X, duration, event)

        # Subsample if too large for prediction
        if len(X_arr) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_arr), max_samples, replace=False)
            X_sub = X_arr[idx]
            y_sub = y_test[idx]
        else:
            X_sub = X_arr
            y_sub = y_test

        risk_scores = self.model.predict(X_sub)
        result = concordance_index_censored(
            y_sub["event"], y_sub["time"], risk_scores
        )
        return result[0]  # C-index


def build_rsf_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build Random Survival Forest models.

    Returns:
        Dict with model results
    """
    from recoverytwin.survival.cox_model import prepare_cox_features

    print("\n--- Building Random Survival Forest Models ---")

    results = {}

    for include_t in [False, True]:
        variant = "rsf_xt" if include_t else "rsf_x"
        name = f"rsf_{variant}"
        print(f"\n  Training {name}...")

        X_train, feat_names, dur_train, evt_train, _ = prepare_cox_features(
            train_df, include_treatment=include_t
        )
        X_val, _, dur_val, evt_val, _ = prepare_cox_features(
            val_df, include_treatment=include_t
        )
        X_test, _, dur_test, evt_test, _ = prepare_cox_features(
            test_df, include_treatment=include_t
        )

        model = RSFModel(
            include_treatment=include_t,
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=30,
            random_state=42,
        )

        # Subsample training data if too large (RSF memory management)
        max_train = 15000
        if len(X_train) > max_train:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), max_train, replace=False)
            X_train_sub = X_train.values[idx]
            dur_train_sub = dur_train[idx]
            evt_train_sub = evt_train[idx]
        else:
            X_train_sub = X_train.values
            dur_train_sub = dur_train
            evt_train_sub = evt_train

        model.fit(
            X_train_sub, dur_train_sub, evt_train_sub,
            X_val.values, dur_val, evt_val,
        )

        train_cindex = model.score(X_train.values, dur_train, evt_train)
        val_cindex = model.score(X_val.values, dur_val, evt_val)
        test_cindex = model.score(X_test.values, dur_test, evt_test)

        print(f"    Train C-index: {train_cindex:.4f}")
        print(f"    Val C-index:   {val_cindex:.4f}")
        print(f"    Test C-index:  {test_cindex:.4f}")

        results[name] = {
            "model": model,
            "feature_names": feat_names,
            "n_features": len(feat_names),
            "train_cindex": float(train_cindex),
            "val_cindex": float(val_cindex),
            "test_cindex": float(test_cindex),
            "include_treatment": include_t,
        }

    return results
