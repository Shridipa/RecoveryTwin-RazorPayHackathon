"""
Cox Proportional Hazards model for time-to-recovery.

Estimates h(t | X, T) where T is the intervention.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from typing import Dict, Any, Tuple, Optional
from recoverytwin.data.feature_availability import get_model_features
from sklearn.preprocessing import LabelEncoder


def prepare_cox_features(
    df: pd.DataFrame,
    include_treatment: bool = True,
    exclude_cols: set = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Prepare features for Cox PH model.

    Only pre-treatment features + optionally treatment.
    Excludes survival outcomes and hidden fields.
    """
    from recoverytwin.survival.kaplan_meier import prepare_survival_data

    data = prepare_survival_data(df)

    blocked = {
        # Outcomes / survival targets
        "recovered", "recovery_time_hours", "revenue_recovered",
        "intervention_cost", "duration", "event",
        # Hidden ground truth
        "customer_fatigue", "true_best_intervention",
        "propensity_0", "propensity_1", "propensity_2", "propensity_3",
        "potential_outcome_0", "potential_outcome_1",
        "potential_outcome_2", "potential_outcome_3",
        # IDs
        "payment_id", "merchant_id", "customer_id", "timestamp",
    }
    if exclude_cols:
        blocked.update(exclude_cols)

    feature_cols = [c for c in data.columns if c not in blocked]

    if not include_treatment and "intervention" in feature_cols:
        feature_cols.remove("intervention")

    X = data[feature_cols].copy()

    # Encode categoricals
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, feature_cols, data["duration"].values, data["event"].values, label_encoders


class CoxPHModel:
    """Cox Proportional Hazards wrapper."""

    def __init__(self, include_treatment: bool = False, penalizer: float = 0.01):
        self.include_treatment = include_treatment
        self.penalizer = penalizer
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, duration: np.ndarray, event: np.ndarray):
        """Fit the Cox PH model."""
        self.feature_names = list(X.columns)

        cox_df = X.copy()
        cox_df["duration"] = duration
        cox_df["event"] = event

        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(cox_df, duration_col="duration", event_col="event")
        return self

    def predict_partial_hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Predict partial hazard (relative risk)."""
        return self.model.predict_partial_hazard(X).values.flatten()

    def predict_survival_function(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict survival function."""
        return self.model.predict_survival_function(X)

    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median survival time."""
        return self.model.predict_median(X).values.flatten()

    def score(self, X: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> float:
        """Compute concordance index."""
        partial_hazard = self.predict_partial_hazard(X)
        return concordance_index(duration, -partial_hazard, event)

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Get hazard ratios with confidence intervals."""
        return self.model.hazard_ratios_


def build_cox_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build Cox PH models for P(recovery | X) and P(recovery | X, T).

    Returns:
        Dict with model results
    """
    print("\n--- Building Cox PH Models ---")

    results = {}

    for include_t in [False, True]:
        variant = "h_t_xt" if include_t else "h_t_x"
        name = f"cox_{variant}"
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

        model = CoxPHModel(include_treatment=include_t, penalizer=0.01)
        model.fit(X_train, dur_train, evt_train)

        # Evaluate
        train_cindex = model.score(X_train, dur_train, evt_train)
        val_cindex = model.score(X_val, dur_val, evt_val)
        test_cindex = model.score(X_test, dur_test, evt_test)

        # Hazard ratios
        hr = model.get_hazard_ratios()

        print(f"    Train C-index: {train_cindex:.4f}")
        print(f"    Val C-index:   {val_cindex:.4f}")
        print(f"    Test C-index:  {test_cindex:.4f}")

        if include_t:
            print(f"\n    Hazard Ratios:")
            for idx in hr.index:
                print(f"      {idx}: {hr[idx]:.4f}")

        results[name] = {
            "model": model,
            "feature_names": feat_names,
            "n_features": len(feat_names),
            "train_cindex": float(train_cindex),
            "val_cindex": float(val_cindex),
            "test_cindex": float(test_cindex),
            "hazard_ratios": {k: float(v) for k, v in hr.items()},
            "include_treatment": include_t,
        }

    return results
