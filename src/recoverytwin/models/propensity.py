"""
Propensity model: P(T | X) using only pre-treatment features.

Multiclass classification of treatment assignment.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple
from recoverytwin.evaluation.metrics import compute_ece


def prepare_pretreatment_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Prepare pre-treatment features only.

    Excludes: recovered, recovery_time_hours, revenue_recovered, intervention_cost,
    true_best_intervention, potential_outcomes, propensities, customer_fatigue,
    intervention (the treatment itself)
    """
    blocked = {
        "recovered", "recovery_time_hours", "revenue_recovered", "intervention_cost",
        "true_best_intervention", "potential_outcome_0", "potential_outcome_1",
        "potential_outcome_2", "potential_outcome_3",
        "propensity_0", "propensity_1", "propensity_2", "propensity_3",
        "customer_fatigue", "intervention",
        "payment_id", "timestamp",
    }

    feature_cols = [c for c in df.columns if c not in blocked]
    X = df[feature_cols].copy()

    # Encode categoricals
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, feature_cols


def build_propensity_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build multiclass propensity model P(T|X).

    Returns:
        Dict with model, metrics, and propensity distributions
    """
    from recoverytwin.evaluation.metrics import compute_metrics

    print("\n--- Building Propensity Model P(T|X) ---")

    # Prepare features
    X_train, feat_names = prepare_pretreatment_features(train_df)
    X_val, _ = prepare_pretreatment_features(val_df)
    X_test, _ = prepare_pretreatment_features(test_df)

    y_train = train_df["intervention"].values
    y_val = val_df["intervention"].values
    y_test = test_df["intervention"].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Fit multiclass logistic regression
    model = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    train_proba = model.predict_proba(X_train_scaled)
    val_proba = model.predict_proba(X_val_scaled)
    test_proba = model.predict_proba(X_test_scaled)

    # Accuracy
    train_acc = (train_pred == y_train).mean()
    val_acc = (val_pred == y_val).mean()
    test_acc = (test_pred == y_test).mean()

    # Log loss
    from sklearn.metrics import log_loss
    val_log_loss = log_loss(y_val, val_proba)
    test_log_loss = log_loss(y_test, test_proba)

    # Multiclass Brier score
    from sklearn.preprocessing import label_binarize
    n_classes = 4
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    test_brier = float(np.mean(np.sum((test_proba - y_test_bin) ** 2, axis=1)))

    # Propensity distributions on test set
    propensity_dist = {}
    for t in range(n_classes):
        mask = y_test == t
        propensity_dist[t] = {
            "n": int(mask.sum()),
            "pct": float(mask.mean()),
        }

    # ESS using inverse propensity weights
    ess_values = {}
    for t in range(n_classes):
        mask = y_test == t
        if mask.sum() > 0:
            props = np.clip(test_proba[mask, t], 1e-6, 1.0)
            weights = 1.0 / props
            ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
            ess_values[t] = {
                "ess": ess,
                "ess_ratio": ess / mask.sum(),
            }

    results = {
        "model": model,
        "scaler": scaler,
        "feature_names": feat_names,
        "n_features": len(feat_names),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "val_log_loss": float(val_log_loss),
        "test_log_loss": float(test_log_loss),
        "test_brier": test_brier,
        "test_propensity_distribution": propensity_dist,
        "test_ess": ess_values,
    }

    print(f"  Features: {len(feat_names)}")
    print(f"  Train accuracy: {train_acc:.4f}  (random baseline: {1/n_classes:.4f})")
    print(f"  Val accuracy:   {val_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Test log loss:  {test_log_loss:.4f}")
    print(f"  Test Brier:     {test_brier:.4f}")

    print("\n  Treatment distribution (test):")
    for t in range(n_classes):
        info = propensity_dist[t]
        ess_info = ess_values.get(t, {})
        print(f"    Treatment {t}: n={info['n']}, pct={info['pct']:.1%}, "
              f"ESS_ratio={ess_info.get('ess_ratio', 0):.3f}")

    return results
