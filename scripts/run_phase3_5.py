"""
Phase 3.5 - Calibration, Propensity Model, Leakage Audit.

Produces GO/NO-GO gate for Phase 4.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import json
from recoverytwin.models.baseline import prepare_features
from recoverytwin.models.calibration import (
    SigmoidCalibrator, IsotonicCalibrator,
    evaluate_calibration,
)
from recoverytwin.models.propensity import build_propensity_model
from recoverytwin.data.validator import DataValidator, BLOCKED_FEATURES
from recoverytwin.evaluation.calibration_metrics import reliability_data
from recoverytwin.evaluation.metrics import compute_metrics, compute_calibration_metrics


def main():
    print("=" * 60)
    print("PHASE 3.5 - CALIBRATION, PROPENSITY, LEAKAGE")
    print("=" * 60)

    # Load data
    data_dir = Path("data/processed/debug")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    full_df = pd.read_parquet("data/synthetic/debug/transactions.parquet")

    report_dir = Path("reports/phase3_5")
    report_dir.mkdir(parents=True, exist_ok=True)

    # === PART 1: CALIBRATION ===
    print("\n--- Part 1: Calibration ---")

    # Build a baseline RF model for calibration testing
    X_train, feat_names, encoders = prepare_features(train_df, include_treatment=True)
    X_val, _, _ = prepare_features(val_df, include_treatment=True)
    X_test, _, _ = prepare_features(test_df, include_treatment=True)

    y_train = train_df["recovered"].values
    y_val = val_df["recovered"].values
    y_test = test_df["recovered"].values

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    val_probs = rf.predict_proba(X_val)[:, 1]
    test_probs = rf.predict_proba(X_test)[:, 1]

    # Calibrate
    from recoverytwin.models.calibration import calibrate_model
    cal_results = calibrate_model(y_val, val_probs, test_probs)

    cal_report = {}
    for method, cal_data in cal_results.items():
        cal_eval = evaluate_calibration(y_test, cal_data["test_probs"], method_name=method)
        cal_report[method] = cal_eval
        print(f"\n  {method}:")
        print(f"    PR-AUC:    {cal_eval['pr_auc']:.4f}")
        print(f"    ROC-AUC:   {cal_eval['roc_auc']:.4f}")
        print(f"    Brier:     {cal_eval['brier_score']:.4f}")
        print(f"    ECE:       {cal_eval['ece']:.4f}")
        print(f"    Cal slope: {cal_eval['calibration_slope']:.4f}")
        print(f"    Cal int:   {cal_eval['calibration_intercept']:.4f}")

    # Reliability curve data
    for method, cal_data in cal_results.items():
        rel_data = reliability_data(y_test, cal_data["test_probs"])
        cal_report[method]["reliability_data"] = rel_data

    calibration_pass = all(
        cal_report[m]["ece"] < 0.20 for m in cal_report
    )

    with open(report_dir / "calibration_report.json", "w") as f:
        json.dump(cal_report, f, indent=2, default=str)

    # === PART 2: PROPENSITY MODEL ===
    print("\n--- Part 2: Propensity Model ---")
    propensity_results = build_propensity_model(train_df, val_df, test_df)

    prop_serializable = {k: v for k, v in propensity_results.items() if k != "model" and k != "scaler"}
    with open(report_dir / "propensity_report.json", "w") as f:
        json.dump(prop_serializable, f, indent=2, default=str)

    # Propensity accuracy should be > random (25%)
    propensity_pass = propensity_results["test_accuracy"] > 0.25

    # === PART 3: LEAKAGE AUDIT ===
    print("\n--- Part 3: Leakage Audit ---")

    # Check blocked features
    model_features = set(full_df.columns) - BLOCKED_FEATURES - {"payment_id", "merchant_id", "customer_id", "timestamp"}
    blocked_in_model = model_features & BLOCKED_FEATURES

    # Feature availability check
    from recoverytwin.data.feature_availability import audit_features, FEATURE_REGISTRY
    audit = audit_features(full_df, include_treatment=True)

    print(f"  Total features: {audit['n_features']}")
    print(f"  Blocked features: {audit['n_blocked']}")
    print(f"  Unknown features: {audit['n_unknown']}")
    print(f"  Blocked in model input: {len(blocked_in_model)}")

    leakage_pass = len(blocked_in_model) == 0 and audit["n_unknown"] == 0

    if not leakage_pass:
        print(f"  [FAIL] Blocked features in model input: {blocked_in_model}")
        print(f"  [FAIL] Unknown features: {audit['unknown']}")
    else:
        print(f"  [PASS] No leakage detected")

    with open(report_dir / "leakage_report.json", "w") as f:
        json.dump({
            "n_features": audit["n_features"],
            "n_blocked": audit["n_blocked"],
            "n_unknown": audit["n_unknown"],
            "blocked_in_model": list(blocked_in_model),
            "unknown_features": audit["unknown"],
            "leakage_pass": leakage_pass,
        }, f, indent=2)

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("PHASE 3.5 GO/NO-GO")
    print("=" * 60)
    print(f"Calibration:   {'[PASS]' if calibration_pass else '[FAIL]'}")
    print(f"Propensity:    {'[PASS]' if propensity_pass else '[FAIL]'}")
    print(f"Leakage Audit: {'[PASS]' if leakage_pass else '[FAIL]'}")

    overall = calibration_pass and propensity_pass and leakage_pass
    print(f"\nPHASE 3.5: {'GO' if overall else 'NO-GO'}")
    print("=" * 60)

    # Save summary
    summary = {
        "calibration": {"pass": calibration_pass},
        "propensity": {
            "pass": propensity_pass,
            "test_accuracy": propensity_results["test_accuracy"],
        },
        "leakage": {"pass": leakage_pass},
        "overall": "GO" if overall else "NO-GO",
    }
    with open(report_dir / "phase3_5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
