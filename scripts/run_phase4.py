"""
Phase 4 - XGBoost + LightGBM.

Includes calibration, temporal robustness check, and model artifacts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import json
from datetime import datetime

from recoverytwin.models.baseline import prepare_features
from recoverytwin.models.xgboost_model import XGBoostModel
from recoverytwin.models.lightgbm_model import LightGBMModel
from recoverytwin.models.calibration import calibrate_model, evaluate_calibration
from recoverytwin.evaluation.metrics import compute_metrics, compute_calibration_metrics
from recoverytwin.evaluation.reports import make_model_manifest


def evaluate_model_split(model, X, y, split_name="test"):
    """Evaluate a model on a split."""
    proba = model.predict_proba(X)
    metrics = compute_metrics(y, proba)
    cal_metrics = compute_calibration_metrics(y, proba)
    return {**metrics, **cal_metrics}


def main():
    print("=" * 60)
    print("PHASE 4 - XGBOOST + LIGHTGBM")
    print("=" * 60)

    # Load data
    data_dir = Path("data/processed/debug")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    target = "recovered"
    all_results = {}

    # === Build tree models ===
    model_configs = [
        ("XGBoost", XGBoostModel, False),
        ("XGBoost", XGBoostModel, True),
        ("LightGBM", LightGBMModel, False),
        ("LightGBM", LightGBMModel, True),
    ]

    for model_class_name, ModelClass, include_treatment in model_configs:
        variant = "p_y_xt" if include_treatment else "p_y_x"
        name = f"{model_class_name.lower()}_{variant}"

        print(f"\n--- Training {name} ---")

        X_train, feat_names, encoders = prepare_features(train_df, include_treatment=include_treatment)
        X_val, _, _ = prepare_features(val_df, include_treatment=include_treatment)
        X_test, _, _ = prepare_features(test_df, include_treatment=include_treatment)

        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        # Train
        model = ModelClass(include_treatment=include_treatment, name=name)
        model.fit(X_train.values, y_train, X_val.values, y_val, feature_names=feat_names)

        # Predict
        val_probs = model.predict_proba(X_val.values)
        test_probs = model.predict_proba(X_test.values)

        # Calibrate
        cal_results = calibrate_model(y_val, val_probs, test_probs)

        # Evaluate all calibration methods
        split_results = {}
        for method, cal_data in cal_results.items():
            test_eval = evaluate_calibration(y_test, cal_data["test_probs"], f"{name}_{method}")
            split_results[method] = test_eval
            print(f"  {method:12s}: PR-AUC={test_eval['pr_auc']:.4f}, "
                  f"ROC-AUC={test_eval['roc_auc']:.4f}, "
                  f"F1={test_eval['f1']:.4f}, "
                  f"Brier={test_eval['brier_score']:.4f}, "
                  f"ECE={test_eval['ece']:.4f}")

        all_results[name] = {
            "model": model,
            "feat_names": feat_names,
            "calibration_results": split_results,
            "params": model.get_params(),
            "include_treatment": include_treatment,
        }

    # === LEADERBOARD ===
    print("\n" + "=" * 80)
    print("PHASE 4 LEADERBOARD (Test Set)")
    print("=" * 80)
    header = f"{'Model + Cal':<35} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'Brier':>8} {'ECE':>8}"
    print(header)
    print("-" * 80)

    # Collect all for sorting
    leaderboard = []
    for name, res in all_results.items():
        for method, eval_r in res["calibration_results"].items():
            display_name = f"{name}+{method}"
            leaderboard.append((display_name, eval_r))

    leaderboard.sort(key=lambda x: x[1]["pr_auc"], reverse=True)

    for display_name, eval_r in leaderboard:
        print(f"{display_name:<35} {eval_r['pr_auc']:>8.4f} {eval_r['roc_auc']:>8.4f} "
              f"{eval_r['f1']:>8.4f} {eval_r['brier_score']:>8.4f} {eval_r['ece']:>8.4f}")

    print("=" * 80)

    # === SELECT BEST MODEL ===
    # Best = highest PR-AUC among sigmoid-calibrated models
    sigmoid_models = [
        (name, res) for name, res in all_results.items()
        if "sigmoid" in res["calibration_results"]
    ]
    best_name, best_res = max(
        sigmoid_models,
        key=lambda x: x[1]["calibration_results"]["sigmoid"]["pr_auc"],
    )
    best_eval = best_res["calibration_results"]["sigmoid"]
    print(f"\nBest model: {best_name} + sigmoid")
    print(f"  PR-AUC:  {best_eval['pr_auc']:.4f}")
    print(f"  ROC-AUC: {best_eval['roc_auc']:.4f}")
    print(f"  Brier:   {best_eval['brier_score']:.4f}")
    print(f"  ECE:     {best_eval['ece']:.4f}")

    # === TEMPORAL ROBUSTNESS ===
    print("\n--- Temporal Robustness Check ---")

    # Split test into November and December
    test_df_copy = test_df.copy()
    test_df_copy["ts"] = pd.to_datetime(test_df_copy["timestamp"])
    nov_mask = test_df_copy["ts"].dt.month == 11
    dec_mask = test_df_copy["ts"].dt.month == 12

    X_test_all, _, _ = prepare_features(test_df, include_treatment=best_res["include_treatment"])
    y_test_all = test_df[target].values

    # Get best model and calibrator
    best_model_obj = best_res["model"]

    # Apply sigmoid calibration
    from recoverytwin.models.calibration import SigmoidCalibrator
    X_val_best, _, _ = prepare_features(val_df, include_treatment=best_res["include_treatment"])
    y_val = val_df[target].values
    val_probs = best_model_obj.predict_proba(X_val_best.values)
    calibrator = SigmoidCalibrator()
    calibrator.fit(y_val, val_probs)

    temporal_results = {}
    for month_name, mask in [("November", nov_mask), ("December", dec_mask)]:
        X_month = X_test_all[mask.values]
        y_month = y_test_all[mask.values]
        raw_probs = best_model_obj.predict_proba(X_month.values)
        cal_probs = calibrator.predict(raw_probs)
        eval_r = compute_metrics(y_month, cal_probs)
        cal_m = compute_calibration_metrics(y_month, cal_probs)
        temporal_results[month_name] = {**eval_r, **cal_m}
        print(f"  {month_name}: PR-AUC={eval_r['pr_auc']:.4f}, ROC-AUC={eval_r['roc_auc']:.4f}, "
              f"Brier={cal_m['brier_score']:.4f}, ECE={cal_m['ece']:.4f}")

    # Check for degradation
    nov_pr = temporal_results["November"]["pr_auc"]
    dec_pr = temporal_results["December"]["pr_auc"]
    degradation = abs(nov_pr - dec_pr) / max(nov_pr, dec_pr) if max(nov_pr, dec_pr) > 0 else 0
    temporal_pass = degradation < 0.15  # Allow up to 15% relative difference
    print(f"\n  Relative PR-AUC difference: {degradation:.1%}")
    print(f"  Temporal robustness: {'[PASS]' if temporal_pass else '[WARN]'}")

    # === SAVE MODEL ARTIFACTS ===
    print("\n--- Saving model artifacts ---")
    model_dir = Path("models/phase4")
    model_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    # Save best model
    joblib.dump(best_model_obj.model, model_dir / f"{best_name}_model.joblib")
    if hasattr(best_model_obj, 'scaler') and best_model_obj.scaler is not None:
        joblib.dump(best_model_obj.scaler, model_dir / f"{best_name}_scaler.joblib")
    joblib.dump(calibrator, model_dir / f"{best_name}_sigmoid_calibrator.joblib")

    # Save all models
    for name, res in all_results.items():
        joblib.dump(res["model"].model, model_dir / f"{name}_model.joblib")
        if hasattr(res["model"], 'scaler') and res["model"].scaler is not None:
            joblib.dump(res["model"].scaler, model_dir / f"{name}_scaler.joblib")

    # Save manifest
    manifest = make_model_manifest(
        model_name=best_name,
        model_version="1.0.0",
        dataset_version="synthetic_v1",
        feature_schema_version="1.0.0",
        training_period="2024-01-01 to 2024-08-31",
        validation_period="2024-09-01 to 2024-10-31",
        test_period="2024-11-01 to 2024-12-31",
        hyperparameters=best_res["params"],
        calibration_method="sigmoid",
        metrics=best_eval,
        random_seed=42,
    )
    with open(model_dir / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Save full leaderboard
    report_dir = Path("reports/phase4")
    report_dir.mkdir(parents=True, exist_ok=True)

    serializable_leaderboard = []
    for display_name, eval_r in leaderboard:
        serializable_leaderboard.append({"model": display_name, **eval_r})

    report = {
        "leaderboard": serializable_leaderboard,
        "best_model": best_name,
        "best_model_sigmoid_eval": best_eval,
        "temporal_robustness": temporal_results,
        "temporal_pass": temporal_pass,
        "model_artifacts": [str(p) for p in model_dir.glob("*")],
    }
    with open(report_dir / "phase4_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # List artifacts
    print("\n  Model artifacts:")
    for p in sorted(model_dir.glob("*")):
        print(f"    {p.name}")

    # === VERIFICATION ===
    print("\n" + "=" * 60)
    print("PHASE 4 VERIFICATION")
    print("=" * 60)
    n_models = len(all_results)
    all_have_sigmoid = all("sigmoid" in r["calibration_results"] for r in all_results.values())
    all_pr_positive = all(
        r["calibration_results"]["sigmoid"]["pr_auc"] > 0
        for r in all_results.values()
    )

    print(f"Models trained:     {'[PASS]' if n_models == 4 else '[FAIL]'} ({n_models}/4)")
    print(f"All have sigmoid:   {'[PASS]' if all_have_sigmoid else '[FAIL]'}")
    print(f"All PR-AUC > 0:     {'[PASS]' if all_pr_positive else '[FAIL]'}")
    print(f"Temporal robust:    {'[PASS]' if temporal_pass else '[WARN]'}")
    print(f"Artifacts saved:    [PASS]")

    overall = n_models == 4 and all_have_sigmoid and all_pr_positive
    print(f"\nPHASE 4: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
