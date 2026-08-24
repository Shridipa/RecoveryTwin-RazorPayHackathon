"""
Phase 3 - Baseline ML models.

Builds Logistic Regression and Random Forest for P(Y|X) and P(Y|X,T).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import json
from recoverytwin.models.baseline import build_baseline_models, print_baseline_leaderboard
from recoverytwin.evaluation.shap_analysis import compute_shap_values, sanity_check_shap


def main():
    print("=" * 60)
    print("PHASE 3 - BASELINE ML")
    print("=" * 60)

    # Load data
    data_dir = Path("data/processed/debug")
    if not (data_dir / "train.parquet").exists():
        print("[FAIL] Processed data not found. Run generate_and_validate.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature audit
    from recoverytwin.data.feature_availability import audit_features
    for include_t in [False, True]:
        audit = audit_features(train_df, include_treatment=include_t)
        variant = "P(Y|X,T)" if include_t else "P(Y|X)"
        print(f"\n  Features ({variant}): {audit['n_features']}, Blocked: {audit['n_blocked']}, Unknown: {audit['n_unknown']}")

    # Build models
    print("\n--- Building baseline models ---")
    results = build_baseline_models(train_df, val_df, test_df)

    # Print leaderboard
    print_baseline_leaderboard(results)

    # SHAP analysis for tree models
    print("\n--- SHAP Analysis ---")
    from recoverytwin.models.baseline import prepare_features
    X_val, feat_names, _ = prepare_features(val_df, include_treatment=True)

    for name, res in results.items():
        if "random_forest" in name:
            print(f"\n  SHAP for {name}:")
            try:
                shap_result = compute_shap_values(
                    res["model"].model, X_val.values, feature_names=feat_names
                )
                if sanity_check_shap(shap_result):
                    print(f"    SHAP sanity check: [PASS]")
                    print(f"    Top 5 features:")
                    for fname, imp in shap_result["top_10"][:5]:
                        print(f"      {fname}: {imp:.4f}")
                else:
                    print(f"    SHAP sanity check: [FAIL]")
            except Exception as e:
                print(f"    SHAP error: {e}")

    # Save results (without model objects)
    report_dir = Path("reports/phase3")
    report_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            "n_features": res["n_features"],
            "include_treatment": res["include_treatment"],
            "train_metrics": res["train_metrics"],
            "val_metrics": res["val_metrics"],
            "test_metrics": res["test_metrics"],
            "val_calibration": res["val_calibration"],
            "test_calibration": res["test_calibration"],
        }

    with open(report_dir / "baseline_report.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nReport saved to {report_dir / 'baseline_report.json'}")

    # Verification
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION")
    print("=" * 60)
    all_ok = all(r["test_metrics"]["pr_auc"] > 0 for r in results.values())
    print(f"Models trained: {'[PASS]' if len(results) == 4 else '[FAIL]'}")
    print(f"PR-AUC > 0:     {'[PASS]' if all_ok else '[FAIL]'}")
    print(f"Overall:        {'PASS' if all_ok and len(results) == 4 else 'FAIL'}")
    print("=" * 60)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
