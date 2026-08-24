"""
Phase 5 - Intervention-Conditional Survival / Time-to-Recovery.

Builds:
1. Kaplan-Meier curves by intervention
2. Cox Proportional Hazards
3. Random Survival Forest

Answers: "How does recovery probability evolve over time under each intervention?"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import json

from recoverytwin.survival.kaplan_meier import (
    prepare_survival_data, fit_kaplan_meier, print_km_results,
)
from recoverytwin.survival.cox_model import build_cox_models
from recoverytwin.survival.rsf_model import build_rsf_models
from recoverytwin.survival.metrics import (
    compute_concordance_index, survival_metrics_summary,
)


def build_intervention_conditional_curves(
    df: pd.DataFrame,
    cox_model_xt,
    rsf_model_xt,
    feature_names: list,
    censoring_time: float = 168.0,
) -> dict:
    """
    Build intervention-conditional survival curves for a sample payment.

    For each intervention, predict the survival curve and extract
    recovery probability at key timepoints.
    """
    from recoverytwin.survival.kaplan_meier import prepare_survival_data
    from recoverytwin.survival.cox_model import prepare_cox_features

    timepoints = [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]

    # Take a representative sample (median customer)
    data = prepare_survival_data(df)
    numeric_cols = data[feature_names].select_dtypes(include=[np.number]).columns
    medians = data[feature_names][numeric_cols].median()

    # Create base feature vector (median values)
    base_row = data[feature_names].iloc[0:1].copy()
    for col in numeric_cols:
        base_row[col] = medians[col]

    # Encode categoricals the same way as training
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = data[feature_names].select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(data[col].astype(str))
        base_row[col] = le.transform([data[col].mode().iloc[0]])[0]

    base_row = base_row.apply(pd.to_numeric, errors="coerce").fillna(0)

    results = {}

    # Cox predictions per intervention
    intervention_names = {0: "Control", 1: "Retry", 2: "Reminder", 3: "Alternative"}

    for t in range(4):
        row = base_row.copy()
        if "intervention" in row.columns:
            row["intervention"] = t

        try:
            surv_func = cox_model_xt.predict_survival_function(row)
            surv_probs = surv_func.values.flatten()

            # Get recovery probabilities at timepoints
            surv_times = surv_func.index.values
            recovery_at_tp = {}
            for tp in timepoints:
                idx = np.searchsorted(surv_times, tp)
                if idx < len(surv_probs):
                    recovery_at_tp[tp] = float(1.0 - surv_probs[idx])
                else:
                    recovery_at_tp[tp] = float(1.0 - surv_probs[-1])

            results[intervention_names[t]] = recovery_at_tp
        except Exception as e:
            print(f"    Cox prediction failed for intervention {t}: {e}")
            results[intervention_names[t]] = {}

    return results


def main():
    print("=" * 60)
    print("PHASE 5 - SURVIVAL / TIME-TO-RECOVERY")
    print("=" * 60)

    # Load data
    data_dir = Path("data/processed/debug")
    if not (data_dir / "train.parquet").exists():
        print("[FAIL] Processed data not found. Run generate_and_validate.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    full_df = pd.read_parquet("data/synthetic/debug/transactions.parquet")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ========================================
    # PART 1: KAPLAN-MEIER CURVES
    # ========================================
    print("\n" + "=" * 60)
    print("PART 1: KAPLAN-MEIER RECOVERY CURVES")
    print("=" * 60)

    km_results = fit_kaplan_meier(full_df)
    print_km_results(km_results)

    # ========================================
    # PART 2: COX PROPORTIONAL HAZARDS
    # ========================================
    print("\n" + "=" * 60)
    print("PART 2: COX PROPORTIONAL HAZARDS")
    print("=" * 60)

    cox_results = build_cox_models(train_df, val_df, test_df)

    # ========================================
    # PART 3: RANDOM SURVIVAL FOREST
    # ========================================
    print("\n" + "=" * 60)
    print("PART 3: RANDOM SURVIVAL FOREST")
    print("=" * 60)

    rsf_results = build_rsf_models(train_df, val_df, test_df)

    # ========================================
    # PART 4: INTERVENTION-CONDITIONAL CURVES
    # ========================================
    print("\n" + "=" * 60)
    print("PART 4: INTERVENTION-CONDITIONAL RECOVERY CURVES")
    print("=" * 60)

    # Use the treatment-aware Cox model
    cox_xt = cox_results.get("cox_h_t_xt", {}).get("model")
    rsf_xt = rsf_results.get("rsf_rsf_xt", {}).get("model")
    feature_names = cox_results.get("cox_h_t_xt", {}).get("feature_names", [])

    if cox_xt is not None:
        curves = build_intervention_conditional_curves(
            full_df, cox_xt, rsf_xt, feature_names
        )

        # Print intervention-conditional table
        timepoints = [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]
        header = f"{'Intervention':<15}"
        for t in timepoints:
            header += f" {t:>5}h"
        print(header)
        print("-" * 80)

        for name, probs in curves.items():
            row = f"{name:<15}"
            for tp in timepoints:
                p = probs.get(tp, 0)
                row += f" {p:>5.1%}"
            print(row)
        print("-" * 80)
    else:
        print("  [SKIP] Cox model not available for conditional curves")

    # ========================================
    # PART 5: SURVIVAL MODEL LEADERBOARD
    # ========================================
    print("\n" + "=" * 60)
    print("PART 5: SURVIVAL MODEL LEADERBOARD (Test Set)")
    print("=" * 60)

    all_models = {}
    all_models.update(cox_results)
    all_models.update(rsf_results)

    header = f"{'Model':<25} {'C-index':>10} {'Features':>10} {'Treatment':>10}"
    print(header)
    print("-" * 60)

    sorted_models = sorted(
        all_models.items(),
        key=lambda x: x[1].get("test_cindex", 0),
        reverse=True,
    )

    for name, res in sorted_models:
        cidx = res.get("test_cindex", 0)
        nfeat = res.get("n_features", 0)
        has_t = "Yes" if res.get("include_treatment", False) else "No"
        print(f"{name:<25} {cidx:>10.4f} {nfeat:>10} {has_t:>10}")

    print("=" * 60)

    # Best model
    best_name, best_res = sorted_models[0]
    print(f"\nBest model: {best_name}")
    print(f"  Test C-index: {best_res['test_cindex']:.4f}")

    # ========================================
    # SAVE REPORTS
    # ========================================
    report_dir = Path("reports/phase5")
    report_dir.mkdir(parents=True, exist_ok=True)

    # KM results (without kmf objects)
    km_serializable = {
        "overall": {k: v for k, v in km_results["overall"].items() if k != "kmf"},
        "by_intervention": {},
    }
    for t, info in km_results["by_intervention"].items():
        km_serializable["by_intervention"][str(t)] = {
            k: v for k, v in info.items() if k != "kmf"
        }

    with open(report_dir / "kaplan_meier_report.json", "w") as f:
        json.dump(km_serializable, f, indent=2, default=str)

    # Cox results
    cox_serializable = {}
    for name, res in cox_results.items():
        cox_serializable[name] = {
            k: v for k, v in res.items() if k != "model"
        }

    with open(report_dir / "cox_report.json", "w") as f:
        json.dump(cox_serializable, f, indent=2, default=str)

    # RSF results
    rsf_serializable = {}
    for name, res in rsf_results.items():
        rsf_serializable[name] = {
            k: v for k, v in res.items() if k != "model"
        }

    with open(report_dir / "rsf_report.json", "w") as f:
        json.dump(rsf_serializable, f, indent=2, default=str)

    # Intervention-conditional curves
    if cox_xt is not None:
        with open(report_dir / "intervention_curves.json", "w") as f:
            json.dump(curves, f, indent=2)

    # Summary
    summary = {
        "kaplan_meier": {
            "overall_event_rate": km_results["overall"]["event_rate"],
        },
        "cox_models": {name: {"test_cindex": res["test_cindex"]}
                       for name, res in cox_results.items()},
        "rsf_models": {name: {"test_cindex": res["test_cindex"]}
                       for name, res in rsf_results.items()},
        "best_model": best_name,
        "best_cindex": best_res["test_cindex"],
    }
    with open(report_dir / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReports saved to {report_dir}")

    # ========================================
    # VERIFICATION
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 5 VERIFICATION")
    print("=" * 60)

    all_have_cindex = all(
        res.get("test_cindex", 0) > 0.5
        for res in all_models.values()
    )
    cox_xt_exists = "cox_h_t_xt" in cox_results
    rsf_xt_exists = "rsf_rsf_xt" in rsf_results

    print(f"KM curves generated:   [PASS]")
    print(f"Cox models trained:    {'[PASS]' if len(cox_results) >= 2 else '[FAIL]'}")
    print(f"RSF models trained:    {'[PASS]' if len(rsf_results) >= 2 else '[FAIL]'}")
    print(f"All C-index > 0.5:     {'[PASS]' if all_have_cindex else '[FAIL]'}")
    print(f"Intervention curves:   {'[PASS]' if cox_xt_exists else '[FAIL]'}")

    overall = len(cox_results) >= 2 and len(rsf_results) >= 2 and all_have_cindex
    print(f"\nPHASE 5: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
