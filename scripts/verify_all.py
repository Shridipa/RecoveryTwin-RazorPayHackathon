"""
RecoveryTwin Complete System Verification.

Runs all phase checks and produces a comprehensive report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pandas as pd
import numpy as np
from datetime import datetime


def check_phase0():
    """Phase 0: Environment + Configuration."""
    checks = []

    # Check configs exist
    for cfg in ["configs/data.yaml", "configs/model.yaml", "configs/policy.yaml"]:
        checks.append(("Config: " + cfg, Path(cfg).exists()))

    # Check src structure
    src_dirs = [
        "src/recoverytwin",
        "src/recoverytwin/simulator",
        "src/recoverytwin/data",
        "src/recoverytwin/models",
        "src/recoverytwin/evaluation",
        "src/recoverytwin/causal",
        "src/recoverytwin/survival",
        "src/recoverytwin/decision",
    ]
    for d in src_dirs:
        checks.append(("Dir: " + d, Path(d).is_dir()))

    # Check requirements.txt
    checks.append(("requirements.txt", Path("requirements.txt").exists()))

    # Check dependencies importable
    deps = ["numpy", "pandas", "sklearn", "scipy", "pyarrow", "xgboost", "lightgbm", "shap"]
    for dep in deps:
        try:
            __import__(dep)
            checks.append((f"Import: {dep}", True))
        except ImportError:
            checks.append((f"Import: {dep}", False))

    return checks


def check_phase1():
    """Phase 1: Data generation."""
    checks = []
    data_path = Path("data/synthetic/debug/transactions.parquet")

    if data_path.exists():
        checks.append(("Data file exists", True))
        df = pd.read_parquet(data_path)
        checks.append(("Has rows", len(df) > 0))
        checks.append((f"Row count ({len(df)})", len(df) >= 40000))

        # Check required columns
        required = ["intervention", "recovered", "recovery_time_hours",
                     "revenue_recovered", "intervention_cost"]
        for col in required:
            checks.append((f"Column: {col}", col in df.columns))

        # Propensity columns
        for i in range(4):
            checks.append((f"Propensity: {i}", f"propensity_{i}" in df.columns))

        # Hidden columns
        for i in range(4):
            checks.append((f"Potential outcome: {i}", f"potential_outcome_{i}" in df.columns))
        checks.append(("true_best_intervention", "true_best_intervention" in df.columns))

        # Treatment distribution
        treatment_dist = df["intervention"].value_counts(normalize=True)
        checks.append(("All 4 treatments", len(treatment_dist) == 4))
    else:
        checks.append(("Data file exists", False))

    return checks


def check_phase2():
    """Phase 2: Validation."""
    checks = []
    report_path = Path("reports/phase2/validation_report.json")

    if report_path.exists():
        checks.append(("Validation report exists", True))
        with open(report_path) as f:
            report = json.load(f)
        checks.append((f"All checks pass ({report['passed']}/{report['total_checks']})",
                       report["all_pass"]))
    else:
        checks.append(("Validation report exists", False))

    # Check temporal splits
    splits = ["data/processed/debug/train.parquet",
              "data/processed/debug/validation.parquet",
              "data/processed/debug/test.parquet"]
    for s in splits:
        checks.append((f"Split: {Path(s).name}", Path(s).exists()))

    if all(Path(s).exists() for s in splits):
        train = pd.read_parquet(splits[0])
        val = pd.read_parquet(splits[1])
        test = pd.read_parquet(splits[2])

        train_max = pd.to_datetime(train["timestamp"]).max()
        val_min = pd.to_datetime(val["timestamp"]).min()
        test_min = pd.to_datetime(test["timestamp"]).min()

        checks.append(("Train < Val", train_max < val_min))
        checks.append(("Val < Test", pd.to_datetime(val["timestamp"]).max() < test_min))
        checks.append((f"Train rows ({len(train)})", len(train) > 10000))
        checks.append((f"Val rows ({len(val)})", len(val) > 5000))
        checks.append((f"Test rows ({len(test)})", len(test) > 5000))

    return checks


def check_phase25():
    """Phase 2.5: Positivity."""
    checks = []
    report_path = Path("reports/phase2_5/overlap_report.json")

    if report_path.exists():
        checks.append(("Overlap report exists", True))
        with open(report_path) as f:
            report = json.load(f)

        checks.append(("Positivity pass", report.get("positivity_pass", False)))
        checks.append(("Support pass", report.get("support_pass", False)))

        # Treatment distribution
        dist = report.get("treatment_percentages", {})
        checks.append((f"Control ({dist.get('0', 0):.1%})",
                       0.05 <= dist.get("0", 0) <= 0.20))
    else:
        checks.append(("Overlap report exists", False))

    return checks


def check_phase3():
    """Phase 3: Baseline ML."""
    checks = []
    report_path = Path("reports/phase3/baseline_report.json")

    if report_path.exists():
        checks.append(("Baseline report exists", True))
        with open(report_path) as f:
            report = json.load(f)

        checks.append((f"Models trained ({len(report)})", len(report) >= 4))

        for name, res in report.items():
            checks.append((f"{name} PR-AUC ({res['test_metrics']['pr_auc']:.4f})",
                           res["test_metrics"]["pr_auc"] > 0))
    else:
        checks.append(("Baseline report exists", False))

    return checks


def check_phase35():
    """Phase 3.5: Calibration, Propensity, Leakage."""
    checks = []
    report_path = Path("reports/phase3_5/phase3_5_summary.json")

    if report_path.exists():
        checks.append(("Phase 3.5 summary exists", True))
        with open(report_path) as f:
            summary = json.load(f)

        checks.append(("Calibration", summary.get("calibration", {}).get("pass", False)))
        checks.append(("Propensity", summary.get("propensity", {}).get("pass", False)))
        checks.append(("Leakage", summary.get("leakage", {}).get("pass", False)))
        checks.append(("Overall GO", summary.get("overall") == "GO"))
    else:
        checks.append(("Phase 3.5 summary exists", False))

    return checks


def check_phase4():
    """Phase 4: XGBoost + LightGBM."""
    checks = []
    report_path = Path("reports/phase4/phase4_report.json")

    if report_path.exists():
        checks.append(("Phase 4 report exists", True))
        with open(report_path) as f:
            report = json.load(f)

        checks.append(("Best model selected", report.get("best_model") is not None))
        checks.append(("Temporal pass", report.get("temporal_pass", False)))

        # Check model artifacts
        model_dir = Path("models/phase4")
        if model_dir.exists():
            artifacts = list(model_dir.glob("*.joblib"))
            checks.append((f"Model artifacts ({len(artifacts)})", len(artifacts) >= 4))
            checks.append(("Manifest", (model_dir / "model_manifest.json").exists()))
        else:
            checks.append(("Model artifacts", False))
    else:
        checks.append(("Phase 4 report exists", False))

    return checks


def check_data():
    """Data quality checks."""
    checks = []
    data_path = Path("data/synthetic/debug/transactions.parquet")
    if not data_path.exists():
        checks.append(("Data exists", False))
        return checks

    df = pd.read_parquet(data_path)
    checks.append(("No missing values", not df.isnull().any().any()))
    checks.append(("No duplicate payments", not df["payment_id"].duplicated().any()))
    checks.append(("Valid treatments", df["intervention"].isin([0, 1, 2, 3]).all()))
    checks.append(("Valid recovery", df["recovered"].isin([0, 1]).all()))
    checks.append(("Positive amounts", (df["amount"] > 0).all()))
    checks.append(("Non-negative revenue", (df["revenue_recovered"] >= 0).all()))

    return checks


def check_leakage():
    """Leakage checks."""
    checks = []
    from recoverytwin.data.validator import BLOCKED_FEATURES

    data_path = Path("data/synthetic/debug/transactions.parquet")
    if not data_path.exists():
        checks.append(("Data exists for leakage check", False))
        return checks

    df = pd.read_parquet(data_path)

    # All blocked features should be in dataset (for evaluation) but not in model input
    blocked_in_data = BLOCKED_FEATURES & set(df.columns)
    checks.append((f"Blocked features in dataset ({len(blocked_in_data)})", len(blocked_in_data) > 0))

    # Propensity columns
    prop_cols = [c for c in df.columns if c.startswith("propensity_")]
    checks.append((f"Propensity columns ({len(prop_cols)})", len(prop_cols) == 4))

    # Hidden outcomes
    hidden_cols = [c for c in df.columns if c.startswith("potential_outcome_")]
    checks.append((f"Potential outcomes ({len(hidden_cols)})", len(hidden_cols) == 4))

    return checks


def check_positivity():
    """Positivity checks."""
    checks = []
    report_path = Path("reports/phase2_5/overlap_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        ess = report.get("ess", {})
        for t, info in ess.items():
            ratio = info.get("ess_ratio", 0)
            checks.append((f"ESS ratio T{t} ({ratio:.3f})", ratio > 0.5))
    else:
        checks.append(("Overlap report", False))

    return checks


def check_calibration():
    """Calibration checks."""
    checks = []
    report_path = Path("reports/phase3_5/calibration_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        for method, eval_r in report.items():
            if isinstance(eval_r, dict) and "ece" in eval_r:
                checks.append((f"{method} ECE ({eval_r['ece']:.4f})", eval_r["ece"] < 0.20))
    else:
        checks.append(("Calibration report", False))

    return checks


def check_models():
    """Model artifact checks."""
    checks = []
    model_dir = Path("models/phase4")
    if model_dir.exists():
        manifests = list(model_dir.glob("model_manifest.json"))
        checks.append(("Manifest file", len(manifests) > 0))
        joblib_files = list(model_dir.glob("*.joblib"))
        checks.append((f"Joblib files ({len(joblib_files)})", len(joblib_files) >= 4))
    else:
        checks.append(("Model directory", False))

    return checks


def check_tests():
    """Test suite check."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    passed = result.returncode == 0
    output = result.stdout.strip()
    return [("Test suite", passed), ("Output", True)]


def run_verification():
    """Run complete verification."""
    print("=" * 60)
    print("             RECOVERYTWIN SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_sections = [
        ("PHASE 0", check_phase0),
        ("PHASE 1", check_phase1),
        ("PHASE 2", check_phase2),
        ("PHASE 2.5", check_phase25),
        ("PHASE 3", check_phase3),
        ("PHASE 3.5", check_phase35),
        ("PHASE 4", check_phase4),
    ]

    detail_sections = [
        ("DATA", check_data),
        ("LEAKAGE", check_leakage),
        ("POSITIVITY", check_positivity),
        ("CALIBRATION", check_calibration),
        ("MODELS", check_models),
        ("TESTS", check_tests),
    ]

    results = {}
    failures = []

    for section_name, check_fn in all_sections + detail_sections:
        print(f"\n--- {section_name} ---")
        try:
            checks = check_fn()
            all_pass = all(passed for _, passed in checks)
            results[section_name] = all_pass

            for check_name, passed in checks:
                status = "[PASS]" if passed else "[FAIL]"
                print(f"  {status} {check_name}")
                if not passed:
                    failures.append(f"{section_name}: {check_name}")

        except Exception as e:
            results[section_name] = False
            print(f"  [FAIL] Exception: {e}")
            failures.append(f"{section_name}: Exception - {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("             RECOVERYTWIN SYSTEM VERIFICATION")
    print("=" * 60)

    for section_name in [s[0] for s in all_sections + detail_sections]:
        status = "[PASS]" if results.get(section_name, False) else "[FAIL]"
        print(f"{section_name:<14} {status}")

    print("=" * 60)

    overall = all(results.values())
    print(f"OVERALL STATUS:")
    print(f"{'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    if failures:
        print("\nFAILURE DETAILS:")
        for f in failures[:10]:
            print(f"  - {f}")

    # Save results
    report_path = Path("reports/verification_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "overall": overall,
            "failures": failures,
        }, f, indent=2)

    return overall


if __name__ == "__main__":
    overall = run_verification()
    sys.exit(0 if overall else 1)
