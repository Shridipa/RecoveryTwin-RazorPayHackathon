"""
Generate synthetic dataset, validate, and split.

Phase 1 + Phase 2 script.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recoverytwin.simulator.payment_environment import PaymentEnvironment, generate_and_save
from recoverytwin.data.splitter import split_and_save
from recoverytwin.data.validator import DataValidator, print_validation_summary


def main():
    print("=" * 60)
    print("RECOVERYTWIN - Data Generation & Validation")
    print("=" * 60)

    # Step 1: Generate
    print("\n--- Step 1: Generating synthetic data ---")
    df, metadata = generate_and_save(
        config_path="configs/data.yaml",
        output_dir="data/synthetic/debug",
    )

    print(f"\nCustomers:       {metadata['n_customers']}")
    print(f"Merchants:        {metadata['n_merchants']}")
    print(f"Transactions:     {metadata['n_transactions']}")
    print(f"Recovery rate:    {metadata['recovery_rate']:.4f}")
    print(f"Revenue at risk:  {metadata['revenue_at_risk']:,.2f}")
    print(f"Revenue recovered:{metadata['revenue_recovered']:,.2f}")
    print(f"\nTreatment distribution:")
    for name, count in metadata["treatment_distribution"].items():
        pct = metadata["treatment_pct"][name]
        print(f"  {name}: {count} ({pct:.1%})")

    # Step 2: Split
    print("\n--- Step 2: Temporal split ---")
    train, val, test, split_info = split_and_save(
        df,
        output_dir="data/processed/debug",
        config_path="configs/data.yaml",
    )
    print(f"Train rows:      {split_info['train_rows']}")
    print(f"Validation rows: {split_info['validation_rows']}")
    print(f"Test rows:       {split_info['test_rows']}")
    print(f"Train recovery:  {split_info['train_recovery_rate']:.4f}")
    print(f"Val recovery:    {split_info['val_recovery_rate']:.4f}")
    print(f"Test recovery:   {split_info['test_recovery_rate']:.4f}")

    # Step 3: Validate
    print("\n--- Step 3: Running validation ---")
    validator = DataValidator()
    summary = validator.run_all(df, train, val, test)
    print_validation_summary(summary)

    # Save validation report
    import json
    report_dir = Path("reports/phase2")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "validation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nValidation report saved to {report_dir / 'validation_report.json'}")

    if not summary["all_pass"]:
        print("\n[FAIL] Validation failed. Fix issues before proceeding.")
        sys.exit(1)
    else:
        print("\n[PASS] All validation checks passed.")

    return summary


if __name__ == "__main__":
    main()
