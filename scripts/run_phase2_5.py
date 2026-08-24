"""
Phase 2.5 - Positivity and overlap analysis.

Verifies that treatment allocation has adequate overlap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import json
from recoverytwin.evaluation.overlap import compute_overlap_diagnostics, print_overlap_report


def main():
    print("=" * 60)
    print("PHASE 2.5 - POSITIVITY / OVERLAP ANALYSIS")
    print("=" * 60)

    # Load data
    data_path = Path("data/synthetic/debug/transactions.parquet")
    if not data_path.exists():
        print("[FAIL] Data not found. Run generate_and_validate.py first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} transactions")

    # Compute overlap diagnostics
    print("\n--- Computing overlap diagnostics ---")
    overlap = compute_overlap_diagnostics(df)
    print_overlap_report(overlap)

    # Save report
    report_dir = Path("reports/phase2_5")
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "overlap_report.json", "w") as f:
        json.dump(overlap, f, indent=2, default=str)

    # GO/NO-GO decision
    positivity_pass = overlap["positivity_pass"]
    support_pass = overlap["support_pass"]

    print("\n" + "=" * 60)
    print("PHASE 2.5 GO/NO-GO")
    print("=" * 60)
    print(f"Positivity (ESS ratio > 0.5):  {'[PASS]' if positivity_pass else '[FAIL]'}")
    print(f"Support (min treatment > 5%):   {'[PASS]' if support_pass else '[FAIL]'}")

    overall = positivity_pass and support_pass
    print(f"\nPHASE 2.5: {'GO' if overall else 'NO-GO'}")
    print("=" * 60)

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
