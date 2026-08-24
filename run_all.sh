#!/bin/bash
set -e

echo "==========================================="
echo "RecoveryTwin - Full Build & Verification"
echo "==========================================="

echo ""
echo "--- Installing dependencies ---"
pip install -r requirements.txt

echo ""
echo "--- Generating data ---"
python scripts/generate_and_validate.py

echo ""
echo "--- Phase 2.5: Positivity ---"
python scripts/run_phase2_5.py

echo ""
echo "--- Phase 3: Baseline ML ---"
python scripts/run_phase3.py

echo ""
echo "--- Phase 3.5: Calibration, Propensity, Leakage ---"
python scripts/run_phase3_5.py

echo ""
echo "--- Phase 4: XGBoost + LightGBM ---"
python scripts/run_phase4.py

echo ""
echo "--- Running tests ---"
python -m pytest -q

echo ""
echo "--- Full verification ---"
python scripts/verify_all.py

echo ""
echo "==========================================="
echo "DONE"
echo "==========================================="
