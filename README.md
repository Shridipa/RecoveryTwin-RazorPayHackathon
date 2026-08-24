# RecoveryTwin

**Counterfactual ML for Payment Recovery**

RecoveryTwin determines which bounded recovery intervention is most likely to recover a failed payment, estimates how much incremental revenue that intervention can create, estimates how long recovery will take, and eventually chooses the financially optimal action.

## Architecture

```
Payment Failure
      |
      v
Decision-Time Feature Snapshot
      |
      v
Recovery Probability Model
      |
      v
Calibrated Probability
      |
      v
Treatment / Propensity Model
      |
      v
Causal / Uplift Model
      |
      v
Counterfactual Outcomes
      |
      v
Survival / Time-to-Recovery Model
      |
      v
Financial Optimization
      |
      v
Policy / Compliance Guardrails
      |
      v
Bounded Recovery Action
      |
      v
Audit Trail
```

## Interventions

| ID | Intervention |
|----|-------------|
| 0 | Control / No intervention |
| 1 | Retry |
| 2 | Reminder |
| 3 | Alternative Payment Method |

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
# Generate synthetic dataset
python scripts/generate_and_validate.py

# Run Phase 2.5 (positivity)
python scripts/run_phase2_5.py

# Run Phase 3 (baseline ML)
python scripts/run_phase3.py

# Run Phase 3.5 (calibration, propensity, leakage)
python scripts/run_phase3_5.py

# Run Phase 4 (XGBoost + LightGBM)
python scripts/run_phase4.py

# Full verification
python scripts/verify_all.py

# Tests
python -m pytest -q
```

## Data

RecoveryTwin uses a synthetic payment environment calibrated using public ecosystem references (UPI statistics, public payment failure datasets). No Razorpay production data is used.

## License

Internal project - not for distribution.
