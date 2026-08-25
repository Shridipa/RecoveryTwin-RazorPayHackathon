# RecoveryTwin

**Counterfactual ML for Payment Recovery**

RecoveryTwin uses predictive intelligence, counterfactual decisioning and financial optimization to determine the best recovery action for every failed payment. It estimates recovery probability, evaluates what would happen under each possible intervention, and selects the financially optimal action — while respecting cost, fatigue, and compliance constraints.

## Architecture

```
                    RECOVERYTWIN
                         │
          ┌──────────────┴──────────────┐
          │                             │
    PAYMENT SIGNALS              CUSTOMER SIGNALS
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
              RECOVERY PREDICTOR
              (XGBoost / LightGBM)
                         │
                         ▼
               PROBABILITY CALIBRATION
              (Sigmoid / Isotonic)
                         │
                         ▼
              CAUSAL / UPLIFT MODEL
              (S-Learner / T-Learner)
                         │
                         ▼
            COUNTERFACTUAL SIMULATOR
          (4 recovery actions compared)
                         │
                         ▼
            TIME-TO-RECOVERY MODEL
           (Cox PH / Random Survival Forest)
                         │
                         ▼
              FINANCIAL OPTIMIZER
          (Expected value × Payment amount − Cost)
                         │
                         ▼
              POLICY GUARDRAILS
          (Fatigue / Retry limits / Eligibility)
                         │
                         ▼
               RECOMMENDED ACTION
          (Retry / Reminder / Alt Method / Stop)
                         │
                         ▼
                    AUDIT LOG
```

## Interventions

| ID | Intervention | Description |
|----|-------------|-------------|
| 0 | Control / No intervention | Do nothing — payment is left unmanaged |
| 1 | Retry | Automatically retry the same payment method |
| 2 | Reminder | Send a reminder to the customer |
| 3 | Alternative Payment Method | Suggest an alternative payment method |

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1. Generate and validate the dataset

Creates the synthetic payment environment and performs the initial
data-quality and leakage checks.

```bash
python scripts/generate_dataset.py
```

### 2. Analyze treatment overlap

Checks whether the four recovery strategies have sufficient statistical
overlap for causal analysis.

```bash
python scripts/analyze_treatment_overlap.py
```

### 3. Train baseline recovery models

Trains Logistic Regression and Random Forest models to establish
baseline recovery prediction performance.

```bash
python scripts/train_baseline_models.py
```

### 4. Calibrate and audit the models

Calibrates recovery probabilities, estimates treatment propensity,
and performs feature-availability and leakage audits.

```bash
python scripts/calibrate_and_audit_models.py
```

### 5. Train advanced ML models

Trains XGBoost and LightGBM recovery models and evaluates them on the
chronological test set.

```bash
python scripts/train_advanced_models.py
```

### 6. Train survival models

Estimates how quickly a failed payment is expected to recover under
different interventions.

```bash
python scripts/train_survival_models.py
```

### 7. Train causal / uplift models

Estimates heterogeneous treatment effects and determines which
interventions are more effective for different payment situations.

```bash
python scripts/train_causal_models.py
```

### 8. Run the counterfactual decision engine

Evaluates all available recovery actions for each payment and selects
the financially preferable bounded action.

```bash
python scripts/run_decision_engine.py
```

### 9. Run financial analysis

Stress-tests the recovery policy under different costs, recovery
rates, treatment degradation and economic conditions.

```bash
python scripts/run_financial_analysis.py
```

### 10. Run complete verification

Runs the complete RecoveryTwin validation suite — data quality,
leakage, positivity, calibration, models, and financial checks.

```bash
python scripts/verify_system.py
```

### 11. Run all tests

```bash
python -m pytest -q
```

## Dashboard

RecoveryTwin includes a FastAPI backend and React operations dashboard.

```bash
# Backend (API docs at http://localhost:8000/docs)
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend (dashboard at http://localhost:5173)
cd frontend
npm install
npm run dev
```

## Benchmark Results

Results from the synthetic evaluation environment on 8,426 unseen test payments:

| Policy | Net Revenue | vs Do Nothing |
|--------|------------|---------------|
| Do Nothing | Rs.18.2L | — |
| Always Retry | Rs.32.2L | +Rs.14.0L |
| Max Probability | Rs.42.1L | +Rs.24.0L |
| **RecoveryTwin** | **Rs.38.2L** | **+Rs.20.0L** |
| Oracle (theoretical max) | Rs.73.2L | +Rs.54.9L |

### Model performance (temporally held-out test set)

| Metric | Value |
|--------|-------|
| XGBoost PR-AUC | 0.516 |
| XGBoost ROC-AUC | 0.709 |
| LightGBM PR-AUC | 0.515 |
| LightGBM ROC-AUC | 0.711 |
| Calibration ECE | 0.012 |
| Survival C-index | 0.665 |
| Policy Robustness (vs Do Nothing) | 100% of scenarios |

### Financial stress testing

- RecoveryTwin outperforms Do Nothing across **all 12 economic scenarios**
- RecoveryTwin remains profitable even with **50% treatment degradation**
- Monte Carlo simulation (1,000 iterations) shows **100% probability of positive net revenue**

## Project Structure

```
RecoveryTwin/
│
├── backend/                    FastAPI production API
├── frontend/                   React operations dashboard
│
├── configs/
│   ├── data.yaml               Data generation parameters
│   ├── model.yaml              Model training configuration
│   ├── policy.yaml             Decision engine policy
│   └── financial.yaml          Financial scenarios & stress tests
│
├── src/recoverytwin/
│   ├── simulator/              Synthetic causal payment environment
│   ├── data/                   Validation, leakage checks
│   ├── models/                 Baseline + advanced ML models
│   ├── evaluation/             Calibration, metrics, benchmarks
│   ├── survival/               Time-to-recovery models
│   ├── causal/                 S-Learner, T-Learner, X-Learner, R-Learner
│   ├── decision/               Counterfactual decision engine
│   └── financial/              Economic model, Monte Carlo, sensitivity
│
├── scripts/                    End-to-end pipeline scripts
├── tests/                      154 tests (ML + financial + backend)
├── reports/                    Verification reports & figures
└── data/                       Synthetic datasets and temporal splits
```

## Data

RecoveryTwin uses a synthetic payment environment calibrated using
public ecosystem references (UPI statistics, public payment failure
datasets). No Razorpay production data is used.

## Verification

```bash
python -m pytest -q             # 154/154 tests
python scripts/verify_system.py # 17/17 verification sections
```

## License

Internal project — not for distribution.
