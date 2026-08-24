"""Backend configuration."""
import os
from pathlib import Path


class Settings:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data" / "processed" / "debug"
    MODELS_DIR = PROJECT_ROOT / "models" / "phase4"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    CONFIGS_DIR = PROJECT_ROOT / "configs"

    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"

    # Data files
    TEST_DATA = DATA_DIR / "test.parquet"
    TRAIN_DATA = DATA_DIR / "train.parquet"

    # Reports
    PHASE7_REPORT = REPORTS_DIR / "phase7" / "phase7_summary.json"
    PHASE7_PREDICTIONS = REPORTS_DIR / "phase7" / "counterfactual_predictions.parquet"
    PHASE8_REPORT = REPORTS_DIR / "phase8" / "phase8_report.json"
    PHASE7_SEGMENTS = REPORTS_DIR / "phase7" / "segment_analysis.csv"
    PHASE8_SEGMENTS = REPORTS_DIR / "phase8" / "segment_analysis.csv"
    PHASE3_REPORT = REPORTS_DIR / "phase3" / "baseline_report.json"
    PHASE35_REPORT = REPORTS_DIR / "phase3_5" / "phase3_5_summary.json"
    PHASE5_SUMMARY = REPORTS_DIR / "phase5" / "phase5_summary.json"
    PHASE6_SUMMARY = REPORTS_DIR / "phase6" / "phase6_summary.json"

    # Blocked fields (never expose to frontend)
    BLOCKED_FIELDS = {
        "recovered", "recovery_time_hours", "revenue_recovered",
        "intervention_cost", "potential_outcome_0", "potential_outcome_1",
        "potential_outcome_2", "potential_outcome_3", "true_best_intervention",
        "propensity_0", "propensity_1", "propensity_2", "propensity_3",
    }


settings = Settings()
