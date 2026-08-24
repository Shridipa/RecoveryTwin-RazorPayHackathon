"""
Report generation utilities.
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def save_report(report: Dict[str, Any], path: str):
    """Save report as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def load_report(path: str) -> Dict[str, Any]:
    """Load report from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_model_manifest(
    model_name: str,
    model_version: str,
    dataset_version: str,
    feature_schema_version: str,
    training_period: str,
    validation_period: str,
    test_period: str,
    hyperparameters: Dict,
    calibration_method: str,
    metrics: Dict,
    random_seed: int,
) -> Dict[str, Any]:
    """Create a model manifest."""
    return {
        "model_name": model_name,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "training_period": training_period,
        "validation_period": validation_period,
        "test_period": test_period,
        "hyperparameters": hyperparameters,
        "calibration_method": calibration_method,
        "metrics": metrics,
        "random_seed": random_seed,
        "timestamp": datetime.now().isoformat(),
    }
