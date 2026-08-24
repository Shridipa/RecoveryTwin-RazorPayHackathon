"""
XGBoost model for recovery prediction.

Supports P(Y|X) and P(Y|X,T) variants.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional


class XGBoostModel:
    """XGBoost wrapper for binary classification."""

    def __init__(self, include_treatment: bool = False, name: str = None,
                 params: Dict = None):
        self.include_treatment = include_treatment
        self.name = name or f"xgboost_{'p_y_xt' if include_treatment else 'p_y_x'}"
        self.model = None
        self.default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "aucpr",
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0,
        }
        if params:
            self.default_params.update(params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feature_names: list = None):
        """Fit the model with optional validation."""
        self.model = xgb.XGBClassifier(**self.default_params)

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {k: v for k, v in self.default_params.items()
                if k not in ["eval_metric", "verbosity", "use_label_encoder"]}
