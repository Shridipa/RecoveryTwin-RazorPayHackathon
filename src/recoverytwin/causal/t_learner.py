"""
T-Learner: Separate model per treatment arm.

For each treatment t, trains P(Y | X, T=t) on the subset receiving t.
Estimates CATE by contrasting separate models:
  tau(x, a) = mu_a(x) - mu_0(x)
where mu_a is trained only on treated units.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder


class TLearner:
    """
    T-Learner for multi-treatment CATE estimation.
    
    Trains a separate model per treatment arm, each conditioned on its own
    treated population. More flexible than S-Learner but can have
    higher variance with small treatment groups.
    """
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42,
                 min_child_samples: int = 30):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.min_child_samples = min_child_samples
        self.models = {}  # treatment_id -> fitted model
        self.feature_names = None
        self.n_treatments = 4
    
    def _prepare_features(self, df: pd.DataFrame, include_treatment: bool = False) -> pd.DataFrame:
        """Prepare features from dataframe, optionally excluding treatment."""
        from recoverytwin.data.validator import BLOCKED_FEATURES
        
        feature_cols = [c for c in df.columns 
                       if c not in BLOCKED_FEATURES 
                       and c not in ['payment_id', 'customer_id', 'merchant_id', 'timestamp']
                       and not c.startswith('potential_outcome')
                       and c != 'true_best_intervention']
        
        X = df[feature_cols].copy()
        
        # Encode categoricals
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if not include_treatment and 'intervention' in X.columns:
            X = X.drop('intervention', axis=1)
        
        return X
    
    def fit(self, df: pd.DataFrame, y_col: str = 'recovered') -> 'TLearner':
        """
        Fit separate models for each treatment arm.
        
        Each model is trained on the subset of data where intervention == t.
        """
        X_full = self._prepare_features(df, include_treatment=False)
        self.feature_names = list(X_full.columns)
        y = df[y_col].values
        treatments = df['intervention'].values
        
        for t in range(self.n_treatments):
            mask = treatments == t
            if mask.sum() < 50:
                print(f"  Warning: only {mask.sum()} samples for treatment {t}")
                continue
            
            X_t = X_full.values[mask]
            y_t = y[mask]
            
            model = LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_t, y_t)
            self.models[t] = model
        
        return self
    
    def predict_counterfactual(self, df: pd.DataFrame, treatment: int) -> np.ndarray:
        """Predict P(Y=1 | X) using the model trained on treatment t."""
        X = self._prepare_features(df, include_treatment=False)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        if treatment not in self.models:
            raise ValueError(f"No model trained for treatment {treatment}")
        
        return self.models[treatment].predict_proba(X.values)[:, 1]
    
    def estimate_cate(self, df: pd.DataFrame, treatments: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Estimate CATE for each treatment vs control.
        
        Returns:
            Dict mapping treatment ID -> CATE array
        """
        if treatments is None:
            treatments = [t for t in range(1, self.n_treatments) if t in self.models]
        
        y0 = self.predict_counterfactual(df, treatment=0)
        
        cate = {}
        for t in treatments:
            if t not in self.models:
                continue
            yt = self.predict_counterfactual(df, treatment=t)
            cate[t] = yt - y0
        
        return cate
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance per treatment model."""
        result = {}
        for t, model in self.models.items():
            importances = model.feature_importances_
            importances = importances / importances.sum()
            result[f"treatment_{t}"] = dict(zip(self.feature_names, importances))
        return result
