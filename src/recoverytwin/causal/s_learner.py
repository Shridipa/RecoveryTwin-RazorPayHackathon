"""
S-Learner: Single model with treatment as a feature.

Estimates CATE by contrasting predictions under different treatment values:
  tau(x, a) = E[Y | X=x, T=a] - E[Y | X=x, T=0]

Uses LightGBM as the base learner for speed and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder


class SLearner:
    """
    S-Learner for multi-treatment CATE estimation.
    
    Trains a single model P(Y | X, T) and estimates treatment effects
    by contrasting predictions under different treatment assignments.
    """
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42,
                 min_child_samples: int = 50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.min_child_samples = min_child_samples
        self.model = None
        self.feature_names = None
        self.n_treatments = 4
    
    def _prepare_features(self, df: pd.DataFrame, include_treatment: bool = True) -> pd.DataFrame:
        """Prepare features from dataframe."""
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
    
    def fit(self, df: pd.DataFrame, y_col: str = 'recovered') -> 'SLearner':
        """Fit the S-Learner."""
        X = self._prepare_features(df, include_treatment=True)
        y = df[y_col].values
        
        self.feature_names = list(X.columns)
        
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X.values, y)
        return self
    
    def predict_counterfactual(self, df: pd.DataFrame, treatment: int) -> np.ndarray:
        """Predict P(Y=1 | X, T=treatment) for all observations."""
        X = self._prepare_features(df, include_treatment=True)
        X['intervention'] = treatment
        # Ensure column order matches training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        return self.model.predict_proba(X.values)[:, 1]
    
    def estimate_cate(self, df: pd.DataFrame, treatments: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Estimate CATE for each treatment vs control.
        
        Returns:
            Dict mapping treatment ID -> CATE array
        """
        if treatments is None:
            treatments = list(range(1, self.n_treatments))
        
        # Predict under control
        y0 = self.predict_counterfactual(df, treatment=0)
        
        cate = {}
        for t in treatments:
            yt = self.predict_counterfactual(df, treatment=t)
            cate[t] = yt - y0
        
        return cate
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        importances = importances / importances.sum()
        return dict(zip(self.feature_names, importances))
