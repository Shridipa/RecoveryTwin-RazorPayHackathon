"""
X-Learner: Two-stage causal estimator.

Stage 1: Train separate models (T-Learner) for each treatment.
Stage 2: Impute the missing potential outcomes for each unit, then
         regress the imputed treatment effects on X to get a more
         efficient CATE estimator.

The X-Learner is particularly useful when one treatment arm is much
larger than others, as it borrows strength across arms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder


class XLearner:
    """
    X-Learner for CATE estimation.
    
    Stage 1: T-Learner (separate models per treatment)
    Stage 2: Impute missing potential outcomes, then regress effects on X
    """
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42,
                 min_child_samples: int = 30):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.min_child_samples = min_child_samples
        
        # Stage 1: T-Learner models
        self.stage1_models = {}
        # Stage 2: effect models (one per treatment vs control)
        self.stage2_models = {}
        self.feature_names = None
        self.n_treatments = 4
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from dataframe."""
        from recoverytwin.data.validator import BLOCKED_FEATURES
        
        feature_cols = [c for c in df.columns 
                       if c not in BLOCKED_FEATURES 
                       and c not in ['payment_id', 'customer_id', 'merchant_id', 'timestamp']
                       and not c.startswith('potential_outcome')
                       and c != 'true_best_intervention']
        
        X = df[feature_cols].copy()
        
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if 'intervention' in X.columns:
            X = X.drop('intervention', axis=1)
        
        return X
    
    def _fit_stage1(self, df: pd.DataFrame, y_col: str = 'recovered'):
        """Stage 1: T-Learner."""
        X_full = self._prepare_features(df)
        self.feature_names = list(X_full.columns)
        y = df[y_col].values
        treatments = df['intervention'].values
        
        for t in range(self.n_treatments):
            mask = treatments == t
            if mask.sum() < 50:
                continue
            
            model = LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_full.values[mask], y[mask])
            self.stage1_models[t] = model
    
    def _impute_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2, Step 1: Impute missing potential outcomes.
        
        For treated units (T=t), we observe Y(t) and impute Y(0).
        For control units (T=0), we observe Y(0) and impute Y(t).
        """
        X = self._prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].values
        
        y = df['recovered'].values
        treatments = df['intervention'].values
        
        effects = pd.DataFrame(index=df.index)
        
        for t in range(1, self.n_treatments):
            if t not in self.stage1_models or 0 not in self.stage1_models:
                continue
            
            # Predict under treatment t
            y_hat_t = self.stage1_models[t].predict_proba(X)[:, 1]
            # Predict under control
            y_hat_0 = self.stage1_models[0].predict_proba(X)[:, 1]
            
            # Imputed effect
            imputed_effect = np.where(
                treatments == t,
                y - y_hat_0,           # treated: observe Y(t), impute Y(0)
                y_hat_t - y,           # control: impute Y(t), observe Y(0)
            )
            effects[f'effect_{t}'] = imputed_effect
        
        return effects
    
    def fit(self, df: pd.DataFrame, y_col: str = 'recovered') -> 'XLearner':
        """Fit the X-Learner."""
        # Stage 1
        self._fit_stage1(df, y_col)
        
        # Stage 2, Step 1: Impute effects
        effects = self._impute_effects(df)
        
        # Stage 2, Step 2: Regress imputed effects on X
        X = self._prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].values
        
        for t in range(1, self.n_treatments):
            col = f'effect_{t}'
            if col not in effects.columns:
                continue
            
            effect_model = LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            effect_model.fit(X, effects[col].values)
            self.stage2_models[t] = effect_model
        
        return self
    
    def predict_counterfactual(self, df: pd.DataFrame, treatment: int) -> np.ndarray:
        """Predict P(Y=1 | X) using the stage-1 model for treatment t."""
        X = self._prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].values
        
        if treatment in self.stage1_models:
            return self.stage1_models[treatment].predict_proba(X)[:, 1]
        raise ValueError(f"No model for treatment {treatment}")
    
    def estimate_cate(self, df: pd.DataFrame, treatments: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Estimate CATE using the stage-2 effect models.
        
        Falls back to stage-1 T-Learner estimates if stage-2 model is missing.
        """
        if treatments is None:
            treatments = [t for t in range(1, self.n_treatments)]
        
        X = self._prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].values
        
        cate = {}
        for t in treatments:
            if t in self.stage2_models:
                cate[t] = self.stage2_models[t].predict(X)
            elif t in self.stage1_models and 0 in self.stage1_models:
                y0 = self.stage1_models[0].predict_proba(X)[:, 1]
                yt = self.stage1_models[t].predict_proba(X)[:, 1]
                cate[t] = yt - y0
            else:
                continue
        
        return cate
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from stage-2 models."""
        result = {}
        for t, model in self.stage2_models.items():
            importances = model.feature_importances_
            if importances.sum() > 0:
                importances = importances / importances.sum()
            result[f"effect_{t}"] = dict(zip(self.feature_names, importances))
        return result
