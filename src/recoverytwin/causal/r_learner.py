"""
R-Learner: Robinson-style orthogonal causal estimator.

Based on Robinson (1988) and Chernozhukov et al. (2018):
1. Partial out the effect of X on Y and on T
2. Regress the residualized Y on the residualized T
3. The coefficient gives a weighted estimate of the CATE

This is more robust to confounding than naive approaches because
it orthogonalizes the treatment effect against covariates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge


class RLearner:
    """
    R-Learner for CATE estimation.
    
    Uses Robinson orthogonalization:
    1. Estimate m(X) = E[Y|X] using a nuisance model
    2. Estimate e(X) = E[T|X] using a propensity model  
    3. Regress (Y - m(X)) on (T - e(X)) to get tau
    """
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42,
                 min_child_samples: int = 30):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.min_child_samples = min_child_samples
        
        # Nuisance models
        self.outcome_model = None      # m(X) = E[Y|X]
        self.propensity_models = {}    # e_t(X) = P(T=t|X)
        # Effect models (one per treatment)
        self.effect_models = {}
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
    
    def fit(self, df: pd.DataFrame, y_col: str = 'recovered') -> 'RLearner':
        """
        Fit the R-Learner.
        
        For each treatment t vs control:
        1. Estimate E[Y|X] (outcome nuisance)
        2. Estimate P(T=t|X) (propensity nuisance)
        3. Residualize: Y_hat = Y - m(X), T_hat = T - e(X)
        4. Regress Y_hat on T_hat to get tau
        """
        X = self._prepare_features(df)
        self.feature_names = list(X.columns)
        X_arr = X.values
        y = df[y_col].values
        treatments = df['intervention'].values
        
        # Step 1: Outcome model m(X) = E[Y|X]
        self.outcome_model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )
        self.outcome_model.fit(X_arr, y)
        m_X = self.outcome_model.predict_proba(X_arr)[:, 1]
        
        # Step 2: Propensity models e_t(X) = P(T=t|X)
        for t in range(self.n_treatments):
            prop_model = LGBMClassifier(
                n_estimators=max(100, self.n_estimators // 2),
                max_depth=max(3, self.max_depth - 1),
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            prop_model.fit(X_arr, (treatments == t).astype(int))
            self.propensity_models[t] = prop_model
        
        # Step 3 & 4: For each treatment t, orthogonalize and regress
        for t in range(1, self.n_treatments):
            # Residualized outcome: Y - m(X)
            Y_res = y - m_X
            
            # Residualized treatment indicator: 1(T=t) - P(T=t|X)
            e_t = self.propensity_models[t].predict_proba(X_arr)[:, 1]
            T_res = (treatments == t).astype(float) - e_t
            
            # Variance weights
            weights = T_res ** 2
            weights = np.maximum(weights, 1e-6)
            
            # Regress Y_res on T_res (weighted)
            # tau = E[w * T_res * Y_res] / E[w * T_res^2]
            # More robust: use a flexible model
            effect_model = LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            
            # Use the residualized treatment as a feature
            meta_X = np.column_stack([X_arr, T_res.reshape(-1, 1)])
            effect_model.fit(meta_X, Y_res, sample_weight=weights)
            self.effect_models[t] = effect_model
        
        return self
    
    def predict_counterfactual(self, df: pd.DataFrame, treatment: int) -> np.ndarray:
        """Predict P(Y=1 | X, T=t) using the nuisance models."""
        X = self._prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X_arr = X[self.feature_names].values
        
        # For the treated model, we can't easily predict P(Y|X,T=t)
        # with just the nuisance models. Use outcome model as approximation.
        if self.outcome_model is not None:
            return self.outcome_model.predict_proba(X_arr)[:, 1]
        raise ValueError("Outcome model not available")
    
    def estimate_cate(self, df: pd.DataFrame, treatments: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Estimate CATE using orthogonalized effect models.
        """
        if treatments is None:
            treatments = [t for t in range(1, self.n_treatments) if t in self.effect_models]
        
        X = self._prepare_features(df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X_arr = X[self.feature_names].values
        
        cate = {}
        for t in treatments:
            if t not in self.effect_models:
                continue
            
            # Residualized treatment indicator
            e_t = self.propensity_models[t].predict_proba(X_arr)[:, 1]
            T_res = 1.0 - e_t  # For counterfactual T=t
            
            meta_X = np.column_stack([X_arr, T_res.reshape(-1, 1)])
            tau = self.effect_models[t].predict(meta_X)
            cate[t] = tau
        
        return cate
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from effect models."""
        result = {}
        for t, model in self.effect_models.items():
            importances = model.feature_importances_
            if importances.sum() > 0:
                importances = importances / importances.sum()
            # Last feature is the residualized treatment indicator
            names = self.feature_names + ['residualized_T']
            result[f"effect_{t}"] = dict(zip(names, importances))
        return result
