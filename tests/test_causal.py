"""
Tests for causal models and evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Load the synthetic dataset."""
    data_path = Path("data/synthetic/debug/transactions.parquet")
    if not data_path.exists():
        pytest.skip("Synthetic data not found")
    return pd.read_parquet(data_path)


@pytest.fixture
def train_test_split(sample_data):
    """Split into train/test."""
    df = sample_data
    train = df[df['timestamp'] < '2024-09-01'].reset_index(drop=True)
    test = df[df['timestamp'] >= '2024-11-01'].reset_index(drop=True)
    return train, test


class TestSLearner:
    def test_fit_and_predict(self, train_test_split):
        from recoverytwin.causal.s_learner import SLearner
        
        train, test = train_test_split
        model = SLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        cate = model.estimate_cate(test, treatments=[1, 2, 3])
        assert 1 in cate
        assert 2 in cate
        assert 3 in cate
        assert len(cate[1]) == len(test)
    
    def test_feature_importance(self, train_test_split):
        from recoverytwin.causal.s_learner import SLearner
        
        train, _ = train_test_split
        model = SLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        imp = model.get_feature_importance()
        assert len(imp) > 0
        assert sum(imp.values()) > 0.9  # Should sum to ~1


class TestTLearner:
    def test_fit_and_predict(self, train_test_split):
        from recoverytwin.causal.t_learner import TLearner
        
        train, test = train_test_split
        model = TLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        cate = model.estimate_cate(test, treatments=[1, 2, 3])
        assert 1 in cate
        assert len(cate[1]) == len(test)
    
    def test_separate_models(self, train_test_split):
        from recoverytwin.causal.t_learner import TLearner
        
        train, _ = train_test_split
        model = TLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        # Should have 4 separate models
        assert len(model.models) == 4


class TestXLearner:
    def test_fit_and_predict(self, train_test_split):
        from recoverytwin.causal.x_learner import XLearner
        
        train, test = train_test_split
        model = XLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        cate = model.estimate_cate(test, treatments=[1, 2, 3])
        assert 1 in cate
        assert len(cate[1]) == len(test)
    
    def test_two_stages(self, train_test_split):
        from recoverytwin.causal.x_learner import XLearner
        
        train, _ = train_test_split
        model = XLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        # Should have stage1 and stage2 models
        assert len(model.stage1_models) >= 2
        assert len(model.stage2_models) >= 2


class TestRLearner:
    def test_fit_and_predict(self, train_test_split):
        from recoverytwin.causal.r_learner import RLearner
        
        train, test = train_test_split
        model = RLearner(n_estimators=50, max_depth=3, min_child_samples=20)
        model.fit(train)
        
        cate = model.estimate_cate(test, treatments=[1, 2, 3])
        assert 1 in cate
        assert len(cate[1]) == len(test)


class TestCausalEvaluation:
    def test_evaluate_ate(self, sample_data):
        from recoverytwin.causal.evaluation import evaluate_ate
        
        # Create dummy CATE estimates
        n = len(sample_data)
        cate_estimates = {
            1: np.full(n, 0.1),
            2: np.full(n, 0.15),
            3: np.full(n, 0.1),
        }
        
        result = evaluate_ate(cate_estimates, sample_data)
        assert 'treatment_1' in result
        assert 'true_ate' in result['treatment_1']
        assert 'estimated_ate' in result['treatment_1']
        assert 'cate_spearman_corr' in result['treatment_1']
    
    def test_evaluate_policy(self, sample_data):
        from recoverytwin.causal.evaluation import evaluate_policy
        
        n = len(sample_data)
        cate_estimates = {
            1: np.random.RandomState(42).randn(n) * 0.1,
            2: np.random.RandomState(42).randn(n) * 0.1 + 0.05,
            3: np.random.RandomState(42).randn(n) * 0.1,
        }
        
        result = evaluate_policy(cate_estimates, sample_data)
        assert 'oracle_value' in result
        assert 'learned_value' in result
        assert 'regret' in result
    
    def test_evaluate_best_action(self, sample_data):
        from recoverytwin.causal.evaluation import evaluate_best_action
        
        n = len(sample_data)
        cate_estimates = {
            1: np.random.RandomState(42).randn(n),
            2: np.random.RandomState(42).randn(n),
            3: np.random.RandomState(42).randn(n),
        }
        
        result = evaluate_best_action(cate_estimates, sample_data)
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
