"""
Phase 6 - Causal / Uplift ML

Builds S-Learner, T-Learner, X-Learner, and R-Learner causal models.
Evaluates estimated CATEs against simulator ground truth (potential outcomes).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from datetime import datetime

from recoverytwin.causal.s_learner import SLearner
from recoverytwin.causal.t_learner import TLearner
from recoverytwin.causal.x_learner import XLearner
from recoverytwin.causal.r_learner import RLearner
from recoverytwin.causal.evaluation import run_full_causal_evaluation


def main():
    print("=" * 60)
    print("PHASE 6 - CAUSAL / UPLIFT ML")
    print("=" * 60)
    
    # Load data
    data_dir = Path("data/processed/debug")
    if not (data_dir / "train.parquet").exists():
        print("[FAIL] Processed data not found. Run generate_dataset.py first.")
        sys.exit(1)
    
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print()
    
    # True ATE for reference
    print("True ATE (treatment - control):")
    for t in range(1, 4):
        ate = test_df[f'potential_outcome_{t}'].mean() - test_df['potential_outcome_0'].mean()
        print(f"  T={t}: {ate:.4f}")
    print()
    
    # ================================================================
    # PART 1: S-Learner
    # ================================================================
    print("=" * 60)
    print("PART 1: S-LEARNER")
    print("=" * 60)
    
    s_learner = SLearner(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=50, random_state=42
    )
    print("  Fitting S-Learner...")
    s_learner.fit(train_df)
    
    s_cate_test = s_learner.estimate_cate(test_df)
    s_eval = run_full_causal_evaluation(s_cate_test, test_df, "s_learner")
    
    print(f"  Estimated ATE:")
    for t in range(1, 4):
        if f'treatment_{t}' in s_eval['ate_evaluation']:
            info = s_eval['ate_evaluation'][f'treatment_{t}']
            print(f"    T={t}: est={info['estimated_ate']:.4f}, true={info['true_ate']:.4f}, "
                  f"error={info['ate_error']:.4f}, corr={info['cate_spearman_corr']:.4f}")
    
    print(f"  Policy value: {s_eval['policy_evaluation']['learned_value']:.4f} "
          f"(oracle: {s_eval['policy_evaluation']['oracle_value']:.4f})")
    print(f"  Regret: {s_eval['policy_evaluation']['regret_pct']:.1f}%")
    print(f"  Best-action accuracy: {s_eval['best_action_evaluation']['accuracy']:.4f}")
    
    # Feature importance
    s_imp = s_learner.get_feature_importance()
    top_features = sorted(s_imp.items(), key=lambda x: -x[1])[:5]
    print(f"  Top features:")
    for name, imp in top_features:
        print(f"    {name}: {imp:.4f}")
    print()
    
    # ================================================================
    # PART 2: T-Learner
    # ================================================================
    print("=" * 60)
    print("PART 2: T-LEARNER")
    print("=" * 60)
    
    t_learner = TLearner(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=30, random_state=42
    )
    print("  Fitting T-Learner...")
    t_learner.fit(train_df)
    
    t_cate_test = t_learner.estimate_cate(test_df)
    t_eval = run_full_causal_evaluation(t_cate_test, test_df, "t_learner")
    
    print(f"  Estimated ATE:")
    for t in range(1, 4):
        if f'treatment_{t}' in t_eval['ate_evaluation']:
            info = t_eval['ate_evaluation'][f'treatment_{t}']
            print(f"    T={t}: est={info['estimated_ate']:.4f}, true={info['true_ate']:.4f}, "
                  f"error={info['ate_error']:.4f}, corr={info['cate_spearman_corr']:.4f}")
    
    print(f"  Policy value: {t_eval['policy_evaluation']['learned_value']:.4f} "
          f"(oracle: {t_eval['policy_evaluation']['oracle_value']:.4f})")
    print(f"  Regret: {t_eval['policy_evaluation']['regret_pct']:.1f}%")
    print(f"  Best-action accuracy: {t_eval['best_action_evaluation']['accuracy']:.4f}")
    print()
    
    # ================================================================
    # PART 3: X-Learner
    # ================================================================
    print("=" * 60)
    print("PART 3: X-LEARNER")
    print("=" * 60)
    
    x_learner = XLearner(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=30, random_state=42
    )
    print("  Fitting X-Learner...")
    x_learner.fit(train_df)
    
    x_cate_test = x_learner.estimate_cate(test_df)
    x_eval = run_full_causal_evaluation(x_cate_test, test_df, "x_learner")
    
    print(f"  Estimated ATE:")
    for t in range(1, 4):
        if f'treatment_{t}' in x_eval['ate_evaluation']:
            info = x_eval['ate_evaluation'][f'treatment_{t}']
            print(f"    T={t}: est={info['estimated_ate']:.4f}, true={info['true_ate']:.4f}, "
                  f"error={info['ate_error']:.4f}, corr={info['cate_spearman_corr']:.4f}")
    
    print(f"  Policy value: {x_eval['policy_evaluation']['learned_value']:.4f} "
          f"(oracle: {x_eval['policy_evaluation']['oracle_value']:.4f})")
    print(f"  Regret: {x_eval['policy_evaluation']['regret_pct']:.1f}%")
    print(f"  Best-action accuracy: {x_eval['best_action_evaluation']['accuracy']:.4f}")
    print()
    
    # ================================================================
    # PART 4: R-Learner
    # ================================================================
    print("=" * 60)
    print("PART 4: R-LEARNER")
    print("=" * 60)
    
    r_learner = RLearner(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=30, random_state=42
    )
    print("  Fitting R-Learner...")
    r_learner.fit(train_df)
    
    r_cate_test = r_learner.estimate_cate(test_df)
    r_eval = run_full_causal_evaluation(r_cate_test, test_df, "r_learner")
    
    print(f"  Estimated ATE:")
    for t in range(1, 4):
        if f'treatment_{t}' in r_eval['ate_evaluation']:
            info = r_eval['ate_evaluation'][f'treatment_{t}']
            print(f"    T={t}: est={info['estimated_ate']:.4f}, true={info['true_ate']:.4f}, "
                  f"error={info['ate_error']:.4f}, corr={info['cate_spearman_corr']:.4f}")
    
    print(f"  Policy value: {r_eval['policy_evaluation']['learned_value']:.4f} "
          f"(oracle: {r_eval['policy_evaluation']['oracle_value']:.4f})")
    print(f"  Regret: {r_eval['policy_evaluation']['regret_pct']:.1f}%")
    print(f"  Best-action accuracy: {r_eval['best_action_evaluation']['accuracy']:.4f}")
    print()
    
    # ================================================================
    # PART 5: LEADERBOARD
    # ================================================================
    print("=" * 60)
    print("PART 5: CAUSAL MODEL LEADERBOARD (Test Set)")
    print("=" * 60)
    
    all_evals = {
        's_learner': s_eval,
        't_learner': t_eval,
        'x_learner': x_eval,
        'r_learner': r_eval,
    }
    
    print(f"{'Model':<15} {'ATE Error':>10} {'CATE Corr':>10} {'Policy Val':>11} {'Regret%':>8} {'BA Acc':>8}")
    print("-" * 72)
    
    best_policy_val = -999
    best_model = None
    
    for name, ev in all_evals.items():
        # Average ATE error across treatments
        ate_errors = [ev['ate_evaluation'][f'treatment_{t}']['ate_error'] 
                      for t in range(1, 4) if f'treatment_{t}' in ev['ate_evaluation']]
        avg_ate_error = np.mean(ate_errors) if ate_errors else 999
        
        # Average CATE correlation
        cate_corrs = [ev['ate_evaluation'][f'treatment_{t}']['cate_spearman_corr']
                      for t in range(1, 4) if f'treatment_{t}' in ev['ate_evaluation']]
        avg_cate_corr = np.mean(cate_corrs) if cate_corrs else 0
        
        policy_val = ev['policy_evaluation']['learned_value']
        regret_pct = ev['policy_evaluation']['regret_pct']
        ba_acc = ev['best_action_evaluation']['accuracy']
        
        print(f"{name:<15} {avg_ate_error:>10.4f} {avg_cate_corr:>10.4f} {policy_val:>11.4f} {regret_pct:>7.1f}% {ba_acc:>8.4f}")
        
        if policy_val > best_policy_val:
            best_policy_val = policy_val
            best_model = name
    
    # Oracle reference
    oracle_val = s_eval['policy_evaluation']['oracle_value']
    random_val = s_eval['policy_evaluation']['random_value']
    control_val = s_eval['policy_evaluation']['control_value']
    
    print("-" * 72)
    print(f"{'Oracle':<15} {'---':>10} {'---':>10} {oracle_val:>11.4f} {'0.0%':>8} {'1.0000':>8}")
    print(f"{'Random':<15} {'---':>10} {'---':>10} {random_val:>11.4f} {'---':>8} {'---':>8}")
    print(f"{'Control':<15} {'---':>10} {'---':>10} {control_val:>11.4f} {'---':>8} {'---':>8}")
    print()
    
    print(f"Best causal model: {best_model}")
    print(f"  Policy value: {best_policy_val:.4f}")
    print(f"  vs Oracle: {best_policy_val / max(oracle_val, 1e-8) * 100:.1f}%")
    print(f"  Lift vs control: +{best_policy_val - control_val:.4f}")
    print()
    
    # ================================================================
    # PART 6: INDIVIDUAL TREATMENT EFFECTS - SAMPLE
    # ================================================================
    print("=" * 60)
    print("PART 6: SAMPLE INDIVIDUAL TREATMENT EFFECTS")
    print("=" * 60)
    
    # Show CATEs for 5 sample payments from test set
    sample_idx = test_df.index[:5]
    sample_df = test_df.loc[sample_idx]
    
    print(f"{'Payment':<12} {'Amount':>8} {'True Best':>10}", end="")
    for t in range(1, 4):
        print(f" {'True tau_' + str(t):>10}", end="")
    print()
    print("-" * 60)
    
    for i, idx in enumerate(sample_idx):
        row = test_df.loc[[idx]]
        amount = row['amount'].iloc[0]
        true_best = int(row['true_best_intervention'].iloc[0])
        
        print(f"{'#' + str(i+1):<12} {amount:>8.0f} {true_best:>10}", end="")
        for t in range(1, 4):
            true_tau = (row[f'potential_outcome_{t}'].iloc[0] - 
                       row['potential_outcome_0'].iloc[0])
            print(f" {true_tau:>10.4f}", end="")
        print()
    
    print()
    
    # Best model's individual predictions
    print(f"Best model ({best_model}) individual CATE estimates:")
    best_eval_data = all_evals[best_model]
    
    best_cate = {
        's_learner': s_cate_test,
        't_learner': t_cate_test,
        'x_learner': x_cate_test,
        'r_learner': r_cate_test,
    }[best_model]
    
    print(f"{'Payment':<12}", end="")
    for t in range(1, 4):
        print(f" {'Est tau_' + str(t):>10}", end="")
    print(f" {'Recommend':>10}")
    print("-" * 60)
    
    for i, idx in enumerate(sample_idx):
        print(f"{'#' + str(i+1):<12}", end="")
        for t in range(1, 4):
            print(f" {best_cate[t][i]:>10.4f}", end="")
        
        # Recommend treatment with highest estimated CATE
        best_t = 1
        best_val = best_cate[1][i]
        for t in range(2, 4):
            if best_cate[t][i] > best_val:
                best_val = best_cate[t][i]
                best_t = t
        print(f" {best_t:>10}")
    
    print()
    
    # ================================================================
    # SAVE REPORTS
    # ================================================================
    report_dir = Path("reports/phase6")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full evaluation
    with open(report_dir / "causal_evaluation.json", "w") as f:
        json.dump(all_evals, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'true_ate': {
            f'treatment_{t}': float(test_df[f'potential_outcome_{t}'].mean() - 
                                    test_df['potential_outcome_0'].mean())
            for t in range(1, 4)
        },
        'models': {},
    }
    
    for name, ev in all_evals.items():
        summary['models'][name] = {
            'policy_value': ev['policy_evaluation']['learned_value'],
            'regret_pct': ev['policy_evaluation']['regret_pct'],
            'best_action_accuracy': ev['best_action_evaluation']['accuracy'],
            'oracle_fraction': ev['policy_evaluation']['fraction_vs_oracle'],
        }
    
    summary['best_model'] = best_model
    summary['oracle_value'] = float(oracle_val)
    summary['random_value'] = float(random_val)
    summary['control_value'] = float(control_val)
    
    with open(report_dir / "phase6_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Reports saved to {report_dir}")
    print()
    
    # ================================================================
    # VERIFICATION
    # ================================================================
    print("=" * 60)
    print("PHASE 6 VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("S-Learner trained", len(s_cate_test) > 0),
        ("T-Learner trained", len(t_cate_test) > 0),
        ("X-Learner trained", len(x_cate_test) > 0),
        ("R-Learner trained", len(r_cate_test) > 0),
        ("All models estimate CATEs", all(len(c) == len(test_df) for c in [s_cate_test.get(1, []), t_cate_test.get(1, []), x_cate_test.get(1, []), r_cate_test.get(1, [])])),
        ("Best model selected", best_model is not None),
        ("Policy value > control", best_policy_val > control_val),
        ("Reports saved", (report_dir / "phase6_summary.json").exists()),
    ]
    
    all_pass = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("PHASE 6: PASS")
    else:
        print("PHASE 6: FAIL")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
