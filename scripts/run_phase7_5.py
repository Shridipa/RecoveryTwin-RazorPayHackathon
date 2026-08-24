"""
Phase 7.5 — Decision Quality Audit

Diagnoses why RecoveryTwin underperforms Max Probability.
Tests counterfactual probability accuracy, CATE accuracy,
action agreement, and financial decomposition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from recoverytwin.causal.s_learner import SLearner
from recoverytwin.decision.engine import (
    DecisionEngine, load_policy_config, evaluate_oracle,
    ACTION_NAMES, ACTION_LABELS, DEFAULT_COSTS,
    time_discount_exponential,
)


def main():
    print("=" * 60)
    print("PHASE 7.5 — DECISION QUALITY AUDIT")
    print("=" * 60)
    print()

    # Load data
    train_df = pd.read_parquet("data/processed/debug/train.parquet")
    test_df = pd.read_parquet("data/processed/debug/test.parquet")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print()

    # Train S-Learner
    s_learner = SLearner(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=50, random_state=42
    )
    s_learner.fit(train_df)

    def predict_fn(df, action):
        return s_learner.predict_counterfactual(df, treatment=action)

    # ================================================================
    # AUDIT #4: COUNTERFACTUAL PROBABILITY ACCURACY
    # ================================================================
    print("=" * 60)
    print("AUDIT #4: COUNTERFACTUAL PROBABILITY ACCURACY")
    print("=" * 60)
    print("(Most critical: is P(Y|X,a) good enough?)")
    print()

    results_by_treatment = {}
    for action in range(4):
        predicted = predict_fn(test_df, action)
        true = test_df[f"potential_outcome_{action}"].values.astype(float)

        mae = np.mean(np.abs(predicted - true))
        rmse = np.sqrt(np.mean((predicted - true) ** 2))

        # Correlation
        corr, _ = stats.spearmanr(predicted, true)

        # Calibration (mean predicted vs mean actual)
        mean_pred = predicted.mean()
        mean_true = true.mean()

        # By predicted probability buckets
        n_buckets = 10
        buckets = pd.qcut(predicted, n_buckets, duplicates='drop')
        cal_data = pd.DataFrame({'pred': predicted, 'true': true, 'bucket': buckets})
        cal = cal_data.groupby('bucket', observed=True).agg(
            mean_pred=('pred', 'mean'),
            mean_true=('true', 'mean'),
            count=('pred', 'count')
        )

        results_by_treatment[action] = {
            'name': ACTION_LABELS[action],
            'mae': float(mae),
            'rmse': float(rmse),
            'spearman_corr': float(corr),
            'mean_predicted': float(mean_pred),
            'mean_true': float(mean_true),
            'calibration_gap': float(abs(mean_pred - mean_true)),
            'calibration_table': cal.to_dict('index'),
        }

        print(f"  {ACTION_LABELS[action]}:")
        print(f"    MAE:            {mae:.4f}")
        print(f"    RMSE:           {rmse:.4f}")
        print(f"    Spearman corr:  {corr:.4f}")
        print(f"    Mean predicted: {mean_pred:.4f}")
        print(f"    Mean true:      {mean_true:.4f}")
        print(f"    Calibration gap:{abs(mean_pred - mean_true):.4f}")
        print()

    # ================================================================
    # AUDIT #5: CATE ACCURACY
    # ================================================================
    print("=" * 60)
    print("AUDIT #5: CATE ACCURACY")
    print("=" * 60)
    print("(Does the model correctly estimate treatment effects?)")
    print()

    for action in [1, 2, 3]:
        p_a = predict_fn(test_df, action)
        p_0 = predict_fn(test_df, 0)
        predicted_cate = p_a - p_0

        true_cate = (
            test_df[f"potential_outcome_{action}"].values.astype(float)
            - test_df["potential_outcome_0"].values.astype(float)
        )

        mae = np.mean(np.abs(predicted_cate - true_cate))
        rmse = np.sqrt(np.mean((predicted_cate - true_cate) ** 2))
        corr, _ = stats.spearmanr(predicted_cate, true_cate)

        # Sign agreement (both positive or both negative)
        sign_agree = np.mean(np.sign(predicted_cate) == np.sign(true_cate))

        # Magnitude agreement (within 5pp)
        within_5pp = np.mean(np.abs(predicted_cate - true_cate) < 0.05)

        print(f"  {ACTION_LABELS[action]} CATE:")
        print(f"    MAE:            {mae:.4f}")
        print(f"    RMSE:           {rmse:.4f}")
        print(f"    Spearman corr:  {corr:.4f}")
        print(f"    Sign agreement: {sign_agree:.1%}")
        print(f"    Within 5pp:     {within_5pp:.1%}")
        print(f"    Mean predicted: {predicted_cate.mean():.4f}")
        print(f"    Mean true:      {true_cate.mean():.4f}")
        print()

    # ================================================================
    # AUDIT #1: ROW-BY-ROW ACTION AGREEMENT
    # ================================================================
    print("=" * 60)
    print("AUDIT #1: ACTION AGREEMENT")
    print("=" * 60)
    print()

    # Build decision tables for different policies
    config = load_policy_config()

    # Policy A: Max Probability (use predicted P(Y|X,a) for all actions, pick best)
    n = len(test_df)
    all_probs = np.column_stack([
        predict_fn(test_df, a) for a in range(4)
    ])
    max_prob_actions = np.argmax(all_probs, axis=1)

    # Policy B: CATE (highest CATE, ignoring cost and survival)
    all_cates = np.column_stack([
        np.zeros(n),
        *[predict_fn(test_df, a) - predict_fn(test_df, 0) for a in [1, 2, 3]]
    ])
    cate_actions = np.argmax(all_cates, axis=1)

    # Policy C: RecoveryTwin (full engine)
    engine = DecisionEngine(config)
    dt = engine.build_decision_table(test_df, predict_fn)
    rt_actions = dt["recommended_action"].values

    # Policy D: Oracle
    true_values = np.column_stack([
        test_df[f"potential_outcome_{a}"].values.astype(float) * test_df["amount"].values
        - DEFAULT_COSTS.get(a, 0)
        for a in range(4)
    ])
    oracle_actions = np.argmax(true_values, axis=1)

    # Agreement matrix
    policies = {
        "MaxProb": max_prob_actions,
        "CATE": cate_actions,
        "RecoveryTwin": rt_actions,
        "Oracle": oracle_actions,
    }

    print("  Pairwise agreement:")
    policy_names = list(policies.keys())
    for i in range(len(policy_names)):
        for j in range(i + 1, len(policy_names)):
            n1, n2 = policy_names[i], policy_names[j]
            agree = np.mean(policies[n1] == policies[n2])
            print(f"    {n1} == {n2}: {agree:.1%}")
    print()

    # True value per policy
    print("  Policy values (using true potential outcomes):")
    for name, actions in policies.items():
        val = sum(
            test_df[f"potential_outcome_{a}"].values[i].astype(float)
            * test_df["amount"].values[i]
            - DEFAULT_COSTS.get(int(a), 0)
            for i, a in enumerate(actions)
        )
        print(f"    {name:<15}: {val:>12,.0f}")
    print()

    # ================================================================
    # AUDIT #2: SURVIVAL CONTRIBUTION
    # ================================================================
    print("=" * 60)
    print("AUDIT #2: SURVIVAL CONTRIBUTION")
    print("=" * 60)
    print()

    # Policy WITHOUT survival adjustment
    config_no_survival = config.copy()
    config_no_survival["time_discount"] = {"enabled": False}
    engine_no_surv = DecisionEngine(config_no_survival)
    dt_no_surv = engine_no_surv.build_decision_table(test_df, predict_fn)

    no_surv_actions = dt_no_surv["recommended_action"].values
    no_surv_val = sum(
        test_df[f"potential_outcome_{a}"].values[i].astype(float)
        * test_df["amount"].values[i]
        - DEFAULT_COSTS.get(int(a), 0)
        for i, a in enumerate(no_surv_actions)
    )

    rt_val = sum(
        test_df[f"potential_outcome_{a}"].values[i].astype(float)
        * test_df["amount"].values[i]
        - DEFAULT_COSTS.get(int(a), 0)
        for i, a in enumerate(rt_actions)
    )

    print(f"  With survival:    {rt_val:>12,.0f}")
    print(f"  Without survival: {no_surv_val:>12,.0f}")
    print(f"  Survival impact:  {rt_val - no_surv_val:>+12,.0f}")
    print()

    # ================================================================
    # AUDIT #3: COST CONTRIBUTION
    # ================================================================
    print("=" * 60)
    print("AUDIT #3: COST CONTRIBUTION")
    print("=" * 60)
    print()

    # Policy WITHOUT cost deduction
    config_no_cost = config.copy()
    config_no_cost["action_costs"] = {0: 0, 1: 0, 2: 0, 3: 0}
    engine_no_cost = DecisionEngine(config_no_cost)
    dt_no_cost = engine_no_cost.build_decision_table(test_df, predict_fn)

    no_cost_actions = dt_no_cost["recommended_action"].values
    no_cost_val = sum(
        test_df[f"potential_outcome_{a}"].values[i].astype(float)
        * test_df["amount"].values[i]
        for i, a in enumerate(no_cost_actions)
    )

    print(f"  With costs:    {rt_val:>12,.0f}")
    print(f"  Without costs: {no_cost_val:>12,.0f}")
    print(f"  Cost impact:   {rt_val - no_cost_val:>+12,.0f}")
    print()

    # ================================================================
    # AUDIT: PURE CATE POLICY (no cost, no survival)
    # ================================================================
    print("=" * 60)
    print("AUDIT: PURE CATE POLICY (no cost, no survival)")
    print("=" * 60)
    print()

    config_pure = config.copy()
    config_pure["time_discount"] = {"enabled": False}
    config_pure["action_costs"] = {0: 0, 1: 0, 2: 0, 3: 0}
    engine_pure = DecisionEngine(config_pure)
    dt_pure = engine_pure.build_decision_table(test_df, predict_fn)

    pure_actions = dt_pure["recommended_action"].values
    pure_val = sum(
        test_df[f"potential_outcome_{a}"].values[i].astype(float)
        * test_df["amount"].values[i]
        for i, a in enumerate(pure_actions)
    )

    print(f"  Pure CATE (no cost, no survival): {pure_val:>12,.0f}")
    print(f"  vs Max Probability:               {sum(true_values[i, max_prob_actions[i]] for i in range(n)):>12,.0f}")
    print()

    # ================================================================
    # ROOT CAUSE ANALYSIS
    # ================================================================
    print("=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    print()

    # Compare: for each payment, what does MaxProb choose vs RT
    agree = max_prob_actions == rt_actions
    disagree = ~agree

    print(f"  Payments where MaxProb == RT:     {agree.sum():>5} ({agree.mean():.1%})")
    print(f"  Payments where MaxProb != RT:     {disagree.sum():>5} ({disagree.mean():.1%})")
    print()

    # When they disagree, who wins?
    maxprob_wins = 0
    rt_wins = 0
    for i in range(n):
        if max_prob_actions[i] != rt_actions[i]:
            maxprob_val = true_values[i, max_prob_actions[i]]
            rt_val_i = true_values[i, rt_actions[i]]
            if maxprob_val > rt_val_i:
                maxprob_wins += 1
            else:
                rt_wins += 1

    print(f"  When they disagree:")
    print(f"    MaxProb wins: {maxprob_wins}")
    print(f"    RT wins:      {rt_wins}")
    print()

    # What actions does RT choose that MaxProb doesn't?
    print("  Action transition (MaxProb -> RT):")
    for from_a in range(4):
        for to_a in range(4):
            mask = (max_prob_actions == from_a) & (rt_actions == to_a)
            if mask.sum() > 0 and from_a != to_a:
                # Value impact of this transition
                val_diff = sum(
                    true_values[i, to_a] - true_values[i, from_a]
                    for i in range(n) if mask[i]
                )
                print(f"    {ACTION_LABELS[from_a]:>20} -> {ACTION_LABELS[to_a]:<20}: "
                      f"{mask.sum():>4} payments, value impact: {val_diff:>+.0f}")
    print()

    # ================================================================
    # DIAGNOSIS
    # ================================================================
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print()

    # Check if the issue is that RT selects actions with negative value
    negative_rt = sum(1 for i in range(n) if true_values[i, rt_actions[i]] < 0)
    negative_maxprob = sum(1 for i in range(n) if true_values[i, max_prob_actions[i]] < 0)

    print(f"  Payments where RT chooses negative-value action:   {negative_rt}")
    print(f"  Payments where MaxProb chooses negative-value action: {negative_maxprob}")
    print()

    # Check: does RT over-select reminder/alternative when retry is better?
    for action in [1, 2, 3]:
        rt_chosen = (rt_actions == action)
        maxprob_chosen = (max_prob_actions == action)
        if rt_chosen.sum() > 0:
            avg_rt_val = true_values[rt_chosen, action].mean()
            # When RT chooses this action but MaxProb would choose something else
            override = rt_chosen & (max_prob_actions != action)
            if override.sum() > 0:
                avg_override_rt = true_values[override, action].mean()
                avg_override_maxprob = true_values[override, max_prob_actions[override]].mean()
                print(f"  RT overrides to {ACTION_LABELS[action]}:")
                print(f"    {override.sum()} payments")
                print(f"    RT value:    {avg_override_rt:>.0f}")
                print(f"    MaxProb val: {avg_override_maxprob:>.0f}")
                print(f"    Difference:  {avg_override_rt - avg_override_maxprob:>+.0f}")
                print()

    # Save audit results
    report_dir = Path("reports/phase7")
    report_dir.mkdir(parents=True, exist_ok=True)

    audit_results = {
        "timestamp": datetime.now().isoformat(),
        "counterfactual_accuracy": results_by_treatment,
        "action_agreement": {
            "maxprob_vs_rt": float(np.mean(max_prob_actions == rt_actions)),
            "maxprob_vs_oracle": float(np.mean(max_prob_actions == oracle_actions)),
            "rt_vs_oracle": float(np.mean(rt_actions == oracle_actions)),
        },
        "financial_decomposition": {
            "maxprob_value": float(sum(true_values[i, max_prob_actions[i]] for i in range(n))),
            "rt_value": float(rt_val),
            "rt_no_survival": float(no_surv_val),
            "rt_no_cost": float(no_cost_val),
            "rt_pure_cate": float(pure_val),
            "oracle_value": float(sum(true_values[i, oracle_actions[i]] for i in range(n))),
        },
        "root_cause": {
            "negative_rt_choices": negative_rt,
            "negative_maxprob_choices": negative_maxprob,
            "maxprob_wins_when_disagree": maxprob_wins,
            "rt_wins_when_disagree": rt_wins,
        },
    }

    with open(report_dir / "decision_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2, default=str)

    print(f"Audit saved to {report_dir / 'decision_audit.json'}")


if __name__ == "__main__":
    main()
