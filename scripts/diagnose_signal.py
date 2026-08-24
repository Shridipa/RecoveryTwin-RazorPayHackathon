"""
Diagnostic script: understand why PR-AUC is ~0.30 instead of ~0.53.

Investigates:
1. Signal-to-noise ratio in the simulator
2. Feature importance / which features carry predictive signal
3. Treatment effect heterogeneity
4. Recovery rate by subgroups
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from collections import Counter


def main():
    df = pd.read_parquet("data/synthetic/debug/transactions.parquet")
    print(f"Dataset: {len(df)} rows, {df.shape[1]} columns")
    print(f"Recovery rate: {df['recovered'].mean():.4f}")

    # 1. Base recovery rate and signal
    print("\n" + "=" * 60)
    print("1. BASE RECOVERY RATE BY TREATMENT")
    print("=" * 60)
    for t in range(4):
        mask = df["intervention"] == t
        rate = df.loc[mask, "recovered"].mean()
        print(f"  Treatment {t}: {rate:.4f} (n={mask.sum()})")

    # 2. Recovery rate by failure reason
    print("\n" + "=" * 60)
    print("2. RECOVERY RATE BY FAILURE REASON")
    print("=" * 60)
    for reason, group in df.groupby("failure_reason"):
        print(f"  {reason:25s}: {group['recovered'].mean():.4f} (n={len(group)})")

    # 3. Recovery rate by customer activity quartiles
    print("\n" + "=" * 60)
    print("3. RECOVERY RATE BY CUSTOMER ACTIVITY QUARTILE")
    print("=" * 60)
    df["activity_q"] = pd.qcut(df["customer_activity"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    for q, group in df.groupby("activity_q"):
        print(f"  {q:10s}: {group['recovered'].mean():.4f} (n={len(group)})")

    # 4. Recovery rate by merchant success rate quartiles
    print("\n" + "=" * 60)
    print("4. RECOVERY RATE BY MERCHANT SUCCESS RATE QUARTILE")
    print("=" * 60)
    df["merchant_q"] = pd.qcut(df["merchant_success_rate"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    for q, group in df.groupby("merchant_q"):
        print(f"  {q:10s}: {group['recovered'].mean():.4f} (n={len(group)})")

    # 5. Recovery rate by amount quartiles
    print("\n" + "=" * 60)
    print("5. RECOVERY RATE BY AMOUNT QUARTILE")
    print("=" * 60)
    df["amount_q"] = pd.qcut(df["amount"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    for q, group in df.groupby("amount_q"):
        print(f"  {q:10s}: {group['recovered'].mean():.4f} (n={len(group)})")

    # 6. Recovery rate by customer success rate quartiles
    print("\n" + "=" * 60)
    print("6. RECOVERY RATE BY CUSTOMER SUCCESS RATE QUARTILE")
    print("=" * 60)
    df["cust_success_q"] = pd.qcut(df["customer_success_rate"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    for q, group in df.groupby("cust_success_q"):
        print(f"  {q:10s}: {group['recovered'].mean():.4f} (n={len(group)})")

    # 7. Treatment effect heterogeneity
    print("\n" + "=" * 60)
    print("7. TREATMENT EFFECT HETEROGENEITY")
    print("=" * 60)
    control_rate = df[df["intervention"] == 0]["recovered"].mean()
    for t in [1, 2, 3]:
        t_rate = df[df["intervention"] == t]["recovered"].mean()
        uplift = t_rate - control_rate
        print(f"  Treatment {t}: rate={t_rate:.4f}, uplift vs control={uplift:+.4f}")

    # 8. Feature correlation with recovery
    print("\n" + "=" * 60)
    print("8. NUMERIC FEATURE CORRELATION WITH RECOVERY")
    print("=" * 60)
    numeric_cols = ["amount", "merchant_success_rate", "merchant_age", "merchant_avg_ticket",
                    "customer_success_rate", "customer_avg_amount", "customer_recency",
                    "customer_frequency", "customer_monetary_value", "customer_activity",
                    "customer_tenure", "customer_transaction_count",
                    "attempt_count", "hours_since_failure", "previous_failures", "previous_recoveries"]
    for col in numeric_cols:
        corr = df[col].corr(df["recovered"])
        print(f"  {col:30s}: {corr:+.4f}")

    # 9. Noise analysis: how much variance is explained?
    print("\n" + "=" * 60)
    print("9. SIGNAL vs NOISE ANALYSIS")
    print("=" * 60)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from recoverytwin.data.feature_availability import get_model_features

    features = get_model_features(df, include_treatment=False)
    X = df[features].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["recovered"].values

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    train_score = rf.score(X, y)
    proba = rf.predict_proba(X)[:, 1]

    from sklearn.metrics import average_precision_score, roc_auc_score
    pr_auc = average_precision_score(y, proba)
    roc_auc = roc_auc_score(y, proba)

    print(f"  Full-data RF accuracy: {train_score:.4f}")
    print(f"  Full-data RF PR-AUC:  {pr_auc:.4f}")
    print(f"  Full-data RF ROC-AUC: {roc_auc:.4f}")
    print(f"  (This is the UPPER BOUND on what any model can achieve)")

    # Feature importance
    importances = sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 features:")
    for fname, imp in importances[:10]:
        print(f"    {fname:30s}: {imp:.4f}")

    # 10. Potential outcome analysis
    print("\n" + "=" * 60)
    print("10. POTENTIAL OUTCOME ANALYSIS (Ground Truth)")
    print("=" * 60)
    for i in range(4):
        po = df[f"potential_outcome_{i}"]
        print(f"  Y({i}) recovery rate: {po.mean():.4f}")

    # How much variance across potential outcomes per individual?
    po_cols = [f"potential_outcome_{i}" for i in range(4)]
    po_matrix = df[po_cols].values
    individual_variance = po_matrix.var(axis=1)
    print(f"\n  Mean individual variance across interventions: {individual_variance.mean():.4f}")
    print(f"  (Higher = more heterogeneous treatment effects = more to learn)")

    # How often is the best intervention different from control?
    true_best = df["true_best_intervention"]
    print(f"  Best intervention = Control: {(true_best == 0).mean():.4f}")
    print(f"  Best intervention = Retry:    {(true_best == 1).mean():.4f}")
    print(f"  Best intervention = Reminder: {(true_best == 2).mean():.4f}")
    print(f"  Best intervention = Alt:      {(true_best == 3).mean():.4f}")

    # 11. Entropy of the outcome
    print("\n" + "=" * 60)
    print("11. OUTCOME ENTROPY")
    print("=" * 60)
    p = df["recovered"].mean()
    entropy = -(p * np.log2(p + 1e-10) + (1-p) * np.log2(1-p + 1e-10))
    print(f"  Outcome entropy: {entropy:.4f} bits")
    print(f"  (Max possible with balanced classes = 1.0)")
    print(f"  (Lower entropy = more predictable)")


if __name__ == "__main__":
    main()
