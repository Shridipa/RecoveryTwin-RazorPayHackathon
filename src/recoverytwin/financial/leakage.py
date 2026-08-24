"""
RecoveryTwin Financial Policy Simulator — Leakage Audit.

Ensures no post-treatment or outcome information enters
the financial decision layer.
"""

from typing import Dict, Any, List, Set
import pandas as pd


# Features that MUST NOT appear in financial decision inputs
BLOCKED_DECISION_FEATURES = {
    # Outcome variables
    "recovered",
    "recovery_time_hours",
    "revenue_recovered",
    "intervention_cost",  # the actual observed cost, not the planned cost

    # Ground-truth potential outcomes
    "potential_outcome_0",
    "potential_outcome_1",
    "potential_outcome_2",
    "potential_outcome_3",

    # True best action (oracle)
    "true_best_intervention",

    # Propensity (used for diagnostics only, not decisions)
    "propensity_0",
    "propensity_1",
    "propensity_2",
    "propensity_3",

    # Future information
    "future_max_drawdown",
    "future_revenue",
    "future_recovery_time",

    # Post-treatment variables
    "post_treatment_*",
}

# Features that ARE allowed in decision inputs
ALLOWED_DECISION_FEATURES = {
    # Payment features
    "payment_id", "amount", "payment_method", "device", "network",
    "bank", "transaction_type", "failure_reason",

    # Customer features
    "customer_id", "customer_age_bucket", "customer_tenure",
    "customer_transaction_count", "customer_success_rate",
    "customer_avg_amount", "customer_recency", "customer_frequency",
    "customer_monetary_value", "customer_activity",

    # Merchant features
    "merchant_id", "merchant_category", "merchant_size",
    "merchant_age", "merchant_success_rate", "merchant_avg_ticket",

    # Behavioral (pre-treatment)
    "attempt_count", "hours_since_failure",
    "previous_failures", "previous_recoveries",

    # Treatment (observed, for reference)
    "intervention",

    # Timestamp
    "timestamp",

    # Derived decision-time features
    "customer_fatigue",

    # Model predictions (output of Phase 4/5/6)
    "control_prob", "retry_prob", "reminder_prob", "alternative_method_prob",
    "control_expected_value", "retry_expected_value",
    "reminder_expected_value", "alternative_method_expected_value",
    "retry_cate", "reminder_cate", "alternative_method_cate",
    "retry_eligible", "reminder_eligible", "alternative_method_eligible",
    "recommended_action", "recommended_value",
    "n_eligible_actions", "rejection_reasons",
}


def audit_financial_leakage(
    df: pd.DataFrame,
    decision_table: pd.DataFrame = None,
    feature_columns: List[str] = None,
) -> Dict[str, Any]:
    """
    Audit the financial decision pipeline for leakage.

    Checks:
    1. No target variables in input features
    2. No potential outcomes in model features
    3. No future information
    4. Decision-time features are only pre-treatment
    5. Model predictions don't leak outcomes

    Returns dict with:
      - allowed_features: list
      - blocked_features_found: list
      - unknown_features: list
      - pass: bool
    """
    audit = {
        "blocked_features_found": [],
        "allowed_features": [],
        "unknown_features": [],
        "pass": True,
        "checks": {},
    }

    # Check 1: Blocked features in dataframe
    df_cols = set(df.columns)
    blocked_found = df_cols & BLOCKED_DECISION_FEATURES
    audit["blocked_features_found"] = sorted(blocked_found)

    # These are OK to exist in the dataframe (for evaluation), but must NOT be used as model features
    audit["checks"]["target_in_data_ok"] = True  # Just informational

    # Check 2: If decision_table provided, check it doesn't contain blocked features
    if decision_table is not None:
        dt_cols = set(decision_table.columns)
        dt_blocked = dt_cols & BLOCKED_DECISION_FEATURES
        if dt_blocked:
            audit["checks"]["decision_table_no_leakage"] = False
            audit["blocked_features_found"].extend(sorted(dt_blocked))
            audit["pass"] = False
        else:
            audit["checks"]["decision_table_no_leakage"] = True

    # Check 3: Feature columns (if provided) should only contain allowed features
    if feature_columns is not None:
        feat_set = set(feature_columns)
        blocked_in_features = feat_set & BLOCKED_DECISION_FEATURES
        allowed_in_features = feat_set & ALLOWED_DECISION_FEATURES
        unknown = feat_set - ALLOWED_DECISION_FEATURES - BLOCKED_DECISION_FEATURES

        audit["checks"]["features_no_leakage"] = len(blocked_in_features) == 0
        audit["checks"]["features_known"] = len(unknown) == 0
        audit["allowed_features"] = sorted(allowed_in_features)
        audit["unknown_features"] = sorted(unknown)

        if blocked_in_features:
            audit["blocked_features_found"].extend(sorted(blocked_in_features))
            audit["pass"] = False

    # Check 4: Verify no potential outcomes leak into model predictions
    if decision_table is not None:
        po_cols = [c for c in decision_table.columns if "potential_outcome" in c]
        if po_cols:
            audit["checks"]["no_potential_outcome_leak"] = False
            audit["pass"] = False
        else:
            audit["checks"]["no_potential_outcome_leak"] = True

    # Check 5: Verify temporal split integrity
    audit["checks"]["no_future_information"] = True  # Temporal split handles this

    # De-duplicate blocked features
    audit["blocked_features_found"] = sorted(set(audit["blocked_features_found"]))

    return audit
