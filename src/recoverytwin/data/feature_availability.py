"""
Feature availability metadata.

Every feature has metadata about when it's available, whether it depends
on treatment or outcome, and whether it's allowed for prediction.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class FeatureMeta:
    feature_name: str
    source: str  # "customer", "merchant", "payment", "behavioral", "treatment", "outcome", "hidden"
    available_at: str  # "pre_treatment", "post_treatment", "post_outcome"
    treatment_dependency: bool
    target_dependency: bool
    allowed_for_prediction: bool
    description: str = ""


# Complete feature registry
FEATURE_REGISTRY = [
    # Merchant features
    FeatureMeta("merchant_id", "merchant", "pre_treatment", False, False, True, "Merchant identifier"),
    FeatureMeta("merchant_category", "merchant", "pre_treatment", False, False, True, "Merchant business category"),
    FeatureMeta("merchant_size", "merchant", "pre_treatment", False, False, True, "Merchant size tier"),
    FeatureMeta("merchant_age", "merchant", "pre_treatment", False, False, True, "Months since merchant onboarded"),
    FeatureMeta("merchant_success_rate", "merchant", "pre_treatment", False, False, True, "Historical payment success rate"),
    FeatureMeta("merchant_avg_ticket", "merchant", "pre_treatment", False, False, True, "Average transaction amount"),

    # Customer features
    FeatureMeta("customer_id", "customer", "pre_treatment", False, False, True, "Customer identifier"),
    FeatureMeta("customer_age_bucket", "customer", "pre_treatment", False, False, True, "Age range bucket"),
    FeatureMeta("customer_tenure", "customer", "pre_treatment", False, False, True, "Months as customer"),
    FeatureMeta("customer_transaction_count", "customer", "pre_treatment", False, False, True, "Total historical transactions"),
    FeatureMeta("customer_success_rate", "customer", "pre_treatment", False, False, True, "Historical success rate"),
    FeatureMeta("customer_avg_amount", "customer", "pre_treatment", False, False, True, "Average transaction amount"),
    FeatureMeta("customer_recency", "customer", "pre_treatment", False, False, True, "Hours since last transaction"),
    FeatureMeta("customer_frequency", "customer", "pre_treatment", False, False, True, "Transaction frequency"),
    FeatureMeta("customer_monetary_value", "customer", "pre_treatment", False, False, True, "Total monetary value"),
    FeatureMeta("customer_activity", "customer", "pre_treatment", False, False, True, "Activity score"),

    # Payment features
    FeatureMeta("payment_id", "payment", "pre_treatment", False, False, True, "Payment identifier"),
    FeatureMeta("timestamp", "payment", "pre_treatment", False, False, True, "Transaction timestamp"),
    FeatureMeta("amount", "payment", "pre_treatment", False, False, True, "Transaction amount"),
    FeatureMeta("payment_method", "payment", "pre_treatment", False, False, True, "Payment method used"),
    FeatureMeta("device", "payment", "pre_treatment", False, False, True, "Device type"),
    FeatureMeta("network", "payment", "pre_treatment", False, False, True, "Network type"),
    FeatureMeta("bank", "payment", "pre_treatment", False, False, True, "Customer bank"),
    FeatureMeta("transaction_type", "payment", "pre_treatment", False, False, True, "Transaction type"),
    FeatureMeta("failure_reason", "payment", "pre_treatment", False, False, True, "Payment failure reason"),

    # Behavioral features
    FeatureMeta("attempt_count", "behavioral", "pre_treatment", False, False, True, "Number of attempts for this payment"),
    FeatureMeta("hours_since_failure", "behavioral", "pre_treatment", False, False, True, "Hours since initial failure"),
    FeatureMeta("previous_failures", "behavioral", "pre_treatment", False, False, True, "Historical failure count"),
    FeatureMeta("previous_recoveries", "behavioral", "pre_treatment", False, False, True, "Historical recovery count"),

    # Treatment (allowed only for P(Y|X,T))
    FeatureMeta("intervention", "treatment", "post_treatment", True, False, True, "Intervention applied"),

    # Outcomes (blocked from model input)
    FeatureMeta("recovered", "outcome", "post_outcome", False, True, False, "Whether payment recovered"),
    FeatureMeta("recovery_time_hours", "outcome", "post_outcome", False, True, False, "Hours until recovery"),
    FeatureMeta("revenue_recovered", "outcome", "post_outcome", False, True, False, "Revenue recovered"),
    FeatureMeta("intervention_cost", "outcome", "post_outcome", True, True, False, "Cost of intervention"),

    # Hidden fields (blocked from model input)
    FeatureMeta("potential_outcome_0", "hidden", "post_outcome", False, True, False, "Potential outcome under control"),
    FeatureMeta("potential_outcome_1", "hidden", "post_outcome", False, True, False, "Potential outcome under retry"),
    FeatureMeta("potential_outcome_2", "hidden", "post_outcome", False, True, False, "Potential outcome under reminder"),
    FeatureMeta("potential_outcome_3", "hidden", "post_outcome", False, True, False, "Potential outcome under alt method"),
    FeatureMeta("true_best_intervention", "hidden", "post_outcome", False, True, False, "True best intervention"),
    FeatureMeta("propensity_0", "hidden", "post_treatment", True, True, False, "Propensity for control"),
    FeatureMeta("propensity_1", "hidden", "post_treatment", True, True, False, "Propensity for retry"),
    FeatureMeta("propensity_2", "hidden", "post_treatment", True, True, False, "Propensity for reminder"),
    FeatureMeta("propensity_3", "hidden", "post_treatment", True, True, False, "Propensity for alternative"),
    FeatureMeta("customer_fatigue", "hidden", "post_treatment", True, True, False, "Customer fatigue score"),
]


def get_model_features(df: pd.DataFrame, include_treatment: bool = False) -> list:
    """Get features allowed for model input."""
    blocked = {f.feature_name for f in FEATURE_REGISTRY if not f.allowed_for_prediction}
    # Also block IDs
    blocked.update({"payment_id", "merchant_id", "customer_id", "timestamp"})

    features = [c for c in df.columns if c not in blocked]

    if not include_treatment and "intervention" in features:
        features.remove("intervention")

    return features


def audit_features(df: pd.DataFrame, include_treatment: bool = False) -> dict:
    """Audit features for a model run."""
    features = get_model_features(df, include_treatment=include_treatment)

    blocked = [f.feature_name for f in FEATURE_REGISTRY if not f.allowed_for_prediction]
    unknown = [c for c in df.columns if c not in {f.feature_name for f in FEATURE_REGISTRY}]

    return {
        "n_features": len(features),
        "features": features,
        "n_blocked": len(blocked),
        "blocked": blocked,
        "n_unknown": len(unknown),
        "unknown": unknown,
    }
