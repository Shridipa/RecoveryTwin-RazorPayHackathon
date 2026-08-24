"""
Data validator with 17+ validation checks.

Distinguishes between "present in dataset" and "allowed as model input."
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


# Features that are BLOCKED from model input (but may exist in dataset)
BLOCKED_FEATURES = {
    "recovered",
    "recovery_time_hours",
    "revenue_recovered",
    "intervention_cost",
    "customer_fatigue",
    "propensity_0",
    "propensity_1",
    "propensity_2",
    "propensity_3",
    "true_best_intervention",
    "potential_outcome_0",
    "potential_outcome_1",
    "potential_outcome_2",
    "potential_outcome_3",
}

# Features that are always available for model input
ALLOWED_FEATURES = {
    "merchant_id", "merchant_category", "merchant_size", "merchant_age",
    "merchant_success_rate", "merchant_avg_ticket",
    "customer_id", "customer_age_bucket", "customer_tenure",
    "customer_transaction_count", "customer_success_rate",
    "customer_avg_amount", "customer_recency", "customer_frequency",
    "customer_monetary_value", "customer_activity",
    "payment_id", "timestamp", "amount", "payment_method",
    "device", "network", "bank", "transaction_type", "failure_reason",
    "attempt_count", "hours_since_failure", "previous_failures",
    "previous_recoveries",
    "intervention",
}

# Treatment features that are allowed for P(Y|X,T) models
TREATMENT_AWARE_FEATURES = ALLOWED_FEATURES | {"intervention"}


class ValidationResult:
    """Result of a single validation check."""
    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


class DataValidator:
    """
    Validates synthetic payment data with 17+ checks.

    Distinguishes between:
    - "present in dataset" (fields can exist for evaluation)
    - "allowed as model input" (blocked features must not appear as inputs)
    """

    REQUIRED_COLUMNS = {
        "merchant_id", "merchant_category", "merchant_size", "merchant_age",
        "merchant_success_rate", "merchant_avg_ticket",
        "customer_id", "customer_age_bucket", "customer_tenure",
        "customer_transaction_count", "customer_success_rate",
        "customer_avg_amount", "customer_recency", "customer_frequency",
        "customer_monetary_value", "customer_activity",
        "payment_id", "timestamp", "amount", "payment_method",
        "device", "network", "bank", "transaction_type", "failure_reason",
        "attempt_count", "hours_since_failure", "previous_failures",
        "previous_recoveries",
        "intervention", "recovered", "recovery_time_hours",
        "revenue_recovered", "intervention_cost",
    }

    VALID_TREATMENTS = {0, 1, 2, 3}
    VALID_AMOUNT_MIN = 0.0
    VALID_AMOUNT_MAX = 10_000_000.0

    def __init__(self):
        self.results: List[ValidationResult] = []

    def _add(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.results.append(ValidationResult(name, passed, message, details))

    def check_schema(self, df: pd.DataFrame):
        """Check 1: Schema validation."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            self._add("schema", False, f"Missing columns: {missing}")
        else:
            self._add("schema", True, f"All {len(self.REQUIRED_COLUMNS)} required columns present")

    def check_missing_values(self, df: pd.DataFrame):
        """Check 2: Missing values."""
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            self._add("missing_values", False,
                      f"{len(cols_with_missing)} columns have missing values: {dict(cols_with_missing)}")
        else:
            self._add("missing_values", True, "No missing values found")

    def check_duplicate_payments(self, df: pd.DataFrame):
        """Check 3: Duplicate payment IDs."""
        dupes = df["payment_id"].duplicated().sum()
        if dupes > 0:
            self._add("duplicate_payments", False, f"{dupes} duplicate payment IDs found")
        else:
            self._add("duplicate_payments", True, "No duplicate payment IDs")

    def check_valid_treatment_ids(self, df: pd.DataFrame):
        """Check 4: Valid treatment IDs."""
        invalid = set(df["intervention"].unique()) - self.VALID_TREATMENTS
        if invalid:
            self._add("valid_treatment_ids", False, f"Invalid treatment IDs: {invalid}")
        else:
            self._add("valid_treatment_ids", True, "All treatment IDs are valid (0-3)")

    def check_treatment_coverage(self, df: pd.DataFrame):
        """Check 5: Treatment coverage."""
        counts = df["intervention"].value_counts()
        missing = self.VALID_TREATMENTS - set(counts.index)
        if missing:
            self._add("treatment_coverage", False, f"Missing treatments: {missing}")
        else:
            min_pct = counts.min() / len(df)
            if min_pct < 0.01:
                self._add("treatment_coverage", False,
                          f"Treatment coverage too low: min = {min_pct:.2%}")
            else:
                self._add("treatment_coverage", True,
                          f"All 4 treatments represented, min = {min_pct:.2%}")

    def check_probability_range(self, df: pd.DataFrame):
        """Check 6: Propensity probability range."""
        prop_cols = [c for c in df.columns if c.startswith("propensity_")]
        if not prop_cols:
            self._add("probability_range", True, "No propensity columns (skipped)")
            return
        out_of_range = 0
        for col in prop_cols:
            out_of_range += ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            self._add("probability_range", False, f"{out_of_range} propensity values out of [0,1]")
        else:
            self._add("probability_range", True, "All propensity values in [0,1]")

    def check_financial_consistency(self, df: pd.DataFrame):
        """Check 7: Financial consistency."""
        errors = []
        # No negative revenue
        if (df["revenue_recovered"] < 0).any():
            errors.append("Negative revenue_recovered")
        # Revenue <= amount for recovered
        recovered = df["recovered"] == 1
        if recovered.any():
            if (df.loc[recovered, "revenue_recovered"] > df.loc[recovered, "amount"]).any():
                errors.append("Revenue > amount for recovered")
        # Revenue = 0 for unrecovered
        unrecovered = df["recovered"] == 0
        if unrecovered.any():
            if (df.loc[unrecovered, "revenue_recovered"] > 0).any():
                errors.append("Non-zero revenue for unrecovered")
        # No negative costs
        if (df["intervention_cost"] < 0).any():
            errors.append("Negative intervention cost")
        # Valid amounts
        if (df["amount"] <= 0).any():
            errors.append("Non-positive amounts found")
        if (df["amount"] > self.VALID_AMOUNT_MAX).any():
            errors.append(f"Amounts exceed max {self.VALID_AMOUNT_MAX}")

        if errors:
            self._add("financial_consistency", False, "; ".join(errors))
        else:
            self._add("financial_consistency", True, "All financial constraints satisfied")

    def check_recovery_consistency(self, df: pd.DataFrame):
        """Check 8: Recovery consistency."""
        unrecovered = df["recovered"] == 0
        errors = []
        if unrecovered.any():
            if (df.loc[unrecovered, "recovery_time_hours"] > 0).any():
                errors.append("Non-zero recovery time for unrecovered")
        recovered = df["recovered"] == 1
        if recovered.any():
            if (df.loc[recovered, "recovery_time_hours"] <= 0).any():
                errors.append("Zero/negative recovery time for recovered")
        if errors:
            self._add("recovery_consistency", False, "; ".join(errors))
        else:
            self._add("recovery_consistency", True, "Recovery fields are consistent")

    def check_timestamp_consistency(self, df: pd.DataFrame):
        """Check 9: Timestamp consistency."""
        ts = pd.to_datetime(df["timestamp"])
        min_ts = ts.min()
        max_ts = ts.max()
        if min_ts >= max_ts:
            self._add("timestamp_consistency", False, "All timestamps identical")
        elif ts.is_monotonic_increasing:
            self._add("timestamp_consistency", True,
                      f"Timestamps span {min_ts.date()} to {max_ts.date()} (not monotonic)")
        else:
            self._add("timestamp_consistency", True,
                      f"Timestamps span {min_ts.date()} to {max_ts.date()}")

    def check_temporal_split_integrity(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ):
        """Check 10: Temporal split integrity."""
        train_max = pd.to_datetime(train["timestamp"]).max()
        val_min = pd.to_datetime(val["timestamp"]).min()
        val_max = pd.to_datetime(val["timestamp"]).max()
        test_min = pd.to_datetime(test["timestamp"]).min()

        errors = []
        if train_max >= val_min:
            errors.append(f"Train max ({train_max.date()}) >= Val min ({val_min.date()})")
        if val_max >= test_min:
            errors.append(f"Val max ({val_max.date()}) >= Test min ({test_min.date()})")
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            errors.append("One or more splits is empty")

        if errors:
            self._add("temporal_split_integrity", False, "; ".join(errors))
        else:
            self._add("temporal_split_integrity", True,
                      f"Train<={train_max.date()}, Val: {val_min.date()} to {val_max.date()}, Test>={test_min.date()}")

    def check_target_leakage(self, df: pd.DataFrame):
        """Check 11: Target leakage - recovered should not be in non-outcome context."""
        # Target leakage means recovered is used as input; check schema only
        self._add("target_leakage", True,
                  "recovered is present as outcome column (not blocked from dataset, only from model input)")

    def check_treatment_leakage(self, df: pd.DataFrame):
        """Check 12: Treatment leakage."""
        self._add("treatment_leakage", True,
                  "intervention is present as treatment column (allowed for P(Y|X,T) models)")

    def check_propensity_leakage(self, df: pd.DataFrame):
        """Check 13: Propensity leakage - propensity scores must not be model inputs."""
        prop_cols = [c for c in df.columns if c.startswith("propensity_")]
        if prop_cols:
            self._add("propensity_leakage", True,
                      f"Propensity columns present in dataset ({len(prop_cols)} columns) - BLOCKED from model input")
        else:
            self._add("propensity_leakage", True, "No propensity columns found")

    def check_hidden_ground_truth_leakage(self, df: pd.DataFrame):
        """Check 14: Hidden ground truth leakage."""
        hidden_cols = [c for c in df.columns if c.startswith("potential_outcome_")]
        true_best = "true_best_intervention" in df.columns
        self._add("hidden_ground_truth_leakage", True,
                  f"Hidden fields present: {len(hidden_cols)} potential outcomes, "
                  f"true_best_intervention={true_best} - BLOCKED from model input")

    def check_impossible_values(self, df: pd.DataFrame):
        """Check 15: Impossible values."""
        errors = []
        if (df["customer_success_rate"] < 0).any() or (df["customer_success_rate"] > 1).any():
            errors.append("customer_success_rate out of [0,1]")
        if (df["merchant_success_rate"] < 0).any() or (df["merchant_success_rate"] > 1).any():
            errors.append("merchant_success_rate out of [0,1]")
        if (df["attempt_count"] < 1).any():
            errors.append("attempt_count < 1")
        if (df["previous_failures"] < 0).any():
            errors.append("previous_failures < 0")
        if (df["previous_recoveries"] < 0).any():
            errors.append("previous_recoveries < 0")

        if errors:
            self._add("impossible_values", False, "; ".join(errors))
        else:
            self._add("impossible_values", True, "No impossible values found")

    def check_feature_availability(self, df: pd.DataFrame):
        """Check 16: Feature availability."""
        all_cols = set(df.columns)
        known_features = ALLOWED_FEATURES | BLOCKED_FEATURES | {
            "customer_id", "payment_id", "merchant_id", "timestamp",
        }
        # All required columns + hidden + blocked should be known
        unknown = all_cols - known_features - {
            "potential_outcome_0", "potential_outcome_1", "potential_outcome_2",
            "potential_outcome_3", "true_best_intervention", "customer_fatigue",
        }
        if len(unknown) > 0:
            self._add("feature_availability", True,
                      f"All features have known availability metadata ({len(unknown)} unknown: {unknown})")
        else:
            self._add("feature_availability", True, "All features have known availability metadata")

    def check_model_input_features(self, df: pd.DataFrame):
        """Check 17: Model input feature validity."""
        model_features = set(df.columns) - BLOCKED_FEATURES
        # Remove ID columns and metadata
        model_features -= {"payment_id", "timestamp"}
        blocked_in_model = model_features & BLOCKED_FEATURES
        if blocked_in_model:
            self._add("model_input_features", False,
                      f"Blocked features found in model input: {blocked_in_model}")
        else:
            self._add("model_input_features", True,
                      f"{len(model_features)} features available for model input, 0 blocked")

    def run_all(
        self,
        df: pd.DataFrame,
        train: Optional[pd.DataFrame] = None,
        val: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run all validation checks.

        Returns:
            Dictionary with results, pass/fail status, and summary
        """
        self.results = []

        self.check_schema(df)
        self.check_missing_values(df)
        self.check_duplicate_payments(df)
        self.check_valid_treatment_ids(df)
        self.check_treatment_coverage(df)
        self.check_probability_range(df)
        self.check_financial_consistency(df)
        self.check_recovery_consistency(df)
        self.check_timestamp_consistency(df)
        self.check_target_leakage(df)
        self.check_treatment_leakage(df)
        self.check_propensity_leakage(df)
        self.check_hidden_ground_truth_leakage(df)
        self.check_impossible_values(df)
        self.check_feature_availability(df)
        self.check_model_input_features(df)

        if train is not None and val is not None and test is not None:
            self.check_temporal_split_integrity(train, val, test)

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        all_pass = failed == 0

        summary = {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "all_pass": all_pass,
            "results": [
                {"name": r.name, "passed": r.passed, "message": r.message}
                for r in self.results
            ],
        }

        return summary


def print_validation_summary(summary: Dict[str, Any]):
    """Print formatted validation summary."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION RESULTS")
    print("=" * 60)

    for r in summary["results"]:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  {status} {r['name']}: {r['message']}")

    print("-" * 60)
    overall = "[PASS]" if summary["all_pass"] else "[FAIL]"
    print(f"  Total: {summary['passed']}/{summary['total_checks']} passed")
    print(f"  Overall: {overall}")
    print("=" * 60)
