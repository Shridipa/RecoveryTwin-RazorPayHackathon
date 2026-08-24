"""
Causal Payment Environment Simulator.

Generates synthetic payment transactions with:
- Realistic customer/merchant/payment features
- Confounded treatment assignment (P(T|X) depends on pre-treatment features)
- Heterogeneous treatment effects
- Potential outcomes for causal evaluation
- Financial consistency checks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import yaml


# Central seed
RANDOM_SEED = 42

# Intervention IDs
CONTROL = 0
RETRY = 1
REMINDER = 2
ALTERNATIVE = 3

INTERVENTION_NAMES = {
    CONTROL: "control",
    RETRY: "retry",
    REMINDER: "reminder",
    ALTERNATIVE: "alternative_method",
}

MERCHANT_CATEGORIES = [
    "electronics", "food_delivery", "travel", "education",
    "subscription", "gaming", "utilities", "fashion",
    "healthcare", "finance",
]

MERCHANT_SIZES = ["small", "medium", "large"]

AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55+"]

PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]

DEVICES = ["mobile", "desktop", "tablet"]

NETWORKS = ["4g", "5g", "wifi", "broadband", "2g"]

BANKS = [
    "SBI", "HDFC", "ICICI", "Axis", "Kotak",
    "PNB", "BankOfBaroda", "YesBank", "IndusInd", "IDBI",
]

FAILURE_REASONS = [
    "insufficient_funds",
    "technical_decline",
    "expired_card",
    "incorrect_pin",
    "network_timeout",
    "bank_unavailable",
    "fraud_suspected",
    "limit_exceeded",
    "account_frozen",
    "customer_abandoned",
]

TRANSACTION_TYPES = ["purchase", "subscription", "transfer", "bill_payment"]


class PaymentEnvironment:
    """
    Causal payment simulator.

    Generates payments with confounded treatment assignment so that:
    - P(T|X) depends on pre-treatment features (failure_reason, customer value, etc.)
    - Each intervention has heterogeneous effects depending on payment characteristics
    - Potential outcomes Y(a) exist for causal evaluation
    - Financial constraints are enforced
    """

    def __init__(self, config_path: Optional[str] = None, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load config if available
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "n_customers": 5000,
            "n_merchants": 500,
            "n_transactions": 50000,
            "seed": RANDOM_SEED,
        }

    def _generate_customers(self, n: int) -> pd.DataFrame:
        """Generate customer profiles."""
        rng = self.rng
        customers = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "customer_age_bucket": rng.choice(AGE_BUCKETS, n, p=[0.15, 0.30, 0.25, 0.18, 0.12]),
            "customer_tenure": np.clip(rng.exponential(12, n).astype(int) + 1, 1, 120),
            "customer_transaction_count": np.clip(rng.negative_binomial(3, 0.3, n), 1, 500),
            "customer_success_rate": np.clip(rng.beta(5, 2, n), 0.0, 1.0),
            "customer_avg_amount": np.clip(rng.lognormal(6.5, 1.0, n), 10, 500000).round(2),
            "customer_recency": np.clip(rng.exponential(48, n), 0.01, 720).round(2),
            "customer_frequency": np.clip(rng.exponential(0.5, n) + 0.1, 0.1, 50).round(2),
            "customer_monetary_value": np.clip(rng.lognormal(8, 1.5, n), 0, 10000000).round(2),
        })

        # Derived activity score
        customers["customer_activity"] = (
            customers["customer_frequency"] * customers["customer_success_rate"]
        ).round(4)

        return customers

    def _generate_merchants(self, n: int) -> pd.DataFrame:
        """Generate merchant profiles."""
        rng = self.rng
        merchants = pd.DataFrame({
            "merchant_id": [f"M{i:04d}" for i in range(n)],
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, n),
            "merchant_size": rng.choice(MERCHANT_SIZES, n, p=[0.5, 0.35, 0.15]),
            "merchant_age": np.clip(rng.exponential(24, n).astype(int) + 1, 1, 240),
            "merchant_success_rate": np.clip(rng.beta(8, 2, n), 0.0, 1.0),
            "merchant_avg_ticket": np.clip(rng.lognormal(6, 1.2, n), 10, 500000).round(2),
        })
        return merchants

    def _generate_transactions(
        self, customers: pd.DataFrame, merchants: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        """Generate payment transactions linked to customers and merchants."""
        rng = self.rng

        # Assign customers and merchants to transactions
        cust_indices = rng.choice(len(customers), n)
        merch_indices = rng.choice(len(merchants), n)

        # Base transaction properties
        amounts = np.clip(rng.lognormal(6.5, 1.2, n), 10, 500000).round(2)

        # Timestamps: uniform across 2024
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")
        timestamps = pd.date_range(start, end, periods=n) + pd.to_timedelta(
            rng.uniform(0, 86400, n), unit="s"
        )

        # Choose failure reason - weighted
        failure_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.10, 0.08, 0.06, 0.07, 0.04, 0.25])
        failure_weights = failure_weights / failure_weights.sum()

        transactions = pd.DataFrame({
            "payment_id": [f"P{i:06d}" for i in range(n)],
            "customer_id": customers.iloc[cust_indices]["customer_id"].values,
            "merchant_id": merchants.iloc[merch_indices]["merchant_id"].values,
            "timestamp": timestamps,
            "amount": amounts,
            "payment_method": rng.choice(PAYMENT_METHODS, n, p=[0.40, 0.30, 0.20, 0.10]),
            "device": rng.choice(DEVICES, n, p=[0.65, 0.25, 0.10]),
            "network": rng.choice(NETWORKS, n, p=[0.35, 0.20, 0.25, 0.15, 0.05]),
            "bank": rng.choice(BANKS, n),
            "transaction_type": rng.choice(TRANSACTION_TYPES, n, p=[0.45, 0.20, 0.15, 0.20]),
            "failure_reason": rng.choice(FAILURE_REASONS, n, p=failure_weights),
        })

        # Merge customer features
        cust_features = [
            "customer_id", "customer_age_bucket", "customer_tenure",
            "customer_transaction_count", "customer_success_rate",
            "customer_avg_amount", "customer_recency", "customer_frequency",
            "customer_monetary_value", "customer_activity",
        ]
        transactions = transactions.merge(customers[cust_features], on="customer_id", how="left")

        # Merge merchant features
        merch_features = [
            "merchant_id", "merchant_category", "merchant_size",
            "merchant_age", "merchant_success_rate", "merchant_avg_ticket",
        ]
        transactions = transactions.merge(merchants[merch_features], on="merchant_id", how="left")

        # Behavioral features
        transactions["attempt_count"] = rng.poisson(1.5, n) + 1
        transactions["hours_since_failure"] = np.clip(rng.exponential(12, n), 0.1, 168).round(2)
        transactions["previous_failures"] = rng.poisson(0.8, n)
        transactions["previous_recoveries"] = np.clip(
            transactions["previous_failures"] * rng.beta(3, 4, n), 0, 20
        ).astype(int)

        return transactions

    def _compute_failure_severity(self, df: pd.DataFrame) -> np.ndarray:
        """Compute failure severity score [0,1] - higher means more likely to recover.

        This score drives the BASE recovery probability under control.
        Stronger feature-driven variance here means better ML signal.
        """
        rng = self.rng
        n = len(df)

        severity = np.zeros(n)

        # --- FAILURE REASON (dominant signal, ~40% of variance) ---
        reason_map = {
            "network_timeout": 0.70,      # Very recoverable
            "technical_decline": 0.65,     # Very recoverable
            "bank_unavailable": 0.60,      # Recoverable
            "expired_card": 0.45,          # Moderately recoverable
            "incorrect_pin": 0.40,         # Moderately recoverable
            "fraud_suspected": 0.25,       # Less recoverable
            "customer_abandoned": 0.20,    # Hard to recover
            "insufficient_funds": 0.30,    # Depends on customer
            "account_frozen": 0.15,        # Very hard
            "limit_exceeded": 0.18,        # Very hard
        }
        for reason, score in reason_map.items():
            mask = (df["failure_reason"] == reason).values
            severity[mask] = score

        # --- CUSTOMER BEHAVIOR (strong signal, ~25% of variance) ---
        # High-activity customers are much more likely to come back
        activity_z = (df["customer_activity"].values - df["customer_activity"].mean()) / (df["customer_activity"].std() + 1e-6)
        severity += np.clip(activity_z, -1.5, 1.5) * 0.15

        # Customer success rate is a strong predictor
        cust_sr = df["customer_success_rate"].values
        severity += (cust_sr - cust_sr.mean()) / (cust_sr.std() + 1e-6) * 0.12

        # Customer tenure: longer tenure = more likely to return
        tenure_z = (df["customer_tenure"].values - df["customer_tenure"].mean()) / (df["customer_tenure"].std() + 1e-6)
        severity += np.clip(tenure_z, -1.5, 1.5) * 0.08

        # Previous recoveries: strong signal of future recovery
        prev_rec = df["previous_recoveries"].values
        severity += np.clip(prev_rec / 5.0, 0, 0.15)

        # --- MERCHANT QUALITY (~10% of variance) ---
        merchant_sr = df["merchant_success_rate"].values
        severity += (merchant_sr - merchant_sr.mean()) / (merchant_sr.std() + 1e-6) * 0.08

        # --- AMOUNT EFFECT: small amounts are easier to recover ---
        amount_z = (df["amount"].values - df["amount"].mean()) / (df["amount"].std() + 1e-6)
        severity -= np.clip(amount_z, -1.5, 1.5) * 0.06

        # --- BEHAVIORAL CONTEXT ---
        # Fewer previous failures = more recoverable
        prev_fail = df["previous_failures"].values
        severity -= np.clip(prev_fail / 5.0, 0, 0.12)

        # Lower attempt count = more recoverable (fresh failure)
        attempt_z = (df["attempt_count"].values - df["attempt_count"].mean()) / (df["attempt_count"].std() + 1e-6)
        severity -= np.clip(attempt_z, -1, 1.5) * 0.06

        # Add moderate noise (but not enough to drown signal)
        severity += rng.normal(0, 0.04, n)

        return np.clip(severity, 0.02, 0.98)

    def _compute_treatment_propensities(
        self, df: pd.DataFrame, severity: np.ndarray
    ) -> np.ndarray:
        """
        Compute treatment propensities P(T|X) that create confounding.

        Treatment assignment depends on pre-treatment features:
        - Technical failures -> more retry
        - High customer responsiveness -> more reminder
        - Payment method failures -> more alternative
        - High severity -> less control
        """
        n = len(df)
        rng = self.rng

        # Base log-odds for each intervention
        log_odds = np.zeros((n, 4))

        # -- Control: slightly higher for low-severity, abandoned --
        # Keep this weak to avoid overly deterministic control assignment
        log_odds[:, CONTROL] = -0.4 - severity * 0.7
        abandoned = (df["failure_reason"] == "customer_abandoned").values
        log_odds[abandoned, CONTROL] += 0.5

        # -- Retry: higher for technical failures, but not too dominant --
        log_odds[:, RETRY] = 0.0 + severity * 0.4
        technical = df["failure_reason"].isin([
            "technical_decline", "network_timeout", "bank_unavailable"
        ]).values
        log_odds[technical, RETRY] += 0.6

        # High attempt count reduces retry propensity (fatigue)
        attempt_penalty = np.clip(df["attempt_count"].values - 2, 0, 5) * 0.3
        log_odds[:, RETRY] -= attempt_penalty

        # -- Reminder: higher for responsive customers, abandoned --
        log_odds[:, REMINDER] = 0.2 + df["customer_activity"].values * 0.3
        log_odds[abandoned, REMINDER] += 0.8

        # -- Alternative: higher for card/method failures, high value --
        log_odds[:, ALTERNATIVE] = 0.1 + severity * 0.4
        card_fail = df["failure_reason"].isin([
            "expired_card", "incorrect_pin"
        ]).values
        log_odds[card_fail, ALTERNATIVE] += 1.2

        # Higher amount -> more likely to try alternative
        amount_z = (df["amount"].values - df["amount"].median()) / (df["amount"].std() + 1e-6)
        log_odds[:, ALTERNATIVE] += np.clip(amount_z, -1, 1) * 0.4

        # Add noise for stochasticity
        log_odds += rng.normal(0, 0.15, log_odds.shape)

        # Convert to probabilities via softmax
        log_odds_shifted = log_odds - log_odds.max(axis=1, keepdims=True)
        probs = np.exp(log_odds_shifted)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def _compute_treatment_effect(
        self, df: pd.DataFrame, treatment: np.ndarray, severity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute heterogeneous treatment effects.

        Returns:
            base_prob: P(recovery under control)
            effect: incremental effect for each intervention
            potential_outcomes: Y(a) for a in {0,1,2,3}
        """
        rng = self.rng
        n = len(df)

        # Base recovery probability under control
        # severity already encodes strong feature-driven signal (0.02 to 0.98)
        # Scale it to a meaningful probability range
        base_prob = np.clip(
            0.03 + severity * 0.42
            + rng.normal(0, 0.02, n),
            0.01, 0.55,
        )

        # Treatment effects (heterogeneous, feature-dependent)
        effects = np.zeros((n, 4))
        effects[:, CONTROL] = 0.0

        technical = df["failure_reason"].isin([
            "technical_decline", "network_timeout", "bank_unavailable"
        ]).values
        card_fail = df["failure_reason"].isin([
            "expired_card", "incorrect_pin"
        ]).values
        abandoned = (df["failure_reason"] == "customer_abandoned").values
        responsive = df["customer_activity"].values > df["customer_activity"].median()

        # --- RETRY: strong for technical failures, weak for abandoned ---
        retry_base = np.where(
            technical,
            0.20 + severity * 0.15,
            np.where(abandoned, 0.02, 0.06 + severity * 0.05),
        )
        # Diminishing returns with attempts
        retry_base *= np.clip(1.0 - (df["attempt_count"].values - 1) * 0.10, 0.3, 1.0)
        effects[:, RETRY] = np.clip(retry_base, 0.0, 0.40)

        # --- REMINDER: strong for responsive/abandoned, weak for technical ---
        reminder_base = np.where(
            abandoned | responsive,
            0.18 + severity * 0.12,
            0.04 + severity * 0.04,
        )
        effects[:, REMINDER] = np.clip(reminder_base, 0.0, 0.35)

        # --- ALTERNATIVE: strong for card/method failures, high-value customers ---
        high_val = df["customer_monetary_value"].values > df["customer_monetary_value"].quantile(0.7)
        alt_base = np.where(
            card_fail,
            0.22 + severity * 0.12,
            np.where(high_val, 0.10 + severity * 0.08, 0.05 + severity * 0.04),
        )
        effects[:, ALTERNATIVE] = np.clip(alt_base, 0.0, 0.45)

        # Potential outcomes Y(a) for each intervention
        potential_probs = np.zeros((n, 4))
        for a in range(4):
            potential_probs[:, a] = np.clip(base_prob + effects[:, a], 0.01, 0.99)

        # Generate binary potential outcomes
        potential_outcomes = np.zeros((n, 4), dtype=int)
        for a in range(4):
            potential_outcomes[:, a] = (rng.random(n) < potential_probs[:, a]).astype(int)

        return base_prob, effects, potential_outcomes

    def _compute_recovery_time(
        self, df: pd.DataFrame, recovered: np.ndarray, treatment: np.ndarray,
        severity: np.ndarray
    ) -> np.ndarray:
        """Compute recovery time in hours for recovered payments."""
        rng = self.rng
        n = len(df)
        times = np.zeros(n)
    
        recovered_mask = recovered == 1

        # Base recovery time scale depends on intervention type
        scale_map = {RETRY: 2.0, REMINDER: 8.0, ALTERNATIVE: 4.0, CONTROL: 12.0}

        # Generate recovery times per treatment for recovered rows
        for t_id, scale in scale_map.items():
            mask = (treatment == t_id) & recovered_mask
            count = mask.sum()
            if count > 0:
                times[mask] = rng.exponential(scale, count)

        # Faster for technical failures (retry) and card (alternative)
        technical = df["failure_reason"].isin([
            "technical_decline", "network_timeout"
        ]).values & recovered_mask
        times[technical] *= 0.6

        # Faster for high-activity customers
        high_activity = (df["customer_activity"].values > df["customer_activity"].median()) & recovered_mask
        times[high_activity] *= 0.8

        times[recovered_mask] = np.clip(times[recovered_mask], 0.1, 168)
        times[~recovered_mask] = 0.0  # No recovery -> 0 hours

        return np.round(times, 2)

    def _compute_intervention_cost(
        self, treatment: np.ndarray, amount: np.ndarray
    ) -> np.ndarray:
        """Compute intervention costs."""
        cost_per_unit = {CONTROL: 0.0, RETRY: 0.50, REMINDER: 1.00, ALTERNATIVE: 2.50}
        costs = np.array([cost_per_unit[t] for t in treatment])

        # Scale cost slightly with amount for alternative
        alt_mask = treatment == ALTERNATIVE
        costs[alt_mask] += amount[alt_mask] * 0.001  # 0.1% of amount

        return np.round(costs, 2)

    def _compute_revenue(
        self, amount: np.ndarray, recovered: np.ndarray
    ) -> np.ndarray:
        """Revenue recovered = amount if recovered, else 0."""
        return np.where(recovered == 1, amount, 0.0)

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate the complete synthetic payment environment.

        Returns:
            df: DataFrame with all features, treatments, and outcomes
            metadata: Summary statistics and generation info
        """
        cfg = self.config
        n_cust = cfg.get("n_customers", 5000)
        n_merch = cfg.get("n_merchants", 500)
        n_txn = cfg.get("n_transactions", 50000)

        print(f"Generating {n_txn} transactions across {n_cust} customers, {n_merchants if 'merchants' in cfg else n_merch} merchants...")

        # Generate entities
        customers = self._generate_customers(n_cust)
        merchants = self._generate_merchants(n_merch)

        # Generate transactions
        df = self._generate_transactions(customers, merchants, n_txn)

        # Compute failure severity
        severity = self._compute_failure_severity(df)

        # Compute treatment propensities
        propensities = self._compute_treatment_propensities(df, severity)

        # Assign treatments based on propensities
        treatment = np.array([
            self.rng.choice(4, p=propensities[i]) for i in range(len(df))
        ])

        # Compute treatment effects and potential outcomes
        base_prob, effects, potential_outcomes = self._compute_treatment_effect(
            df, treatment, severity
        )

        # Compute observed outcomes using assigned treatment
        actual_probs = np.clip(base_prob + effects[np.arange(len(df)), treatment], 0.01, 0.99)
        recovered = (self.rng.random(len(df)) < actual_probs).astype(int)

        # Compute recovery time
        recovery_time = self._compute_recovery_time(df, recovered, treatment, severity)

        # Compute financial fields
        revenue = self._compute_revenue(df["amount"].values, recovered)
        intervention_cost = self._compute_intervention_cost(treatment, df["amount"].values)

        # True best intervention (hidden ground truth)
        true_best = potential_outcomes.sum(axis=0)
        # For each customer, the best intervention is the one with highest potential outcome prob
        true_best_intervention = potential_outcomes.mean(axis=0).argmax()
        # Per-row best
        true_best_per_row = potential_outcomes.mean(axis=1).argmax()  # This is approximate
        # Better: use potential outcome probs
        pot_probs = np.zeros((len(df), 4))
        for a in range(4):
            pot_probs[:, a] = np.clip(base_prob + effects[:, a], 0.01, 0.99)
        true_best_per_row = pot_probs.argmax(axis=1)

        # Add treatment and outcome columns
        df["intervention"] = treatment
        df["recovered"] = recovered
        df["recovery_time_hours"] = recovery_time
        df["revenue_recovered"] = revenue
        df["intervention_cost"] = intervention_cost

        # Hidden columns (for evaluation only, blocked from model input)
        df["potential_outcome_0"] = potential_outcomes[:, 0]
        df["potential_outcome_1"] = potential_outcomes[:, 1]
        df["potential_outcome_2"] = potential_outcomes[:, 2]
        df["potential_outcome_3"] = potential_outcomes[:, 3]
        df["true_best_intervention"] = true_best_per_row

        # Propensity scores
        df["propensity_0"] = propensities[:, 0]
        df["propensity_1"] = propensities[:, 1]
        df["propensity_2"] = propensities[:, 2]
        df["propensity_3"] = propensities[:, 3]

        # Customer fatigue (accumulated intervention pressure)
        df["customer_fatigue"] = np.clip(
            df["previous_failures"] + (treatment != CONTROL).astype(int) * 0.5,
            0, 10,
        ).round(2)

        # Metadata
        metadata = {
            "n_customers": n_cust,
            "n_merchants": n_merch,
            "n_transactions": len(df),
            "recovery_rate": float(recovered.mean()),
            "treatment_distribution": {
                INTERVENTION_NAMES[i]: int((treatment == i).sum())
                for i in range(4)
            },
            "treatment_pct": {
                INTERVENTION_NAMES[i]: float((treatment == i).mean())
                for i in range(4)
            },
            "revenue_at_risk": float(df["amount"].sum()),
            "revenue_recovered": float(revenue.sum()),
            "mean_recovery_time": float(recovery_time[recovered == 1].mean()) if recovered.any() else 0,
            "mean_intervention_cost": float(intervention_cost.mean()),
            "seed": self.seed,
        }

        return df, metadata

    def validate_financial(self, df: pd.DataFrame) -> list:
        """Run financial consistency checks."""
        errors = []

        # No negative recovery
        if (df["revenue_recovered"] < 0).any():
            errors.append("Negative revenue_recovered found")

        # Revenue <= amount for recovered payments
        recovered = df["recovered"] == 1
        if (df.loc[recovered, "revenue_recovered"] > df.loc[recovered, "amount"]).any():
            errors.append("Revenue recovered > amount for some transactions")

        # Revenue = 0 for unrecovered
        unrecovered = df["recovered"] == 0
        if (df.loc[unrecovered, "revenue_recovered"] > 0).any():
            errors.append("Non-zero revenue for unrecovered transactions")

        # Valid probabilities
        for col in ["propensity_0", "propensity_1", "propensity_2", "propensity_3"]:
            if col in df.columns:
                if (df[col] < 0).any() or (df[col] > 1).any():
                    errors.append(f"{col} out of [0,1] range")

        # Valid treatment IDs
        if not df["intervention"].isin([0, 1, 2, 3]).all():
            errors.append("Invalid intervention IDs found")

        # Recovery time > 0 only for recovered
        if (df.loc[unrecovered, "recovery_time_hours"] > 0).any():
            errors.append("Non-zero recovery time for unrecovered transactions")

        return errors


def generate_and_save(config_path: Optional[str] = None, output_dir: str = "data/synthetic/debug") -> Tuple[pd.DataFrame, Dict]:
    """Generate dataset and save to parquet."""
    env = PaymentEnvironment(config_path=config_path)
    df, metadata = env.generate()

    # Validate financial
    fin_errors = env.validate_financial(df)
    if fin_errors:
        print("FINANCIAL VALIDATION ERRORS:")
        for e in fin_errors:
            print(f"  [FAIL] {e}")
    else:
        print("Financial validation: [PASS]")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / "transactions.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} rows to {parquet_path}")

    # Save metadata
    import json
    meta_path = output_path / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {meta_path}")

    return df, metadata


if __name__ == "__main__":
    df, meta = generate_and_save()
    print("\n=== GENERATION SUMMARY ===")
    print(f"Customers:       {meta['n_customers']}")
    print(f"Merchants:        {meta['n_merchants']}")
    print(f"Transactions:     {meta['n_transactions']}")
    print(f"Recovery rate:    {meta['recovery_rate']:.4f}")
    print(f"Revenue at risk:  {meta['revenue_at_risk']:,.2f}")
    print(f"Revenue recovered:{meta['revenue_recovered']:,.2f}")
    print(f"\nTreatment distribution:")
    for name, count in meta["treatment_distribution"].items():
        pct = meta["treatment_pct"][name]
        print(f"  {name}: {count} ({pct:.1%})")
