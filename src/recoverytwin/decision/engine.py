"""
Counterfactual Decision Engine for RecoveryTwin.

Combines:
  - Phase 4 predictive recovery probability P(Y|X,T)
  - Phase 6 causal/uplift CATE estimates
  - Phase 5 time-to-recovery survival information
  - Intervention costs, customer fatigue, policy constraints

into a single financial decision per failed payment.

Core formulation:
  p_a  = P(Y(a)=1 | X)          -- recovery probability under action a
  p_0  = P(Y(0)=1 | X)          -- control probability
  CATE_a = p_a - p_0             -- incremental recovery probability

  IncrementalRevenue_a = Amount * CATE_a
  NetValue_a = IncrementalRevenue_a - Cost_a
  TimeAdjustedValue_a = NetValue_a * TimeDiscount(a)
  a* = argmax(TimeAdjustedValue_a)  subject to Eligible(a)=True
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml


# ============================================================
# ACTION DEFINITIONS
# ============================================================

ACTION_NAMES = {
    0: "control",
    1: "retry",
    2: "reminder",
    3: "alternative_method",
}

ACTION_LABELS = {
    0: "Do Nothing",
    1: "Retry",
    2: "Reminder",
    3: "Alternative Method",
}

# Default intervention costs (synthetic, configurable via policy.yaml)
DEFAULT_COSTS = {
    0: 0.0,
    1: 0.50,
    2: 1.00,
    3: 2.50,
}

DEFAULT_FATIGUE_COST = {
    0: 0.0,
    1: 1.0,
    2: 1.0,
    3: 2.0,
}


def load_policy_config(config_path: str = "configs/policy.yaml") -> Dict:
    """Load policy configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        # Return defaults
        return {
            "action_costs": DEFAULT_COSTS,
            "fatigue_cost": DEFAULT_FATIGUE_COST,
            "max_retries": 3,
            "max_interventions": 5,
            "max_fatigue": 4.0,
            "cooldown_hours": 24,
            "min_incremental_value": 0.0,
            "time_discount": {
                "enabled": True,
                "half_life_hours": 24.0,
            },
        }

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Map intervention_costs dict to action cost mapping
    ic = cfg.get("intervention_costs", {})
    action_costs = {
        0: ic.get("control", 0.0),
        1: ic.get("retry", 0.50),
        2: ic.get("reminder", 1.00),
        3: ic.get("alternative_method", 2.50),
    }

    limits = cfg.get("limits", {})
    time_cfg = cfg.get("financial", {}).get("time_discount", {})

    return {
        "action_costs": action_costs,
        "fatigue_cost": DEFAULT_FATIGUE_COST,
        "max_retries": limits.get("max_retry_attempts", 3),
        "max_interventions": limits.get("max_total_interventions", 5),
        "max_fatigue": limits.get("customer_fatigue_threshold", 4),
        "cooldown_hours": limits.get("cooldown_hours", 24),
        "min_incremental_value": cfg.get("financial", {}).get("min_incremental_value", 0.0),
        "time_discount": {
            "enabled": time_cfg.get("enabled", True),
            "half_life_hours": time_cfg.get("half_life_hours", 24.0),
        },
    }


# ============================================================
# TIME DISCOUNT FUNCTIONS
# ============================================================

def time_discount_exponential(hours: float, half_life: float = 24.0) -> float:
    """
    Exponential time discount: value decays as recovery takes longer.

    Returns a value in (0, 1] where 1 = instant recovery.
    half_life: number of hours at which discount = 0.5
    """
    if hours <= 0:
        return 1.0
    return np.exp(-0.693 * hours / max(half_life, 0.1))


def time_discount_step(hours: float, thresholds: Dict[float, float] = None) -> float:
    """
    Step-function time discount.
    thresholds: dict mapping hours -> discount factor.
    """
    if thresholds is None:
        thresholds = {1.0: 0.95, 6.0: 0.85, 24.0: 0.70, 72.0: 0.50}

    discount = 1.0
    for h in sorted(thresholds.keys()):
        if hours > h:
            discount = thresholds[h]
        else:
            break
    return discount


# ============================================================
# POLICY ELIGIBILITY CHECKS
# ============================================================

class PolicyFilter:
    """Enforces policy constraints on action selection."""

    def __init__(self, config: Dict):
        self.config = config
        self.max_retries = config.get("max_retries", 3)
        self.max_interventions = config.get("max_interventions", 5)
        self.max_fatigue = config.get("max_fatigue", 4.0)
        self.cooldown_hours = config.get("cooldown_hours", 24)

    def check_eligibility(
        self,
        action: int,
        attempt_count: int,
        previous_failures: int,
        customer_fatigue: float,
        hours_since_failure: float,
        recovered: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if action is eligible given current payment state.

        Returns (eligible, reason_string).
        """
        # Stopping rule: if already recovered, no intervention needed
        if recovered:
            return False, "Payment already recovered"

        # Control is always eligible
        if action == 0:
            return True, "Control always eligible"

        # Maximum intervention count
        if attempt_count >= self.max_interventions:
            return False, f"Max interventions ({self.max_interventions}) exceeded"

        # Retry-specific limits
        if action == 1:
            if previous_failures >= self.max_retries:
                return False, f"Max retry attempts ({self.max_retries}) exceeded"

        # Customer fatigue limit
        fatigue_cost = self.config.get("fatigue_cost", DEFAULT_FATIGUE_COST).get(action, 1.0)
        if customer_fatigue + fatigue_cost > self.max_fatigue:
            return False, f"Customer fatigue ({customer_fatigue:.1f} + {fatigue_cost:.0f} = {customer_fatigue + fatigue_cost:.1f}) exceeds threshold ({self.max_fatigue})"

        # Cooldown check (hours since last failure)
        if hours_since_failure < self.cooldown_hours / 24.0:
            # Allow but flag
            pass  # Cooldown is soft — flag but don't block

        return True, "Eligible"


# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    """
    Counterfactual Decision Engine.

    For each failed payment:
      1. Generate counterfactual recovery probabilities for all actions
      2. Estimate CATEs
      3. Calculate financial values
      4. Apply policy constraints
      5. Select optimal action
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = load_policy_config()
        self.config = config
        self.action_costs = config["action_costs"]
        self.fatigue_cost = config.get("fatigue_cost", DEFAULT_FATIGUE_COST)
        self.time_discount_cfg = config.get("time_discount", {"enabled": True, "half_life_hours": 24.0})
        self.min_incremental_value = config.get("min_incremental_value", 0.0)
        self.policy_filter = PolicyFilter(config)

    def generate_counterfactual_table(
        self,
        df: pd.DataFrame,
        predict_fn,
    ) -> pd.DataFrame:
        """
        Generate counterfactual recovery probabilities for all actions.

        predict_fn(df, action) -> np.ndarray of P(Y=1|X, action)
        """
        result = df[["payment_id", "amount"]].copy()

        control_probs = predict_fn(df, 0)
        result["control_prob"] = control_probs

        for action in [1, 2, 3]:
            probs = predict_fn(df, action)
            action_name = ACTION_NAMES[action]
            result[f"{action_name}_prob"] = probs

        return result

    def calculate_cate(
        self,
        counterfactual_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate CATE for each action vs control."""
        result = counterfactual_table.copy()
        p0 = result["control_prob"].values

        for action in [1, 2, 3]:
            action_name = ACTION_NAMES[action]
            pa = result[f"{action_name}_prob"].values
            result[f"{action_name}_cate"] = pa - p0

        return result

    def calculate_financial_value(
        self,
        df: pd.DataFrame,
        counterfactual_table: pd.DataFrame,
        survival_probs: Dict[int, Dict[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate financial value for each action.

        For each action:
          incremental_revenue = amount * CATE
          net_value = incremental_revenue - cost
          time_adjusted_value = net_value * time_discount
        """
        result = counterfactual_table.copy()
        amounts = df["amount"].values

        for action in [1, 2, 3]:
            action_name = ACTION_NAMES[action]
            cate = result[f"{action_name}_cate"].values
            prob = result[f"{action_name}_prob"].values

            # Incremental revenue
            incremental_rev = amounts * cate
            result[f"{action_name}_incremental_revenue"] = incremental_rev

            # Intervention cost
            cost = self.action_costs.get(action, 0.0)
            result[f"{action_name}_cost"] = cost

            # Net incremental value (before time discount)
            net_value = incremental_rev - cost
            result[f"{action_name}_net_value"] = net_value

            # Time discount
            if self.time_discount_cfg.get("enabled", True) and survival_probs and action in survival_probs:
                # Use median recovery time from survival model
                median_time = survival_probs[action].get("median_hours", 24.0)
                half_life = self.time_discount_cfg.get("half_life_hours", 24.0)
                td = time_discount_exponential(median_time, half_life)
            else:
                # Default: no time discount if survival info unavailable
                td = 1.0

            result[f"{action_name}_time_discount"] = td
            result[f"{action_name}_time_adjusted_value"] = net_value * td

        return result

    def select_actions(
        self,
        df: pd.DataFrame,
        financial_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select optimal action for each payment.

        Applies policy constraints, then selects argmax(time_adjusted_value).
        Falls back to control if no intervention has positive value.
        """
        result = financial_table.copy()

        n = len(result)
        recommended = np.zeros(n, dtype=int)
        recommended_value = np.zeros(n)
        eligible_mask = np.ones((n, 4), dtype=bool)
        rejection_reasons = [[] for _ in range(n)]

        for i in range(n):
            amount = df["amount"].iloc[i]
            attempt_count = df["attempt_count"].iloc[i]
            previous_failures = df["previous_failures"].iloc[i]
            fatigue = df["customer_fatigue"].iloc[i]
            hours_since = df["hours_since_failure"].iloc[i]
            recovered = bool(df["recovered"].iloc[i])

            best_action = 0
            best_value = 0.0  # Control always has value 0

            for action in range(4):
                action_name = ACTION_NAMES[action]

                if action == 0:
                    # Control is always eligible with value 0
                    continue

                # Check eligibility
                eligible, reason = self.policy_filter.check_eligibility(
                    action=action,
                    attempt_count=attempt_count,
                    previous_failures=previous_failures,
                    customer_fatigue=fatigue,
                    hours_since_failure=hours_since,
                    recovered=recovered,
                )

                if not eligible:
                    eligible_mask[i, action] = False
                    rejection_reasons[i].append(f"{ACTION_LABELS[action]}: {reason}")
                    continue

                # Get time-adjusted value
                tav_col = f"{action_name}_time_adjusted_value"
                if tav_col in result.columns:
                    value = result[tav_col].iloc[i]
                else:
                    value = 0.0

                # Check minimum incremental value threshold
                if value < self.min_incremental_value:
                    eligible_mask[i, action] = False
                    rejection_reasons[i].append(f"{ACTION_LABELS[action]}: Net value ({value:.2f}) below threshold")
                    continue

                if value > best_value:
                    best_value = value
                    best_action = action

            recommended[i] = best_action
            recommended_value[i] = best_value

        result["recommended_action"] = recommended
        result["recommended_action_name"] = [ACTION_LABELS[a] for a in recommended]
        result["recommended_value"] = recommended_value

        for action in [0, 1, 2, 3]:
            action_name = ACTION_NAMES[action]
            result[f"{action_name}_eligible"] = eligible_mask[:, action]

        result["rejection_reasons"] = ["; ".join(r) for r in rejection_reasons]
        result["n_eligible_actions"] = eligible_mask.sum(axis=1)

        return result

    def build_decision_table(
        self,
        df: pd.DataFrame,
        predict_fn,
        survival_probs: Dict[int, Dict[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Full decision pipeline for a batch of payments.

        Returns complete decision table with all intermediate calculations.
        """
        # Step 1: Counterfactual predictions
        cf_table = self.generate_counterfactual_table(df, predict_fn)

        # Step 2: CATE
        cf_table = self.calculate_cate(cf_table)

        # Step 3: Financial value
        fin_table = self.calculate_financial_value(df, cf_table, survival_probs)

        # Step 4: Action selection
        decision_table = self.select_actions(df, fin_table)

        return decision_table


# ============================================================
# ORACLE EVALUATION
# ============================================================

def evaluate_oracle(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    action_costs: Dict[int, float],
    time_discount_enabled: bool = True,
    half_life_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Evaluate policies against oracle (ground truth) on the test set.

    Oracle = action with highest TRUE net financial value.
    """
    n = len(df)
    amounts = df["amount"].values
    costs = np.array([action_costs.get(a, 0.0) for a in range(4)])

    # True recovery outcomes for each action (potential outcomes)
    true_probs = np.column_stack([
        df[f"potential_outcome_{a}"].values.astype(float)
        for a in range(4)
    ])

    # True recovery time (when available)
    true_recovery_time = df["recovery_time_hours"].values

    # True financial value per action:
    # For recovered payments (potential_outcome=1): revenue = amount, cost = action_cost
    # For non-recovered: revenue = 0, cost = 0 (no intervention needed if not recovered)
    # But we simulate: value = potential_outcome * amount - costs[action]
    true_value = np.zeros((n, 4))
    for a in range(4):
        recovered = true_probs[:, a]  # Binary: 0 or 1
        true_value[:, a] = recovered * amounts - costs[a]

    # Oracle policy
    oracle_actions = np.argmax(true_value, axis=1)

    # Build policy comparison
    policies = {}

    # 1. Do Nothing (control)
    policies["do_nothing"] = {
        "actions": np.zeros(n, dtype=int),
        "value": true_value[:, 0].sum(),
        "recovery_rate": true_probs[:, 0].mean(),
        "n_interventions": 0,
    }

    # 2. Always Retry
    policies["always_retry"] = {
        "actions": np.ones(n, dtype=int),
        "value": true_value[:, 1].sum(),
        "recovery_rate": true_probs[:, 1].mean(),
        "n_interventions": n,
    }

    # 3. Max Probability (predictive)
    # Use the counterfactual predictions from decision_table
    prob_cols = [f"{ACTION_NAMES[a]}_prob" for a in [1, 2, 3]]
    if all(c in decision_table.columns for c in prob_cols):
        pred_probs = np.column_stack([
            decision_table["control_prob"].values,
            *[decision_table[c].values for c in prob_cols],
        ])
        # Select action with highest predicted probability (including control)
        max_prob_actions = np.argmax(pred_probs, axis=1)
        # But always prefer non-control if control is not the best
        # Actually: if best predicted prob is control, choose control
        policies["max_probability"] = {
            "actions": max_prob_actions,
            "value": sum(true_value[i, a] for i, a in enumerate(max_prob_actions)),
            "recovery_rate": np.mean([true_probs[i, a] for i, a in enumerate(max_prob_actions)]),
            "n_interventions": int(np.sum(max_prob_actions > 0)),
        }
    else:
        policies["max_probability"] = policies["do_nothing"].copy()

    # 4. CATE Policy (highest CATE)
    cate_cols = [f"{ACTION_NAMES[a]}_cate" for a in [1, 2, 3]]
    if all(c in decision_table.columns for c in cate_cols):
        cates = np.column_stack([
            np.zeros(n),  # control CATE = 0
            *[decision_table[c].values for c in cate_cols],
        ])
        cate_actions = np.argmax(cates, axis=1)
        policies["cate_policy"] = {
            "actions": cate_actions,
            "value": sum(true_value[i, a] for i, a in enumerate(cate_actions)),
            "recovery_rate": np.mean([true_probs[i, a] for i, a in enumerate(cate_actions)]),
            "n_interventions": int(np.sum(cate_actions > 0)),
        }
    else:
        policies["cate_policy"] = policies["do_nothing"].copy()

    # 5. RecoveryTwin (our engine)
    if "recommended_action" in decision_table.columns:
        rt_actions = decision_table["recommended_action"].values
        policies["recoverytwin"] = {
            "actions": rt_actions,
            "value": sum(true_value[i, a] for i, a in enumerate(rt_actions)),
            "recovery_rate": np.mean([true_probs[i, a] for i, a in enumerate(rt_actions)]),
            "n_interventions": int(np.sum(rt_actions > 0)),
        }

    # 6. Oracle
    policies["oracle"] = {
        "actions": oracle_actions,
        "value": true_value.sum(axis=1).sum() - true_value[np.arange(n), oracle_actions].sum() + true_value[np.arange(n), oracle_actions].sum(),
        "value": sum(true_value[i, a] for i, a in enumerate(oracle_actions)),
        "recovery_rate": np.mean([true_probs[i, a] for i, a in enumerate(oracle_actions)]),
        "n_interventions": int(np.sum(oracle_actions > 0)),
    }
    # Fix: recalculate oracle value cleanly
    policies["oracle"]["value"] = sum(true_value[i, oracle_actions[i]] for i in range(n))

    return {
        "policies": policies,
        "true_value_matrix": true_value,
        "oracle_actions": oracle_actions,
        "n_samples": n,
    }


# ============================================================
# SEGMENT ANALYSIS
# ============================================================

def segment_analysis(
    df: pd.DataFrame,
    decision_table: pd.DataFrame,
    oracle_result: Dict,
) -> pd.DataFrame:
    """Analyze policy performance by segment."""
    segments = []

    # Amount buckets
    df_copy = df.copy()
    df_copy["amount_bucket"] = pd.cut(
        df_copy["amount"],
        bins=[0, 500, 2000, 5000, 500000],
        labels=["low", "medium", "high", "very_high"],
    )

    # Failure reason
    for group_col, group_name in [
        ("amount_bucket", "amount_bucket"),
        ("failure_reason", "failure_reason"),
        ("customer_activity", "customer_activity_bucket"),
    ]:
        if group_col == "customer_activity":
            df_copy[group_col] = pd.cut(
                df_copy[group_col],
                bins=3,
                labels=["low", "medium", "high"],
            )

        if group_col not in df_copy.columns:
            continue

        for seg_val, seg_df_idx in df_copy.groupby(group_col).groups.items():
            if len(seg_df_idx) < 10:
                continue

            seg_idx = list(seg_df_idx)
            seg_dt = decision_table.iloc[seg_idx]
            seg_df = df.iloc[seg_idx]

            # RecoveryTwin allocation
            rt_actions = seg_dt["recommended_action"].values if "recommended_action" in seg_dt.columns else np.zeros(len(seg_idx))
            action_dist = {ACTION_LABELS[a]: float(np.mean(rt_actions == a)) for a in range(4)}

            # True value
            amounts = seg_df["amount"].values
            true_po = np.column_stack([
                seg_df[f"potential_outcome_{a}"].values.astype(float)
                for a in range(4)
            ])

            # Oracle value
            oracle_actions_seg = oracle_result["oracle_actions"][seg_idx]
            oracle_val = sum(
                true_po[i, oracle_actions_seg[i]] * amounts[i] - DEFAULT_COSTS.get(oracle_actions_seg[i], 0)
                for i in range(len(seg_idx))
            )

            # RT value
            rt_val = sum(
                true_po[i, rt_actions[i]] * amounts[i] - DEFAULT_COSTS.get(int(rt_actions[i]), 0)
                for i in range(len(seg_idx))
            )

            segments.append({
                "segment": group_name,
                "value": str(seg_val),
                "n": len(seg_idx),
                "oracle_value": float(oracle_val),
                "rt_value": float(rt_val),
                "regret": float(oracle_val - rt_val),
                "regret_pct": float((oracle_val - rt_val) / max(abs(oracle_val), 1e-8) * 100),
                **{f"pct_{k}": v for k, v in action_dist.items()},
            })

    return pd.DataFrame(segments)


# ============================================================
# STRESS TESTS
# ============================================================

def run_stress_tests(
    df: pd.DataFrame,
    predict_fn,
    base_config: Dict,
    survival_probs: Dict[int, Dict[float, float]] = None,
) -> pd.DataFrame:
    """
    Run financial sensitivity analysis under different cost/timing assumptions.
    """
    results = []

    # 1. Cost sensitivity
    for cost_multiplier in [0.1, 1.0, 5.0]:
        cfg = base_config.copy()
        cfg["action_costs"] = {
            a: c * cost_multiplier for a, c in base_config["action_costs"].items()
        }
        engine = DecisionEngine(cfg)
        dt = engine.build_decision_table(df, predict_fn, survival_probs)
        rt_actions = dt["recommended_action"].values
        dist = {ACTION_NAMES[a]: float(np.mean(rt_actions == a)) for a in range(4)}
        results.append({
            "test": "cost_sensitivity",
            "param": f"cost_multiplier={cost_multiplier}",
            "control_pct": dist.get("control", 0),
            "retry_pct": dist.get("retry", 0),
            "reminder_pct": dist.get("reminder", 0),
            "alternative_method_pct": dist.get("alternative_method", 0),
            "n_interventions": int(np.sum(rt_actions > 0)),
        })

    # 2. Fatigue tolerance
    for fatigue_max in [2.0, 4.0, 8.0]:
        cfg = base_config.copy()
        cfg["max_fatigue"] = fatigue_max
        engine = DecisionEngine(cfg)
        dt = engine.build_decision_table(df, predict_fn, survival_probs)
        rt_actions = dt["recommended_action"].values
        dist = {ACTION_NAMES[a]: float(np.mean(rt_actions == a)) for a in range(4)}
        results.append({
            "test": "fatigue_sensitivity",
            "param": f"max_fatigue={fatigue_max}",
            "control_pct": dist.get("control", 0),
            "retry_pct": dist.get("retry", 0),
            "reminder_pct": dist.get("reminder", 0),
            "alternative_method_pct": dist.get("alternative_method", 0),
            "n_interventions": int(np.sum(rt_actions > 0)),
        })

    # 3. Time discount sensitivity
    for half_life in [6.0, 24.0, 72.0]:
        cfg = base_config.copy()
        cfg["time_discount"] = {"enabled": True, "half_life_hours": half_life}
        engine = DecisionEngine(cfg)
        dt = engine.build_decision_table(df, predict_fn, survival_probs)
        rt_actions = dt["recommended_action"].values
        dist = {ACTION_NAMES[a]: float(np.mean(rt_actions == a)) for a in range(4)}
        results.append({
            "test": "time_discount_sensitivity",
            "param": f"half_life={half_life}h",
            "control_pct": dist.get("control", 0),
            "retry_pct": dist.get("retry", 0),
            "reminder_pct": dist.get("reminder", 0),
            "alternative_method_pct": dist.get("alternative_method", 0),
            "n_interventions": int(np.sum(rt_actions > 0)),
        })

    return pd.DataFrame(results)
