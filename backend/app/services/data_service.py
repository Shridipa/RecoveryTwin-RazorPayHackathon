"""
Data Service — loads all RecoveryTwin reports, data, and cached results.

When report/data files are missing (e.g. on Railway), generates realistic
demo data so the API always returns usable responses.
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.app.config import settings


def _demo_overview() -> Dict[str, Any]:
    """Realistic demo data matching actual RecoveryTwin benchmarks."""
    return {
        "total_payments": 8426,
        "at_risk_revenue": 115200000,
        "expected_recovery": 3822601,
        "incremental_recovery": 1999675,
        "recovery_rate": 0.386,
        "intervention_rate": 0.812,
        "policy_regret": 0.478,
        "robustness_vs_baseline": 1.0,
        "robustness_vs_max_prob": 0.33,
        "oracle_revenue": 7315627,
        "do_nothing_revenue": 1822926,
        "max_probability_revenue": 4214668,
    }


def _demo_policies() -> List[Dict[str, Any]]:
    return [
        {"name": "Do Nothing", "net_revenue": 1822926, "incremental_revenue": 0,
         "recovery_rate": 0.115, "n_interventions": 0, "total_cost": 0},
        {"name": "Always Retry", "net_revenue": 3217224, "incremental_revenue": 1394299,
         "recovery_rate": 0.317, "n_interventions": 8426, "total_cost": 4213},
        {"name": "Max Probability", "net_revenue": 4214668, "incremental_revenue": 2391742,
         "recovery_rate": 0.402, "n_interventions": 8418, "total_cost": 12637},
        {"name": "Recovery Twin", "net_revenue": 3822601, "incremental_revenue": 1999675,
         "recovery_rate": 0.386, "n_interventions": 6841, "total_cost": 10261},
        {"name": "Oracle", "net_revenue": 7315627, "incremental_revenue": 5492701,
         "recovery_rate": 0.612, "n_interventions": 8426, "total_cost": 0},
    ]


def _demo_actions() -> Dict[str, Any]:
    return {"control": 1679, "retry": 2847, "reminder": 2312, "alternative_method": 1588}


def _demo_payments(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate realistic payment data for demo mode."""
    rng = np.random.RandomState(seed)
    n = min(n, 200)
    payment_ids = [f"RZP_{hashlib.md5(str(i).encode()).hexdigest()[:8].upper()}" for i in range(n)]
    failure_reasons = rng.choice(
        ["technical_decline", "insufficient_funds", "card_expired",
         "network_timeout", "auth_failure", "bank_decline"], n,
        p=[0.30, 0.20, 0.15, 0.15, 0.12, 0.08]
    )
    amounts = rng.lognormal(mean=7.5, sigma=1.2, size=n).clip(50, 200000).round(2)
    customer_activity = rng.beta(2, 2, n).round(3)
    customer_tenure = rng.poisson(18, n).clip(1, 120)
    previous_recoveries = rng.poisson(1.5, n).clip(0, 10)
    merchant_ids = [f"M{rng.randint(1, 500):04d}" for _ in range(n)]

    # Generate counterfactual probs
    base = rng.beta(2, 5, n)
    retry_p = (base + rng.uniform(0.05, 0.35, n)).clip(0.01, 0.99).round(4)
    reminder_p = (base + rng.uniform(0.03, 0.25, n)).clip(0.01, 0.99).round(4)
    alt_p = (base + rng.uniform(0.02, 0.20, n)).clip(0.01, 0.99).round(4)

    best_ev = np.maximum(
        amounts * base,
        np.maximum(amounts * retry_p - 0.50, np.maximum(amounts * reminder_p - 1.0, amounts * alt_p - 2.5))
    )
    best_action = np.where(best_ev == amounts * base, "do_nothing",
                  np.where(best_ev == amounts * retry_p - 0.50, "retry",
                  np.where(best_ev == amounts * reminder_p - 1.0, "reminder", "alternative_method")))

    df = pd.DataFrame({
        "payment_id": payment_ids,
        "amount": amounts,
        "failure_reason": failure_reasons,
        "customer_activity": customer_activity,
        "customer_tenure": customer_tenure,
        "previous_recoveries": previous_recoveries,
        "previous_failures": rng.poisson(2, n).clip(0, 15),
        "attempt_count": rng.poisson(1, n).clip(0, 5),
        "merchant_id": merchant_ids,
        "payment_method": rng.choice(["upi", "card", "netbanking", "wallet"], n),
        "device": rng.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1]),
        "control_prob": base.round(4),
        "retry_prob": retry_p,
        "reminder_prob": reminder_p,
        "alternative_method_prob": alt_p,
        "recommended_action_name": best_action,
        "recommended_value": best_ev.round(2),
        "control_expected_value": (amounts * base).round(2),
        "retry_expected_value": (amounts * retry_p - 0.50).round(2),
        "reminder_expected_value": (amounts * reminder_p - 1.0).round(2),
        "alternative_method_expected_value": (amounts * alt_p - 2.5).round(2),
        "retry_eligible": rng.choice([True, True, True, False], n),
        "reminder_eligible": rng.choice([True, True, True, True, False], n),
        "alternative_method_eligible": rng.choice([True, True, False], n),
    })
    return df


def _demo_scenarios() -> List[Dict[str, Any]]:
    return [
        {"name": "BASELINE", "description": "Current operating conditions",
         "recoverytwin_revenue": 3822601, "incremental": 1999675, "regret": 0.478, "beats_do_nothing": True},
        {"name": "HIGH_INTERVENTION_COST", "description": "Intervention costs +50%",
         "recoverytwin_revenue": 3751200, "incremental": 1928274, "regret": 0.485, "beats_do_nothing": True},
        {"name": "LOW_INTERVENTION_COST", "description": "Intervention costs -50%",
         "recoverytwin_revenue": 3894000, "incremental": 2071074, "regret": 0.471, "beats_do_nothing": True},
        {"name": "HIGH_FATIGUE", "description": "Customer fatigue threshold reduced to 3",
         "recoverytwin_revenue": 3650000, "incremental": 1827074, "regret": 0.495, "beats_do_nothing": True},
        {"name": "LOW_FATIGUE", "description": "No fatigue constraints",
         "recoverytwin_revenue": 3898000, "incremental": 2075074, "regret": 0.470, "beats_do_nothing": True},
        {"name": "SLOW_RECOVERY", "description": "Recovery time +50%",
         "recoverytwin_revenue": 3701000, "incremental": 1878074, "regret": 0.491, "beats_do_nothing": True},
        {"name": "FAST_RECOVERY", "description": "Recovery time -30%",
         "recoverytwin_revenue": 3912000, "incremental": 2089074, "regret": 0.468, "beats_do_nothing": True},
        {"name": "TREATMENT_DEGRADATION_20", "description": "Treatment effectiveness -20%",
         "recoverytwin_revenue": 3580000, "incremental": 1757074, "regret": 0.502, "beats_do_nothing": True},
        {"name": "TREATMENT_DEGRADATION_40", "description": "Treatment effectiveness -40%",
         "recoverytwin_revenue": 3340000, "incremental": 1517074, "regret": 0.528, "beats_do_nothing": True},
        {"name": "HIGH_PAYMENT_VALUE", "description": "Payment amounts +30%",
         "recoverytwin_revenue": 4969000, "incremental": 2599577, "regret": 0.478, "beats_do_nothing": True},
        {"name": "LOW_PAYMENT_VALUE", "description": "Payment amounts -30%",
         "recoverytwin_revenue": 2676000, "incremental": 1399872, "regret": 0.478, "beats_do_nothing": True},
        {"name": "ADVERSE_COMBINED", "description": "Multiple adverse conditions simultaneously",
         "recoverytwin_revenue": 2850000, "incremental": 1027074, "regret": 0.560, "beats_do_nothing": True},
    ]


def _demo_monte_carlo() -> Dict[str, Any]:
    rng = np.random.RandomState(42)
    def _mc(mean, std):
        samples = rng.normal(mean, std, 1000)
        return {
            "mean_net_revenue": round(float(np.mean(samples)), 0),
            "median_net_revenue": round(float(np.median(samples)), 0),
            "p5": round(float(np.percentile(samples, 5)), 0),
            "p25": round(float(np.percentile(samples, 25)), 0),
            "p75": round(float(np.percentile(samples, 75)), 0),
            "p95": round(float(np.percentile(samples, 95)), 0),
        }
    return {
        "recoverytwin": _mc(3822601, 300000),
        "do_nothing": _mc(1822926, 150000),
        "max_probability": _mc(4214668, 350000),
    }


def _demo_breakeven() -> List[Dict[str, Any]]:
    return [
        {"action": "retry", "breakeven_cost": 8.50},
        {"action": "reminder", "breakeven_cost": 5.20},
        {"action": "alternative_method", "breakeven_cost": 12.30},
    ]


def _demo_robustness() -> Dict[str, Any]:
    return {
        "robustness": {
            "vs_baseline": 1.0,
            "vs_max_prob": 0.33,
            "n_scenarios": 12,
        },
        "worst_case": {
            "scenario": "ADVERSE_COMBINED",
            "net_revenue": 2850000,
            "incremental": 1027074,
            "beats_do_nothing": True,
        },
    }


def _demo_model_metrics() -> Dict[str, Any]:
    return {
        "predictive": {
            "xgboost_p_y_xt": {"pr_auc": 0.516, "roc_auc": 0.709, "brier_score": 0.153},
            "lightgbm_p_y_xt": {"pr_auc": 0.515, "roc_auc": 0.711, "brier_score": 0.152},
            "random_forest_p_y_xt": {"pr_auc": 0.492, "roc_auc": 0.685, "brier_score": 0.161},
            "logistic_p_y_xt": {"pr_auc": 0.429, "roc_auc": 0.621, "brier_score": 0.178},
        },
        "calibration": {
            "none": {"ece": 0.0163, "brier_score": 0.153},
            "sigmoid": {"ece": 0.0139, "brier_score": 0.152},
            "isotonic": {"ece": 0.0163, "brier_score": 0.154},
        },
        "survival": {"rsf_cindex": 0.665, "cox_cindex": 0.628},
        "causal": {
            "best_model": "s_learner_xgboost",
            "ate_error": 0.0342,
            "cate_correlation": 0.124,
            "policy_value": -1.308,
            "policy_regret": 0.478,
        },
        "decision": {
            "incremental_recovery": 1999675,
            "policy_regret_pct": 47.8,
            "overall_recovery_rate": 0.386,
        },
    }


class DataService:
    """Singleton service that loads and caches all RecoveryTwin data."""

    def __init__(self):
        self._test_data: Optional[pd.DataFrame] = None
        self._train_data: Optional[pd.DataFrame] = None
        self._decision_table: Optional[pd.DataFrame] = None
        self._phase7_summary: Optional[Dict] = None
        self._phase8_report: Optional[Dict] = None
        self._phase3_report: Optional[Dict] = None
        self._phase5_summary: Optional[Dict] = None
        self._phase6_summary: Optional[Dict] = None
        self._loaded = False
        self._demo_mode = False

    def load(self):
        """Load all data. Called once at startup. Falls back to demo mode if files missing."""
        if self._loaded:
            return

        self._test_data = self._load_parquet(settings.TEST_DATA)
        self._train_data = self._load_parquet(settings.TRAIN_DATA)
        self._phase7_summary = self._load_json(settings.PHASE7_REPORT)
        self._phase8_report = self._load_json(settings.PHASE8_REPORT)
        self._phase3_report = self._load_json(settings.PHASE3_REPORT)
        self._phase5_summary = self._load_json(settings.PHASE5_SUMMARY)
        self._phase6_summary = self._load_json(settings.PHASE6_SUMMARY)

        if settings.PHASE7_PREDICTIONS.exists():
            self._decision_table = pd.read_parquet(settings.PHASE7_PREDICTIONS)

        # Check if we have enough data to operate normally
        has_data = self._test_data is not None and self._phase8_report is not None

        if not has_data:
            print("[DataService] Report/data files not found — entering DEMO MODE")
            self._demo_mode = True
            self._test_data = _demo_payments()
        else:
            self._demo_mode = False

        self._loaded = True

    @property
    def demo_mode(self) -> bool:
        return self._demo_mode

    def _load_parquet(self, path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _load_json(self, path: Path) -> Optional[Dict]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @property
    def test_data(self) -> pd.DataFrame:
        return self._test_data

    @property
    def train_data(self) -> pd.DataFrame:
        return self._train_data

    @property
    def decision_table(self) -> pd.DataFrame:
        return self._decision_table

    @property
    def phase7(self) -> Dict:
        return self._phase7_summary or {}

    @property
    def phase8(self) -> Dict:
        return self._phase8_report or {}

    @property
    def phase3(self) -> Dict:
        return self._phase3_report or {}

    @property
    def phase5(self) -> Dict:
        return self._phase5_summary or {}

    @property
    def phase6(self) -> Dict:
        return self._phase6_summary or {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Overview metrics ──

    def get_overview(self) -> Dict[str, Any]:
        if self._demo_mode:
            return _demo_overview()

        p8 = self._phase8_report or {}
        p7 = self._phase7_summary or {}
        bp = p8.get("baseline_policies", {})
        rob = p8.get("robustness", {})
        test = self._test_data
        n_test = len(test) if test is not None else 0

        rt = bp.get("recoverytwin", {})
        dn = bp.get("do_nothing", {})
        oracle = bp.get("oracle", {})
        at_risk = float(test["amount"].sum()) if test is not None else 0

        return {
            "total_payments": n_test,
            "at_risk_revenue": round(at_risk, 0),
            "expected_recovery": round(rt.get("net_revenue", 0), 0),
            "incremental_recovery": round(rt.get("net_revenue", 0) - dn.get("net_revenue", 0), 0),
            "recovery_rate": round(rt.get("recovery_rate", 0), 4),
            "intervention_rate": round(rt.get("n_interventions", 0) / max(n_test, 1), 4),
            "policy_regret": round(p7.get("policy_regret_pct", 0) / 100, 4),
            "robustness_vs_baseline": rob.get("vs_baseline", 0),
            "robustness_vs_max_prob": rob.get("vs_max_prob", 0),
            "oracle_revenue": round(oracle.get("net_revenue", 0), 0),
            "do_nothing_revenue": round(dn.get("net_revenue", 0), 0),
            "max_probability_revenue": round(bp.get("max_probability", {}).get("net_revenue", 0), 0),
        }

    # ── Policies ──

    def get_policies(self) -> List[Dict[str, Any]]:
        if self._demo_mode:
            return _demo_policies()

        p8 = self._phase8_report or {}
        bp = p8.get("baseline_policies", {})
        dn_rev = bp.get("do_nothing", {}).get("net_revenue", 0)

        policies = []
        for name, pol in bp.items():
            policies.append({
                "name": name.replace("_", " ").title(),
                "net_revenue": round(pol.get("net_revenue", 0), 0),
                "incremental_revenue": round(pol.get("net_revenue", 0) - dn_rev, 0),
                "recovery_rate": round(pol.get("recovery_rate", 0), 4),
                "n_interventions": pol.get("n_interventions", 0),
                "total_cost": round(pol.get("total_cost", 0), 0),
            })
        return policies

    # ── Action distribution ──

    def get_action_distribution(self) -> Dict[str, Any]:
        if self._demo_mode:
            return _demo_actions()
        p7 = self._phase7_summary or {}
        return p7.get("action_allocation", {})

    # ── Payments ──

    def get_payments(
        self,
        page: int = 1,
        page_size: int = 25,
        search: str = None,
        failure_reason: str = None,
        recommended_action: str = None,
        min_amount: float = None,
        max_amount: float = None,
    ) -> Dict[str, Any]:
        if self._test_data is None:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        df = self._test_data.copy()
        dt = self._decision_table

        if dt is not None:
            merge_cols = ["payment_id"]
            extra_cols = [c for c in dt.columns if c not in df.columns and c != "payment_id"]
            if extra_cols:
                df = df.merge(dt[merge_cols + extra_cols], on="payment_id", how="left")

        if search:
            mask = df["payment_id"].str.contains(search, case=False, na=False)
            df = df[mask]
        if failure_reason:
            df = df[df["failure_reason"] == failure_reason]
        if recommended_action and "recommended_action_name" in df.columns:
            df = df[df["recommended_action_name"].str.lower() == recommended_action.lower()]
        if min_amount is not None:
            df = df[df["amount"] >= min_amount]
        if max_amount is not None:
            df = df[df["amount"] <= max_amount]

        total = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]

        items = []
        for _, row in page_df.iterrows():
            item = {
                "payment_id": row.get("payment_id", ""),
                "amount": round(float(row.get("amount", 0)), 2),
                "failure_reason": row.get("failure_reason", "unknown"),
                "customer_activity": round(float(row.get("customer_activity", 0)), 3),
                "customer_tenure": int(row.get("customer_tenure", 0)),
                "previous_recoveries": int(row.get("previous_recoveries", 0)),
                "merchant_id": row.get("merchant_id", ""),
                "recommended_action": row.get("recommended_action_name", "unknown"),
                "control_prob": round(float(row.get("control_prob", 0)), 4) if "control_prob" in row.index else None,
                "retry_prob": round(float(row.get("retry_prob", 0)), 4) if "retry_prob" in row.index else None,
                "reminder_prob": round(float(row.get("reminder_prob", 0)), 4) if "reminder_prob" in row.index else None,
                "alternative_method_prob": round(float(row.get("alternative_method_prob", 0)), 4) if "alternative_method_prob" in row.index else None,
                "recommended_value": round(float(row.get("recommended_value", 0)), 2) if "recommended_value" in row.index else None,
            }
            items.append(item)

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }

    def get_payment(self, payment_id: str) -> Optional[Dict[str, Any]]:
        if self._test_data is None:
            return None

        df = self._test_data
        row_mask = df["payment_id"] == payment_id
        if not row_mask.any():
            return None

        row = df[row_mask].iloc[0]
        result = {
            "payment_id": row["payment_id"],
            "amount": round(float(row["amount"]), 2),
            "failure_reason": row["failure_reason"],
            "customer_activity": round(float(row.get("customer_activity", 0)), 3),
            "customer_tenure": int(row.get("customer_tenure", 0)),
            "previous_recoveries": int(row.get("previous_recoveries", 0)),
            "previous_failures": int(row.get("previous_failures", 0)),
            "attempt_count": int(row.get("attempt_count", 0)),
            "merchant_id": row.get("merchant_id", ""),
            "payment_method": row.get("payment_method", ""),
            "device": row.get("device", ""),
        }

        # Use inline decision data if available (works in demo mode too)
        if "control_prob" in df.columns:
            result["actions"] = [
                {"action": "do_nothing", "action_id": 0,
                 "probability": round(float(row.get("control_prob", 0)), 4),
                 "cost": 0,
                 "net_value": round(float(row.get("control_expected_value", 0)), 2),
                 "eligible": True},
                {"action": "retry", "action_id": 1,
                 "probability": round(float(row.get("retry_prob", 0)), 4),
                 "cost": 0.50,
                 "net_value": round(float(row.get("retry_expected_value", 0)), 2),
                 "eligible": bool(row.get("retry_eligible", True))},
                {"action": "reminder", "action_id": 2,
                 "probability": round(float(row.get("reminder_prob", 0)), 4),
                 "cost": 1.00,
                 "net_value": round(float(row.get("reminder_expected_value", 0)), 2),
                 "eligible": bool(row.get("reminder_eligible", True))},
                {"action": "alternative_method", "action_id": 3,
                 "probability": round(float(row.get("alternative_method_prob", 0)), 4),
                 "cost": 2.50,
                 "net_value": round(float(row.get("alternative_method_expected_value", 0)), 2),
                 "eligible": bool(row.get("alternative_method_eligible", True))},
            ]
            result["recommended_action"] = row.get("recommended_action_name", "unknown")
            result["recommended_value"] = round(float(row.get("recommended_value", 0)), 2)
            result["explanation"] = self._generate_explanation_from_row(row)

        # Also check decision_table for richer data
        if self._decision_table is not None:
            dt_row_mask = self._decision_table["payment_id"] == payment_id
            if dt_row_mask.any():
                dt_row = self._decision_table[dt_row_mask].iloc[0]
                result["actions"] = [
                    {"action": "do_nothing", "action_id": 0,
                     "probability": round(float(dt_row.get("control_prob", 0)), 4),
                     "cost": 0,
                     "net_value": round(float(dt_row.get("control_expected_value", 0)), 2),
                     "eligible": True},
                    {"action": "retry", "action_id": 1,
                     "probability": round(float(dt_row.get("retry_prob", 0)), 4),
                     "cost": 0.50,
                     "net_value": round(float(dt_row.get("retry_expected_value", 0)), 2),
                     "eligible": bool(dt_row.get("retry_eligible", True))},
                    {"action": "reminder", "action_id": 2,
                     "probability": round(float(dt_row.get("reminder_prob", 0)), 4),
                     "cost": 1.00,
                     "net_value": round(float(dt_row.get("reminder_expected_value", 0)), 2),
                     "eligible": bool(dt_row.get("reminder_eligible", True))},
                    {"action": "alternative_method", "action_id": 3,
                     "probability": round(float(dt_row.get("alternative_method_prob", 0)), 4),
                     "cost": 2.50,
                     "net_value": round(float(dt_row.get("alternative_method_expected_value", 0)), 2),
                     "eligible": bool(dt_row.get("alternative_method_eligible", True))},
                ]
                result["recommended_action"] = dt_row.get("recommended_action_name", "unknown")
                result["recommended_value"] = round(float(dt_row.get("recommended_value", 0)), 2)
                result["explanation"] = self._generate_explanation(dt_row, row)

        return result

    def _generate_explanation_from_row(self, row) -> Dict[str, Any]:
        rec_action = row.get("recommended_action_name", "unknown")
        ctrl_prob = float(row.get("control_prob", 0))

        prob_map = {
            "retry": "retry_prob",
            "reminder": "reminder_prob",
            "alternative_method": "alternative_method_prob",
            "do_nothing": "control_prob",
        }
        rec_prob = float(row.get(prob_map.get(rec_action, "control_prob"), 0))
        incremental_pp = rec_prob - ctrl_prob
        amount = float(row.get("amount", 0))

        reason_codes = []
        if incremental_pp > 0.2:
            reason_codes.append("strong_treatment_response")
        if amount > 5000:
            reason_codes.append("high_payment_value")
        if float(row.get("customer_activity", 0)) > 0.6:
            reason_codes.append("active_customer")
        if int(row.get("previous_recoveries", 0)) > 0:
            reason_codes.append("prior_recovery_history")

        return {
            "recommended_action": rec_action,
            "recovery_probability": round(rec_prob, 4),
            "control_probability": round(ctrl_prob, 4),
            "incremental_probability_pp": round(incremental_pp, 4),
            "incremental_revenue": round(incremental_pp * amount, 2),
            "reason_codes": reason_codes,
            "amount": round(amount, 2),
        }

    def _generate_explanation(self, dt_row, raw_row) -> Dict[str, Any]:
        return self._generate_explanation_from_row(dt_row)

    # ── Analytics ──

    def get_model_metrics(self) -> Dict[str, Any]:
        if self._demo_mode:
            return _demo_model_metrics()

        result = {"predictive": {}, "calibration": {}, "survival": {}, "causal": {}, "decision": {}}

        p3 = self._phase3_report or {}
        for name, res in p3.items():
            if isinstance(res, dict) and "test_metrics" in res:
                tm = res["test_metrics"]
                result["predictive"][name] = {
                    "pr_auc": round(tm.get("pr_auc", 0), 4),
                    "roc_auc": round(tm.get("roc_auc", 0), 4),
                    "brier_score": round(tm.get("brier_score", 0), 4),
                }

        cal_path = settings.REPORTS_DIR / "phase3_5" / "calibration_report.json"
        if cal_path.exists():
            with open(cal_path, "r") as f:
                cal = json.load(f)
            for method, eval_r in cal.items():
                if isinstance(eval_r, dict) and "ece" in eval_r:
                    result["calibration"][method] = {
                        "ece": round(eval_r["ece"], 4),
                        "brier_score": round(eval_r.get("brier_score", 0), 4),
                    }

        p5 = self._phase5_summary or {}
        result["survival"] = {"rsf_cindex": p5.get("best_rsf_cindex", 0), "cox_cindex": p5.get("best_cox_cindex", 0)}

        p6 = self._phase6_summary or {}
        best = p6.get("best_model")
        models = p6.get("models", {})
        if best and best in models:
            m = models[best]
            result["causal"] = {
                "best_model": best,
                "ate_error": round(m.get("ate_error", 0), 4),
                "cate_correlation": round(m.get("cate_correlation", 0), 4),
                "policy_value": round(m.get("policy_value", 0), 4),
                "policy_regret": round(m.get("policy_regret", 0), 4),
            }

        p7 = self._phase7_summary or {}
        result["decision"] = {
            "incremental_recovery": round(p7.get("recoverytwin_incremental", 0), 0),
            "policy_regret_pct": round(p7.get("policy_regret_pct", 0), 1),
            "overall_recovery_rate": round(p7.get("overall_recovery_rate", 0), 4),
        }
        return result

    def get_scenarios(self) -> List[Dict[str, Any]]:
        if self._demo_mode:
            return _demo_scenarios()

        p8 = self._phase8_report or {}
        scenarios = p8.get("scenario_results", [])
        return [
            {
                "name": s.get("scenario", ""),
                "description": s.get("description", ""),
                "recoverytwin_revenue": round(s["policies"].get("recoverytwin", {}).get("net_revenue", 0), 0),
                "incremental": round(s.get("recoverytwin_incremental", 0), 0),
                "regret": round(s.get("policy_regret", 0), 4),
                "beats_do_nothing": s.get("beats_do_nothing", False),
            }
            for s in scenarios
        ]

    def get_stress_test_results(self) -> Dict[str, Any]:
        if self._demo_mode:
            return {"cost_sensitivity": [], "degradation_sensitivity": [], "time_discount_sensitivity": []}

        p8 = self._phase8_report or {}

        def _clean(obj):
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        return _clean({
            "cost_sensitivity": p8.get("cost_sensitivity", []),
            "degradation_sensitivity": p8.get("degradation_sensitivity", []),
            "time_discount_sensitivity": p8.get("time_discount_sensitivity", []),
        })

    def get_monte_carlo(self) -> Dict[str, Any]:
        if self._demo_mode:
            return _demo_monte_carlo()

        p8 = self._phase8_report or {}
        mc = p8.get("monte_carlo", {})
        result = {}
        for pol_name, pol_mc in mc.items():
            if isinstance(pol_mc, dict) and "mean_net_revenue" in pol_mc:
                result[pol_name] = {k: v for k, v in pol_mc.items() if k != "raw"}
        return result

    def get_breakeven(self) -> List[Dict[str, Any]]:
        if self._demo_mode:
            return _demo_breakeven()
        p8 = self._phase8_report or {}
        return p8.get("breakeven_analysis", [])

    def get_robustness(self) -> Dict[str, Any]:
        if self._demo_mode:
            return _demo_robustness()
        p8 = self._phase8_report or {}
        return {"robustness": p8.get("robustness", {}), "worst_case": p8.get("worst_case", {})}

    def get_segments(self) -> List[Dict[str, Any]]:
        if self._demo_mode:
            return []
        p8 = self._phase8_report or {}
        return p8.get("segment_analysis", [])


# Global singleton
data_service = DataService()
