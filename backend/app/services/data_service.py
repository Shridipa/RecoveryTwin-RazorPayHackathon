"""
Data Service — loads all RecoveryTwin reports, data, and cached results.

This is the single source of truth for the API layer.
All data comes from existing RecoveryTwin outputs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.app.config import settings


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

    def load(self):
        """Load all data. Called once at startup."""
        if self._loaded:
            return

        self._test_data = self._load_parquet(settings.TEST_DATA)
        self._train_data = self._load_parquet(settings.TRAIN_DATA)
        self._phase7_summary = self._load_json(settings.PHASE7_REPORT)
        self._phase8_report = self._load_json(settings.PHASE8_REPORT)
        self._phase3_report = self._load_json(settings.PHASE3_REPORT)
        self._phase5_summary = self._load_json(settings.PHASE5_SUMMARY)
        self._phase6_summary = self._load_json(settings.PHASE6_SUMMARY)

        # Load decision table (counterfactual predictions)
        if settings.PHASE7_PREDICTIONS.exists():
            self._decision_table = pd.read_parquet(settings.PHASE7_PREDICTIONS)

        self._loaded = True

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
        """Compute overview metrics from existing reports."""
        if not self._loaded:
            return {}

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
        """Get all policy comparison data."""
        if not self._loaded:
            return []

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
        """Get recommended action distribution from Phase 7."""
        p7 = self._phase7_summary or {}
        alloc = p7.get("action_allocation", {})
        return alloc

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
        """Get paginated payment list with decision data."""
        if self._test_data is None:
            return {"items": [], "total": 0, "page": page, "page_size": page_size}

        df = self._test_data.copy()
        dt = self._decision_table

        # Merge decision data if available
        if dt is not None:
            merge_cols = ["payment_id"]
            extra_cols = [c for c in dt.columns if c not in df.columns and c != "payment_id"]
            if extra_cols:
                df = df.merge(dt[merge_cols + extra_cols], on="payment_id", how="left")

        # Filters
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

        # Paginate
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
        """Get detailed information for a single payment."""
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

        # Add counterfactual predictions if available
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

                # CATE values
                for action_name in ["retry", "reminder", "alternative_method"]:
                    cate_col = f"{action_name}_cate"
                    if cate_col in dt_row.index:
                        result[f"{action_name}_cate"] = round(float(dt_row[cate_col]), 4)

                # Explanation
                result["explanation"] = self._generate_explanation(dt_row, row)

        return result

    def _generate_explanation(self, dt_row, raw_row) -> Dict[str, Any]:
        """Generate a structured explanation from measurable model outputs."""
        rec_action = dt_row.get("recommended_action_name", "unknown")
        rec_prob = 0
        ctrl_prob = float(dt_row.get("control_prob", 0))

        action_prob_map = {
            "retry": "retry_prob",
            "reminder": "reminder_prob",
            "alternative_method": "alternative_method_prob",
            "Do Nothing": "control_prob",
        }
        prob_col = action_prob_map.get(rec_action, "control_prob")
        rec_prob = float(dt_row.get(prob_col, 0))

        incremental_pp = rec_prob - ctrl_prob
        amount = float(raw_row.get("amount", 0))
        incremental_revenue = incremental_pp * amount

        reason_codes = []
        if incremental_pp > 0.2:
            reason_codes.append("strong_treatment_response")
        if amount > 5000:
            reason_codes.append("high_payment_value")
        if float(raw_row.get("customer_activity", 0)) > 0.6:
            reason_codes.append("active_customer")
        if int(raw_row.get("previous_recoveries", 0)) > 0:
            reason_codes.append("prior_recovery_history")

        return {
            "recommended_action": rec_action,
            "recovery_probability": round(rec_prob, 4),
            "control_probability": round(ctrl_prob, 4),
            "incremental_probability_pp": round(incremental_pp, 4),
            "incremental_revenue": round(incremental_revenue, 2),
            "reason_codes": reason_codes,
            "amount": round(amount, 2),
        }

    # ── Analytics ──

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get ML model evaluation metrics from existing reports."""
        result = {
            "predictive": {},
            "calibration": {},
            "survival": {},
            "causal": {},
            "decision": {},
        }

        # Phase 3 — predictive
        p3 = self._phase3_report or {}
        for name, res in p3.items():
            if isinstance(res, dict) and "test_metrics" in res:
                tm = res["test_metrics"]
                result["predictive"][name] = {
                    "pr_auc": round(tm.get("pr_auc", 0), 4),
                    "roc_auc": round(tm.get("roc_auc", 0), 4),
                    "brier_score": round(tm.get("brier_score", 0), 4),
                }

        # Phase 3.5 — calibration
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

        # Phase 5 — survival
        p5 = self._phase5_summary or {}
        result["survival"] = {
            "rsf_cindex": p5.get("best_rsf_cindex", 0),
            "cox_cindex": p5.get("best_cox_cindex", 0),
        }

        # Phase 6 — causal
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

        # Phase 7 — decision
        p7 = self._phase7_summary or {}
        result["decision"] = {
            "incremental_recovery": round(p7.get("recoverytwin_incremental", 0), 0),
            "policy_regret_pct": round(p7.get("policy_regret_pct", 0), 1),
            "overall_recovery_rate": round(p7.get("overall_recovery_rate", 0), 4),
        }

        return result

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Get available scenarios from Phase 8."""
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
        """Get stress test results from Phase 8."""
        p8 = self._phase8_report or {}

        def _clean(obj):
            """Replace NaN/inf with None for JSON serialization."""
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
        """Get Monte Carlo simulation results."""
        p8 = self._phase8_report or {}
        mc = p8.get("monte_carlo", {})
        # Remove raw data to keep response small
        result = {}
        for pol_name, pol_mc in mc.items():
            if isinstance(pol_mc, dict) and "mean_net_revenue" in pol_mc:
                result[pol_name] = {k: v for k, v in pol_mc.items() if k != "raw"}
        return result

    def get_breakeven(self) -> List[Dict[str, Any]]:
        """Get break-even analysis."""
        p8 = self._phase8_report or {}
        return p8.get("breakeven_analysis", [])

    def get_robustness(self) -> Dict[str, Any]:
        """Get robustness and worst-case results."""
        p8 = self._phase8_report or {}
        return {
            "robustness": p8.get("robustness", {}),
            "worst_case": p8.get("worst_case", {}),
        }

    def get_segments(self) -> List[Dict[str, Any]]:
        """Get segment analysis results."""
        p8 = self._phase8_report or {}
        return p8.get("segment_analysis", [])


# Global singleton
data_service = DataService()
