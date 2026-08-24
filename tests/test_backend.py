"""
Backend API Tests.

Tests all FastAPI endpoints using the TestClient.
Does not require a running server.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.data_service import data_service


@pytest.fixture(scope="module")
def client():
    """Create test client with loaded data."""
    data_service.load()
    with TestClient(app) as c:
        yield c


# Blocked fields that must never appear in API responses
BLOCKED_FIELDS = {
    "recovered", "recovery_time_hours", "revenue_recovered",
    "intervention_cost", "potential_outcome_0", "potential_outcome_1",
    "potential_outcome_2", "potential_outcome_3", "true_best_intervention",
    "propensity_0", "propensity_1", "propensity_2", "propensity_3",
}


class TestHealth:
    def test_health_status(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert "RecoveryTwin" in data["service"]

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "RecoveryTwin API"


class TestOverview:
    def test_overview_fields(self, client):
        r = client.get("/api/overview")
        assert r.status_code == 200
        data = r.json()
        assert "total_payments" in data
        assert "expected_recovery" in data
        assert "incremental_recovery" in data
        assert data["total_payments"] > 0

    def test_overview_positive_recovery(self, client):
        r = client.get("/api/overview")
        data = r.json()
        assert data["expected_recovery"] > 0
        assert data["incremental_recovery"] > 0

    def test_policies(self, client):
        r = client.get("/api/policies")
        assert r.status_code == 200
        policies = r.json()
        assert len(policies) >= 4
        names = [p["name"] for p in policies]
        assert "Recoverytwin" in names
        assert "Do Nothing" in names

    def test_actions(self, client):
        r = client.get("/api/actions")
        assert r.status_code == 200


class TestPayments:
    def test_list_payments(self, client):
        r = client.get("/api/payments?page=1&page_size=5")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] > 0
        assert len(data["items"]) <= 5
        assert data["page"] == 1

    def test_payment_fields(self, client):
        r = client.get("/api/payments?page=1&page_size=1")
        item = r.json()["items"][0]
        assert "payment_id" in item
        assert "amount" in item
        assert "failure_reason" in item
        assert "recommended_action" in item

    def test_payment_filter_amount(self, client):
        r = client.get("/api/payments?min_amount=10000&page_size=5")
        data = r.json()
        for item in data["items"]:
            assert item["amount"] >= 10000

    def test_payment_detail(self, client):
        r = client.get("/api/payments?page=1&page_size=1")
        pid = r.json()["items"][0]["payment_id"]
        r2 = client.get(f"/api/payments/{pid}")
        assert r2.status_code == 200
        data = r2.json()
        assert "actions" in data
        assert len(data["actions"]) == 4
        assert "explanation" in data

    def test_payment_not_found(self, client):
        r = client.get("/api/payments/nonexistent_999")
        assert r.status_code == 404

    def test_counterfactual_actions(self, client):
        r = client.get("/api/payments?page=1&page_size=1")
        pid = r.json()["items"][0]["payment_id"]
        r2 = client.get(f"/api/payments/{pid}")
        actions = r2.json()["actions"]
        action_names = [a["action"] for a in actions]
        assert "do_nothing" in action_names
        assert "retry" in action_names
        assert "reminder" in action_names
        assert "alternative_method" in action_names


class TestDecisions:
    def test_decision_endpoint(self, client):
        r = client.get("/api/payments?page=1&page_size=1")
        pid = r.json()["items"][0]["payment_id"]
        r2 = client.get(f"/api/decisions/{pid}")
        assert r2.status_code == 200
        data = r2.json()
        assert "recommended_action" in data
        assert "explanation" in data
        assert "cate" in data

    def test_decision_not_found(self, client):
        r = client.get("/api/decisions/nonexistent_999")
        assert r.status_code == 404


class TestAnalytics:
    def test_model_metrics(self, client):
        r = client.get("/api/analytics/models")
        assert r.status_code == 200
        data = r.json()
        assert "predictive" in data
        assert "calibration" in data
        assert "survival" in data
        assert "causal" in data
        assert len(data["predictive"]) >= 2

    def test_segments(self, client):
        r = client.get("/api/analytics/segments")
        assert r.status_code == 200

    def test_stress_tests(self, client):
        r = client.get("/api/analytics/stress-tests")
        assert r.status_code == 200


class TestScenarios:
    def test_scenarios_list(self, client):
        r = client.get("/api/scenarios")
        assert r.status_code == 200
        scenarios = r.json()
        assert len(scenarios) >= 5
        names = [s["name"] for s in scenarios]
        assert "BASELINE" in names

    def test_monte_carlo(self, client):
        r = client.get("/api/financial/monte-carlo")
        assert r.status_code == 200
        data = r.json()
        assert "recoverytwin" in data
        rt = data["recoverytwin"]
        assert "mean_net_revenue" in rt
        assert rt["p5"] <= rt["p50"] <= rt["p95"]

    def test_breakeven(self, client):
        r = client.get("/api/financial/breakeven")
        assert r.status_code == 200
        assert len(r.json()) >= 2

    def test_robustness(self, client):
        r = client.get("/api/financial/robustness")
        assert r.status_code == 200
        data = r.json()
        assert "robustness" in data
        assert "worst_case" in data


class TestLeakage:
    def _check_no_leakage(self, response_body: str):
        for field in BLOCKED_FIELDS:
            assert field not in response_body, f"Leakage: {field} found in response"

    def test_overview_no_leakage(self, client):
        r = client.get("/api/overview")
        self._check_no_leakage(json.dumps(r.json()))

    def test_payments_no_leakage(self, client):
        r = client.get("/api/payments?page=1&page_size=5")
        self._check_no_leakage(json.dumps(r.json()))

    def test_payment_detail_no_leakage(self, client):
        r = client.get("/api/payments?page=1&page_size=1")
        pid = r.json()["items"][0]["payment_id"]
        r2 = client.get(f"/api/payments/{pid}")
        self._check_no_leakage(json.dumps(r2.json()))

    def test_decision_no_leakage(self, client):
        r = client.get("/api/payments?page=1&page_size=1")
        pid = r.json()["items"][0]["payment_id"]
        r2 = client.get(f"/api/decisions/{pid}")
        self._check_no_leakage(json.dumps(r2.json()))

    def test_scenarios_no_leakage(self, client):
        r = client.get("/api/scenarios")
        self._check_no_leakage(json.dumps(r.json()))

    def test_monte_carlo_no_leakage(self, client):
        r = client.get("/api/financial/monte-carlo")
        self._check_no_leakage(json.dumps(r.json()))

    def test_analytics_no_leakage(self, client):
        r = client.get("/api/analytics/models")
        self._check_no_leakage(json.dumps(r.json()))
