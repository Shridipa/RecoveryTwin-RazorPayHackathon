"""Pydantic response models."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: bool
    timestamp: str


class OverviewResponse(BaseModel):
    total_payments: int
    at_risk_revenue: float
    expected_recovery: float
    incremental_recovery: float
    recovery_rate: float
    intervention_rate: float
    policy_regret: float
    robustness_vs_baseline: float
    robustness_vs_max_prob: float
    oracle_revenue: float
    do_nothing_revenue: float
    max_probability_revenue: float


class PolicyResponse(BaseModel):
    name: str
    net_revenue: float
    incremental_revenue: float
    recovery_rate: float
    n_interventions: int
    total_cost: float


class ActionDistribution(BaseModel):
    control: Optional[float] = None
    retry: Optional[float] = None
    reminder: Optional[float] = None
    alternative_method: Optional[float] = None


class PaymentItem(BaseModel):
    payment_id: str
    amount: float
    failure_reason: str
    customer_activity: float
    customer_tenure: int
    previous_recoveries: int
    merchant_id: str
    recommended_action: str
    control_prob: Optional[float] = None
    retry_prob: Optional[float] = None
    reminder_prob: Optional[float] = None
    alternative_method_prob: Optional[float] = None
    recommended_value: Optional[float] = None


class PaymentListResponse(BaseModel):
    items: List[PaymentItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class ActionDetail(BaseModel):
    action: str
    action_id: int
    probability: float
    cost: float
    net_value: float
    eligible: bool


class Explanation(BaseModel):
    recommended_action: str
    recovery_probability: float
    control_probability: float
    incremental_probability_pp: float
    incremental_revenue: float
    reason_codes: List[str]
    amount: float


class PaymentDetailResponse(BaseModel):
    payment_id: str
    amount: float
    failure_reason: str
    customer_activity: float
    customer_tenure: int
    previous_recoveries: int
    previous_failures: int
    attempt_count: int
    merchant_id: str
    payment_method: str
    device: str
    actions: Optional[List[ActionDetail]] = None
    recommended_action: Optional[str] = None
    recommended_value: Optional[float] = None
    retry_cate: Optional[float] = None
    reminder_cate: Optional[float] = None
    alternative_method_cate: Optional[float] = None
    explanation: Optional[Explanation] = None


class ModelMetric(BaseModel):
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None
    ece: Optional[float] = None


class ModelMetricsResponse(BaseModel):
    predictive: Dict[str, Any]
    calibration: Dict[str, Any]
    survival: Dict[str, Any]
    causal: Dict[str, Any]
    decision: Dict[str, Any]


class ScenarioItem(BaseModel):
    name: str
    description: str
    recoverytwin_revenue: float
    incremental: float
    regret: float
    beats_do_nothing: bool


class MonteCarloResult(BaseModel):
    mean_net_revenue: float
    median_net_revenue: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    prob_positive_net: float


class RobustnessResponse(BaseModel):
    robustness: Dict[str, Any]
    worst_case: Dict[str, Any]


class SegmentItem(BaseModel):
    segment_column: str
    segment_value: str
    n: int
    do_nothing: float
    recoverytwin: float
    max_probability: float
    oracle: float
    rt_incremental: float
    regret: float
    recovery_rate_rt: float
    intervention_rate_rt: float
