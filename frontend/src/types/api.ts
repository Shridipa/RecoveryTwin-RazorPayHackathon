export interface Overview {
  total_payments: number;
  at_risk_revenue: number;
  expected_recovery: number;
  incremental_recovery: number;
  recovery_rate: number;
  intervention_rate: number;
  policy_regret: number;
  robustness_vs_baseline: number;
  robustness_vs_max_prob: number;
  oracle_revenue: number;
  do_nothing_revenue: number;
  max_probability_revenue: number;
}

export interface Policy {
  name: string;
  net_revenue: number;
  incremental_revenue: number;
  recovery_rate: number;
  n_interventions: number;
  total_cost: number;
}

export interface PaymentItem {
  payment_id: string;
  amount: number;
  failure_reason: string;
  customer_activity: number;
  customer_tenure: number;
  previous_recoveries: number;
  merchant_id: string;
  recommended_action: string;
  control_prob: number | null;
  retry_prob: number | null;
  reminder_prob: number | null;
  alternative_method_prob: number | null;
  recommended_value: number | null;
}

export interface PaymentListResponse {
  items: PaymentItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ActionDetail {
  action: string;
  action_id: number;
  probability: number;
  cost: number;
  net_value: number;
  eligible: boolean;
}

export interface Explanation {
  recommended_action: string;
  recovery_probability: number;
  control_probability: number;
  incremental_probability_pp: number;
  incremental_revenue: number;
  reason_codes: string[];
  amount: number;
}

export interface PaymentDetail {
  payment_id: string;
  amount: number;
  failure_reason: string;
  customer_activity: number;
  customer_tenure: number;
  previous_recoveries: number;
  previous_failures: number;
  attempt_count: number;
  merchant_id: string;
  payment_method: string;
  device: string;
  actions: ActionDetail[];
  recommended_action: string;
  recommended_value: number;
  retry_cate: number | null;
  reminder_cate: number | null;
  alternative_method_cate: number | null;
  explanation: Explanation;
}

export interface Decision {
  payment_id: string;
  amount: number;
  failure_reason: string;
  recommended_action: string;
  recommended_value: number;
  actions: ActionDetail[];
  explanation: Explanation;
  cate: Record<string, number | null>;
}

export interface ModelMetrics {
  predictive: Record<string, { pr_auc: number; roc_auc: number; brier_score: number }>;
  calibration: Record<string, { ece: number; brier_score: number }>;
  survival: { rsf_cindex: number; cox_cindex: number };
  causal: { best_model: string; ate_error: number; cate_correlation: number; policy_value: number; policy_regret: number };
  decision: { incremental_recovery: number; policy_regret_pct: number; overall_recovery_rate: number };
}

export interface ScenarioItem {
  name: string;
  description: string;
  recoverytwin_revenue: number;
  incremental: number;
  regret: number;
  beats_do_nothing: boolean;
}

export interface MonteCarloResult {
  mean_net_revenue: number;
  median_net_revenue: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  prob_positive_net: number;
}

export interface RobustnessData {
  robustness: {
    vs_baseline: number;
    vs_max_prob: number;
    n_scenarios: number;
    n_beats_baseline?: number;
    n_beats_max_prob?: number;
  };
  worst_case: {
    scenario: string;
    description: string;
    net_revenue: number;
    incremental_over_do_nothing: number;
    policy_regret: number;
    beats_do_nothing: boolean;
  };
}
