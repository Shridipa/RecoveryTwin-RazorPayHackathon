const BASE = import.meta.env.VITE_API_URL || '/api';

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getHealth: () => fetchJSON<{ status: string; models_loaded: boolean; timestamp: string }>('/health'),
  getOverview: () => fetchJSON<import('../types/api').Overview>('/overview'),
  getPolicies: () => fetchJSON<import('../types/api').Policy[]>('/policies'),
  getActions: () => fetchJSON<Record<string, number>>('/actions'),

  getPayments: (params: Record<string, string | number> = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== '') qs.set(k, String(v)); });
    return fetchJSON<import('../types/api').PaymentListResponse>(`/payments?${qs}`);
  },
  getPayment: (id: string) => fetchJSON<import('../types/api').PaymentDetail>(`/payments/${id}`),
  getDecision: (id: string) => fetchJSON<import('../types/api').Decision>(`/decisions/${id}`),

  getModelMetrics: () => fetchJSON<import('../types/api').ModelMetrics>('/analytics/models'),
  getSegments: () => fetchJSON<unknown[]>('/analytics/segments'),
  getStressTests: () => fetchJSON<unknown>('/analytics/stress-tests'),

  getScenarios: () => fetchJSON<import('../types/api').ScenarioItem[]>('/scenarios'),
  getMonteCarlo: () => fetchJSON<Record<string, import('../types/api').MonteCarloResult>>('/financial/monte-carlo'),
  getBreakeven: () => fetchJSON<{ action: number; breakeven_cost: number }[]>('/financial/breakeven'),
  getRobustness: () => fetchJSON<import('../types/api').RobustnessData>('/financial/robustness'),
};
