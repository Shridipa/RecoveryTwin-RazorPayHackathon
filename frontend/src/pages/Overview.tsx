import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { inr, pct, actionLabel, failureLabel } from '../utils/format';
import { Loading, ErrorState } from '../components/States';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ACTION_COLORS: Record<string, string> = {
  'Do Nothing': '#94a3b8', Retry: '#2563eb', Reminder: '#f59e0b', 'Alt Method': '#8b5cf6', Alternative: '#8b5cf6',
};

export default function OverviewPage() {
  const { data: overview, isLoading, error, refetch } = useQuery({ queryKey: ['overview'], queryFn: api.getOverview });
  const { data: policies } = useQuery({ queryKey: ['policies'], queryFn: api.getPolicies });
  const { data: actions } = useQuery({ queryKey: ['actions'], queryFn: api.getActions });
  const { data: payments } = useQuery({ queryKey: ['payments', 1, 8], queryFn: () => api.getPayments({ page: 1, page_size: 8 }) });

  if (isLoading) return <Loading text="Loading recovery intelligence..." />;
  if (error || !overview) return <ErrorState message="Unable to connect to RecoveryTwin engine." onRetry={() => refetch()} />;

  const policyData = (policies || []).map(p => ({ name: p.name, value: p.net_revenue, isRT: p.name === 'Recoverytwin' }));
  const actionData = actions ? Object.entries(actions)
    .filter(([k]) => !['overall_recovery_rate', 'intervention_rate'].includes(k))
    .map(([k, v]) => ({ name: actionLabel(k), value: (v as number) * 100 }))
    .filter(a => a.value > 0.5) : [];

  return (
    <>
      <div className="page-header">
        <h2>Recovery Command Center</h2>
        <p>Monitor failed payments and recovery opportunities</p>
      </div>

      <div className="metrics-row">
        <div className="metric-card">
          <div className="label">Payments at Risk</div>
          <div className="value">{overview.total_payments.toLocaleString()}</div>
        </div>
        <div className="metric-card">
          <div className="label">Value at Risk</div>
          <div className="value">{inr(overview.at_risk_revenue)}</div>
        </div>
        <div className="metric-card accent">
          <div className="label">Expected Recovery</div>
          <div className="value">{inr(overview.expected_recovery)}</div>
          <div className="sub">RecoveryTwin policy</div>
        </div>
        <div className="metric-card success">
          <div className="label">Recovery Opportunity</div>
          <div className="value">+{inr(overview.incremental_recovery)}</div>
          <div className="sub">vs. doing nothing</div>
        </div>
      </div>

      <div className="grid-2 mb-20">
        <div className="card">
          <div className="card-header"><h3>Policy Comparison</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={policyData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tickFormatter={(v: number) => inr(v)} tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => inr(v)} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {policyData.map((e, i) => <Cell key={i} fill={e.isRT ? '#2563eb' : ['#94a3b8','#60a5fa','#fbbf24','#a78bfa','#34d399'][i] || '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3>Recommended Actions</h3></div>
          <div className="card-body">
            {actionData.map(a => (
              <div key={a.name} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <span style={{ width: 110, fontSize: 12, fontWeight: 600 }}>{a.name}</span>
                <div style={{ flex: 1, height: 8, background: 'var(--border-light)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${a.value}%`, height: '100%', background: ACTION_COLORS[a.name] || '#94a3b8', borderRadius: 4, transition: 'width 0.4s' }} />
                </div>
                <span style={{ width: 44, textAlign: 'right', fontSize: 13, fontWeight: 700 }}>{a.value.toFixed(0)}%</span>
              </div>
            ))}
            <div style={{ display: 'flex', gap: 24, marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border-light)' }}>
              <div>
                <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>Recovery Rate</div>
                <div style={{ fontSize: 20, fontWeight: 700, marginTop: 2 }}>{pct(overview.recovery_rate)}</div>
              </div>
              <div>
                <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>Intervention Rate</div>
                <div style={{ fontSize: 20, fontWeight: 700, marginTop: 2 }}>{pct(overview.intervention_rate)}</div>
              </div>
              <div>
                <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>Policy Robustness</div>
                <div style={{ fontSize: 20, fontWeight: 700, marginTop: 2, color: 'var(--success)' }}>{pct(overview.robustness_vs_baseline)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top Payments */}
      <div className="card">
        <div className="card-header">
          <h3>Payments Requiring Attention</h3>
          <a href="/app/payments" style={{ fontSize: 12, color: 'var(--accent)', textDecoration: 'none', fontWeight: 600 }}>View all</a>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Payment</th>
                <th>Amount</th>
                <th>Issue</th>
                <th>P(Retry)</th>
                <th>P(Reminder)</th>
                <th>Recommendation</th>
                <th>Expected Value</th>
              </tr>
            </thead>
            <tbody>
              {(payments?.items || []).map(p => (
                <tr key={p.payment_id} className="clickable" onClick={() => window.location.href = `/app/payments/${p.payment_id}`}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{p.payment_id}</td>
                  <td style={{ fontWeight: 600 }}>{inr(p.amount)}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{failureLabel(p.failure_reason)}</td>
                  <td>{p.retry_prob != null ? pct(p.retry_prob) : '—'}</td>
                  <td>{p.reminder_prob != null ? pct(p.reminder_prob) : '—'}</td>
                  <td><span className={`badge badge-${p.recommended_action?.toLowerCase() === 'retry' ? 'retry' : p.recommended_action?.toLowerCase()?.includes('reminder') ? 'reminder' : p.recommended_action?.toLowerCase()?.includes('alternative') ? 'alternative' : 'control'}`}>{p.recommended_action}</span></td>
                  <td style={{ fontWeight: 600 }}>{p.recommended_value != null ? inr(p.recommended_value) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
