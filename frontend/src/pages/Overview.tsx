import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const fmt = (n: number) => `Rs.${(n / 100000).toFixed(1)}L`;
const fmtFull = (n: number) => `Rs.${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

const COLORS = ['#94a3b8', '#3b82f6', '#f59e0b', '#8b5cf6', '#10b981'];
const ACTION_NAMES: Record<string, string> = {
  control: 'Do Nothing', retry: 'Retry', reminder: 'Reminder', alternative_method: 'Alt Method'
};

export default function OverviewPage() {
  const { data: overview, isLoading } = useQuery({ queryKey: ['overview'], queryFn: api.getOverview });
  const { data: policies } = useQuery({ queryKey: ['policies'], queryFn: api.getPolicies });
  const { data: actions } = useQuery({ queryKey: ['actions'], queryFn: api.getActions });

  if (isLoading || !overview) {
    return <div className="empty-state"><div className="skeleton" style={{ width: '100%', height: 400 }} /></div>;
  }

  const policyData = (policies || []).map(p => ({
    name: p.name,
    value: p.net_revenue,
    isRT: p.name === 'Recoverytwin',
  }));

  const actionData = actions ? Object.entries(actions)
    .filter(([k]) => k !== 'overall_recovery_rate')
    .map(([k, v]) => ({ name: ACTION_NAMES[k] || k, value: v * 100 }))
    .filter(a => a.value > 0) : [];

  return (
    <>
      <div className="page-header">
        <h2>Revenue Recovery Command Center</h2>
        <p>Counterfactual decision intelligence for failed payment recovery</p>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="label">Revenue at Risk</div>
          <div className="value">{fmt(overview.at_risk_revenue)}</div>
          <div className="sub">{overview.total_payments.toLocaleString()} failed payments</div>
        </div>
        <div className="metric-card primary">
          <div className="label">Expected Recovery</div>
          <div className="value">{fmt(overview.expected_recovery)}</div>
          <div className="sub">RecoveryTwin policy</div>
        </div>
        <div className="metric-card success">
          <div className="label">Incremental Recovery</div>
          <div className="value">+{fmt(overview.incremental_recovery)}</div>
          <div className="sub">vs. doing nothing</div>
        </div>
        <div className="metric-card">
          <div className="label">Policy Regret</div>
          <div className="value" style={{ color: overview.policy_regret < 0.5 ? 'var(--warning)' : 'var(--danger)' }}>
            {pct(overview.policy_regret)}
          </div>
          <div className="sub">vs. oracle (perfect info)</div>
        </div>
      </div>

      <div className="grid-2 mb-24">
        <div className="card">
          <div className="card-header"><h3>Policy Comparison</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={policyData} layout="vertical" margin={{ left: 20, right: 20 }}>
                <XAxis type="number" tickFormatter={(v: number) => fmt(v)} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: number) => fmtFull(v)} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {policyData.map((entry, i) => (
                    <Cell key={i} fill={entry.isRT ? '#2563eb' : COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3>Action Distribution</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={actionData} margin={{ left: 20, right: 20 }}>
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v: number) => `${v}%`} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Bar dataKey="value" fill="#2563eb" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Decision Coverage</h3>
        </div>
        <div className="card-body" style={{ display: 'flex', gap: 32 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Recovery Rate</div>
            <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{pct(overview.recovery_rate)}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Intervention Rate</div>
            <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{pct(overview.intervention_rate)}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Robustness (vs Do Nothing)</div>
            <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4, color: 'var(--success)' }}>{pct(overview.robustness_vs_baseline)}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Total Payments</div>
            <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{overview.total_payments.toLocaleString()}</div>
          </div>
        </div>
      </div>
    </>
  );
}
