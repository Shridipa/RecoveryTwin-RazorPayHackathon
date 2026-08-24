import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const fmt = (n: number) => `Rs.${(n / 100000).toFixed(1)}L`;
const fmtFull = (n: number) => `Rs.${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export default function ScenariosPage() {
  const { data: scenarios, isLoading } = useQuery({ queryKey: ['scenarios'], queryFn: api.getScenarios });

  if (isLoading || !scenarios) {
    return <div className="empty-state"><div className="skeleton" style={{ width: '100%', height: 400 }} /></div>;
  }

  const baseline = scenarios.find(s => s.name === 'BASELINE');
  const chartData = scenarios.map(s => ({
    name: s.name.replace(/_/g, ' '),
    value: s.recoverytwin_revenue,
    isBaseline: s.name === 'BASELINE',
  }));

  return (
    <>
      <div className="page-header">
        <h2>Scenario Lab</h2>
        <p>Stress-test the recovery policy under different economic conditions</p>
      </div>

      <div className="metrics-grid mb-24">
        <div className="metric-card primary">
          <div className="label">Baseline Recovery</div>
          <div className="value">{fmt(baseline?.recoverytwin_revenue || 0)}</div>
          <div className="sub">Default parameters</div>
        </div>
        <div className="metric-card success">
          <div className="label">Scenarios Passing</div>
          <div className="value">{scenarios.filter(s => s.beats_do_nothing).length}/{scenarios.length}</div>
          <div className="sub">Beats doing nothing</div>
        </div>
        <div className="metric-card">
          <div className="label">Avg Policy Regret</div>
          <div className="value">{(scenarios.reduce((s, sc) => s + sc.regret, 0) / scenarios.length * 100).toFixed(0)}%</div>
          <div className="sub">Across all scenarios</div>
        </div>
        <div className="metric-card warning">
          <div className="label">Worst Case</div>
          <div className="value">{fmt(Math.min(...scenarios.map(s => s.recoverytwin_revenue)))}</div>
          <div className="sub">Minimum recovery</div>
        </div>
      </div>

      <div className="card mb-24">
        <div className="card-header"><h3>Scenario Comparison — RecoveryTwin Net Revenue</h3></div>
        <div className="card-body">
          <ResponsiveContainer width="100%" height={Math.max(300, scenarios.length * 32)}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 40 }}>
              <XAxis type="number" tickFormatter={(v: number) => fmt(v)} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => fmtFull(v)} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.isBaseline ? '#2563eb' : entry.value >= (baseline?.recoverytwin_revenue || 0) ? '#10b981' : '#f59e0b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-header"><h3>Scenario Details</h3></div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Description</th>
                <th>RT Revenue</th>
                <th>Incremental</th>
                <th>Regret</th>
                <th>Beats DN?</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map(s => (
                <tr key={s.name} style={s.name === 'BASELINE' ? { background: '#f8faff' } : {}}>
                  <td style={{ fontWeight: 600 }}>{s.name.replace(/_/g, ' ')}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{s.description}</td>
                  <td style={{ fontWeight: 600 }}>{fmtFull(s.recoverytwin_revenue)}</td>
                  <td style={{ color: s.incremental > 0 ? 'var(--success)' : 'var(--danger)' }}>
                    {s.incremental > 0 ? '+' : ''}{fmtFull(s.incremental)}
                  </td>
                  <td>{(s.regret * 100).toFixed(0)}%</td>
                  <td>
                    <span className={`badge ${s.beats_do_nothing ? 'badge-success' : 'badge-danger'}`}>
                      {s.beats_do_nothing ? 'Yes' : 'No'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
