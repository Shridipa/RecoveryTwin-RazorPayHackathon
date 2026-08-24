import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

const fmt = (n: number) => `Rs.${(n / 100000).toFixed(1)}L`;
const fmtFull = (n: number) => `Rs.${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export default function FinancialPage() {
  const { data: mc, isLoading: mcLoading } = useQuery({ queryKey: ['monte-carlo'], queryFn: api.getMonteCarlo });
  const { data: be, isLoading: beLoading } = useQuery({ queryKey: ['breakeven'], queryFn: api.getBreakeven });
  const { data: rob, isLoading: robLoading } = useQuery({ queryKey: ['robustness'], queryFn: api.getRobustness });

  if (mcLoading || beLoading || robLoading) {
    return <div className="empty-state"><div className="skeleton" style={{ width: '100%', height: 400 }} /></div>;
  }

  const mcData = mc ? Object.entries(mc).map(([name, v]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    mean: v.mean_net_revenue,
    p5: v.p5,
    p95: v.p95,
    p_positive: v.prob_positive_net,
  })) : [];

  const rtMC = mc?.recoverytwin;

  return (
    <>
      <div className="page-header">
        <h2>Financial Analysis</h2>
        <p>Monte Carlo simulation, break-even analysis, and policy robustness</p>
      </div>

      {/* Monte Carlo Summary */}
      {rtMC && (
        <div className="metrics-grid mb-24">
          <div className="metric-card primary">
            <div className="label">Expected Net Revenue</div>
            <div className="value">{fmt(rtMC.mean_net_revenue)}</div>
            <div className="sub">RecoveryTwin (500 simulations)</div>
          </div>
          <div className="metric-card success">
            <div className="label">P(Net Positive)</div>
            <div className="value">{(rtMC.prob_positive_net * 100).toFixed(0)}%</div>
            <div className="sub">Probability of positive outcome</div>
          </div>
          <div className="metric-card">
            <div className="label">P5 — P95 Range</div>
            <div className="value" style={{ fontSize: 18 }}>{fmt(rtMC.p5)} — {fmt(rtMC.p95)}</div>
            <div className="sub">90% confidence interval</div>
          </div>
          <div className="metric-card">
            <div className="label">Median</div>
            <div className="value">{fmt(rtMC.median_net_revenue)}</div>
            <div className="sub">50th percentile</div>
          </div>
        </div>
      )}

      <div className="grid-2 mb-24">
        {/* Monte Carlo Distribution */}
        <div className="card">
          <div className="card-header"><h3>Monte Carlo — Net Revenue Distribution</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={mcData} margin={{ left: 20, right: 20 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tickFormatter={(v: number) => fmt(v)} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => fmtFull(v)} />
                <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
                  {mcData.map((_, i) => (
                    <Cell key={i} fill={i === 2 ? '#2563eb' : '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Break-Even */}
        <div className="card">
          <div className="card-header"><h3>Break-Even Analysis</h3></div>
          <div className="card-body">
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Maximum intervention cost before RecoveryTwin loses to doing nothing
            </p>
            {be && be.map(b => (
              <div key={b.action} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 0', borderBottom: '1px solid var(--border-light)' }}>
                <span style={{ fontWeight: 600, textTransform: 'capitalize' }}>
                  {b.action === 1 ? 'Retry' : b.action === 2 ? 'Reminder' : 'Alternative Method'}
                </span>
                <span style={{ fontWeight: 700, color: 'var(--accent)' }}>Rs.{b.breakeven_cost.toFixed(2)}</span>
              </div>
            ))}
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 12 }}>
              Current costs: Retry Rs.0.50, Reminder Rs.1.00, Alternative Rs.2.50
            </p>
          </div>
        </div>
      </div>

      {/* Robustness */}
      {rob && (
        <div className="grid-2 mb-24">
          <div className="card">
            <div className="card-header"><h3>Policy Robustness</h3></div>
            <div className="card-body">
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
                  RecoveryTwin beats Do Nothing
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ flex: 1, height: 8, background: 'var(--border-light)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ width: `${(rob.robustness.vs_baseline || 0) * 100}%`, height: '100%', background: 'var(--success)', borderRadius: 4 }} />
                  </div>
                  <span style={{ fontWeight: 700, fontSize: 16 }}>{((rob.robustness.vs_baseline || 0) * 100).toFixed(0)}%</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                  {rob.robustness.n_beats_baseline || 0}/{rob.robustness.n_scenarios || 0} scenarios
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
                  RecoveryTwin beats Max Probability
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ flex: 1, height: 8, background: 'var(--border-light)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ width: `${(rob.robustness.vs_max_prob || 0) * 100}%`, height: '100%', background: 'var(--warning)', borderRadius: 4 }} />
                  </div>
                  <span style={{ fontWeight: 700, fontSize: 16 }}>{((rob.robustness.vs_max_prob || 0) * 100).toFixed(0)}%</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                  {rob.robustness.n_beats_max_prob || 0}/{rob.robustness.n_scenarios || 0} scenarios
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header"><h3>Worst Case Analysis</h3></div>
            <div className="card-body">
              {rob.worst_case && (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>Scenario</div>
                    <div style={{ fontWeight: 700, fontSize: 16 }}>{rob.worst_case.scenario?.replace(/_/g, ' ') || '—'}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{rob.worst_case.description}</div>
                  </div>
                  <table>
                    <tbody>
                      <tr>
                        <td style={{ color: 'var(--text-secondary)' }}>Net Revenue</td>
                        <td style={{ fontWeight: 700 }}>{fmtFull(rob.worst_case.net_revenue || 0)}</td>
                      </tr>
                      <tr>
                        <td style={{ color: 'var(--text-secondary)' }}>Incremental vs Do Nothing</td>
                        <td style={{ fontWeight: 700, color: (rob.worst_case.incremental_over_do_nothing || 0) > 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {rob.worst_case.incremental_over_do_nothing > 0 ? '+' : ''}{fmtFull(rob.worst_case.incremental_over_do_nothing || 0)}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ color: 'var(--text-secondary)' }}>Policy Regret</td>
                        <td>{((rob.worst_case.policy_regret || 0) * 100).toFixed(1)}%</td>
                      </tr>
                      <tr>
                        <td style={{ color: 'var(--text-secondary)' }}>Still Profitable?</td>
                        <td>
                          <span className={`badge ${rob.worst_case.beats_do_nothing ? 'badge-success' : 'badge-danger'}`}>
                            {rob.worst_case.beats_do_nothing ? 'Yes' : 'No'}
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
