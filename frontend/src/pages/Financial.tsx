import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { inr, inrFull } from '../utils/format';
import { Loading, ErrorState } from '../components/States';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function FinancialPage() {
  const { data: mc, isLoading: mcL, error: mcE, refetch: mcR } = useQuery({ queryKey: ['monte-carlo'], queryFn: api.getMonteCarlo });
  const { data: be, isLoading: beL } = useQuery({ queryKey: ['breakeven'], queryFn: api.getBreakeven });
  const { data: rob, isLoading: robL, error: robE, refetch: robR } = useQuery({ queryKey: ['robustness'], queryFn: api.getRobustness });

  if (mcL || beL || robL) return <Loading text="Loading financial analysis..." />;
  if (mcE || robE) return <ErrorState message="Unable to load financial data." onRetry={() => { mcR(); robR(); }} />;

  const rtMC = mc?.recoverytwin;
  const mcData = mc ? Object.entries(mc).map(([k, v]) => ({
    name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()), value: v.mean_net_revenue,
  })) : [];

  return (
    <>
      <div className="page-header">
        <h2>Financial Impact</h2>
        <p>Is the recovery policy profitable under real-world conditions?</p>
      </div>

      {rtMC && (
        <div className="metrics-row mb-20">
          <div className="metric-card accent">
            <div className="label">Expected Net Revenue</div>
            <div className="value">{inr(rtMC.mean_net_revenue)}</div>
            <div className="sub">Based on 500 financial simulations</div>
          </div>
          <div className="metric-card success">
            <div className="label">Probability of Positive Outcome</div>
            <div className="value">{(rtMC.prob_positive_net * 100).toFixed(0)}%</div>
          </div>
          <div className="metric-card">
            <div className="label">90% Confidence Range</div>
            <div className="value" style={{ fontSize: 18 }}>{inr(rtMC.p5)} — {inr(rtMC.p95)}</div>
            <div className="sub">P5 to P95 outcomes</div>
          </div>
          <div className="metric-card">
            <div className="label">Median Outcome</div>
            <div className="value">{inr(rtMC.median_net_revenue)}</div>
          </div>
        </div>
      )}

      <div className="grid-2 mb-20">
        {/* Monte Carlo */}
        <div className="card">
          <div className="card-header"><h3>Financial Risk Simulation</h3></div>
          <div className="card-body">
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Distribution of net recovery across 500 simulated scenarios. Higher bars mean more likely outcomes.
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={mcData} margin={{ left: 10, right: 10 }}>
                <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-15} textAnchor="end" height={50} />
                <YAxis tickFormatter={(v: number) => inr(v)} tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v: number) => inrFull(v)} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {mcData.map((_, i) => <Cell key={i} fill={i === 2 ? '#2563eb' : '#cbd5e1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Break-Even */}
        <div className="card">
          <div className="card-header"><h3>Break-Even Analysis</h3></div>
          <div className="card-body">
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14 }}>
              Maximum intervention cost before RecoveryTwin stops being more profitable than doing nothing.
            </p>
            {be && be.map(b => (
              <div key={b.action} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--border-light)' }}>
                <span style={{ fontWeight: 600 }}>{b.action === 1 ? 'Retry' : b.action === 2 ? 'Reminder' : 'Alt Method'}</span>
                <span style={{ fontWeight: 700, color: 'var(--accent)' }}>Rs.{b.breakeven_cost.toFixed(2)}</span>
              </div>
            ))}
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 12 }}>
              Current costs: Retry Rs.0.50, Reminder Rs.1.00, Alt Method Rs.2.50 — well below break-even.
            </p>
          </div>
        </div>
      </div>

      {/* Robustness */}
      {rob && (
        <div className="grid-2">
          <div className="card">
            <div className="card-header"><h3>Policy Robustness</h3></div>
            <div className="card-body">
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
                How often does RecoveryTwin outperform simpler strategies across all tested conditions?
              </p>
              <div style={{ marginBottom: 18 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6 }}>RecoveryTwin vs. Doing Nothing</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ flex: 1, height: 10, background: 'var(--border-light)', borderRadius: 5, overflow: 'hidden' }}>
                    <div style={{ width: `${(rob.robustness.vs_baseline || 0) * 100}%`, height: '100%', background: 'var(--success)', borderRadius: 5 }} />
                  </div>
                  <span style={{ fontWeight: 700, fontSize: 18 }}>{((rob.robustness.vs_baseline || 0) * 100).toFixed(0)}%</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{rob.robustness.n_beats_baseline || 0}/{rob.robustness.n_scenarios || 0} scenarios</div>
              </div>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6 }}>RecoveryTwin vs. Max Probability</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ flex: 1, height: 10, background: 'var(--border-light)', borderRadius: 5, overflow: 'hidden' }}>
                    <div style={{ width: `${(rob.robustness.vs_max_prob || 0) * 100}%`, height: '100%', background: 'var(--warning)', borderRadius: 5 }} />
                  </div>
                  <span style={{ fontWeight: 700, fontSize: 18 }}>{((rob.robustness.vs_max_prob || 0) * 100).toFixed(0)}%</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{rob.robustness.n_beats_max_prob || 0}/{rob.robustness.n_scenarios || 0} scenarios</div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header"><h3>Worst Case</h3></div>
            <div className="card-body">
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14 }}>
                The scenario where RecoveryTwin performs the poorest.
              </p>
              {rob.worst_case && (
                <table>
                  <tbody>
                    <tr><td style={{ color: 'var(--text-secondary)' }}>Scenario</td><td style={{ fontWeight: 700 }}>{(rob.worst_case.scenario || '').replace(/_/g, ' ')}</td></tr>
                    <tr><td style={{ color: 'var(--text-secondary)' }}>Net Revenue</td><td style={{ fontWeight: 700 }}>{inr(rob.worst_case.net_revenue || 0)}</td></tr>
                    <tr><td style={{ color: 'var(--text-secondary)' }}>vs. Doing Nothing</td>
                      <td style={{ fontWeight: 700, color: (rob.worst_case.incremental_over_do_nothing || 0) > 0 ? 'var(--success)' : 'var(--danger)' }}>
                        {(rob.worst_case.incremental_over_do_nothing || 0) > 0 ? '+' : ''}{inr(rob.worst_case.incremental_over_do_nothing || 0)}
                      </td>
                    </tr>
                    <tr><td style={{ color: 'var(--text-secondary)' }}>Still Profitable?</td>
                      <td><span className={`badge ${rob.worst_case.beats_do_nothing ? 'badge-success' : 'badge-danger'}`}>{rob.worst_case.beats_do_nothing ? 'Yes' : 'No'}</span></td>
                    </tr>
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
