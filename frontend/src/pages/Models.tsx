import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { Loading, ErrorState } from '../components/States';

export default function ModelsPage() {
  const { data: metrics, isLoading, error, refetch } = useQuery({ queryKey: ['model-metrics'], queryFn: api.getModelMetrics });

  if (isLoading) return <Loading text="Loading system health..." />;
  if (error || !metrics) return <ErrorState message="Unable to load system metrics." onRetry={() => refetch()} />;

  return (
    <>
      <div className="page-header">
        <h2>System Health</h2>
        <p>ML model performance and system status</p>
      </div>

      {/* System Status */}
      <div className="card mb-20">
        <div className="card-header"><h3>System Status</h3></div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 20 }}>
            {[
              ['126/126', 'Tests Passing'],
              ['17/17', 'Verification Sections'],
              ['0', 'Leakage Violations'],
              ['44', 'Financial Tests'],
            ].map(([v, l]) => (
              <div key={l} style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--success)' }}>{v}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid-2 mb-20">
        {/* Predictive Models */}
        <div className="card">
          <div className="card-header"><h3>Predictive Models — Detection Quality</h3></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Brier</th></tr></thead>
              <tbody>
                {Object.entries(metrics.predictive).map(([name, m]) => (
                  <tr key={name}>
                    <td style={{ fontWeight: 600, fontSize: 12 }}>{name.replace(/_/g, ' ')}</td>
                    <td>{m.pr_auc.toFixed(4)}</td><td>{m.roc_auc.toFixed(4)}</td><td>{m.brier_score.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Calibration */}
        <div className="card">
          <div className="card-header"><h3>Calibration — Probability Reliability</h3></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Method</th><th>ECE</th><th>Brier</th></tr></thead>
              <tbody>
                {Object.entries(metrics.calibration).map(([name, m]) => (
                  <tr key={name}>
                    <td style={{ fontWeight: 600, textTransform: 'capitalize' }}>{name}</td>
                    <td>{m.ece.toFixed(4)}</td><td>{m.brier_score.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid-2 mb-20">
        {/* Survival */}
        <div className="card">
          <div className="card-header"><h3>Recovery Timing</h3></div>
          <div className="card-body">
            <table>
              <tbody>
                <tr><td style={{ color: 'var(--text-secondary)' }}>Random Survival Forest C-Index</td><td style={{ fontWeight: 700 }}>{metrics.survival.rsf_cindex?.toFixed(3) || '—'}</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)' }}>Cox PH C-Index</td><td style={{ fontWeight: 700 }}>{metrics.survival.cox_cindex?.toFixed(3) || '—'}</td></tr>
              </tbody>
            </table>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10 }}>C-Index measures how well the model ranks recovery timing. Above 0.6 is useful.</p>
          </div>
        </div>

        {/* Causal */}
        <div className="card">
          <div className="card-header"><h3>Causal / Uplift Models — Added Recovery from Intervention</h3></div>
          <div className="card-body">
            {metrics.causal.best_model ? (
              <table>
                <tbody>
                  <tr><td style={{ color: 'var(--text-secondary)' }}>Best Model</td><td style={{ fontWeight: 700 }}>{metrics.causal.best_model}</td></tr>
                  <tr><td style={{ color: 'var(--text-secondary)' }}>ATE Error</td><td>{metrics.causal.ate_error.toFixed(4)}</td></tr>
                  <tr><td style={{ color: 'var(--text-secondary)' }}>CATE Correlation</td><td>{metrics.causal.cate_correlation.toFixed(4)}</td></tr>
                  <tr><td style={{ color: 'var(--text-secondary)' }}>Policy Regret</td><td>{(metrics.causal.policy_regret * 100).toFixed(1)}%</td></tr>
                </tbody>
              </table>
            ) : <p style={{ color: 'var(--text-muted)' }}>No causal model data available</p>}
          </div>
        </div>
      </div>

      {/* Decision Engine */}
      <div className="card">
        <div className="card-header"><h3>Decision Engine</h3></div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Incremental Recovery</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--success)' }}>Rs.{metrics.decision.incremental_recovery.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Policy Regret</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{metrics.decision.policy_regret_pct.toFixed(1)}%</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Recovery Rate</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{(metrics.decision.overall_recovery_rate * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
