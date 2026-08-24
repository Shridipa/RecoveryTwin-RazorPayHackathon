import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '../services/api';
import { inr, pct, actionLabel, failureLabel } from '../utils/format';
import { Loading, ErrorState } from '../components/States';
import type { ActionDetail } from '../types/api';

const REASON_TEXT: Record<string, string> = {
  strong_treatment_response: 'Failure pattern is historically recoverable with this intervention',
  high_payment_value: 'Payment value makes recovery economically worthwhile',
  active_customer: 'Customer has strong recent payment activity',
  prior_recovery_history: 'Customer has successfully recovered payments before',
};

export default function PaymentDetailPage() {
  const { paymentId } = useParams<{ paymentId: string }>();
  const [whatIfAmount, setWhatIfAmount] = useState<number | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['payment', paymentId],
    queryFn: () => api.getPayment(paymentId!),
    enabled: !!paymentId,
  });

  if (isLoading) return <Loading text="Loading payment intelligence..." />;
  if (error || !data) return <ErrorState message="Payment not found or engine unavailable." />;

  const actions = data.actions || [];
  const rec = data.recommended_action || '';
  const exp = data.explanation;
  const maxProb = Math.max(...actions.map(a => a.probability), 0.01);
  const amount = whatIfAmount ?? data.amount;

  return (
    <>
      <div className="page-header">
        <Link to="/app/payments" style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: 'var(--accent)', fontSize: 13, marginBottom: 10, textDecoration: 'none', fontWeight: 500 }}>
          <ArrowLeft size={14} /> Back to Payments
        </Link>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <h2>{data.payment_id}</h2>
          <span style={{ fontSize: 20, fontWeight: 700 }}>{inr(data.amount)}</span>
        </div>
        <p style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>{failureLabel(data.failure_reason)}</p>
      </div>

      {/* Recommendation Banner */}
      <div className="card mb-20" style={{ borderLeft: '4px solid var(--accent)' }}>
        <div className="card-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Recommended Action</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--accent)' }}>{actionLabel(rec)}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Recovery Probability</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{pct(exp?.recovery_probability || 0)}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Net Expected Value</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--success)' }}>{inr(data.recommended_value || 0)}</div>
          </div>
        </div>
      </div>

      <div className="grid-2 mb-20">
        {/* Payment Context */}
        <div className="card">
          <div className="card-header"><h3>Payment Context</h3></div>
          <div className="card-body">
            <table>
              <tbody>
                {[
                  ['Amount', inr(data.amount)],
                  ['Failure Reason', failureLabel(data.failure_reason)],
                  ['Customer Activity', `${(data.customer_activity * 100).toFixed(0)}%`],
                  ['Customer Tenure', `${data.customer_tenure} months`],
                  ['Previous Recoveries', String(data.previous_recoveries)],
                  ['Attempt Count', String(data.attempt_count)],
                  ['Payment Method', data.payment_method?.replace(/_/g, ' ')],
                ].map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ color: 'var(--text-secondary)', fontWeight: 500, padding: '7px 0', borderBottom: '1px solid var(--border-light)' }}>{k}</td>
                    <td style={{ fontWeight: 600, padding: '7px 0', borderBottom: '1px solid var(--border-light)', textAlign: 'right' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Why This Decision */}
        <div className="card">
          <div className="card-header"><h3>Why {actionLabel(rec)}?</h3></div>
          <div className="card-body">
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Incremental Impact</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--success)' }}>+{(exp?.incremental_probability_pp || 0) > 0 ? '+' : ''}{((exp?.incremental_probability_pp || 0) * 100).toFixed(1)}pp</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>vs. doing nothing</div>
            </div>
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>Expected Incremental Revenue</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{inr(exp?.incremental_revenue || 0)}</div>
            </div>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8 }}>Decision Factors</div>
            {(exp?.reason_codes || []).length > 0 ? exp!.reason_codes.map(code => (
              <div key={code} className="explanation-factor">
                <CheckCircle2 size={14} className="factor-check" />
                <span>{REASON_TEXT[code] || code.replace(/_/g, ' ')}</span>
              </div>
            )) : (
              <div className="explanation-factor">
                <CheckCircle2 size={14} className="factor-check" />
                <span>Action selected based on model's financial value optimization</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Counterfactual Analysis */}
      <div className="card mb-20">
        <div className="card-header"><h3>What-If Analysis — All Recovery Options</h3></div>
        <div className="card-body">
          <div className="action-bar">
            {actions.map((a: ActionDetail) => {
              const isRec = a.action === rec;
              return (
                <div key={a.action} className={`action-row ${isRec ? 'recommended' : ''} ${!a.eligible ? 'ineligible' : ''}`}>
                  <div className="action-name">{actionLabel(a.action)}</div>
                  <div className="prob">{(a.probability * 100).toFixed(0)}%</div>
                  <div className="bar-wrap">
                    <div className="bar" style={{ width: `${(a.probability / maxProb) * 100}%`, background: isRec ? 'var(--accent)' : a.eligible ? '#cbd5e1' : '#e2e8f0' }} />
                  </div>
                  <div className="value">{inr(a.net_value)}</div>
                  <div className="tag">{isRec ? 'RECOMMENDED' : !a.eligible ? 'INELIGIBLE' : ''}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* What-If Amount Slider */}
      <div className="card mb-20">
        <div className="card-header"><h3>What-If: Payment Amount</h3></div>
        <div className="card-body">
          <div className="slider-group">
            <label>
              <span>Payment amount</span>
              <span style={{ fontWeight: 700 }}>{inr(amount)}</span>
            </label>
            <input type="range" min={50} max={Math.max(data.amount * 3, 50000)} step={50}
              value={whatIfAmount ?? data.amount}
              onChange={e => setWhatIfAmount(Number(e.target.value))} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 12 }}>
            {actions.map(a => (
              <div key={a.action} style={{ textAlign: 'center', padding: 12, borderRadius: 'var(--radius)', background: a.action === rec ? 'var(--accent-bg)' : 'var(--bg)', border: `1px solid ${a.action === rec ? 'var(--accent)' : 'var(--border)'}` }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>{actionLabel(a.action)}</div>
                <div style={{ fontSize: 18, fontWeight: 700 }}>{inr(amount * a.probability)}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>expected recovery</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid-3">
        <div className="card"><div className="card-body" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>Do Nothing Probability</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{pct(exp?.control_probability || 0)}</div>
        </div></div>
        <div className="card"><div className="card-body" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>Recommended Probability</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4, color: 'var(--accent)' }}>{pct(exp?.recovery_probability || 0)}</div>
        </div></div>
        <div className="card"><div className="card-body" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>Incremental Revenue</div>
          <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4, color: 'var(--success)' }}>{inr(exp?.incremental_revenue || 0)}</div>
        </div></div>
      </div>
    </>
  );
}
