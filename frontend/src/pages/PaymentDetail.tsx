import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft } from 'lucide-react';
import { api } from '../services/api';
import type { ActionDetail } from '../types/api';

const ACTION_LABELS: Record<string, string> = {
  do_nothing: 'Do Nothing', retry: 'Retry', reminder: 'Reminder', alternative_method: 'Alt Method'
};

const REASON_LABELS: Record<string, string> = {
  strong_treatment_response: 'Strong treatment response predicted',
  high_payment_value: 'High payment value makes recovery worthwhile',
  active_customer: 'Customer has high activity level',
  prior_recovery_history: 'Customer has recovered payments before',
};

export default function PaymentDetailPage() {
  const { paymentId } = useParams<{ paymentId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ['payment', paymentId],
    queryFn: () => api.getPayment(paymentId!),
    enabled: !!paymentId,
  });

  if (isLoading) {
    return <div className="empty-state"><div className="skeleton" style={{ width: '100%', height: 500 }} /></div>;
  }

  if (!data) {
    return <div className="empty-state"><p>Payment not found</p></div>;
  }

  const actions = data.actions || [];
  const recAction = data.recommended_action || '';
  const explanation = data.explanation;
  const maxProb = Math.max(...actions.map(a => a.probability), 0.01);

  return (
    <>
      <div className="page-header">
        <Link to="/payments" style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: 'var(--accent)', fontSize: 13, marginBottom: 12, textDecoration: 'none' }}>
          <ArrowLeft size={14} /> Back to Payment Queue
        </Link>
        <h2>Payment {data.payment_id}</h2>
        <p>Rs.{data.amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })} — {data.failure_reason.replace(/_/g, ' ')}</p>
      </div>

      <div className="grid-2 mb-24">
        <div className="card">
          <div className="card-header"><h3>Payment Context</h3></div>
          <div className="card-body">
            <table>
              <tbody>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Amount</td><td style={{ fontWeight: 700 }}>Rs.{data.amount.toLocaleString('en-IN')}</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Failure Reason</td><td style={{ textTransform: 'capitalize' }}>{data.failure_reason.replace(/_/g, ' ')}</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Customer Activity</td><td>{(data.customer_activity * 100).toFixed(0)}%</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Customer Tenure</td><td>{data.customer_tenure} months</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Previous Recoveries</td><td>{data.previous_recoveries}</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Attempt Count</td><td>{data.attempt_count}</td></tr>
                <tr><td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Payment Method</td><td style={{ textTransform: 'capitalize' }}>{data.payment_method.replace(/_/g, ' ')}</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        {explanation && (
          <div className="card">
            <div className="card-header"><h3>Why {ACTION_LABELS[recAction] || recAction}?</h3></div>
            <div className="card-body">
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>Incremental Recovery</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--success)' }}>+{(explanation.incremental_probability_pp * 100).toFixed(1)}pp</div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>vs. doing nothing</div>
              </div>
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>Expected Incremental Revenue</div>
                <div style={{ fontSize: 20, fontWeight: 700 }}>Rs.{explanation.incremental_revenue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
              </div>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>Decision Factors</div>
              {explanation.reason_codes.map(code => (
                <div key={code} style={{ padding: '6px 0', fontSize: 13, borderBottom: '1px solid var(--border-light)' }}>
                  {REASON_LABELS[code] || code}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="card mb-24">
        <div className="card-header"><h3>Counterfactual Action Analysis</h3></div>
        <div className="card-body">
          <div className="action-bar">
            {actions.map((a: ActionDetail) => {
              const isRec = a.action === recAction;
              return (
                <div key={a.action} className={`action-row ${isRec ? 'recommended' : ''}`}>
                  <div className="action-name">{ACTION_LABELS[a.action] || a.action}</div>
                  <div className="prob">{(a.probability * 100).toFixed(1)}%</div>
                  <div className="bar-wrap">
                    <div className="bar" style={{ width: `${(a.probability / maxProb) * 100}%`, background: isRec ? 'var(--accent)' : '#cbd5e1' }} />
                  </div>
                  <div className="value">Rs.{a.net_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
                  <div className="tag">{isRec ? 'RECOMMENDED' : a.eligible ? '' : 'INELIGIBLE'}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="grid-3">
        <div className="card">
          <div className="card-body" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)' }}>Control Probability</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4 }}>{((explanation?.control_probability || 0) * 100).toFixed(1)}%</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)' }}>Recommended Probability</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4, color: 'var(--accent)' }}>{((explanation?.recovery_probability || 0) * 100).toFixed(1)}%</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, color: 'var(--text-secondary)' }}>Net Expected Value</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4 }}>Rs.{(data.recommended_value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
          </div>
        </div>
      </div>
    </>
  );
}
