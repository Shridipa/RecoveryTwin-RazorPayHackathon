import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { api } from '../services/api';
import { inr, pct, actionLabel, failureLabel } from '../utils/format';
import { Loading, ErrorState, Empty } from '../components/States';

const REASONS = ['technical_decline', 'network_timeout', 'insufficient_funds', 'expired_card', 'bank_unavailable', 'customer_abandoned', 'incorrect_pin', 'limit_exceeded', 'fraud_suspected', 'account_frozen'];

const ACTION_BADGE: Record<string, string> = {
  retry: 'badge-retry', reminder: 'badge-reminder', alternative_method: 'badge-alternative', 'Do Nothing': 'badge-control',
};

export default function PaymentsPage() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [reason, setReason] = useState('');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['payments', page, search, reason],
    queryFn: () => api.getPayments({ page, page_size: 25, ...(search ? { search } : {}), ...(reason ? { failure_reason: reason } : {}) }),
  });

  if (error) return <ErrorState message="Unable to load payments." onRetry={() => refetch()} />;

  return (
    <>
      <div className="page-header">
        <h2>Payments</h2>
        <p>{data?.total?.toLocaleString() || '...'} failed payments</p>
      </div>

      <div className="card">
        <div className="card-header" style={{ gap: 10 }}>
          <div style={{ position: 'relative', flex: '0 0 220px' }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text" placeholder="Search payment ID..." value={search}
              onChange={e => { setSearch(e.target.value); setPage(1); }}
              style={{ padding: '7px 10px 7px 30px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: 13, width: '100%' }}
            />
          </div>
          <select value={reason} onChange={e => { setReason(e.target.value); setPage(1); }}
            style={{ padding: '7px 12px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: 13 }}>
            <option value="">All failure reasons</option>
            {REASONS.map(r => <option key={r} value={r}>{failureLabel(r)}</option>)}
          </select>
        </div>

        {isLoading ? <Loading /> : !data?.items.length ? <Empty text="No payments match your filters." /> : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Payment</th><th>Amount</th><th>Issue</th><th>P(Retry)</th><th>P(Reminder)</th><th>Recommendation</th><th>Expected Value</th></tr>
                </thead>
                <tbody>
                  {data.items.map(p => (
                    <tr key={p.payment_id} className="clickable" onClick={() => navigate(`/app/payments/${p.payment_id}`)}>
                      <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{p.payment_id}</td>
                      <td style={{ fontWeight: 600 }}>{inr(p.amount)}</td>
                      <td style={{ color: 'var(--text-secondary)' }}>{failureLabel(p.failure_reason)}</td>
                      <td>{p.retry_prob != null ? pct(p.retry_prob) : '—'}</td>
                      <td>{p.reminder_prob != null ? pct(p.reminder_prob) : '—'}</td>
                      <td><span className={`badge ${ACTION_BADGE[p.recommended_action?.toLowerCase()] || 'badge-control'}`}>{actionLabel(p.recommended_action)}</span></td>
                      <td style={{ fontWeight: 600 }}>{p.recommended_value != null ? inr(p.recommended_value) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>Page {data.page} of {data.total_pages} ({data.total.toLocaleString()} payments)</span>
              <div style={{ display: 'flex', gap: 8 }}>
                <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button>
                <button disabled={page >= data.total_pages} onClick={() => setPage(p => p + 1)}>Next</button>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
