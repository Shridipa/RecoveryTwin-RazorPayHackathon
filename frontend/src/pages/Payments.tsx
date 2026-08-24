import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

const ACTION_BADGE: Record<string, string> = {
  'Do Nothing': 'badge-control', retry: 'badge-retry', Retry: 'badge-retry',
  reminder: 'badge-reminder', Reminder: 'badge-reminder',
  alternative_method: 'badge-alternative', 'Alternative Method': 'badge-alternative',
  unknown: 'badge-control',
};

export default function PaymentsPage() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [failureFilter, setFailureFilter] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['payments', page, search, failureFilter],
    queryFn: () => api.getPayments({ page, page_size: 25, ...(search ? { search } : {}), ...(failureFilter ? { failure_reason: failureFilter } : {}) }),
  });

  return (
    <>
      <div className="page-header">
        <h2>Payment Queue</h2>
        <p>{data?.total?.toLocaleString() || '...'} failed payments requiring recovery decisions</p>
      </div>

      <div className="card">
        <div className="card-header" style={{ gap: 12 }}>
          <input
            type="text"
            placeholder="Search payment ID..."
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1); }}
            style={{ padding: '6px 12px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: 13, width: 200 }}
          />
          <select
            value={failureFilter}
            onChange={e => { setFailureFilter(e.target.value); setPage(1); }}
            style={{ padding: '6px 12px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: 13 }}
          >
            <option value="">All failure reasons</option>
            <option value="technical_decline">Technical Decline</option>
            <option value="network_timeout">Network Timeout</option>
            <option value="insufficient_funds">Insufficient Funds</option>
            <option value="expired_card">Expired Card</option>
            <option value="bank_unavailable">Bank Unavailable</option>
            <option value="customer_abandoned">Customer Abandoned</option>
            <option value="incorrect_pin">Incorrect PIN</option>
            <option value="limit_exceeded">Limit Exceeded</option>
            <option value="fraud_suspected">Fraud Suspected</option>
            <option value="account_frozen">Account Frozen</option>
          </select>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Payment ID</th>
                <th>Amount</th>
                <th>Failure Reason</th>
                <th>P(Retry)</th>
                <th>P(Reminder)</th>
                <th>Recommended</th>
                <th>Expected Value</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                Array.from({ length: 10 }).map((_, i) => (
                  <tr key={i}><td colSpan={7}><div className="skeleton" style={{ height: 20 }} /></td></tr>
                ))
              ) : data?.items.map(item => (
                <tr
                  key={item.payment_id}
                  onClick={() => navigate(`/payments/${item.payment_id}`)}
                  style={{ cursor: 'pointer' }}
                >
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{item.payment_id}</td>
                  <td style={{ fontWeight: 600 }}>Rs.{item.amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                  <td style={{ textTransform: 'capitalize' }}>{item.failure_reason.replace(/_/g, ' ')}</td>
                  <td>{item.retry_prob != null ? `${(item.retry_prob * 100).toFixed(1)}%` : '—'}</td>
                  <td>{item.reminder_prob != null ? `${(item.reminder_prob * 100).toFixed(1)}%` : '—'}</td>
                  <td>
                    <span className={`badge ${ACTION_BADGE[item.recommended_action] || 'badge-control'}`}>
                      {item.recommended_action}
                    </span>
                  </td>
                  <td style={{ fontWeight: 600 }}>
                    {item.recommended_value != null ? `Rs.${item.recommended_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {data && (
          <div className="pagination">
            <span>Page {data.page} of {data.total_pages} ({data.total.toLocaleString()} payments)</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button>
              <button disabled={page >= data.total_pages} onClick={() => setPage(p => p + 1)}>Next</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
