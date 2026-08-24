import { AlertCircle, RefreshCw } from 'lucide-react';

export function Loading({ text = 'Loading...' }: { text?: string }) {
  return (
    <div style={{ padding: 48, textAlign: 'center' }}>
      <div className="skeleton" style={{ width: 200, height: 20, margin: '0 auto 12px' }} />
      <div className="skeleton" style={{ width: 300, height: 14, margin: '0 auto' }} />
      <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 16 }}>{text}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div style={{ padding: 48, textAlign: 'center' }}>
      <AlertCircle size={32} color="var(--danger)" style={{ marginBottom: 12 }} />
      <p style={{ fontSize: 14, color: 'var(--text-primary)', marginBottom: 4 }}>Unable to load data</p>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="btn">
          <RefreshCw size={14} /> Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ text = 'No data available' }: { text?: string }) {
  return (
    <div style={{ padding: 48, textAlign: 'center' }}>
      <p style={{ fontSize: 14, color: 'var(--text-muted)' }}>{text}</p>
    </div>
  );
}
