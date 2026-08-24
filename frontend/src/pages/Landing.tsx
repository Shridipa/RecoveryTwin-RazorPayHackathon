import { Link } from 'react-router-dom';
import { ArrowRight, Shield, Zap, Target } from 'lucide-react';

export default function LandingPage() {
  return (
    <div style={{ minHeight: '100vh', background: '#fff' }}>
      {/* Nav */}
      <nav style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 48px', borderBottom: '1px solid var(--border)' }}>
        <div>
          <span style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.01em' }}>RecoveryTwin</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 12, textTransform: 'uppercase', letterSpacing: '0.08em' }}>AI Revenue Recovery</span>
        </div>
        <Link to="/app" className="btn btn-primary" style={{ textDecoration: 'none' }}>
          Open Dashboard <ArrowRight size={14} />
        </Link>
      </nav>

      {/* Hero */}
      <section style={{ maxWidth: 1100, margin: '0 auto', padding: '80px 48px 60px', display: 'flex', gap: 60, alignItems: 'center' }}>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: 42, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.15, marginBottom: 16 }}>
            Recover more from every failed payment.
          </h1>
          <p style={{ fontSize: 16, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 28, maxWidth: 480 }}>
            RecoveryTwin evaluates recovery options using predictive intelligence, counterfactual decisioning and financial optimization to recommend the action with the highest expected value.
          </p>
          <div style={{ display: 'flex', gap: 12 }}>
            <Link to="/app" className="btn btn-primary btn-lg" style={{ textDecoration: 'none' }}>
              Open Recovery Center <ArrowRight size={16} />
            </Link>
            <a href="#how-it-works" className="btn btn-lg" style={{ textDecoration: 'none' }}>
              How it works
            </a>
          </div>
        </div>

        {/* Product Visualization */}
        <div style={{ flex: '0 0 420px' }}>
          <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 12, padding: 24, boxShadow: 'var(--shadow-lg)' }}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>Failed Payment</div>
              <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4 }}>Rs.8,500</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Bank Decline</div>
            </div>
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 14 }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>Recovery Analysis</div>
              {[
                { name: 'Do Nothing', prob: 18, color: '#94a3b8' },
                { name: 'Retry', prob: 72, color: '#2563eb', selected: true },
                { name: 'Reminder', prob: 46, color: '#f59e0b' },
                { name: 'Alt Method', prob: 61, color: '#8b5cf6' },
              ].map(a => (
                <div key={a.name} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, padding: '6px 10px', borderRadius: 6, background: a.selected ? 'var(--accent-bg)' : 'transparent' }}>
                  <span style={{ width: 90, fontSize: 12, fontWeight: a.selected ? 700 : 500, color: a.selected ? 'var(--accent)' : 'var(--text-secondary)' }}>{a.name}</span>
                  <div style={{ flex: 1, height: 5, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${a.prob}%`, height: '100%', background: a.color, borderRadius: 3 }} />
                  </div>
                  <span style={{ width: 40, textAlign: 'right', fontSize: 13, fontWeight: 700 }}>{a.prob}%</span>
                </div>
              ))}
            </div>
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 14, textAlign: 'center' }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>Recommended</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent)', marginTop: 4 }}>Retry</div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>Expected value Rs.6,120</div>
            </div>
          </div>
        </div>
      </section>

      {/* Value Props */}
      <section style={{ background: 'var(--bg)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 48px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 32 }}>
          {[
            { icon: Target, title: 'Recover', desc: 'Identify payments with the highest recovery opportunity across your entire failed-payment portfolio.' },
            { icon: Zap, title: 'Decide', desc: 'Compare what would happen under each possible recovery action before committing resources.' },
            { icon: Shield, title: 'Protect', desc: 'Apply financial limits, customer fatigue rules and compliance constraints automatically.' },
          ].map(v => (
            <div key={v.title}>
              <v.icon size={24} color="var(--accent)" style={{ marginBottom: 12 }} />
              <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>{v.title}</h3>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{v.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" style={{ maxWidth: 1100, margin: '0 auto', padding: '60px 48px' }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, textAlign: 'center', marginBottom: 40 }}>How RecoveryTwin works</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 24 }}>
          {[
            { num: '01', title: 'Understand', desc: 'Ingest the failed payment, customer history, merchant context and failure reason.' },
            { num: '02', title: 'Predict', desc: 'Estimate the likelihood and timing of recovery under each possible intervention.' },
            { num: '03', title: 'Compare', desc: 'Evaluate what would happen under each action using counterfactual analysis.' },
            { num: '04', title: 'Act', desc: 'Choose the highest-value action that respects all business constraints.' },
          ].map(s => (
            <div key={s.num}>
              <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--border)', marginBottom: 8 }}>{s.num}</div>
              <h4 style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>{s.title}</h4>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Benchmark */}
      <section style={{ background: 'var(--bg)', borderTop: '1px solid var(--border)' }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 48px', textAlign: 'center' }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Simulation Benchmark</h2>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 32 }}>Results from RecoveryTwin's synthetic evaluation environment</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 24 }}>
            {[
              { value: 'Rs.38.2L', label: 'Expected net recovery' },
              { value: '12', label: 'Scenarios tested' },
              { value: '100%', label: 'Positive-net scenarios' },
              { value: '50K', label: 'Synthetic transactions' },
            ].map(m => (
              <div key={m.label}>
                <div style={{ fontSize: 28, fontWeight: 700 }}>{m.value}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>{m.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ padding: '24px 48px', borderTop: '1px solid var(--border)', textAlign: 'center' }}>
        <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          RecoveryTwin — Built for Razorpay AI Revenue Recovery. Benchmark results from synthetic evaluation environment.
        </p>
      </footer>
    </div>
  );
}
