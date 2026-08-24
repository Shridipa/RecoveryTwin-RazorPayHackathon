import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { inr, inrFull } from '../utils/format';
import { Loading, ErrorState } from '../components/States';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const SCENARIO_LABELS: Record<string, string> = {
  BASELINE: 'Baseline', LOW_RECOVERY: 'Low Recovery', HIGH_RECOVERY: 'High Recovery',
  HIGH_COST: 'High Cost', LOW_COST: 'Low Cost', HIGH_FATIGUE: 'High Fatigue',
  LOW_FATIGUE: 'Low Fatigue', HIGH_VALUE: 'High Value', LOW_VALUE: 'Low Value',
  DEGRADATION_20: 'Degradation 20%', DEGRADATION_40: 'Degradation 40%',
  ADVERSE_COMBINED: 'Worst Case',
};

export default function ScenariosPage() {
  const [selected, setSelected] = useState('BASELINE');
  const { data: scenarios, isLoading, error, refetch } = useQuery({ queryKey: ['scenarios'], queryFn: api.getScenarios });

  if (isLoading) return <Loading text="Loading scenarios..." />;
  if (error || !scenarios) return <ErrorState message="Unable to load scenarios." onRetry={() => refetch()} />;

  const active = scenarios.find(s => s.name === selected) || scenarios[0];
  const baseline = scenarios.find(s => s.name === 'BASELINE');
  const chartData = scenarios.map(s => ({
    name: SCENARIO_LABELS[s.name] || s.name, value: s.recoverytwin_revenue, name_raw: s.name,
  }));
  const passing = scenarios.filter(s => s.beats_do_nothing).length;

  return (
    <>
      <div className="page-header">
        <h2>Scenario Lab</h2>
        <p>Stress-test the recovery policy under different conditions</p>
      </div>

      <div className="metrics-row mb-20">
        <div className="metric-card accent">
          <div className="label">Baseline Recovery</div>
          <div className="value">{inr(baseline?.recoverytwin_revenue || 0)}</div>
        </div>
        <div className="metric-card success">
          <div className="label">Scenarios Passing</div>
          <div className="value">{passing}/{scenarios.length}</div>
          <div className="sub">Beat doing nothing</div>
        </div>
        <div className="metric-card">
          <div className="label">Selected Scenario</div>
          <div className="value" style={{ fontSize: 18 }}>{SCENARIO_LABELS[active.name] || active.name}</div>
          <div className="sub">{active.description}</div>
        </div>
        <div className="metric-card">
          <div className="label">Scenario Recovery</div>
          <div className="value">{inr(active.recoverytwin_revenue)}</div>
          <div className="sub">{active.beats_do_nothing ? 'Better than doing nothing' : 'Below doing nothing'}</div>
        </div>
      </div>

      <div className="grid-2 mb-20">
        <div className="card">
          <div className="card-header"><h3>Scenario Comparison</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={Math.max(260, scenarios.length * 28)}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
                <XAxis type="number" tickFormatter={(v: number) => inr(v)} tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => inrFull(v)} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} onClick={(d: any) => d?.name_raw && setSelected(d.name_raw)}>
                  {chartData.map((e, i) => (
                    <Cell key={i} fill={e.name_raw === selected ? '#2563eb' : e.name_raw === 'BASELINE' ? '#60a5fa' : '#cbd5e1'} style={{ cursor: 'pointer' }} />
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
                <tr><th>Scenario</th><th>RT Revenue</th><th>Incremental</th><th>Regret</th><th>Profitable?</th></tr>
              </thead>
              <tbody>
                {scenarios.map(s => (
                  <tr key={s.name} className="clickable" onClick={() => setSelected(s.name)}
                    style={s.name === selected ? { background: 'var(--accent-bg)' } : {}}>
                    <td style={{ fontWeight: s.name === selected ? 700 : 500 }}>{SCENARIO_LABELS[s.name] || s.name}</td>
                    <td style={{ fontWeight: 600 }}>{inr(s.recoverytwin_revenue)}</td>
                    <td style={{ color: s.incremental > 0 ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>
                      {s.incremental > 0 ? '+' : ''}{inr(s.incremental)}
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
      </div>
    </>
  );
}
