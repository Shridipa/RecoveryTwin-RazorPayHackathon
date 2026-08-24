import { Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, CreditCard, BrainCircuit, FlaskConical, BarChart3, Activity, Shield } from 'lucide-react';
import OverviewPage from './pages/Overview';
import PaymentsPage from './pages/Payments';
import PaymentDetailPage from './pages/PaymentDetail';
import ScenariosPage from './pages/Scenarios';
import ModelsPage from './pages/Models';
import FinancialPage from './pages/Financial';

const nav = [
  { section: 'OVERVIEW', items: [
    { to: '/', icon: LayoutDashboard, label: 'Command Center' },
  ]},
  { section: 'RECOVERY', items: [
    { to: '/payments', icon: CreditCard, label: 'Payment Queue' },
  ]},
  { section: 'INTELLIGENCE', items: [
    { to: '/models', icon: BrainCircuit, label: 'Model Performance' },
  ]},
  { section: 'FINANCIAL', items: [
    { to: '/scenarios', icon: FlaskConical, label: 'Scenario Lab' },
    { to: '/financial', icon: BarChart3, label: 'Financial Analysis' },
  ]},
];

export default function App() {
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>RecoveryTwin</h1>
          <p>Revenue Recovery Intelligence</p>
        </div>
        {nav.map(section => (
          <div className="sidebar-section" key={section.section}>
            <div className="sidebar-section-label">{section.section}</div>
            {section.items.map(item => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
              >
                <item.icon />
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}
        <div className="sidebar-status">
          <div className="status-dot" />
          <span>ML System Healthy</span>
        </div>
      </aside>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/payments" element={<PaymentsPage />} />
          <Route path="/payments/:paymentId" element={<PaymentDetailPage />} />
          <Route path="/scenarios" element={<ScenariosPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/financial" element={<FinancialPage />} />
        </Routes>
      </main>
    </div>
  );
}
