import { useState, useEffect } from 'react';
import { Routes, Route, NavLink, Link, Navigate, useLocation } from 'react-router-dom';
import { LayoutDashboard, CreditCard, FlaskConical, BarChart3, Activity, Shield, Zap, ArrowRight } from 'lucide-react';
import { api } from './services/api';
import LandingPage from './pages/Landing';
import OverviewPage from './pages/Overview';
import PaymentsPage from './pages/Payments';
import PaymentDetailPage from './pages/PaymentDetail';
import ScenariosPage from './pages/Scenarios';
import ModelsPage from './pages/Models';
import FinancialPage from './pages/Financial';

function DashboardLayout() {
  const [online, setOnline] = useState(true);
  useEffect(() => {
    const check = () => api.getHealth().then(() => setOnline(true)).catch(() => setOnline(false));
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>RecoveryTwin</h1>
          <p>Revenue Recovery</p>
        </div>
        <div className="sidebar-section">
          <NavLink to="/app" end className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
            <LayoutDashboard /> Command Center
          </NavLink>
          <NavLink to="/app/payments" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
            <CreditCard /> Payments
          </NavLink>
          <NavLink to="/app/scenarios" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
            <FlaskConical /> Scenario Lab
          </NavLink>
          <NavLink to="/app/financial" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
            <BarChart3 /> Financial Impact
          </NavLink>
        </div>
        <div className="sidebar-divider" />
        <div className="sidebar-section">
          <NavLink to="/app/models" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
            <Shield /> System Health
          </NavLink>
        </div>
        <div className="sidebar-status">
          <div className={`status-dot ${online ? '' : 'offline'}`} />
          <span>{online ? 'Engine Online' : 'Engine Offline'}</span>
        </div>
      </aside>
      <main className="main-content">
        <Routes>
          <Route path="/app" element={<OverviewPage />} />
          <Route path="/app/payments" element={<PaymentsPage />} />
          <Route path="/app/payments/:paymentId" element={<PaymentDetailPage />} />
          <Route path="/app/scenarios" element={<ScenariosPage />} />
          <Route path="/app/financial" element={<FinancialPage />} />
          <Route path="/app/models" element={<ModelsPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  const location = useLocation();
  const isLanding = location.pathname === '/';

  return isLanding ? (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/app/*" element={<DashboardLayout />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  ) : <DashboardLayout />;
}
