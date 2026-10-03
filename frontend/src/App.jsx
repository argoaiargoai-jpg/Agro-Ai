import { Route, Routes } from 'react-router-dom'
import { GuestOnly, RequireAdmin, RequireAuth } from './components/guards'
import AppLayout from './layouts/AppLayout'
import Admin from './pages/admin/Admin'
import Analyze from './pages/Analyze'
import AnalysisResult from './pages/AnalysisResult'
import Dashboard from './pages/Dashboard'
import ForgotPassword from './pages/ForgotPassword'
import GoogleDone from './pages/GoogleDone'
import History from './pages/History'
import Landing from './pages/Landing'
import Login from './pages/Login'
import { Forbidden, NotFound } from './pages/NotFound'
import Profile from './pages/Profile'
import Register from './pages/Register'
import Settings from './pages/Settings'
import VerifyOtp from './pages/VerifyOtp'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route element={<GuestOnly />}>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
      </Route>
      <Route path="/verify" element={<VerifyOtp />} />
      <Route path="/auth/google/done" element={<GoogleDone />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/analysis/:id" element={<AnalysisResult />} />
          <Route path="/history" element={<History />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/forbidden" element={<Forbidden />} />
          <Route element={<RequireAdmin />}>
            <Route path="/admin" element={<Admin />} />
          </Route>
        </Route>
      </Route>
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
