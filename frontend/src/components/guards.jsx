import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { PageLoader } from './ui'

export function RequireAuth() {
  const { status } = useAuth()
  const loc = useLocation()
  if (status === 'loading') return <PageLoader />
  if (status === 'anon') return <Navigate to="/login" replace state={{ from: loc.pathname + loc.search }} />
  return <Outlet />
}

export function RequireAdmin() {
  const { isAdmin } = useAuth()
  return isAdmin ? <Outlet /> : <Navigate to="/forbidden" replace />
}

export function GuestOnly() {
  const { status } = useAuth()
  if (status === 'loading') return <PageLoader />
  if (status === 'authed') return <Navigate to="/dashboard" replace />
  return <Outlet />
}
