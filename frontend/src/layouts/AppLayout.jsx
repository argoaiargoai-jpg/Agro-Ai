import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { History, LayoutDashboard, LogOut, Menu, ScanLine, Settings, ShieldCheck, UserRound, X, Megaphone } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useConfig } from '../context/ConfigContext'
import { Logo } from '../components/ui'
import { initials } from '../lib/format'

const NAV = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/analyze', label: 'Analyze plant', icon: ScanLine },
  { to: '/history', label: 'History', icon: History },
  { to: '/profile', label: 'Profile', icon: UserRound },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export default function AppLayout() {
  const { user, isAdmin, logout } = useAuth()
  const { config } = useConfig()
  const [open, setOpen] = useState(false)
  const nav = useNavigate()
  const loc = useLocation()
  useEffect(() => setOpen(false), [loc.pathname])

  const link = ({ to, label, icon: Icon }) => (
    <NavLink key={to} to={to} className={({ isActive }) => `side-link${isActive ? ' active' : ''}`}>
      <Icon size={19} />{label}
    </NavLink>
  )

  return (
    <div className="shell">
      <aside className={`sidebar${open ? ' open' : ''}`} aria-label="Primary">
        <div className="side-top">
          <Logo to="/dashboard" light />
          <button className="btn btn-ghost btn-icon side-close" onClick={() => setOpen(false)} aria-label="Close menu"><X size={20} /></button>
        </div>
        <nav className="side-nav">
          {NAV.map(link)}
          {isAdmin && (<>
            <p className="side-section">Administration</p>
            {link({ to: '/admin', label: 'Admin console', icon: ShieldCheck })}
          </>)}
        </nav>
        <div className="side-user">
          <div className="avatar">{initials(user?.full_name)}</div>
          <div className="side-user-meta">
            <strong>{user?.full_name}</strong>
            <span>{isAdmin ? 'Administrator' : 'Farmer'}</span>
          </div>
          <button className="btn btn-ghost btn-icon" style={{ color: '#bbf7d0' }} onClick={async () => { await logout(); nav('/login') }} aria-label="Sign out" title="Sign out">
            <LogOut size={18} />
          </button>
        </div>
      </aside>
      {open && <div className="scrim" onClick={() => setOpen(false)} />}
      <div className="main">
        <header className="topbar">
          <button className="btn btn-secondary btn-icon menu-btn" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <div className="topbar-title"><Logo to="/dashboard" /></div>
          <div className="topbar-right">
            <div className="avatar sm" aria-hidden="true">{initials(user?.full_name)}</div>
          </div>
        </header>
        {config.announcement && (
          <div className="announce"><Megaphone size={16} /> {config.announcement}</div>
        )}
        {config.maintenance_mode && (
          <div className="announce warn">AGRO AI is in maintenance mode — new analyses are temporarily paused.</div>
        )}
        <main className="content"><Outlet /></main>
      </div>
    </div>
  )
}
