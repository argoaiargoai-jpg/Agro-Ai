import { Link } from 'react-router-dom'
import { Camera, ShieldCheck, Sprout } from 'lucide-react'
import { Logo } from './ui'
import { Leaf } from './Reveal'

export default function AuthShell({ title, subtitle, children, footer }) {
  return (
    <div className="auth">
      <aside className="auth-art">
        <Logo light />
        <div>
          <h2>Know the health of every field, leaf by leaf.</h2>
          <p>AGRO AI turns a photo from your phone into clear, actionable crop insight.</p>
          <ul>
            <li><Camera size={20} /> Snap or upload a photo of any leaf</li>
            <li><Sprout size={20} /> Track the health of your crops over time</li>
            <li><ShieldCheck size={20} /> Secure accounts with email verification</li>
          </ul>
        </div>
        <small style={{ color: '#86c79b' }}>© {new Date().getFullYear()} AGRO AI</small>
        <Leaf size={46} className="float-leaf" style={{ left: '12%', top: '22%', animation: 'sway 8s ease-in-out infinite' }} color="#86efac" />
        <Leaf size={30} className="float-leaf" style={{ right: '14%', top: '40%', animation: 'sway 6s -2s ease-in-out infinite' }} color="#bef264" />
        <Leaf size={56} className="float-leaf" style={{ right: '28%', bottom: '14%', animation: 'sway 10s -4s ease-in-out infinite' }} color="#4ade80" />
        <svg className="leafbg" width="420" height="420" viewBox="0 0 32 32" aria-hidden="true"><path d="M9 22c0-8 5-13 14-13 0 9-5 14-13 14l-1-1z" fill="#fff" /></svg>
      </aside>
      <main className="auth-main">
        <div className="auth-card">
          <div className="auth-mobile-logo"><Logo /></div>
          <div>
            <h1>{title}</h1>
            {subtitle && <p className="muted" style={{ marginTop: 8 }}>{subtitle}</p>}
          </div>
          {children}
          {footer && <p className="auth-foot">{footer}</p>}
          <p className="auth-foot small"><Link to="/">← Back to home</Link></p>
        </div>
      </main>
    </div>
  )
}
