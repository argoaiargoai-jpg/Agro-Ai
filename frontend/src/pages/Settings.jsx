import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PageHead } from '../components/AnalysisBits'
import { Alert, Button, Field, Modal, PasswordMeter, Switch } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { passwordIssue, passwordStrength } from '../lib/validate'

const LANGS = { en: 'English', hi: 'हिन्दी', te: 'తెలుగు', es: 'Español', fr: 'Français' }

function Preferences() {
  const { user, setUser } = useAuth()
  const toast = useToast()
  const p = user.preferences
  async function update(patch) {
    try { setUser(await api.patch('/users/me/preferences', patch)); toast('Preference saved.') }
    catch (e) { toast(errorMessage(e), 'error') }
  }
  return (
    <div className="card card-pad">
      <h3 style={{ marginBottom: 6 }}>Preferences</h3>
      <div className="setting-row">
        <div className="grow"><strong>Language</strong><span className="small muted">Interface language</span></div>
        <select className="select input" style={{ width: 160 }} aria-label="Language" value={p.language} onChange={(e) => update({ language: e.target.value })}>
          {Object.entries(LANGS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>
      <div className="setting-row">
        <div className="grow"><strong>Units</strong><span className="small muted">Measurements shown in reports</span></div>
        <select className="select input" style={{ width: 160 }} aria-label="Units" value={p.units} onChange={(e) => update({ units: e.target.value })}>
          <option value="metric">Metric</option><option value="imperial">Imperial</option>
        </select>
      </div>
      <div className="setting-row">
        <div className="grow"><strong>Email notifications</strong><span className="small muted">Account and analysis updates</span></div>
        <Switch label="Email notifications" checked={!!p.email_notifications} onChange={(v) => update({ email_notifications: v })} />
      </div>
      <div className="setting-row">
        <div className="grow"><strong>Weekly summary</strong><span className="small muted">A digest of your field activity</span></div>
        <Switch label="Weekly summary" checked={!!p.weekly_summary} onChange={(v) => update({ weekly_summary: v })} />
      </div>
    </div>
  )
}

function Password() {
  const { user, logout } = useAuth()
  const nav = useNavigate()
  const toast = useToast()
  const [f, setF] = useState({ current: '', next: '' })
  const [errs, setErrs] = useState({})
  const [banner, setBanner] = useState(null)
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    const v = {}
    if (user.has_password && !f.current) v.current_password = 'Enter your current password.'
    const issue = passwordIssue(f.next)
    if (issue) v.new_password = issue
    setErrs(v)
    if (Object.keys(v).length) return
    setBusy(true); setBanner(null)
    try {
      await api.post('/auth/change-password', { current_password: f.current || null, new_password: f.next })
      await logout().catch(() => {})
      toast('Password changed. Please sign in again.')
      nav('/login', { replace: true })
    } catch (err) {
      if (err instanceof ApiError && err.fields) setErrs(err.fields)
      setBanner(errorMessage(err)); setBusy(false)
    }
  }

  return (
    <form className="card card-pad form" onSubmit={submit} noValidate>
      <div><h3>{user.has_password ? 'Change password' : 'Set a password'}</h3>
        <p className="small muted" style={{ marginTop: 4 }}>You'll be signed out on all devices afterwards.</p></div>
      {banner && <Alert tone="error">{banner}</Alert>}
      {user.has_password && <Field label="Current password" type="password" autoComplete="current-password" value={f.current} onChange={(e) => setF({ ...f, current: e.target.value })} error={errs.current_password} />}
      <div className="field">
        <Field label="New password" type="password" autoComplete="new-password" value={f.next} onChange={(e) => setF({ ...f, next: e.target.value })} error={errs.new_password} />
        {f.next && <PasswordMeter score={passwordStrength(f.next)} />}
      </div>
      <div><Button type="submit" loading={busy}>Update password</Button></div>
    </form>
  )
}

function DangerZone() {
  const { user, logout } = useAuth()
  const nav = useNavigate()
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [pw, setPw] = useState('')
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  async function del() {
    setBusy(true); setErr(null)
    try {
      await api.post('/users/me/delete', { password: pw || null })
      await logout().catch(() => {})
      toast('Your account has been deleted.')
      nav('/', { replace: true })
    } catch (e) { setErr(errorMessage(e)); setBusy(false) }
  }
  return (
    <div className="card card-pad danger-zone">
      <h3 style={{ color: 'var(--error)' }}>Delete account</h3>
      <p className="muted" style={{ margin: '6px 0 14px' }}>Permanently remove your account, scans and uploaded images. This cannot be undone.</p>
      <Button variant="outline-danger" onClick={() => setOpen(true)}>Delete my account</Button>
      {open && (
        <Modal title="Delete your account?" onClose={() => setOpen(false)}
          actions={<><Button variant="secondary" onClick={() => setOpen(false)}>Cancel</Button><Button variant="danger" loading={busy} onClick={del}>Delete forever</Button></>}>
          <p className="muted">All of your data will be erased immediately.</p>
          {err && <Alert tone="error">{err}</Alert>}
          {user.has_password && <Field label="Confirm your password" type="password" value={pw} onChange={(e) => setPw(e.target.value)} />}
        </Modal>
      )}
    </div>
  )
}

export default function Settings() {
  return (
    <>
      <PageHead title="Settings" subtitle="Manage preferences, security and your account." />
      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 18 }}><Preferences /><DangerZone /></div>
        <Password />
      </div>
    </>
  )
}
