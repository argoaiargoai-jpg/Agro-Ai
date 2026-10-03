import { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import AuthShell from '../components/AuthShell'
import { Alert, Button, Field, GoogleIcon } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { useConfig } from '../context/ConfigContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { OAUTH_ERRORS } from '../lib/format'
import { isEmail } from '../lib/validate'

export const googleUrl = `${import.meta.env.VITE_API_URL || ''}/api/v1/auth/google/login`

export function GoogleButton({ label = 'Continue with Google' }) {
  const { config } = useConfig()
  const [busy, setBusy] = useState(false)
  // Coming back with the browser's Back button restores this page from cache: don't leave the button stuck on "loading".
  useEffect(() => {
    const reset = (e) => { if (e.persisted) setBusy(false) }
    window.addEventListener('pageshow', reset)
    return () => window.removeEventListener('pageshow', reset)
  }, [])
  function go() {
    if (busy) return
    setBusy(true)
    window.location.href = googleUrl
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <Button variant="secondary" block onClick={go} loading={busy} disabled={!config.google_enabled || busy} type="button">
        {!busy && <GoogleIcon />} {busy ? 'Connecting to Google…' : label}
      </Button>
      {!config.google_enabled && <p className="field-hint" style={{ textAlign: 'center' }}>Google sign-in isn't configured on this server yet.</p>}
    </div>
  )
}

export default function Login() {
  const { login } = useAuth()
  const nav = useNavigate()
  const loc = useLocation()
  const [params] = useSearchParams()
  const [form, setForm] = useState({ email: '', password: '' })
  const [errs, setErrs] = useState({})
  const [banner, setBanner] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const code = params.get('oauth_error')
    if (code) setBanner(OAUTH_ERRORS[code] || 'Google sign-in failed. Please try again.')
  }, [params])

  const set = (k) => (e) => { setForm({ ...form, [k]: e.target.value }); setErrs({ ...errs, [k]: undefined }) }

  async function submit(e) {
    e.preventDefault()
    const v = {}
    if (!isEmail(form.email)) v.email = 'Enter a valid email address.'
    if (!form.password) v.password = 'Enter your password.'
    setErrs(v)
    if (Object.keys(v).length) return
    setBusy(true); setBanner(null)
    try {
      await login(form.email.trim(), form.password)
      nav(loc.state?.from || '/dashboard', { replace: true })
    } catch (err) {
      if (err instanceof ApiError && err.code === 'email_not_verified') {
        let sent = null
        try { sent = await api.post('/auth/resend-otp', { email: form.email.trim() }, { auth: false }) } catch { /* cooldown: code already sent */ }
        nav('/verify', { state: { email: form.email.trim().toLowerCase(), devOtp: sent?.dev_otp, resendIn: sent?.resend_in_seconds } })
        return
      }
      if (err instanceof ApiError && err.fields) setErrs(err.fields)
      setBanner(errorMessage(err))
    } finally { setBusy(false) }
  }

  return (
    <AuthShell title="Welcome back" subtitle="Sign in to your AGRO AI account." footer={<>New to AGRO AI? <Link to="/register">Create an account</Link></>}>
      {banner && <Alert tone="error">{banner}</Alert>}
      <form className="form" onSubmit={submit} noValidate>
        <Field label="Email" type="email" autoComplete="email" placeholder="you@farm.com" value={form.email} onChange={set('email')} error={errs.email} autoFocus />
        <Field label="Password" type="password" autoComplete="current-password" placeholder="Your password" value={form.password} onChange={set('password')} error={errs.password} />
        <div style={{ textAlign: 'right', marginTop: -6 }}><Link to="/forgot-password" className="small">Forgot password?</Link></div>
        <Button type="submit" size="lg" block loading={busy}>Sign in</Button>
      </form>
      <div className="divider">or</div>
      <GoogleButton />
    </AuthShell>
  )
}
