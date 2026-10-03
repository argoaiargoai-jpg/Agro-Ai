import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthShell from '../components/AuthShell'
import { Alert, Button, Field, PasswordMeter } from '../components/ui'
import { useConfig } from '../context/ConfigContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { isEmail, passwordIssue, passwordStrength } from '../lib/validate'
import { GoogleButton } from './Login'

export default function Register() {
  const nav = useNavigate()
  const { config } = useConfig()
  const [form, setForm] = useState({ full_name: '', email: '', password: '', confirm: '' })
  const [errs, setErrs] = useState({})
  const [banner, setBanner] = useState(null)
  const [busy, setBusy] = useState(false)
  const set = (k) => (e) => { setForm({ ...form, [k]: e.target.value }); setErrs({ ...errs, [k]: undefined }) }

  async function submit(e) {
    e.preventDefault()
    const v = {}
    if (form.full_name.trim().length < 2) v.full_name = 'Enter your full name.'
    if (!isEmail(form.email)) v.email = 'Enter a valid email address.'
    const pw = passwordIssue(form.password)
    if (pw) v.password = pw
    if (form.confirm !== form.password) v.confirm = 'Passwords do not match.'
    setErrs(v)
    if (Object.keys(v).length) return
    setBusy(true); setBanner(null)
    try {
      const r = await api.post('/auth/register', { full_name: form.full_name.trim(), email: form.email.trim(), password: form.password }, { auth: false })
      nav('/verify', { state: { email: r.email, devOtp: r.dev_otp, resendIn: r.resend_in_seconds } })
    } catch (err) {
      if (err instanceof ApiError && err.fields) setErrs(err.fields)
      setBanner(errorMessage(err))
    } finally { setBusy(false) }
  }

  return (
    <AuthShell title="Create your account" subtitle="Start monitoring crop health in minutes." footer={<>Already have an account? <Link to="/login">Sign in</Link></>}>
      {!config.registration_enabled && <Alert tone="warn">New registrations are currently closed.</Alert>}
      {banner && <Alert tone="error">{banner}</Alert>}
      <form className="form" onSubmit={submit} noValidate>
        <Field label="Full name" autoComplete="name" placeholder="Asha Patel" value={form.full_name} onChange={set('full_name')} error={errs.full_name} autoFocus />
        <Field label="Email" type="email" autoComplete="email" placeholder="you@farm.com" value={form.email} onChange={set('email')} error={errs.email} />
        <div className="field">
          <Field label="Password" type="password" autoComplete="new-password" placeholder="At least 8 characters" value={form.password} onChange={set('password')} error={errs.password} hint="Use 8+ characters with a letter and a number." />
          {form.password && <PasswordMeter score={passwordStrength(form.password)} />}
        </div>
        <Field label="Confirm password" type="password" autoComplete="new-password" value={form.confirm} onChange={set('confirm')} error={errs.confirm} />
        <Button type="submit" size="lg" block loading={busy} disabled={!config.registration_enabled}>Create account</Button>
      </form>
      <div className="divider">or</div>
      <GoogleButton label="Sign up with Google" />
    </AuthShell>
  )
}
