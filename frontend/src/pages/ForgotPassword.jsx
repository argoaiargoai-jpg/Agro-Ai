import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthShell from '../components/AuthShell'
import { Alert, Button, Field, OtpInput, PasswordMeter } from '../components/ui'
import { useToast } from '../context/ToastContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { isEmail, passwordIssue, passwordStrength } from '../lib/validate'

export default function ForgotPassword() {
  const nav = useNavigate()
  const toast = useToast()
  const [step, setStep] = useState(1)
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [pw, setPw] = useState('')
  const [errs, setErrs] = useState({})
  const [banner, setBanner] = useState(null)
  const [devOtp, setDevOtp] = useState(null)
  const [busy, setBusy] = useState(false)

  async function sendCode(e) {
    e.preventDefault()
    if (!isEmail(email)) return setErrs({ email: 'Enter a valid email address.' })
    setBusy(true); setBanner(null); setErrs({})
    try {
      const r = await api.post('/auth/forgot-password', { email: email.trim() }, { auth: false })
      setDevOtp(r.dev_otp); setStep(2)
    } catch (err) { setBanner(errorMessage(err)) } finally { setBusy(false) }
  }

  async function reset(e) {
    e.preventDefault()
    const v = {}
    if (code.length < 6) v.code = 'Enter the 6-digit code.'
    const issue = passwordIssue(pw)
    if (issue) v.new_password = issue
    setErrs(v)
    if (Object.keys(v).length) return
    setBusy(true); setBanner(null)
    try {
      await api.post('/auth/reset-password', { email: email.trim(), code, new_password: pw }, { auth: false })
      toast('Password updated. Sign in with your new password.')
      nav('/login', { replace: true })
    } catch (err) {
      if (err instanceof ApiError && err.fields) setErrs(err.fields)
      setBanner(errorMessage(err))
    } finally { setBusy(false) }
  }

  return (
    <AuthShell title="Reset your password" subtitle={step === 1 ? "Enter your email and we'll send you a code." : <>Enter the code sent to <strong>{email}</strong> and choose a new password.</>}
      footer={<Link to="/login">Back to sign in</Link>}>
      {banner && <Alert tone="error">{banner}</Alert>}
      {step === 1 ? (
        <form className="form" onSubmit={sendCode} noValidate>
          <Field label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} error={errs.email} autoFocus placeholder="you@farm.com" />
          <Button type="submit" size="lg" block loading={busy}>Send reset code</Button>
        </form>
      ) : (
        <form className="form" onSubmit={reset} noValidate>
          {devOtp && <Alert tone="warn"><strong>Development mode.</strong> Your code is <code style={{ fontWeight: 700 }}>{devOtp}</code>.</Alert>}
          <OtpInput value={code} onChange={setCode} invalid={!!errs.code} />
          {errs.code && <p className="field-error" role="alert">{errs.code}</p>}
          <div className="field">
            <Field label="New password" type="password" autoComplete="new-password" value={pw} onChange={(e) => setPw(e.target.value)} error={errs.new_password} />
            {pw && <PasswordMeter score={passwordStrength(pw)} />}
          </div>
          <Button type="submit" size="lg" block loading={busy}>Update password</Button>
        </form>
      )}
    </AuthShell>
  )
}
