import { useEffect, useRef, useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { MailCheck } from 'lucide-react'
import AuthShell from '../components/AuthShell'
import { Alert, Button, OtpInput } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { api, errorMessage } from '../lib/api'

export function useCountdown(initial = 0) {
  const [left, setLeft] = useState(initial)
  useEffect(() => {
    if (left <= 0) return
    const t = setTimeout(() => setLeft((x) => x - 1), 1000)
    return () => clearTimeout(t)
  }, [left])
  return [left, setLeft]
}

export default function VerifyOtp() {
  const { state } = useLocation()
  const { verifyOtp } = useAuth()
  const nav = useNavigate()
  const [code, setCode] = useState('')
  const [error, setError] = useState(null)
  const [info, setInfo] = useState(null)
  const [busy, setBusy] = useState(false)
  const [devOtp, setDevOtp] = useState(state?.devOtp)
  const [wait, setWait] = useCountdown(state?.resendIn ?? 0)
  const submitting = useRef(false)

  async function verify(c) {
    if (submitting.current) return
    submitting.current = true; setBusy(true); setError(null)
    try {
      await verifyOtp(state.email, c)
      nav('/dashboard', { replace: true })
    } catch (err) {
      setError(errorMessage(err)); setCode('')
    } finally {
      submitting.current = false; setBusy(false)
      setTimeout(() => document.querySelector('.otp input')?.focus(), 0) // inputs were disabled while verifying
    }
  }

  useEffect(() => { if (code.length === 6) verify(code) }, [code]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!state?.email) return <Navigate to="/register" replace />

  async function resend() {
    setError(null); setInfo(null)
    try {
      const r = await api.post('/auth/resend-otp', { email: state.email }, { auth: false })
      setInfo('A new code has been sent.'); setDevOtp(r.dev_otp); setWait(r.resend_in_seconds); setCode('')
    } catch (err) { setError(errorMessage(err)) }
  }

  return (
    <AuthShell title="Check your email" subtitle={<>We sent a 6-digit code to <strong>{state.email}</strong>. It expires in 10 minutes.</>}
      footer={<>Wrong email? <Link to="/register">Start over</Link></>}>
      <div style={{ display: 'grid', placeItems: 'center' }}>
        <div className="state-icon" style={{ width: 56, height: 56, borderRadius: 16, background: 'var(--mint-100)', color: 'var(--forest)', display: 'grid', placeItems: 'center' }}><MailCheck size={28} /></div>
      </div>
      {error && <Alert tone="error">{error}</Alert>}
      {info && <Alert tone="success">{info}</Alert>}
      {devOtp && (
        <Alert tone="warn" data-testid="dev-otp">
          <strong>Development mode.</strong> Email isn't sent; your code is <code style={{ fontWeight: 700, letterSpacing: 2 }}>{devOtp}</code>.{' '}
          <button className="btn btn-ghost btn-sm" onClick={() => setCode(devOtp)} type="button">Fill it in</button>
        </Alert>
      )}
      <OtpInput value={code} onChange={setCode} invalid={!!error} disabled={busy} />
      <Button size="lg" block loading={busy} disabled={code.length < 6} onClick={() => verify(code)}>Verify email</Button>
      <p className="auth-foot">
        Didn't get it?{' '}
        <button className="btn btn-ghost btn-sm" onClick={resend} disabled={wait > 0} type="button">{wait > 0 ? `Resend in ${wait}s` : 'Resend code'}</button>
      </p>
    </AuthShell>
  )
}
