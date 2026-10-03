import { useState } from 'react'
import { Mail, ShieldCheck } from 'lucide-react'
import { PageHead } from '../components/AnalysisBits'
import { Alert, Button, Field } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { fmtDate, initials } from '../lib/format'

export default function Profile() {
  const { user, setUser } = useAuth()
  const toast = useToast()
  const [form, setForm] = useState({ full_name: user.full_name, phone: user.phone || '', farm_name: user.farm_name || '', region: user.region || '' })
  const [errs, setErrs] = useState({})
  const [banner, setBanner] = useState(null)
  const [busy, setBusy] = useState(false)
  const set = (k) => (e) => { setForm({ ...form, [k]: e.target.value }); setErrs({ ...errs, [k]: undefined }) }
  const dirty = ['full_name', 'phone', 'farm_name', 'region'].some((k) => form[k] !== (user[k] || ''))

  async function save(e) {
    e.preventDefault()
    if (form.full_name.trim().length < 2) return setErrs({ full_name: 'Enter your full name.' })
    setBusy(true); setBanner(null)
    try {
      setUser(await api.patch('/users/me', form))
      toast('Profile saved.')
    } catch (err) {
      if (err instanceof ApiError && err.fields) setErrs(err.fields)
      setBanner(errorMessage(err))
    } finally { setBusy(false) }
  }

  return (
    <>
      <PageHead title="Profile" subtitle="Your personal and farm details." />
      <div className="grid grid-main">
        <form className="card card-pad form" onSubmit={save} noValidate>
          {banner && <Alert tone="error">{banner}</Alert>}
          <Field label="Full name" value={form.full_name} onChange={set('full_name')} error={errs.full_name} autoComplete="name" />
          <div className="form-row">
            <Field label="Farm name" value={form.farm_name} onChange={set('farm_name')} error={errs.farm_name} placeholder="Green Acres" />
            <Field label="Region" value={form.region} onChange={set('region')} error={errs.region} placeholder="e.g. Punjab" />
          </div>
          <Field label="Phone" type="tel" value={form.phone} onChange={set('phone')} error={errs.phone} placeholder="+91 98765 43210" autoComplete="tel" />
          <div><Button type="submit" loading={busy} disabled={!dirty}>Save changes</Button></div>
        </form>
        <div className="card card-pad" style={{ display: 'grid', gap: 18 }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            <div className="avatar lg">{initials(user.full_name)}</div>
            <div style={{ minWidth: 0 }}>
              <h3 style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{user.full_name}</h3>
              <span className={`badge ${user.role === 'admin' ? 'dark' : ''}`} style={{ marginTop: 6 }}>{user.role === 'admin' ? 'Administrator' : 'Member'}</span>
            </div>
          </div>
          <dl className="kv" style={{ gridTemplateColumns: '100px 1fr' }}>
            <dt><Mail size={14} style={{ display: 'inline' }} /> Email</dt><dd>{user.email}</dd>
            <dt><ShieldCheck size={14} style={{ display: 'inline' }} /> Sign-in</dt><dd style={{ textTransform: 'capitalize' }}>{user.auth_provider}</dd>
            <dt>Member since</dt><dd>{fmtDate(user.created_at)}</dd>
          </dl>
        </div>
      </div>
    </>
  )
}
