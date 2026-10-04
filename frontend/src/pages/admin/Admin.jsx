import { useEffect, useState } from 'react'
import { Activity, Bot, Cpu, Database, CheckCircle2, Image as ImageIcon, KeyRound, ScrollText, Search, SlidersHorizontal, UserCheck, Users, XCircle } from 'lucide-react'
import { PageHead } from '../../components/AnalysisBits'
import { Alert, Button, EmptyState, ErrorState, Field, Modal, Pagination, Skeleton, Switch } from '../../components/ui'
import { CountUp } from '../../components/Reveal'
import { Columns, Donut } from '../../components/charts'
import { OUTCOME } from '../../lib/outcome'
import { useAuth } from '../../context/AuthContext'
import { useConfig } from '../../context/ConfigContext'
import { useToast } from '../../context/ToastContext'
import { api, ApiError, errorMessage } from '../../lib/api'
import { fmtDate, fmtDateTime, initials } from '../../lib/format'
import { useAsync } from '../../lib/useAsync'


const OUTCOME_COLORS = { healthy: '#16a34a', disease: '#f59e0b', unresolved: '#94a3b8', no_plant: '#a78bfa' }
const AI_BADGE = { ready: ['ok', 'Ready'], not_configured: ['warn', 'Not configured'], disabled: ['gray', 'Disabled'], unknown_provider: ['err', 'Unknown provider'] }

function SystemStatus() {
  const health = useAsync(() => api.get('/health'))
  const ml = useAsync(() => api.get('/ml/info'))
  const ai = useAsync(() => api.get('/admin/ai'))
  const [aiTone, aiLabel] = AI_BADGE[ai.data?.state] || ['gray', ai.loading ? '…' : 'Unknown']
  const items = [
    { icon: Database, name: 'Backend & database', tone: health.data?.database === 'ok' ? 'ok' : health.error ? 'err' : 'gray', label: health.data ? (health.data.database === 'ok' ? 'Healthy' : 'Database issue') : health.error ? 'Unreachable' : '…',
      note: health.data ? `${health.data.environment} environment` : '' },
    { icon: Cpu, name: 'AGRO AI ML model', tone: ml.data?.available ? 'ok' : ml.error ? 'err' : 'gray', label: ml.data ? (ml.data.available ? 'Installed / Ready' : 'Not installed') : ml.error ? 'Unreachable' : '…',
      note: ml.data?.available ? `${ml.data.model.split(' (')[0]} · version ${ml.data.version} · ${ml.data.supported_crops.length} crops` : '' },
    { icon: Bot, name: 'AI provider', tone: aiTone, label: aiLabel,
      note: ai.data ? `${ai.data.active_provider} · key ${ai.data.providers?.find((p) => p.id === ai.data.active_provider || p.name === ai.data.active_provider)?.configured ? 'configured (hidden)' : 'not set'} · ${Object.entries(ai.data.usage_24h || {}).map(([k, v]) => `${v} ${k}`).join(', ') || 'no calls in 24h'}` : '' },
  ]
  return (
    <div className="grid grid-3" style={{ marginBottom: 18 }} data-testid="system-status">
      {items.map(({ icon: I, name, tone, label, note }) => (
        <div key={name} className="card card-pad">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}><span className="stat-icon" style={{ margin: 0 }}><I size={18} /></span><strong>{name}</strong></div>
          <span className={`badge ${tone}`}>{label}</span>
          {note && <p className="small muted" style={{ marginTop: 8 }}>{note}</p>}
        </div>
      ))}
    </div>
  )
}

function Overview() {
  const { data: s, loading, error, reload } = useAsync(() => api.get('/admin/stats'))
  if (error) return <div className="card"><ErrorState message={errorMessage(error)} onRetry={reload} /></div>
  const cards = [
    { icon: Users, label: 'Total users', value: s?.users_total },
    { icon: UserCheck, label: 'Verified users', value: s?.users_verified },
    { icon: Activity, label: 'Active (7 days)', value: s?.users_active_7d },
    { icon: ImageIcon, label: 'Total scans', value: s?.analyses_total },
  ]
  const days = Array.from({ length: 14 }, (_, i) => new Date(Date.now() - (13 - i) * 864e5).toISOString().slice(0, 10))
  const counts = Object.fromEntries((s?.signups_14d || []).map((d) => [d.date, d.count]))
  const series = days.map((date) => ({ date, count: counts[date] || 0 }))
  const max = Math.max(1, ...series.map((d) => d.count))
  const total = series.reduce((n, d) => n + d.count, 0)
  const o = s?.outcomes || {}
  const analysed = Object.values(o).reduce((n, v) => n + v, 0)
  const analysesDay = s?.activity_14d || []
  return (
    <>
      <SystemStatus />
      <div className="grid grid-4" style={{ marginBottom: 18 }}>
        {cards.map(({ icon: I, label, value }) => (
          <div key={label} className="card stat"><div className="stat-icon"><I size={20} /></div>
            {loading ? <Skeleton h={34} w={60} /> : <div className="stat-value"><CountUp value={value} /></div>}<div className="stat-label">{label}</div></div>
        ))}
      </div>
      <div className="grid-2c">
        <div className="card chart-card">
          <h3>Analyses, last 14 days</h3><div className="sub">All users · {s?.ai_assisted ?? 0} AI-assisted overall</div>
          {loading ? <Skeleton h={140} /> : analysesDay.every((d) => d.count === 0) ? <div className="chart-empty">No analyses in this period.</div> : <Columns series={analysesDay} label="Analyses per day" />}
        </div>
        <div className="card chart-card">
          <h3>Outcomes</h3><div className="sub">All analyses with a result</div>
          {loading ? <Skeleton h={160} /> : analysed === 0 ? <div className="chart-empty">No results yet.</div>
            : <Donut size={140} stroke={18} data={Object.keys(OUTCOME).map((k) => ({ key: k, label: OUTCOME[k].label, value: o[k] || 0, color: OUTCOME_COLORS[k] }))} centerValue={analysed} centerLabel="analysed" />}
        </div>
      </div>
      <div className="card card-pad">
        <h3 style={{ marginBottom: 16 }}>Sign-ups, last 14 days</h3>
        {loading ? <Skeleton h={140} /> : (
          <>
            <div className="bars" role="img" aria-label={`Sign-ups per day, ${total} in total`}>
              {series.map((d) => (
                <div key={d.date} className="bar-col" title={`${d.date}: ${d.count}`}>
                  <span className="bar-n">{d.count || ''}</span>
                  <i style={{ height: d.count ? `${Math.max((d.count / max) * 100, 6)}%` : '3px', opacity: d.count ? 1 : .35 }} />
                  <span className="bar-d">{d.date.slice(8)}</span>
                </div>
              ))}
            </div>
            {total === 0 && <p className="muted small" style={{ marginTop: 10 }}>No sign-ups in this period.</p>}
          </>
        )}
      </div>
    </>
  )
}

function UsersTab() {
  const { user: me } = useAuth()
  const toast = useToast()
  const [page, setPage] = useState(1)
  const [q, setQ] = useState('')
  const [dq, setDq] = useState('')
  const [role, setRole] = useState('')
  const [target, setTarget] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { const t = setTimeout(() => { setDq(q.trim()); setPage(1) }, 300); return () => clearTimeout(t) }, [q])
  const { data, loading, error, reload } = useAsync(() => {
    const p = new URLSearchParams({ page, page_size: 10 })
    if (dq) p.set('q', dq); if (role) p.set('role', role)
    return api.get(`/admin/users?${p}`)
  }, [page, dq, role])

  async function apply(patch) {
    setBusy(true)
    try { await api.patch(`/admin/users/${target.id}`, patch); toast('User updated.'); setTarget(null); reload() }
    catch (e) { toast(errorMessage(e), 'error') } finally { setBusy(false) }
  }

  return (
    <div className="card">
      <div className="toolbar">
        <label className="search"><span className="sr-only">Search users</span><Search size={17} /><input className="input" placeholder="Search name or email" value={q} onChange={(e) => setQ(e.target.value)} /></label>
        <select className="select input" aria-label="Filter by role" value={role} onChange={(e) => { setRole(e.target.value); setPage(1) }}>
          <option value="">All roles</option><option value="user">Members</option><option value="admin">Admins</option>
        </select>
      </div>
      {error ? <ErrorState message={errorMessage(error)} onRetry={reload} /> : loading && !data ? <div style={{ padding: 22 }}><Skeleton h={200} /></div>
        : data.items.length === 0 ? <EmptyState icon={Users} title="No users found">Try a different search.</EmptyState> : (
          <>
            <div className="table-wrap" style={{ opacity: loading ? .6 : 1 }}>
              <table className="table">
                <thead><tr><th>User</th><th>Role</th><th>Status</th><th>Joined</th><th>Last sign-in</th><th /></tr></thead>
                <tbody>
                  {data.items.map((u) => (
                    <tr key={u.id}>
                      <td><div style={{ display: 'flex', gap: 12, alignItems: 'center' }}><div className="avatar sm">{initials(u.full_name)}</div>
                        <div style={{ minWidth: 0 }}><strong>{u.full_name}</strong><div className="small muted">{u.email}</div></div></div></td>
                      <td><span className={`badge ${u.role === 'admin' ? 'dark' : 'gray'}`}>{u.role}</span></td>
                      <td>{!u.is_active ? <span className="badge err">Disabled</span> : u.is_verified ? <span className="badge ok">Active</span> : <span className="badge warn">Unverified</span>}</td>
                      <td>{fmtDate(u.created_at)}</td><td>{fmtDateTime(u.last_login_at)}</td>
                      <td style={{ textAlign: 'right' }}>{u.id === me.id ? <span className="small muted">You</span> : <Button variant="secondary" size="sm" onClick={() => setTarget(u)}>Manage</Button>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} pageSize={10} total={data.total} onChange={setPage} />
          </>
        )}
      {target && (
        <Modal title={`Manage ${target.full_name}`} onClose={() => setTarget(null)} actions={<Button variant="secondary" onClick={() => setTarget(null)}>Close</Button>}>
          <p className="small muted">{target.email}</p>
          <div className="setting-row"><div className="grow"><strong>Administrator</strong><span className="small muted">Full access to this console</span></div>
            <Switch label="Administrator" checked={target.role === 'admin'} disabled={busy} onChange={(v) => apply({ role: v ? 'admin' : 'user' })} /></div>
          <div className="setting-row"><div className="grow"><strong>Account active</strong><span className="small muted">Disabled users are signed out and blocked</span></div>
            <Switch label="Account active" checked={target.is_active} disabled={busy} onChange={(v) => apply({ is_active: v })} /></div>
        </Modal>
      )}
    </div>
  )
}

function SystemTab() {
  const toast = useToast()
  const { reload: reloadPublic } = useConfig()
  const { data, loading, error, reload } = useAsync(() => api.get('/admin/settings'))
  const [f, setF] = useState(null)
  const [errs, setErrs] = useState({})
  const [busy, setBusy] = useState(false)
  useEffect(() => { if (data) setF({ ...data, supported_crops: data.supported_crops.join(', ') }) }, [data])
  if (error) return <div className="card"><ErrorState message={errorMessage(error)} onRetry={reload} /></div>
  if (loading || !f) return <div className="card card-pad"><Skeleton h={220} /></div>

  async function save(e) {
    e.preventDefault(); setBusy(true); setErrs({})
    const crops = f.supported_crops.split(',').map((c) => c.trim()).filter(Boolean)
    try {
      await api.put('/admin/settings', { values: { ...f, max_upload_mb: Number(f.max_upload_mb), supported_crops: crops } })
      toast('Settings saved.'); reloadPublic(); reload()
    } catch (err) { if (err instanceof ApiError && err.fields) setErrs(err.fields); toast(errorMessage(err), 'error') } finally { setBusy(false) }
  }

  return (
    <form className="card card-pad form" onSubmit={save} noValidate>
      <div className="setting-row" style={{ paddingTop: 0 }}><div className="grow"><strong>Open registration</strong><span className="small muted">Allow new users to create accounts</span></div>
        <Switch label="Open registration" checked={f.registration_enabled} onChange={(v) => setF({ ...f, registration_enabled: v })} /></div>
      <div className="setting-row"><div className="grow"><strong>Maintenance mode</strong><span className="small muted">Pause new uploads and show a banner</span></div>
        <Switch label="Maintenance mode" checked={f.maintenance_mode} onChange={(v) => setF({ ...f, maintenance_mode: v })} /></div>
      <Field label="Announcement banner" maxLength={300} value={f.announcement} onChange={(e) => setF({ ...f, announcement: e.target.value })} error={errs.announcement} hint="Shown to every signed-in user. Leave empty to hide." />
      <Field label="Max upload size (MB)" type="number" min={1} max={25} value={f.max_upload_mb} onChange={(e) => setF({ ...f, max_upload_mb: e.target.value })} error={errs.max_upload_mb} />
      <Field label="Crop suggestions (legacy)" value={f.supported_crops} onChange={(e) => setF({ ...f, supported_crops: e.target.value })} error={errs.supported_crops} hint="Legacy list. It no longer limits what customers can analyze." />
      <div><Button type="submit" loading={busy}>Save settings</Button></div>
    </form>
  )
}

function AuditTab() {
  const { data, loading, error, reload } = useAsync(() => api.get('/admin/audit-log?limit=50'))
  return (
    <div className="card">
      {error ? <ErrorState message={errorMessage(error)} onRetry={reload} /> : loading ? <div style={{ padding: 22 }}><Skeleton h={160} /></div>
        : data.length === 0 ? <EmptyState icon={ScrollText} title="No activity yet">Admin actions will be recorded here.</EmptyState> : (
          <div className="table-wrap"><table className="table">
            <thead><tr><th>When</th><th>Actor</th><th>Action</th><th>Details</th></tr></thead>
            <tbody>{data.map((l) => (
              <tr key={l.id}><td>{fmtDateTime(l.created_at)}</td><td>{l.actor || '—'}</td><td><span className="badge gray">{l.action}</span></td>
                <td className="small muted" style={{ maxWidth: 360, wordBreak: 'break-word' }}>{l.meta ? JSON.stringify(l.meta) : '—'}</td></tr>
            ))}</tbody>
          </table></div>
        )}
    </div>
  )
}

const AI_STATE = {
  ready: { tone: 'ok', label: 'Ready', text: 'The active provider has an API key and AI guidance is on.' },
  not_configured: { tone: 'warn', label: 'Not configured', text: 'No API key is set on the server for the active provider. Analyses will show only our model\'s result.' },
  disabled: { tone: 'gray', label: 'Disabled', text: 'AI guidance is switched off. Analyses show only our model\'s result.' },
  unknown_provider: { tone: 'err', label: 'Unknown provider', text: 'The saved provider is not available in this version. Pick another one.' },
}
const TEST_TEXT = { ready: 'Connection works', not_configured: 'No API key configured', timeout: 'Timed out', rate_limited: 'Rate limited', unavailable: 'Provider unavailable',
  misconfigured: 'Key or model rejected', bad_response: 'Unexpected answer', blocked: 'Blocked by provider' }

function AITab() {
  const toast = useToast()
  const { data, loading, error, reload } = useAsync(() => api.get('/admin/ai'))
  const [busy, setBusy] = useState(false)
  const [testing, setTesting] = useState(false)
  const [test, setTest] = useState(null)
  const [confirmTest, setConfirmTest] = useState(false)
  if (error) return <div className="card"><ErrorState message={errorMessage(error)} onRetry={reload} /></div>
  if (loading && !data) return <div className="card card-pad"><Skeleton h={260} /></div>
  const st = AI_STATE[data.state] || AI_STATE.not_configured

  async function save(patch) {
    setBusy(true)
    try { await api.put('/admin/ai', patch); toast('AI settings saved.'); setTest(null); reload() }
    catch (e) { toast(errorMessage(e), 'error') } finally { setBusy(false) }
  }
  async function runTest() {
    setTesting(true); setTest(null)
    try { setTest(await api.post('/admin/ai/test', { provider: data.active_provider })) }
    catch (e) { setTest({ ok: false, state: 'error', message: errorMessage(e) }) } finally { setTesting(false) }
  }
  const L = data.limits
  return (
    <div style={{ display: 'grid', gap: 18 }} data-testid="ai-tab">
      <div className="card card-pad form">
        <div className="setting-row" style={{ paddingTop: 0 }}>
          <div className="grow"><strong>AI guidance</strong><span className="small muted">After our model runs, the original image is always sent to the selected AI provider, which inspects it independently (our model's result is only a hint) and writes the explanation and advice.</span></div>
          <span className={`badge ${st.tone}`} data-testid="ai-state">{st.label}</span>
          <Switch label="Enable AI guidance" checked={data.enabled} disabled={busy} onChange={(v) => save({ enabled: v })} />
        </div>
        <Alert tone={st.tone === 'ok' ? 'success' : st.tone === 'err' ? 'error' : 'info'}>{st.text}</Alert>
        <div className="field">
          <label htmlFor="ai-provider">Active provider</label>
          <select id="ai-provider" className="select input" value={data.active_provider} disabled={busy} onChange={(e) => save({ provider: e.target.value })}>
            {data.providers.map((p) => <option key={p.name} value={p.name}>{p.display_name}{p.configured ? '' : ' (no key)'}</option>)}
          </select>
        </div>
      </div>

      <div className="card card-pad" style={{ display: 'grid', gap: 12 }}>
        <h3 style={{ fontSize: 16 }}>Providers</h3>
        {data.providers.map((p) => (
          <div key={p.name} className={`prov${p.name === data.active_provider ? ' active' : ''}`}>
            <Bot size={22} />
            <div className="grow"><strong>{p.display_name}</strong><span className="small muted">Model: {p.model}</span></div>
            <span className={`badge ${p.configured ? 'ok' : 'warn'}`}>{p.configured ? <><KeyRound size={13} /> Key set on server</> : 'No key on server'}</span>
            {p.name === data.active_provider && <span className="badge dark">Active</span>}
          </div>
        ))}
        <p className="small muted">API keys are read from the server environment (for Gemini: <code>GEMINI_API_KEY_1</code> and <code>GEMINI_API_KEY_2</code>). They are never stored in the database, returned by the API, or shown here.</p>
        <div><Button variant="secondary" size="sm" loading={testing} onClick={runTest}>Test connection</Button></div>
        {test && (
          <Alert tone={test.ok ? 'success' : 'error'} data-testid="ai-test-result">
            <strong>{test.ok ? <>Connection works</> : (TEST_TEXT[test.state] || 'Test failed')}</strong>
            <div className="small" style={{ marginTop: 4 }}>{test.message}{test.latency_ms ? ` · ${test.latency_ms} ms` : ''}{test.model ? ` · ${test.model}` : ''}</div>
            {test.detail && <div className="small muted" style={{ marginTop: 4 }}>Detail: {test.detail}</div>}
          </Alert>
        )}
      </div>

      <div className="card card-pad" style={{ display: 'grid', gap: 12, borderLeft: '4px solid var(--warning)', background: data.openrouter_test_mode ? 'var(--warning-bg)' : undefined }} data-testid="openrouter-test-mode">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span className="badge warn">TEST / ADMIN ONLY</span>
          <h3 style={{ fontSize: 16, margin: 0 }}>AI Provider Testing</h3>
        </div>
        <div className="setting-row" style={{ paddingTop: 0 }}>
          <div className="grow"><strong>OpenRouter Test Mode</strong>
            <span className="small muted">⚠ When enabled, Gemini will be bypassed for testing and the AI enrichment request will be sent directly to OpenRouter. The model and specialists still run first. Turn it off again after testing.</span></div>
          <Switch label="OpenRouter Test Mode" checked={!!data.openrouter_test_mode} disabled={busy}
            onChange={(v) => (v ? setConfirmTest(true) : save({ openrouter_test_mode: false }))} />
        </div>
        <div data-testid="openrouter-test-status" style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span className={`badge ${data.openrouter_test_mode ? 'warn' : 'ok'}`}>{data.openrouter_test_mode ? '● OpenRouter Test Mode Active' : '● Normal Mode'}</span>
          <span className="small muted">OpenRouter key: {data.openrouter?.configured ? 'set on server' : 'not set on server'} · model {data.openrouter?.model}</span>
        </div>
        {data.openrouter_test_mode && !data.openrouter?.configured && <Alert tone="warn">Test mode is on but no OpenRouter key is set on the server, so every analysis will fall back to the specialist result.</Alert>}
      </div>
      {confirmTest && (
        <Modal title="Enable OpenRouter Test Mode?" onClose={() => setConfirmTest(false)}
          actions={<><Button variant="secondary" onClick={() => setConfirmTest(false)}>Cancel</Button><Button loading={busy} onClick={async () => { await save({ openrouter_test_mode: true }); setConfirmTest(false) }}>Enable test mode</Button></>}>
          <p className="muted">Gemini will be bypassed for every new analysis until you turn this off. Real users' analyses will use the free OpenRouter route instead. Use it for a short test only.</p>
        </Modal>
      )}

      <div className="card card-pad" style={{ display: 'grid', gap: 12 }}>
        <h3 style={{ fontSize: 16 }}>Safeguards</h3>
        <div className="limit-grid">
          <div><b>{L.timeout_seconds}s</b><span className="small muted">Request timeout</span></div>
          <div><b>{L.max_retries}</b><span className="small muted">Retries (timeouts/5xx only)</span></div>
          <div><b>{L.max_concurrency}</b><span className="small muted">Simultaneous calls</span></div>
          <div><b>{L.user_hourly_limit}/h</b><span className="small muted">Per user</span></div>
        </div>
        <h3 style={{ fontSize: 16, marginTop: 6 }}>Last 24 hours</h3>
        {Object.keys(data.usage_24h).length === 0 ? <p className="small muted">No provider calls yet.</p> : (
          <div className="chips">{Object.entries(data.usage_24h).map(([k, v]) => <span key={k} className={`badge ${k === 'completed' ? 'ok' : 'warn'}`}>{k.replace('_', ' ')}: {v}</span>)}</div>
        )}
      </div>
    </div>
  )
}

const TABS = [
  ['overview', 'Overview', Activity, Overview],
  ['users', 'Users', Users, UsersTab],
  ['ai', 'AI provider', Bot, AITab],
  ['system', 'System settings', SlidersHorizontal, SystemTab],
  ['audit', 'Audit log', ScrollText, AuditTab],
]

export default function Admin() {
  const [tab, setTab] = useState('overview')
  const Active = TABS.find((t) => t[0] === tab)[3]
  return (
    <>
      <PageHead title="Admin console" subtitle="Manage users and platform configuration." />
      <div className="tabs" role="tablist" style={{ marginBottom: 20, maxWidth: '100%', overflowX: 'auto' }}>
        {TABS.map(([k, label, I]) => <button key={k} role="tab" className="tab" aria-selected={tab === k} onClick={() => setTab(k)}><I size={16} />{label}</button>)}
      </div>
      <Active />
    </>
  )
}
