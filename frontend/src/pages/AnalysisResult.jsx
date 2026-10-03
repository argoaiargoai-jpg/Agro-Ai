import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { AlertTriangle, ArrowLeft, Check, Clock, Eye, HelpCircle, ImageOff, ShieldCheck, Sparkles, Sprout, Trash2, TriangleAlert } from 'lucide-react'
import AuthedImage from '../components/AuthedImage'
import { PageHead, StatusBadge } from '../components/AnalysisBits'
import { Alert, Button, EmptyState, ErrorState, Modal, Skeleton } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { useToast } from '../context/ToastContext'
import { api, ApiError, errorMessage } from '../lib/api'
import { fmtBytes, fmtDateTime } from '../lib/format'
import { useAsync } from '../lib/useAsync'

const TYPE = {
  DISEASE: { tone: 'warn', icon: AlertTriangle, title: 'Disease detected' },
  HEALTHY: { tone: 'ok', icon: Sprout, title: 'Looks healthy' },
  UNKNOWN: { tone: 'unk', icon: HelpCircle, title: 'Unknown or unsupported' },
  NO_PLANT: { tone: 'np', icon: ImageOff, title: 'No plant detected' },
}

/** The basic result shown when no detailed report exists (guidance off / unavailable). Same visual language, no internals. */
function BasicResult({ ml }) {
  const t = TYPE[ml.classification_type] || TYPE.UNKNOWN
  const Icon = t.icon
  const pct = Math.round(ml.confidence * 100)
  const label = ml.classification_type === 'NO_PLANT' ? 'Certainty it is not a plant' : 'Confidence'
  return (
    <div className={`card card-pad res res-${t.tone}`} data-testid="basic-result" data-type={ml.classification_type}>
      <div className="res-top"><span className="res-icon"><Icon size={22} /></span><div><span className="res-kicker">{t.title}</span>
        <h3 className="res-title">{ml.crop ? (ml.disease ? `${ml.crop} — ${ml.disease}` : `${ml.crop} — healthy`) : t.title}</h3></div></div>
      <p>{ml.message}</p>
      <div className="res-meter" aria-label={`${label} ${pct}%`}>
        <div className="res-meter-row"><span>{label}</span><strong>{pct}%</strong></div>
        <div className="res-bar"><i style={{ width: `${pct}%` }} /></div>
      </div>
      {(ml.classification_type === 'UNKNOWN' || ml.classification_type === 'NO_PLANT') && (
        <p className="small muted" style={{ marginTop: 12 }}>AGRO AI currently supports: {ml.supported_crops.join(', ')}. For best results, photograph a single leaf in good light.</p>
      )}
    </div>
  )
}

const SEVERITY = { none: 'None', mild: 'Mild', moderate: 'Moderate', severe: 'Severe', unknown: 'Unknown' }
const RISK = { low: 'Low', moderate: 'Moderate', high: 'High', unknown: 'Unknown' }
const FINAL = {
  DISEASE: { tone: 'warn', icon: AlertTriangle, kicker: 'Disease detected' },
  HEALTHY: { tone: 'ok', icon: Sprout, kicker: 'Looks healthy' },
  UNCERTAIN: { tone: 'unk', icon: HelpCircle, kicker: 'Uncertain result' },
  REJECTED: { tone: 'np', icon: ImageOff, kicker: 'Image not suitable' },
}

function Section({ title, items, tone, icon: Icon }) {
  if (!items || items.length === 0) return null
  return (
    <section className={`rep-sec${tone ? ` ${tone}` : ''}`}>
      <h4>{Icon && <Icon size={16} />}{title}</h4>
      <ul>{items.map((t, i) => <li key={i}>{t}</li>)}</ul>
    </section>
  )
}

function Report({ f }) {
  const t = FINAL[f.status] || FINAL.UNCERTAIN
  const Icon = t.icon
  const pct = f.affected_percentage
  return (
    <div className="report" data-testid="final-report" data-status={f.status}>
      <div className={`card card-pad res res-${t.tone}`}>
        <div className="res-top"><span className="res-icon"><Icon size={22} /></span>
          <div><span className="res-kicker">{t.kicker}</span><h3 className="res-title">{f.headline}</h3></div></div>
        {(f.plant || f.crop) && <p className="muted">Plant: <strong>{f.plant || f.crop}</strong>{f.crop && f.plant && f.crop !== f.plant ? ` · Crop: ${f.crop}` : ''}</p>}
        {f.rejection_reason && <p>{f.rejection_reason}</p>}
        {f.status !== 'REJECTED' && (
          <div className="rep-facts">
            {f.status !== 'UNCERTAIN' && f.severity && f.severity !== 'unknown' && <span className={`pill sev-${f.severity}`}>Severity: {SEVERITY[f.severity]}</span>}
            <span className="pill">{pct == null ? "Affected area: can't be estimated from this photo" : `Affected area: about ${Math.round(pct)}% (visual estimate)`}</span>
            {f.status !== 'UNCERTAIN' && f.identification_confidence && <span className="pill">Certainty: {f.identification_confidence}</span>}
          </div>
        )}
      </div>

      {f.status !== 'REJECTED' && (
        <div className="rep-grid">
          <Section title="Symptoms observed" items={f.symptoms} icon={Eye} />
          <Section title="Immediate actions" items={f.immediate_actions} icon={Check} />
          <Section title="Treatment options" items={f.treatment} />
          <Section title="Prevention" items={f.prevention} icon={ShieldCheck} />
          {f.spread_risk && (f.spread_risk.level !== 'unknown' || f.spread_risk.explanation) && (
            <section className="rep-sec"><h4>Spread risk: <span className={`pill risk-${f.spread_risk.level}`}>{RISK[f.spread_risk.level]}</span></h4>
              {f.spread_risk.explanation && <p className="small">{f.spread_risk.explanation}</p>}</section>
          )}
          <Section title="Warnings" items={f.warnings} tone="warn" icon={TriangleAlert} />
          <Section title="Monitoring & recovery" items={f.monitoring} />
          {f.ai_notes && <section className="rep-sec"><h4><Sparkles size={16} />Additional notes</h4><p className="small">{f.ai_notes}</p></section>}
        </div>
      )}
      <p className="small muted rep-foot">{f.disclaimer}</p>
    </div>
  )
}

function GuidanceBanner({ a, running, onRetry }) {
  const err = a.result?.ai_error
  if (!err) return null
  if (a.status === 'completed') {            // intentional ML-only state (AI off / not set up): informational, nothing to retry
    return <Alert tone="info" data-testid="ai-info">{err.message}</Alert>
  }
  const wait = err.retry_after ? ` Try again in about ${err.retry_after >= 120 ? `${Math.round(err.retry_after / 60)} minutes` : `${err.retry_after} seconds`}.` : ''
  return (
    <div className="card card-pad res res-wait" data-testid="ai-error" data-code={err.code}>
      <h3><AlertTriangle size={18} /> Detailed guidance isn't available right now</h3>
      <p>{err.message}{wait}</p>
      <p className="small muted">Your basic result above is still valid. {err.retryable ? 'You can try again for the detailed guidance.' : 'Trying again won’t help right now; please check back later.'}</p>
      {err.retryable && <Button size="sm" loading={running} onClick={onRetry}>Try again</Button>}
    </div>
  )
}

/** Admin-only: the internal ML / AI facts that customers never see (kept for debugging and auditing). */
function AdminInternals({ a }) {
  const r = a.result || {}
  const internal = {
    ml: r.ml, ai_status: a.ai_status, ai_provider: a.ai_provider, ai_model: a.ai_model, ai_error: r.ai_error, ai_analysis: r.ai,
    disagreement: r.final?.disagreement ?? null, disease_source: r.final?.disease_source ?? null, stage: r.stage, prompt_version: r.prompt_version,
    ml_completed_at: a.ml_completed_at, ai_completed_at: a.ai_completed_at,
  }
  return (
    <details className="card card-pad admin-internals" data-testid="admin-internals">
      <summary>Internal details (administrators only)</summary>
      <pre>{JSON.stringify(internal, null, 2)}</pre>
    </details>
  )
}

export default function AnalysisResult() {
  const { id } = useParams()
  const nav = useNavigate()
  const toast = useToast()
  const { isAdmin } = useAuth()
  const { data: a, loading, error, reload } = useAsync(() => api.get(`/analyses/${id}`), [id])
  const [confirm, setConfirm] = useState(false)
  const [busy, setBusy] = useState(false)
  const [running, setRunning] = useState(false)

  if (error) {
    const missing = error instanceof ApiError && error.status === 404
    return missing
      ? <div className="card"><EmptyState title="Analysis not found" action={<Link to="/history" className="btn btn-primary">Back to history</Link>}>It may have been deleted, or it belongs to another account.</EmptyState></div>
      : <div className="card"><ErrorState message={errorMessage(error)} onRetry={reload} /></div>
  }

  const ml = a?.result?.ml
  const final = a?.result?.final
  const mlOnly = a?.result?.stage === 'ml_only'
  const steps = [
    ['Image uploaded', !!a],
    ['Image analysis', !!ml],
    [mlOnly ? 'Agricultural guidance (not available)' : 'Agricultural guidance', !!final],
  ]

  async function retryGuidance() {            // no ?force: the stored ML result is kept, only the guidance step runs again
    setRunning(true)
    try { await api.post(`/analyses/${id}/analyze`); reload() }
    catch (e) { toast(errorMessage(e), 'error'); reload() }
    finally { setRunning(false) }
  }

  async function runAnalysis() {
    setRunning(true)
    try { await api.post(`/analyses/${id}/analyze?force=true`); reload() }
    catch (e) { toast(errorMessage(e), 'error'); reload() }
    finally { setRunning(false) }
  }

  async function del() {
    setBusy(true)
    try { await api.del(`/analyses/${id}`); toast('Analysis deleted.'); nav('/history', { replace: true }) }
    catch (e) { toast(errorMessage(e), 'error'); setBusy(false); setConfirm(false) }
  }

  return (
    <>
      <Link to="/history" className="small" style={{ display: 'inline-flex', gap: 6, alignItems: 'center', marginBottom: 14 }}><ArrowLeft size={15} /> History</Link>
      <PageHead title={a?.crop_type ? `${a.crop_type} analysis` : 'Crop analysis'} subtitle={a ? fmtDateTime(a.created_at) : ' '}>
        {a && <StatusBadge status={a.status} />}
        <Button variant="outline-danger" size="sm" icon={Trash2} onClick={() => setConfirm(true)} disabled={!a}>Delete</Button>
      </PageHead>

      <div className="grid grid-main">
        <div className="card card-pad">
          <div className="result-img">{loading ? <Skeleton h="100%" /> : <AuthedImage analysisId={id} alt="Analyzed crop" style={{ width: '100%', height: '100%' }} />}</div>
        </div>
        <div style={{ display: 'grid', gap: 18 }}>
          <div className="card card-pad">
            <h3 style={{ fontSize: 16, marginBottom: 16 }}>Progress</h3>
            <div className="timeline">
              {steps.map(([label, done]) => (
                <div key={label} className={`tl-item${done ? ' done' : ''}`}>
                  <div className={`tl-dot${done ? '' : ' todo'}`}>{done ? <Check size={15} /> : <Clock size={14} />}</div>
                  <div><strong>{label}</strong></div>
                </div>
              ))}
            </div>
          </div>

          {a && a.status === 'failed' && (
            <div className="card card-pad res res-err">
              <h3><AlertTriangle size={18} /> Analysis failed</h3>
              <p>{a.error_message || 'The analysis failed. Try uploading a clearer photo.'}</p>
              <Button variant="secondary" size="sm" loading={running} onClick={runAnalysis}>Try again</Button>
            </div>
          )}
          {a && a.status === 'uploaded' && (
            <div className="card card-pad res res-wait">
              <h3><Clock size={18} /> Not analyzed yet</h3>
              <p>Your image is saved. Run the analysis to check it.</p>
              <Button size="sm" loading={running} onClick={runAnalysis}>Analyze now</Button>
            </div>
          )}
          {ml && !final && <BasicResult ml={ml} />}
          {a && <GuidanceBanner a={a} running={running} onRetry={retryGuidance} />}

          <div className="card card-pad">
            <h3 style={{ fontSize: 16, marginBottom: 14 }}>Details</h3>
            {a ? (
              <dl className="kv">
                <dt>Crop</dt><dd>{a.crop_type || '—'}</dd>
                <dt>Source</dt><dd style={{ textTransform: 'capitalize' }}>{a.source}</dd>
                <dt>File</dt><dd>{a.original_filename || '—'} · {fmtBytes(a.size_bytes)}</dd>
                <dt>Notes</dt><dd>{a.notes || '—'}</dd>
              </dl>
            ) : <Skeleton h={90} />}
          </div>
        </div>
      </div>

      {final && <Report f={final} />}

      {isAdmin && a && <AdminInternals a={a} />}

      {confirm && (
        <Modal title="Delete this analysis?" onClose={() => setConfirm(false)}
          actions={<><Button variant="secondary" onClick={() => setConfirm(false)}>Cancel</Button><Button variant="danger" loading={busy} onClick={del}>Delete</Button></>}>
          <p className="muted">The photo and its results will be permanently removed.</p>
        </Modal>
      )}
    </>
  )
}
