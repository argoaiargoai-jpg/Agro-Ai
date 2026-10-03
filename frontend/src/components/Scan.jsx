import { AlertTriangle, Check } from 'lucide-react'
import { stepStates } from '../lib/outcome'

/** The user's own image with a scanning beam, animated corner brackets and a soft pulse while the analysis runs. */
export function ScanPreview({ src, active, alt = 'Image being analyzed' }) {
  return (
    <div className={`scan${active ? ' on' : ''}`}>
      <img src={src} alt={alt} />
      {active && (
        <>
          <span className="scan-grid" aria-hidden="true" /><span className="scan-glow" aria-hidden="true" /><span className="scan-beam" aria-hidden="true" />
          <span className="scan-particles" aria-hidden="true"><i /><i /><i /><i /><i /></span>
          <span className="scan-corner tl" /><span className="scan-corner tr" /><span className="scan-corner bl" /><span className="scan-corner br" />
          <span className="scan-chip" role="status"><span className="pulse-dot" /> Analyzing image…</span>
        </>
      )}
    </div>
  )
}

/** Step list from `steps` (buildSteps) and `stage` (what the backend reports). States: pending, active, completed, failed. */
export function StageList({ steps, stage, failedAt = null, title = 'AGRO AI analysis', compact = false, onRetry = null }) {
  const states = stepStates(steps, stage, failedAt)
  const done = states.filter((s) => s === 'done').length
  return (
    <div className={`stages${compact ? ' compact' : ''}`} aria-live="polite">
      <div className="stages-title"><span className="pulse-dot" /> {title}</div>
      <div className="stages-bar" role="progressbar" aria-valuemin={0} aria-valuemax={steps.length} aria-valuenow={done} aria-label="Analysis progress"><i style={{ width: `${(done / steps.length) * 100}%` }} /></div>
      <ol>
        {steps.map((s, i) => {
          const st = states[i]
          return (
            <li key={s.key} className={st}>
              <span className="stage-dot">{st === 'done' ? <Check size={13} strokeWidth={3} /> : st === 'failed' ? <AlertTriangle size={12} strokeWidth={3} /> : st === 'now' ? <i /> : null}</span>
              <span>
                <strong>{s.label}</strong>
                {st === 'now' && s.hint && <small>{s.hint}…</small>}
                {st === 'failed' && <small>Temporarily unavailable{onRetry ? ' · ' : ''}{onRetry && <button type="button" className="linklike" onClick={onRetry}>Retry</button>}</small>}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
