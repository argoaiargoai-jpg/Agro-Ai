import { Check } from 'lucide-react'
import { STEPS, stepState } from '../lib/outcome'

/** The user's own image with a subtle scanning beam while the analysis runs. */
export function ScanPreview({ src, active, alt = 'Image being analyzed' }) {
  return (
    <div className={`scan${active ? ' on' : ''}`}>
      <img src={src} alt={alt} />
      {active && (<><span className="scan-grid" aria-hidden="true" /><span className="scan-beam" aria-hidden="true" /><span className="scan-corner tl" /><span className="scan-corner tr" /><span className="scan-corner bl" /><span className="scan-corner br" /></>)}
    </div>
  )
}

/** Step list driven by the real backend stage ('uploading' | 'ml' | 'guidance' | 'done'). */
export function StageList({ stage, title = 'AGRO AI ML MODEL' }) {
  return (
    <div className="stages" aria-live="polite">
      <div className="stages-title"><span className="pulse-dot" /> {title}</div>
      <ol>
        {STEPS.map((s, i) => {
          const st = stepState(stage, i)
          return (
            <li key={s.key} className={st}>
              <span className="stage-dot">{st === 'done' ? <Check size={13} strokeWidth={3} /> : st === 'now' ? <i /> : null}</span>
              <span><strong>{s.label}</strong>{st === 'now' && s.hint && <small>{s.hint}…</small>}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
