import { STATUS } from '../lib/format'
import { describeAnalysis, OUTCOME } from '../lib/outcome'

export const StatusBadge = ({ status }) => {
  const s = STATUS[status] || { label: status, tone: 'gray' }
  return <span className={`badge ${s.tone}`}>{s.label}</span>
}

export const PageHead = ({ title, subtitle, children }) => (
  <div className="page-head">
    <div><h1>{title}</h1>{subtitle && <p>{subtitle}</p>}</div>
    {children && <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>{children}</div>}
  </div>
)

/** The customer-facing result tag for an analysis; before a result exists it falls back to the processing status. */
export const OutcomeTag = ({ analysis }) => {
  const { outcome } = describeAnalysis(analysis)
  if (!outcome) return <StatusBadge status={analysis.status} />
  const o = OUTCOME[outcome]
  return <span className={`otag ${o.tone}`} data-outcome={outcome}>{o.label}</span>
}
