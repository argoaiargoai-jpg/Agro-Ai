import { STATUS } from '../lib/format'

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
