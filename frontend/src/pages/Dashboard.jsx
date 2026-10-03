import { Link } from 'react-router-dom'
import { CheckCircle2, Clock, Image as ImageIcon, ScanLine, Sprout } from 'lucide-react'
import AuthedImage from '../components/AuthedImage'
import { PageHead, StatusBadge } from '../components/AnalysisBits'
import { Button, EmptyState, ErrorState, Skeleton } from '../components/ui'
import { CountUp } from '../components/Reveal'
import { useAuth } from '../context/AuthContext'
import { api, errorMessage } from '../lib/api'
import { fmtDateTime } from '../lib/format'
import { useAsync } from '../lib/useAsync'

export default function Dashboard() {
  const { user } = useAuth()
  const { data, loading, error, reload } = useAsync(() => api.get('/analyses/summary'))
  const by = data?.by_status || {}
  const stats = [
    { icon: ImageIcon, label: 'Total scans', value: data?.total },
    { icon: Clock, label: 'Awaiting analysis', value: (by.uploaded || 0) + (by.queued || 0) + (by.processing || 0) + (by.partial || 0) },
    { icon: CheckCircle2, label: 'Completed', value: by.completed || 0 },
  ]
  return (
    <>
      <PageHead title={`Hello, ${user.full_name.split(' ')[0]} 👋`} subtitle="Here's what's happening across your fields.">
        <Link to="/analyze" className="btn btn-primary"><ScanLine size={18} /> New analysis</Link>
      </PageHead>

      {error ? <div className="card"><ErrorState message={errorMessage(error)} onRetry={reload} /></div> : (
        <>
          <div className="grid grid-3" style={{ marginBottom: 22 }}>
            {stats.map(({ icon: I, label, value }) => (
              <div key={label} className="card stat">
                <div className="stat-icon"><I size={20} /></div>
                {loading ? <Skeleton h={34} w={70} /> : <div className="stat-value"><CountUp value={value} /></div>}
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>
          <div className="card">
            <div className="card-head"><h3>Recent scans</h3><Link to="/history" className="small">View all</Link></div>
            {loading ? (
              <div style={{ padding: 22, display: 'grid', gap: 14 }}>{[0, 1, 2].map((i) => <Skeleton key={i} h={52} />)}</div>
            ) : data.recent.length === 0 ? (
              <EmptyState icon={Sprout} title="No scans yet" action={<Link to="/analyze" className="btn btn-primary">Analyze your first crop</Link>}>
                Upload or capture a photo of a leaf to start building your crop history.
              </EmptyState>
            ) : data.recent.map((a) => (
              <Link key={a.id} to={`/analysis/${a.id}`} className="row-link">
                <div className="thumb"><AuthedImage analysisId={a.id} alt="" style={{ width: '100%', height: '100%' }} /></div>
                <div className="grow"><strong>{a.crop_type || 'Unspecified crop'}</strong><span className="small muted">{fmtDateTime(a.created_at)}</span></div>
                <StatusBadge status={a.status} />
              </Link>
            ))}
          </div>
        </>
      )}
    </>
  )
}
