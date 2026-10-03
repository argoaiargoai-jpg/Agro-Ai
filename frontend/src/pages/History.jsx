import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { History as HistoryIcon, Search } from 'lucide-react'
import AuthedImage from '../components/AuthedImage'
import { OutcomeTag, PageHead, StatusBadge } from '../components/AnalysisBits'
import { EmptyState, ErrorState, Pagination, Skeleton } from '../components/ui'
import { api, errorMessage } from '../lib/api'
import { fmtDateTime } from '../lib/format'
import { headline } from '../lib/outcome'
import { useAsync } from '../lib/useAsync'

const PAGE = 10

export default function History() {
  const [page, setPage] = useState(1)
  const [status, setStatus] = useState('')
  const [q, setQ] = useState('')
  const [dq, setDq] = useState('')
  useEffect(() => { const t = setTimeout(() => { setDq(q.trim()); setPage(1) }, 300); return () => clearTimeout(t) }, [q])

  const { data, loading, error, reload } = useAsync(() => {
    const p = new URLSearchParams({ page, page_size: PAGE })
    if (status) p.set('status', status)
    if (dq) p.set('q', dq)
    return api.get(`/analyses?${p}`)
  }, [page, status, dq])
  const filtered = !!(status || dq)

  return (
    <>
      <PageHead title="History" subtitle="Every crop scan you've submitted." />
      <div className="card">
        <div className="toolbar">
          <label className="search"><span className="sr-only">Search</span><Search size={17} /><input className="input" placeholder="Search by crop or filename" value={q} onChange={(e) => setQ(e.target.value)} /></label>
          <select className="select input" aria-label="Filter by status" value={status} onChange={(e) => { setStatus(e.target.value); setPage(1) }}>
            <option value="">All statuses</option>
            <option value="uploaded">Awaiting analysis</option>
            <option value="processing">Analyzing</option>
            <option value="partial">Guidance pending</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        {error ? <ErrorState message={errorMessage(error)} onRetry={reload} /> : loading && !data ? (
          <div style={{ padding: 22, display: 'grid', gap: 14 }}>{[0, 1, 2, 3].map((i) => <Skeleton key={i} h={52} />)}</div>
        ) : data.items.length === 0 ? (
          filtered
            ? <EmptyState icon={Search} title="No matching scans">Try a different search or clear the filters.</EmptyState>
            : <EmptyState icon={HistoryIcon} title="Nothing here yet" action={<Link to="/analyze" className="btn btn-primary">Analyze a crop</Link>}>Your scans will appear here once you submit a photo.</EmptyState>
        ) : (
          <>
            <div style={{ opacity: loading ? 0.6 : 1, transition: 'opacity .15s' }}>
              {data.items.map((a) => (
                <Link key={a.id} to={`/analysis/${a.id}`} className="hist-row">
                  <div className="thumb"><AuthedImage analysisId={a.id} alt="" style={{ width: '100%', height: '100%' }} /></div>
                  <div className="grow"><strong>{headline(a)}</strong><span className="small muted">{fmtDateTime(a.created_at)} · {a.source === 'camera' ? 'Camera' : 'Upload'}{a.crop_type ? ` · ${a.crop_type}` : ''}</span></div>
                  <div className="hist-tags"><OutcomeTag analysis={a} />{a.result && a.status !== 'completed' && <StatusBadge status={a.status} />}</div>
                </Link>
              ))}
            </div>
            <Pagination page={page} pageSize={PAGE} total={data.total} onChange={setPage} />
          </>
        )}
      </div>
    </>
  )
}
