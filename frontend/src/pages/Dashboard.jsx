import { Link } from 'react-router-dom'
import { Activity, AlertTriangle, BarChart3, Cpu, Image as ImageIcon, Leaf, ScanLine, Sparkles, Sprout, Stethoscope } from 'lucide-react'
import AuthedImage from '../components/AuthedImage'
import { OutcomeTag, PageHead } from '../components/AnalysisBits'
import { Columns, Donut, HBars } from '../components/charts'
import { EmptyState, ErrorState, Skeleton } from '../components/ui'
import { CountUp } from '../components/Reveal'
import { useAuth } from '../context/AuthContext'
import { api, errorMessage } from '../lib/api'
import { fmtDateTime } from '../lib/format'
import { headline, OUTCOME } from '../lib/outcome'
import { useAsync } from '../lib/useAsync'

const COLORS = { healthy: '#16a34a', disease: '#f59e0b', unresolved: '#94a3b8', no_plant: '#a78bfa' }

function ModelCard() {
  const { data: info, loading } = useAsync(() => api.get('/ml/info'))
  const input = info?.model?.match(/(\d+)\s*x\s*(\d+)/i)
  const arch = info?.model?.split(' (')[0]
  return (
    <div className="card card-pad model-card" data-testid="model-card">
      <h3><Cpu size={16} /> AGRO AI ML Model</h3>
      {loading ? <Skeleton h={110} style={{ opacity: 0.25 }} /> : (
        <dl>
          <dt>Architecture</dt><dd>{arch || 'MobileNetV3-Small'}</dd>
          <dt>Input</dt><dd>{input ? `${input[1]} × ${input[2]}` : '224 × 224'}</dd>
          <dt>Inference</dt><dd>ONNX Runtime</dd>
          <dt>Trained on</dt><dd>{info?.supported_crops ? `${info.supported_crops.length} crops` : '—'}</dd>
          <dt>Status</dt><dd className="ready">{info?.available ? <><span className="pulse-dot" /> Ready</> : 'Unavailable'}</dd>
        </dl>
      )}
      <p className="fine">ML-powered image analysis, extended by agricultural AI guidance for a broader range of plants. Results are guidance, not a guarantee.</p>
    </div>
  )
}

function ChartEmpty({ icon: I = BarChart3, children }) {
  return <div className="chart-empty"><I size={26} />{children}</div>
}

export default function Dashboard() {
  const { user } = useAuth()
  const { data, loading, error, reload } = useAsync(() => api.get('/analyses/summary'))
  const o = data?.outcomes || { healthy: 0, disease: 0, unresolved: 0, no_plant: 0 }
  const analysed = o.healthy + o.disease + o.unresolved + o.no_plant
  const stats = [
    { icon: ImageIcon, label: 'Total analyses', value: data?.total, cls: '' },
    { icon: Sprout, label: 'Healthy plants', value: o.healthy, cls: '' },
    { icon: AlertTriangle, label: 'Diseases detected', value: o.disease, cls: 'warn' },
    { icon: Sparkles, label: 'AI-assisted analyses', value: data?.ai_assisted, cls: 'info' },
  ]
  const donut = Object.keys(OUTCOME).map((k) => ({ key: k, label: OUTCOME[k].label, value: o[k], color: COLORS[k] }))
  const activityTotal = (data?.activity_14d || []).reduce((n, d) => n + d.count, 0)

  return (
    <>
      <PageHead title={`Hello, ${user.full_name.split(' ')[0]} 👋`} subtitle="Your plant-health overview, built from your own analyses.">
        <Link to="/analyze" className="btn btn-primary"><ScanLine size={18} /> New analysis</Link>
      </PageHead>

      {error ? <div className="card"><ErrorState message={errorMessage(error)} onRetry={reload} /></div> : (
        <>
          <div className="grid-4s">
            {stats.map(({ icon: I, label, value, cls }) => (
              <div key={label} className={`card stat stat-tint ${cls}`}>
                <div className="stat-icon"><I size={20} /></div>
                {loading ? <Skeleton h={34} w={70} /> : <div className="stat-value"><CountUp value={value ?? 0} /></div>}
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>

          {!loading && data.total === 0 ? (
            <div className="card" style={{ marginBottom: 18 }}>
              <EmptyState icon={Leaf} title="No analyses yet" action={<Link to="/analyze" className="btn btn-primary"><ScanLine size={18} /> Analyze your first plant</Link>}>
                Upload or capture a clear photo of a leaf. Your charts, history and plant-health trends will appear here as you analyze.
              </EmptyState>
            </div>
          ) : (
            <>
              <div className="grid-2c">
                <div className="card chart-card">
                  <h3>Analysis activity</h3>
                  <div className="sub">Analyses per day, last 14 days</div>
                  {loading ? <Skeleton h={150} /> : activityTotal === 0
                    ? <ChartEmpty icon={Activity}>No analyses in the last 14 days.</ChartEmpty>
                    : <Columns series={data.activity_14d} label="Analyses per day" />}
                </div>
                <div className="card chart-card">
                  <h3>Results overview</h3>
                  <div className="sub">How your analyses turned out</div>
                  {loading ? <Skeleton h={168} /> : analysed === 0
                    ? <ChartEmpty icon={BarChart3}>Results will show here once an analysis completes.</ChartEmpty>
                    : <Donut data={donut} centerValue={analysed} centerLabel="analysed" />}
                </div>
              </div>

              <div className="grid-2e">
                <div className="card chart-card">
                  <h3>Disease distribution</h3>
                  <div className="sub">Most frequent conditions detected</div>
                  {loading ? <Skeleton h={120} /> : data.top_diseases.length === 0
                    ? <ChartEmpty icon={Stethoscope}>No diseases detected yet. 🌱</ChartEmpty>
                    : <HBars items={data.top_diseases} color="#f59e0b" />}
                </div>
                <div className="card chart-card">
                  <h3>Plants &amp; crops</h3>
                  <div className="sub">Plants identified in your analyses</div>
                  {loading ? <Skeleton h={120} /> : data.crops.length === 0
                    ? <ChartEmpty icon={Sprout}>No plants identified yet.</ChartEmpty>
                    : <HBars items={data.crops} />}
                </div>
              </div>
            </>
          )}

          <div className="grid-2c">
            <div className="card" style={{ overflow: 'hidden' }}>
              <div className="card-head"><h3>Recent activity</h3><Link to="/history" className="small">View all</Link></div>
              {loading ? (
                <div style={{ padding: 22, display: 'grid', gap: 14 }}>{[0, 1, 2].map((i) => <Skeleton key={i} h={52} />)}</div>
              ) : data.recent.length === 0 ? (
                <ChartEmpty icon={Sprout}>Nothing here yet. Your latest analyses will appear in this timeline.</ChartEmpty>
              ) : (
                <div className="activity-list">
                  {data.recent.map((a) => (
                    <Link key={a.id} to={`/analysis/${a.id}`} className="activity-item">
                      <div className="thumb"><AuthedImage analysisId={a.id} alt="" style={{ width: '100%', height: '100%' }} /></div>
                      <div className="grow"><strong>{headline(a)}</strong><span className="small muted">{fmtDateTime(a.created_at)}</span></div>
                      <OutcomeTag analysis={a} />
                    </Link>
                  ))}
                </div>
              )}
            </div>
            <ModelCard />
          </div>
        </>
      )}
    </>
  )
}
