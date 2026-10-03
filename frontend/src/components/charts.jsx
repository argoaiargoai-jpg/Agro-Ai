/** Donut chart (pure SVG). `data`: [{key,label,value,color}]. Shows `centerValue`/`centerLabel` in the hole. */
export function Donut({ data, centerValue, centerLabel, size = 168, stroke = 22 }) {
  const total = data.reduce((n, d) => n + d.value, 0)
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  let offset = 0
  return (
    <div className="donut-wrap">
      <svg className="donut" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img"
        aria-label={data.map((d) => `${d.label}: ${d.value}`).join(', ')}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--mint-100)" strokeWidth={stroke} />
        {total > 0 && data.filter((d) => d.value > 0).map((d) => {
          const len = (d.value / total) * c
          const el = (
            <circle key={d.key} cx={size / 2} cy={size / 2} r={r} fill="none" stroke={d.color} strokeWidth={stroke}
              strokeDasharray={`${len} ${c}`} strokeDashoffset={-offset} transform={`rotate(-90 ${size / 2} ${size / 2})`}
              className="donut-arc" style={{ '--c': c }} />
          )
          offset += len
          return el
        })}
        <text x="50%" y="48%" textAnchor="middle" className="donut-num">{centerValue}</text>
        <text x="50%" y="62%" textAnchor="middle" className="donut-cap">{centerLabel}</text>
      </svg>
      <ul className="legend">
        {data.map((d) => (
          <li key={d.key}><i style={{ background: d.color }} /><span>{d.label}</span><strong>{d.value}</strong></li>
        ))}
      </ul>
    </div>
  )
}

/** Horizontal bars: items [{name,count}] */
export function HBars({ items, color = 'var(--leaf-600)' }) {
  const max = Math.max(1, ...items.map((i) => i.count))
  return (
    <ul className="hbars">
      {items.map((i, n) => (
        <li key={i.name}>
          <div className="hbar-top"><span title={i.name}>{i.name}</span><strong>{i.count}</strong></div>
          <div className="hbar-track"><i style={{ width: `${Math.max((i.count / max) * 100, 4)}%`, background: color, animationDelay: `${n * 70}ms` }} /></div>
        </li>
      ))}
    </ul>
  )
}

/** Column chart for the last N days: series [{date,count}] */
export function Columns({ series, label }) {
  const max = Math.max(1, ...series.map((d) => d.count))
  const total = series.reduce((n, d) => n + d.count, 0)
  return (
    <div className="bars" role="img" aria-label={`${label}, ${total} in total`}>
      {series.map((d) => (
        <div key={d.date} className="bar-col" title={`${d.date}: ${d.count}`}>
          <span className="bar-n">{d.count || ''}</span>
          <i style={{ height: d.count ? `${Math.max((d.count / max) * 100, 6)}%` : '3px', opacity: d.count ? 1 : 0.35 }} />
          <span className="bar-d">{d.date.slice(8)}</span>
        </div>
      ))}
    </div>
  )
}
