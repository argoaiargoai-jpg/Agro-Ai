/** Lightweight inline vector art (no image files, no network): leaves, a scanned leaf, field rows. Decorative, hidden from screen readers. */
export function LeafScanArt({ className = '' }) {
  return (
    <svg className={`art leafscan ${className}`} viewBox="0 0 260 220" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="ls-g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#4ade80" /><stop offset="1" stopColor="#15803d" /></linearGradient>
        <linearGradient id="ls-b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#86efac" stopOpacity="0" /><stop offset="1" stopColor="#86efac" stopOpacity=".9" /></linearGradient>
        <clipPath id="ls-c"><path d="M130 24C70 40 34 92 52 150c18 50 74 58 112 24 42-38 52-108-34-150z" /></clipPath>
      </defs>
      <circle cx="130" cy="112" r="96" fill="#dcfce7" opacity=".7" />
      <path d="M130 24C70 40 34 92 52 150c18 50 74 58 112 24 42-38 52-108-34-150z" fill="url(#ls-g)" />
      <g stroke="#ecfdf5" strokeOpacity=".55" strokeWidth="2" fill="none" strokeLinecap="round">
        <path d="M70 168C92 128 112 92 132 36" /><path d="M96 134l30-6M108 108l28-4M88 152l24-8M120 80l22-2" />
      </g>
      <circle cx="150" cy="118" r="7" fill="#f59e0b" opacity=".9" /><circle cx="108" cy="146" r="5" fill="#f59e0b" opacity=".75" />
      <g clipPath="url(#ls-c)"><rect className="art-beam" x="30" y="20" width="200" height="34" fill="url(#ls-b)" /></g>
      <g stroke="#16a34a" strokeWidth="4" fill="none" strokeLinecap="round" className="art-brackets">
        <path d="M34 54V30h24M226 54V30h-24M34 170v24h24M226 170v24h-24" />
      </g>
    </svg>
  )
}

export function FieldRows({ className = '' }) {
  return (
    <svg className={`art fieldrows ${className}`} viewBox="0 0 600 90" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <path d="M0 62C90 40 170 74 300 52s220 22 300-6V90H0z" fill="#bbf7d0" opacity=".7" />
      <path d="M0 74C120 54 200 88 320 68s190 12 280-6V90H0z" fill="#86efac" opacity=".75" />
      <path d="M0 84C100 70 220 92 330 80s190 4 270-4V90H0z" fill="#4ade80" opacity=".85" />
      {[40, 120, 205, 290, 380, 470, 555].map((x, i) => (
        <g key={x} transform={`translate(${x} ${50 + (i % 3) * 5})`}>
          <g className="sprout" style={{ animationDelay: `${i * 0.25}s` }}>
            <path d="M0 24V6" stroke="#15803d" strokeWidth="3" strokeLinecap="round" />
            <path d="M0 10C-10 10-14 2-12-6c10 0 14 6 12 16zM0 14c10 0 14-6 12-14-10 0-14 6-12 14z" fill="#16a34a" />
          </g>
        </g>
      ))}
    </svg>
  )
}
