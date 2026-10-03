import { useEffect, useRef, useState } from 'react'

/** Fades/slides children in when scrolled into view. */
export function Reveal({ as: Tag = 'div', variant = '', delay = 0, className = '', children, ...rest }) {
  const ref = useRef(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    // Reveal once the element's top is above the lower edge of the viewport. Checking on scroll (not just
    // IntersectionObserver) means fast scrolls and #anchor jumps can't skip an element and leave it hidden.
    const check = () => {
      if (el.getBoundingClientRect().top < window.innerHeight - 40) {
        setSeen(true)
        window.removeEventListener('scroll', check)
        window.removeEventListener('resize', check)
      }
    }
    check()
    window.addEventListener('scroll', check, { passive: true })
    window.addEventListener('resize', check)
    return () => { window.removeEventListener('scroll', check); window.removeEventListener('resize', check) }
  }, [])
  return <Tag ref={ref} className={`reveal ${variant} ${seen ? 'in' : ''} ${className}`} style={{ '--d': `${delay}ms` }} {...rest}>{children}</Tag>
}

/** Counts up to `value` once mounted. */
export function CountUp({ value, duration = 900 }) {
  const [n, setN] = useState(0)
  useEffect(() => {
    if (typeof value !== 'number') return
    let raf, start
    const tick = (t) => {
      start ??= t
      const p = Math.min((t - start) / duration, 1)
      setN(Math.round(value * (1 - Math.pow(1 - p, 3))))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])
  return <>{typeof value === 'number' ? n : value}</>
}

export const Leaf = ({ size = 40, color = '#22c55e', style, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" style={style} className={className}>
    <path d="M6 26C6 13 14 5 27 5c0 13-8 21-20 21z" fill={color} /><path d="M7 26c5-8 10-13 17-17" stroke="rgba(255,255,255,.55)" strokeWidth="1.4" strokeLinecap="round" fill="none" />
  </svg>
)
