import { Link } from 'react-router-dom'
import { forwardRef, useEffect, useId, useRef, useState } from 'react'
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, Eye, EyeOff, Info, Leaf } from 'lucide-react'

export function Logo({ to = '/', light = false }) {
  return (
    <Link to={to} className="logo" style={light ? { color: '#fff' } : undefined} aria-label="AGRO AI home">
      <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="9" fill={light ? '#22C55E' : '#166534'} />
        <path d="M9 22c0-8 5-13 14-13 0 9-5 14-13 14l-1-1z" fill={light ? '#14532d' : '#22C55E'} />
        <path d="M9 23c3-5 6-8 11-10" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" fill="none" />
      </svg>
      <span>AGRO <small style={light ? { color: '#86efac' } : undefined}>AI</small></span>
    </Link>
  )
}

export function Button({ variant = 'primary', size, block, loading, children, className = '', icon: Icon, ...rest }) {
  const cls = ['btn', `btn-${variant}`, size && `btn-${size}`, block && 'btn-block', className].filter(Boolean).join(' ')
  return (
    <button className={cls} disabled={loading || rest.disabled} {...rest}>
      {loading ? <span className="spinner" aria-hidden="true" /> : Icon ? <Icon size={size === 'sm' ? 15 : 18} /> : null}
      {children}
    </button>
  )
}

export const Field = forwardRef(function Field(
  { label, error, hint, type = 'text', as = 'input', children, id: idProp, ...rest }, ref,
) {
  const auto = useId()
  const id = idProp || auto
  const [show, setShow] = useState(false)
  const isPw = type === 'password'
  const describedBy = error ? `${id}-err` : hint ? `${id}-hint` : undefined
  const common = { id, ref, 'aria-invalid': error ? 'true' : undefined, 'aria-describedby': describedBy, ...rest }
  let control
  if (as === 'select') control = <select className="select input" {...common}>{children}</select>
  else if (as === 'textarea') control = <textarea className="textarea" {...common} />
  else control = <input className="input" type={isPw && show ? 'text' : type} {...common} />
  return (
    <div className="field">
      {label && <label htmlFor={id}>{label}</label>}
      {isPw ? (
        <div className="input-wrap">
          {control}
          <button type="button" className="btn btn-ghost btn-icon toggle" onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide password' : 'Show password'}>
            {show ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        </div>
      ) : control}
      {error ? <p id={`${id}-err`} className="field-error" role="alert"><AlertCircle size={14} />{error}</p>
        : hint ? <p id={`${id}-hint`} className="field-hint">{hint}</p> : null}
    </div>
  )
})

export function Switch({ checked, onChange, label, disabled }) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} aria-label={label} disabled={disabled} />
      <span />
    </label>
  )
}

const ICONS = { error: AlertCircle, warn: AlertTriangle, info: Info, success: CheckCircle2 }
export function Alert({ tone = 'info', children, ...rest }) {
  const Icon = ICONS[tone]
  return <div className={`alert alert-${tone}`} role={tone === 'error' ? 'alert' : 'status'} {...rest}><Icon size={18} /><div>{children}</div></div>
}

export const Spinner = ({ lg }) => <span className={`spinner${lg ? ' lg' : ''}`} role="status" aria-label="Loading" />
export const PageLoader = () => <div className="page-loader"><Spinner lg /></div>
export const Skeleton = ({ h = 16, w = '100%', style }) => <div className="skeleton" style={{ height: h, width: w, ...style }} aria-hidden="true" />

export function EmptyState({ icon: Icon = Leaf, title, children, action }) {
  return (
    <div className="state">
      <div className="state-icon"><Icon size={28} /></div>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}

export function ErrorState({ title = "We couldn't load this", message, onRetry }) {
  return (
    <div className="state error" role="alert">
      <div className="state-icon"><AlertCircle size={28} /></div>
      <h3>{title}</h3>
      <p>{message || 'Please check your connection and try again.'}</p>
      {onRetry && <Button variant="secondary" onClick={onRetry}>Try again</Button>}
    </div>
  )
}

export function Modal({ title, children, onClose, actions }) {
  const ref = useRef(null)
  useEffect(() => {
    const prev = document.activeElement
    ref.current?.querySelector('input, button')?.focus()
    const onKey = (e) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('keydown', onKey); prev?.focus?.() }
  }, [onClose])
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title} ref={ref}>
        <h3>{title}</h3>
        {children}
        <div className="modal-actions">{actions}</div>
      </div>
    </div>
  )
}

export function Pagination({ page, pageSize, total, onChange }) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  const from = total ? (page - 1) * pageSize + 1 : 0
  return (
    <div className="pagination">
      <span className="small muted">{from}–{Math.min(page * pageSize, total)} of {total}</span>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <Button variant="secondary" size="sm" disabled={page <= 1} onClick={() => onChange(page - 1)} aria-label="Previous page"><ChevronLeft size={16} /></Button>
        <span className="small">Page {page} / {pages}</span>
        <Button variant="secondary" size="sm" disabled={page >= pages} onClick={() => onChange(page + 1)} aria-label="Next page"><ChevronRight size={16} /></Button>
      </div>
    </div>
  )
}

export function OtpInput({ value, onChange, invalid, disabled }) {
  const refs = useRef([])
  const digits = Array.from({ length: 6 }, (_, i) => value[i] || '')
  const focus = (i) => refs.current[Math.max(0, Math.min(i, 5))]?.focus()

  // Value is always a contiguous digit string, so typing in any box fills the next empty slot.
  // Multi-digit input (paste, SMS/one-time-code autofill) is spread across the boxes.
  const handleChange = (i, raw) => {
    const typed = raw.replace(/\D/g, '')
    const at = Math.min(i, value.length)
    if (!typed) { onChange(value.slice(0, i) + value.slice(i + 1)); return }
    const next = (typed.length > 1 ? value.slice(0, at) + typed : value.slice(0, at) + typed + value.slice(at + 1)).slice(0, 6)
    onChange(next)
    focus(Math.min(at + typed.length, 5))
  }

  return (
    <div className={`otp${invalid ? ' invalid' : ''}`} role="group" aria-label="6-digit verification code">
      {digits.map((d, i) => (
        <input
          key={i} ref={(el) => (refs.current[i] = el)} value={d} disabled={disabled}
          inputMode="numeric" autoComplete={i === 0 ? 'one-time-code' : 'off'}
          aria-label={`Digit ${i + 1}`}
          onChange={(e) => handleChange(i, e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Backspace' && !digits[i] && i > 0) { e.preventDefault(); onChange(value.slice(0, i - 1)); focus(i - 1) }
            if (e.key === 'ArrowLeft') focus(i - 1)
            if (e.key === 'ArrowRight') focus(i + 1)
          }}
          onFocus={(e) => e.target.select()}
        />
      ))}
    </div>
  )
}

export function PasswordMeter({ score }) {
  const labels = ['Too weak', 'Weak', 'Okay', 'Good', 'Strong']
  const colors = ['#dc2626', '#f59e0b', '#f59e0b', '#22c55e', '#16a34a']
  return (
    <div aria-live="polite">
      <div style={{ display: 'flex', gap: 4, marginTop: 2 }}>
        {[0, 1, 2, 3].map((i) => <div key={i} style={{ flex: 1, height: 4, borderRadius: 4, background: i < score ? colors[score] : '#e1e8df' }} />)}
      </div>
      <p className="field-hint" style={{ marginTop: 4 }}>{labels[score]}</p>
    </div>
  )
}

export function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.9 6.1C12.4 13.6 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4.1 7.1-10.1 7.1-17.5z" />
      <path fill="#FBBC05" d="M10.5 28.7c-.5-1.5-.8-3-.8-4.7s.3-3.2.8-4.7l-7.9-6.1C.9 16.4 0 20.1 0 24s.9 7.6 2.6 10.8l7.9-6.1z" />
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.5-5.8c-2.1 1.4-4.9 2.3-8.4 2.3-6.3 0-11.6-4.1-13.5-9.8l-7.9 6.1C6.5 42.6 14.6 48 24 48z" />
    </svg>
  )
}
