export const fmtDate = (iso) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '—'
export const fmtDateTime = (iso) =>
  iso ? new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '—'
export const fmtBytes = (n) => (n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`)
export const initials = (name = '') => name.split(/\s+/).filter(Boolean).slice(0, 2).map((p) => p[0].toUpperCase()).join('') || '?'

export const STATUS = {
  uploaded: { label: 'Awaiting analysis', tone: 'gray' },
  queued: { label: 'Queued', tone: 'warn' },
  partial: { label: 'Guidance pending', tone: 'warn' },
  processing: { label: 'Analyzing', tone: 'warn' },
  completed: { label: 'Completed', tone: 'ok' },
  failed: { label: 'Failed', tone: 'err' },
}

export const OAUTH_ERRORS = {
  oauth_cancelled: 'Google sign-in was cancelled.',
  oauth_state_invalid: "Google sign-in couldn't be completed. Please try again.",
  oauth_exchange_failed: "Google sign-in couldn't be completed. Please try again.",
  oauth_email_unverified: 'Your Google email address is not verified.',
  oauth_unreachable: "Google sign-in couldn't be completed. Please try again.",
  oauth_conflict: 'This email is linked to a different Google account.',
  account_disabled: 'This account has been disabled.',
  registration_disabled: 'New registrations are currently closed.',
  google_not_configured: 'Google sign-in is not configured on this server.',
}
