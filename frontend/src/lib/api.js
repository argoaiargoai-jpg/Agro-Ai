const BASE = `${import.meta.env.VITE_API_URL || ''}/api/v1`
const SESSION_FLAG = 'agro_has_session'

export class ApiError extends Error {
  constructor(status, code, message, fields) {
    super(message)
    this.status = status
    this.code = code
    this.fields = fields || null
  }
}

let accessToken = null
let refreshing = null
let onAuthLost = () => {}

export const setAuthLostHandler = (fn) => { onAuthLost = fn }
export const setAccessToken = (t) => {
  accessToken = t
  try { t ? localStorage.setItem(SESSION_FLAG, '1') : localStorage.removeItem(SESSION_FLAG) } catch { /* private mode */ }
}
export const mayHaveSession = () => {
  try { return localStorage.getItem(SESSION_FLAG) === '1' } catch { return true }
}

async function send(path, { method = 'GET', body, form, auth = true, signal, raw = false } = {}) {
  const headers = {}
  let payload
  if (form) payload = form
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }
  if (auth && accessToken) headers.Authorization = `Bearer ${accessToken}`
  let res
  try {
    res = await fetch(BASE + path, { method, headers, body: payload, credentials: 'include', signal })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(0, 'network_error', "Can't reach the AGRO AI server. Check your connection and try again.")
  }
  if (raw && res.ok) return res
  let data = null
  try { data = await res.json() } catch { /* empty / non-JSON */ }
  if (!res.ok) {
    const err = data?.error
    throw new ApiError(res.status, err?.code || 'http_error', err?.message || `Request failed (${res.status})`, err?.fields)
  }
  return data
}

/** Single-flight refresh so parallel 401s (and React StrictMode) never race the rotating cookie. */
export function refreshSession() {
  if (!refreshing) {
    refreshing = send('/auth/refresh', { method: 'POST', auth: false })
      .then((d) => { setAccessToken(d.access_token); return d })
      .catch((e) => { setAccessToken(null); throw e })
      .finally(() => { refreshing = null })
  }
  return refreshing
}

export async function request(path, opts = {}) {
  try {
    return await send(path, opts)
  } catch (e) {
    const retryable = opts.auth !== false && e instanceof ApiError && e.status === 401 && ['token_expired', 'unauthorized'].includes(e.code)
    if (!retryable) throw e
    try {
      await refreshSession()
    } catch {
      onAuthLost()
      throw e
    }
    return send(path, opts)
  }
}

export const api = {
  get: (p, o) => request(p, o),
  post: (p, body, o) => request(p, { method: 'POST', body, ...o }),
  put: (p, body, o) => request(p, { method: 'PUT', body, ...o }),
  patch: (p, body, o) => request(p, { method: 'PATCH', body, ...o }),
  del: (p, o) => request(p, { method: 'DELETE', ...o }),
  upload: (p, form, o) => request(p, { method: 'POST', form, ...o }),
  blob: async (p) => (await request(p, { raw: true })).blob(),
}

export const errorMessage = (e) => (e instanceof ApiError ? e.message : 'Something went wrong. Please try again.')
