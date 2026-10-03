import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { api, mayHaveSession, refreshSession, setAccessToken, setAuthLostHandler } from '../lib/api'

const AuthContext = createContext(null)
export const useAuth = () => useContext(AuthContext)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [status, setStatus] = useState('loading') // loading | authed | anon

  const adopt = useCallback((data) => {
    setAccessToken(data.access_token)
    setUser(data.user)
    setStatus('authed')
    return data.user
  }, [])

  const clear = useCallback(() => {
    setAccessToken(null)
    setUser(null)
    setStatus('anon')
  }, [])

  useEffect(() => {
    setAuthLostHandler(clear)
    if (!mayHaveSession()) { setStatus('anon'); return }
    refreshSession().then(adopt).catch(clear)
  }, [adopt, clear])

  const value = useMemo(() => ({
    user, status, isAdmin: user?.role === 'admin',
    adoptSession: adopt,
    login: async (email, password) => adopt(await api.post('/auth/login', { email, password }, { auth: false })),
    verifyOtp: async (email, code) => adopt(await api.post('/auth/verify-otp', { email, code }, { auth: false })),
    completeGoogle: async () => adopt(await refreshSession()),
    logout: async () => {
      try { await api.post('/auth/logout', undefined, { auth: false }) } finally { clear() }
    },
    setUser,
    refreshUser: async () => setUser(await api.get('/users/me')),
  }), [user, status, adopt, clear])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
