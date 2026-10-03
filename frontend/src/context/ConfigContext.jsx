import { createContext, useContext, useEffect, useState } from 'react'
import { api } from '../lib/api'

const DEFAULTS = { registration_enabled: true, maintenance_mode: false, announcement: '', max_upload_mb: 10, supported_crops: [], google_enabled: false }
const ConfigContext = createContext({ config: DEFAULTS, loaded: false, reload: () => {} })
export const useConfig = () => useContext(ConfigContext)

export function ConfigProvider({ children }) {
  const [config, setConfig] = useState(DEFAULTS)
  const [loaded, setLoaded] = useState(false)
  const reload = () => api.get('/config/public', { auth: false }).then(setConfig).catch(() => {}).finally(() => setLoaded(true))
  useEffect(() => { reload() }, [])
  return <ConfigContext.Provider value={{ config, loaded, reload }}>{children}</ConfigContext.Provider>
}
