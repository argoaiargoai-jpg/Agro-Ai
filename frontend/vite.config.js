import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const backend = process.env.VITE_BACKEND_URL || 'http://localhost:8000'

export default defineConfig(({ mode }) => {
  if (mode === 'production') {
    const env = loadEnv(mode, process.cwd(), 'VITE_')   // .env files AND real environment variables
    // 1) Anything starting with VITE_ is compiled into the public JavaScript. Refuse to build if it looks like a secret.
    const risky = Object.keys(env).filter((k) => /KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL/i.test(k))
    if (risky.length) throw new Error(`Refusing to build: ${risky.join(', ')} would be exposed to every visitor. Secrets belong on the backend only.`)
    // 2) The API must be reached over HTTPS from a production site.
    const api = env.VITE_API_URL
    if (api && !/^https:\/\/[^/\s]+(:\d+)?$/.test(api.trim())) {
      throw new Error(`VITE_API_URL must be an https:// origin without a path or trailing slash (got "${api}"). Example: https://agroai-api.onrender.com`)
    }
  }
  return {
    plugins: [react()],
    server: { port: 5173, proxy: { '/api': { target: backend, changeOrigin: false } } },
    preview: { port: 5173, proxy: { '/api': { target: backend } } },
    test: { environment: 'node' },
  }
})
