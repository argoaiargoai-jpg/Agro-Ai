import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

// Customers must never see model/provider names or a comparison between them. Admin-only code is excluded on purpose.
const NAMES = /mobilenet|onnx|gemini|plantix|kindwise|pl@?nt\s?net|disagree|confidence routing|fallback|first-stage|model said|supported crops|supports:|single leaf/i
const ML_WORD = /\bML\b/                                      // case-sensitive on purpose: `ml` is a normal identifier in code
const FORBIDDEN = { match: (t) => t.match(NAMES) || t.match(ML_WORD) }

const root = fileURLToPath(new URL('..', import.meta.url))      // frontend/src
const read = (rel) => readFileSync(join(root, rel), 'utf8')

function visibleText(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')                           // comments are not shown
    .replace(/^\s*\/\/.*$/gm, '')
    .replace(/\s\/\/ .*$/gm, '')
}

describe('customer-facing copy', () => {
  const pages = ['Analyze', 'Dashboard', 'History', 'Landing', 'Login', 'Register', 'Profile', 'Settings', 'VerifyOtp', 'ForgotPassword'].map((n) => `pages/${n}.jsx`)
  const parts = ['components/Scan.jsx', 'components/AgriArt.jsx', 'components/AnalysisBits.jsx', 'components/AuthShell.jsx', 'layouts/AppLayout.jsx', 'lib/outcome.js', 'lib/format.js']

  it.each([...pages, ...parts])('%s contains no model or provider terminology', (file) => {
    const m = FORBIDDEN.match(visibleText(read(file)))
    expect(m && m[0]).toBeNull()
  })

  it('the result page is clean once the administrators-only panel is excluded', () => {
    const src = visibleText(read('pages/AnalysisResult.jsx'))
    const customer = src.replace(/function AdminInternals[\s\S]*?\nexport default/, 'export default')
    const m = FORBIDDEN.match(customer)
    expect(m && m[0]).toBeNull()
  })

  it('the admin-only panel really is gated behind the admin role', () => {
    expect(read('pages/AnalysisResult.jsx')).toMatch(/\{isAdmin && a && <AdminInternals a=\{a\} \/>\}/)
  })

  it('no page shows a list of supported crops any more', () => {
    for (const f of readdirSync(join(root, 'pages')).filter((x) => x.endsWith('.jsx'))) {
      expect(read(`pages/${f}`)).not.toMatch(/config\.supported_crops|supported_crops\.(map|join)/)
    }
  })
})
