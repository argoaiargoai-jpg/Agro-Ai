/** Customer-level outcome of an analysis, derived from the stored result. No internal ML/AI wording ever reaches the UI. */
export const OUTCOME = {
  healthy: { label: 'Healthy', tone: 'ok' },
  disease: { label: 'Disease detected', tone: 'warn' },
  unresolved: { label: 'Needs a clearer photo', tone: 'unk' },
  no_plant: { label: 'No plant detected', tone: 'np' },
}

const FINAL_TO_OUTCOME = { DISEASE: 'disease', HEALTHY: 'healthy', UNCERTAIN: 'unresolved', REJECTED: 'no_plant' }
const ML_TO_OUTCOME = { DISEASE: 'disease', HEALTHY: 'healthy', UNKNOWN: 'unresolved', NO_PLANT: 'no_plant' }

/** @returns {{outcome: string|null, plant: string|null, condition: string|null}} */
export function describeAnalysis(a) {
  const r = a?.result
  const f = r?.final
  const ml = r?.ml
  if (f && FINAL_TO_OUTCOME[f.status]) {
    const outcome = FINAL_TO_OUTCOME[f.status]
    return { outcome, plant: f.plant || f.crop || null, condition: outcome === 'disease' ? (f.disease || null) : outcome === 'healthy' ? 'Healthy' : null }
  }
  if (ml && ML_TO_OUTCOME[ml.classification_type]) {
    const outcome = ML_TO_OUTCOME[ml.classification_type]
    return { outcome, plant: ml.crop || null, condition: outcome === 'disease' ? (ml.disease || null) : outcome === 'healthy' ? 'Healthy' : null }
  }
  return { outcome: null, plant: null, condition: null }
}

/** One-line headline for lists: "Tomato — Early Blight", "Grape — Healthy", "No plant detected". */
export function headline(a) {
  const { outcome, plant, condition } = describeAnalysis(a)
  if (!outcome) return a?.crop_type || 'Awaiting analysis'
  if (outcome === 'disease' || outcome === 'healthy') return [plant || a?.crop_type, condition].filter(Boolean).join(' — ') || OUTCOME[outcome].label
  return OUTCOME[outcome].label
}

/**
 * The visible analysis steps. They follow the REAL backend workflow: `plan` is what the backend says will run
 * (e.g. ["guidance"], ["disease","guidance"], ["identify","disease","guidance"]). Labels are generic: no provider is ever named.
 */
export const STEP_LABELS = {
  received: { label: 'Image received' },
  ml: { label: 'AGRO AI deep-learning analysis', hint: 'Examining the image' },
  identify: { label: 'Plant identification', hint: 'Identifying the plant' },
  disease: { label: 'Agricultural disease analysis', hint: 'Checking for diseases and disorders' },
  guidance: { label: 'AI guidance', hint: 'Generating agricultural guidance' },
  final: { label: 'Final result' },
}

export function buildSteps(plan) {
  const mid = (Array.isArray(plan) ? plan : ['guidance']).filter((k) => STEP_LABELS[k])      // [] = the backend stopped after the first stage
  return ['received', 'ml', ...mid, 'final'].map((key) => ({ key, ...STEP_LABELS[key] }))
}

/**
 * state per step: 'done' | 'now' | 'todo' | 'failed'.
 * stage: 'uploading' | 'ml' | 'ml_done' | 'identify' | 'disease' | 'guidance' | 'done'  (what the backend reports)
 * failedAt: step key that failed (e.g. 'guidance'): earlier steps are done, that one is 'failed', later ones stay 'todo'.
 */
export function stepStates(steps, stage, failedAt = null) {
  const keys = steps.map((s) => s.key)
  if (failedAt && keys.includes(failedAt)) {
    const f = keys.indexOf(failedAt)
    return keys.map((_, i) => (i < f ? 'done' : i === f ? 'failed' : 'todo'))
  }
  if (stage === 'done') return keys.map(() => 'done')
  const active = stage === 'uploading' ? 'received' : stage === 'ml' ? 'ml' : stage === 'ml_done' ? keys[2] : keys.includes(stage) ? stage : 'received'
  const at = keys.indexOf(active)
  return keys.map((_, i) => (i < at ? 'done' : i === at ? 'now' : 'todo'))
}
