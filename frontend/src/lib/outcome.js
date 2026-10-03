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

/** Real analysis stage -> which of the 4 visible steps is done / active. Based only on backend state. */
export const STEPS = [
  { key: 'received', label: 'Image received' },
  { key: 'ml', label: 'AGRO AI ML model', hint: 'Preprocessing and ML-powered image analysis' },
  { key: 'intel', label: 'Agricultural intelligence', hint: 'Generating agricultural guidance' },
  { key: 'final', label: 'Final result' },
]
const ORDER = { uploading: 0, ml: 1, guidance: 2, done: 4 }
export function stepState(stage, index) {
  const at = ORDER[stage] ?? 0
  return index < at ? 'done' : index === at ? 'now' : 'todo'
}
