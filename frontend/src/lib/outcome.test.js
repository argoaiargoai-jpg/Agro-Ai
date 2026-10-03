import { describe, expect, it } from 'vitest'
import { buildSteps, describeAnalysis, headline, stepStates } from './outcome'

const ml = (t, extra = {}) => ({ result: { ml: { classification_type: t, ...extra } } })

describe('describeAnalysis', () => {
  it('maps the four ML states to customer outcomes', () => {
    expect(describeAnalysis(ml('DISEASE', { crop: 'Tomato', disease: 'Early Blight' }))).toMatchObject({ outcome: 'disease', plant: 'Tomato', condition: 'Early Blight' })
    expect(describeAnalysis(ml('HEALTHY', { crop: 'Grape' }))).toMatchObject({ outcome: 'healthy', condition: 'Healthy' })
    expect(describeAnalysis(ml('UNKNOWN')).outcome).toBe('unresolved')
    expect(describeAnalysis(ml('NO_PLANT')).outcome).toBe('no_plant')
  })
  it('prefers the unified final report over the basic ML result', () => {
    const a = { result: { ml: { classification_type: 'UNKNOWN' }, final: { status: 'DISEASE', plant: 'Rose', disease: 'Black Spot' } } }
    expect(describeAnalysis(a)).toMatchObject({ outcome: 'disease', plant: 'Rose', condition: 'Black Spot' })
    expect(describeAnalysis({ result: { final: { status: 'REJECTED' } } }).outcome).toBe('no_plant')
    expect(describeAnalysis({ result: { final: { status: 'UNCERTAIN' } } }).outcome).toBe('unresolved')
  })
  it('has no outcome before a result exists', () => {
    expect(describeAnalysis({ result: null }).outcome).toBeNull()
    expect(describeAnalysis(undefined).outcome).toBeNull()
  })
})

describe('headline', () => {
  it('never shows internal terms', () => {
    const all = [ml('UNKNOWN'), ml('NO_PLANT'), ml('DISEASE', { crop: 'Apple', disease: 'Apple Scab' }), ml('HEALTHY', { crop: 'Apple' })].map(headline).join(' ')
    expect(all).not.toMatch(/UNKNOWN|NO_PLANT|unsupported|ML|Gemini/)
    expect(headline(ml('DISEASE', { crop: 'Apple', disease: 'Apple Scab' }))).toBe('Apple — Apple Scab')
  })
  it('falls back to the chosen crop before analysis', () => {
    expect(headline({ crop_type: 'Corn', result: null })).toBe('Corn')
    expect(headline({ result: null })).toBe('Awaiting analysis')
  })
})

describe('analysis steps follow the real backend plan', () => {
  const keys = (plan) => buildSteps(plan).map((s) => s.key)
  it('shows only the steps the backend will run, never provider names', () => {
    expect(keys(['guidance'])).toEqual(['received', 'ml', 'guidance', 'final'])
    expect(keys(['disease', 'guidance'])).toEqual(['received', 'ml', 'disease', 'guidance', 'final'])
    expect(keys(['identify', 'disease', 'guidance'])).toEqual(['received', 'ml', 'identify', 'disease', 'guidance', 'final'])
    expect(keys(undefined)).toEqual(['received', 'ml', 'guidance', 'final'])
    expect(keys([])).toEqual(['received', 'ml', 'final'])                                                // guidance switched off: nothing is promised
    expect(keys(['plantix', 'guidance'])).toEqual(['received', 'ml', 'guidance', 'final'])            // unknown names are ignored
    const text = buildSteps(['identify', 'disease', 'guidance']).map((s) => `${s.label} ${s.hint || ''}`).join(' ')
    expect(text).not.toMatch(/plantix|kindwise|plant\.?net|gemini|mobilenet|api/i)
  })
  it('marks pending / active / completed along the real stage', () => {
    const steps = buildSteps(['identify', 'disease', 'guidance'])
    expect(stepStates(steps, 'uploading')).toEqual(['now', 'todo', 'todo', 'todo', 'todo', 'todo'])
    expect(stepStates(steps, 'ml')).toEqual(['done', 'now', 'todo', 'todo', 'todo', 'todo'])
    expect(stepStates(steps, 'ml_done')).toEqual(['done', 'done', 'now', 'todo', 'todo', 'todo'])
    expect(stepStates(steps, 'disease')).toEqual(['done', 'done', 'done', 'now', 'todo', 'todo'])
    expect(stepStates(steps, 'guidance')).toEqual(['done', 'done', 'done', 'done', 'now', 'todo'])
    expect(stepStates(steps, 'done')).toEqual(Array(6).fill('done'))
  })
  it('the first plan step becomes active when the model finishes, whatever it is', () => {
    expect(stepStates(buildSteps(['guidance']), 'ml_done')).toEqual(['done', 'done', 'now', 'todo'])
    expect(stepStates(buildSteps(['disease', 'guidance']), 'ml_done')).toEqual(['done', 'done', 'now', 'todo', 'todo'])
  })
  it('shows failure at the failing step with earlier steps completed', () => {
    expect(stepStates(buildSteps(['identify', 'disease', 'guidance']), 'guidance', 'guidance')).toEqual(['done', 'done', 'done', 'done', 'failed', 'todo'])
    expect(stepStates(buildSteps(['guidance']), 'ml', 'guidance')).toEqual(['done', 'done', 'failed', 'todo'])
  })
})
