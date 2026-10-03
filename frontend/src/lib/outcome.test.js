import { describe, expect, it } from 'vitest'
import { describeAnalysis, headline, stepState } from './outcome'

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

describe('stepState', () => {
  it('follows the real backend stage', () => {
    expect([0, 1, 2, 3].map((i) => stepState('uploading', i))).toEqual(['now', 'todo', 'todo', 'todo'])
    expect([0, 1, 2, 3].map((i) => stepState('ml', i))).toEqual(['done', 'now', 'todo', 'todo'])
    expect([0, 1, 2, 3].map((i) => stepState('guidance', i))).toEqual(['done', 'done', 'now', 'todo'])
    expect([0, 1, 2, 3].map((i) => stepState('done', i))).toEqual(['done', 'done', 'done', 'done'])
  })
})
