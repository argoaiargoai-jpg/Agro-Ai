import { describe, expect, it } from 'vitest'
import { isEmail, passwordIssue, passwordStrength } from './validate.js'
import { fmtBytes, initials } from './format.js'

describe('validate', () => {
  it('validates emails', () => {
    expect(isEmail('a@b.co')).toBe(true)
    expect(isEmail('a@b')).toBe(false)
    expect(isEmail('a b@c.com')).toBe(false)
  })
  it('matches backend password policy', () => {
    expect(passwordIssue('short1')).toBeTruthy()
    expect(passwordIssue('allletters')).toBeTruthy()
    expect(passwordIssue('12345678')).toBeTruthy()
    expect(passwordIssue('a1'.repeat(40))).toBeTruthy()
    expect(passwordIssue('Farmer123')).toBeNull()
  })
  it('scores strength', () => {
    expect(passwordStrength('abc')).toBe(0)
    expect(passwordStrength('Farmer123!!long')).toBe(4)
  })
})

describe('format', () => {
  it('formats bytes and initials', () => {
    expect(fmtBytes(2048)).toBe('2 KB')
    expect(fmtBytes(5 * 1024 * 1024)).toBe('5.0 MB')
    expect(initials('Test Farmer')).toBe('TF')
    expect(initials('')).toBe('?')
  })
})
