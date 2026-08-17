import { describe, expect, it } from 'vitest'
import { formatDuration, isMeasuredMs, shareOfTotal } from './latency'

describe('latency formatting', () => {
  it('formats backend milliseconds without inventing values', () => {
    expect(formatDuration(4200)).toBe('4.20 s')
    expect(formatDuration(889)).toBe('889 ms')
    expect(formatDuration(0)).toBe('0 ms')
    expect(formatDuration(0.4)).toBe('<1 ms')
  })

  it('skips missing, null, and non-finite fields', () => {
    expect(formatDuration(undefined)).toBeNull()
    expect(formatDuration(null)).toBeNull()
    expect(formatDuration(Number.NaN)).toBeNull()
    expect(isMeasuredMs(undefined)).toBe(false)
    expect(isMeasuredMs(Number.POSITIVE_INFINITY)).toBe(false)
  })

  it('does not invent a bar width when totals are missing', () => {
    expect(shareOfTotal(889, undefined)).toBe(0)
    expect(shareOfTotal(undefined, 4200)).toBe(0)
  })
})
