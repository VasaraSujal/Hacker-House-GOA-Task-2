export type LatencyValue = number | null | undefined

export function isMeasuredMs(value: LatencyValue): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function formatDuration(ms: LatencyValue): string | null {
  if (!isMeasuredMs(ms)) return null
  if (ms === 0) return '0 ms'
  if (ms > 0 && ms < 1) return '<1 ms'
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  return `${Math.round(ms)} ms`
}

export function shareOfTotal(value: LatencyValue, total: LatencyValue): number {
  if (!isMeasuredMs(value) || !isMeasuredMs(total) || total <= 0) return 0
  if (value <= 0) return 0
  return Math.min(100, Math.max((value / total) * 100, 1))
}
