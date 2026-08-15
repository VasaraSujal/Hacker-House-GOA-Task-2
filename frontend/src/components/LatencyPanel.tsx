import { FiClock } from 'react-icons/fi'
import type { VoiceLatency } from '../types/rag'

function duration(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`
}

export function LatencyPanel({ latency }: { latency: VoiceLatency }) {
  const rows = [
    { label: 'Speech-to-text', value: latency.stt_ms, color: 'bg-cyan-300' },
    { label: 'RAG core', value: latency.rag_core_ms, color: 'bg-violet-300' },
    { label: 'Generation', value: latency.generation_ms, color: 'bg-amber-300' },
    { label: 'Grounding', value: latency.grounding_ms, color: 'bg-emerald-300' },
  ]
  const total = Math.max(latency.total_ms, 1)

  return (
    <section className="result-card">
      <div className="flex items-center justify-between">
        <div className="section-label">
          <FiClock aria-hidden="true" />
          Request performance
        </div>
        <span className="font-mono text-sm font-semibold text-white">{duration(latency.total_ms)}</span>
      </div>
      <div className="mt-5 space-y-4">
        {rows.map((row) => (
          <div key={row.label}>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-slate-400">{row.label}</span>
              <span className="font-mono text-slate-300">{duration(row.value)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
              <div
                className={`h-full rounded-full ${row.color}`}
                style={{ width: `${Math.max((row.value / total) * 100, row.value > 0 ? 1 : 0)}%` }}
                role="meter"
                aria-label={`${row.label} latency`}
                aria-valuenow={Math.round(row.value)}
                aria-valuemin={0}
                aria-valuemax={Math.round(total)}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-5 border-t border-white/8 pt-4 text-xs leading-5 text-slate-500">
        Actual timing for this request. External speech recognition and generation account for most of the wait.
      </p>
    </section>
  )
}
