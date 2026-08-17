import { useState } from 'react'
import { FiClock } from 'react-icons/fi'
import { formatDuration, isMeasuredMs, shareOfTotal } from '../lib/latency'
import type { VoiceLatency } from '../types/rag'

const SUMMARY_ROWS: Array<{ label: string; key: keyof VoiceLatency; color: string }> = [
  { label: 'Speech-to-text', key: 'stt_ms', color: 'bg-cyan-300' },
  { label: 'RAG core', key: 'rag_core_ms', color: 'bg-violet-300' },
  { label: 'Answer generation', key: 'generation_ms', color: 'bg-amber-300' },
  { label: 'Grounding', key: 'grounding_ms', color: 'bg-emerald-300' },
]

const DETAIL_ROWS: Array<{ label: string; key: keyof VoiceLatency }> = [
  { label: 'Speech-to-text', key: 'stt_ms' },
  { label: 'Embedding', key: 'embedding_ms' },
  { label: 'Dense retrieval', key: 'dense_retrieval_ms' },
  { label: 'BM25', key: 'bm25_ms' },
  { label: 'Retrieval wall', key: 'retrieval_wall_ms' },
  { label: 'Fusion', key: 'fusion_ms' },
  { label: 'Reranking', key: 'reranking_ms' },
  { label: 'Generation', key: 'generation_ms' },
  { label: 'Grounding', key: 'grounding_ms' },
  { label: 'RAG core', key: 'rag_core_ms' },
]

function LatencyRow({
  label,
  value,
  total,
  color,
}: {
  label: string
  value: VoiceLatency[keyof VoiceLatency]
  total: VoiceLatency['total_ms']
  color: string
}) {
  const formatted = formatDuration(value)
  if (!formatted) return null

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-300">{formatted}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${shareOfTotal(value, total)}%` }}
          role="meter"
          aria-label={`${label} latency`}
          aria-valuenow={isMeasuredMs(value) ? Math.round(value) : 0}
          aria-valuemin={0}
          aria-valuemax={isMeasuredMs(total) ? Math.round(total) : 0}
        />
      </div>
    </div>
  )
}

export function LatencyPanel({ latency }: { latency: VoiceLatency }) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const totalLabel = formatDuration(latency.total_ms)
  const ragLabel = formatDuration(latency.rag_core_ms)
  const sttLabel = formatDuration(latency.stt_ms)

  return (
    <section className="result-card">
      <div className="section-label">
        <FiClock aria-hidden="true" />
        Request performance
      </div>

      {totalLabel ? (
        <p className="mt-4 font-mono text-4xl font-semibold tracking-tight text-white">{totalLabel}</p>
      ) : (
        <p className="mt-4 text-sm text-slate-500">Total latency was not reported for this request.</p>
      )}
      <p className="mt-1 text-xs text-slate-500">Full request · microphone to grounded answer</p>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
        {sttLabel && (
          <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2">
            <dt className="text-slate-500">Speech-to-text</dt>
            <dd className="mt-1 font-mono text-sm text-cyan-200">{sttLabel}</dd>
          </div>
        )}
        {ragLabel && (
          <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2">
            <dt className="text-slate-500">RAG core</dt>
            <dd className="mt-1 font-mono text-sm text-violet-200">{ragLabel}</dd>
          </div>
        )}
      </dl>

      <div className="mt-5 space-y-4">
        {SUMMARY_ROWS.map((row) => (
          <LatencyRow
            key={row.key}
            label={row.label}
            value={latency[row.key]}
            total={latency.total_ms}
            color={row.color}
          />
        ))}
      </div>

      <button
        type="button"
        className="mt-5 text-xs font-medium text-cyan-200 underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
        aria-expanded={detailsOpen}
        onClick={() => setDetailsOpen((open) => !open)}
      >
        {detailsOpen ? 'Hide pipeline timings' : 'Show pipeline timings'}
      </button>

      {detailsOpen && (
        <dl className="mt-4 space-y-2 border-t border-white/8 pt-4 text-xs">
          {DETAIL_ROWS.map((row) => {
            const formatted = formatDuration(latency[row.key])
            if (!formatted) return null
            return (
              <div key={row.key} className="flex items-center justify-between gap-3">
                <dt className="text-slate-500">{row.label}</dt>
                <dd className="font-mono text-slate-300">{formatted}</dd>
              </div>
            )
          })}
        </dl>
      )}
    </section>
  )
}
