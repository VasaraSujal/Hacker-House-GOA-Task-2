import { useState } from 'react'
import { FiAlertTriangle, FiCheckCircle, FiClock } from 'react-icons/fi'
import { formatDuration, isMeasuredMs } from '../lib/latency'
import type { VoiceLatency } from '../types/rag'

const RAG_TARGET_MS = 200

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

export function LatencyPanel({ latency }: { latency: VoiceLatency }) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const ragCoreMs = latency.rag_core_ms
  const ragMeasured = isMeasuredMs(ragCoreMs)
  const underTarget = ragMeasured && ragCoreMs < RAG_TARGET_MS
  const ragLabel = ragMeasured ? `${Math.round(ragCoreMs)} ms` : null
  const totalLabel = formatDuration(latency.total_ms)
  const sttLabel = formatDuration(latency.stt_ms)

  return (
    <section
      className={`result-card ${
        ragMeasured
          ? underTarget
            ? 'border-emerald-300/25'
            : 'border-amber-300/25'
          : ''
      }`}
    >
      <div className="section-label">
        <FiClock aria-hidden="true" />
        RAG latency
      </div>

      {ragLabel ? (
        <>
          <p className="mt-4 font-mono text-5xl font-semibold tracking-tight text-white">
            {ragLabel}
          </p>
          <p
            className={`mt-4 flex items-center gap-2 text-sm font-medium ${
              underTarget ? 'text-emerald-200' : 'text-amber-100'
            }`}
          >
            {underTarget ? (
              <FiCheckCircle aria-hidden="true" />
            ) : (
              <FiAlertTriangle aria-hidden="true" />
            )}
            {underTarget ? 'Under 200 ms target' : 'Above 200 ms target'}
          </p>
        </>
      ) : (
        <p className="mt-4 text-base font-medium text-slate-400">
          RAG latency unavailable
        </p>
      )}
      <p className="mt-2 text-xs text-slate-500">Retrieval → Grounding</p>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
        {sttLabel && (
          <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2">
            <dt className="text-slate-500">Speech-to-text</dt>
            <dd className="mt-1 font-mono text-sm text-cyan-200">{sttLabel}</dd>
          </div>
        )}
        {totalLabel && (
          <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2">
            <dt className="text-slate-500">Total voice request</dt>
            <dd className="mt-1 font-mono text-sm text-slate-300">{totalLabel}</dd>
          </div>
        )}
      </dl>

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
