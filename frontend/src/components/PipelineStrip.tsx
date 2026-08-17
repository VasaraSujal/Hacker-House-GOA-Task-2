import { FiCheckCircle, FiCpu, FiMessageSquare, FiMic, FiSearch, FiType } from 'react-icons/fi'
import { formatDuration } from '../lib/latency'
import type { VoiceLatency } from '../types/rag'

const STAGES = [
  { label: 'Voice', key: 'audio_validation_ms' as const, icon: FiMic },
  { label: 'STT', key: 'stt_ms' as const, icon: FiType },
  { label: 'Retrieval', key: 'retrieval_wall_ms' as const, icon: FiSearch },
  { label: 'Rerank', key: 'reranking_ms' as const, icon: FiCpu },
  { label: 'Ground', key: 'grounding_ms' as const, icon: FiCheckCircle },
  { label: 'Answer', key: 'generation_ms' as const, icon: FiMessageSquare },
]

export function PipelineStrip({ latency }: { latency: VoiceLatency }) {
  return (
    <section className="result-card overflow-hidden">
      <div className="section-label">RAG pipeline</div>
      <ol className="mt-4 flex flex-wrap items-stretch gap-2">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon
          const timing = formatDuration(latency[stage.key])
          return (
            <li
              key={stage.label}
              className="flex min-w-[6.5rem] flex-1 items-center gap-2 rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2"
            >
              <Icon className="size-4 shrink-0 text-cyan-200" aria-hidden="true" />
              <div className="min-w-0">
                <p className="text-xs text-slate-200">
                  {index + 1}. {stage.label}
                </p>
                {timing && <p className="font-mono text-[11px] text-slate-500">{timing}</p>}
              </div>
            </li>
          )
        })}
      </ol>
      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:grid-cols-4">
        {[
          ['Dense retrieval', latency.dense_retrieval_ms],
          ['BM25', latency.bm25_ms],
          ['Fusion', latency.fusion_ms],
          ['Reranking', latency.reranking_ms],
          ['Extractive generation', latency.generation_ms],
          ['Grounding', latency.grounding_ms],
        ].map(([label, value]) => {
          const formatted = formatDuration(value as VoiceLatency[keyof VoiceLatency])
          if (!formatted) return null
          return (
            <div key={String(label)}>
              <dt className="text-slate-500">{label}</dt>
              <dd className="font-mono text-slate-300">{formatted}</dd>
            </div>
          )
        })}
      </dl>
    </section>
  )
}
