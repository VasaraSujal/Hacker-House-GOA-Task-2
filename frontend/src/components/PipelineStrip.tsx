import { FiCheckCircle, FiCpu, FiMessageSquare, FiMic, FiSearch, FiType } from 'react-icons/fi'

const STAGES = [
  { label: 'Voice', icon: FiMic },
  { label: 'STT', icon: FiType },
  { label: 'Retrieval', icon: FiSearch },
  { label: 'Rerank', icon: FiCpu },
  { label: 'Ground', icon: FiCheckCircle },
  { label: 'Answer', icon: FiMessageSquare },
]

export function PipelineStrip() {
  return (
    <section className="result-card overflow-hidden">
      <div className="section-label">RAG pipeline</div>
      <ol className="mt-4 flex flex-wrap items-stretch gap-2">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon
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
              </div>
            </li>
          )
        })}
      </ol>
      <p className="mt-4 text-xs text-slate-500">
        Detailed backend timings are available from the RAG latency card.
      </p>
    </section>
  )
}
