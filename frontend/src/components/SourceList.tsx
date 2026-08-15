import { useState } from 'react'
import { FiChevronDown, FiFileText } from 'react-icons/fi'
import type { SourceDocument } from '../types/rag'

export function SourceList({ sources }: { sources: SourceDocument[] }) {
  const [expanded, setExpanded] = useState<string | null>(sources[0]?.chunk_id ?? null)

  return (
    <section className="result-card">
      <div className="flex items-center justify-between">
        <div className="section-label">
          <FiFileText aria-hidden="true" />
          Retrieved sources
        </div>
        <span className="font-mono text-xs text-slate-500">{sources.length} chunks</span>
      </div>
      <div className="mt-4 space-y-2">
        {sources.map((source, index) => {
          const open = expanded === source.chunk_id
          return (
            <article key={source.chunk_id} className="rounded-xl border border-white/8 bg-slate-950/50">
              <button
                type="button"
                onClick={() => setExpanded(open ? null : source.chunk_id)}
                aria-expanded={open}
                className="flex w-full items-center gap-3 px-4 py-3 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
              >
                <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-white/5 font-mono text-[11px] text-cyan-200">
                  {index + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-slate-200">Document {source.document_id}</span>
                  <span className="block font-mono text-[10px] text-slate-600">Chunk {source.chunk_id}</span>
                </span>
                <span className="font-mono text-[11px] text-slate-500">{source.score.toFixed(4)}</span>
                <FiChevronDown className={`text-slate-500 transition ${open ? 'rotate-180' : ''}`} />
              </button>
              {open && <p className="border-t border-white/8 px-4 py-4 text-sm leading-6 text-slate-400">{source.text}</p>}
            </article>
          )
        })}
      </div>
    </section>
  )
}
