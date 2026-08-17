import { useEffect, useId, useRef, useState } from 'react'
import { FiCheckCircle, FiFileText, FiX } from 'react-icons/fi'
import type { SourceDocument, VoiceRagResponse } from '../types/rag'
import { SourceList } from './SourceList'

export function EvidenceCard({ result }: { result: VoiceRagResponse }) {
  const [open, setOpen] = useState(false)
  const titleId = useId()
  const closeRef = useRef<HTMLButtonElement>(null)
  const count = result.sources.length
  const grounded = result.grounded && !result.refused

  useEffect(() => {
    if (!open) return
    closeRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = previous
    }
  }, [open])

  return (
    <section className="result-card">
      <div className="section-label">
        <FiFileText aria-hidden="true" />
        Evidence
      </div>
      <p className="mt-4 flex items-start gap-2 text-sm leading-6 text-slate-200">
        {grounded ? (
          <>
            <FiCheckCircle className="mt-0.5 size-4 shrink-0 text-emerald-300" aria-hidden="true" />
            <span>Grounded using {count} retrieved {count === 1 ? 'passage' : 'passages'}</span>
          </>
        ) : (
          <>
            <FiFileText className="mt-0.5 size-4 shrink-0 text-amber-200" aria-hidden="true" />
            <span>
              {count
                ? `${count} retrieved ${count === 1 ? 'passage' : 'passages'} available for inspection. The answer was withheld.`
                : 'No retrieved passages are shown for this refusal.'}
            </span>
          </>
        )}
      </p>
      <dl className="mt-4 grid gap-2 text-xs text-slate-500 sm:grid-cols-3">
        <div>
          <dt>Knowledge base</dt>
          <dd className="mt-1 text-slate-300">MSMARCO-XI validation subset</dd>
        </div>
        <div>
          <dt>Retrieval</dt>
          <dd className="mt-1 text-slate-300">Dense + BM25</dd>
        </div>
        <div>
          <dt>Fusion</dt>
          <dd className="mt-1 text-slate-300">RRF</dd>
        </div>
      </dl>
      {count > 0 && (
        <button type="button" className="secondary-button mt-5" onClick={() => setOpen(true)}>
          View evidence
        </button>
      )}

      {open && (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-slate-950/70 p-4 sm:items-center">
          <button
            type="button"
            className="absolute inset-0"
            aria-label="Close evidence"
            onClick={() => setOpen(false)}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            className="relative z-10 max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-2xl border border-white/10 bg-[#0b1220] p-5 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id={titleId} className="text-base font-semibold text-white">
                  Retrieved passages
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Engineering evidence for this request. {count} {count === 1 ? 'passage' : 'passages'}.
                </p>
              </div>
              <button
                ref={closeRef}
                type="button"
                className="secondary-button px-3 py-2"
                onClick={() => setOpen(false)}
              >
                <FiX aria-hidden="true" /> Close
              </button>
            </div>
            <div className="mt-4">
              <SourceList sources={result.sources as SourceDocument[]} />
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
