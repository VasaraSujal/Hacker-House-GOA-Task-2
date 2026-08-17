import { FiAlertTriangle, FiCheckCircle, FiMic, FiSlash } from 'react-icons/fi'
import type { VoiceRagResponse } from '../types/rag'

export function AnswerPanel({ result }: { result: VoiceRagResponse }) {
  const refused = result.refused
  const grounded = result.grounded && !refused

  return (
    <div className="space-y-4">
      <section className="result-card">
        <div className="section-label">
          <FiMic aria-hidden="true" />
          You asked
        </div>
        <blockquote className="mt-4 text-xl font-medium leading-8 text-white sm:text-2xl">
          “{result.transcript}”
        </blockquote>
      </section>

      <section
        className={`result-card ${refused ? 'border-amber-300/25' : 'border-emerald-300/20'}`}
        aria-live="polite"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="section-label">
            {refused ? <FiSlash aria-hidden="true" /> : <FiCheckCircle aria-hidden="true" />}
            {refused ? 'Knowledge-base refusal' : 'Grounded answer'}
          </div>
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs ${
              grounded ? 'bg-emerald-300/10 text-emerald-200' : 'bg-amber-300/10 text-amber-100'
            }`}
          >
            {grounded ? (
              <>
                <FiCheckCircle aria-hidden="true" />
                Grounded in retrieved context
              </>
            ) : (
              <>
                <FiAlertTriangle aria-hidden="true" />
                {refused ? 'Unable to answer from knowledge base' : 'Grounding not verified'}
              </>
            )}
          </span>
        </div>

        <p className="mt-5 whitespace-pre-wrap text-[17px] leading-8 text-slate-100">{result.answer}</p>

        {refused && (
          <p className="mt-4 border-t border-white/8 pt-4 text-sm text-amber-100/80">
            Grounding: Not available. Not enough relevant evidence was found in the knowledge base.
          </p>
        )}
      </section>
    </div>
  )
}
