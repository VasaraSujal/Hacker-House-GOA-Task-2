import { FiCheckCircle, FiMessageSquare, FiShield, FiSlash } from 'react-icons/fi'
import type { VoiceRagResponse } from '../types/rag'

export function AnswerPanel({ result }: { result: VoiceRagResponse }) {
  return (
    <div className="space-y-4">
      <section className="result-card">
        <div className="section-label">
          <FiMessageSquare aria-hidden="true" />
          You asked
        </div>
        <blockquote className="mt-4 text-xl font-medium leading-8 text-white">
          “{result.transcript}”
        </blockquote>
      </section>

      <section className={`result-card ${result.refused ? 'border-amber-300/20' : 'border-emerald-300/15'}`}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="section-label">
            {result.refused ? <FiSlash aria-hidden="true" /> : <FiShield aria-hidden="true" />}
            {result.refused ? 'Knowledge-base refusal' : 'Grounded answer'}
          </div>
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs ${
              result.grounded ? 'bg-emerald-300/10 text-emerald-300' : 'bg-amber-300/10 text-amber-200'
            }`}
          >
            <FiCheckCircle aria-hidden="true" />
            {result.grounded ? 'Grounded in retrieved context' : 'Grounding not verified'}
          </span>
        </div>
        <p className="mt-5 whitespace-pre-wrap text-[15px] leading-7 text-slate-200">{result.answer}</p>
      </section>
    </div>
  )
}
