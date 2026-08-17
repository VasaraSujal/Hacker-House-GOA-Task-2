import type { VoiceRagResponse } from '../types/rag'

export function RequestMeta({ result }: { result: VoiceRagResponse }) {
  const rows = [
    ['Request ID', result.request_id || 'Not reported'],
    ['Evidence passages', String(result.sources.length)],
    ['Grounded', result.grounded && !result.refused ? 'Yes' : 'No'],
    ['Retrieval', 'Dense + BM25'],
    ['Answer mode', 'Extractive'],
    ['Knowledge base', 'MSMARCO-XI validation subset'],
  ]

  return (
    <section className="rounded-2xl border border-white/8 px-4 py-4 text-xs text-slate-500">
      <p className="font-mono uppercase tracking-[0.18em]">Request details</p>
      <dl className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt>{label}</dt>
            <dd className="mt-1 truncate font-mono text-slate-400">{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
