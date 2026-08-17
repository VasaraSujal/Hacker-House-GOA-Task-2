export function ProcessingState() {
  const stages = ['Transcribing', 'Retrieving knowledge', 'Grounding answer']

  return (
    <section className="result-card" aria-live="polite" aria-busy="true">
      <p className="font-medium text-white">Processing your question</p>
      <p className="mt-1 text-sm text-slate-500">
        The backend is transcribing, retrieving, and checking grounding. No estimated percentage is shown.
      </p>
      <ol className="mt-5 space-y-3">
        {stages.map((stage, index) => (
          <li key={stage} className="flex items-center gap-3 text-sm text-slate-300">
            <span className="relative flex size-5 shrink-0 items-center justify-center">
              <span className="absolute size-5 animate-ping rounded-full bg-cyan-300/10" />
              <span className="size-2 rounded-full bg-cyan-300" />
            </span>
            <span>
              {index + 1}. {stage}…
            </span>
          </li>
        ))}
      </ol>
    </section>
  )
}
