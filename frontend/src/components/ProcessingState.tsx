export function ProcessingState() {
  return (
    <section className="result-card flex items-center gap-4" aria-live="polite">
      <span className="relative flex size-10 shrink-0 items-center justify-center">
        <span className="absolute size-10 animate-ping rounded-full bg-cyan-300/10" />
        <span className="size-3 rounded-full bg-cyan-300" />
      </span>
      <div>
        <p className="font-medium text-white">Processing your question</p>
        <p className="mt-1 text-sm text-slate-500">
          The backend is transcribing, retrieving, generating, and checking grounding. This usually takes a few seconds.
        </p>
      </div>
    </section>
  )
}
