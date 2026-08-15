import { FiMic, FiSquare, FiX } from 'react-icons/fi'
import type { ExperienceState } from '../types/rag'

interface VoiceRecorderProps {
  state: ExperienceState
  elapsedSeconds: number
  supported: boolean
  onStart: () => void
  onStop: () => void
  onCancel: () => void
}

function formatTime(seconds: number) {
  return `00:${seconds.toString().padStart(2, '0')}`
}

export function VoiceRecorder({
  state,
  elapsedSeconds,
  supported,
  onStart,
  onStop,
  onCancel,
}: VoiceRecorderProps) {
  const recording = state === 'recording'
  const processing = state === 'processing'

  return (
    <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/70 px-6 py-10 text-center sm:px-10">
      <div className="ambient-orb" aria-hidden="true" />
      <div className="relative mx-auto flex max-w-xl flex-col items-center">
        <p className="mb-5 font-mono text-[11px] uppercase tracking-[0.3em] text-slate-500">
          {recording ? 'Listening now' : processing ? 'Voice pipeline active' : 'Ask the knowledge base'}
        </p>

        <button
          type="button"
          disabled={processing || !supported}
          onClick={recording ? onStop : onStart}
          aria-label={recording ? 'Stop voice recording' : 'Start voice recording'}
          className={`mic-button group ${recording ? 'is-recording' : ''} ${processing ? 'is-processing' : ''}`}
        >
          <span className="mic-ripple" aria-hidden="true" />
          {processing ? (
            <span className="size-9 animate-spin rounded-full border-2 border-slate-600 border-t-cyan-300" />
          ) : recording ? (
            <FiSquare className="size-8" aria-hidden="true" />
          ) : (
            <FiMic className="size-10 transition-transform group-hover:scale-105" aria-hidden="true" />
          )}
        </button>

        <h1 className="mt-7 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          {recording ? 'Recording your question' : processing ? 'Finding a grounded answer' : 'What would you like to know?'}
        </h1>
        <p className="mt-3 max-w-md text-sm leading-6 text-slate-400">
          {recording
            ? 'Speak naturally, then stop when you are finished.'
            : processing
              ? 'Transcribing, retrieving context, generating, and validating your answer.'
              : 'Your audio is transcribed by ElevenLabs, then answered only from retrieved context.'}
        </p>

        {recording && (
          <div className="mt-6 flex items-center gap-4">
            <span className="rounded-full bg-rose-400/10 px-4 py-2 font-mono text-sm text-rose-300">
              <span className="mr-2 inline-block size-2 animate-pulse rounded-full bg-rose-400" />
              {formatTime(elapsedSeconds)}
            </span>
            <button
              type="button"
              onClick={onCancel}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm text-slate-400 transition hover:border-white/20 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
            >
              <FiX aria-hidden="true" /> Cancel
            </button>
          </div>
        )}

        {!recording && !processing && (
          <span className="mt-6 text-xs text-slate-500">
            {supported ? 'Click the microphone to begin · 30 second limit' : 'MediaRecorder is not supported in this browser'}
          </span>
        )}
      </div>
    </section>
  )
}
