import { useCallback, useEffect, useState } from 'react'
import { FiAlertCircle, FiArrowRight, FiRefreshCw } from 'react-icons/fi'
import { getHealth, submitVoiceQuery } from './api/ragApi'
import { AnswerPanel } from './components/AnswerPanel'
import { Header } from './components/Header'
import { LatencyPanel } from './components/LatencyPanel'
import { ProcessingState } from './components/ProcessingState'
import { SourceList } from './components/SourceList'
import { VoiceRecorder } from './components/VoiceRecorder'
import { useVoiceRecorder } from './hooks/useVoiceRecorder'
import type { ExperienceState, VoiceRagResponse } from './types/rag'

function App() {
  const [state, setState] = useState<ExperienceState>('idle')
  const [result, setResult] = useState<VoiceRagResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [connected, setConnected] = useState<boolean | null>(null)

  const checkHealth = useCallback(() => {
    setConnected(null)
    getHealth()
      .then((health) => setConnected(health.status === 'ok' && health.stt_configured))
      .catch(() => setConnected(false))
  }, [])

  useEffect(checkHealth, [checkHealth])

  const handleRecording = useCallback(async (blob: Blob) => {
    setState('processing')
    setError(null)
    try {
      const response = await submitVoiceQuery(blob)
      setResult(response)
      setState(response.refused ? 'refused' : 'success')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Something went wrong. Please try again.')
      setState('error')
    }
  }, [])

  const recorder = useVoiceRecorder({ onRecordingReady: handleRecording, maxDurationSeconds: 30 })

  useEffect(() => {
    if (recorder.error) {
      setError(recorder.error)
      setState('error')
    }
  }, [recorder.error])

  const start = async () => {
    setResult(null)
    setError(null)
    recorder.setError(null)
    setState('idle')
    await recorder.startRecording()
  }

  useEffect(() => {
    if (recorder.isRecording) setState('recording')
  }, [recorder.isRecording])

  return (
    <div className="min-h-screen bg-[#06090f] text-slate-200">
      <div className="noise-layer" aria-hidden="true" />
      <div className="relative mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8">
        <Header connected={connected} />

        <main className="py-8 sm:py-12">
          <div className="mb-8 grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.26em] text-cyan-300">Voice-first retrieval</p>
              <h2 className="mt-3 max-w-3xl text-4xl font-semibold tracking-[-0.04em] text-white sm:text-6xl">
                Ask. Retrieve. Answer with evidence.
              </h2>
            </div>
            <p className="max-w-sm text-sm leading-6 text-slate-500">
              ElevenLabs speech recognition meets hybrid retrieval and transparent grounding. Every result shows its sources and real request latency.
            </p>
          </div>

          <VoiceRecorder
            state={state}
            elapsedSeconds={recorder.elapsedSeconds}
            supported={recorder.supported}
            onStart={start}
            onStop={recorder.stopRecording}
            onCancel={() => {
              recorder.cancelRecording()
              setState('idle')
            }}
          />

          {error && (
            <section className="mt-5 flex flex-col gap-4 rounded-2xl border border-rose-300/15 bg-rose-300/5 p-5 sm:flex-row sm:items-center" role="alert">
              <FiAlertCircle className="size-6 shrink-0 text-rose-300" aria-hidden="true" />
              <div className="flex-1">
                <p className="font-medium text-rose-100">Something went wrong</p>
                <p className="mt-1 text-sm text-rose-200/65">{error}</p>
              </div>
              <button type="button" onClick={start} className="secondary-button">
                <FiRefreshCw aria-hidden="true" /> Try again
              </button>
            </section>
          )}

          {state === 'processing' && <div className="mt-5"><ProcessingState /></div>}

          {result && (state === 'success' || state === 'refused') && (
            <div className="mt-6 grid items-start gap-5 lg:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.75fr)]">
              <div className="min-w-0 space-y-5">
                <AnswerPanel result={result} />
                <SourceList sources={result.sources} />
              </div>
              <div className="min-w-0 space-y-5 lg:sticky lg:top-5">
                <LatencyPanel latency={result.latency} />
                <button type="button" onClick={start} className="primary-button w-full">
                  Ask another question <FiArrowRight aria-hidden="true" />
                </button>
              </div>
            </div>
          )}

          {!result && state === 'idle' && !error && (
            <section className="mt-6 grid gap-3 sm:grid-cols-3">
              {['What is a corporation?', 'What is benthos?', 'What is Bayern Munich?'].map((question) => (
                <div key={question} className="rounded-xl border border-white/8 bg-white/[0.025] px-4 py-4">
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-slate-600">Try asking</p>
                  <p className="mt-2 text-sm text-slate-400">“{question}”</p>
                </div>
              ))}
            </section>
          )}
        </main>

        <footer className="flex flex-col gap-2 border-t border-white/8 py-6 text-xs text-slate-600 sm:flex-row sm:justify-between">
          <span>HH Goa 2026 · Voice RAG</span>
          <span>Actual request timings · No fake streaming · Grounded responses</span>
        </footer>
      </div>
    </div>
  )
}

export default App
