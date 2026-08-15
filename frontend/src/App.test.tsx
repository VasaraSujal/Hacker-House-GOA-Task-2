import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { getHealth, submitVoiceQuery } from './api/ragApi'
import type { VoiceRagResponse } from './types/rag'

vi.mock('./api/ragApi', () => ({
  getHealth: vi.fn(),
  submitVoiceQuery: vi.fn(),
}))

const response = (refused = false): VoiceRagResponse => ({
  transcript: refused ? "Who won yesterday's cricket match?" : 'What is a corporation?',
  answer: refused ? 'I could not find enough relevant information in the knowledge base.' : 'A corporation is a legal entity.',
  sources: refused
    ? []
    : [{ text: 'Supporting context', score: 0.91, document_id: 'doc-1', chunk_id: 'chunk-1', metadata: {} }],
  grounded: !refused,
  refused,
  request_id: 'req-1',
  latency: {
    audio_validation_ms: 1,
    stt_ms: 1000,
    transcript_validation_ms: 1,
    query_processing_ms: 2,
    embedding_ms: 10,
    dense_retrieval_ms: 12,
    bm25_ms: 8,
    retrieval_wall_ms: 20,
    fusion_ms: 2,
    relevance_guard_ms: 2,
    reranking_ms: 10,
    context_building_ms: 2,
    generation_ms: 3000,
    grounding_ms: 50,
    rag_core_ms: 48,
    rag_ms: 3100,
    total_ms: 4200,
  },
})

async function recordQuestion() {
  const user = userEvent.setup()
  await user.click(await screen.findByRole('button', { name: 'Start voice recording' }))
  const stop = await screen.findByRole('button', { name: 'Stop voice recording' })
  await user.click(stop)
}

describe('voice experience', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    } as unknown as MediaStream)
    vi.mocked(getHealth).mockResolvedValue({
      status: 'ok',
      qdrant: 'ok',
      embeddings: 'ok',
      bm25: 'ok',
      elevenlabs_configured: true,
      stt_configured: true,
    })
  })

  it('moves through recording and renders a grounded result', async () => {
    vi.mocked(submitVoiceQuery).mockResolvedValue(response())
    render(<App />)

    await recordQuestion()

    expect(await screen.findByText('“What is a corporation?”')).toBeInTheDocument()
    expect(screen.getByText('A corporation is a legal entity.')).toBeInTheDocument()
    expect(screen.getByText('Grounded in retrieved context')).toBeInTheDocument()
    expect(screen.getByText('4.20 s')).toBeInTheDocument()
  })

  it('shows a knowledge-base refusal as a valid result', async () => {
    vi.mocked(submitVoiceQuery).mockResolvedValue(response(true))
    render(<App />)

    await recordQuestion()

    expect(await screen.findByText('Knowledge-base refusal')).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
  })

  it('shows an application error when the request fails', async () => {
    vi.mocked(submitVoiceQuery).mockRejectedValue(new Error('Unable to connect to the RAG backend. Please try again.'))
    render(<App />)

    await recordQuestion()

    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to connect to the RAG backend')
  })

  it('explains a denied microphone permission', async () => {
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(
      new DOMException('Permission denied', 'NotAllowedError'),
    )
    render(<App />)

    await userEvent.click(screen.getByRole('button', { name: 'Start voice recording' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Microphone access was denied')
    expect(submitVoiceQuery).not.toHaveBeenCalled()
  })
})
