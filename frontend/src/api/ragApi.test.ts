import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, submitVoiceQuery } from './ragApi'
import type { VoiceRagResponse } from '../types/rag'

afterEach(() => vi.restoreAllMocks())

describe('submitVoiceQuery', () => {
  it('sends browser audio as multipart form data', async () => {
    const payload: VoiceRagResponse = {
      transcript: 'What is a corporation?',
      answer: 'A corporation is a legal entity.',
      sources: [{ text: 'Evidence', score: 0.9, document_id: 'doc-1', chunk_id: 'chunk-1', metadata: {} }],
      grounded: true,
      refused: false,
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
        reranking_ms: 0,
        context_building_ms: 2,
        generation_ms: 3000,
        grounding_ms: 50,
        rag_core_ms: 48,
        rag_ms: 3100,
        total_ms: 4200,
      },
    }
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    )

    const result = await submitVoiceQuery(new Blob(['audio'], { type: 'audio/webm;codecs=opus' }))

    const [, options] = fetchMock.mock.calls[0]
    if (!options || !(options.body instanceof FormData)) {
      throw new Error('Expected a multipart request')
    }
    expect(options.method).toBe('POST')
    expect(options.body.get('audio')).toBeInstanceOf(File)
    expect(result).toMatchObject({
      transcript: 'What is a corporation?',
      answer: 'A corporation is a legal entity.',
      grounded: true,
      refused: false,
    })
    expect(result.sources).toHaveLength(1)
    expect(result.latency.total_ms).toBe(4200)
  })

  it('maps rate limits to a safe user-facing error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: { error: 'private provider detail', code: 'stt_rate_limited' } }), {
        status: 429,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await expect(submitVoiceQuery(new Blob(['audio'], { type: 'audio/webm' }))).rejects.toMatchObject({
      status: 429,
      message: 'The service is temporarily rate-limited. Please try again shortly.',
    } satisfies Partial<ApiError>)
  })
})
