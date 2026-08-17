import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { LatencyPanel } from './LatencyPanel'

describe('RAG latency panel', () => {
  it('uses rag_core_ms as the primary under-target metric', () => {
    render(
      <LatencyPanel
        latency={{ rag_core_ms: 169.39, stt_ms: 889, total_ms: 1058.21 }}
      />,
    )

    expect(screen.getByText('169 ms')).toBeInTheDocument()
    expect(screen.getByText('Under 200 ms target')).toBeInTheDocument()
    expect(screen.getByText('889 ms')).toBeInTheDocument()
    expect(screen.getByText('1.06 s')).toBeInTheDocument()
  })

  it('honestly reports an above-target RAG measurement', () => {
    render(
      <LatencyPanel
        latency={{ rag_core_ms: 296.12, stt_ms: 120, total_ms: 420 }}
      />,
    )

    expect(screen.getByText('296 ms')).toBeInTheDocument()
    expect(screen.getByText('Above 200 ms target')).toBeInTheDocument()
  })

  it('does not substitute STT or total latency when rag_core_ms is missing', () => {
    render(<LatencyPanel latency={{ stt_ms: 169.39, total_ms: 296.12 }} />)

    expect(screen.getByText('RAG latency unavailable')).toBeInTheDocument()
    expect(screen.queryByText('Under 200 ms target')).not.toBeInTheDocument()
    expect(screen.queryByText('Above 200 ms target')).not.toBeInTheDocument()
    expect(screen.getByText('169 ms')).toBeInTheDocument()
    expect(screen.getByText('296 ms')).toBeInTheDocument()
  })

  it('keeps component timings behind an accessible disclosure', async () => {
    const user = userEvent.setup()
    render(
      <LatencyPanel
        latency={{
          rag_core_ms: 169.39,
          embedding_ms: 160.25,
          bm25_ms: 1.1,
          total_ms: 1058.21,
        }}
      />,
    )

    expect(screen.queryByText('Embedding')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Show pipeline timings' }))
    expect(screen.getByText('Embedding')).toBeInTheDocument()
    expect(screen.getByText('BM25')).toBeInTheDocument()
  })
})
