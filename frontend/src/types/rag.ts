export interface SourceDocument {
  text: string
  score: number
  document_id: string
  chunk_id: string
  metadata: Record<string, unknown>
}

export interface VoiceLatency {
  request_parsing_ms?: number | null
  query_processing_ms?: number | null
  embedding_ms?: number | null
  dense_retrieval_ms?: number | null
  bm25_ms?: number | null
  retrieval_wall_ms?: number | null
  fusion_ms?: number | null
  relevance_guard_ms?: number | null
  reranking_ms?: number | null
  context_building_ms?: number | null
  generation_ms?: number | null
  grounding_ms?: number | null
  rag_core_ms?: number | null
  component_sum_ms?: number | null
  unaccounted_ms?: number | null
  total_ms?: number | null
  audio_validation_ms?: number | null
  stt_ms?: number | null
  transcript_validation_ms?: number | null
  rag_ms?: number | null
}

export interface VoiceRagResponse {
  transcript: string
  answer: string
  sources: SourceDocument[]
  grounded: boolean
  refused: boolean
  request_id: string
  latency: VoiceLatency
}

export interface HealthResponse {
  status: string
  qdrant: string
  embeddings: string
  bm25: string
  elevenlabs_configured: boolean
  stt_configured: boolean
}

export interface WarmupResponse {
  status: string
  warmup_ms: number
  qdrant: string
  bm25: string
  retrieval_mode: string
}

export type SystemStatus =
  | 'preparing'
  | 'warming'
  | 'ready'
  | 'degraded'
  | 'unavailable'

export type ExperienceState =
  | 'idle'
  | 'recording'
  | 'processing'
  | 'success'
  | 'refused'
  | 'error'

