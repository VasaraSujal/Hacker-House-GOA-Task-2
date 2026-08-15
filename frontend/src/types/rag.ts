export interface SourceDocument {
  text: string
  score: number
  document_id: string
  chunk_id: string
  metadata: Record<string, unknown>
}

export interface VoiceLatency {
  audio_validation_ms: number
  stt_ms: number
  transcript_validation_ms: number
  query_processing_ms: number
  embedding_ms: number
  dense_retrieval_ms: number
  bm25_ms: number
  retrieval_wall_ms: number
  fusion_ms: number
  relevance_guard_ms: number
  reranking_ms: number
  context_building_ms: number
  generation_ms: number
  grounding_ms: number
  rag_core_ms: number
  rag_ms: number
  total_ms: number
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

export type ExperienceState =
  | 'idle'
  | 'recording'
  | 'processing'
  | 'success'
  | 'refused'
  | 'error'
