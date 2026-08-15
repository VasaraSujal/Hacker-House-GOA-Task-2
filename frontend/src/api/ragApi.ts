import type { HealthResponse, VoiceRagResponse } from '../types/rag'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  readonly status: number
  readonly code?: string

  constructor(
    message: string,
    status: number,
    code?: string,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function filenameForMime(mimeType: string): string {
  if (mimeType.includes('webm')) return 'voice-query.webm'
  if (mimeType.includes('mp4')) return 'voice-query.m4a'
  if (mimeType.includes('mpeg')) return 'voice-query.mp3'
  return 'voice-query.wav'
}

async function parseError(response: Response): Promise<ApiError> {
  let message = 'The voice pipeline encountered an error. Please try again.'
  let code: string | undefined
  try {
    const payload = await response.json()
    const detail = payload.detail ?? payload
    message = detail.error ?? detail.detail ?? message
    code = detail.code
  } catch {
    // Keep the safe public fallback.
  }

  if (response.status === 413) message = 'Audio file is too large.'
  if (response.status === 429) message = 'The service is temporarily rate-limited. Please try again shortly.'
  if (response.status >= 500) message = 'The voice pipeline encountered an error. Please try again.'
  return new ApiError(message, response.status, code)
}

export async function submitVoiceQuery(audioBlob: Blob): Promise<VoiceRagResponse> {
  const form = new FormData()
  form.append('audio', audioBlob, filenameForMime(audioBlob.type))
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/voice/query`, {
      method: 'POST',
      body: form,
    })
  } catch {
    throw new ApiError('Unable to connect to the RAG backend. Please try again.', 0)
  }
  if (!response.ok) throw await parseError(response)
  return response.json() as Promise<VoiceRagResponse>
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw await parseError(response)
  return response.json() as Promise<HealthResponse>
}
