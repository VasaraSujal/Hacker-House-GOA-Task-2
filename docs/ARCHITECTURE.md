# Architecture

## Live request flow

```text
Chrome microphone
  → React / Vite on Vercel (HTTPS)
  → FastAPI on Render (HTTPS)
  → ElevenLabs Scribe v2
  → validated transcript
  → hybrid retrieval
      ├─ dense MiniLM query embedding → Qdrant Cloud
      └─ persisted in-memory BM25
  → Reciprocal Rank Fusion
  → optional bounded reranker
  → context builder
  → ElevenLabs generation
  → grounding guard
  → transcript + answer + sources + latency
  → React result UI
```

The browser communicates only with FastAPI. ElevenLabs and Qdrant credentials
remain server-side.

## Runtime boundaries

### Frontend

- Static React 19 bundle hosted by Vercel
- Native MediaRecorder; WebM/Opus preferred when supported
- Only public build variable: `VITE_API_BASE_URL`
- No fake streaming; one final response per request

### Backend

- One Uvicorn worker to avoid duplicating model/BM25 memory
- MiniLM model loaded once and cached
- BM25 pickle loaded once at startup
- Shared Qdrant and ElevenLabs HTTP clients
- Explicit CORS allow-list and in-process voice rate limiter
- Production startup fails if required retrieval/STT dependencies are missing

### Storage

- Qdrant Cloud: 11,478 vectors, 384 dimensions, Cosine distance, payload text
- Container image: persisted BM25 index for the same 11,478 chunk IDs
- Local Docker Qdrant and snapshot remain the rollback/reference copy

## Dataset disclosure

The current live target uses a validated subset of
`ai4bharat/MSMARCO-XI`, configuration `hi`, validation split:

- 500 query records
- 9,989 English/Hindi passages
- 11,478 chunks

The full approximately 55–56 GB snapshot is not indexed. Stage 5 projected
hundreds of GiB of vector/sparse storage and weeks of local CPU ingestion.
The loader, chunker, embedding, and Qdrant upsert paths remain streaming and
batched for a future larger environment.

## Failure behavior

- Invalid/oversized audio: controlled 4xx response
- Voice request limit: 429
- ElevenLabs timeout/rate limit: bounded retries and controlled response
- Qdrant query failure: controlled RAG error
- Missing production dependency at startup: process fails before becoming ready
- Grounding failure/irrelevant retrieval: successful knowledge-base refusal
