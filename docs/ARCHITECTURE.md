# Architecture

## Two runtime profiles

The system supports a full hybrid local RAG mode and a memory-constrained
production mode for Render Free (~512 MB RAM).

### Full local mode

```text
Chrome microphone
  → React / Vite
  → FastAPI
  → ElevenLabs Scribe v2
  → local MiniLM query embedding
  → Qdrant (local or cloud copy of hh_goa_rag)
  + persisted BM25
  → Reciprocal Rank Fusion
  → optional CrossEncoder reranker
  → ElevenLabs generative answer
  → grounding guard
  → transcript + answer + sources + latency
```

Config:

```text
RETRIEVAL_MODE=local
ANSWER_MODE=generative
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Render Free mode

```text
Chrome microphone
  → React / Vite on Vercel (HTTPS)
  → FastAPI on Render Free (HTTPS)
  → ElevenLabs Scribe v2
  → Qdrant Cloud hosted inference (dense)
  + persisted BM25
  → Reciprocal Rank Fusion
  → lexical light reranker
  → extractive grounded answer
  → grounding guard
  → transcript + answer + sources + latency
```

Config:

```text
RETRIEVAL_MODE=cloud_dense_sparse
ANSWER_MODE=extractive
QDRANT_COLLECTION=hh_goa_voice_rag_prod
QDRANT_INFERENCE_MODEL=intfloat/multilingual-e5-small
```

Render Free must not load PyTorch or SentenceTransformers. Dense embedding
inference runs inside Qdrant Cloud. The browser communicates only with FastAPI.
ElevenLabs and Qdrant credentials remain server-side.

## Why a separate production collection

Local vectors were embedded with
`paraphrase-multilingual-MiniLM-L12-v2` (384-d, Cosine, normalized). That exact
model is not available on Qdrant Cloud Inference. Free hosted models on the
cluster include `intfloat/multilingual-e5-small` and
`sentence-transformers/all-MiniLM-L6-v2` (both 384-d).

Document and query embeddings must use the same model. Therefore production
uses a dedicated collection (`hh_goa_voice_rag_prod`) rebuilt with hosted
inference. Local `hh_goa_rag` is never overwritten.

## Runtime boundaries

### Frontend

- Static React 19 bundle hosted by Vercel
- Native MediaRecorder; WebM/Opus preferred when supported
- Only public build variable: `VITE_API_BASE_URL`
- No fake streaming; one final response per request

### Backend

- One Uvicorn worker (`WEB_CONCURRENCY=1`)
- Local mode: MiniLM loaded once and cached
- Free mode: no local embedding model; Qdrant Cloud Document inference
- BM25 pickle loaded once at startup
- Shared Qdrant and ElevenLabs HTTP clients
- Explicit CORS allow-list and in-process voice rate limiter
- Production startup fails if required retrieval/STT dependencies are missing

### Storage

- Local/reference: `hh_goa_rag`, 11,478 vectors, paraphrase-multilingual MiniLM
- Production Free: `hh_goa_voice_rag_prod`, 11,478 vectors, multilingual-e5-small
  via Qdrant Cloud Inference
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
