# HH Goa 2026 — Voice-Enabled RAG (Stage 6B)

Production-grade **voice RAG application** for Hacker House Goa 2026 Shortlisting Task 2. Stages 1–5 are verified locally. Stage 6B adds a Render Free profile that keeps the full local RAG intact while moving dense embedding inference to Qdrant Cloud and using extractive answers so the API can run within ~512 MB RAM.

**Modes:**

- **Full local:** `RETRIEVAL_MODE=local`, `ANSWER_MODE=generative` (Torch MiniLM + optional ElevenLabs generation)
- **Render Free:** `RETRIEVAL_MODE=cloud_dense_sparse`, `ANSWER_MODE=extractive` (Qdrant Cloud inference + BM25 + lexical rerank; no Torch)

See `docs/DEPLOYMENT.md`, `docs/ARCHITECTURE.md`, and `benchmarks/STAGE6B_FREE_DEPLOYMENT_REPORT.md`.

## HH Goa Task 2 requirements

The full submission needs voice-enabled RAG over MSMARCO-XI, ElevenLabs STT, multiple chunking strategies, vector retrieval, hybrid search, latency percentiles, harnessing, retries, guardrails, and grounded refusals.

The voice route transcribes audio, then delegates to the unchanged text boundary: `RAGPipeline.run(transcript)`.

## Current Stage 5 scope

Implemented through Stage 5:

- MSMARCO-XI inspection (streaming; no 55 GB RAM load)
- Batched / resumable ingestion with bounded batch retries and progress metrics
- Isolated ingestion scale harness (`scripts/run_scale_ingest.py`)
- Capacity estimates with MEASURED vs ESTIMATE separation
- Fixed, sentence, semantic, and metadata-aware chunking
- Local multilingual embeddings
- Qdrant dense retrieval
- BM25 sparse retrieval
- Reciprocal Rank Fusion (configurable weighted fusion)
- Local cross-encoder reranking
- Context builder
- ElevenLabs LLM generation (abstracted `LLMProvider`)
- Input, relevance, and grounding guardrails
- FastAPI `POST /api/rag/query`, `POST /api/voice/query`, and `GET /health`
- Voice endpoint rate limiting and Qdrant timeout configuration
- Retrieval metrics (Recall@K, P@K, MRR, nDCG)
- P50 / P70 / P100 latency harness
- ElevenLabs `scribe_v2` batch STT with bounded transient retries
- Multipart audio validation (WAV, MP3, M4A, WebM; configurable size cap)
- Tests with mocked external APIs plus real ElevenLabs STT smoke tests
- React, Vite, TypeScript, and Tailwind CSS frontend
- Native MediaRecorder capture with detected browser MIME type and track cleanup
- Transcript, answer, sources, grounding/refusal, and actual latency UI
- Explicit development CORS origins and real Chrome voice-flow automation

Not claimed yet: a public HTTPS demo URL. Full 55–56 GB ingestion, social posting, submission form, and TTS playback remain out of scope until Stage 6 public deploy is unblocked.

## Architecture

```text
Text Query
    → Input Guard + query preprocessing
    → Dense Retrieval (local embeddings + Qdrant)
    +  BM25
    → Hybrid fusion (RRF)
    → Relevance guard (skip LLM if evidence is weak)
    → Optional local reranker (disabled in latency-first profile)
    → Context builder
    → ElevenLabs LLM
    → Grounding guard
    → Structured RAGResponse
```

Stage 4 browser flow:

```text
Microphone → MediaRecorder → multipart /api/voice/query → ElevenLabs STT
  → RAGPipeline → ElevenLabs generation → grounding → React result UI
```

## Technology stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn + Pydantic |
| Dataset | Hugging Face `datasets` (streaming parquet) |
| Embeddings | Sentence Transformers (configurable) |
| Vectors | Qdrant (Docker) |
| Sparse | `rank-bm25` |
| Rerank | local CrossEncoder |
| LLM | ElevenLabs Agents API |
| Eval | NumPy / scikit-learn-style metrics |
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS |

LangChain / LangGraph are not used. Orchestration is a small custom `RAGPipeline`.

## Dataset

[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) is MS MARCO translated into 14 Indic languages. Each row is a **query** with candidate **passages**, not a standalone document dump.

Typical fields (confirmed by `scripts/inspect_dataset.py`):

- `query`, `Eng_Query`, `Answer`, `Eng_Answer`, `query_id`, `query_type`
- `source_lang`, `target_lang`
- `passages.is_selected`, `passages.English_passages`, `passages.Translated_passages`

Relevance labels are `passages.is_selected`. Document IDs are derived from `query_id + passage index + language + text hash` because the dataset does not provide global passage IDs.

## Dataset size

The Hub snapshot is about **55.6 GB**. Train per language is on the order of ~778k examples; validation ~97k. Do not `list(dataset)` or `load_dataset(...)` without streaming.

Ingestion streams parquet files (`validation/hinval.parquet`, etc.) in batches, embeds a batch, upserts a batch, and writes a checkpoint.

## Dataset inspection

```powershell
cd backend
python scripts/inspect_dataset.py
```

The script reports Hub files, language configs, schema, samples, and a capped count. It does not download the full corpus.

**Development subset:** Hindi (`hi`) **validation** split, `INGEST_MODE=subset`, `MAX_DOCUMENTS=500` query records (about 5k passages if both English and translated passages are indexed). Hindi is the dataset default and includes English originals plus Indic translations, with `is_selected` labels for evaluation.

## Ingestion

```powershell
docker compose up -d
cd backend
python scripts/ingest.py
```

Flow: stream → clean → chunk → batch embed → Qdrant upsert → BM25 update → checkpoint.

`INGEST_MODE=full` streams an entire split without a record cap. It still does not load 55 GB into RAM.

Interrupted jobs resume from `data/checkpoints/ingest.json`.

## Chunking strategies

Configured with `CHUNK_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.

| Strategy | Behavior |
| --- | --- |
| `fixed` | Character windows with overlap (e.g. 300 / 500 / 800) |
| `sentence` | Sentence boundaries (Latin `.?!` and Indic `।॥`) packed to max size |
| `semantic` | Consecutive sentences grouped while Jaccard similarity stays above a threshold |
| `metadata` | Same body as sentence; every chunk still carries document metadata |

Every chunk stores `document_id`, `chunk_id`, `language`, `chunk_strategy`, `position`, and `text`.

## Embedding model

`EMBEDDING_MODEL` is the single source of truth. The Qdrant collection dimension is taken from the loaded model.

| Candidate | Dim | Speed | Memory | Indic / multilingual | Notes |
| --- | --- | --- | --- | --- | --- |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | ~420 MB | 50+ languages including Hindi | Current indexed model |
| `intfloat/multilingual-e5-small` | 384 | Fast | ~470 MB | 100 languages | Measured Stage 2 recommendation; requires re-index |
| `BAAI/bge-m3` | 1024 | Slow | ~2.2 GB | 100+ languages | Strong candidate; heavier than the current CPU budget |

The encoder is loaded once and reused. Embeddings are L2-normalized when `EMBEDDING_NORMALIZE=true`.

## Qdrant

```powershell
docker compose up -d
```

Default: `http://localhost:6333`, collection `hh_goa_rag`, cosine distance. Payloads store chunk text, document/chunk metadata, language, and strategy. Upserts are batched; there is no per-chunk HTTP call.

## BM25

`rank-bm25` over the same development chunks, persisted at `data/indexes/bm25.pkl`. Unicode word tokens. The class is a narrow interface so a later sparse engine can replace it.

## Hybrid retrieval

Dense top-K and BM25 top-K are fused with **Reciprocal Rank Fusion** (default) or min-max **weighted** fusion. Results are not concatenated. Tunables: `DENSE_TOP_K`, `BM25_TOP_K`, `HYBRID_TOP_K`, `FUSION_METHOD`, `RRF_K`.

## Reranking

Local cross-encoder, loaded once when `ENABLE_RERANKER=true`. The latency-first profile disables it after measurement: it improved MRR from 0.3996 to 0.5369 but added roughly 805 ms mean retrieval latency. The configured model remains `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

## ElevenLabs generation

`LLMProvider` → `ElevenLabsProvider`. Uses the ElevenLabs Agents API (`/v1/convai/agents/create` and `/v1/convai/agents/{id}/simulate-conversation`) with a text-only agent, prompt override containing retrieved context, retries on 429/5xx, and timeouts. Model is `ELEVENLABS_MODEL` (default `gemini-2.0-flash` for latency).

The API key stays in `.env`. It is never logged.

STT is a separate interface (`rag/stt`) and is not wired in this stage.

## Guardrails

1. **Input** — empty, non-string, oversized, punctuation-only queries.
2. **Relevance** — skip the LLM when retrieval is empty or below `RELEVANCE_MIN_SCORE`.
3. **Grounding** — lexical token overlap between answer and context; explicit refusals count as grounded; unsupported answers are replaced with the refusal message.

Refusal text:

`I couldn't find enough relevant information in the provided knowledge base to answer this question.`

## API

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`POST /api/rag/query`

```json
{ "query": "What is the capital of France?" }
```

```json
{
  "query": "...",
  "answer": "...",
  "sources": [],
  "grounded": true,
  "refused": false,
  "latency": {
    "request_parsing_ms": 0,
    "query_processing_ms": 0,
    "embedding_ms": 0,
    "dense_retrieval_ms": 0,
    "bm25_ms": 0,
    "retrieval_wall_ms": 0,
    "fusion_ms": 0,
    "relevance_guard_ms": 0,
    "reranking_ms": 0,
    "context_building_ms": 0,
    "generation_ms": 0,
    "grounding_ms": 0,
    "rag_core_ms": 0,
    "component_sum_ms": 0,
    "unaccounted_ms": 0,
    "total_ms": 0
  }
}
```

`GET /health` reports process liveness plus Qdrant / embedding / BM25 / ElevenLabs configuration.

## Stage 3 — Voice backend

```text
Audio upload
  → validate filename, MIME type, and configured size
  → ElevenLabs Speech-to-Text (scribe_v2)
  → existing RAGPipeline.run(transcript)
  → ElevenLabs generation
  → grounding guard
  → structured voice response
```

`POST /api/voice/query` accepts one multipart file field named `audio`.
Supported upload extensions are WAV, MP3, M4A, and WebM, with matching
ElevenLabs-compatible MIME types. The default cap is `MAX_AUDIO_SIZE_MB=20`.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/voice/query" `
  -F "audio=@C:\path\to\question.wav;type=audio/wav"
```

The response includes `transcript`, the normal grounded answer/source fields,
and `audio_validation_ms`, `stt_ms`, `transcript_validation_ms`, every existing
RAG component timing, `rag_ms`, and `total_ms`. Swagger documents the endpoint
at `/docs`.

Real run evidence (20 repeated requests over one short WAV, 2026-08-15):

| Component | P50 | P70 | P100 |
| --- | ---: | ---: | ---: |
| ElevenLabs STT | 1,022.52 ms | 1,053.97 ms | 1,480.71 ms |
| RAG (including generation) | 3,177.44 ms | 3,313.75 ms | 4,763.56 ms |
| ElevenLabs generation | 3,037.46 ms | 3,180.69 ms | 4,490.44 ms |
| Full voice pipeline | 4,201.59 ms | 4,419.22 ms | 6,244.36 ms |

This is a warm, repeated-audio benchmark and not a production latency claim.
The full voice pipeline does **not** meet 200 ms; generation is the primary
bottleneck and STT is the secondary external bottleneck.

## Stage 4 — Frontend

The frontend lives in `frontend/`. It requests microphone permission only after
the user clicks the microphone, chooses a browser-supported WebM/MP4/WAV MIME
type, limits recording to 30 seconds, and releases every audio track on stop,
cancel, error, and unmount.

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev -- --host=127.0.0.1
```

`VITE_API_BASE_URL` defaults to `http://127.0.0.1:8000`. This is the only
frontend API setting; the ElevenLabs key remains exclusively in `backend/.env`.
Microphone access requires a supported browser and a secure context (localhost
is accepted during development). The backend permits only the comma-separated
origins in `CORS_ORIGINS`, which defaults to localhost and 127.0.0.1 on port
5173.

Frontend validation:

```powershell
npm run build
npm run lint
npm test
npm run test:browser
```

The browser smoke test uses Chrome with real WAV speech through MediaRecorder.
It verifies a grounded corporation query twice, a mobile refusal query, CORS,
source and latency visibility, microphone track cleanup, and horizontal
overflow. Actual request latency is shown in the UI; no end-to-end 200 ms claim
or fake streaming is used.

## Stage 5 — Scaling and production readiness

Stage 5 does **not** ingest the full 55–56 GB dataset. It measures capacity first.

```powershell
cd backend
# Stops are recommended before large runs so the embedding model has free RAM.
python scripts/run_scale_ingest.py --records 1000 10000
```

Measured on this host (11.34 GiB RAM):

| Scale | Result |
| --- | --- |
| Production index | 500 query records / 11,478 chunks |
| Isolated 1K records | 22,573 chunks in 701 s · 32.2 chunks/s · peak RSS 3.33 GiB |
| 10K / 50K / 100K | Skipped — insufficient free RAM after the 1K peak |

Decision: **Option D** — keep the documented Hindi validation development subset.
Full-corpus ESTIMATEs and production-hardening notes are in
`benchmarks/STAGE5_SCALING_PRODUCTION_REPORT.md` and
`benchmarks/scaling/scaling_report.md`.

## Evaluation

```powershell
python scripts/evaluate.py
```

Where MSMARCO-XI `is_selected` labels exist for the indexed subset, this compares:

1. Dense retrieval  
2. BM25  
3. Hybrid  
4. Hybrid + reranker  

Metrics: Recall@5, Recall@10, Precision@5, MRR, nDCG@10, and measured retrieval latency. `evaluate_chunking.py` and `evaluate_embeddings.py` compare candidates on fixed corpora and qrels.

## Latency benchmarking

```powershell
python scripts/benchmark.py
```

Uses `time.perf_counter()` over 100 valid unique indexed queries and writes min/mean/median/P50/P70/P90/P95/P100 for every stage. Reports are written to `benchmarks/latency` as JSON, CSV, and Markdown. No response cache is used.

The HH Goa end-to-end target is under 200 ms. The project **does not claim full
text-to-answer <200 ms**: ElevenLabs generation alone measured 4,299.34 ms P50.

## Stage 2 measured results (2026-08-15)

Development index: 500 query records, 11,478 chunks, Hindi validation with
English and translated passages. Full benchmark: 100 valid unique queries,
3 warmups, no response cache.

| Component | P50 | P70 | P100 |
| --- | ---: | ---: | ---: |
| Embedding | 77.48 ms | 87.55 ms | 159.61 ms |
| Qdrant | 19.05 ms | 22.10 ms | 52.53 ms |
| BM25 | 37.64 ms | 47.35 ms | 129.86 ms |
| Retrieval wall | 97.66 ms | 114.73 ms | 188.32 ms |
| Reranker | 917.44 ms | 992.32 ms | 1,881.39 ms |
| ElevenLabs generation | 4,299.34 ms | 5,410.99 ms | 14,858.73 ms |
| Grounding | 0.30 ms | 0.46 ms | 1.34 ms |
| Full text-to-answer | 5,368.20 ms | 6,382.56 ms | 15,997.77 ms |

Measured bottlenecks: ElevenLabs generation first, local reranking second.

The optimization cycle disabled reranking in the latency-first runtime profile
and retained parallel dense/BM25 retrieval:

- RAG core P50: **1,020.79 ms → 52.87 ms**
- Sequential retrieval P50: **66.04 ms**
- Parallel retrieval P50: **52.67 ms**
- Retrieval quality trade-off: MRR **0.5369 → 0.3996**, nDCG@10
  **0.5969 → 0.4994**

The optimized RAG core is below 200 ms P50 on this 11,478-chunk local index.
The complete text-to-answer path is not.

Retrieval ablation on 50 labeled indexed queries:

| Pipeline | Recall@5 | Recall@10 | MRR | nDCG@10 | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.4600 | 0.6000 | 0.3100 | 0.3764 | 83.09 ms |
| BM25 | 0.6800 | 0.8200 | 0.3161 | 0.4344 | 39.45 ms |
| Hybrid RRF | 0.6200 | 0.8200 | 0.3996 | 0.4994 | 83.35 ms |
| Hybrid weighted | 0.6200 | 0.8000 | 0.3778 | 0.4761 | 83.36 ms |
| Hybrid RRF + reranker | 0.7400 | 0.7800 | 0.5369 | 0.5969 | 888.60 ms |

Embedding comparison on 46 labeled queries / 1,058 fixed chunks selected
`intfloat/multilingual-e5-small` as the next-index recommendation: Recall@10
0.9130 versus 0.7500 and mean query embedding 18.55 ms versus 21.38 ms.
The active 11,478-vector collection remains MiniLM because switching embedding
spaces requires a clean re-index.

Chunking comparison found fixed, sentence, and metadata-aware effectively tied
on this short-passage subset (Recall@10 0.8152). Semantic chunking produced
3,106 chunks versus ~1,058 and reduced Recall@10 to 0.7174.

Detailed evidence is under `benchmarks/`; generated values are actual execution
results except capacity figures, which are explicitly labeled extrapolations.

## Local setup

Prerequisites: Python 3.11+, Docker Desktop.

```powershell
cd "hh-goa-voice-rag"
docker compose up -d
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# set ELEVENLABS_API_KEY in .env
python scripts/inspect_dataset.py
python scripts/ingest.py
uvicorn app.main:app --reload
```

First ingest downloads the embedding model and streams a Hindi validation parquet slice (hundreds of MB, not 55 GB).

## Environment variables

See `backend/.env.example`. Important knobs:

- Dataset: `DATASET_CONFIG`, `DATASET_SPLIT`, `INGEST_MODE`, `MAX_DOCUMENTS`, `BATCH_SIZE`
- Index languages: `INDEX_ENGLISH`, `INDEX_TRANSLATED`
- Models: `EMBEDDING_MODEL`, `RERANKER_MODEL`, `ENABLE_RERANKER`, `ELEVENLABS_MODEL`
- Retrieval: `DENSE_TOP_K`, `BM25_TOP_K`, `HYBRID_TOP_K`, `RERANK_TOP_K`, `PARALLEL_RETRIEVAL`
- Chunking: `CHUNK_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- Guardrails: `RELEVANCE_MIN_SCORE`, `GROUNDING_MIN_OVERLAP`, `MAX_QUERY_CHARS`
- ElevenLabs: `ELEVENLABS_API_KEY`, `ELEVENLABS_AGENT_ID` (optional; created on first use)

Never commit `.env`.

## Testing

```powershell
cd backend
pytest -q
```

External APIs, Qdrant, and GPU models are mocked in unit tests. The core pipeline is tested as text-in / structured-response-out.

Frontend unit tests mock MediaRecorder and network access. The repeatable browser
smoke test uses installed Chrome and real ElevenLabs/backend requests. The core
pipeline contract remains `run(query: str) -> RAGResponse`.

Known limitations: responses are final rather than streamed, Chrome's fake
microphone may repeat short audio as it loops the fixture, and observed latency
varies with external ElevenLabs services. TTS and deployment are intentionally
outside Stage 4.
