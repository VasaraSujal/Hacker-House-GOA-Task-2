# Production Deployment

Target architecture: Vercel static frontend, Render Free Docker backend,
Qdrant Cloud Inference, and ElevenLabs STT. Kubernetes and full-corpus
ingestion are intentionally out of scope.

## Prerequisites

- Qdrant Cloud cluster with Inference enabled
- Free hosted dense model (recommended: `intfloat/multilingual-e5-small`)
- Render Free web service (512 MB) using `backend/Dockerfile.free`
- Vercel project
- ElevenLabs API key with STT permissions
- Docker for local Free-image verification

The deployed knowledge base is the validated Hindi validation subset:
500 query records, 9,989 passages, and 11,478 chunks. It is not the complete
55–56 GB MSMARCO-XI snapshot.

## Profiles

| Profile | Image | Retrieval | Answer |
|---|---|---|---|
| Full local | `backend/Dockerfile` | local MiniLM + Qdrant | generative ElevenLabs |
| Render Free | `backend/Dockerfile.free` | Qdrant Cloud inference + BM25 | extractive |

## Qdrant Cloud production collection

Do **not** query the existing `hh_goa_rag` vectors with a different embedding
model. Rebuild a dedicated collection:

```powershell
cd backend
$env:SOURCE_QDRANT_URL="http://127.0.0.1:6333"
$env:SOURCE_QDRANT_COLLECTION="hh_goa_rag"
$env:TARGET_QDRANT_URL="https://<cluster>.<region>.cloud.qdrant.io:6333"
$env:TARGET_QDRANT_API_KEY="<secret>"
$env:TARGET_QDRANT_COLLECTION="hh_goa_voice_rag_prod"
$env:QDRANT_INFERENCE_MODEL="intfloat/multilingual-e5-small"
python scripts/rebuild_cloud_collection.py
```

Expected result: exactly 11,478 points, Cosine 384-d, payload fields
`text`, `document_id`, `chunk_id`. Local source collection is never deleted.

Optional payload/vector copy of the old model (reference only):

```powershell
python scripts/migrate_qdrant.py
```

## Backend on Render Free

Use root `render.yaml` and `backend/Dockerfile.free`.

```text
docker build -f backend/Dockerfile.free -t hh-goa-voice-rag-api:free .
```

The Free image does **not** install Torch or SentenceTransformers. It embeds
the BM25 pickle and FastAPI runtime only.

Required Render environment variables:

```text
APP_ENV=production
RETRIEVAL_MODE=cloud_dense_sparse
ANSWER_MODE=extractive
WEB_CONCURRENCY=1
ELEVENLABS_API_KEY=<secret>
QDRANT_URL=https://<qdrant-cloud-host>:6333
QDRANT_API_KEY=<secret>
QDRANT_COLLECTION=hh_goa_voice_rag_prod
QDRANT_INFERENCE_MODEL=intfloat/multilingual-e5-small
QDRANT_INFERENCE_DIMENSION=384
CORS_ORIGINS=https://<vercel-production-domain>
BM25_INDEX_PATH=/app/data/indexes/bm25.pkl
VOICE_RATE_LIMIT_PER_MINUTE=20
QDRANT_TIMEOUT_S=30
```

Render supplies `PORT`; the container binds `0.0.0.0:${PORT}` with one worker.
`GET /health` is the platform health check. Production startup fails if Qdrant
is unavailable/empty, BM25 is missing, or ElevenLabs STT is not configured.

## Frontend on Vercel

Deploy the `frontend/` directory. Set:

```text
VITE_API_BASE_URL=https://<render-service>.onrender.com
```

Rebuild after changing this variable because Vite injects it at build time.
Never put ElevenLabs or Qdrant credentials in any `VITE_` variable.

After Vercel assigns the final domain, set that exact HTTPS origin in Render's
`CORS_ORIGINS` and redeploy the backend.

## Smoke checks

```powershell
curl.exe https://<backend>/health
curl.exe -X POST https://<backend>/api/rag/query `
  -H "Content-Type: application/json" `
  -d '{"query":"What is a corporation?"}'
curl.exe -X POST https://<backend>/api/voice/query `
  -F "audio=@data/smoke/what-is-a-corporation.wav;type=audio/wav"
```

Then use the public Vercel page in Chrome:

1. Ask “What is a corporation?” and verify transcript, grounded answer, sources,
   and request latency.
2. Ask “Who won yesterday's cricket match?” and verify knowledge-base refusal.
3. Confirm HTTPS and no CORS errors.

## Rollback

- Keep the local Qdrant collection and snapshot unchanged.
- Render: redeploy the previous immutable image/revision.
- Vercel: promote the previous successful deployment.
- Qdrant: do not delete `hh_goa_rag` during app rollback.
- If cloud rebuild validation fails, stop and continue using the local system.

## Known limitations

- The deployment uses the validated subset, not the full MSMARCO-XI snapshot.
- Render Free mode uses extractive answers, not ElevenLabs generation.
- Dense embeddings in Free mode use multilingual-e5-small, not the local
  paraphrase-multilingual MiniLM collection.
- BM25 is an in-memory pickle suitable for ~11K chunks, not millions.
- Rate limiting is per backend process and is not a distributed limiter.
- External ElevenLabs STT dominates end-to-end voice latency.
- End-to-end voice latency is multi-second; it is not under 200 ms.
