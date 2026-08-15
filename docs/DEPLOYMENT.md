# Production Deployment

Target architecture: Vercel static frontend, Render Docker backend, Qdrant Cloud,
and ElevenLabs APIs. Kubernetes and full-corpus ingestion are intentionally out
of scope.

## Prerequisites

- Qdrant Cloud cluster with enough capacity for 11,478 384-dimensional vectors
- Render Standard (2 GB RAM minimum; 4 GB preferred for startup margin)
- Vercel project
- ElevenLabs API key with STT and Conversational AI permissions
- Docker for local image verification

The deployed knowledge base is the validated Hindi validation subset:
500 query records, 9,989 passages, and 11,478 chunks. It is not the complete
55–56 GB MSMARCO-XI snapshot.

## Qdrant Cloud

1. Create a Qdrant Cloud cluster; do not expose the local Docker service.
2. Set migration variables in the current shell. Do not store keys in files:

   ```powershell
   $env:TARGET_QDRANT_URL="https://<cluster>.<region>.cloud.qdrant.io:6333"
   $env:TARGET_QDRANT_API_KEY="<secret>"
   cd backend
   python scripts/migrate_qdrant.py
   ```

3. The script copies payloads and vectors in batches without deleting the local
   collection. It must finish with exactly 11,478 target points and verify
   `text`, `document_id`, and `chunk_id` payload fields.

A local snapshot can be recreated with `python scripts/backup_qdrant.py`.
Snapshots under `backups/` are excluded from Git and Docker.

## Backend on Render

The root `render.yaml` and `backend/Dockerfile` are the deployment contract.
The image embeds the persisted BM25 artifact (`data/indexes/bm25.pkl`) and
pre-caches the MiniLM embedding model. It never embeds `.env` or provider keys.

Local image verification (MEASURED on this workstation):

```text
docker build -f backend/Dockerfile -t hh-goa-voice-rag-api:stage6 .
# image size ≈ 1.98 GB (CPU torch; no CUDA)
# container /health → ok; BM25 11478 docs; embeddings dim=384
# voice fixture → HTTP 200, grounded=true, 5 sources
```

Render still needs a Git-connected service or a pushed registry image; this
project tree currently has no `.git` directory.

Required Render environment variables:

```text
APP_ENV=production
ELEVENLABS_API_KEY=<secret>
QDRANT_URL=https://<qdrant-cloud-host>:6333
QDRANT_API_KEY=<secret>
QDRANT_COLLECTION=hh_goa_rag
CORS_ORIGINS=https://<vercel-production-domain>
BM25_INDEX_PATH=/app/data/indexes/bm25.pkl
VOICE_RATE_LIMIT_PER_MINUTE=20
QDRANT_TIMEOUT_S=30
```

Render supplies `PORT`; the container binds `0.0.0.0:${PORT}` with one worker.
One worker avoids duplicating the embedding model and in-memory BM25 index.
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

1. Ask “What is a corporation?” and verify transcript, grounded answer, five
   production sources, and request latency.
2. Ask “Who won yesterday's cricket match?” and verify knowledge-base refusal.
3. Record twice and verify the microphone indicator turns off each time.
4. Confirm the voice request is HTTPS and has no CORS errors.

## Rollback

- Keep the local Qdrant collection and snapshot unchanged.
- Render: redeploy the previous immutable image/revision.
- Vercel: promote the previous successful deployment.
- Qdrant: do not delete or overwrite the cloud collection during app rollback.
- If cloud migration validation fails, stop and continue using the local system;
  never delete the source collection.

## Known limitations

- The deployment uses the validated subset, not the full MSMARCO-XI snapshot.
- BM25 is an in-memory pickle suitable for ~11K chunks, not millions.
- Rate limiting is per backend process and is not a distributed limiter.
- External ElevenLabs STT and generation dominate end-to-end latency.
- End-to-end voice latency is multi-second; it is not under 200 ms.
