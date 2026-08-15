# Stage 6 Deployment Report

Status: **BLOCKED — public provider authentication required**

Deployment artifacts are prepared. The production Docker image was built and
smoke-tested locally against the verified development Qdrant index. No public
URL or cloud production result is claimed until Render, Vercel, and Qdrant Cloud
are authenticated and a real public-browser test succeeds.

## Deployment checklist

| Item | Result |
|---|---|
| Frontend URL | NOT AVAILABLE |
| Backend URL | NOT AVAILABLE |
| Qdrant | Local source verified (11,478 points, green); cloud target pending |
| Dataset/index | Validated development subset: 11,478 chunks |
| Frontend build | PASS |
| Frontend lint | PASS |
| Frontend tests | 6 passed |
| Backend tests | 57 passed |
| Health | Local container PASS; production pending |
| Voice | Local container PASS; production pending |
| Refusal | Local PASS historically; production pending |
| Sources | Local container PASS (5 production-index sources) |
| Grounding | Local container PASS (`grounded: true`) |
| CORS | Local PASS historically; production pending |
| HTTPS | Pending provider deployment |

## Prepared artifacts

- `backend/Dockerfile` (CPU-only torch; no CUDA)
- `backend/requirements.prod.txt`
- `.dockerignore`
- `render.yaml`
- `frontend/vercel.json`
- `frontend/.env.production.example`
- `backend/scripts/backup_qdrant.py`
- `backend/scripts/migrate_qdrant.py`
- `docs/DEPLOYMENT.md`
- `docs/ARCHITECTURE.md`

## Local container verification (MEASURED)

Image: `hh-goa-voice-rag-api:stage6` (~1.98 GB)

Health:

```json
{"status":"ok","qdrant":"ok","embeddings":"ok (dim=384)","bm25":"ok (11478 docs)","elevenlabs_configured":true,"stt_configured":true}
```

Voice fixture `data/smoke/what-is-a-corporation.wav` against the container
(local Qdrant via `host.docker.internal`):

| Metric | Value |
|---|---:|
| HTTP | 200 |
| Transcript | What is a corporation? |
| Sources | 5 |
| Grounded | true |
| Refused | false |
| STT | 1140.721 ms |
| RAG core | 98.339 ms |
| Generation | 3330.578 ms |
| Full voice total | 4571.459 ms |

Local Qdrant point count remained **11,478** before and after image build and
container smoke tests. The local index was not deleted.

## Local vs production benchmark

Production measurements remain blank until real public HTTPS requests run.

| Metric | Local measured | Production |
|---|---:|---:|
| Health | PASS | PENDING |
| RAG core P50 | ~66–98 ms (prior + container) | PENDING |
| STT P50 | ~1,141–1,214 ms | PENDING |
| Generation P50 | ~3,331–4,171 ms | PENDING |
| Full voice P50 | ~4,202–5,452 ms across prior runs | PENDING |
| Full voice P100 | ~6,244 ms | PENDING |

Do not claim end-to-end voice latency below 200 ms.

## Dataset disclosure

The deployment target uses the validated development subset/index
(`ai4bharat/MSMARCO-XI`, `hi`, `validation`, 500 query records → 11,478 chunks).
The full approximately 55 GB MSMARCO-XI snapshot was not locally indexed because
Stage 5 capacity analysis projected hundreds of GiB and impractical local
ingestion time. Ingestion remains streaming, batched, resumable, and suitable
for a larger environment.

Current BM25 is an in-memory pickle suitable for this subset scale. A
multi-million-chunk corpus would require a disk-backed sparse retrieval redesign.

## Secrets / safety

- Real provider keys exist only in local `backend/.env` (not embedded in the
  Docker image; `.env` is dockerignored).
- Frontend production bundle scan: no `localhost`, no `127.0.0.1`, no API key
  material; `VITE_API_BASE_URL` injected as HTTPS placeholder during audit build.
- Git is **not** initialized under the project tree. No automatic push was
  performed.

## Blocking inputs for COMPLETE

1. Create a Qdrant Cloud cluster and provide `QDRANT_URL` + `QDRANT_API_KEY`
   (shell/provider secrets only).
2. Authenticate a Render deployment path (Git remote or container registry).
   This tree currently has no `.git` directory.
3. Authenticate Vercel and deploy `frontend/` with
   `VITE_API_BASE_URL=https://<render-service>`.
4. Set Render `CORS_ORIGINS` to the exact Vercel HTTPS origin (no `*`).
5. Run public HTTPS health/RAG/voice/refusal/CORS browser tests and fill
   production benchmark cells above.

## Host notes from this session

- C: was critically low (~0.45 GiB) after an earlier aborted CUDA image pull.
- Safe cleanup moved Hugging Face cache to `E:\ml-cache\huggingface` and cleared
  npm cache; C: recovered to ~8.9 GiB.
- CPU-only Dockerfile rebuild succeeded; local Qdrant remained intact.
