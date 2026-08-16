# Stage 6B Free Deployment Report

Status: **LOCALLY COMPLETE — Render Free redeploy pending push**

The Free profile runs without Torch/SentenceTransformers, uses Qdrant Cloud
hosted inference on a dedicated production collection, and stays under 512 MB
in local Docker simulation.

## Checklist

| Item | Result |
|---|---|
| Local full RAG preserved | PASS (`RETRIEVAL_MODE=local` unchanged) |
| Lightweight Free mode | PASS (`cloud_dense_sparse` + `extractive`) |
| No local embedding model in Free image | PASS (297 MB image vs 1.98 GB full) |
| Qdrant Cloud vector retrieval | PASS (`hh_goa_voice_rag_prod`) |
| Hosted model | `intfloat/multilingual-e5-small` (384-d, free) |
| Production vectors | **11,478** verified |
| BM25 + RRF + light reranker | PASS |
| Extractive answer | PASS |
| Guardrails / cricket refusal | PASS |
| ElevenLabs STT | PASS |
| Backend tests | **65 passed** |
| Frontend tests | **6 passed** |
| Frontend lint / build | PASS / PASS |
| Startup RSS (Free container) | **162.8 MB** MEASURED |
| Peak RSS after 20 queries | **~166.7 MB** MEASURED |
| Post-voice RSS | **~161–167 MB** MEASURED |
| Public Render URL | pending redeploy of Free image |

## Architecture

```text
Browser → ElevenLabs STT → FastAPI Free
  ├─ Qdrant Cloud Document inference (dense)
  └─ BM25 pickle
       → RRF → LexicalLightReranker → extractive answer → grounding
```

## Latency (MEASURED, Free Docker simulation)

| Metric | Value |
|---|---:|
| RAG P50 | 195.28 ms |
| RAG P70 | 207.34 ms |
| RAG P100 | 638.08 ms |
| Voice STT | 1068.234 ms |
| Voice RAG core | 200.878 ms |
| Voice generation (extractive) | 0.451 ms |
| Full voice total | 1270.723 ms |

Do **not** report full voice as sub-200 ms. RAG P50 is near the 200 ms target;
P70/P100 can exceed it due to Qdrant Cloud inference variance.

## Local vs Free (summary)

See `benchmarks/cloud_mode_comparison.md`.

| Metric | Local Full | Cloud Free |
|---|---:|---:|
| Startup RSS | higher (Torch) | ~163 MB |
| Peak RSS | — | ~167 MB |
| RAG P50 | ~66–98 ms warm local historically; 1350 ms cold sample | 195 ms |
| Generation P50 | ~3–4 s ElevenLabs | ~0.4 ms extractive |
| Full voice P50 | ~4.5–5.5 s | ~1.27 s (STT dominated) |

## Dataset disclosure

Validated development subset only (`ai4bharat/MSMARCO-XI`, `hi`, validation,
500 queries → 11,478 chunks). Full ~55–56 GB snapshot is not indexed.

## Deploy next step

1. Commit/push Stage 6B files (including `Dockerfile.free`, `render.yaml`).
2. On Render: use `backend/Dockerfile.free`, plan Free, env from `render.yaml`.
3. Set `CORS_ORIGINS` to the Vercel HTTPS origin.
4. Confirm public `/health`, RAG, voice, refusal.

## Remaining limitations

- Free answers are extractive, not generative ElevenLabs prose.
- Dense model differs from local MiniLM (compatible new collection required).
- RAG P100 may exceed 200 ms on cold/cloud variance.
- Public Render redeploy not executed from this agent session.
