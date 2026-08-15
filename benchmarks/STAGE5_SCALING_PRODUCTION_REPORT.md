# Stage 5 — Scaling & Production Readiness Report

## Dataset

| Field | Value |
|---|---|
| Dataset | `ai4bharat/MSMARCO-XI` |
| Configuration | `hi` (Hindi) |
| Split | `validation` |
| Current development size | **MEASURED** 500 query records → 9,989 passages → 11,478 chunks |
| Provided total size | ~55–56 GB Hub snapshot (ESTIMATE ~11M query records across languages/splits) |

Schema verification (code + prior inspect script):

- Records are **queries** with nested `passages.English_passages` / `Translated_passages` / `is_selected`.
- Ingestion indexes **passages**, never queries-as-documents.
- Deterministic passage IDs: `query_id-lang-index-hash`.
- Deterministic chunk IDs: `sha1(document_id|strategy|position|text[:64])[:16]`.
- Qdrant point IDs: `uuid5(NAMESPACE_URL, chunk_id)`.

## Current index (MEASURED)

| Metric | Value |
|---:|
| Query records | 500 |
| Passages | 9,989 |
| Chunks / Qdrant vectors / BM25 entries | 11,478 / 11,478 / 11,478 |
| Unique chunk IDs | 11,478 (0 duplicates) |
| Unique document IDs | 9,989 |
| Duplicate text hashes | 177 (expected cross-language/near-duplicate text) |
| Embedding dimension | 384 (Cosine) |
| BM25 file | 15.57 MB |
| Qdrant Docker volume | ~594 MB (includes WAL/segments; not linear at this scale) |
| Languages | en 5,731 · hin 5,747 |

Production index remained unchanged after isolated scale tests.

## Ingestion architecture (audited)

```text
HF parquet stream (validation/hinval.parquet)
  → parse QueryRecord / Passage
  → sentence chunker
  → batch embed (LocalEmbeddingProvider, model loaded once)
  → Qdrant batch upsert (wait=True)
  → BM25 add + rebuild + pickle save
  → checkpoint (bounded processed query-id window)
```

Hardening added in Stage 5:

- Bounded batch retries with structured failure logs (`batch_id`, query range, retry count).
- Progress lines with chunks/s, elapsed time, and RSS.
- Isolated scale harness that never writes production collection/index/checkpoint.
- Configurable `INGEST_MAX_BATCH_RETRIES`, `QDRANT_TIMEOUT_S`, `VOICE_RATE_LIMIT_PER_MINUTE`.

## Scale testing

| Scale | Status | Chunks | Wall | Peak RSS | Throughput |
|---|---|---:|---:|---:|---:|
| 500 (prod baseline) | MEASURED | 11,478 | n/a | n/a | n/a |
| 1,000 records | MEASURED | 22,573 | 701.0 s | 3,328.5 MB | 32.2 chunks/s |
| 10,000 | SKIPPED | — | — | — | free RAM 2.17 GiB after 1K |
| 50,000 / 100,000 | SKIPPED | — | — | — | safety policy |

1K breakdown (MEASURED): embedding 538.1 s · Qdrant upsert 15.4 s · BM25 31.7 s.

Qdrant search latency from prior copy benchmark (MEASURED): P50 ≈ 30.7–30.8 ms at 1K / 10K / 11.5K vectors.

## Full dataset estimate (ESTIMATE)

Basis: 22.573 chunks/record from measured 1K; 384-d float32 vectors; BM25 ≈ 1.35 KB/chunk from measured 1K file.

| Corpus | Records | Chunks | Raw vectors | BM25 | Ingest @ 32 c/s |
|---|---:|---:|---:|---:|---:|
| Hindi validation | 97,941 | ~2.21M | ~3.2 GiB | ~2.8 GiB | ~19 h |
| All 14 validation | ~1.37M | ~31.0M | ~44 GiB | ~39 GiB | ~11 days |
| Full ~55 GB snapshot | ~11M | ~248M | ~355 GiB | ~311 GiB | ~89 days |

Additional ESTIMATE: Qdrant on-disk with payload/HNSW overhead ≈ 1.3×–2.0× (raw vectors + text). Full-corpus BM25 cannot stay in-process RAM on this host.

### Decision: **Option D**

Keep the documented Hindi validation development subset (500 records / 11,478 chunks).

Reasons:

1. MEASURED peak RSS for only 1K records was **3.33 GiB** on an **11.34 GiB** machine.
2. In-memory `rank_bm25` stores tokenized corpus + payloads; full corpus is not viable locally.
3. Streaming/batching architecture is already correct for larger environments.
4. Submitting a silent tiny index without documentation would be dishonest; this subset is explicit, measurable, and reproducible.

## Production readiness

| Area | Finding | Action |
|---|---|---|
| API secrets | ElevenLabs key server-side only | Unchanged |
| CORS | Explicit localhost/127.0.0.1:5173 | Unchanged; production URL via `CORS_ORIGINS` |
| Audio upload | Size cap, extension allow-list, MIME base-type normalization | Unchanged |
| Rate limiting | Voice endpoint can burn STT+LLM quota | **Added** in-process sliding window (`VOICE_RATE_LIMIT_PER_MINUTE`, default 20) |
| Timeouts | STT/LLM already configurable | **Added** `QDRANT_TIMEOUT_S` |
| Logging | request_id + latency fields; no audio body / API keys | Audited OK |
| Health | Lightweight dependency status | Audited OK (`configured` vs live ping for Qdrant) |
| Batch failures | Silent loss risk | **Added** bounded retries + error logs |
| BM25 scaling | In-memory Okapi; full corpus limitation | Documented; no premature rewrite |
| Reranker | Disabled in latency profile; loads once when enabled | Audited OK |
| Deployment | Out of scope | Not performed |

## Physical microphone

**NOT AVAILABLE** in this agent environment.

Automated Chrome MediaRecorder smoke (Stage 4) with real WAV fixtures remains the verified browser path. A human physical-microphone acceptance test is still recommended before demo day.

## Regressions

| Check | Result |
|---|---|
| Backend pytest | **57 passed** |
| Frontend build | **PASS** |
| Frontend lint | **PASS** |
| Frontend tests | **6 passed** |
| `GET /health` | ok · qdrant ok · bm25 11478 docs · STT configured |
| `POST /api/rag/query` | grounded answer (corporation) |
| `POST /api/voice/query` | transcript + grounded answer (~5.9 s total) |
| Physical microphone | **NOT AVAILABLE** |
| Git | No `.git` directory in the project tree; no commit/push performed |

## Remaining risks

- Full MSMARCO-XI cannot be indexed on this laptop.
- BM25 rebuild-on-every-batch will dominate at larger subset sizes (already 31.7 s of 1K wall).
- Qdrant volume size at small N is overhead-dominated; do not linearly extrapolate the 594 MB / 11k figure.
- Physical microphone UX not re-verified in Stage 5.

## Recommended Stage 6

1. Deployment packaging (API + Qdrant + static frontend) with explicit production CORS origin.
2. Optional cloud/larger-RAM ingestion of Hindi validation (or a labeled larger subset).
3. Replace in-memory BM25 with a disk-backed sparse index before multi-million chunk corpora.
4. Human physical-microphone acceptance on the deployed HTTPS origin.
