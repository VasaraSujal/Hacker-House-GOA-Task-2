# Stage 6C — Production Tail-Latency Optimization

Status: **IMPLEMENTED AND VERIFIED LOCALLY; MATCHED PRODUCTION AFTER-BENCHMARK
PENDING DEPLOYMENT**

No architecture, embedding model, dataset, Qdrant collection, retrieval depth,
RRF behavior, reranker, grounding, guardrail, or frontend behavior was changed.

## Production baseline

Target: `https://hacker-house-goa-task-2.onrender.com/api/rag/query`

- Warmup: 3 requests (excluded)
- Measured: 66 requests
- Successful / failed: 66 / 0
- Query mix: 45 knowledge-base-tagged / 21 unsupported-tagged
- Refused: 24
- Every response returned five sources

| Component | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---:|---:|---:|
| embedding | 162.58 | 172.15 | 246.20 |
| dense retrieval | 0.00 | 0.00 | 0.00 |
| BM25 | 113.73 | 158.97 | 286.49 |
| retrieval wall | 163.03 | 172.79 | 287.61 |
| RRF / fusion | 0.10 | 0.11 | 0.17 |
| reranking | 0.39 | 0.51 | 86.08 |
| generation | 0.25 | 0.32 | 3.43 |
| grounding | 0.13 | 0.16 | 85.80 |
| RAG core | 163.81 | 173.38 | 295.33 |
| total | 165.24 | 175.88 | 296.83 |

Cloud inference and Qdrant search occur in one remote operation, so the existing
instrumentation attributes that wall time to `embedding_ms`; the residual
`dense_retrieval_ms` is zero.

Raw results:
`benchmarks/stage6c_baseline_production/production_latency_report.json`.

## Profiling and bottleneck analysis

The high BM25 tail was recurrent, not one removable sample: 8 of 66 measured
requests had BM25 latency at or above 200 ms. It was influenced by both query
work and host scheduling:

- BM25 latency correlation with query words: **0.707**
- BM25 latency correlation with query characters: **0.600**
- BM25 latency correlation with embedding latency: **0.613**
- Off-topic BM25 P50/P100: **163.27 / 280.78 ms**
- Knowledge-base BM25 P50/P100: **91.36 / 286.49 ms**
- Identical queries varied substantially across their three repetitions; for
  example, the tallest-mountain query measured 159.5, 269.0, and 286.5 ms.
- Two reranking pauses near 86 ms and one grounding pause near 86 ms show that
  scheduling pauses can occur outside BM25 too. This is consistent with shared
  Render Free CPU contention/throttling, although application timings alone
  cannot prove the host scheduler's exact cause.

The implementation profile found the principal avoidable work:

1. `rank_bm25.get_scores()` scanned all 11,478 per-document frequency
   dictionaries once for every query token.
2. A long-query profile spent 4.65 of 5.17 seconds in scoring across 100 calls,
   including 9.18 million dictionary lookups.
3. Sorting every one of the 11,478 scores consumed about 10% of local search
   time even though only the top 20 were needed.
4. Tokenization, normalization, payload construction, RRF, reranking,
   generation, and grounding were not material steady-state bottlenecks.
5. BM25 was loaded once and reused; it was not rebuilt per request.

Changing `top_k` alone did not solve the scan. Local P50 values for candidate
sizes 5, 20, 100, and 1,000 were approximately 21.66, 21.52, 22.47, and
23.43 ms. The production value remains 20 to preserve RRF inputs and quality.

No early refusal shortcut was introduced because it would alter retrieval and
guardrail behavior.

## Optimization

`BM25Index` now lazily builds a sparse term-postings representation from the
already loaded BM25 document frequencies:

- Query scoring visits only documents containing each query token.
- The BM25Okapi IDF, `k1`, `b`, document lengths, and formula are unchanged.
- Duplicate query terms retain their original repeated contribution.
- A thresholded partial selection replaces the full result sort.
- Cutoff ties retain the old score-descending, document-index-ascending order.
- The postings build is lock-protected and occurs once on first search.
- Ingestion behavior is preserved: an index rebuild invalidates the postings,
  and the next search rebuilds them lazily.

The first local search spends about 262 ms building postings. It is a one-time
cold cost covered by the documented three-request warmup.

## Retrieval and behavior quality

All **496** labeled evaluation queries produced the same top-20 rankings before
and after optimization. The canonical rankings have the same SHA-256:

`e17ec903a103637d30ac4fff2fc43976d6ad87e13a29e7134067bc0a5dd1a037`

| Metric | Before | After |
|---|---:|---:|
| Recall@5 | 0.639449 | 0.639449 |
| Recall@10 | 0.773522 | 0.773522 |
| Precision@5 | 0.137500 | 0.137500 |
| MRR | 0.374151 | 0.374151 |
| nDCG@10 | 0.464059 | 0.464059 |

The required behavior checks also produced identical answers and source
document IDs:

| Query | Grounded | Refused | Sources |
|---|---:|---:|---:|
| What is a corporation? | true | false | 5 |
| What is a shareholder? | true | false | 5 |
| How do shareholders vote? | true | false | 5 |
| Who won yesterday's cricket match? | true | true | 5 |

## Local post-optimization measurements

Direct warm BM25 search:

- P50: **0.327 ms**
- P100: **1.005 ms**
- 496-query quality run: **15,305 ms before → 430 ms after**

Production-like Free Docker, 3 warmups plus 66 measured HTTP requests:

| Component | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---:|---:|---:|
| embedding | 200.27 | 212.50 | 319.48 |
| BM25 | 1.00 | 1.12 | 2.89 |
| retrieval wall | 200.43 | 212.75 | 319.67 |
| RAG core | 201.06 | 213.30 | 320.38 |
| total | 202.11 | 214.35 | 321.46 |

This local HTTP run confirms that BM25 no longer controls the tail. Its total
latency is not directly comparable with Render because dense inference crosses
a different network path from the development workstation.

## Memory safety

| Measurement | Before | After |
|---|---:|---:|
| Free Docker startup RSS | ~162.8 MiB | 160.2 MiB |
| Free Docker post-query peak | ~166.7 MiB | 201.6 MiB |
| Direct postings RSS increment | — | 23.7 MiB |
| Image size | 297 MB | 297 MB |

The measured peak remains about 310 MiB below the 512 MiB Render Free limit.
No model, duplicate BM25 index, Torch, SentenceTransformers, or ONNX dependency
was added.

## Regression tests

- Backend: **70 passed**
- Frontend: **6 passed**
- Frontend lint: **PASS**
- Frontend production build: **PASS**
- Targeted BM25 / Free mode: **15 passed**

## Production after-benchmark and decision

The optimized code is intentionally uncommitted and unpushed. Therefore the
live Render service still runs the baseline implementation, and rerunning it
would not measure this optimization. A matched 3-warmup/66-request production
after-benchmark must be run after the user reviews and deploys the worktree.

Current decision: **KEEP for deployment validation**. The optimization is
meaningful, exact-ranking-preserving, and memory-safe. Do not pursue candidate
reduction or architectural changes.

The under-200 ms requirement is **not claimed as fully met**. The measured
production baseline has RAG P50/P70 below 200 ms but P100 at 295.33 ms. Final
target status must use the pending matched production after-benchmark.
