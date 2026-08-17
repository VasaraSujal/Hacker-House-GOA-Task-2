# Production RAG Latency Report (Render)

- Timestamp (UTC): 2026-08-16T17:48:18.985108+00:00
- Target: https://hacker-house-goa-task-2.onrender.com
- Endpoint: /api/rag/query
- Warmup requests (excluded): 3
- Measured requests: 66
- Successful: 66
- Failures: 0
- Refused (among success): 24
- KB-tagged / off-topic-tagged: 45 / 21

| Component | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
|---|---:|---:|---:|---:|
| embedding_ms | 162.58 | 172.15 | 246.20 | 170.82 |
| dense_retrieval_ms | 0.00 | 0.00 | 0.00 | 0.00 |
| bm25_ms | 113.73 | 158.97 | 286.49 | 122.31 |
| retrieval_wall_ms | 163.03 | 172.79 | 287.61 | 177.15 |
| fusion_ms | 0.10 | 0.11 | 0.17 | 0.10 |
| reranking_ms | 0.39 | 0.51 | 86.08 | 2.90 |
| generation_ms | 0.25 | 0.32 | 3.43 | 0.25 |
| grounding_ms | 0.13 | 0.16 | 85.80 | 1.41 |
| rag_core_ms | 163.81 | 173.38 | 295.33 | 181.62 |
| total_ms | 165.24 | 175.88 | 296.83 | 183.14 |

## Embedding ~475 ms diagnosis

- embedding P50/P70/P100: 162.58 / 172.15 / 246.20 ms
- first measured embedding_ms: 157.90
- remaining measured embedding P50: 162.68
- Verdict: steady-state embedding is generally under 200 ms

## <200 ms requirement

- rag_core P50/P70/P100 all < 200 ms: **NOT MET** (P50=163.81, P70=173.38, P100=295.33)

Values are server-reported stage timings from live HTTP responses. No estimates.
