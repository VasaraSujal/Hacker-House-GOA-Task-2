# Production RAG Latency Report (Render)

- Timestamp (UTC): 2026-08-18T13:33:09.440407+00:00
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
| embedding_ms | 161.92 | 162.80 | 171.46 | 162.29 |
| dense_retrieval_ms | 0.00 | 0.00 | 0.00 | 0.00 |
| bm25_ms | 0.73 | 0.79 | 1.87 | 0.72 |
| retrieval_wall_ms | 162.15 | 163.03 | 171.62 | 162.56 |
| fusion_ms | 0.10 | 0.10 | 0.15 | 0.10 |
| reranking_ms | 0.44 | 0.55 | 0.77 | 0.35 |
| generation_ms | 0.28 | 0.38 | 0.53 | 0.24 |
| grounding_ms | 0.16 | 0.21 | 1.06 | 0.14 |
| rag_core_ms | 163.03 | 163.94 | 171.77 | 163.22 |
| total_ms | 164.33 | 165.68 | 174.67 | 164.88 |

## Embedding ~475 ms diagnosis

- embedding P50/P70/P100: 161.92 / 162.80 / 171.46 ms
- first measured embedding_ms: 164.36
- remaining measured embedding P50: 161.91
- Verdict: steady-state embedding is generally under 200 ms

## <200 ms requirement

- rag_core P50/P70/P100 all < 200 ms: **MET**

Values are server-reported stage timings from live HTTP responses. No estimates.
