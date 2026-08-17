# Production RAG Latency Report (Render)

- Timestamp (UTC): 2026-08-16T16:09:07.517468+00:00
- Target: https://hacker-house-goa-task-2.onrender.com
- Endpoint: /api/rag/query
- Warmup requests (excluded): 3
- Measured requests: 22
- Successful: 22
- Failures: 0
- Refused (among success): 8
- KB-tagged / off-topic-tagged: 15 / 7

| Component | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
|---|---:|---:|---:|---:|
| embedding_ms | 167.36 | 180.19 | 211.06 | 173.92 |
| bm25_ms | 127.19 | 174.98 | 293.42 | 135.32 |
| retrieval_wall_ms | 168.45 | 186.50 | 294.95 | 183.88 |
| fusion_ms | 0.10 | 0.10 | 0.18 | 0.10 |
| reranking_ms | 0.41 | 0.52 | 0.71 | 0.32 |
| generation_ms | 0.25 | 0.31 | 2.55 | 0.33 |
| grounding_ms | 0.13 | 0.16 | 0.51 | 0.12 |
| rag_core_ms | 169.39 | 186.62 | 296.12 | 184.48 |
| total_ms | 173.07 | 193.75 | 299.08 | 188.48 |

## Embedding ~475 ms diagnosis

- embedding P50/P70/P100: 167.36 / 180.19 / 211.06 ms
- first measured embedding_ms: 167.27
- remaining measured embedding P50: 167.46
- Verdict: steady-state embedding is generally under 200 ms

## <200 ms requirement

- rag_core P50/P70/P100 all < 200 ms: **NOT MET** (P50=169.39, P70=186.62, P100=296.12)

Values are server-reported stage timings from live HTTP responses. No estimates.
