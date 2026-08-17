# Production RAG Latency Report (Render)

- Timestamp (UTC): 2026-08-16T17:55:48.300414+00:00
- Target: http://127.0.0.1:8011
- Endpoint: /api/rag/query
- Warmup requests (excluded): 3
- Measured requests: 66
- Successful: 66
- Failures: 0
- Refused (among success): 24
- KB-tagged / off-topic-tagged: 45 / 21

| Component | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
|---|---:|---:|---:|---:|
| embedding_ms | 200.27 | 212.50 | 319.48 | 211.02 |
| dense_retrieval_ms | 0.00 | 0.00 | 0.00 | 0.00 |
| bm25_ms | 1.00 | 1.12 | 2.89 | 1.03 |
| retrieval_wall_ms | 200.43 | 212.75 | 319.67 | 211.22 |
| fusion_ms | 0.08 | 0.09 | 0.20 | 0.09 |
| reranking_ms | 0.36 | 0.41 | 0.54 | 0.26 |
| generation_ms | 0.22 | 0.25 | 0.37 | 0.16 |
| grounding_ms | 0.13 | 0.14 | 0.23 | 0.10 |
| rag_core_ms | 201.06 | 213.30 | 320.38 | 211.71 |
| total_ms | 202.11 | 214.35 | 321.46 | 212.73 |

## Embedding ~475 ms diagnosis

- embedding P50/P70/P100: 200.27 / 212.50 / 319.48 ms
- first measured embedding_ms: 195.52
- remaining measured embedding P50: 200.69
- Verdict: mixed / variable embedding latency; inspect sample series

## <200 ms requirement

- rag_core P50/P70/P100 all < 200 ms: **NOT MET** (P50=201.06, P70=213.30, P100=320.38)

Values are server-reported stage timings from live HTTP responses. No estimates.
