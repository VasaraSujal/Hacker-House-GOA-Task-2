# RAG Latency Report

- Timestamp (UTC): 2026-08-15T09:48:33.296743+00:00
- Requests: 100 (100 unique)
- Invalid indexed queries excluded: 1
- Index chunks: 11478
- Retrieval mode: parallel
- Response cache: disabled
- Request parsing: not applicable (direct in-process pipeline harness)
- Model initialization: 49180.62 ms (excluded from warm percentiles)
- Refused responses: 22

| Component | P50 (ms) | P70 (ms) | P90 (ms) | P95 (ms) | P100 (ms) | Mean (ms) |
|---|---:|---:|---:|---:|---:|---:|
| request_parsing_ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| query_processing_ms | 0.02 | 0.02 | 0.03 | 0.04 | 0.10 | 0.02 |
| embedding_ms | 77.48 | 87.55 | 104.93 | 111.37 | 159.61 | 77.98 |
| dense_retrieval_ms | 19.05 | 22.10 | 32.34 | 35.14 | 52.53 | 20.67 |
| bm25_ms | 37.64 | 47.35 | 63.11 | 76.24 | 129.86 | 41.16 |
| retrieval_wall_ms | 97.66 | 114.73 | 130.49 | 141.98 | 188.32 | 98.91 |
| fusion_ms | 0.14 | 0.17 | 0.20 | 0.23 | 0.39 | 0.15 |
| relevance_guard_ms | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.01 |
| reranking_ms | 917.44 | 992.32 | 1270.95 | 1313.51 | 1881.39 | 936.51 |
| context_building_ms | 0.10 | 0.12 | 0.15 | 0.16 | 1.00 | 0.11 |
| generation_ms | 4299.34 | 5410.99 | 9468.28 | 11564.60 | 14858.73 | 5219.43 |
| grounding_ms | 0.30 | 0.46 | 0.63 | 0.86 | 1.34 | 0.35 |
| rag_core_ms | 1020.79 | 1109.54 | 1387.28 | 1437.16 | 2023.94 | 1036.08 |
| component_sum_ms | 5367.97 | 6382.37 | 10374.34 | 12391.25 | 15997.58 | 6255.51 |
| unaccounted_ms | 0.20 | 0.22 | 0.26 | 0.29 | 0.32 | 0.20 |
| total_ms | 5368.20 | 6382.56 | 10374.47 | 12391.42 | 15997.77 | 6255.71 |

Primary measured bottleneck: **generation_ms** (4299.34 ms P50).
Secondary measured bottleneck: **reranking_ms** (917.44 ms P50).

RAG core P50: **1020.79 ms**.
Full text-to-answer P50: **5368.20 ms**.

The full total includes ElevenLabs generation. No values are estimated or fabricated.
