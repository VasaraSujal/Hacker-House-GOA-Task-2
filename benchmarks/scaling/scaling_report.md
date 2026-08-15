# Scaling Report

MEASURED isolated ingestions. Production collection `hh_goa_rag` was not modified.

Host at test time: **11.34 GiB RAM**, **~2.95 GiB free** before 1K, **71.36 GiB free** on `E:`.

## Measured scales

| Records | Status | Chunks | Wall (s) | Peak RSS (MB) | Chunks/s | BM25 (bytes) |
|---:|---|---:|---:|---:|---:|---:|
| 500 | measured_baseline | 11,478 | n/a | n/a | n/a | 15,574,429 |
| 1,000 | measured | 22,573 | 701.01 | 3,328.5 | 32.2 | 30,361,984 |
| 10,000 | skipped |  |  |  |  |  |
| 50,000 | skipped |  |  |  |  |  |
| 100,000 | skipped |  |  |  |  |  |

Skip reasons:

- **10K**: insufficient free RAM after 1K (free 2.17 GiB; 1K already peaked at 3.33 GiB RSS).
- **50K / 100K**: refused by safety policy after measured 1K peak on this host.

## Notes

- Isolated 1K used temporary collection `hh_goa_rag_scale_1000` and was deleted afterward.
- Qdrant reported 22,553 points vs 22,573 emitted chunks (20 deterministic ID collisions overwritten by upsert).
- Embedding dominated wall time (538s of 701s). Qdrant upsert was 15.4s; BM25 rebuild/save 31.7s.
- Existing Qdrant search-latency copy benchmark (Stage 2/3): P50 ≈ 30.7–30.8 ms at 1K / 10K / 11.5K vectors.

## Capacity estimates (ESTIMATE unless labeled MEASURED)

| Corpus | Kind | Est. records | Est. chunks | Raw vectors (GiB) | BM25 (GiB) |
|---|---|---:|---:|---:|---:|
| development_500 | MEASURED | 500 | 11,478 | 0.016 | 0.015 |
| hindi_validation | ESTIMATE | 97,941 | 2,210,830 | 3.16 | 2.77 |
| all_14_validation | ESTIMATE | 1,371,174 | 30,951,620 | 44.3 | 38.7 |
| full_55GB snapshot | ESTIMATE | ~11,000,000 | ~248,000,000 | ~355 | ~311 |

Decision: **Option D** — retain the documented development subset; do not ingest the full 55–56 GB corpus on this machine.
