# Cloud Free vs Local Full Comparison

MEASURED on 2026-08-16 against the validated 11,478-chunk subset.

## Setup

| Mode | Config |
|---|---|
| Local full | `RETRIEVAL_MODE=local`, `ANSWER_MODE=generative`, collection `hh_goa_rag`, paraphrase-multilingual-MiniLM-L12-v2 |
| Cloud Free | `RETRIEVAL_MODE=cloud_dense_sparse`, `ANSWER_MODE=extractive`, collection `hh_goa_voice_rag_prod`, `intfloat/multilingual-e5-small` via Qdrant Cloud Inference |

Local source collection was **not** deleted. Production collection was rebuilt with hosted inference (11,478 points verified).

## Query: What is a corporation?

| Metric | Local Full | Cloud Free |
|---|---|---|
| Grounded | true | true |
| Refused | false | false |
| Sources | 5 | 5 |
| Shared top chunk IDs | `1b6bcdc540c0ed94`, `41df27a7cf521f58`, `2ff6bd7ef85ffd17` | same overlap present |
| RAG core | 1350.484 ms (cold-ish local) | ~195–208 ms |
| Generation | 4552.148 ms (ElevenLabs) | ~0.4 ms (extractive) |

Cloud Free answer (extractive, source sentences only) correctly defines a corporation from retrieved passages.

## Query: Who won yesterday's cricket match?

| Metric | Local Full | Cloud Free |
|---|---|---|
| Expected | refusal | refusal |
| Cloud Free | — | `refused=true`, grounding-safe refusal message |

Cloud Free lexical coverage guard blocks weak dense hits (e.g. unrelated “yesterday” passages).

## Notes

- Scores are not bit-identical across embedding models; that is expected.
- Functional equivalence for the corporation query is acceptable: overlapping corporation/shareholder sources and a grounded extractive answer.
- Full voice Free mode remains STT-dominated (~1.0–1.1 s STT + ~200 ms RAG).
