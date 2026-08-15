# Stage 2 Engineering Report

## Current system

- Indexed query records: 500
- Indexed chunks: 11,478
- Active embedding: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Retrieval: parallel dense + BM25, RRF
- Runtime reranker: disabled in latency-first profile
- Evaluated reranker: `mmarco-mMiniLMv2-L12-H384-v1`
- Generation: ElevenLabs Agents API, `gemini-2.0-flash`
- Response cache: disabled

## Retrieval quality

Evaluation used 50 labeled English/Hindi queries already represented in the
index.

| Pipeline | Recall@5 | Recall@10 | MRR | nDCG@10 | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.4600 | 0.6000 | 0.3100 | 0.3764 | 83.09 ms |
| BM25 | 0.6800 | 0.8200 | 0.3161 | 0.4344 | 39.45 ms |
| Hybrid RRF | 0.6200 | 0.8200 | 0.3996 | 0.4994 | 83.35 ms |
| Hybrid weighted | 0.6200 | 0.8000 | 0.3778 | 0.4761 | 83.36 ms |
| Hybrid RRF + reranker | 0.7400 | 0.7800 | 0.5369 | 0.5969 | 888.60 ms |

RRF was retained. Reranking gives the best ranking quality, but its latency is
incompatible with the stated 200 ms target on this CPU.

## Full latency

The full benchmark used 100 valid unique queries, three warmups, and no response
cache.

| Component | P50 | P70 | P100 |
| --- | ---: | ---: | ---: |
| Embedding | 77.48 ms | 87.55 ms | 159.61 ms |
| Qdrant | 19.05 ms | 22.10 ms | 52.53 ms |
| BM25 | 37.64 ms | 47.35 ms | 129.86 ms |
| Retrieval wall | 97.66 ms | 114.73 ms | 188.32 ms |
| Reranking | 917.44 ms | 992.32 ms | 1,881.39 ms |
| Context building | 0.10 ms | 0.12 ms | 1.00 ms |
| ElevenLabs generation | 4,299.34 ms | 5,410.99 ms | 14,858.73 ms |
| Grounding | 0.30 ms | 0.46 ms | 1.34 ms |
| Total | 5,368.20 ms | 6,382.56 ms | 15,997.77 ms |

Primary bottleneck: ElevenLabs generation. Secondary bottleneck: local
cross-encoder reranking.

## Optimization result

Two 100-query no-generation runs isolated the core:

- Before, reranker enabled: RAG core P50 1,020.79 ms.
- After, reranker disabled + parallel retrieval: RAG core P50 52.87 ms.
- Sequential retrieval P50: 66.04 ms.
- Parallel retrieval P50: 52.67 ms.

This is a measured 94.8% reduction in core P50. It trades away ranking quality:
MRR falls from 0.5369 to 0.3996 and nDCG@10 from 0.5969 to 0.4994.

## Chunking and embeddings

Fixed, sentence, and metadata-aware chunking tied at Recall@10 0.8152 on the
46-query/50-record short-passage experiment. Semantic chunking created about
three times as many chunks and fell to Recall@10 0.7174. Sentence chunking stays
as the interpretable default.

`multilingual-e5-small` materially beat the active MiniLM in the fixed-corpus
embedding experiment: Recall@10 0.9130 versus 0.7500, MRR 0.6113 versus 0.3938,
and mean query embedding 18.55 ms versus 21.38 ms. It is recommended for the
next clean re-index; the active Qdrant collection cannot mix embedding spaces.

## Scaling and capacity

Measured Qdrant P50 was 30.70 ms at 1,000 vectors, 30.74 ms at 10,000, and
30.84 ms at 11,478. This does not validate million-vector performance.

Extrapolating the measured 22.96 chunks/query-record to the approximate 55 GB
snapshot gives roughly 252.5 million chunks, 361 GiB raw float32 vectors,
636–978 GiB Qdrant storage, and 319 GiB for the current Python BM25
serialization. These are planning estimates, not measured full-dataset usage.
The in-memory BM25 design is not suitable at that scale.

## Guardrails

Five live pipeline cases passed: relevant, irrelevant, weak retrieval, unsafe,
and empty input. Four deterministic grounding cases passed after increasing
the minimum lexical-overlap threshold from 0.12 to 0.20; the old threshold had
accepted an unsupported answer at 0.167 overlap.

## Recommendation

Carry forward parallel RRF retrieval with reranking disabled by default, while
keeping reranking configurable for quality-first/offline use. Re-index with
`multilingual-e5-small` before the final quality benchmark. Replace in-memory
BM25 before large-corpus ingestion.

The optimized local RAG core is below 200 ms P50 on the 11,478-chunk
development index. The complete text-to-answer path is not: ElevenLabs alone
is over four seconds P50. Voice integration would add STT latency, so the
submission must not claim end-to-end <200 ms without changing the generation
architecture and measuring again.
