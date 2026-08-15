# Retrieval Ablation Report

- Queries: 50
- Index chunks: 11478
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

| Pipeline | Recall@5 | Recall@10 | Precision@5 | MRR | nDCG@10 | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.4600 | 0.6000 | 0.0920 | 0.3100 | 0.3764 | 63.63 |
| bm25 | 0.6800 | 0.8200 | 0.1360 | 0.3161 | 0.4344 | 33.22 |
| hybrid_rrf | 0.6200 | 0.8200 | 0.1240 | 0.3996 | 0.4994 | 63.89 |
| hybrid_weighted | 0.6200 | 0.8000 | 0.1240 | 0.3778 | 0.4761 | 63.91 |
| hybrid_rrf+reranker | 0.7400 | 0.7800 | 0.1480 | 0.5369 | 0.5969 | 800.69 |
