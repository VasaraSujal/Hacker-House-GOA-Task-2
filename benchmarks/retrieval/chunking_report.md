# Chunking Strategy Evaluation

- Source records: 50
- Evaluation queries: 46
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Strategy | Chunks | Avg chars | Est. raw index (MiB) | Recall@5 | Recall@10 | MRR | nDCG@10 | Mean retrieval (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 1057 | 292.5 | 2.08 | 0.6957 | 0.8152 | 0.4072 | 0.5058 | 2.26 |
| sentence | 1058 | 294.4 | 2.08 | 0.6957 | 0.8152 | 0.4034 | 0.5029 | 2.18 |
| semantic | 3106 | 98.0 | 5.07 | 0.5870 | 0.7174 | 0.3712 | 0.4627 | 4.95 |
| metadata | 1058 | 294.4 | 2.08 | 0.6957 | 0.8152 | 0.4034 | 0.5029 | 2.23 |
