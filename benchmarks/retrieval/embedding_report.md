# Embedding Model Evaluation

- Queries: 46
- Chunks: 1058

| Model | Dim | Parameter MiB | Query embed (ms) | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 448.8 | 21.38 | 0.5870 | 0.7500 | 0.3938 | 0.4873 |
| intfloat/multilingual-e5-small | 384 | 448.8 | 18.55 | 0.6630 | 0.9130 | 0.6113 | 0.6788 |
