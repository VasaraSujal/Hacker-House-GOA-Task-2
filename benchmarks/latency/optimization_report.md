# Measured Optimization Cycle

- Queries per configuration: 100
- Generation disabled in after-runs to isolate the RAG core.
- Response cache disabled.

Before: reranker enabled, RAG core P50 **1020.79 ms**.
After: reranker disabled, parallel retrieval, RAG core P50 **52.87 ms**.

Sequential retrieval P50: **66.04 ms**.
Parallel retrieval P50: **52.67 ms**.

Quality trade-off (50 labeled queries):
- RRF + reranker: MRR 0.5369, nDCG@10 0.5969.
- RRF without reranker: MRR 0.3996, nDCG@10 0.4994.

ElevenLabs remains outside this optimization: its measured P50 is **4299.34 ms**.
