# Stage 2 Benchmark Evidence

All latency and quality values in this directory were produced by actual local
execution on 2026-08-15. Capacity values are extrapolations and are labeled as
such in their reports.

## Reproduce

Run from `backend` with Qdrant available and the development index ingested:

```powershell
python scripts/evaluate.py
python scripts/benchmark.py
python scripts/benchmark_optimization.py
python scripts/evaluate_chunking.py
python scripts/evaluate_embeddings.py
python scripts/evaluate_guardrails.py
python scripts/benchmark_scaling.py
python scripts/estimate_capacity.py
```

`benchmark.py` includes ElevenLabs generation and therefore consumes API calls.
`benchmark_optimization.py` deliberately uses a deterministic local refusal
provider to isolate RAG-core latency; it does not represent full answer latency.

## Reports

- `latency/latency_report.*`: 100-query full text-to-answer percentiles.
- `latency/optimization_report.*`: reranker and concurrency before/after.
- `retrieval/retrieval_report.*`: dense/BM25/fusion/reranker ablation.
- `retrieval/chunking_report.*`: four chunking strategies on one corpus.
- `retrieval/embedding_report.*`: two multilingual embedding models.
- `retrieval/guardrail_grounding_report.*`: live guardrails and grounding cases.
- `scaling/scaling_latency.*`: measured Qdrant latency at 1K, 10K, 11,478.
- `scaling/scaling_report.*`: storage extrapolation from the measured index.

The benchmark excludes one malformed 7,783-character dataset query because it
exceeds the production input limit; this exclusion is recorded in latency
metadata. No response cache was enabled.
