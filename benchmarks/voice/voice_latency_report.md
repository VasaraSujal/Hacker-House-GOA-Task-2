# Voice Pipeline Latency Report

- Requests: 20
- Audio: `what-is-a-corporation.wav`
- STT: `scribe_v2`

| Component | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
|---|---:|---:|---:|---:|
| audio_validation_ms | 0.00 | 0.00 | 0.00 | 0.00 |
| stt_ms | 1022.52 | 1053.97 | 1480.71 | 1040.32 |
| rag_ms | 3177.44 | 3313.75 | 4763.56 | 3337.86 |
| generation_ms | 3037.46 | 3180.69 | 4490.44 | 3196.76 |
| grounding_ms | 0.43 | 0.46 | 0.61 | 0.44 |
| total_ms | 4201.59 | 4419.22 | 6244.36 | 4378.24 |
