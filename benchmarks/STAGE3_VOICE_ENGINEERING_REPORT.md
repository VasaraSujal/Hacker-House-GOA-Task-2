# Stage 3 Voice Engineering Report

## Architecture

```text
Audio file upload
  → filename / MIME / size validation
  → ElevenLabs Speech-to-Text (`scribe_v2`)
  → existing InputGuard + RAGPipeline.run(transcript)
  → ElevenLabs generation (`gemini-2.0-flash`)
  → grounding guard
  → structured voice response
```

`/api/voice/query` is a thin adapter: it does not duplicate retrieval,
reranking, context construction, generation, or grounding logic.

## Implementation

- `rag/stt/base.py`: typed provider interface and transcript result.
- `rag/stt/elevenlabs.py`: real multipart `POST /v1/speech-to-text` provider.
- `app/api/routes/voice.py`: multipart upload endpoint, validation, controlled
  errors, structured logs, and latency composition.
- `app/models/schemas.py`: voice response and latency schema.
- `scripts/voice_benchmark.py`: configurable real STT benchmark.

The provider uses `file` and required `model_id=scribe_v2`, the documented
ElevenLabs batch STT contract. Retries are bounded to transient network,
timeout, 429, and selected 5xx responses. Authentication, invalid audio, and
other 4xx errors are not retried.

## Tests and live verification

- Unit/integration test suite: **49 passed**.
- Mocked voice-path tests cover valid WAV, unsupported extension, empty
  transcript, multipart contract, and controlled STT rate-limit failure.
- Live `/health`: `status=ok`, STT configured.
- Real WAV smoke test: the file saying “What is a corporation?” transcribed
  exactly to that text, produced a grounded answer, and returned 5 sources.
- Real unsupported WAV test: “Who won yesterday's cricket match?” transcribed
  exactly and returned the existing no-context refusal (`refused=true`).

## Measured 20-request voice benchmark

One short WAV was submitted repeatedly with warm local models and no response
cache. This measures the real ElevenLabs STT and generation APIs.

| Component | P50 | P70 | P100 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Audio validation | 0.00 ms | 0.00 ms | 0.00 ms | 0.00 ms |
| ElevenLabs STT | 1,022.52 ms | 1,053.97 ms | 1,480.71 ms | 1,040.32 ms |
| RAG including generation | 3,177.44 ms | 3,313.75 ms | 4,763.56 ms | 3,337.86 ms |
| ElevenLabs generation | 3,037.46 ms | 3,180.69 ms | 4,490.44 ms | 3,196.76 ms |
| Grounding | 0.43 ms | 0.46 ms | 0.61 ms | 0.44 ms |
| Full voice pipeline | 4,201.59 ms | 4,419.22 ms | 6,244.36 ms | 4,378.24 ms |

## Comparison and bottleneck

Stage 2's optimized local RAG core measured 52.87 ms P50 with reranking
disabled. The Stage 3 voice endpoint's full pipeline measured 4,201.59 ms P50.

The largest contributor is ElevenLabs generation (3,037.46 ms P50), followed
by ElevenLabs STT (1,022.52 ms P50). The project must not claim full
voice-to-answer under 200 ms. The 200 ms result remains only for the local
RAG-core profile, not for external STT or generation.

## Known limitations and Stage 4 recommendation

- The benchmark repeats one short WAV; varied speakers, accents, background
  noise, and multilingual audio need a separate quality dataset.
- Upload handling caps in-memory bytes at the configured limit. For very large
  production uploads, add streamed temporary-file handling.
- The current generation design uses an external API and dominates latency.

Stage 4 should add only the browser audio capture/frontend and must re-measure
browser upload plus STT/RAG/generation latency. It should not claim a full
sub-200 ms experience without changing the generation architecture and
repeating this benchmark.
