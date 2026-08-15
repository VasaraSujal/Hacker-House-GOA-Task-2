# Stage 4 Frontend Engineering Report

## Architecture

React 19, TypeScript, Vite, Tailwind CSS, and React Icons provide the browser
client. `useVoiceRecorder` owns MediaRecorder and microphone cleanup;
`ragApi.ts` is the sole HTTP boundary. The backend API key is never exposed.

```text
Browser microphone → MediaRecorder Blob → multipart /api/voice/query
→ ElevenLabs STT → RAGPipeline → grounding → typed React result
```

The UI distinguishes idle, recording, processing, grounded success,
knowledge-base refusal, and application error states. It displays the backend
transcript, unchanged answer, source chunks, grounding status, and actual
request timing.

## Automated verification

- Frontend production build: passed
- Frontend lint: passed with no warnings
- Frontend tests: 6 passed
- Backend regression tests: 53 passed
- CORS preflight: passed for `http://localhost:5173` and
  `http://127.0.0.1:5173`

Unit tests cover multipart API submission, safe rate-limit errors, recording
state transitions, grounded success, refusal, backend failure, and denied
microphone permission. External APIs and microphone hardware are mocked only in
unit tests.

## Real Chrome smoke tests

Chrome was launched with real WAV fixtures supplied as its microphone input.
The frontend still used getUserMedia, MediaRecorder, Blob generation, multipart
upload, and the live backend/ElevenLabs path.

### Relevant query

- Input audio: “What is a corporation?”
- Transcript: “What is a corporation? What is a corporation?” (Chrome's fake
  microphone loops the short fixture)
- Result: grounded answer
- Sources: 5
- Second recording in the same page: passed
- Microphone tracks released after both recordings: passed
- Latest measured request: STT 1,213.99 ms; RAG core 65.83 ms; generation
  4,171.32 ms; full pipeline 5,451.52 ms

### Unsupported query

- Input audio: “Who won yesterday's cricket match?”
- Result: successful knowledge-base refusal, not an application error
- Refusal text: “I couldn't find enough relevant information in the provided
  knowledge base to answer this question.”
- Microphone tracks released: passed
- 390 × 844 mobile viewport with no horizontal overflow: passed
- Latest measured request: STT 1,083.73 ms; RAG core 67.88 ms; generation
  3,387.50 ms; full pipeline 4,539.50 ms

Screenshots:

- `benchmarks/stage4-browser-relevant.png`
- `benchmarks/stage4-browser-refusal-mobile.png`

## Fixes discovered through browser testing

The real browser emits `audio/webm;codecs=opus`. Backend validation originally
compared this exact value with `audio/webm`, causing HTTP 400. Validation now
compares the normalized base media type while preserving the original MIME
value for ElevenLabs. A regression test covers this browser contract.

The source/result grid also exposed a mobile min-content overflow. Adding
`min-width: 0` to the two grid children resolved it without altering desktop
layout.

## Limitations

- The backend returns a complete response; the UI does not claim streaming.
- External STT and generation dominate latency. No 200 ms end-to-end claim is
  made.
- Browser MIME support varies; the recorder chooses the first compatible
  backend-supported type.
- Automated microphone verification used Chrome's fake microphone device with
  real speech audio, not physical microphone hardware.

## Recommended Stage 5

Plan deployment and environment-specific origins, then run a short manual
physical-microphone acceptance test on the deployed HTTPS origin. Do not ingest
the full dataset until deployment capacity and retrieval-quality goals are
defined.
