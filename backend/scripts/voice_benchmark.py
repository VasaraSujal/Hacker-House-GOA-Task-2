"""Benchmark the real ElevenLabs STT → RAG → generation voice path."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.dependencies import build_pipeline
from app.models.schemas import VoiceLatencyBreakdown
from evaluation.latency_eval import summarize
from rag.stt.elevenlabs import ElevenLabsSTTProvider

STAGES = ["audio_validation_ms", "stt_ms", "rag_ms", "generation_ms", "grounding_ms", "total_ms"]


def main() -> int:
    settings = get_settings()
    audio_path = Path(os.environ.get("VOICE_BENCHMARK_AUDIO", "")).expanduser()
    if not audio_path.is_file():
        print("VOICE_BENCHMARK_AUDIO must point to a readable WAV, MP3, M4A, or WebM file.")
        return 2
    n = max(1, settings.voice_benchmark_queries)
    audio = audio_path.read_bytes()
    content_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".webm": "audio/webm"}
    content_type = content_types.get(audio_path.suffix.lower())
    if not content_type:
        print("Unsupported benchmark file extension.")
        return 2

    pipeline = build_pipeline(settings)
    stt = ElevenLabsSTTProvider(
        settings.elevenlabs_api_key,
        model=settings.elevenlabs_stt_model,
        api_base=settings.elevenlabs_api_base,
        timeout_s=settings.elevenlabs_stt_timeout_s,
        max_retries=settings.elevenlabs_stt_max_retries,
    )
    samples = []
    for i in range(1, n + 1):
        started = time.perf_counter()
        validation_started = time.perf_counter()
        if not audio:
            raise RuntimeError("Benchmark audio is empty")
        validation_ms = (time.perf_counter() - validation_started) * 1000
        transcript = stt.transcribe(audio, filename=audio_path.name, content_type=content_type)
        rag_started = time.perf_counter()
        response = pipeline.run(transcript.text)
        rag_ms = (time.perf_counter() - rag_started) * 1000
        rag_latency = response.latency.model_dump()
        rag_latency.pop("total_ms", None)
        rag_latency.pop("component_sum_ms", None)
        rag_latency.pop("unaccounted_ms", None)
        total_ms = (time.perf_counter() - started) * 1000
        component_sum_ms = validation_ms + transcript.latency_ms + rag_ms
        latency = VoiceLatencyBreakdown(
            **rag_latency,
            audio_validation_ms=validation_ms,
            stt_ms=transcript.latency_ms,
            rag_ms=rag_ms,
            component_sum_ms=component_sum_ms,
            unaccounted_ms=max(0.0, total_ms - component_sum_ms),
            total_ms=total_ms,
        )
        samples.append(
            {
                "sequence": i,
                "transcript": transcript.text,
                "grounded": response.grounded,
                "refused": response.refused,
                **latency.model_dump(),
            }
        )
        print(f"[{i}/{n}] stt={latency.stt_ms:.1f}ms total={latency.total_ms:.1f}ms")

    reports = {stage: asdict(summarize(stage, [row[stage] for row in samples])) for stage in STAGES}
    output_dir = settings.benchmark_output_dir / "voice"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "requests": n,
            "audio_filename": audio_path.name,
            "audio_bytes": len(audio),
            "stt_model": settings.elevenlabs_stt_model,
            "generation_model": settings.elevenlabs_model,
            "note": "Repeated requests reuse the same local audio file; provider and model connections are warm.",
        },
        "stages": reports,
        "samples": samples,
    }
    (output_dir / "voice_latency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Voice Pipeline Latency Report",
        "",
        f"- Requests: {n}",
        f"- Audio: `{audio_path.name}`",
        f"- STT: `{settings.elevenlabs_stt_model}`",
        "",
        "| Component | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, values in reports.items():
        lines.append(f"| {stage} | {values['p50_ms']:.2f} | {values['p70_ms']:.2f} | {values['p100_ms']:.2f} | {values['mean_ms']:.2f} |")
    (output_dir / "voice_latency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote voice report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
