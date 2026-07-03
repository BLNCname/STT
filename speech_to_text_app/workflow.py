from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from speech_to_text_app.media import (
    CommandRunner,
    ensure_supported_media,
    probe_duration_seconds,
)
from speech_to_text_app.transcriber import FasterWhisperTranscriber, format_transcript


@dataclass(frozen=True)
class TranscriptionResult:
    source: Path
    duration_seconds: float
    transcript: str


def transcribe_media_file(
    source: Path,
    work_dir: Path,
    transcriber: FasterWhisperTranscriber,
    language: str | None = None,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    runner: CommandRunner | None = None,
) -> TranscriptionResult:
    ensure_supported_media(source)
    if runner is None:
        duration = probe_duration_seconds(source, ffprobe_path=ffprobe_path)
    else:
        duration = probe_duration_seconds(
            source, ffprobe_path=ffprobe_path, runner=runner
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    segments = transcriber.transcribe(source, language=language)
    return TranscriptionResult(
        source=source,
        duration_seconds=duration,
        transcript=format_transcript(segments),
    )
