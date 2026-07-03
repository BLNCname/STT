from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable, Protocol


MAX_MEDIA_SECONDS = 60 * 60

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


class MediaValidationError(ValueError):
    """Raised when a selected media file cannot be transcribed."""


class ProcessResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str]], ProcessResult]


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def ensure_supported_media(path: Path) -> Path:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise MediaValidationError(
            f"Unsupported media format '{path.suffix}'. Supported formats: {supported}"
        )
    return path


def ensure_within_duration_limit(duration_seconds: float) -> float:
    if duration_seconds > MAX_MEDIA_SECONDS:
        raise MediaValidationError("Selected media is longer than 1 hour.")
    return duration_seconds


def build_audio_extract_command(
    ffmpeg_path: str, source: Path, target: Path
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]


def probe_duration_seconds(
    path: Path,
    ffprobe_path: str = "ffprobe",
    runner: CommandRunner | None = None,
) -> float:
    if runner is None:
        return _probe_duration_seconds_with_av(path)

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = runner(command)
    if result.returncode != 0:
        message = result.stderr.strip() or "ffprobe failed."
        raise MediaValidationError(message)
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise MediaValidationError("Could not read media duration.") from exc
    return ensure_within_duration_limit(duration)


def _probe_duration_seconds_with_av(path: Path) -> float:
    try:
        import av
    except ImportError as exc:
        raise MediaValidationError(
            "PyAV is not installed; cannot read media duration."
        ) from exc

    try:
        with av.open(str(path)) as container:
            if container.duration is None:
                raise MediaValidationError("Could not read media duration.")
            return ensure_within_duration_limit(
                seconds_from_av_duration(container.duration, av.time_base)
            )
    except MediaValidationError:
        raise
    except Exception as exc:
        raise MediaValidationError(str(exc)) from exc


def seconds_from_av_duration(duration: int, time_base: int | float) -> float:
    return float(duration / time_base)


def extract_audio_to_wav(
    source: Path,
    target: Path,
    ffmpeg_path: str = "ffmpeg",
    runner: CommandRunner = _run_command,
) -> Path:
    command = build_audio_extract_command(ffmpeg_path, source, target)
    result = runner(command)
    if result.returncode != 0:
        message = result.stderr.strip() or "ffmpeg conversion failed."
        raise MediaValidationError(message)
    return target
