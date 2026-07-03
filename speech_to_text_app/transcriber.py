from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Protocol

from speech_to_text_app.profiles import (
    TranscriptionProfile,
    profile_from_id,
)
from speech_to_text_app.runtime import (
    configure_cuda_dll_search_paths,
    find_bundled_model,
)


DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptionError(RuntimeError):
    """Raised when speech recognition fails."""


class WhisperModelProtocol(Protocol):
    def transcribe(self, audio_path: str, **options): ...


ModelFactory = Callable[..., WhisperModelProtocol]
ModelResolver = Callable[[str], Path | None]
BatchedPipelineFactory = Callable[[WhisperModelProtocol], WhisperModelProtocol]


def _default_cpu_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def _load_default_model_factory():
    configure_cuda_dll_search_paths()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Install application dependencies first."
        ) from exc
    return WhisperModel


def _load_default_batched_pipeline_factory():
    try:
        from faster_whisper import BatchedInferencePipeline
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper batched inference is not available."
        ) from exc
    return BatchedInferencePipeline


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        cpu_threads: int | None = None,
        profile: TranscriptionProfile | None = None,
        model_factory: ModelFactory | None = None,
        batched_pipeline_factory: BatchedPipelineFactory | None = None,
        model_resolver: ModelResolver = find_bundled_model,
    ):
        self._profile = profile or profile_from_id("exam_accuracy")
        if profile is None:
            self._profile = TranscriptionProfile(
                id="custom",
                label="Custom",
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                language="en",
                beam_size=5,
                condition_on_previous_text=True,
                vad_filter=True,
                batched=False,
                batch_size=1,
            )
        self._model_size = self._profile.model_size
        self._device = self._profile.device
        self._compute_type = self._profile.compute_type
        self._cpu_threads = cpu_threads or _default_cpu_threads()
        self._model_factory = model_factory
        self._batched_pipeline_factory = batched_pipeline_factory
        self._model_resolver = model_resolver
        self._model: WhisperModelProtocol | None = None
        self._batched_model = None

    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptSegment]:
        try:
            model = (
                self._get_batched_model()
                if self._profile.batched
                else self._get_model()
            )
            options = {
                "beam_size": self._profile.beam_size,
                "vad_filter": self._profile.vad_filter,
                "condition_on_previous_text": self._profile.condition_on_previous_text,
            }
            selected_language = language or self._profile.language
            if selected_language:
                options["language"] = selected_language
            if self._profile.batched:
                segments, _info = model.transcribe(
                    str(audio_path),
                    batch_size=self._profile.batch_size,
                    **options,
                )
            else:
                segments, _info = model.transcribe(str(audio_path), **options)
            return [
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                )
                for segment in segments
            ]
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc

    def _get_model(self) -> WhisperModelProtocol:
        if self._model is None:
            factory = self._model_factory or _load_default_model_factory()
            model_ref = self._model_size
            bundled_model = self._model_resolver(self._model_size)
            if bundled_model is not None:
                model_ref = str(bundled_model)
            self._model = factory(
                model_ref,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
            )
        return self._model

    def _get_batched_model(self):
        if self._batched_model is None:
            factory = (
                self._batched_pipeline_factory
                or _load_default_batched_pipeline_factory()
            )
            self._batched_model = factory(self._get_model())
        return self._batched_model


def format_timestamp(seconds: float) -> str:
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def format_transcript(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(
            f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {text}"
        )
    return "\n".join(lines)
