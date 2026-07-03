from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TranscriptionProfile:
    id: str = "custom"
    label: str = "Custom"
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "en"
    beam_size: int = 5
    condition_on_previous_text: bool = True
    vad_filter: bool = True
    batched: bool = False
    batch_size: int = 1

    def with_language(self, language: str | None) -> "TranscriptionProfile":
        return replace(self, language=language)


TRANSCRIPTION_PROFILES = {
    "exam_accuracy": TranscriptionProfile(
        id="exam_accuracy",
        label="Exam Accuracy - GPU large-v3",
        model_size="large-v3",
        device="cuda",
        compute_type="float16",
        language="en",
        beam_size=5,
        condition_on_previous_text=True,
        vad_filter=True,
        batched=False,
        batch_size=1,
    ),
    "fast_exam": TranscriptionProfile(
        id="fast_exam",
        label="Fast Exam - GPU distil-large-v3",
        model_size="distil-large-v3",
        device="cuda",
        compute_type="float16",
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
        batched=False,
        batch_size=1,
    ),
    "cpu_fallback": TranscriptionProfile(
        id="cpu_fallback",
        label="Emergency CPU fallback - tiny",
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
        batched=False,
        batch_size=1,
    ),
}

DEFAULT_PROFILE_ID = "exam_accuracy"


def profile_from_id(profile_id: str | None) -> TranscriptionProfile:
    return TRANSCRIPTION_PROFILES.get(
        profile_id or DEFAULT_PROFILE_ID,
        TRANSCRIPTION_PROFILES[DEFAULT_PROFILE_ID],
    )
