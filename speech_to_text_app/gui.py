from __future__ import annotations

from pathlib import Path

from speech_to_text_app.profiles import (
    DEFAULT_PROFILE_ID,
    TRANSCRIPTION_PROFILES,
    profile_from_id,
)

MODEL_CHOICES = {
    profile.label: profile.id for profile in TRANSCRIPTION_PROFILES.values()
}
DEFAULT_MODEL_CHOICE = profile_from_id(DEFAULT_PROFILE_ID).label

LANGUAGE_CHOICES = {
    "Auto": None,
    "Russian": "ru",
    "English": "en",
}


def model_size_from_choice(choice: str) -> str:
    return profile_from_id(profile_id_from_choice(choice)).model_size


def profile_id_from_choice(choice: str) -> str:
    return MODEL_CHOICES.get(choice, DEFAULT_PROFILE_ID)


def language_code_from_choice(choice: str) -> str | None:
    return LANGUAGE_CHOICES.get(choice)


def default_transcript_path(source: Path) -> Path:
    return source.with_suffix(".txt")
