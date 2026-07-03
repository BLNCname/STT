from pathlib import Path
import unittest

from speech_to_text_app.media import (
    MAX_MEDIA_SECONDS,
    SUPPORTED_EXTENSIONS,
    MediaValidationError,
    build_audio_extract_command,
    ensure_supported_media,
    ensure_within_duration_limit,
    extract_audio_to_wav,
    seconds_from_av_duration,
    probe_duration_seconds,
)


class MediaValidationTests(unittest.TestCase):
    def test_accepts_common_audio_and_video_extensions_case_insensitively(self):
        for name in [
            "meeting.MP4",
            "voice.mp3",
            "clip.wav",
            "camera.AVI",
            "interview.mkv",
            "recording.m4a",
            "archive.webm",
        ]:
            with self.subTest(name=name):
                self.assertEqual(Path(name), ensure_supported_media(Path(name)))

        self.assertIn(".mp4", SUPPORTED_EXTENSIONS)
        self.assertIn(".avi", SUPPORTED_EXTENSIONS)
        self.assertIn(".wav", SUPPORTED_EXTENSIONS)

    def test_rejects_unsupported_media_extension(self):
        with self.assertRaisesRegex(MediaValidationError, "Unsupported media format"):
            ensure_supported_media(Path("notes.txt"))

    def test_rejects_media_longer_than_one_hour(self):
        with self.assertRaisesRegex(MediaValidationError, "longer than 1 hour"):
            ensure_within_duration_limit(MAX_MEDIA_SECONDS + 0.1)

    def test_accepts_media_at_one_hour_limit(self):
        self.assertEqual(
            MAX_MEDIA_SECONDS, ensure_within_duration_limit(MAX_MEDIA_SECONDS)
        )

    def test_converts_pyav_microsecond_duration_to_seconds(self):
        self.assertEqual(1.694286, seconds_from_av_duration(1694286, 1000000))

    def test_builds_ffmpeg_command_for_fast_cpu_transcription_audio(self):
        command = build_audio_extract_command(
            ffmpeg_path="ffmpeg",
            source=Path("input.mp4"),
            target=Path("work/audio.wav"),
        )

        self.assertEqual(
            [
                "ffmpeg",
                "-y",
                "-i",
                "input.mp4",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "work\\audio.wav",
            ],
            command,
        )

    def test_probes_duration_with_ffprobe(self):
        calls = []

        def runner(command):
            calls.append(command)
            return _ProcessResult(returncode=0, stdout="3599.42\n", stderr="")

        duration = probe_duration_seconds(
            Path("input.mp4"), ffprobe_path="ffprobe", runner=runner
        )

        self.assertEqual(3599.42, duration)
        self.assertEqual(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                "input.mp4",
            ],
            calls[0],
        )

    def test_rejects_unreadable_duration(self):
        def runner(command):
            return _ProcessResult(returncode=0, stdout="N/A", stderr="")

        with self.assertRaisesRegex(
            MediaValidationError, "Could not read media duration"
        ):
            probe_duration_seconds(Path("input.mp4"), runner=runner)

    def test_extracts_audio_with_ffmpeg(self):
        calls = []

        def runner(command):
            calls.append(command)
            return _ProcessResult(returncode=0, stdout="", stderr="")

        target = extract_audio_to_wav(
            source=Path("input.mp4"),
            target=Path("work/audio.wav"),
            ffmpeg_path="ffmpeg",
            runner=runner,
        )

        self.assertEqual(Path("work/audio.wav"), target)
        self.assertEqual(
            build_audio_extract_command(
                "ffmpeg", Path("input.mp4"), Path("work/audio.wav")
            ),
            calls[0],
        )

    def test_reports_ffmpeg_conversion_errors(self):
        def runner(command):
            return _ProcessResult(returncode=1, stdout="", stderr="Invalid data found")

        with self.assertRaisesRegex(MediaValidationError, "Invalid data found"):
            extract_audio_to_wav(
                source=Path("input.mp4"),
                target=Path("work/audio.wav"),
                runner=runner,
            )


class _ProcessResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


if __name__ == "__main__":
    unittest.main()
