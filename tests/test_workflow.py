from pathlib import Path
import tempfile
import unittest

from speech_to_text_app.transcriber import TranscriptSegment
from speech_to_text_app.workflow import TranscriptionResult, transcribe_media_file


class TranscriptionWorkflowTests(unittest.TestCase):
    def test_transcribes_supported_media_through_audio_pipeline(self):
        commands = []

        def runner(command):
            commands.append(command)
            if command[0] == "ffprobe":
                return _ProcessResult(returncode=0, stdout="12.5\n", stderr="")
            return _ProcessResult(returncode=0, stdout="", stderr="")

        transcriber = _FakeTranscriber([TranscriptSegment(0.0, 1.0, "hello")])

        with tempfile.TemporaryDirectory() as temp_dir:
            result = transcribe_media_file(
                source=Path("meeting.mp4"),
                work_dir=Path(temp_dir),
                transcriber=transcriber,
                runner=runner,
            )

        self.assertEqual(
            TranscriptionResult(
                source=Path("meeting.mp4"),
                duration_seconds=12.5,
                transcript="[00:00:00.000 - 00:00:01.000] hello",
            ),
            result,
        )
        self.assertEqual(Path("meeting.mp4"), transcriber.audio_path)
        self.assertEqual("ffprobe", commands[0][0])
        self.assertEqual(1, len(commands))

    def test_passes_language_to_transcriber(self):
        def runner(command):
            return _ProcessResult(returncode=0, stdout="1\n", stderr="")

        transcriber = _FakeTranscriber([TranscriptSegment(0.0, 1.0, "privet")])

        with tempfile.TemporaryDirectory() as temp_dir:
            transcribe_media_file(
                source=Path("voice.wav"),
                work_dir=Path(temp_dir),
                transcriber=transcriber,
                language="ru",
                runner=runner,
            )

        self.assertEqual("ru", transcriber.language)


class _FakeTranscriber:
    def __init__(self, segments):
        self._segments = segments
        self.audio_path = None
        self.language = None

    def transcribe(self, audio_path, language=None):
        self.audio_path = audio_path
        self.language = language
        return self._segments


class _ProcessResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


if __name__ == "__main__":
    unittest.main()
