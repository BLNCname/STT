from pathlib import Path
import unittest

from speech_to_text_app.transcriber import (
    FasterWhisperTranscriber,
    TranscriptSegment,
    TranscriptionProfile,
    TranscriptionError,
    format_transcript,
)


class TranscriptFormattingTests(unittest.TestCase):
    def test_formats_segments_with_timestamps_and_text(self):
        transcript = format_transcript(
            [
                TranscriptSegment(start=0.0, end=1.25, text=" Hello world "),
                TranscriptSegment(start=62.5, end=65.0, text="second line"),
            ]
        )

        self.assertEqual(
            "[00:00:00.000 - 00:00:01.250] Hello world\n"
            "[00:01:02.500 - 00:01:05.000] second line",
            transcript,
        )

    def test_skips_blank_segments(self):
        transcript = format_transcript(
            [
                TranscriptSegment(start=0.0, end=1.0, text=""),
                TranscriptSegment(start=1.0, end=2.0, text="  kept  "),
            ]
        )

        self.assertEqual("[00:00:01.000 - 00:00:02.000] kept", transcript)


class FasterWhisperTranscriberTests(unittest.TestCase):
    def test_defaults_to_exam_accuracy_gpu_settings(self):
        factory = _RecordingModelFactory([_WhisperSegment(0.0, 1.0, "text")])

        transcriber = FasterWhisperTranscriber(
            model_factory=factory,
            model_resolver=lambda model_size: None,
        )
        transcriber.transcribe(Path("audio.wav"), language="en")

        self.assertEqual("large-v3", factory.model_size)
        self.assertEqual("cuda", factory.options["device"])
        self.assertEqual("float16", factory.options["compute_type"])
        self.assertGreaterEqual(factory.options["cpu_threads"], 1)

    def test_uses_bundled_tiny_model_when_available(self):
        factory = _RecordingModelFactory([_WhisperSegment(0.0, 1.0, "text")])

        transcriber = FasterWhisperTranscriber(
            model_factory=factory,
            model_resolver=lambda model_size: Path("C:/models/faster-whisper-tiny"),
        )
        transcriber.transcribe(Path("audio.wav"))

        self.assertEqual("C:\\models\\faster-whisper-tiny", factory.model_size)

    def test_profile_controls_model_and_transcription_options(self):
        factory = _RecordingModelFactory([_WhisperSegment(0.0, 1.0, "text")])
        profile = TranscriptionProfile(
            model_size="large-v3",
            device="cuda",
            compute_type="float16",
            language="en",
            beam_size=5,
            condition_on_previous_text=True,
            vad_filter=True,
            batched=False,
            batch_size=1,
        )

        transcriber = FasterWhisperTranscriber(
            profile=profile,
            model_factory=factory,
            model_resolver=lambda model_size: None,
        )
        transcriber.transcribe(Path("audio.wav"))

        self.assertEqual("large-v3", factory.model_size)
        self.assertEqual("cuda", factory.options["device"])
        self.assertEqual("float16", factory.options["compute_type"])
        self.assertEqual("en", factory.model.options["language"])
        self.assertEqual(5, factory.model.options["beam_size"])
        self.assertTrue(factory.model.options["condition_on_previous_text"])

    def test_batched_profile_uses_batched_inference_pipeline(self):
        factory = _RecordingModelFactory([_WhisperSegment(0.0, 1.0, "base")])
        batched_factory = _RecordingBatchedPipelineFactory(
            [_WhisperSegment(0.0, 1.0, "batched")]
        )
        profile = TranscriptionProfile(batched=True, batch_size=16)

        transcriber = FasterWhisperTranscriber(
            profile=profile,
            model_factory=factory,
            batched_pipeline_factory=batched_factory,
            model_resolver=lambda model_size: None,
        )
        segments = transcriber.transcribe(Path("audio.wav"))

        self.assertEqual([TranscriptSegment(0.0, 1.0, "batched")], segments)
        self.assertIs(factory.model, batched_factory.model)
        self.assertEqual("audio.wav", batched_factory.pipeline.audio_path)
        self.assertEqual(16, batched_factory.pipeline.options["batch_size"])

    def test_converts_whisper_segments_to_transcript_segments(self):
        factory = _RecordingModelFactory(
            [
                _WhisperSegment(0.0, 1.25, " first "),
                _WhisperSegment(2.0, 3.0, "second"),
            ]
        )

        transcriber = FasterWhisperTranscriber(
            model_factory=factory,
            model_resolver=lambda model_size: None,
        )
        segments = transcriber.transcribe(Path("audio.wav"), language="ru")

        self.assertEqual(
            [
                TranscriptSegment(start=0.0, end=1.25, text=" first "),
                TranscriptSegment(start=2.0, end=3.0, text="second"),
            ],
            segments,
        )
        self.assertEqual("audio.wav", factory.model.audio_path)
        self.assertEqual("ru", factory.model.options["language"])
        self.assertEqual(5, factory.model.options["beam_size"])
        self.assertTrue(factory.model.options["vad_filter"])

    def test_wraps_model_errors(self):
        transcriber = FasterWhisperTranscriber(model_factory=_FailingModelFactory())

        with self.assertRaisesRegex(TranscriptionError, "model failed"):
            transcriber.transcribe(Path("audio.wav"))


class _WhisperSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeWhisperModel:
    def __init__(self, segments):
        self._segments = segments
        self.audio_path = None
        self.options = None

    def transcribe(self, audio_path, **options):
        self.audio_path = audio_path
        self.options = options
        return self._segments, object()


class _RecordingModelFactory:
    def __init__(self, segments):
        self._segments = segments
        self.model_size = None
        self.options = None
        self.model = None

    def __call__(self, model_size, **options):
        self.model_size = model_size
        self.options = options
        self.model = _FakeWhisperModel(self._segments)
        return self.model


class _FakeBatchedPipeline:
    def __init__(self, segments):
        self._segments = segments
        self.audio_path = None
        self.options = None

    def transcribe(self, audio_path, **options):
        self.audio_path = audio_path
        self.options = options
        return self._segments, object()


class _RecordingBatchedPipelineFactory:
    def __init__(self, segments):
        self._segments = segments
        self.model = None
        self.pipeline = None

    def __call__(self, model):
        self.model = model
        self.pipeline = _FakeBatchedPipeline(self._segments)
        return self.pipeline


class _FailingModelFactory:
    def __call__(self, model_size, **options):
        raise RuntimeError("model failed")


if __name__ == "__main__":
    unittest.main()
