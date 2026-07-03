import unittest

from speech_to_text_app.profiles import (
    DEFAULT_PROFILE_ID,
    TRANSCRIPTION_PROFILES,
    profile_from_id,
)


class TranscriptionProfileTests(unittest.TestCase):
    def test_default_profile_prioritizes_exam_accuracy_on_cuda(self):
        profile = profile_from_id(DEFAULT_PROFILE_ID)

        self.assertEqual("exam_accuracy", profile.id)
        self.assertEqual("large-v3", profile.model_size)
        self.assertEqual("cuda", profile.device)
        self.assertEqual("float16", profile.compute_type)
        self.assertEqual("en", profile.language)
        self.assertEqual(5, profile.beam_size)
        self.assertFalse(profile.batched)

    def test_fast_exam_uses_english_distil_model_without_batching(self):
        profile = profile_from_id("fast_exam")

        self.assertEqual("distil-large-v3", profile.model_size)
        self.assertEqual("cuda", profile.device)
        self.assertEqual("float16", profile.compute_type)
        self.assertEqual("en", profile.language)
        self.assertEqual(1, profile.beam_size)
        self.assertFalse(profile.condition_on_previous_text)
        self.assertFalse(profile.batched)
        self.assertEqual(1, profile.batch_size)

    def test_unknown_profile_falls_back_to_exam_accuracy(self):
        self.assertEqual(
            profile_from_id(DEFAULT_PROFILE_ID), profile_from_id("unknown")
        )

    def test_cpu_fallback_is_available_but_not_default(self):
        self.assertIn("cpu_fallback", TRANSCRIPTION_PROFILES)
        self.assertNotEqual("cpu_fallback", DEFAULT_PROFILE_ID)


if __name__ == "__main__":
    unittest.main()
