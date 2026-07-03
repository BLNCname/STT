from pathlib import Path
import unittest

from speech_to_text_app.gui import (
    DEFAULT_MODEL_CHOICE,
    default_transcript_path,
    language_code_from_choice,
    model_size_from_choice,
    profile_id_from_choice,
)


class GuiHelperTests(unittest.TestCase):
    def test_default_model_choice_is_exam_accuracy_gpu(self):
        self.assertEqual("large-v3", model_size_from_choice(DEFAULT_MODEL_CHOICE))
        self.assertEqual("exam_accuracy", profile_id_from_choice(DEFAULT_MODEL_CHOICE))

    def test_maps_language_choice_to_whisper_language_code(self):
        self.assertIsNone(language_code_from_choice("Auto"))
        self.assertEqual("ru", language_code_from_choice("Russian"))
        self.assertEqual("en", language_code_from_choice("English"))

    def test_uses_source_stem_for_default_transcript_path(self):
        self.assertEqual(
            Path("C:/media/interview.txt"),
            default_transcript_path(Path("C:/media/interview.mp4")),
        )


if __name__ == "__main__":
    unittest.main()
