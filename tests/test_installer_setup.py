from pathlib import Path
import tempfile
import unittest

from installer.setup import install_dir, install_payload, parse_args, resource_path


class InstallerSetupTests(unittest.TestCase):
    def test_install_dir_uses_local_app_data(self):
        self.assertEqual(
            Path("C:/Users/Test/AppData/Local/SpeechToText"),
            install_dir("C:/Users/Test/AppData/Local"),
        )

    def test_install_dir_can_be_overridden_for_verification(self):
        self.assertEqual(
            Path("C:/Temp/InstallCheck"),
            install_dir(target_dir="C:/Temp/InstallCheck"),
        )

    def test_resource_path_uses_pyinstaller_temp_dir_when_available(self):
        self.assertEqual(
            Path("C:/Temp/_MEI/SpeechToText.exe"),
            resource_path("SpeechToText.exe", base_dir="C:/Temp/_MEI"),
        )

    def test_install_payload_copies_application_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "payload" / "SpeechToText"
            source.mkdir(parents=True)
            (source / "SpeechToText.exe").write_text("exe", encoding="utf-8")
            (source / "model.bin").write_text("model", encoding="utf-8")
            target_dir = Path(temp_dir) / "installed"

            target = install_payload(source, target_dir)

            self.assertEqual(target_dir / "SpeechToText.exe", target)
            self.assertEqual(
                "model", (target_dir / "model.bin").read_text(encoding="utf-8")
            )

    def test_parse_args_supports_safe_verification_mode(self):
        args = parse_args(["--target-dir", "C:/Temp/App", "--no-shortcut", "--no-run"])

        self.assertEqual("C:/Temp/App", args.target_dir)
        self.assertFalse(args.create_shortcut)
        self.assertFalse(args.run_after_install)


if __name__ == "__main__":
    unittest.main()
