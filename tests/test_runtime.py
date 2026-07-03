import os
from pathlib import Path
import tempfile
import unittest

from speech_to_text_app.runtime import (
    configure_cuda_dll_search_paths,
    find_bundled_model,
    find_runtime_tool,
    resolve_executable_path,
)


class RuntimeToolTests(unittest.TestCase):
    def test_prefers_bundled_tool_next_to_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tool = base / "ffmpeg.exe"
            tool.write_text("", encoding="utf-8")

            self.assertEqual(
                str(tool), find_runtime_tool("ffmpeg", executable_dir=base)
            )

    def test_uses_path_name_when_tool_is_not_bundled(self):
        self.assertEqual(
            "missing-tool", find_runtime_tool("missing-tool", executable_dir=None)
        )

    def test_resolve_executable_path_preserves_existing_normal_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = Path(temp_dir) / "tool.exe"
            tool.write_text("", encoding="utf-8")

            self.assertEqual(str(tool.resolve()), resolve_executable_path(tool))

    def test_finds_bundled_tiny_model_in_pyinstaller_temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            model_dir = base / "models" / "faster-whisper-tiny"
            model_dir.mkdir(parents=True)

            self.assertEqual(model_dir, find_bundled_model("tiny", temp_dir=base))

    def test_finds_local_large_model_in_project_dir_for_source_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            model_dir = base / "models" / "faster-whisper-large-v3"
            model_dir.mkdir(parents=True)

            self.assertEqual(
                model_dir,
                find_bundled_model("large-v3", project_dir=base),
            )

    def test_configures_nvidia_cuda_dll_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            cublas_dir = base / "nvidia" / "cublas" / "bin"
            runtime_dir = base / "nvidia" / "cuda_runtime" / "bin"
            cublas_dir.mkdir(parents=True)
            runtime_dir.mkdir(parents=True)
            (cublas_dir / "cublas64_12.dll").write_text("", encoding="utf-8")
            (runtime_dir / "cudart64_12.dll").write_text("", encoding="utf-8")
            added_paths = []

            def add_dll_directory(path):
                added_paths.append(Path(path))
                return object()

            old_path = os.environ.get("PATH", "")
            try:
                configured_paths = configure_cuda_dll_search_paths(
                    roots=[base],
                    add_dll_directory=add_dll_directory,
                )
                path_parts = os.environ["PATH"].split(os.pathsep)
            finally:
                os.environ["PATH"] = old_path

        self.assertEqual([cublas_dir, runtime_dir], configured_paths)
        self.assertEqual([cublas_dir, runtime_dir], added_paths)
        self.assertEqual(str(runtime_dir), path_parts[0])
        self.assertEqual(str(cublas_dir), path_parts[1])


if __name__ == "__main__":
    unittest.main()
