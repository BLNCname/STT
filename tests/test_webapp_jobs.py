from pathlib import Path
import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from speech_to_text_app.transcriber import TranscriptSegment
from speech_to_text_app.webapp import JobManager, create_server, parse_args
from speech_to_text_app.web_upload import parse_multipart_form
from speech_to_text_app.profiles import profile_from_id
from speech_to_text_app.hardware import CudaStatus


class WebAppJobTests(unittest.TestCase):
    def test_parse_multipart_form_reads_file_and_fields(self):
        boundary = "----speech-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="profile"\r\n\r\n'
            "exam_accuracy\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="language"\r\n\r\n'
            "en\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="meeting.mp4"\r\n'
            "Content-Type: video/mp4\r\n\r\n"
            "media-bytes\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        form = parse_multipart_form(
            _BytesReader(body),
            {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )

        self.assertEqual("exam_accuracy", form.getfirst("profile"))
        self.assertEqual("en", form.getfirst("language"))
        self.assertEqual("meeting.mp4", form.files["file"].filename)
        self.assertEqual(b"media-bytes", form.files["file"].file.read())

    def test_runs_transcription_job_to_success(self):
        def runner(command):
            if command[0] == "ffprobe":
                return _ProcessResult(returncode=0, stdout="3\n", stderr="")
            return _ProcessResult(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "meeting.mp4"
            source.write_bytes(b"media")
            seen_profiles = []

            def transcriber_factory(profile):
                seen_profiles.append(profile)
                return _FakeTranscriber([TranscriptSegment(0.0, 1.0, "hello")])

            manager = JobManager(
                work_root=root / "jobs",
                transcriber_factory=transcriber_factory,
                runner=runner,
            )

            job_id = manager.start(
                source=source, profile_id="exam_accuracy", language=None
            )
            self.assertTrue(manager.wait(job_id, timeout=5))

            snapshot = manager.get(job_id)

        self.assertEqual("succeeded", snapshot.status)
        self.assertEqual("[00:00:00.000 - 00:00:01.000] hello", snapshot.transcript)
        self.assertEqual(3.0, snapshot.duration_seconds)
        self.assertIsNone(snapshot.error)
        self.assertEqual("large-v3", seen_profiles[0].model_size)
        self.assertEqual("cuda", seen_profiles[0].device)
        self.assertEqual("en", seen_profiles[0].language)

    def test_records_failed_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "notes.txt"
            source.write_text("not media", encoding="utf-8")
            manager = JobManager(
                work_root=root / "jobs",
                transcriber_factory=lambda profile: _FakeTranscriber([]),
            )

            job_id = manager.start(
                source=source, profile_id="exam_accuracy", language=None
            )
            self.assertTrue(manager.wait(job_id, timeout=5))

            snapshot = manager.get(job_id)

        self.assertEqual("failed", snapshot.status)
        self.assertIn("Unsupported media format", snapshot.error)

    def test_create_server_binds_job_manager_to_handler_class(self):
        server = create_server(
            hardware_probe=lambda: CudaStatus(
                available=True,
                device_count=1,
                supported_compute_types=["float16"],
                error=None,
            )
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with urlopen(f"http://{host}:{port}/api/hardware", timeout=5) as response:
                self.assertEqual(
                    {
                        "available": True,
                        "device_count": 1,
                        "supported_compute_types": ["float16"],
                        "error": None,
                    },
                    json.loads(response.read().decode("utf-8")),
                )
            with self.assertRaises(HTTPError) as context:
                urlopen(f"http://{host}:{port}/api/jobs/missing", timeout=5)
            self.assertEqual(404, context.exception.code)
            payload = json.loads(context.exception.read().decode("utf-8"))
            self.assertEqual({"error": "Job not found"}, payload)
        finally:
            server.shutdown()
            server.server_close()

    def test_default_transcriber_factory_uses_profile_keyword(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = JobManager(work_root=Path(temp_dir) / "jobs")
            profile = profile_from_id("exam_accuracy")

            transcriber = manager._transcriber_factory(profile)

        self.assertEqual("large-v3", transcriber._profile.model_size)
        self.assertEqual("cuda", transcriber._profile.device)
        self.assertEqual("float16", transcriber._profile.compute_type)

    def test_parse_args_supports_fixed_port_without_browser_open(self):
        args = parse_args(["--port", "8766", "--no-open"])

        self.assertEqual(8766, args.port)
        self.assertFalse(args.open_browser)


class _FakeTranscriber:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path, language=None):
        return self._segments


class _ProcessResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _BytesReader:
    def __init__(self, data):
        self._data = data

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._data)
        chunk = self._data[:size]
        self._data = self._data[size:]
        return chunk


if __name__ == "__main__":
    unittest.main()
