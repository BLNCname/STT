from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import uuid

from speech_to_text_app.media import CommandRunner
from speech_to_text_app.profiles import profile_from_id
from speech_to_text_app.transcriber import FasterWhisperTranscriber
from speech_to_text_app.workflow import transcribe_media_file


@dataclass(frozen=True)
class JobSnapshot:
    id: str
    status: str
    transcript: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass
class _JobRecord:
    snapshot: JobSnapshot
    thread: threading.Thread | None = None


class JobManager:
    def __init__(
        self,
        work_root: Path,
        transcriber_factory=None,
        runner: CommandRunner | None = None,
    ):
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._transcriber_factory = transcriber_factory or (
            lambda profile: FasterWhisperTranscriber(profile=profile)
        )
        self._runner = runner
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()

    def start(self, source: Path, profile_id: str, language: str | None) -> str:
        job_id = uuid.uuid4().hex
        record = _JobRecord(snapshot=JobSnapshot(id=job_id, status="queued"))
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, source, profile_id, language),
            daemon=True,
        )
        record.thread = thread
        with self._lock:
            self._jobs[job_id] = record
        thread.start()
        return job_id

    def get(self, job_id: str) -> JobSnapshot:
        with self._lock:
            record = self._jobs[job_id]
            return record.snapshot

    def wait(self, job_id: str, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._jobs[job_id].thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _set_snapshot(self, job_id: str, snapshot: JobSnapshot) -> None:
        with self._lock:
            self._jobs[job_id].snapshot = snapshot

    def _run_job(
        self, job_id: str, source: Path, profile_id: str, language: str | None
    ) -> None:
        self._set_snapshot(job_id, JobSnapshot(id=job_id, status="running"))
        try:
            profile = profile_from_id(profile_id)
            if language is not None:
                profile = profile.with_language(language)
            transcriber = self._transcriber_factory(profile)
            job_dir = self.work_root / job_id
            kwargs = {
                "source": source,
                "work_dir": job_dir,
                "transcriber": transcriber,
                "language": profile.language,
            }
            if self._runner is not None:
                kwargs["runner"] = self._runner
            result = transcribe_media_file(**kwargs)
            self._set_snapshot(
                job_id,
                JobSnapshot(
                    id=job_id,
                    status="succeeded",
                    transcript=result.transcript,
                    duration_seconds=result.duration_seconds,
                ),
            )
        except Exception as exc:
            self._set_snapshot(
                job_id,
                JobSnapshot(id=job_id, status="failed", error=str(exc)),
            )
