from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import webbrowser

from speech_to_text_app.gui import LANGUAGE_CHOICES, MODEL_CHOICES
from speech_to_text_app.hardware import probe_cuda_status
from speech_to_text_app.jobs import JobManager
from speech_to_text_app.media import SUPPORTED_EXTENSIONS
from speech_to_text_app.web_upload import parse_multipart_form, safe_upload_name


class WebAppHandler(BaseHTTPRequestHandler):
    manager: JobManager
    hardware_probe = staticmethod(probe_cuda_status)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._send_html(INDEX_HTML)
            return
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            try:
                snapshot = self.manager.get(job_id)
            except KeyError:
                self._send_json({"error": "Job not found"}, status=404)
                return
            self._send_json(asdict(snapshot))
            return
        if self.path == "/api/hardware":
            self._send_json(asdict(self.hardware_probe()))
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/jobs":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            form = parse_multipart_form(self.rfile, self.headers)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        file_item = form.files.get("file")
        if file_item is None or not getattr(file_item, "filename", None):
            self._send_json({"error": "No media file uploaded"}, status=400)
            return

        upload_dir = self.manager.work_root / "uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)
        source = upload_dir / safe_upload_name(file_item.filename)
        with source.open("wb") as target:
            shutil.copyfileobj(file_item.file, target)

        profile_id = form.getfirst("profile", form.getfirst("model", "exam_accuracy"))
        language = form.getfirst("language", "") or None
        job_id = self.manager.start(
            source=source, profile_id=profile_id, language=language
        )
        self._send_json({"job_id": job_id}, status=202)

    def log_message(self, format, *args) -> None:
        return

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=True)
    return parser.parse_args(argv)


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    hardware_probe=probe_cuda_status,
) -> ThreadingHTTPServer:
    work_root = Path(tempfile.gettempdir()) / "speech-to-text-jobs"
    manager = JobManager(work_root=work_root)

    class BoundWebAppHandler(WebAppHandler):
        pass

    BoundWebAppHandler.manager = manager
    BoundWebAppHandler.hardware_probe = staticmethod(hardware_probe)
    return ThreadingHTTPServer((host, port), BoundWebAppHandler)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = create_server(host=args.host, port=args.port)
    host, port = server.server_address
    if args.open_browser:
        webbrowser.open(f"http://{host}:{port}/")
    server.serve_forever()


MODEL_OPTIONS_HTML = "\n".join(
    f'<option value="{value}">{label}</option>'
    for label, value in MODEL_CHOICES.items()
)
LANGUAGE_OPTIONS_HTML = "\n".join(
    f'<option value="{value or ""}">{label}</option>'
    for label, value in LANGUAGE_CHOICES.items()
)
ACCEPT_EXTENSIONS = ",".join(sorted(SUPPORTED_EXTENSIONS))

INDEX_HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Speech To Text</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #5d6673;
      --line: #d7dce3;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font: 14px/1.45 "Segoe UI", system-ui, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    form {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 180px 150px auto;
      gap: 10px;
      align-items: end;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    input, select, button, textarea {{
      font: inherit;
    }}
    input[type="file"], select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 7px 9px;
    }}
    button {{
      min-height: 38px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: #fff;
      font-weight: 650;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button:disabled {{
      background: #9aa4b2;
      cursor: default;
    }}
    .status {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      min-height: 42px;
      color: var(--muted);
    }}
    .status strong {{ color: var(--text); }}
    .status .error {{ color: var(--danger); }}
    textarea {{
      width: 100%;
      min-height: 360px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      color: var(--text);
    }}
    .actions {{
      display: flex;
      justify-content: flex-end;
      margin-top: 10px;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 12px; }}
      form {{ grid-template-columns: 1fr; }}
      textarea {{ min-height: 320px; }}
      .status {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Speech To Text</h1>
    <form id="transcribe-form">
      <label>File
        <input id="file" name="file" type="file" accept="{ACCEPT_EXTENSIONS}" required>
      </label>
      <label>Profile
        <select id="profile" name="profile">{MODEL_OPTIONS_HTML}</select>
      </label>
      <label>Language
        <select id="language" name="language">{LANGUAGE_OPTIONS_HTML}</select>
      </label>
      <button id="start" type="submit">Transcribe</button>
    </form>
    <div class="status">
      <span><span id="status">Ready</span> <span id="gpu-status"></span></span>
      <strong id="duration"></strong>
    </div>
    <textarea id="transcript" spellcheck="false"></textarea>
    <div class="actions">
      <button id="save" type="button" disabled>Save TXT</button>
    </div>
  </main>
  <script>
    const form = document.getElementById("transcribe-form");
    const start = document.getElementById("start");
    const statusLine = document.getElementById("status");
    const gpuStatus = document.getElementById("gpu-status");
    const durationLine = document.getElementById("duration");
    const transcript = document.getElementById("transcript");
    const save = document.getElementById("save");
    let currentFileName = "transcript.txt";

    function setBusy(isBusy) {{
      start.disabled = isBusy;
      save.disabled = isBusy || transcript.value.trim().length === 0;
    }}

    function setStatus(text, isError = false) {{
      statusLine.textContent = text;
      statusLine.className = isError ? "error" : "";
    }}

    function formatDuration(seconds) {{
      if (seconds === null || seconds === undefined) return "";
      const mins = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60).toString().padStart(2, "0");
      return `${{mins}}:${{secs}}`;
    }}

    async function loadHardwareStatus() {{
      try {{
        const response = await fetch("/api/hardware");
        const status = await response.json();
        if (status.available) {{
          gpuStatus.textContent = "GPU: CUDA float16";
          gpuStatus.className = "";
        }} else {{
          gpuStatus.textContent = `GPU: ${{status.error || "unavailable"}}`;
          gpuStatus.className = "error";
        }}
      }} catch (error) {{
        gpuStatus.textContent = "";
      }}
    }}

    async function pollJob(jobId) {{
      const response = await fetch(`/api/jobs/${{jobId}}`);
      const job = await response.json();
      if (job.status === "succeeded") {{
        transcript.value = job.transcript || "";
        durationLine.textContent = formatDuration(job.duration_seconds);
        setStatus("Done");
        setBusy(false);
        return;
      }}
      if (job.status === "failed") {{
        setStatus(job.error || "Transcription failed", true);
        setBusy(false);
        return;
      }}
      setStatus(job.status === "queued" ? "Queued" : "Transcribing");
      setTimeout(() => pollJob(jobId), 1000);
    }}

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const file = document.getElementById("file").files[0];
      if (!file) return;
      currentFileName = file.name.replace(/\\.[^.]+$/, "") + ".txt";
      transcript.value = "";
      durationLine.textContent = "";
      setBusy(true);
      setStatus("Uploading");
      try {{
        const response = await fetch("/api/jobs", {{
          method: "POST",
          body: new FormData(form)
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Upload failed");
        pollJob(payload.job_id);
      }} catch (error) {{
        setStatus(error.message, true);
        setBusy(false);
      }}
    }});

    save.addEventListener("click", () => {{
      const blob = new Blob([transcript.value], {{ type: "text/plain;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = currentFileName;
      link.click();
      URL.revokeObjectURL(url);
    }});

    loadHardwareStatus();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
