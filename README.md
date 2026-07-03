# Speech To Text

Windows desktop application for local audio and video transcription.

The app wraps `faster-whisper` in a lightweight local web UI, accepts common
media containers, validates file duration before transcription, and can be
packaged as a standalone Windows executable with an installer.

## Features

- Transcribes audio and video files with `faster-whisper`.
- Supports MP4, MP3, WAV, AVI, MKV, MOV, M4A, WEBM and other PyAV-readable
  containers.
- Rejects media longer than 1 hour before inference starts.
- Uses CPU/int8 by default for low-power systems.
- Provides configurable transcription profiles, including CUDA/float16
  profiles when compatible NVIDIA runtime libraries are available.
- Reads media metadata through PyAV, so the packaged app does not depend on a
  separate FFmpeg installation.
- Builds to `SpeechToText.exe` and `SpeechToText-Setup.exe`.

## Tech Stack

- Python 3.11+
- `faster-whisper`
- `ctranslate2`
- PyAV
- PyInstaller
- Standard-library `unittest`

## Project Structure

```text
speech_to_text_app/   application code
tests/                unit and packaging tests
installer/            Windows setup bootstrapper
scripts/              build scripts
```

## Development

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m speech_to_text_app
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Build

Build the executable and installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

Build artifacts are written to:

- `dist\SpeechToText.exe`
- `dist\installer\SpeechToText-Setup.exe`

## Model Files

Whisper model directories are intentionally not committed to this repository.
The build script prepares the local model assets when needed.
