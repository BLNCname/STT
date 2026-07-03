from __future__ import annotations

from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol


class HeaderMapping(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class BodyReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    file: BinaryIO


@dataclass(frozen=True)
class UploadForm:
    fields: dict[str, str]
    files: dict[str, UploadedFile]

    def getfirst(self, name: str, default: str | None = None) -> str | None:
        return self.fields.get(name, default)


def safe_upload_name(filename: str | None) -> str:
    name = Path(filename or "media").name
    return name or "media"


def parse_multipart_form(rfile: BodyReader, headers: HeaderMapping) -> UploadForm:
    content_type = headers.get("Content-Type", "") or ""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data")

    content_length_header = headers.get("Content-Length", "0") or "0"
    try:
        content_length = int(content_length_header)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length") from exc

    body = rfile.read(content_length)
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("utf-8")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise ValueError("Invalid multipart body")

    fields: dict[str, str] = {}
    files: dict[str, UploadedFile] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is None:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")
            continue
        files[name] = UploadedFile(filename=filename, file=BytesIO(payload))

    return UploadForm(fields=fields, files=files)
