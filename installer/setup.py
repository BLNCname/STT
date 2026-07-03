from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir")
    parser.add_argument("--no-shortcut", dest="create_shortcut", action="store_false")
    parser.add_argument("--no-run", dest="run_after_install", action="store_false")
    parser.set_defaults(create_shortcut=True, run_after_install=True)
    return parser.parse_args(argv)


def install_dir(
    local_appdata: str | None = None, target_dir: str | None = None
) -> Path:
    if target_dir:
        return Path(target_dir)
    base = local_appdata or os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return Path(base) / "SpeechToText"


def resource_path(name: str, base_dir: str | None = None) -> Path:
    base = Path(base_dir or getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def adjacent_resource_path(name: str, base_dir: str | None = None) -> Path:
    if base_dir:
        base = Path(base_dir)
    elif getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    return base / name


def install_payload(source_dir: Path, target_dir: Path) -> Path:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Installer payload not found: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    target = target_dir / "SpeechToText.exe"
    if not target.exists():
        raise FileNotFoundError(
            f"Application executable not found after install: {target}"
        )
    return target


def create_desktop_shortcut(target: Path) -> None:
    command = (
        "$desktop=[Environment]::GetFolderPath('Desktop'); "
        "$shortcut=Join-Path $desktop 'Speech To Text.lnk'; "
        "$shell=New-Object -ComObject WScript.Shell; "
        "$link=$shell.CreateShortcut($shortcut); "
        f"$link.TargetPath='{str(target)}'; "
        f"$link.WorkingDirectory='{str(target.parent)}'; "
        "$link.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_dir = adjacent_resource_path("SpeechToText")
    target_dir = install_dir(target_dir=args.target_dir)
    target = install_payload(source_dir, target_dir)
    if args.create_shortcut:
        create_desktop_shortcut(target)
    if args.run_after_install:
        subprocess.Popen([str(target)], cwd=str(target_dir))


if __name__ == "__main__":
    main()
