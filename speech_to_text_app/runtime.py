from __future__ import annotations

from pathlib import Path
import os
import shutil
import site
import sys


_CUDA_DLL_DIRECTORY_HANDLES = []
_CUDA_DLL_DIRECTORIES: set[Path] = set()


def resolve_executable_path(path: str | Path) -> str:
    value = Path(path)
    if os.path.islink(value):
        target = Path(os.readlink(value))
        if not target.is_absolute():
            target = value.parent / target
        return str(target.resolve(strict=True))
    try:
        return str(value.resolve(strict=True))
    except OSError:
        return str(value)


def executable_dir() -> Path | None:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def _existing_cuda_dll_dirs(roots: list[Path]) -> list[Path]:
    subdirs = [
        Path("nvidia/cublas/bin"),
        Path("nvidia/cuda_runtime/bin"),
        Path("nvidia/cuda_nvrtc/bin"),
        Path("nvidia/cudnn/bin"),
        Path("ctranslate2"),
    ]
    found: list[Path] = []
    for root in roots:
        for subdir in subdirs:
            candidate = root / subdir
            if candidate.is_dir() and any(candidate.glob("*.dll")):
                resolved = candidate.resolve()
                if resolved not in found:
                    found.append(resolved)
    return found


def _default_cuda_dll_roots() -> list[Path]:
    roots: list[Path] = []
    for value in [pyinstaller_temp_dir(), executable_dir()]:
        if value is not None:
            roots.append(value)
            roots.append(value / "_internal")
    for value in [Path(sys.prefix) / "Lib" / "site-packages", Path.cwd()]:
        roots.append(value)
    for value in site.getsitepackages():
        roots.append(Path(value))
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    unique_roots: list[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def configure_cuda_dll_search_paths(
    roots: list[Path] | None = None,
    add_dll_directory=None,
) -> list[Path]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    if add_dll_directory is None:
        add_dll_directory = os.add_dll_directory

    configured: list[Path] = []
    for directory in _existing_cuda_dll_dirs(roots or _default_cuda_dll_roots()):
        if directory in _CUDA_DLL_DIRECTORIES:
            continue
        _CUDA_DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(directory)))
        _CUDA_DLL_DIRECTORIES.add(directory)
        _prepend_to_process_path(directory)
        configured.append(directory)
    return configured


def _prepend_to_process_path(directory: Path) -> None:
    path_key = next((key for key in os.environ if key.lower() == "path"), "PATH")
    path_value = os.environ.get(path_key, "")
    path_parts = [part for part in path_value.split(os.pathsep) if part]
    directory_text = str(directory)
    if any(part.lower() == directory_text.lower() for part in path_parts):
        return
    os.environ[path_key] = os.pathsep.join([directory_text, *path_parts])


def pyinstaller_temp_dir() -> Path | None:
    value = getattr(sys, "_MEIPASS", None)
    if value:
        return Path(value)
    return None


def find_runtime_tool(
    tool_name: str,
    executable_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> str:
    exe_name = tool_name if tool_name.lower().endswith(".exe") else f"{tool_name}.exe"
    env_key = f"STT_{Path(exe_name).stem.upper()}_PATH"
    env_value = os.environ.get(env_key)
    if env_value and Path(env_value).exists():
        return resolve_executable_path(env_value)

    for base in [
        executable_dir,
        temp_dir,
        pyinstaller_temp_dir(),
        globals()["executable_dir"](),
    ]:
        if base is None:
            continue
        candidate = base / exe_name
        if os.path.lexists(candidate):
            return resolve_executable_path(candidate)

    winget_link = (
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / exe_name
    )
    if os.path.lexists(winget_link):
        return resolve_executable_path(winget_link)

    path_value = shutil.which(exe_name) or shutil.which(tool_name)
    if path_value:
        return resolve_executable_path(path_value)
    return tool_name


def find_bundled_model(
    model_size: str,
    executable_dir: Path | None = None,
    temp_dir: Path | None = None,
    project_dir: Path | None = None,
) -> Path | None:
    model_dir_name = f"faster-whisper-{model_size}"
    for base in [
        temp_dir,
        pyinstaller_temp_dir(),
        executable_dir,
        executable_dir / "_internal" if executable_dir is not None else None,
        globals()["executable_dir"](),
        globals()["executable_dir"]() / "_internal"
        if globals()["executable_dir"]() is not None
        else None,
        project_dir,
        Path.cwd(),
    ]:
        if base is None:
            continue
        candidate = base / "models" / model_dir_name
        if candidate.exists():
            return candidate
    return None
