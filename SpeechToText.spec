# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


def collect_optional_submodules(package):
    try:
        return collect_submodules(package)
    except Exception:
        return []


datas = collect_data_files("faster_whisper")
models_root = Path("models")
if models_root.exists():
    for model_dir in models_root.glob("faster-whisper-*"):
        if model_dir.is_dir():
            datas.append((str(model_dir), f"models/{model_dir.name}"))

binaries = []
for package_name in ("ctranslate2", "onnxruntime", "nvidia"):
    binaries.extend(collect_dynamic_libs(package_name))

hiddenimports = []
for package_name in ("faster_whisper", "ctranslate2", "av", "tokenizers", "onnxruntime", "nvidia"):
    hiddenimports.extend(collect_optional_submodules(package_name))

a = Analysis(
    ["speech_to_text_app/__main__.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpeechToText",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    [],
    name="SpeechToText",
    strip=False,
    upx=True,
)
