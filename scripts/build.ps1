param(
    [switch]$NoInstaller,
    [switch]$SkipDependencyInstall,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Dist = Join-Path $Root "dist"
$AppDir = Join-Path $Dist "SpeechToText"
$Exe = Join-Path $AppDir "SpeechToText.exe"
$InstallerDir = Join-Path $Dist "installer"
$SetupExe = Join-Path $InstallerDir "SpeechToText-Setup.exe"

if (-not (Test-Path $Python)) {
    python -m venv $Venv
}

if (-not $SkipDependencyInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $Root "requirements-build.txt")
}

$Models = @(
    @{ Repo = "Systran/faster-whisper-large-v3"; Dir = "faster-whisper-large-v3" },
    @{ Repo = "Systran/faster-distil-whisper-large-v3"; Dir = "faster-whisper-distil-large-v3" },
    @{ Repo = "Systran/faster-whisper-tiny"; Dir = "faster-whisper-tiny" }
)
if (-not $SkipModelDownload) {
    $Hf = Join-Path $Venv "Scripts\hf.exe"
    foreach ($Model in $Models) {
        $ModelDir = Join-Path $Root ("models\" + $Model.Dir)
        if (-not (Test-Path (Join-Path $ModelDir "model.bin"))) {
            & $Hf download $Model.Repo --local-dir $ModelDir --quiet
        }
    }
}

& $Python -m PyInstaller (Join-Path $Root "SpeechToText.spec") --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller application build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $Exe)) {
    throw "PyInstaller did not create $Exe"
}

if ($NoInstaller) {
    Write-Host "Built executable: $Exe"
    exit 0
}

if (-not (Test-Path $InstallerDir)) {
    New-Item -ItemType Directory -Path $InstallerDir | Out-Null
}

$InstallerBuild = Join-Path $Root "build\installer"
$InstallerSpec = Join-Path $Root "build\installer-spec"
$SetupScript = Join-Path $Root "installer\setup.py"

& $Python -m PyInstaller $SetupScript `
    --name "SpeechToText-Setup" `
    --onefile `
    --windowed `
    --noconfirm `
    --clean `
    --paths $Root `
    --distpath $InstallerDir `
    --workpath $InstallerBuild `
    --specpath $InstallerSpec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller installer build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $SetupExe)) {
    throw "PyInstaller did not create $SetupExe"
}

$InstallerPayload = Join-Path $InstallerDir "SpeechToText"
$ResolvedInstallerDir = (Resolve-Path $InstallerDir).Path
if (Test-Path $InstallerPayload) {
    $ResolvedPayload = (Resolve-Path $InstallerPayload).Path
    if (-not $ResolvedPayload.StartsWith($ResolvedInstallerDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected installer payload path: $ResolvedPayload"
    }
    Remove-Item -LiteralPath $InstallerPayload -Recurse -Force
}
Copy-Item -LiteralPath $AppDir -Destination $InstallerPayload -Recurse -Force

Write-Host "Built executable: $Exe"
Write-Host "Built installer: $SetupExe"
Write-Host "Installer payload: $InstallerPayload"
