# Build the DTT WFT Automation Platform as a standalone Windows application.
#
#   .\build_exe.ps1                 # onedir  — fastest (ship the whole folder)
#   .\build_exe.ps1 -OneFile        # onefile — ONE .exe you can just send
#   .\build_exe.ps1 -Clean          # discard PyInstaller's cache first
#   .\build_exe.ps1 -Console        # keep a console window (shows tracebacks)
#
# Output:
#   onedir   dist\DTT-Platform\DTT-Platform.exe   (ship the WHOLE folder)
#   onefile  dist-onefile\DTT-Platform.exe        (single self-contained file)
#
# Measured on this project (555k-sample .dat recording, 38 channels):
#   onedir    startup 2s    full analysis 37s
#   onefile   startup 12s   full analysis 52s
# onefile unpacks itself to a temp folder on every launch, and the GUI relaunches
# its own executable per analysis, so that cost is paid again per run. Use
# onedir for day-to-day work and onefile when handing the app to someone.

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console,
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Prefer an active venv, else the interpreter that has PySide6 installed.
$python = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" }
          else { (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $python -or -not (Test-Path $python)) {
    throw "No Python interpreter found. Activate your venv or add python to PATH."
}
Write-Host "Python : $python" -ForegroundColor Cyan

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
    & $python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller install failed" }
}

# Fail early on a missing runtime dependency rather than after a 3-minute build.
Write-Host "Checking dependencies..." -ForegroundColor Cyan
& $python -c @'
import importlib, sys
missing = [m for m in ("PySide6", "numpy", "scipy", "pandas", "matplotlib",
                       "rainflow", "pptx", "openpyxl")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit("Missing packages: " + ", ".join(missing) +
             "\nRun: pip install -r dtt/requirements.txt -r gui/requirements.txt")
print("  all present")
'@
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }

$buildArgs = @("DTT-Platform.spec", "--noconfirm")
if ($Clean)   { $buildArgs += "--clean" }
if ($OneFile) { $buildArgs += @("--distpath", "dist-onefile"); $env:DTT_BUILD_ONEFILE = "1" }
if ($Console) { $env:DTT_BUILD_CONSOLE = "1" }

Write-Host ("Building {0} (a few minutes)..." -f $(if ($OneFile) {"onefile"} else {"onedir"})) -ForegroundColor Cyan
& $python -m PyInstaller @buildArgs
$rc = $LASTEXITCODE
Remove-Item Env:\DTT_BUILD_CONSOLE -ErrorAction SilentlyContinue
Remove-Item Env:\DTT_BUILD_ONEFILE -ErrorAction SilentlyContinue
if ($rc -ne 0) { throw "PyInstaller build failed" }

if ($OneFile) {
    $exe = "dist-onefile\DTT-Platform.exe"
    if (-not (Test-Path $exe)) { throw "Build reported success but $exe is missing" }
    $mb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    $ship = "this single file — nothing else needed"
} else {
    $exe = "dist\DTT-Platform\DTT-Platform.exe"
    if (-not (Test-Path $exe)) { throw "Build reported success but $exe is missing" }
    $mb = [math]::Round(((Get-ChildItem "dist\DTT-Platform" -Recurse -File |
                          Measure-Object Length -Sum).Sum / 1MB), 1)
    $ship = "the WHOLE dist\DTT-Platform folder — the .exe alone will not run"
}

Write-Host ""
Write-Host "Build complete" -ForegroundColor Green
Write-Host "  exe    : $exe"
Write-Host "  size   : $mb MB"
Write-Host "  ship   : $ship"
Write-Host "  studies: %USERPROFILE%\DTT-Platform\outputs"
Write-Host ""
Write-Host "Headless run:" -ForegroundColor Cyan
Write-Host "  $exe --run-pipeline --raw `"<imc folder>`" --vehicle `"PV`" --study `"Run1`""
