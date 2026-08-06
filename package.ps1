# Package the project for sharing.
#
#   .\package.ps1                # source-code zip (no confidential data)
#   .\package.ps1 -App           # zip the built onedir app for a colleague
#   .\package.ps1 -Source -App   # both
#
# All paths are relative to this script, so it works from any checkout on any
# machine. Nothing is hardcoded to one developer's folder.
#
# The source zip uses an ALLOWLIST: only the code paths named below are added.
# Anything not listed - recordings, exports, reports, internal documents - can
# never be swept in by accident, which a denylist cannot guarantee.

[CmdletBinding()]
param(
    [switch]$Source,
    [switch]$App
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $Source -and -not $App) { $Source = $true }   # default

# Code that is safe to share.
$IncludeDirs  = @("dtt", "gui", ".github")
$IncludeFiles = @("README.md", ".gitignore", "DTT-Platform.spec",
                  "build_exe.ps1", "package.ps1", "build-requirements.txt",
                  "generate_sample_data.py")

# Pruned even inside the allowlisted directories.
$SkipDirs = @("__pycache__", ".pytest_cache", ".ipynb_checkpoints", "outputs")
$SkipExt  = @(".pyc", ".pyo", ".csv", ".raw", ".dat", ".pptx", ".xlsx",
              ".png", ".jpg", ".log", ".zip", ".exe")

function New-SourceZip {
    $out = Join-Path $PSScriptRoot "DTT-Platform-source.zip"
    Remove-Item $out -ErrorAction SilentlyContinue
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) ("dtt_src_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    try {
        foreach ($d in $IncludeDirs) {
            if (-not (Test-Path $d)) { Write-Warning "missing dir: $d"; continue }
            Get-ChildItem $d -Recurse -File | ForEach-Object {
                $rel = $_.FullName.Substring($PSScriptRoot.Length).TrimStart('\')
                $parts = $rel -split '\\'
                if ($parts | Where-Object { $SkipDirs -contains $_.ToLower() }) { return }
                if ($SkipExt -contains $_.Extension.ToLower()) { return }
                $dest = Join-Path $staging $rel
                New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
                Copy-Item $_.FullName $dest
            }
        }
        foreach ($f in $IncludeFiles) {
            if (Test-Path $f) { Copy-Item $f (Join-Path $staging $f) }
            else { Write-Warning "missing file: $f" }
        }
        Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $out -CompressionLevel Optimal
    } finally {
        Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
    }
    $n = (Get-ChildItem $out).Length / 1MB
    Write-Host ("Source zip : {0}  ({1:N2} MB)" -f $out, $n) -ForegroundColor Green
}

function New-AppZip {
    $app = Join-Path $PSScriptRoot "dist\DTT-Platform"
    if (-not (Test-Path $app)) { throw "Not built. Run .\build_exe.ps1 first." }
    $out = Join-Path $PSScriptRoot "DTT-Platform-app-windows-x64.zip"
    Remove-Item $out -ErrorAction SilentlyContinue
    Write-Host "Compressing the app folder (a minute or two)..." -ForegroundColor Cyan
    Compress-Archive -Path $app -DestinationPath $out -CompressionLevel Optimal
    $n = (Get-ChildItem $out).Length / 1MB
    Write-Host ("App zip    : {0}  ({1:N1} MB)" -f $out, $n) -ForegroundColor Green
    Write-Host "  Recipient must extract the WHOLE folder; the .exe alone will not run."
}

if ($Source) { New-SourceZip }
if ($App)    { New-AppZip }
