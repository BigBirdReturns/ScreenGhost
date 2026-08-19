[CmdletBinding()]
param(
    [string]$Venv = "artifacts\dti\venv"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo

function Resolve-Python {
    if ($env:SG_PYTHON) { return $env:SG_PYTHON }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }
    throw "Python was not found. Set SG_PYTHON to a Python 3.11+ executable."
}

$Python = Resolve-Python
$Version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$Parts = $Version.Trim().Split(".")
if ([int]$Parts[0] -lt 3 -or ([int]$Parts[0] -eq 3 -and [int]$Parts[1] -lt 11)) {
    throw "Python 3.11+ is required; found $Version"
}

$VenvPath = Join-Path $Repo $Venv
$VenvParent = Split-Path $VenvPath -Parent
if ($VenvParent) { New-Item -ItemType Directory -Force -Path $VenvParent | Out-Null }
if (-not (Test-Path $VenvPath)) {
    & $Python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create $VenvPath" }
}
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { throw "Virtual-environment Python was not created." }

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $VenvPython -m pip install -r (Join-Path $Repo "requirements-dti.txt")
if ($LASTEXITCODE -ne 0) { throw "DTI dependency install failed" }

& $VenvPython -m compileall -q (Join-Path $Repo "experiments\dti")
if ($LASTEXITCODE -ne 0) { throw "DTI compile gate failed" }
& $VenvPython -m pytest (Join-Path $Repo "tests\dti") -q
if ($LASTEXITCODE -ne 0) { throw "DTI deterministic tests failed" }
& $VenvPython -m experiments.dti validate-profile (Join-Path $Repo "configs\dti\profile.example.json")
if ($LASTEXITCODE -ne 0) { throw "DTI profile validation failed" }

Write-Host "DTI environment ready: $VenvPython"
