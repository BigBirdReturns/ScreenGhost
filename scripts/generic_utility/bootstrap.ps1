[CmdletBinding()]
param(
    [string]$Venv = ".venv-generic-utility",
    [switch]$Models,
    [switch]$InstallChromium
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
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
if ([int]$Parts[0] -lt 3 -or ([int]$Parts[0] -eq 3 -and [int]$Parts[1] -lt 11)) { throw "Python 3.11+ is required; found $Version" }
$VenvPath = Join-Path $Repo $Venv
if (-not (Test-Path $VenvPath)) { & $Python -m venv $VenvPath }
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = Join-Path $VenvPath "bin/python" }
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Repo "requirements-generic-utility.txt")
if ($Models) {
    & $VenvPython -m pip install -r (Join-Path $Repo "requirements-generic-utility-models.txt")
}
if ($InstallChromium) { & $VenvPython -m playwright install chromium }
Write-Host "Generic Utility environment ready: $VenvPython"
