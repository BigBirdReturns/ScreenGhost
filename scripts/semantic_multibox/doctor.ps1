[CmdletBinding()]
param([string]$Python)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $candidate = Join-Path $Repo ".venv-semantic-multibox\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate }
    elseif ($env:SG_PYTHON) { $Python = $env:SG_PYTHON }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
$env:PYTHONPATH = $Repo
& $Python -m experiments.emulator_fleet doctor
exit $LASTEXITCODE
