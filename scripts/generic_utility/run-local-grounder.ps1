[CmdletBinding()]
param(
    [string]$Python,
    [string]$Model = "osunlp/UGround-V1-2B",
    [string]$Out = "log/generic_utility/grounding-local",
    [string]$DType = "float16"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $candidate = Join-Path $Repo ".venv-generic-utility\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate } else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
$env:PYTHONPATH = $Repo
& $Python -m experiments.generic_utility grounding-local --model $Model --dtype $DType --out (Join-Path $Repo $Out)
