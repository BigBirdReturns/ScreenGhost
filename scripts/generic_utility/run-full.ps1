[CmdletBinding()]
param(
    [string]$Python,
    [string]$Out = "log/generic_utility/full",
    [switch]$SkipBrowser,
    [switch]$RequireBrowser,
    [switch]$SkipFullRepo
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Python) {
    $candidate = Join-Path $Repo ".venv-generic-utility\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate }
    elseif ($env:SG_PYTHON) { $Python = $env:SG_PYTHON }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
$OutPath = Join-Path $Repo $Out
$env:PYTHONPATH = $Repo
$env:PYTHONDONTWRITEBYTECODE = "1"
$VerifyArgs = @(
    (Join-Path $Repo "VERIFY_GENERIC_UTILITY_CAMPAIGN.py"),
    "--out", $OutPath
)
if ($SkipBrowser) { $VerifyArgs += "--skip-browser" }
if ($RequireBrowser) { $VerifyArgs += "--require-browser" }
if ($SkipFullRepo) { $VerifyArgs += "--skip-full-repo" }
& $Python @VerifyArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Generic Utility campaign passed. Receipts: $OutPath"
